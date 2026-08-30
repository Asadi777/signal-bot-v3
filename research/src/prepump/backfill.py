"""OHLCV backfill orchestration.

Guarantees this module is responsible for:

* **idempotent** — the partition (exchange, market_type, symbol, UTC date) is
  written atomically; running the same range twice produces the same content
  checksum and never appends duplicates.
* **resumable** — an interrupted run resumes from the raw sidecar and the
  partition sidecars, and does not re-download work already on disk.
* **auditable** — every partition names the source revision it came from, and
  the run manifest names every partition, artifact, gap and error.

Raw bytes are kept exactly as served. Normalized output is always derivable
from raw, which is what makes "deterministic subject to source revisions"
checkable rather than aspirational.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from prepump.io.parquet import is_current, partition_state, write_partition
from prepump.io.paths import PartitionKey, ensure_dir, partition_dir, raw_path
from prepump.logging import get_logger
from prepump.manifest import RunManifest
from prepump.normalize import ohlcv
from prepump.normalize.schema import validate_table
from prepump.normalize.symbols import SymbolSpec
from prepump.net.client import BannedError, NotFoundError
from prepump.timeutils import MINUTE_MS, date_str, utc_now, utc_now_ms
from prepump.version import COLLECTOR_VERSION, SCHEMA_VERSION

log = get_logger(__name__)

DATASET = "ohlcv_1m"


@dataclass
class BackfillSummary:
    partitions_written: int = 0
    partitions_skipped: int = 0
    rows_written: int = 0
    units_fetched: int = 0
    units_skipped: int = 0
    units_missing: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "partitions_written": self.partitions_written,
            "partitions_skipped": self.partitions_skipped,
            "rows_written": self.rows_written,
            "units_fetched": self.units_fetched,
            "units_skipped": self.units_skipped,
            "units_missing": self.units_missing,
            "errors": self.errors,
        }


def _raw_meta_path(settings, source_name: str, spec: SymbolSpec, interval: str, filename: str) -> Path:
    return raw_path(settings.raw_dir, source_name, spec.market_type, spec.symbol, interval, filename + ".meta.json")


def _unit_filename(unit_key: str, unit_label: str) -> str:
    safe = unit_label or unit_key
    return safe.replace(":", "_").replace("/", "_")


def backfill_symbol(
    settings,
    source,
    spec: SymbolSpec,
    start_ms: int,
    end_ms: int,
    manifest: RunManifest,
    *,
    interval: str = "1m",
    interval_ms: int = MINUTE_MS,
    resume: bool = True,
    save_raw: bool = True,
) -> BackfillSummary:
    summary = BackfillSummary()
    descriptor = source.descriptor

    for unit in source.plan(spec, start_ms, end_ms):
        filename = unit.url.rsplit("/", 1)[-1] if "/" in unit.url else _unit_filename(unit.key, unit.label)
        if not filename.endswith((".zip", ".csv", ".json")):
            filename = _unit_filename(unit.key, unit.label) + ".json"
        meta_path = _raw_meta_path(settings, descriptor.name, spec, interval, filename)

        if resume and _unit_is_complete(settings, meta_path, spec):
            summary.units_skipped += 1
            manifest.bump("units_skipped")
            log.info("unit already complete", extra={"unit": unit.key})
            continue

        try:
            result = source.fetch(spec, unit)
        except NotFoundError:
            # The source simply does not have this month for this symbol. That
            # is a coverage fact about the source, recorded as a gap, not an
            # error that should fail the run.
            summary.units_missing += 1
            manifest.record_gap(
                kind="SOURCE_UNIT_ABSENT",
                symbol=spec.symbol,
                market_type=spec.market_type,
                source=descriptor.name,
                unit=unit.key,
                url=unit.url,
            )
            log.warning("unit absent at source", extra={"unit": unit.key})
            continue
        except BannedError:
            raise  # a hard stop must abort the run, never be retried in a loop
        except Exception as exc:
            summary.errors.append(f"{unit.key}: {exc}")
            manifest.record_error(f"fetch:{unit.key}", exc)
            log.error("unit fetch failed", extra={"unit": unit.key, "error": str(exc)})
            continue

        summary.units_fetched += 1
        ingested_at_ms = utc_now_ms()
        by_date = ohlcv.split_by_utc_date(result.klines)

        if save_raw and result.raw_bytes is not None:
            _write_raw(settings, descriptor.name, spec, interval, filename, result, sorted(by_date))

        manifest.raw_artifacts.append(
            {
                "unit": unit.key,
                "url": unit.url,
                "source_revision": result.source_revision,
                "rows": len(result.klines),
                "dates": sorted(by_date),
                "meta": result.meta,
            }
        )

        for date, klines in sorted(by_date.items()):
            key = PartitionKey(spec.exchange, spec.market_type, spec.symbol, date)
            part_dir = partition_dir(settings.normalized_dir, key, DATASET)
            existing = partition_state(part_dir)
            if resume and is_current(
                existing,
                source_revision=result.source_revision,
                schema_version=SCHEMA_VERSION,
                collector_version=COLLECTOR_VERSION,
            ):
                summary.partitions_skipped += 1
                manifest.partitions_skipped.append({"path": str(part_dir), "rows": existing.rows})
                continue

            table = ohlcv.to_table(
                klines,
                spec=spec,
                interval=interval,
                interval_ms=interval_ms,
                source=descriptor.name,
                source_revision=result.source_revision,
                run_id=manifest.run_id,
                ingested_at_ms=ingested_at_ms,
                availability_method=descriptor.availability_method,
                availability_delta_ms=descriptor.availability_delta_ms,
                availability_uncertainty_ms=descriptor.availability_uncertainty_ms,
            )
            validate_table(table)
            meta = write_partition(
                table,
                part_dir,
                source=descriptor.name,
                source_revision=result.source_revision,
                schema_version=SCHEMA_VERSION,
                collector_version=COLLECTOR_VERSION,
                run_id=manifest.run_id,
            )
            summary.partitions_written += 1
            summary.rows_written += meta.rows
            manifest.partitions_written.append(
                {"path": str(part_dir), "rows": meta.rows, "content_sha256": meta.content_sha256}
            )

    return summary


def _write_raw(settings, source_name, spec, interval, filename, result, dates: list[str]) -> None:
    target = raw_path(settings.raw_dir, source_name, spec.market_type, spec.symbol, interval, filename)
    ensure_dir(target.parent)
    target.write_bytes(result.raw_bytes)
    meta = {
        "filename": filename,
        "source": source_name,
        "source_revision": result.source_revision,
        "fetched_at": utc_now().isoformat(timespec="milliseconds"),
        "rows": len(result.klines),
        "dates": dates,
        "collector_version": COLLECTOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "detail": result.meta,
    }
    (target.parent / (filename + ".meta.json")).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _unit_is_complete(settings, meta_path: Path, spec: SymbolSpec) -> bool:
    """True when a previous run already produced every partition of this unit.

    The check compares against the revision recorded when the unit was last
    fetched, so a *resumed* run cannot notice that the source has since revised
    the file: knowing that would require downloading it again, which is exactly
    what resume exists to avoid. Detecting upstream revisions is therefore the
    job of a ``resume=False`` run (``--no-resume``), which re-fetches and
    rewrites any partition whose revision moved. Run one periodically; the
    published checksums make the comparison cheap and exact.
    """
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    revision = meta.get("source_revision")
    dates = meta.get("dates") or []
    if not revision or not dates:
        return False
    for date in dates:
        key = PartitionKey(spec.exchange, spec.market_type, spec.symbol, date)
        state = partition_state(partition_dir(settings.normalized_dir, key, DATASET))
        if not is_current(
            state,
            source_revision=revision,
            schema_version=SCHEMA_VERSION,
            collector_version=COLLECTOR_VERSION,
        ):
            return False
    return True


def expected_dates(start_ms: int, end_ms: int) -> list[str]:
    out, cursor = [], start_ms
    while cursor <= end_ms:
        out.append(date_str(cursor))
        cursor += 86_400_000
    return sorted(set(out))
