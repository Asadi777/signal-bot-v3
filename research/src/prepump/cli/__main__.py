"""Reproducible command-line entry points.

Every command that touches data writes a run manifest, so any output can be
traced back to the code version, config hash and source revisions that made it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml

from prepump import backfill as backfill_mod
from prepump import fixtures as fixtures_mod
from prepump.config import load_settings
from prepump.io.parquet import read_dataset, write_partition
from prepump.io.paths import ensure_dir
from prepump.logging import configure_logging, get_logger
from prepump.manifest import RunManifest
from prepump.matrix.model import Matrix
from prepump.matrix.render import render as render_matrix_md
from prepump.matrix.validate import validate_matrix
from prepump.net.client import build_client
from prepump.normalize.schema import OHLCV_SCHEMA, STAGE_A_DISALLOWED_COLUMNS
from prepump.quality.report import build_report, write_report
from prepump.sources.binance_rest import BinanceRestSource
from prepump.sources.binance_vision import BinanceVisionSource
from prepump.sources.bybit_rest import BybitRestSource
from prepump.timeutils import dt_to_ms
from prepump.universe import load_universe
from prepump.version import COLLECTOR_VERSION, MATRIX_SCHEMA_VERSION, METADATA_SCHEMA_VERSION, SCHEMA_VERSION

app = typer.Typer(add_completion=False, help="Phase 0A research data pipeline.")
log = get_logger("prepump.cli")

DEFAULT_UNIVERSE = Path("config/smoke_universe_v1.yaml")
DEFAULT_MATRIX = Path("config/data_source_matrix_v1.yaml")


def _parse_bound(value: str, *, end: bool) -> int:
    """Accept YYYY-MM or YYYY-MM-DD and return an inclusive epoch-ms bound."""
    parts = value.split("-")
    if len(parts) == 2:
        from prepump.timeutils import month_bounds_ms

        first, last = month_bounds_ms(value)
        return last if end else first
    if len(parts) == 3:
        day = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt_to_ms(day) + 86_400_000 - 1 if end else dt_to_ms(day)
    raise typer.BadParameter(f"expected YYYY-MM or YYYY-MM-DD, got {value!r}")


def _make_source(name: str, settings, client, interval: str = "1m"):
    if name == "binance_vision":
        return BinanceVisionSource(client, settings.binance_vision_base_url, interval)
    if name == "binance_rest":
        return BinanceRestSource(client, settings.binance_spot_rest_base_url, settings.binance_futures_rest_base_url, interval)
    if name == "bybit_rest":
        return BybitRestSource(client, settings.bybit_rest_base_url, interval)
    raise typer.BadParameter(f"unknown source {name!r}")


@app.command()
def version() -> None:
    """Print the version metadata stamped onto every output."""
    typer.echo(
        json.dumps(
            {
                "collector_version": COLLECTOR_VERSION,
                "ohlcv_schema_version": SCHEMA_VERSION,
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "matrix_schema_version": MATRIX_SCHEMA_VERSION,
            },
            indent=2,
        )
    )


@app.command("schema-info")
def schema_info() -> None:
    """Print the normalized OHLCV contract and the Stage A column policy."""
    fields = [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in OHLCV_SCHEMA]
    typer.echo(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "fields": fields,
                "stage_a_disallowed_columns": list(STAGE_A_DISALLOWED_COLUMNS),
                "note": "Stage A column policy is provisional pending the decision in docs/decisions/0002.",
            },
            indent=2,
        )
    )


@app.command("make-fixtures")
def make_fixtures(
    out: Path = typer.Option(Path("data/fixtures/vision"), help="Root of the offline archive tree."),
    months: str = typer.Option("2024-01,2024-02", help="Comma-separated YYYY-MM list."),
    seed: int = typer.Option(42),
) -> None:
    """Generate the synthetic offline archive used for network-free runs."""
    month_list = [m.strip() for m in months.split(",") if m.strip()]
    written = fixtures_mod.build_archive(ensure_dir(out), month_list, seed=seed)
    typer.echo(f"wrote {len(written)} fixture archives under {out}")
    typer.echo(f"base url: {fixtures_mod.fixture_base_url(out)}")


@app.command("backfill-ohlcv")
def backfill_ohlcv(
    universe: Path = typer.Option(DEFAULT_UNIVERSE, help="Universe config file."),
    start: str = typer.Option(..., help="Inclusive start, YYYY-MM or YYYY-MM-DD."),
    end: str = typer.Option(..., help="Inclusive end, YYYY-MM or YYYY-MM-DD."),
    source: str = typer.Option("binance_vision", help="binance_vision | binance_rest | bybit_rest"),
    base_url: str = typer.Option("", help="Override the source base URL (supports file:// for offline runs)."),
    exchange: str = typer.Option("", help="Filter the universe by exchange."),
    market_type: str = typer.Option("", help="Filter the universe by market type."),
    data_root: Path = typer.Option(None, help="Override the output root."),
    resume: bool = typer.Option(True, help="Skip units and partitions already complete on disk."),
) -> None:
    """Backfill 1-minute OHLCV for the smoke universe."""
    overrides = {}
    if data_root:
        overrides["data_root"] = data_root
    if base_url:
        overrides["binance_vision_base_url" if source == "binance_vision" else "bybit_rest_base_url"] = base_url
    settings = load_settings(**overrides)
    configure_logging(settings.log_level)

    uni = load_universe(universe)
    specs = [
        entry.spec
        for entry in uni.filter(exchange or None, market_type or None)
    ]
    if not specs:
        raise typer.BadParameter("no symbols selected from the universe after filtering")

    start_ms, end_ms = _parse_bound(start, end=False), _parse_bound(end, end=True)
    manifest = RunManifest.start(
        "backfill-ohlcv",
        settings.config_hash(),
        {
            "universe": str(universe),
            "universe_version": uni.version,
            "universe_status": uni.status,
            "universe_limitations": uni.limitations,
            "source": source,
            "start": start,
            "end": end,
            "symbols": [spec.symbol for spec in specs],
            "resume": resume,
        },
    )
    configure_logging(settings.log_level, manifest.run_id)
    client = build_client(settings)
    collector = _make_source(source, settings, client)

    totals = {"written": 0, "skipped": 0, "rows": 0, "missing_units": 0}
    for spec in specs:
        if not collector.supports(spec):
            manifest.record_gap(
                kind="SOURCE_DOES_NOT_SUPPORT_SYMBOL",
                symbol=spec.symbol,
                market_type=spec.market_type,
                source=collector.descriptor.name,
            )
            continue
        summary = backfill_mod.backfill_symbol(settings, collector, spec, start_ms, end_ms, manifest, resume=resume)
        totals["written"] += summary.partitions_written
        totals["skipped"] += summary.partitions_skipped
        totals["rows"] += summary.rows_written
        totals["missing_units"] += summary.units_missing

    manifest.http_stats = client.stats.to_dict()
    manifest.counters.update(totals)
    path = manifest.finish(settings.manifests_dir)
    typer.echo(json.dumps({"run_id": manifest.run_id, "manifest": str(path), **totals}, indent=2))


@app.command("collect-metadata")
def collect_metadata(
    exchange: str = typer.Option("all", help="binance | bybit | all"),
    data_root: Path = typer.Option(None),
) -> None:
    """Snapshot exchange symbol/market metadata (append-only by collection date)."""
    from prepump.io.paths import PartitionKey, partition_dir
    from prepump.sources import exchange_meta
    from prepump.timeutils import date_str, utc_now_ms

    settings = load_settings(**({"data_root": data_root} if data_root else {}))
    configure_logging(settings.log_level)
    manifest = RunManifest.start("collect-metadata", settings.config_hash(), {"exchange": exchange})
    configure_logging(settings.log_level, manifest.run_id)
    client = build_client(settings)

    jobs = []
    if exchange in ("binance", "all"):
        jobs.append(("binance", "spot", lambda: exchange_meta.collect_binance_spot(client, settings.binance_spot_rest_base_url, manifest.run_id)))
        jobs.append(("binance", "futures", lambda: exchange_meta.collect_binance_futures(client, settings.binance_futures_rest_base_url, manifest.run_id)))
    if exchange in ("bybit", "all"):
        for category in ("spot", "linear"):
            jobs.append(("bybit", category, lambda c=category: exchange_meta.collect_bybit(client, settings.bybit_rest_base_url, c, manifest.run_id)))

    today = date_str(utc_now_ms())
    written = 0
    for venue, label, job in jobs:
        try:
            rows = job()
        except Exception as exc:
            manifest.record_error(f"metadata:{venue}:{label}", exc)
            log.error("metadata collection failed", extra={"exchange": venue, "scope": label, "error": str(exc)})
            continue
        if not rows:
            manifest.record_gap(kind="EMPTY_METADATA_RESPONSE", exchange=venue, scope=label)
            continue
        table = exchange_meta.rows_to_table(rows)
        key = PartitionKey(venue, label, "_all", today)
        part_dir = partition_dir(settings.normalized_dir, key, "exchange_metadata")
        meta = write_partition(
            table,
            part_dir,
            source=rows[0]["source"],
            source_revision=rows[0]["source_revision"],
            schema_version=METADATA_SCHEMA_VERSION,
            collector_version=COLLECTOR_VERSION,
            run_id=manifest.run_id,
        )
        manifest.partitions_written.append({"path": str(part_dir), "rows": meta.rows})
        written += meta.rows

    manifest.http_stats = client.stats.to_dict()
    manifest.counters["metadata_rows"] = written
    path = manifest.finish(settings.manifests_dir)
    typer.echo(json.dumps({"run_id": manifest.run_id, "rows": written, "manifest": str(path)}, indent=2))


@app.command("quality-report")
def quality_report(
    data_root: Path = typer.Option(None),
    dataset: str = typer.Option("ohlcv_1m"),
    stem: str = typer.Option("data_quality"),
) -> None:
    """Generate the data-quality report for the normalized dataset."""
    settings = load_settings(**({"data_root": data_root} if data_root else {}))
    configure_logging(settings.log_level)
    table = read_dataset(settings.normalized_dir, dataset)
    if table is None:
        raise typer.Exit(code=typer.echo(f"no partitions found under {settings.normalized_dir / dataset}") or 1)

    notes = [
        "Missing minutes are classified, never filled. A one-minute gap on an illiquid pair is most likely "
        "'no trade occurred', not lost data.",
        "available_at is derived from candle close for archive data and is not a measured availability time; "
        "see available_at_method and available_at_uncertainty_ms on every row.",
        "This dataset is a provisional smoke-test sample. It is not a research universe and must not be "
        "generalized (Master Spec §65.3).",
    ]
    report = build_report(table, settings.normalized_dir, dataset=dataset, notes=notes)
    json_path, md_path = write_report(report, settings.reports_dir, stem)
    typer.echo(json.dumps({"json": str(json_path), "markdown": str(md_path), **report["totals"]}, indent=2))


@app.command("validate-matrix")
def validate_matrix_cmd(path: Path = typer.Option(DEFAULT_MATRIX)) -> None:
    """Validate the Data Source Matrix against the evidence rules."""
    matrix = Matrix.model_validate(yaml.safe_load(Path(path).read_text()))
    problems = validate_matrix(matrix)
    if problems:
        for problem in problems:
            typer.echo(str(problem))
        raise typer.Exit(code=1)
    typer.echo(f"matrix OK: {len(matrix.rows)} rows, schema {matrix.schema_version}")


@app.command("render-matrix")
def render_matrix_cmd(
    path: Path = typer.Option(DEFAULT_MATRIX),
    out: Path = typer.Option(Path("../docs/data_source_matrix_v1.md")),
) -> None:
    """Render the machine-readable matrix to the Markdown template layout."""
    matrix = Matrix.model_validate(yaml.safe_load(Path(path).read_text()))
    problems = validate_matrix(matrix)
    if problems:
        for problem in problems:
            typer.echo(str(problem))
        raise typer.Exit(code=1)
    ensure_dir(out.parent)
    out.write_text(render_matrix_md(matrix))
    typer.echo(f"wrote {out}")


@app.command("probe-bybit-depth")
def probe_bybit_depth(
    symbol: str = typer.Option("BTCUSDT"),
    market_type: str = typer.Option("linear_perpetual"),
    out: Path = typer.Option(Path("data/reports/bybit_depth_probe.json")),
) -> None:
    """Measure how far back Bybit actually serves 1-minute candles."""
    from prepump.normalize.symbols import make_spec

    settings = load_settings()
    configure_logging(settings.log_level)
    client = build_client(settings)
    source = BybitRestSource(client, settings.bybit_rest_base_url)
    result = source.probe_depth(make_spec("bybit", market_type, symbol))
    ensure_dir(out.parent)
    out.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps(result, indent=2))


@app.command("selftest-offline")
def selftest_offline(
    data_root: Path = typer.Option(Path("data/selftest")),
    months: str = typer.Option("2024-01,2024-02"),
) -> None:
    """Run the whole pipeline offline and assert the acceptance properties.

    Proves, without touching the network: idempotency, resume, deterministic
    content checksums, schema validity and report generation.
    """
    from prepump.io.paths import PARTITION_SIDECAR

    settings_root = ensure_dir(data_root)
    fixture_root = settings_root / "fixtures" / "vision"
    month_list = [m.strip() for m in months.split(",") if m.strip()]
    fixtures_mod.build_archive(ensure_dir(fixture_root), month_list)

    settings = load_settings(
        data_root=settings_root,
        binance_vision_base_url=fixtures_mod.fixture_base_url(fixture_root),
    )
    configure_logging(settings.log_level)
    universe = Path(__file__).resolve().parents[3] / "config" / "smoke_universe_offline.yaml"
    uni = load_universe(universe)

    start_ms = _parse_bound(month_list[0], end=False)
    end_ms = _parse_bound(month_list[-1], end=True)

    def run(resume: bool) -> tuple[dict, RunManifest]:
        manifest = RunManifest.start("selftest-offline", settings.config_hash(), {"resume": resume})
        client = build_client(settings)
        source = BinanceVisionSource(client, settings.binance_vision_base_url)
        for spec in uni.specs:
            backfill_mod.backfill_symbol(settings, source, spec, start_ms, end_ms, manifest, resume=resume)
        manifest.http_stats = client.stats.to_dict()
        manifest.finish(settings.manifests_dir)
        checksums = {
            str(sidecar.parent.relative_to(settings.normalized_dir)): json.loads(sidecar.read_text())["content_sha256"]
            for sidecar in sorted((settings.normalized_dir / "ohlcv_1m").rglob(PARTITION_SIDECAR))
        }
        return checksums, manifest

    first, manifest_one = run(resume=True)
    second, manifest_two = run(resume=True)

    checks = {
        "partitions_written_first_run": len(manifest_one.partitions_written),
        "partitions_written_second_run": len(manifest_two.partitions_written),
        "partitions_skipped_second_run": len(manifest_two.partitions_skipped),
        "identical_content_checksums": first == second,
        "second_run_wrote_nothing": len(manifest_two.partitions_written) == 0,
        "no_errors": not (manifest_one.errors or manifest_two.errors),
    }

    table = read_dataset(settings.normalized_dir, "ohlcv_1m")
    checks["rows"] = 0 if table is None else table.num_rows
    report = build_report(table, settings.normalized_dir, run_id=manifest_one.run_id, notes=["Synthetic offline fixture data."])
    write_report(report, settings.reports_dir, "selftest_data_quality")
    checks["report_symbols"] = report["totals"]["symbols"]
    checks["report_found_duplicates"] = report["totals"]["duplicate_rows"] > 0
    checks["report_found_gaps"] = report["totals"]["missing_minutes"] > 0

    passed = (
        checks["identical_content_checksums"]
        and checks["second_run_wrote_nothing"]
        and checks["no_errors"]
        and checks["rows"] > 0
    )
    typer.echo(json.dumps({"passed": passed, **checks}, indent=2))
    raise typer.Exit(code=0 if passed else 1)


if __name__ == "__main__":
    app()
