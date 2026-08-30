"""Backfill behaviour: idempotency, resume, gaps and hard stops."""

import json

import pytest

from prepump import backfill as backfill_mod
from prepump.config import Settings
from prepump.io.paths import PARTITION_SIDECAR
from prepump.manifest import RunManifest
from prepump.net.client import BannedError, NotFoundError
from prepump.normalize.symbols import make_spec
from prepump.sources.base import FetchResult, FetchUnit, SourceDescriptor
from prepump.normalize.schema import AvailabilityMethod
from prepump.timeutils import MINUTE_MS
from tests.helpers import kline

START = 1_704_067_200_000  # 2024-01-01
SPEC = make_spec("binance", "spot", "TESTUSDT", "TEST", "USDT")


class FakeSource:
    """Two units: one day of candles each, on consecutive days."""

    def __init__(self, fail_on: set[str] | None = None, revision="sha256:rev1", units=2):
        self.descriptor = SourceDescriptor(
            name="fake_source",
            exchange="binance",
            availability_method=AvailabilityMethod.DERIVED_FROM_CLOSE,
            availability_delta_ms=1,
            availability_uncertainty_ms=2000,
        )
        self.fail_on = fail_on or set()
        self.revision = revision
        self.units = units
        self.fetched: list[str] = []

    def supports(self, spec):
        return True

    def plan(self, spec, start_ms, end_ms):
        return [
            FetchUnit(key=f"u{i}", url=f"https://fake/{spec.symbol}-{i}.zip", label=f"unit{i}")
            for i in range(self.units)
        ]

    def fetch(self, spec, unit):
        self.fetched.append(unit.key)
        if unit.key in self.fail_on:
            raise RuntimeError(f"boom on {unit.key}")
        day = int(unit.key[1:])
        base = START + day * 86_400_000
        klines = [kline(i, open_time_ms=base + i * MINUTE_MS) for i in range(10)]
        return FetchResult(klines=klines, source_revision=self.revision, raw_bytes=b"raw", raw_filename="f.zip")


def settings_for(tmp_path) -> Settings:
    return Settings(data_root=tmp_path)


def run(settings, source, resume=True):
    manifest = RunManifest.start("test", settings.config_hash())
    summary = backfill_mod.backfill_symbol(
        settings, source, SPEC, START, START + 2 * 86_400_000, manifest, resume=resume
    )
    return summary, manifest


def test_first_run_writes_partitions_and_raw(tmp_path):
    settings = settings_for(tmp_path)
    summary, manifest = run(settings, FakeSource())
    assert summary.partitions_written == 2
    assert summary.rows_written == 20
    assert manifest.raw_artifacts and manifest.partitions_written
    raw_files = list((settings.raw_dir).rglob("*.meta.json"))
    assert len(raw_files) == 2


def test_second_run_is_a_no_op(tmp_path):
    settings = settings_for(tmp_path)
    run(settings, FakeSource())
    source = FakeSource()
    summary, _ = run(settings, source)
    assert summary.partitions_written == 0
    assert summary.units_skipped == 2
    assert source.fetched == [], "a resumed run must not re-download completed units"


def test_rerun_does_not_duplicate_rows(tmp_path):
    settings = settings_for(tmp_path)
    run(settings, FakeSource())
    run(settings, FakeSource())
    from prepump.io.parquet import read_dataset

    table = read_dataset(settings.normalized_dir)
    times = table.column("event_time").to_pylist()
    assert len(times) == len(set(times)) == 20


def test_interrupted_run_resumes_only_the_missing_unit(tmp_path):
    settings = settings_for(tmp_path)
    summary, manifest = run(settings, FakeSource(fail_on={"u1"}))
    assert summary.partitions_written == 1
    assert manifest.errors and "boom" in manifest.errors[0]["error"]

    source = FakeSource()
    summary2, manifest2 = run(settings, source)
    assert source.fetched == ["u1"], "only the missing unit should be refetched"
    assert summary2.partitions_written == 1
    assert summary2.units_skipped == 1
    assert not manifest2.errors


def test_content_is_identical_across_runs(tmp_path):
    def checksums(root):
        return {
            str(p.parent.relative_to(root)): json.loads(p.read_text())["content_sha256"]
            for p in sorted((root / "ohlcv_1m").rglob(PARTITION_SIDECAR))
        }

    first_dir, second_dir = tmp_path / "one", tmp_path / "two"
    run(Settings(data_root=first_dir), FakeSource())
    run(Settings(data_root=second_dir), FakeSource())
    assert checksums(first_dir / "normalized") == checksums(second_dir / "normalized")


def test_no_resume_run_rewrites_a_revised_partition(tmp_path):
    settings = settings_for(tmp_path)
    run(settings, FakeSource(revision="sha256:rev1"))
    summary, _ = run(settings, FakeSource(revision="sha256:rev2"), resume=False)
    assert summary.partitions_written == 2, "a revised source must not be silently kept as stale"


def test_resumed_run_cannot_see_an_upstream_revision(tmp_path):
    # Documented limitation, not an oversight: noticing a revision requires
    # re-downloading the unit, which is precisely what resume avoids. Periodic
    # --no-resume runs are what catch source revisions.
    settings = settings_for(tmp_path)
    run(settings, FakeSource(revision="sha256:rev1"))
    source = FakeSource(revision="sha256:rev2")
    summary, _ = run(settings, source, resume=True)
    assert summary.units_skipped == 2 and source.fetched == []


def test_missing_unit_is_recorded_as_a_gap_not_an_error(tmp_path):
    class MissingSource(FakeSource):
        def fetch(self, spec, unit):
            raise NotFoundError("absent", status=404, url=unit.url)

    settings = settings_for(tmp_path)
    summary, manifest = run(settings, MissingSource())
    assert summary.units_missing == 2
    assert not manifest.errors
    assert all(gap["kind"] == "SOURCE_UNIT_ABSENT" for gap in manifest.gaps)


def test_ban_aborts_the_run(tmp_path):
    class BannedSource(FakeSource):
        def fetch(self, spec, unit):
            raise BannedError("banned", status=418, url=unit.url)

    with pytest.raises(BannedError):
        run(settings_for(tmp_path), BannedSource())
