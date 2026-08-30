"""End-to-end run against the offline fixture archive.

Same collector code as a live run: only the base URL changes, from https to
file://. This is what makes the acceptance properties checkable in CI without
network access or redistributed exchange data.
"""

import json

from prepump import backfill as backfill_mod
from prepump import fixtures as fixtures_mod
from prepump.config import Settings
from prepump.io.parquet import read_dataset
from prepump.io.paths import PARTITION_SIDECAR
from prepump.manifest import RunManifest
from prepump.net.client import build_client
from prepump.normalize.schema import validate_table
from prepump.quality.report import build_report, render_markdown
from prepump.sources.binance_vision import BinanceVisionSource
from prepump.timeutils import month_bounds_ms
from prepump.universe import load_universe
from pathlib import Path

UNIVERSE = Path(__file__).resolve().parents[2] / "config" / "smoke_universe_offline.yaml"
MONTHS = ["2024-01"]


def build(tmp_path):
    fixture_root = tmp_path / "fixtures"
    fixtures_mod.build_archive(fixture_root, MONTHS)
    settings = Settings(data_root=tmp_path / "data", binance_vision_base_url=fixtures_mod.fixture_base_url(fixture_root))
    return settings


def run_backfill(settings, resume=True):
    universe = load_universe(UNIVERSE)
    manifest = RunManifest.start("offline-e2e", settings.config_hash())
    client = build_client(settings)
    source = BinanceVisionSource(client, settings.binance_vision_base_url)
    start_ms, _ = month_bounds_ms(MONTHS[0])
    _, end_ms = month_bounds_ms(MONTHS[-1])
    for spec in universe.specs:
        backfill_mod.backfill_symbol(settings, source, spec, start_ms, end_ms, manifest, resume=resume)
    return manifest


def checksums(settings):
    return {
        str(p.parent.relative_to(settings.normalized_dir)): json.loads(p.read_text())["content_sha256"]
        for p in sorted((settings.normalized_dir / "ohlcv_1m").rglob(PARTITION_SIDECAR))
    }


def test_offline_pipeline_is_idempotent_and_deterministic(tmp_path):
    settings = build(tmp_path)
    first = run_backfill(settings)
    before = checksums(settings)
    assert not first.errors
    assert len(before) > 0

    second = run_backfill(settings)
    assert second.partitions_written == [], "a completed range must not be rewritten"
    assert checksums(settings) == before


def test_dataset_is_schema_valid_and_covers_every_symbol(tmp_path):
    settings = build(tmp_path)
    run_backfill(settings)
    table = read_dataset(settings.normalized_dir)
    validate_table(table)
    assert set(table.column("symbol").to_pylist()) == {"FIXTUSDT", "GAPYUSDT", "DEADUSDT", "PERPUSDT"}
    assert set(table.column("market_type").to_pylist()) == {"spot", "linear_perpetual"}


def test_report_finds_the_injected_defects(tmp_path):
    settings = build(tmp_path)
    manifest = run_backfill(settings)
    table = read_dataset(settings.normalized_dir)
    report = build_report(table, settings.normalized_dir, run_id=manifest.run_id)

    assert report["totals"]["duplicate_rows"] >= 1
    assert report["dq_flag_totals"].get("OHLC_VIOLATION", 0) >= 1
    assert report["dq_flag_totals"].get("ZERO_VOLUME_WITH_RANGE", 0) >= 1

    by_symbol = {item["symbol"]: item for item in report["per_symbol"]}
    # The thin symbol is mostly one-minute holes; the dense one has a real hole.
    assert by_symbol["GAPYUSDT"]["gap_runs"].get("LIKELY_NO_TRADE", 0) > 100
    assert by_symbol["FIXTUSDT"]["longest_gap_minutes"] > 15
    # Delisting: the series stops well before month end.
    assert by_symbol["DEADUSDT"]["dates_present"] <= 10

    text = render_markdown(report)
    assert "# Data Quality Report" in text and "Gap classification" in text


def test_interrupted_partition_is_rewritten_not_trusted(tmp_path):
    settings = build(tmp_path)
    run_backfill(settings)
    victim = sorted((settings.normalized_dir / "ohlcv_1m").rglob(PARTITION_SIDECAR))[0]
    victim.unlink()  # simulates a crash between the data file and its sidecar
    manifest = run_backfill(settings)
    assert len(manifest.partitions_written) == 1


def test_fixture_archives_are_byte_identical_across_builds(tmp_path):
    # The source_revision of every row is the digest of the archive file, so a
    # generator that stamped a build time into the zip would make the pipeline
    # look non-deterministic when it is not.
    first = fixtures_mod.build_archive(tmp_path / "a", MONTHS)
    second = fixtures_mod.build_archive(tmp_path / "b", MONTHS)
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]


def test_content_checksums_match_across_independent_roots(tmp_path):
    one, two = build(tmp_path / "one"), build(tmp_path / "two")
    run_backfill(one)
    run_backfill(two)
    assert checksums(one) == checksums(two)
