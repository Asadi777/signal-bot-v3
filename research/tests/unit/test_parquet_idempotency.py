import json

from prepump.io.parquet import is_current, partition_state, read_partition, write_partition
from prepump.io.paths import PARTITION_FILE, PARTITION_SIDECAR
from prepump.version import COLLECTOR_VERSION, SCHEMA_VERSION
from tests.helpers import make_table

WRITE_ARGS = dict(
    source="unit_test",
    source_revision="sha256:deadbeef",
    schema_version=SCHEMA_VERSION,
    collector_version=COLLECTOR_VERSION,
    run_id="run-1",
)


def test_write_then_read_roundtrip(tmp_path):
    table = make_table()
    meta = write_partition(table, tmp_path / "p", **WRITE_ARGS)
    assert meta.rows == table.num_rows
    assert read_partition(tmp_path / "p").num_rows == table.num_rows


def test_rerun_produces_the_same_content_checksum(tmp_path):
    # Different run_id and ingested_at, identical source data: the content
    # checksum must not move, or "deterministic" means nothing.
    first = write_partition(make_table(run_id="run-1", ingested_at_ms=1), tmp_path / "a", **WRITE_ARGS)
    second_args = {**WRITE_ARGS, "run_id": "run-2"}
    second = write_partition(make_table(run_id="run-2", ingested_at_ms=999), tmp_path / "b", **second_args)
    assert first.content_sha256 == second.content_sha256


def test_different_data_changes_the_checksum(tmp_path):
    from tests.helpers import kline

    first = write_partition(make_table(), tmp_path / "a", **WRITE_ARGS)
    second = write_partition(make_table([kline(0, close=1.0)]), tmp_path / "b", **WRITE_ARGS)
    assert first.content_sha256 != second.content_sha256


def test_partition_without_sidecar_counts_as_unwritten(tmp_path):
    part = tmp_path / "p"
    write_partition(make_table(), part, **WRITE_ARGS)
    (part / PARTITION_SIDECAR).unlink()
    assert partition_state(part) is None


def test_partition_with_corrupt_sidecar_counts_as_unwritten(tmp_path):
    part = tmp_path / "p"
    write_partition(make_table(), part, **WRITE_ARGS)
    (part / PARTITION_SIDECAR).write_text("{not json")
    assert partition_state(part) is None


def test_is_current_requires_matching_versions_and_revision(tmp_path):
    part = tmp_path / "p"
    write_partition(make_table(), part, **WRITE_ARGS)
    state = partition_state(part)
    assert is_current(state, source_revision="sha256:deadbeef", schema_version=SCHEMA_VERSION, collector_version=COLLECTOR_VERSION)
    assert not is_current(state, source_revision="sha256:other", schema_version=SCHEMA_VERSION, collector_version=COLLECTOR_VERSION)
    assert not is_current(state, source_revision="sha256:deadbeef", schema_version="ohlcv_1m.v2", collector_version=COLLECTOR_VERSION)
    assert not is_current(state, source_revision="sha256:deadbeef", schema_version=SCHEMA_VERSION, collector_version="9.9.9")
    assert not is_current(None, source_revision="x", schema_version="y", collector_version="z")


def test_no_temporary_files_are_left_behind(tmp_path):
    part = tmp_path / "p"
    write_partition(make_table(), part, **WRITE_ARGS)
    names = sorted(child.name for child in part.iterdir())
    assert names == sorted([PARTITION_FILE, PARTITION_SIDECAR])


def test_sidecar_records_the_provenance_a_reader_needs(tmp_path):
    part = tmp_path / "p"
    write_partition(make_table(), part, **WRITE_ARGS)
    payload = json.loads((part / PARTITION_SIDECAR).read_text())
    assert {"rows", "content_sha256", "source", "source_revision", "schema_version", "collector_version", "run_id", "written_at"} <= set(payload)
