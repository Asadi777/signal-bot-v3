"""Atomic, idempotent partition writes.

Write protocol, in order:

1. serialize to a temporary file in the destination directory;
2. fsync and ``os.replace`` it onto the final name (atomic on POSIX);
3. write the sidecar the same way.

A crash therefore leaves either no partition, or a partition without a sidecar.
Both are treated as "not written" by :func:`partition_state`, so a resumed run
rewrites them rather than trusting a possibly truncated file.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepump.io.checksums import content_checksum
from prepump.io.paths import PARTITION_FILE, PARTITION_SIDECAR, ensure_dir
from prepump.timeutils import utc_now

# Fixed writer settings: determinism requires that nothing about the encoding
# depends on the machine or the library defaults of the day.
_WRITER_KWARGS = dict(
    compression="zstd",
    compression_level=3,
    version="2.6",
    row_group_size=100_000,
    write_statistics=True,
    store_schema=True,
)


@dataclass(frozen=True)
class PartitionMeta:
    rows: int
    content_sha256: str
    source: str
    source_revision: str
    schema_version: str
    collector_version: str
    run_id: str
    written_at: str

    def to_dict(self) -> dict:
        return dict(
            rows=self.rows,
            content_sha256=self.content_sha256,
            source=self.source,
            source_revision=self.source_revision,
            schema_version=self.schema_version,
            collector_version=self.collector_version,
            run_id=self.run_id,
            written_at=self.written_at,
        )


def _atomic_write(path: Path, writer) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        with tmp.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_partition(
    table: pa.Table,
    part_dir: Path,
    *,
    source: str,
    source_revision: str,
    schema_version: str,
    collector_version: str,
    run_id: str,
) -> PartitionMeta:
    meta = PartitionMeta(
        rows=table.num_rows,
        content_sha256=content_checksum(table),
        source=source,
        source_revision=source_revision,
        schema_version=schema_version,
        collector_version=collector_version,
        run_id=run_id,
        written_at=utc_now().isoformat(timespec="milliseconds"),
    )
    _atomic_write(part_dir / PARTITION_FILE, lambda tmp: pq.write_table(table, tmp, **_WRITER_KWARGS))
    _atomic_write(
        part_dir / PARTITION_SIDECAR,
        lambda tmp: tmp.write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=True) + "\n"),
    )
    return meta


def partition_state(part_dir: Path) -> PartitionMeta | None:
    """Return the sidecar of a complete partition, or None if it needs writing."""
    data_file = part_dir / PARTITION_FILE
    sidecar = part_dir / PARTITION_SIDECAR
    if not data_file.exists() or not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text())
        return PartitionMeta(**payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def is_current(meta: PartitionMeta | None, *, source_revision: str, schema_version: str, collector_version: str) -> bool:
    """True when an existing partition can be reused by a resumed run.

    A changed source revision, schema version or collector version all
    invalidate the partition: reusing it would silently mix contracts inside
    one dataset.
    """
    if meta is None:
        return False
    return (
        meta.source_revision == source_revision
        and meta.schema_version == schema_version
        and meta.collector_version == collector_version
    )


def read_partition(part_dir: Path) -> pa.Table:
    return pq.read_table(part_dir / PARTITION_FILE)


def read_dataset(root: Path, dataset: str = "ohlcv_1m") -> pa.Table | None:
    """Read every complete partition of a dataset into one table."""
    base = root / dataset
    if not base.exists():
        return None
    tables = [
        read_partition(sidecar.parent)
        for sidecar in sorted(base.rglob(PARTITION_SIDECAR))
        if (sidecar.parent / PARTITION_FILE).exists()
    ]
    if not tables:
        return None
    return pa.concat_tables(tables)
