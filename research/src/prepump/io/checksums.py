"""Content hashing.

Two different hashes are used and must not be confused:

* ``sha256_bytes`` fingerprints a raw artifact exactly as the source served it.
* ``content_checksum`` fingerprints the *deterministic* part of a normalized
  table. It deliberately excludes ``ingested_at`` and ``run_id`` so that two
  runs of the same code over the same source revision compare equal. Those two
  columns are the only permitted sources of run-to-run variation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa

VOLATILE_COLUMNS = ("ingested_at", "run_id")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def content_checksum(table: pa.Table) -> str:
    """Deterministic hash of a normalized table, ignoring volatile columns."""
    keep = [name for name in table.schema.names if name not in VOLATILE_COLUMNS]
    projected = table.select(keep)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, projected.schema) as writer:
        writer.write_table(projected, max_chunksize=len(projected) or 1)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()
