"""Output layout.

Three trees, never mixed:

* ``raw/``        bytes exactly as fetched, plus a sidecar describing the fetch.
* ``normalized/`` Parquet in the project schema, Hive-partitioned.
* ``reports/``    derived, disposable, always regenerable from normalized.

The normalized partition (exchange, market_type, symbol, date) is the unit of
idempotency: it is written atomically and either exists complete or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PARTITION_FILE = "part-000.parquet"
PARTITION_SIDECAR = "_partition.json"


@dataclass(frozen=True)
class PartitionKey:
    exchange: str
    market_type: str
    symbol: str
    date: str  # YYYY-MM-DD, UTC

    def relative(self, dataset: str = "ohlcv_1m") -> Path:
        return Path(
            dataset,
            f"exchange={self.exchange}",
            f"market_type={self.market_type}",
            f"symbol={self.symbol}",
            f"date={self.date}",
        )


def partition_dir(normalized_root: Path, key: PartitionKey, dataset: str = "ohlcv_1m") -> Path:
    return normalized_root / key.relative(dataset)


def raw_path(raw_root: Path, source: str, market_type: str, symbol: str, interval: str, filename: str) -> Path:
    return raw_root / source / market_type / symbol / interval / filename


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
