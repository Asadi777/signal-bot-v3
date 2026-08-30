"""Raw kline rows -> the normalized contract.

Row-level data-quality flags are attached here because they are cheap and
purely local. Dataset-level findings (gaps, duplicates, ordering) belong to
:mod:`prepump.quality` because they need the whole partition or more.

Nothing in this module ever invents a value: no gap filling, no interpolation,
no carrying a price forward. A missing minute stays missing and is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from prepump.normalize.schema import OHLCV_SCHEMA, AvailabilityMethod, DQFlag
from prepump.normalize.symbols import SymbolSpec
from prepump.timeutils import MINUTE_MS
from prepump.version import COLLECTOR_VERSION, SCHEMA_VERSION


@dataclass(frozen=True)
class RawKline:
    """One candle as served by a source, with units already normalized to ms."""

    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    base_volume: float
    close_time_ms: int
    quote_volume: float | None = None
    trade_count: int | None = None
    taker_buy_base: float | None = None
    taker_buy_quote: float | None = None


def row_flags(kline: RawKline, interval_ms: int, spec: SymbolSpec) -> list[str]:
    flags: list[str] = []
    prices = (kline.open, kline.high, kline.low, kline.close)
    if any(p <= 0 for p in prices):
        flags.append(DQFlag.NON_POSITIVE_PRICE)
    if not (kline.low <= min(kline.open, kline.close) and max(kline.open, kline.close) <= kline.high):
        flags.append(DQFlag.OHLC_VIOLATION)
    if kline.base_volume < 0 or (kline.quote_volume is not None and kline.quote_volume < 0):
        flags.append(DQFlag.NEGATIVE_VOLUME)
    if kline.base_volume == 0 and kline.high != kline.low:
        # Price moved inside a candle that reports no traded volume: either the
        # source is inconsistent or the candle is synthetic. Never silently
        # accepted as a normal observation.
        flags.append(DQFlag.ZERO_VOLUME_WITH_RANGE)
    if kline.open_time_ms % interval_ms != 0:
        flags.append(DQFlag.NOT_MINUTE_ALIGNED)
    if kline.close_time_ms != kline.open_time_ms + interval_ms - 1:
        flags.append(DQFlag.CLOSE_TIME_MISMATCH)
    if kline.quote_volume is None:
        flags.append(DQFlag.QUOTE_VOLUME_MISSING)
    if spec.inferred:
        flags.append(DQFlag.INFERRED_SYMBOL_SPLIT)
    return flags


def to_table(
    klines: list[RawKline],
    *,
    spec: SymbolSpec,
    interval: str,
    interval_ms: int,
    source: str,
    source_revision: str,
    run_id: str,
    ingested_at_ms: int,
    availability_method: AvailabilityMethod,
    availability_delta_ms: int,
    availability_uncertainty_ms: int,
) -> pa.Table:
    """Build a schema-valid table. Rows are sorted by ``event_time``."""
    ordered = sorted(klines, key=lambda k: k.open_time_ms)
    n = len(ordered)

    columns = {
        "exchange": [spec.exchange] * n,
        "market_type": [spec.market_type] * n,
        "symbol": [spec.symbol] * n,
        "symbol_canonical": [spec.canonical] * n,
        "base_asset": [spec.base_asset] * n,
        "quote_asset": [spec.quote_asset] * n,
        "interval": [interval] * n,
        "event_time": [k.open_time_ms for k in ordered],
        "event_time_close": [k.close_time_ms for k in ordered],
        "available_at": [k.close_time_ms + availability_delta_ms for k in ordered],
        "available_at_method": [str(availability_method)] * n,
        "available_at_uncertainty_ms": [availability_uncertainty_ms] * n,
        "ingested_at": [ingested_at_ms] * n,
        "open": [k.open for k in ordered],
        "high": [k.high for k in ordered],
        "low": [k.low for k in ordered],
        "close": [k.close for k in ordered],
        "base_volume": [k.base_volume for k in ordered],
        "quote_volume": [k.quote_volume for k in ordered],
        "trade_count": [k.trade_count for k in ordered],
        "taker_buy_base": [k.taker_buy_base for k in ordered],
        "taker_buy_quote": [k.taker_buy_quote for k in ordered],
        "source": [source] * n,
        "source_revision": [source_revision] * n,
        "collector_version": [COLLECTOR_VERSION] * n,
        "schema_version": [SCHEMA_VERSION] * n,
        "run_id": [run_id] * n,
        "dq_flags": [row_flags(k, interval_ms, spec) for k in ordered],
    }

    arrays = []
    for field in OHLCV_SCHEMA:
        arrays.append(pa.array(columns[field.name], type=field.type))
    return pa.Table.from_arrays(arrays, schema=OHLCV_SCHEMA)


def split_by_utc_date(klines: list[RawKline]) -> dict[str, list[RawKline]]:
    """Group klines by their UTC calendar date, the partition unit."""
    from prepump.timeutils import date_str

    out: dict[str, list[RawKline]] = {}
    for kline in klines:
        out.setdefault(date_str(kline.open_time_ms), []).append(kline)
    return out


MINUTE_INTERVAL_MS = MINUTE_MS
