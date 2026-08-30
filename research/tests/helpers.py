"""Shared test builders."""

from __future__ import annotations

from prepump.normalize import ohlcv
from prepump.normalize.ohlcv import RawKline
from prepump.normalize.schema import AvailabilityMethod
from prepump.normalize.symbols import make_spec
from prepump.timeutils import MINUTE_MS

START_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z
SPEC = make_spec("binance", "spot", "TESTUSDT", "TEST", "USDT")


def kline(index: int, **overrides) -> RawKline:
    open_time = overrides.pop("open_time_ms", START_MS + index * MINUTE_MS)
    base = dict(
        open_time_ms=open_time,
        open=100.0 + index,
        high=101.0 + index,
        low=99.0 + index,
        close=100.5 + index,
        base_volume=10.0,
        close_time_ms=open_time + MINUTE_MS - 1,
        quote_volume=1000.0,
        trade_count=7,
        taker_buy_base=5.0,
        taker_buy_quote=500.0,
    )
    base.update(overrides)
    return RawKline(**base)


def make_table(klines=None, spec=SPEC, run_id="test-run", ingested_at_ms=START_MS):
    klines = klines if klines is not None else [kline(i) for i in range(5)]
    return ohlcv.to_table(
        klines,
        spec=spec,
        interval="1m",
        interval_ms=MINUTE_MS,
        source="unit_test",
        source_revision="sha256:deadbeef",
        run_id=run_id,
        ingested_at_ms=ingested_at_ms,
        availability_method=AvailabilityMethod.DERIVED_FROM_CLOSE,
        availability_delta_ms=1,
        availability_uncertainty_ms=2000,
    )
