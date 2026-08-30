"""Bybit v5 klines.

Two differences from Binance that the code has to respect: the response is
ordered newest-first, and it carries no trade count or taker-side split, so
those columns stay null rather than being invented.

The historical depth Bybit actually serves for 1-minute candles is unverified
(it is one of the open questions in the Phase 0A plan). :func:`probe_depth`
measures it instead of assuming it, and the answer belongs in the Data Source
Matrix as evidence.
"""

from __future__ import annotations

import json

from prepump.io.checksums import sha256_bytes
from prepump.logging import get_logger
from prepump.net.client import HttpClient
from prepump.normalize.ohlcv import RawKline
from prepump.normalize.schema import AvailabilityMethod
from prepump.normalize.symbols import SymbolSpec
from prepump.sources.base import FetchResult, FetchUnit, SourceDescriptor
from prepump.timeutils import MINUTE_MS, detect_epoch_ms, utc_now_ms

log = get_logger(__name__)

MAX_LIMIT = 1000
_CATEGORIES = {"spot": "spot", "linear_perpetual": "linear", "inverse_perpetual": "inverse"}


class BybitRestSource:
    def __init__(self, client: HttpClient, base_url: str, interval: str = "1m") -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.interval = interval
        self.interval_ms = MINUTE_MS
        self.descriptor = SourceDescriptor(
            name="bybit_rest_kline",
            exchange="bybit",
            availability_method=AvailabilityMethod.DERIVED_FROM_CLOSE,
            availability_delta_ms=1,
            availability_uncertainty_ms=2000,
            notes="Newest-first pagination; no trade_count or taker split; 1m depth must be measured.",
        )

    def supports(self, spec: SymbolSpec) -> bool:
        return spec.exchange == "bybit" and spec.market_type in _CATEGORIES

    @property
    def url(self) -> str:
        return f"{self.base_url}/v5/market/kline"

    def plan(self, spec: SymbolSpec, start_ms: int, end_ms: int) -> list[FetchUnit]:
        return [
            FetchUnit(
                key=f"{spec.exchange}:{spec.market_type}:{spec.symbol}:{start_ms}-{end_ms}",
                url=self.url,
                label=f"{start_ms}-{end_ms}",
            )
        ]

    def fetch(self, spec: SymbolSpec, unit: FetchUnit) -> FetchResult:
        start_ms, end_ms = (int(part) for part in unit.label.split("-"))
        klines = self.fetch_range(spec, start_ms, end_ms)
        revision = sha256_bytes(f"{unit.url}|{unit.label}|{len(klines)}".encode())
        return FetchResult(
            klines=klines,
            source_revision=f"endpoint:{revision[:32]}",
            meta={"url": unit.url, "start_ms": start_ms, "end_ms": end_ms},
        )

    def fetch_range(self, spec: SymbolSpec, start_ms: int, end_ms: int) -> list[RawKline]:
        out: dict[int, RawKline] = {}
        cursor_end = end_ms
        while cursor_end >= start_ms:
            params = {
                "category": _CATEGORIES[spec.market_type],
                "symbol": spec.symbol,
                "interval": "1",
                "start": start_ms,
                "end": cursor_end,
                "limit": MAX_LIMIT,
            }
            payload = json.loads(self.client.get(self.url, params=params).content)
            if payload.get("retCode") not in (0, None):
                raise ValueError(f"bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
            rows = (payload.get("result") or {}).get("list") or []
            if not rows:
                break
            batch = [_row_to_kline(row) for row in rows]
            for kline in batch:
                if start_ms <= kline.open_time_ms <= end_ms:
                    out[kline.open_time_ms] = kline
            oldest = min(k.open_time_ms for k in batch)
            next_end = oldest - 1
            if next_end >= cursor_end:
                raise ValueError(f"pagination did not advance at end={cursor_end} for {spec.symbol}")
            cursor_end = next_end
            if len(rows) < MAX_LIMIT:
                break
        return [out[key] for key in sorted(out)]

    def probe_depth(self, spec: SymbolSpec, max_lookback_days: int = 1200) -> dict:
        """Binary-search how far back 1-minute candles are actually served."""
        now = utc_now_ms()
        low, high = 0, max_lookback_days
        deepest: int | None = None
        while low <= high:
            mid = (low + high) // 2
            window_start = now - mid * 86_400_000
            rows = self.fetch_range(spec, window_start, window_start + 60 * MINUTE_MS)
            if rows:
                deepest = mid
                low = mid + 1
            else:
                high = mid - 1
        return {
            "symbol": spec.symbol,
            "market_type": spec.market_type,
            "deepest_days_with_1m_data": deepest,
            "probed_at_ms": now,
            "method": "binary search over one-hour probe windows",
        }


def _row_to_kline(row: list) -> RawKline:
    # [startTime, open, high, low, close, volume, turnover]
    open_ms = detect_epoch_ms(int(float(row[0])))
    return RawKline(
        open_time_ms=open_ms,
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        base_volume=float(row[5]),
        close_time_ms=open_ms + MINUTE_MS - 1,
        quote_volume=float(row[6]) if len(row) > 6 else None,
        trade_count=None,
        taker_buy_base=None,
        taker_buy_quote=None,
    )
