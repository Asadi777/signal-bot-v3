"""Binance REST klines.

Not the backfill path. This exists for two jobs the archive cannot do:
fetching the current, not-yet-published month, and serving as an independent
endpoint to cross-check the archive against (deliverable F asks for exactly
that comparison).

Pagination walks forward by ``startTime``; the last returned open time plus one
interval is the next start, which is resumable and cannot loop forever because
progress is asserted on every page.
"""

from __future__ import annotations

from prepump.io.checksums import sha256_bytes
from prepump.logging import get_logger
from prepump.net.client import HttpClient
from prepump.normalize.ohlcv import RawKline
from prepump.normalize.schema import AvailabilityMethod
from prepump.normalize.symbols import SymbolSpec
from prepump.sources.base import FetchResult, FetchUnit, SourceDescriptor
from prepump.timeutils import MINUTE_MS, detect_epoch_ms

log = get_logger(__name__)

MAX_LIMIT = 1000
_PATHS = {"spot": "/api/v3/klines", "linear_perpetual": "/fapi/v1/klines"}


class BinanceRestSource:
    def __init__(self, client: HttpClient, spot_base_url: str, futures_base_url: str, interval: str = "1m") -> None:
        self.client = client
        self.spot_base_url = spot_base_url.rstrip("/")
        self.futures_base_url = futures_base_url.rstrip("/")
        self.interval = interval
        self.interval_ms = MINUTE_MS
        self.descriptor = SourceDescriptor(
            name="binance_rest_klines",
            exchange="binance",
            availability_method=AvailabilityMethod.DERIVED_FROM_CLOSE,
            availability_delta_ms=1,
            availability_uncertainty_ms=2000,
            notes="Weight-limited REST endpoint; used for the current month and cross-checks only.",
        )

    def supports(self, spec: SymbolSpec) -> bool:
        return spec.exchange == "binance" and spec.market_type in _PATHS

    def _url(self, spec: SymbolSpec) -> str:
        base = self.spot_base_url if spec.market_type == "spot" else self.futures_base_url
        return base + _PATHS[spec.market_type]

    def plan(self, spec: SymbolSpec, start_ms: int, end_ms: int) -> list[FetchUnit]:
        """One unit per range; pagination happens inside :meth:`fetch`."""
        return [
            FetchUnit(
                key=f"{spec.exchange}:{spec.market_type}:{spec.symbol}:{start_ms}-{end_ms}",
                url=self._url(spec),
                params={"symbol": spec.symbol, "interval": self.interval, "limit": MAX_LIMIT},
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
            meta={"url": unit.url, "start_ms": start_ms, "end_ms": end_ms, "pages": self._last_pages},
        )

    _last_pages = 0

    def fetch_range(self, spec: SymbolSpec, start_ms: int, end_ms: int) -> list[RawKline]:
        url = self._url(spec)
        out: list[RawKline] = []
        cursor = start_ms
        pages = 0
        while cursor <= end_ms:
            params = {
                "symbol": spec.symbol,
                "interval": self.interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": MAX_LIMIT,
            }
            response = self.client.get(url, params=params)
            self._log_weight(response.headers)
            rows = response.content and _json(response.content) or []
            pages += 1
            if not rows:
                break
            batch = [_row_to_kline(row) for row in rows]
            out.extend(k for k in batch if start_ms <= k.open_time_ms <= end_ms)
            next_cursor = batch[-1].open_time_ms + self.interval_ms
            if next_cursor <= cursor:
                # Defensive: a source that stops advancing would spin forever.
                raise ValueError(f"pagination did not advance at cursor={cursor} for {spec.symbol}")
            cursor = next_cursor
            if len(rows) < MAX_LIMIT:
                break
        self._last_pages = pages
        return out

    @staticmethod
    def _log_weight(headers: dict) -> None:
        used = headers.get("x-mbx-used-weight-1m") or headers.get("X-MBX-USED-WEIGHT-1M")
        if used:
            log.info("binance weight", extra={"used_weight_1m": used})


def _json(content: bytes):
    import json

    return json.loads(content)


def _row_to_kline(row: list) -> RawKline:
    return RawKline(
        open_time_ms=detect_epoch_ms(int(row[0])),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        base_volume=float(row[5]),
        close_time_ms=detect_epoch_ms(int(row[6])),
        quote_volume=float(row[7]),
        trade_count=int(row[8]),
        taker_buy_base=float(row[9]),
        taker_buy_quote=float(row[10]),
    )
