"""Offline fixture archive.

Builds a local tree that mirrors the Binance public-archive layout exactly, so
the real collector runs against ``file://`` with no code path of its own. That
matters twice over: the pipeline is testable end to end without network access,
and the shipped sample dataset contains no redistributed exchange data, which
keeps the licensing question out of the repository entirely.

The generated series are deterministic given a seed and deliberately contain
the defects the quality report is supposed to find:

* a low-liquidity symbol with scattered no-trade minutes and one multi-hour hole;
* a symbol whose data stops mid-month, standing in for a delisting;
* one duplicated candle, one OHLC violation and one zero-volume candle with a
  non-zero price range.
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

from prepump.io.paths import ensure_dir
from prepump.timeutils import MINUTE_MS, month_bounds_ms

SYNTHETIC_MARKER = "SYNTHETIC-FIXTURE-NOT-REAL-MARKET-DATA"


@dataclass(frozen=True)
class FixtureSymbol:
    symbol: str
    market_type: str
    start_price: float
    volatility: float
    liquidity: float
    missing_rate: float = 0.0
    stops_at_day: int | None = None
    inject_defects: bool = False


DEFAULT_SYMBOLS = (
    FixtureSymbol("FIXTUSDT", "spot", 100.0, 0.0012, 1500.0, missing_rate=0.0, inject_defects=True),
    FixtureSymbol("GAPYUSDT", "spot", 0.045, 0.004, 12.0, missing_rate=0.35),
    FixtureSymbol("DEADUSDT", "spot", 2.5, 0.003, 60.0, missing_rate=0.05, stops_at_day=10),
    FixtureSymbol("PERPUSDT", "linear_perpetual", 12.0, 0.002, 400.0, missing_rate=0.01),
)

_MARKET_PATHS = {"spot": "spot", "linear_perpetual": "futures/um"}


def _candles(symbol: FixtureSymbol, month: str, seed: int) -> list[list]:
    start_ms, end_ms = month_bounds_ms(month)
    rng = random.Random(f"{symbol.symbol}-{month}-{seed}")
    price = symbol.start_price * (1 + rng.uniform(-0.05, 0.05))
    rows: list[list] = []
    open_time = start_ms
    day_one = start_ms
    while open_time <= end_ms:
        if symbol.stops_at_day is not None and (open_time - day_one) // 86_400_000 >= symbol.stops_at_day:
            break
        if rng.random() < symbol.missing_rate:
            open_time += MINUTE_MS
            continue
        drift = rng.gauss(0, symbol.volatility)
        open_price = price
        close_price = max(1e-8, open_price * (1 + drift))
        high = max(open_price, close_price) * (1 + abs(rng.gauss(0, symbol.volatility / 2)))
        low = min(open_price, close_price) * (1 - abs(rng.gauss(0, symbol.volatility / 2)))
        volume = max(0.0, rng.gauss(symbol.liquidity, symbol.liquidity / 3))
        quote_volume = volume * (high + low) / 2
        trades = max(0, int(rng.gauss(volume / 3, volume / 9)))
        taker_base = volume * rng.uniform(0.3, 0.7)
        rows.append(
            [
                open_time,
                f"{open_price:.8f}",
                f"{high:.8f}",
                f"{low:.8f}",
                f"{close_price:.8f}",
                f"{volume:.8f}",
                open_time + MINUTE_MS - 1,
                f"{quote_volume:.8f}",
                trades,
                f"{taker_base:.8f}",
                f"{taker_base * (high + low) / 2:.8f}",
                "0",
            ]
        )
        price = close_price
        open_time += MINUTE_MS

    if symbol.inject_defects and len(rows) > 500:
        # A long no-trade hole that is clearly not "nobody traded for a minute".
        del rows[200:380]
        # An exact duplicate of one candle, as some source revisions have shipped.
        rows.insert(120, list(rows[119]))
        # high below close: impossible, must be flagged not silently accepted.
        broken = rows[400]
        broken[2] = f"{float(broken[4]) * 0.5:.8f}"
        # Zero volume while the price range is non-zero.
        zero = rows[420]
        zero[5] = "0.00000000"
        zero[7] = "0.00000000"
    return rows


def build_archive(root: Path, months: list[str], symbols=DEFAULT_SYMBOLS, seed: int = 42) -> list[Path]:
    """Write the fixture archive tree; returns the created zip paths."""
    written: list[Path] = []
    for symbol in symbols:
        market_path = _MARKET_PATHS[symbol.market_type]
        for month in months:
            rows = _candles(symbol, month, seed)
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerows(rows)
            csv_name = f"{symbol.symbol}-1m-{month}.csv"
            zip_name = f"{symbol.symbol}-1m-{month}.zip"
            target_dir = root / "data" / market_path / "monthly" / "klines" / symbol.symbol / "1m"
            ensure_dir(target_dir)
            zip_path = target_dir / zip_name
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
                # A default writestr stamps the current time into the zip
                # header, which would make the archive bytes -- and therefore
                # the source_revision derived from them -- differ on every
                # build. Real archive files on the exchange host are fixed
                # bytes, so the fixture must be too, or it would report a
                # determinism failure that the pipeline does not have.
                entry = zipfile.ZipInfo(csv_name, date_time=(2020, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o600 << 16
                archive.writestr(entry, buffer.getvalue())
            data = payload.getvalue()
            zip_path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            (target_dir / (zip_name + ".CHECKSUM")).write_text(f"{digest}  {zip_name}\n")
            written.append(zip_path)
    (root / "README.txt").write_text(
        f"{SYNTHETIC_MARKER}\n\nGenerated by prepump.fixtures for offline pipeline testing.\n"
        "Layout mirrors the Binance public data archive. Values are synthetic and\n"
        "must never be used for research conclusions.\n"
    )
    return written


def fixture_base_url(root: Path) -> str:
    return "file://" + str(Path(root).resolve())
