"""Symbol identity.

The exchange-native string (``BTCUSDT``) is ambiguous across venues and market
types, so every record also carries a canonical id::

    binance:spot:BTC-USDT
    binance:linear_perpetual:BTC-USDT
    bybit:spot:BTC-USDT

Base/quote come from the universe config or exchange metadata whenever
possible. The suffix-based inference below is a fallback only, and records
which path was used so a downstream reader can distinguish the two.
"""

from __future__ import annotations

from dataclasses import dataclass

# Longest first: FDUSD must win over USD, USDT over USD.
KNOWN_QUOTES = (
    "FDUSD",
    "TUSD",
    "BUSD",
    "USDT",
    "USDC",
    "USDP",
    "DAI",
    "BTC",
    "ETH",
    "BNB",
    "EUR",
    "TRY",
    "BRL",
    "JPY",
    "GBP",
    "AUD",
    "USD",
)

MARKET_TYPES = ("spot", "linear_perpetual", "inverse_perpetual", "futures")


class SymbolError(ValueError):
    pass


@dataclass(frozen=True)
class SymbolSpec:
    exchange: str
    market_type: str
    symbol: str
    base_asset: str
    quote_asset: str
    inferred: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.market_type not in MARKET_TYPES:
            raise SymbolError(f"unknown market_type {self.market_type!r}, expected one of {MARKET_TYPES}")
        if not self.symbol or not self.base_asset or not self.quote_asset:
            raise SymbolError(f"incomplete symbol spec: {self!r}")

    @property
    def canonical(self) -> str:
        return f"{self.exchange}:{self.market_type}:{self.base_asset}-{self.quote_asset}"


def split_symbol(symbol: str) -> tuple[str, str]:
    """Best-effort split of an exchange-native symbol into (base, quote)."""
    upper = symbol.upper()
    for quote in KNOWN_QUOTES:
        if upper.endswith(quote) and len(upper) > len(quote):
            return upper[: -len(quote)], quote
    raise SymbolError(f"cannot infer base/quote for {symbol!r}; declare them explicitly")


def make_spec(
    exchange: str,
    market_type: str,
    symbol: str,
    base_asset: str | None = None,
    quote_asset: str | None = None,
    note: str = "",
) -> SymbolSpec:
    if base_asset and quote_asset:
        return SymbolSpec(exchange, market_type, symbol.upper(), base_asset.upper(), quote_asset.upper(), False, note)
    base, quote = split_symbol(symbol)
    return SymbolSpec(exchange, market_type, symbol.upper(), base, quote, True, note)
