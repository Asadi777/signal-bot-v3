import pytest

from prepump.normalize.symbols import SymbolError, make_spec, split_symbol


def test_explicit_assets_are_not_inferred():
    spec = make_spec("binance", "spot", "btcusdt", "BTC", "USDT")
    assert spec.symbol == "BTCUSDT"
    assert spec.canonical == "binance:spot:BTC-USDT"
    assert spec.inferred is False


def test_inference_prefers_the_longest_quote():
    # FDUSD must not be split as F + DUSD or ...USD.
    assert split_symbol("ETHFDUSD") == ("ETH", "FDUSD")
    assert split_symbol("BTCUSDT") == ("BTC", "USDT")


def test_inference_is_flagged_on_the_spec():
    assert make_spec("binance", "spot", "BTCUSDT").inferred is True


def test_unknown_quote_raises_rather_than_guessing():
    with pytest.raises(SymbolError):
        split_symbol("WEIRDPAIR1")


def test_market_type_is_validated():
    with pytest.raises(SymbolError):
        make_spec("binance", "options", "BTCUSDT", "BTC", "USDT")


def test_same_symbol_differs_across_market_types():
    spot = make_spec("binance", "spot", "BTCUSDT", "BTC", "USDT")
    perp = make_spec("binance", "linear_perpetual", "BTCUSDT", "BTC", "USDT")
    assert spot.canonical != perp.canonical
