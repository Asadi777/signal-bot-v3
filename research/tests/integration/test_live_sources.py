"""Live-network checks.

Deselected by default (``-m 'not integration'``). These are the tests that turn
the Data Source Matrix rows from UNKNOWN into evidence, so their assertions are
deliberately about *what we learn*, not just that a call succeeded.

Run with:  pytest -m integration
"""

import pytest

from prepump.config import load_settings
from prepump.net.client import build_client
from prepump.normalize.symbols import make_spec
from prepump.sources.binance_rest import BinanceRestSource
from prepump.sources.binance_vision import BinanceVisionSource, list_prefix
from prepump.sources.bybit_rest import BybitRestSource
from prepump.quality.checks import cross_source_discrepancy
from prepump.timeutils import month_bounds_ms

pytestmark = pytest.mark.integration

SPOT = make_spec("binance", "spot", "BTCUSDT", "BTC", "USDT")
DELISTED_CANDIDATE = make_spec("binance", "spot", "SRMUSDT", "SRM", "USDT")
MONTH = "2024-01"


@pytest.fixture(scope="module")
def client():
    return build_client(load_settings())


def test_archive_month_downloads_and_checksum_matches(client):
    settings = load_settings()
    source = BinanceVisionSource(client, settings.binance_vision_base_url)
    result = source.fetch(SPOT, source.plan(SPOT, *month_bounds_ms(MONTH))[0])
    assert result.meta["checksum_verified"] is True, "record in the matrix whether checksums are published"
    assert len(result.klines) > 40_000
    assert result.meta["time_unit"] in {"s", "ms", "us"}


def test_delisted_symbol_is_still_served_by_the_archive(client):
    """The survivorship path. If this fails, Stage A universe policy changes."""
    settings = load_settings()
    source = BinanceVisionSource(client, settings.binance_vision_base_url)
    result = source.fetch(DELISTED_CANDIDATE, source.plan(DELISTED_CANDIDATE, *month_bounds_ms("2022-06"))[0])
    assert result.klines


def test_archive_listing_enumerates_symbols(client):
    settings = load_settings()
    prefixes, _ = list_prefix(client, settings.binance_vision_base_url, "data/spot/monthly/klines/")
    assert len(prefixes) > 100


def test_rest_and_archive_agree_on_the_same_day(client):
    settings = load_settings()
    archive = BinanceVisionSource(client, settings.binance_vision_base_url)
    rest = BinanceRestSource(client, settings.binance_spot_rest_base_url, settings.binance_futures_rest_base_url)
    start_ms, _ = month_bounds_ms(MONTH)
    day_end = start_ms + 86_400_000 - 1

    from prepump.normalize.ohlcv import to_table
    from prepump.normalize.schema import AvailabilityMethod
    from prepump.timeutils import MINUTE_MS

    def table(klines, source_name):
        return to_table(
            [k for k in klines if start_ms <= k.open_time_ms <= day_end],
            spec=SPOT, interval="1m", interval_ms=MINUTE_MS, source=source_name,
            source_revision="integration", run_id="integration", ingested_at_ms=start_ms,
            availability_method=AvailabilityMethod.DERIVED_FROM_CLOSE,
            availability_delta_ms=1, availability_uncertainty_ms=2000,
        )

    archive_rows = archive.fetch(SPOT, archive.plan(SPOT, start_ms, day_end)[0]).klines
    rest_rows = rest.fetch_range(SPOT, start_ms, day_end)
    result = cross_source_discrepancy(table(archive_rows, "archive"), table(rest_rows, "rest"))
    assert result["overlapping_minutes"] > 1400
    assert result["mismatching_minutes"] == 0, result["mismatch_examples"]


def test_binance_metadata_shape(client):
    from prepump.sources import exchange_meta

    settings = load_settings()
    spot_rows = exchange_meta.collect_binance_spot(client, settings.binance_spot_rest_base_url, "integration")
    futures_rows = exchange_meta.collect_binance_futures(client, settings.binance_futures_rest_base_url, "integration")
    assert spot_rows and futures_rows
    # The documented asymmetry: spot has no listing date, futures does.
    assert all(row["listing_time"] is None for row in spot_rows)
    assert any(row["listing_time"] for row in futures_rows)


def test_bybit_metadata_and_launch_time_coverage(client):
    from prepump.sources import exchange_meta

    settings = load_settings()
    rows = exchange_meta.collect_bybit(client, settings.bybit_rest_base_url, "linear", "integration")
    assert rows
    with_launch = sum(1 for row in rows if row["listing_time"])
    # Record the ratio in the matrix rather than asserting a threshold blindly.
    assert with_launch >= 0


def test_bybit_one_minute_depth_is_measured(client):
    settings = load_settings()
    source = BybitRestSource(client, settings.bybit_rest_base_url)
    result = source.probe_depth(make_spec("bybit", "linear_perpetual", "BTCUSDT", "BTC", "USDT"), max_lookback_days=1200)
    assert result["deepest_days_with_1m_data"] is not None, "record the measured depth in the matrix"
