"""Pagination boundaries for both REST sources.

The cases that actually break collectors: a page that is exactly the limit, a
short final page, an empty response, and a source that stops advancing.
"""

import json

import pytest

from prepump.net.client import HttpClient, Response
from prepump.net.ratelimit import TokenBucket
from prepump.normalize.symbols import make_spec
from prepump.sources.binance_rest import MAX_LIMIT, BinanceRestSource
from prepump.sources.bybit_rest import BybitRestSource
from prepump.timeutils import MINUTE_MS

START = 1_704_067_200_000
SPEC = make_spec("binance", "spot", "TESTUSDT", "TEST", "USDT")
BYBIT_SPEC = make_spec("bybit", "linear_perpetual", "TESTUSDT", "TEST", "USDT")


def binance_rows(start_ms, count):
    return [
        [start_ms + i * MINUTE_MS, "1", "2", "0.5", "1.5", "10", start_ms + i * MINUTE_MS + MINUTE_MS - 1, "15", 3, "5", "7", "0"]
        for i in range(count)
    ]


def client_for(pages):
    sequence = iter(pages)

    def transport(url, params, timeout):
        return Response(200, json.dumps(next(sequence)).encode(), {}, url)

    return HttpClient(bucket=TokenBucket(rate_per_second=1e6, sleeper=lambda s: None), transport=transport, sleeper=lambda s: None)


def test_short_page_ends_pagination():
    source = BinanceRestSource(client_for([binance_rows(START, 10)]), "http://a", "http://b")
    rows = source.fetch_range(SPEC, START, START + 100 * MINUTE_MS)
    assert len(rows) == 10


def test_exact_limit_page_triggers_another_request():
    pages = [binance_rows(START, MAX_LIMIT), binance_rows(START + MAX_LIMIT * MINUTE_MS, 5)]
    source = BinanceRestSource(client_for(pages), "http://a", "http://b")
    rows = source.fetch_range(SPEC, START, START + (MAX_LIMIT + 100) * MINUTE_MS)
    assert len(rows) == MAX_LIMIT + 5


def test_empty_page_ends_pagination_without_error():
    source = BinanceRestSource(client_for([[]]), "http://a", "http://b")
    assert source.fetch_range(SPEC, START, START + 10 * MINUTE_MS) == []


def test_rows_outside_the_requested_window_are_dropped():
    pages = [binance_rows(START - 5 * MINUTE_MS, 10)]
    source = BinanceRestSource(client_for(pages), "http://a", "http://b")
    rows = source.fetch_range(SPEC, START, START + 2 * MINUTE_MS)
    assert [r.open_time_ms for r in rows] == [START, START + MINUTE_MS, START + 2 * MINUTE_MS]


def test_non_advancing_source_raises_instead_of_looping():
    page = binance_rows(START, MAX_LIMIT)
    sequence = iter([page, page, page])

    def transport(url, params, timeout):
        return Response(200, json.dumps(next(sequence)).encode(), {}, url)

    client = HttpClient(bucket=TokenBucket(rate_per_second=1e6, sleeper=lambda s: None), transport=transport, sleeper=lambda s: None)
    source = BinanceRestSource(client, "http://a", "http://b")
    source.interval_ms = 0  # simulate a source that never moves the cursor
    with pytest.raises(ValueError, match="did not advance"):
        source.fetch_range(SPEC, START, START + 10_000 * MINUTE_MS)


def bybit_page(start_ms, count):
    """Bybit returns newest first, which is the trap this test guards."""
    rows = [
        [str(start_ms + i * MINUTE_MS), "1", "2", "0.5", "1.5", "10", "15"]
        for i in range(count)
    ]
    return {"retCode": 0, "result": {"list": list(reversed(rows))}}


def test_bybit_descending_pages_are_reassembled_in_order():
    # A full page means "there may be more"; the short page ends pagination.
    pages = [bybit_page(START + 500 * MINUTE_MS, 1000), bybit_page(START, 500)]
    sequence = iter(pages)

    def transport(url, params, timeout):
        return Response(200, json.dumps(next(sequence)).encode(), {}, url)

    client = HttpClient(bucket=TokenBucket(rate_per_second=1e6, sleeper=lambda s: None), transport=transport, sleeper=lambda s: None)
    source = BybitRestSource(client, "http://bybit")
    rows = source.fetch_range(BYBIT_SPEC, START, START + 1499 * MINUTE_MS)
    times = [r.open_time_ms for r in rows]
    assert times == sorted(times)
    assert len(times) == len(set(times)) == 1500


def test_bybit_error_code_raises():
    def transport(url, params, timeout):
        return Response(200, json.dumps({"retCode": 10001, "retMsg": "bad"}).encode(), {}, url)

    client = HttpClient(bucket=TokenBucket(rate_per_second=1e6, sleeper=lambda s: None), transport=transport, sleeper=lambda s: None)
    source = BybitRestSource(client, "http://bybit")
    with pytest.raises(ValueError, match="10001"):
        source.fetch_range(BYBIT_SPEC, START, START + MINUTE_MS)


def test_bybit_leaves_unavailable_columns_null():
    def transport(url, params, timeout):
        return Response(200, json.dumps(bybit_page(START, 1)).encode(), {}, url)

    client = HttpClient(bucket=TokenBucket(rate_per_second=1e6, sleeper=lambda s: None), transport=transport, sleeper=lambda s: None)
    rows = BybitRestSource(client, "http://bybit").fetch_range(BYBIT_SPEC, START, START + MINUTE_MS)
    assert rows[0].trade_count is None and rows[0].taker_buy_base is None
