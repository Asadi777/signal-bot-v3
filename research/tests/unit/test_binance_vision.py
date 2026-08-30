import csv
import hashlib
import io
import zipfile

import pytest

from prepump.net.client import HttpClient, NotFoundError, Response
from prepump.net.ratelimit import TokenBucket
from prepump.normalize.symbols import make_spec
from prepump.sources.binance_vision import BinanceVisionSource, parse_kline_zip
from prepump.timeutils import MINUTE_MS

START = 1_704_067_200_000
SPEC = make_spec("binance", "spot", "TESTUSDT", "TEST", "USDT")
HEADER = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_base", "taker_buy_quote", "ignore"]


def build_zip(rows, header=False, name="TESTUSDT-1m-2024-01.csv"):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if header:
        writer.writerow(HEADER)
    writer.writerows(rows)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(name, buffer.getvalue())
    return payload.getvalue()


def rows(count=3, scale=1):
    return [
        [
            (START + i * MINUTE_MS) * scale,
            "1.0", "2.0", "0.5", "1.5", "10.0",
            (START + i * MINUTE_MS + MINUTE_MS - 1) * scale,
            "15.0", 3, "5.0", "7.0", "0",
        ]
        for i in range(count)
    ]


def test_parses_without_header():
    klines, meta = parse_kline_zip(build_zip(rows()))
    assert len(klines) == 3
    assert meta["header_detected"] is False
    assert klines[0].open_time_ms == START


def test_parses_with_header():
    klines, meta = parse_kline_zip(build_zip(rows(), header=True))
    assert len(klines) == 3
    assert meta["header_detected"] is True


def test_detects_microsecond_timestamps():
    klines, meta = parse_kline_zip(build_zip(rows(scale=1000)))
    assert meta["time_unit"] == "us"
    assert klines[0].open_time_ms == START


def test_empty_archive_yields_no_rows():
    klines, meta = parse_kline_zip(build_zip([]))
    assert klines == [] and meta["rows"] == 0


def test_multiple_csvs_are_rejected():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("a.csv", "1,2\n")
        archive.writestr("b.csv", "1,2\n")
    with pytest.raises(ValueError, match="exactly one CSV"):
        parse_kline_zip(payload.getvalue())


def make_source(payload, checksum: str | None):
    def transport(url, params, timeout):
        if url.endswith(".CHECKSUM"):
            if checksum is None:
                return Response(404, b"", {}, url)
            return Response(200, f"{checksum}  file.zip\n".encode(), {}, url)
        return Response(200, payload, {}, url)

    client = HttpClient(bucket=TokenBucket(rate_per_second=1e6, sleeper=lambda s: None), transport=transport, sleeper=lambda s: None)
    return BinanceVisionSource(client, "https://archive.test")


def test_fetch_verifies_the_published_checksum():
    payload = build_zip(rows())
    source = make_source(payload, hashlib.sha256(payload).hexdigest())
    unit = source.plan(SPEC, START, START)[0]
    result = source.fetch(SPEC, unit)
    assert len(result.klines) == 3
    assert result.source_revision == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert result.meta["checksum_verified"] is True


def test_checksum_mismatch_aborts_before_any_output():
    payload = build_zip(rows())
    source = make_source(payload, "0" * 64)
    unit = source.plan(SPEC, START, START)[0]
    with pytest.raises(ValueError, match="checksum mismatch"):
        source.fetch(SPEC, unit)


def test_missing_checksum_is_recorded_not_fatal():
    payload = build_zip(rows())
    source = make_source(payload, None)
    result = source.fetch(SPEC, source.plan(SPEC, START, START)[0])
    assert result.meta["checksum_verified"] is False


def test_plan_covers_every_month_in_the_range():
    from prepump.timeutils import dt_to_ms
    from datetime import datetime, timezone

    end = dt_to_ms(datetime(2024, 3, 15, tzinfo=timezone.utc))
    units = BinanceVisionSource(None, "https://archive.test").plan(SPEC, START, end)
    assert [u.label for u in units] == ["2024-01", "2024-02", "2024-03"]


def test_url_layout_differs_per_market_type():
    source = BinanceVisionSource(None, "https://archive.test")
    perp = make_spec("binance", "linear_perpetual", "TESTUSDT", "TEST", "USDT")
    assert "/data/spot/monthly/klines/" in source.file_url(SPEC, "2024-01")
    assert "/data/futures/um/monthly/klines/" in source.file_url(perp, "2024-01")


def test_absent_month_surfaces_as_not_found():
    def transport(url, params, timeout):
        return Response(404, b"", {}, url)

    client = HttpClient(bucket=TokenBucket(rate_per_second=1e6, sleeper=lambda s: None), transport=transport, sleeper=lambda s: None)
    source = BinanceVisionSource(client, "https://archive.test")
    with pytest.raises(NotFoundError):
        source.fetch(SPEC, source.plan(SPEC, START, START)[0])
