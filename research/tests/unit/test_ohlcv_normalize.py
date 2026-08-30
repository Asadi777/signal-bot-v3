from prepump.normalize.ohlcv import split_by_utc_date, to_table
from prepump.normalize.schema import DQFlag
from prepump.timeutils import MINUTE_MS
from tests.helpers import START_MS, kline, make_table


def test_rows_are_sorted_by_event_time():
    table = make_table([kline(2), kline(0), kline(1)])
    times = [ts.timestamp() * 1000 for ts in table.column("event_time").to_pylist()]
    assert times == sorted(times)


def test_available_at_is_close_plus_documented_delta():
    table = make_table([kline(0)])
    close = table.column("event_time_close").to_pylist()[0].timestamp() * 1000
    available = table.column("available_at").to_pylist()[0].timestamp() * 1000
    assert available - close == 1
    assert table.column("available_at_method").to_pylist()[0] == "DERIVED_FROM_CLOSE"
    # An approximation must never look like a measurement.
    assert table.column("available_at_uncertainty_ms").to_pylist()[0] > 0


def test_available_at_is_never_before_event_time():
    table = make_table()
    for event, available in zip(table.column("event_time").to_pylist(), table.column("available_at").to_pylist()):
        assert available > event


def test_ohlc_violation_is_flagged():
    table = make_table([kline(0, high=50.0)])
    assert DQFlag.OHLC_VIOLATION in table.column("dq_flags").to_pylist()[0]


def test_non_positive_price_is_flagged():
    table = make_table([kline(0, low=-1.0)])
    assert DQFlag.NON_POSITIVE_PRICE in table.column("dq_flags").to_pylist()[0]


def test_negative_volume_is_flagged():
    assert DQFlag.NEGATIVE_VOLUME in make_table([kline(0, base_volume=-5.0)]).column("dq_flags").to_pylist()[0]


def test_zero_volume_with_price_range_is_flagged():
    flags = make_table([kline(0, base_volume=0.0)]).column("dq_flags").to_pylist()[0]
    assert DQFlag.ZERO_VOLUME_WITH_RANGE in flags


def test_zero_volume_without_range_is_not_flagged():
    flat = kline(0, base_volume=0.0, open=100.0, high=100.0, low=100.0, close=100.0)
    assert DQFlag.ZERO_VOLUME_WITH_RANGE not in make_table([flat]).column("dq_flags").to_pylist()[0]


def test_misaligned_and_mismatched_timestamps_are_flagged():
    flags = make_table([kline(0, open_time_ms=START_MS + 30_000, close_time_ms=START_MS + 30_000)]).column(
        "dq_flags"
    ).to_pylist()[0]
    assert DQFlag.NOT_MINUTE_ALIGNED in flags
    assert DQFlag.CLOSE_TIME_MISMATCH in flags


def test_missing_quote_volume_is_flagged_not_invented():
    table = make_table([kline(0, quote_volume=None)])
    assert table.column("quote_volume").to_pylist()[0] is None
    assert DQFlag.QUOTE_VOLUME_MISSING in table.column("dq_flags").to_pylist()[0]


def test_split_by_utc_date_groups_on_the_partition_key():
    day_two = START_MS + 24 * 60 * MINUTE_MS
    grouped = split_by_utc_date([kline(0), kline(0, open_time_ms=day_two)])
    assert sorted(grouped) == ["2024-01-01", "2024-01-02"]


def test_versions_are_stamped_on_every_row():
    table = make_table()
    assert set(table.column("schema_version").to_pylist()) == {"ohlcv_1m.v1"}
    assert len(set(table.column("run_id").to_pylist())) == 1
