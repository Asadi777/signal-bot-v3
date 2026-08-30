from datetime import datetime, timedelta, timezone

import pytest

from prepump.timeutils import (
    TimeUnitError,
    detect_epoch_ms,
    detect_time_unit,
    dt_to_ms,
    minute_grid,
    month_bounds_ms,
    month_range,
    require_utc,
)


def test_require_utc_rejects_naive():
    with pytest.raises(ValueError):
        require_utc(datetime(2024, 1, 1))


def test_require_utc_rejects_non_utc_offset():
    tehran = timezone(timedelta(hours=3, minutes=30))
    with pytest.raises(ValueError):
        require_utc(datetime(2024, 1, 1, tzinfo=tehran))


def test_require_utc_accepts_utc():
    value = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert require_utc(value) is value


@pytest.mark.parametrize(
    "raw,unit",
    [
        (1_704_067_200, "s"),
        (1_704_067_200_000, "ms"),
        (1_704_067_200_000_000, "us"),
        (1_704_067_200_000_000_000, "ns"),
    ],
)
def test_epoch_unit_detection(raw, unit):
    # Binance archives have shipped all of these; guessing would shift candles
    # by orders of magnitude without any downstream check noticing.
    assert detect_time_unit(raw) == unit
    assert detect_epoch_ms(raw) == 1_704_067_200_000


def test_epoch_detection_rejects_nonsense():
    with pytest.raises(TimeUnitError):
        detect_epoch_ms(42)


def test_month_range_spans_year_boundary():
    start = dt_to_ms(datetime(2023, 11, 15, tzinfo=timezone.utc))
    end = dt_to_ms(datetime(2024, 2, 3, tzinfo=timezone.utc))
    assert month_range(start, end) == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_month_bounds_are_inclusive_and_utc():
    first, last = month_bounds_ms("2024-02")
    assert first == dt_to_ms(datetime(2024, 2, 1, tzinfo=timezone.utc))
    assert last == dt_to_ms(datetime(2024, 3, 1, tzinfo=timezone.utc)) - 1


def test_minute_grid_is_inclusive():
    start = dt_to_ms(datetime(2024, 1, 1, tzinfo=timezone.utc))
    grid = minute_grid(start, start + 3 * 60_000)
    assert len(grid) == 4 and grid[0] == start
