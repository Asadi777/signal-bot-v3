"""UTC-only time helpers.

Every timestamp in this project is UTC with millisecond precision. Naive
datetimes are rejected rather than coerced: a silently localized timestamp
would corrupt point-in-time correctness in a way no downstream check can
detect.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

MINUTE_MS = 60_000
DAY_MS = 86_400_000

# Plausibility window used to auto-detect the time unit of a source column.
# Binance archives have shipped open_time in seconds, milliseconds and (in
# newer futures dumps) microseconds; guessing wrong shifts every candle by
# three orders of magnitude, so we detect instead of assuming.
_YEAR_2001_MS = 978_307_200_000
_YEAR_2100_MS = 4_102_444_800_000


class TimeUnitError(ValueError):
    """Raised when a raw timestamp cannot be mapped to a plausible epoch."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_ms() -> int:
    return int(utc_now().timestamp() * 1000)


def require_utc(value: datetime) -> datetime:
    """Return ``value`` unchanged, or raise if it is naive or not UTC."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"naive datetime is not allowed: {value!r}")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"non-UTC datetime is not allowed: {value!r}")
    return value


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def dt_to_ms(value: datetime) -> int:
    return int(require_utc(value).timestamp() * 1000)


def detect_epoch_ms(raw: int) -> int:
    """Normalize an epoch integer of unknown unit to milliseconds.

    Accepts seconds, milliseconds, microseconds or nanoseconds and returns
    milliseconds. Raises when no interpretation lands in [2001, 2100).
    """
    candidates = ((raw * 1000, "s"), (raw, "ms"), (raw // 1000, "us"), (raw // 1_000_000, "ns"))
    for value, _unit in candidates:
        if _YEAR_2001_MS <= value < _YEAR_2100_MS:
            return int(value)
    raise TimeUnitError(f"cannot interpret {raw!r} as an epoch timestamp")


def detect_time_unit(raw: int) -> str:
    """Return the detected unit name ('s' | 'ms' | 'us' | 'ns') for ``raw``."""
    for value, unit in ((raw * 1000, "s"), (raw, "ms"), (raw // 1000, "us"), (raw // 1_000_000, "ns")):
        if _YEAR_2001_MS <= value < _YEAR_2100_MS:
            return unit
    raise TimeUnitError(f"cannot interpret {raw!r} as an epoch timestamp")


def floor_to_minute_ms(ms: int) -> int:
    return (ms // MINUTE_MS) * MINUTE_MS


def date_str(ms: int) -> str:
    """UTC calendar date of ``ms``, used as the partition key."""
    return ms_to_dt(ms).strftime("%Y-%m-%d")


def month_str(ms: int) -> str:
    return ms_to_dt(ms).strftime("%Y-%m")


def month_range(start_ms: int, end_ms: int) -> list[str]:
    """Inclusive list of ``YYYY-MM`` months covering [start_ms, end_ms]."""
    if end_ms < start_ms:
        return []
    cur = ms_to_dt(start_ms).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last = ms_to_dt(end_ms)
    out: list[str] = []
    while cur.year < last.year or (cur.year == last.year and cur.month <= last.month):
        out.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=28) + timedelta(days=8)).replace(day=1)
    return out


def month_bounds_ms(month: str) -> tuple[int, int]:
    """Return [first_ms, last_ms_inclusive] of the UTC month ``YYYY-MM``."""
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    nxt = (start.replace(day=28) + timedelta(days=8)).replace(day=1)
    return dt_to_ms(start), dt_to_ms(nxt) - 1


def minute_grid(start_ms: int, end_ms: int) -> list[int]:
    """Every minute open-time in [start_ms, end_ms], both aligned inclusive."""
    first = floor_to_minute_ms(start_ms)
    return list(range(first, floor_to_minute_ms(end_ms) + MINUTE_MS, MINUTE_MS))
