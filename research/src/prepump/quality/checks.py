"""Dataset-level data-quality checks.

The one judgement call worth stating plainly: a minute with no candle is not
automatically a defect. Binance emits no candle for a minute in which nothing
traded, which is the normal state of a low-liquidity pair at 03:00 UTC. Calling
every such minute "missing data" would bury the real collection gaps under
millions of false alarms, so gaps are classified by run length and reported
separately, never filled.

Nothing here mutates data. Findings are descriptive; remediation is a human
decision recorded in the matrix or the limitations section.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pyarrow as pa

from prepump.timeutils import MINUTE_MS, date_str

# Gap-length buckets, in minutes. A single missing minute on an illiquid pair is
# almost certainly "no trade"; a multi-hour hole is almost certainly ours or an
# exchange halt. The middle band is explicitly unclassified rather than guessed.
GAP_NO_TRADE_MAX = 1
GAP_SHORT_MAX = 15
GAP_LONG_MAX = 1440

GAP_LIKELY_NO_TRADE = "LIKELY_NO_TRADE"
GAP_SHORT_UNCLASSIFIED = "SHORT_GAP_UNCLASSIFIED"
GAP_LONG_NEEDS_INVESTIGATION = "LONG_GAP_NEEDS_INVESTIGATION"
GAP_FULL_DAY_OR_MORE = "FULL_DAY_OR_MORE_MISSING"


@dataclass
class SymbolQuality:
    exchange: str
    market_type: str
    symbol: str
    rows: int = 0
    first_event_time_ms: int | None = None
    last_event_time_ms: int | None = None
    duplicate_rows: int = 0
    duplicate_examples: list[str] = field(default_factory=list)
    out_of_order_rows: int = 0
    expected_minutes: int = 0
    present_minutes: int = 0
    missing_minutes: int = 0
    gap_runs: dict = field(default_factory=dict)
    longest_gap_minutes: int = 0
    gap_examples: list[dict] = field(default_factory=list)
    dq_flag_counts: dict = field(default_factory=dict)
    zero_volume_minutes: int = 0
    dates_present: int = 0

    def to_dict(self) -> dict:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "rows": self.rows,
            "first_event_time_ms": self.first_event_time_ms,
            "last_event_time_ms": self.last_event_time_ms,
            "dates_present": self.dates_present,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_examples": self.duplicate_examples,
            "out_of_order_rows": self.out_of_order_rows,
            "expected_minutes": self.expected_minutes,
            "present_minutes": self.present_minutes,
            "missing_minutes": self.missing_minutes,
            "missing_pct": round(100 * self.missing_minutes / self.expected_minutes, 4) if self.expected_minutes else 0.0,
            "gap_runs": self.gap_runs,
            "longest_gap_minutes": self.longest_gap_minutes,
            "gap_examples": self.gap_examples,
            "dq_flag_counts": self.dq_flag_counts,
            "zero_volume_minutes": self.zero_volume_minutes,
        }


def _to_ms(value) -> int:
    """Arrow timestamps come back as datetimes; normalize to epoch ms."""
    if isinstance(value, int):
        return value
    return int(value.timestamp() * 1000)


def analyze(table: pa.Table) -> list[SymbolQuality]:
    """Per-symbol quality findings for a normalized OHLCV table."""
    data = table.to_pydict()
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, (exchange, market_type, symbol) in enumerate(
        zip(data["exchange"], data["market_type"], data["symbol"])
    ):
        grouped[(exchange, market_type, symbol)].append(index)

    results = []
    for (exchange, market_type, symbol), indices in sorted(grouped.items()):
        results.append(_analyze_group(exchange, market_type, symbol, indices, data))
    return results


def _analyze_group(exchange, market_type, symbol, indices, data) -> SymbolQuality:
    quality = SymbolQuality(exchange=exchange, market_type=market_type, symbol=symbol, rows=len(indices))

    times = [_to_ms(data["event_time"][i]) for i in indices]
    quality.out_of_order_rows = sum(1 for a, b in zip(times, times[1:]) if b < a)

    counts = Counter(times)
    duplicates = {ts: c for ts, c in counts.items() if c > 1}
    quality.duplicate_rows = sum(c - 1 for c in duplicates.values())
    quality.duplicate_examples = [
        f"{date_str(ts)}T{_hhmm(ts)}Z x{count}" for ts, count in sorted(duplicates.items())[:5]
    ]

    unique_times = sorted(counts)
    if unique_times:
        quality.first_event_time_ms = unique_times[0]
        quality.last_event_time_ms = unique_times[-1]
        quality.present_minutes = len(unique_times)
        span = (unique_times[-1] - unique_times[0]) // MINUTE_MS + 1
        quality.expected_minutes = span
        quality.missing_minutes = span - len(unique_times)
        quality.dates_present = len({date_str(ts) for ts in unique_times})
        gap_runs, longest, examples = _gap_runs(unique_times)
        quality.gap_runs = gap_runs
        quality.longest_gap_minutes = longest
        quality.gap_examples = examples

    flag_counter: Counter = Counter()
    zero_volume = 0
    for i in indices:
        for flag in data["dq_flags"][i] or []:
            flag_counter[flag] += 1
        if (data["base_volume"][i] or 0) == 0:
            zero_volume += 1
    quality.dq_flag_counts = dict(sorted(flag_counter.items()))
    quality.zero_volume_minutes = zero_volume
    return quality


def _hhmm(ms: int) -> str:
    from prepump.timeutils import ms_to_dt

    return ms_to_dt(ms).strftime("%H:%M")


def classify_gap(minutes: int) -> str:
    if minutes <= GAP_NO_TRADE_MAX:
        return GAP_LIKELY_NO_TRADE
    if minutes <= GAP_SHORT_MAX:
        return GAP_SHORT_UNCLASSIFIED
    if minutes <= GAP_LONG_MAX:
        return GAP_LONG_NEEDS_INVESTIGATION
    return GAP_FULL_DAY_OR_MORE


def _gap_runs(unique_times: list[int]) -> tuple[dict, int, list[dict]]:
    buckets: Counter = Counter()
    examples: list[dict] = []
    longest = 0
    for previous, current in zip(unique_times, unique_times[1:]):
        gap = (current - previous) // MINUTE_MS - 1
        if gap <= 0:
            continue
        label = classify_gap(gap)
        buckets[label] += 1
        longest = max(longest, gap)
        if len(examples) < 10 and label != GAP_LIKELY_NO_TRADE:
            examples.append(
                {
                    "after": f"{date_str(previous)}T{_hhmm(previous)}Z",
                    "missing_minutes": gap,
                    "classification": label,
                }
            )
    return dict(sorted(buckets.items())), longest, examples


def cross_source_discrepancy(
    left: pa.Table,
    right: pa.Table,
    *,
    price_tolerance: float = 1e-9,
    volume_tolerance: float = 1e-6,
) -> dict:
    """Compare two tables of the same symbol from different endpoints.

    Deliverable F asks for cross-source discrepancies where tested. Overlap is
    matched on the primary key; rows present in only one source are reported
    separately from rows that disagree, because they mean different things.
    """
    def index(table: pa.Table) -> dict[int, dict]:
        data = table.to_pydict()
        return {
            _to_ms(data["event_time"][i]): {
                "close": data["close"][i],
                "high": data["high"][i],
                "low": data["low"][i],
                "open": data["open"][i],
                "base_volume": data["base_volume"][i],
            }
            for i in range(table.num_rows)
        }

    left_index, right_index = index(left), index(right)
    shared = sorted(set(left_index) & set(right_index))
    mismatches = []
    for ts in shared:
        a, b = left_index[ts], right_index[ts]
        for column, tolerance in (
            ("open", price_tolerance),
            ("high", price_tolerance),
            ("low", price_tolerance),
            ("close", price_tolerance),
            ("base_volume", volume_tolerance),
        ):
            left_value, right_value = a[column] or 0.0, b[column] or 0.0
            scale = max(abs(left_value), abs(right_value), 1.0)
            if abs(left_value - right_value) / scale > tolerance:
                mismatches.append({"event_time_ms": ts, "column": column, "left": left_value, "right": right_value})
                break
    return {
        "overlapping_minutes": len(shared),
        "only_in_left": len(set(left_index) - set(right_index)),
        "only_in_right": len(set(right_index) - set(left_index)),
        "mismatching_minutes": len(mismatches),
        "mismatch_examples": mismatches[:10],
    }
