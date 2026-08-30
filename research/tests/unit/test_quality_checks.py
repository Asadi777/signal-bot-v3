import pyarrow as pa

from prepump.quality.checks import (
    GAP_LIKELY_NO_TRADE,
    GAP_LONG_NEEDS_INVESTIGATION,
    GAP_SHORT_UNCLASSIFIED,
    analyze,
    classify_gap,
    cross_source_discrepancy,
)
from prepump.timeutils import MINUTE_MS
from tests.helpers import START_MS, kline, make_table


def test_clean_series_reports_no_findings():
    result = analyze(make_table())[0]
    assert result.rows == 5
    assert result.missing_minutes == 0
    assert result.duplicate_rows == 0
    assert result.out_of_order_rows == 0


def test_single_missing_minute_is_classified_as_no_trade():
    table = make_table([kline(0), kline(2)])
    result = analyze(table)[0]
    assert result.missing_minutes == 1
    assert result.gap_runs == {GAP_LIKELY_NO_TRADE: 1}
    # A one-minute hole on an illiquid pair is normal; it must not be surfaced
    # as something to investigate.
    assert result.gap_examples == []


def test_long_gap_is_surfaced_for_investigation():
    table = make_table([kline(0), kline(200)])
    result = analyze(table)[0]
    assert result.gap_runs == {GAP_LONG_NEEDS_INVESTIGATION: 1}
    assert result.longest_gap_minutes == 199
    assert result.gap_examples and result.gap_examples[0]["classification"] == GAP_LONG_NEEDS_INVESTIGATION


def test_gap_classification_boundaries():
    assert classify_gap(1) == GAP_LIKELY_NO_TRADE
    assert classify_gap(2) == GAP_SHORT_UNCLASSIFIED
    assert classify_gap(15) == GAP_SHORT_UNCLASSIFIED
    assert classify_gap(16) == GAP_LONG_NEEDS_INVESTIGATION
    assert classify_gap(2000) == "FULL_DAY_OR_MORE_MISSING"


def test_duplicates_are_counted_and_exemplified():
    table = make_table([kline(0), kline(0), kline(1)])
    result = analyze(table)[0]
    assert result.duplicate_rows == 1
    assert result.duplicate_examples


def test_out_of_order_rows_are_detected():
    # analyze() reads rows as stored, so build the table unsorted on purpose.
    table = make_table()
    reversed_table = table.take(list(reversed(range(table.num_rows))))
    assert analyze(reversed_table)[0].out_of_order_rows == 4


def test_row_flags_are_aggregated():
    table = make_table([kline(0, high=0.1), kline(1)])
    result = analyze(table)[0]
    assert result.dq_flag_counts.get("OHLC_VIOLATION") == 1


def test_symbols_are_reported_separately():
    from prepump.normalize.symbols import make_spec

    other = make_table(spec=make_spec("binance", "spot", "OTHERUSDT", "OTHER", "USDT"))
    combined = pa.concat_tables([make_table(), other])
    assert {item.symbol for item in analyze(combined)} == {"TESTUSDT", "OTHERUSDT"}


def test_cross_source_agreement_and_disagreement():
    left = make_table([kline(0), kline(1), kline(2)])
    right = make_table([kline(0), kline(1, close=999.0), kline(3)])
    result = cross_source_discrepancy(left, right)
    assert result["overlapping_minutes"] == 2
    assert result["mismatching_minutes"] == 1
    assert result["only_in_left"] == 1
    assert result["only_in_right"] == 1
