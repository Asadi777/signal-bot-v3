# Data Quality Report

- Generated at: `2026-08-30T11:42:16+00:00`
- Schema version: `ohlcv_1m.v1`
- Collector version: `0.1.0`
- Dataset: `ohlcv_1m` — 103 partitions, 11.392 MiB
- Partitions by source: `{'binance_vision_monthly_klines': 103}`

## Totals

| Metric | Value |
|---|---:|
| symbols | 4 |
| rows | 131,329 |
| expected minutes | 148,320 |
| missing minutes | 16,992 |
| duplicate rows | 1 |
| out of order rows | 0 |
| zero volume minutes | 197 |

## Row-level flags

| Flag | Rows |
|---|---:|
| `OHLC_VIOLATION` | 1 |
| `ZERO_VOLUME_WITH_RANGE` | 197 |

## Per symbol

| Symbol | Market | Rows | Range (UTC) | Missing min | Missing % | Dupes | Out of order | Longest gap (min) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `PERPUSDT` | linear_perpetual | 44,207 | 2024-01-01 00:00 → 2024-01-31 23:59 | 433 | 0.97 | 0 | 0 | 2 |
| `DEADUSDT` | spot | 13,681 | 2024-01-01 00:00 → 2024-01-10 23:59 | 719 | 4.9931 | 0 | 0 | 2 |
| `FIXTUSDT` | spot | 44,461 | 2024-01-01 00:00 → 2024-01-31 23:59 | 180 | 0.4032 | 1 | 0 | 180 |
| `GAPYUSDT` | spot | 28,980 | 2024-01-01 00:00 → 2024-01-31 23:59 | 15,660 | 35.0806 | 0 | 0 | 9 |

## Gap classification

A missing minute is not automatically a defect: an exchange emits no candle for a minute with no trades.
Gaps are therefore classified by run length and never filled.

| Symbol | Gap classification | Occurrences |
|---|---|---:|
| `PERPUSDT` | `LIKELY_NO_TRADE` | 429 |
| `PERPUSDT` | `SHORT_GAP_UNCLASSIFIED` | 2 |
| `DEADUSDT` | `LIKELY_NO_TRADE` | 675 |
| `DEADUSDT` | `SHORT_GAP_UNCLASSIFIED` | 22 |
| `FIXTUSDT` | `LONG_GAP_NEEDS_INVESTIGATION` | 1 |
| `GAPYUSDT` | `LIKELY_NO_TRADE` | 6,520 |
| `GAPYUSDT` | `SHORT_GAP_UNCLASSIFIED` | 3,578 |

### Gaps needing investigation (first 10)

| Symbol | After | Missing minutes | Classification |
|---|---|---:|---|
| `PERPUSDT` | 2024-01-06T18:24Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `PERPUSDT` | 2024-01-10T21:11Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `DEADUSDT` | 2024-01-01T11:03Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `DEADUSDT` | 2024-01-01T14:28Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `DEADUSDT` | 2024-01-02T06:29Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `DEADUSDT` | 2024-01-03T10:51Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `DEADUSDT` | 2024-01-03T13:22Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `DEADUSDT` | 2024-01-03T18:04Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `DEADUSDT` | 2024-01-05T10:26Z | 2 | `SHORT_GAP_UNCLASSIFIED` |
| `DEADUSDT` | 2024-01-05T14:26Z | 2 | `SHORT_GAP_UNCLASSIFIED` |

## Cross-source comparison

Not run in this report. Requires live access to a second endpoint for the same symbol and range.

## Notes and known limitations

- Missing minutes are classified, never filled. A one-minute gap on an illiquid pair is most likely 'no trade occurred', not lost data.
- available_at is derived from candle close for archive data and is not a measured availability time; see available_at_method and available_at_uncertainty_ms on every row.
- This dataset is a provisional smoke-test sample. It is not a research universe and must not be generalized (Master Spec §65.3).

