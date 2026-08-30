# Normalized OHLCV schema — `ohlcv_1m.v1`

Dataset: `data/normalized/ohlcv_1m`
Partitioning: `exchange / market_type / symbol / date` (UTC calendar date)
Primary key: `(exchange, market_type, symbol, interval, event_time)`
File format: Parquet, zstd level 3, format version 2.6, 100k-row row groups

## Columns

| Column | Type | Null | Meaning |
|---|---|---|---|
| `exchange` | string | no | `binance`, `bybit` |
| `market_type` | string | no | `spot`, `linear_perpetual`, `inverse_perpetual`, `futures` |
| `symbol` | string | no | exchange-native symbol, upper case |
| `symbol_canonical` | string | no | `exchange:market_type:BASE-QUOTE` |
| `base_asset` / `quote_asset` | string | no | from config or exchange metadata where possible |
| `interval` | string | no | `1m` |
| `event_time` | timestamp[ms, UTC] | no | candle open — the time key |
| `event_time_close` | timestamp[ms, UTC] | no | candle close |
| `available_at` | timestamp[ms, UTC] | no | earliest defensible usable time |
| `available_at_method` | string | no | `DERIVED_FROM_CLOSE` / `MEASURED` / `APPROXIMATED` |
| `available_at_uncertainty_ms` | int32 | no | explicit uncertainty on the above |
| `ingested_at` | timestamp[ms, UTC] | no | when this pipeline stored the row |
| `open` / `high` / `low` / `close` | float64 | no | prices as served |
| `base_volume` | float64 | no | base-asset volume |
| `quote_volume` | float64 | yes | null when the source does not provide it |
| `trade_count` | int64 | yes | null on sources without it (Bybit) |
| `taker_buy_base` / `taker_buy_quote` | float64 | yes | null on sources without them |
| `source` | string | no | e.g. `binance_vision_monthly_klines` |
| `source_revision` | string | no | `sha256:<file digest>` for archives |
| `collector_version` | string | no | collection/normalization logic version |
| `schema_version` | string | no | this contract's version |
| `run_id` | string | no | the run that wrote the row |
| `dq_flags` | list<string> | no | row-level quality flags, possibly empty |

## The three timestamps

`event_time` says **when the market event happened**. It is the correct field
for ordering and for computing windows.

`available_at` says **when we could first have used it**. It is the only field
allowed in the eligibility rule of Master Spec §61.1:

```
a feature may enter a prediction at time T only if available_at <= T
```

`ingested_at` says **when we actually got it**. For live collection it can be
later than `available_at`; reconstructing what the system really knew must use
both.

### Why `available_at` is derived, and what that costs

For a closed candle the earliest defensible availability is its close time: the
value was determined at that instant and disseminated immediately. The archive
file that republishes it later is an artefact of our access path, not of when
the market knew the number.

So archive rows carry `available_at = event_time_close + 1ms`,
`available_at_method = DERIVED_FROM_CLOSE`, and a non-zero
`available_at_uncertainty_ms` covering plausible dissemination latency.

This is an approximation and is labelled as one. A **measured** `available_at`
requires a live websocket collector timestamping arrivals, which is out of
Phase 0A scope. Nothing in this dataset should be read as evidence about
sub-second availability.

## Data-quality flags

| Flag | Condition |
|---|---|
| `OHLC_VIOLATION` | not `low <= min(open, close) <= max(open, close) <= high` |
| `NON_POSITIVE_PRICE` | any of OHLC is `<= 0` |
| `NEGATIVE_VOLUME` | base or quote volume `< 0` |
| `ZERO_VOLUME_WITH_RANGE` | zero volume while `high != low` |
| `CLOSE_TIME_MISMATCH` | `close_time != open_time + interval - 1` |
| `NOT_MINUTE_ALIGNED` | `open_time` not on the minute grid |
| `QUOTE_VOLUME_MISSING` | source provided no quote volume |
| `INFERRED_SYMBOL_SPLIT` | base/quote were inferred, not declared |

Flags describe; they never modify. No row is dropped, corrected or filled.

## Missing minutes

A minute with no candle is **not** automatically missing data: exchanges emit
no candle for a minute with no trades. The quality report classifies gaps by
run length (`LIKELY_NO_TRADE`, `SHORT_GAP_UNCLASSIFIED`,
`LONG_GAP_NEEDS_INVESTIGATION`, `FULL_DAY_OR_MORE_MISSING`) and never fills
them. Distinguishing "no trade" from "we lost data" with certainty requires
trade-level data, which Phase 0A does not collect; the middle band is labelled
unclassified rather than guessed.

## Determinism

Two runs over the same source revision produce identical partition content
checksums. The checksum deliberately excludes `ingested_at` and `run_id` —
they are the only fields allowed to vary between runs, and they are recorded
because provenance matters more than making the files byte-identical.

## Stage A column policy (open)

`trade_count`, `taker_buy_base` and `taker_buy_quote` arrive free in the same
candle payload but are arguably microstructure. Master Spec §57.1 forbids
mixing microstructure into the Stage A baseline and does not say which side of
the line these fall on. They are therefore always stored and flagged
`STAGE_A_DISALLOWED_COLUMNS` until the decision in
`docs/decisions/0002-stage-a-column-policy.md` is made.

## Exchange metadata schema — `exchange_metadata.v1`

Dataset: `data/normalized/exchange_metadata`, partitioned by exchange, scope
and collection date, append-only by date. Carries symbol identity, status,
contract specifications, `listing_time` with a `listing_time_method`
(`EXCHANGE_PROVIDED` or `UNAVAILABLE_AT_SOURCE`), and a `known_gaps` list.

Binance spot exposes no listing date at all; futures exposes `onboardDate`;
Bybit exposes `launchTime`. Delisted symbols vanish from all three. A series of
dated snapshots is the only membership history these endpoints can give us,
which is why the cadence should start immediately — every uncollected day is
history that cannot be recovered later.
