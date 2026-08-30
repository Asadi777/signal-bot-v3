# 0002 — Are taker-side columns Stage A or Stage C?

**Status:** OPEN — needs Majid's decision
**Date:** 2026-08-30

## Question

Binance candles include `trade_count`, `taker_buy_base` and
`taker_buy_quote` in the same free payload as OHLCV. Master Spec §57.1 says
Stage A is "OHLCV only" and that microstructure must not be mixed into the
baseline. It does not say which side of that line these three columns fall on.

## Why it matters

Taker-buy imbalance and trade count are among the more informative cheap
features available. Including them could materially raise Stage A's measured
lift; excluding them keeps Stage A a clean price/volume baseline and lets the
same columns show up as part of Stage C's incremental lift instead. Either is
defensible; deciding after seeing the numbers is not.

## Current handling

Both columns are always collected and stored, and are listed in
`STAGE_A_DISALLOWED_COLUMNS`. Nothing consumes them yet.

## Options

1. **Stage A** — treat them as part of the candle. Simple, and honest about
   what a free candle contains, but weakens the "OHLCV only" boundary.
2. **Stage C** — treat them as microstructure. Keeps the staged design clean
   and gives Stage C something to measure even before order-book data exists.
3. **Both, reported separately** — run Stage A twice, with and without. Costs
   one extra run and answers the question with data instead of taste.

Recommended: option 3, then adopt whichever boundary the numbers support and
record it here.
