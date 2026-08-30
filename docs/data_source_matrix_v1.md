# Crypto Pre-Pump Project — Data Source Matrix v1 (partially populated)

> Every row is UNKNOWN. No source may be promoted above UNKNOWN until the seven evidence items are actually collected, and none could be collected in the environment this file was written in: the session's egress policy refused CONNECT to data.binance.vision, api.binance.com and api.bybit.com, so no live documentation check, sample, depth check or rate-limit test was possible. The descriptive columns below therefore record what each row is *expected* to show and exactly how to verify it; they are not findings. Filling this in is the first task of the network-enabled run.

| Domain | Provider / Product | Endpoint / Dataset | Exchanges / Chains | Symbol Coverage | Historical Start | Granularity | Raw Backfill | event_time Quality | available_at / Latency Reliability | ingested_at Capturable | Revisions / Backfills | Rate Limits | Monthly Cost | Bulk Backfill Cost | Licensing / Redistribution | PIT Suitability | Known Gaps | Status | Evidence / Docs | Reviewed At | Reviewer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Derivatives / OI | Binance / Futures open interest | GET /futures/data/openInterestHist | Binance USDⓈ-M | Perpetual contracts | DOCUMENTED HARD LIMIT: only the latest 30 days are available from this endpoint | 5m and coarser | Not possible beyond 30 days from this endpoint. See the archive metrics row below for the alternative | Sampled series, not events | UNVERIFIED | Yes | UNVERIFIED | Weight-based | 0 | May require a paid vendor if retention is as short as expected | UNVERIFIED | UNVERIFIED | 30-day retention means Stage B history CANNOT be reconstructed from this endpoint. This is a documented fact, not a suspicion; Granularity is 5m at best, far coarser than the 1m onset labelling | UNKNOWN | https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics — 'Only the data of the latest 30 days is available', limit max 500 | — | — |
| Derivatives / OI | Binance / Public data archive — futures metrics | data.binance.vision /data/futures/um/daily/metrics/{SYMBOL}/ | Binance USDⓈ-M | Perpetuals; per-symbol coverage unmeasured | UNVERIFIED — this is now the most valuable single thing to measure in the whole matrix | Reported as 5m rows containing sumOpenInterest and sumOpenInterestValue | Yes, if coverage is deep — this is the only free path to historical open interest we have found | Sampled series timestamps, not events | Derived from the sample timestamp; sampling delay unmeasured | Yes | UNVERIFIED | Static file host | 0 | 0 plus bandwidth | UNVERIFIED, same terms as the rest of the archive | Plausible for 5m-granularity features; useless for minute-level onset work | Found by search, not yet confirmed against the host; the project README does not list a metrics dataset, so it is undocumented in the official repo; 5m granularity cannot support a 15-minute lead-time claim on its own | UNKNOWN | — | — | — |
| Exchange Flows | TBD / TBD | — | — | — | — | — | — | — | Label publication lag is the whole problem: exchange wallet labels are assigned after the fact | — | Labels are revised retroactively, which is a leakage risk | — | — | — | — | Suspect until label revision history is available | A vendor that cannot tell us when a label was assigned cannot be used point-in-time | UNKNOWN | — | — | — |
| Fundamental / Tokenomics | TBD / TBD | — | — | — | — | — | — | Unlock schedules are known in advance, which makes them unusually clean point-in-time features | — | — | Supply figures are silently restated by aggregators | — | — | — | — | — | Not investigated in Phase 0A | UNKNOWN | — | — | — |
| Funding / Basis | Binance / Funding rate history | GET /fapi/v1/fundingRate | Binance USDⓈ-M | Perpetuals | UNVERIFIED, expected long | Per funding interval | Expected yes | Funding timestamps | UNVERIFIED | Yes | UNVERIFIED | Weight-based | 0 | 0 | UNVERIFIED | UNVERIFIED | — | UNKNOWN | — | — | — |
| Liquidations | Binance / Force order stream | websocket forceOrder / no documented historical REST endpoint | Binance USDⓈ-M | Perpetuals | Expected none — likely forward-collection only | Per event | Expected not possible from the exchange | Event timestamps | Live only | Yes | n/a | n/a | 0 live | Vendor-dependent | UNVERIFIED | Good live, absent historically | A domain that cannot be backfilled cannot contribute to a historical Stage B test; this must be stated before anyone builds features on it | UNKNOWN | — | — | — |
| Listing / Delisting Metadata | Binance / exchangeInfo | GET /api/v3/exchangeInfo, GET /fapi/v1/exchangeInfo | Binance spot and futures | Currently listed symbols only | Not applicable; this is a live snapshot | Snapshot per collection | No. History exists only as the series of snapshots we start taking now | n/a | Snapshot collection time | Yes | Symbols appear and disappear without notice | Weight-based; one call per collection | 0 | n/a | UNVERIFIED | Weak. Membership-as-of-time can only be approximated from snapshot series plus archive coverage | Spot exchangeInfo exposes no listing date at all; futures exposes onboardDate; Delisted symbols vanish, taking their delisting date with them | UNKNOWN | — | — | — |
| Listing / Delisting Metadata | Bybit / v5 instruments-info | GET /v5/market/instruments-info?category=spot\|linear | Bybit | Currently listed instruments | n/a | Snapshot | No | n/a | Snapshot collection time | Yes | UNVERIFIED | UNVERIFIED | 0 | n/a | UNVERIFIED | Better than Binance spot if launchTime is populated | launchTime presence and reliability unverified | UNKNOWN | — | — | — |
| News / Catalysts | TBD / TBD | — | — | — | — | — | — | Publication time versus our observation time differ materially here | — | — | Articles are edited and re-timestamped | — | — | — | — | — | Not investigated in Phase 0A | UNKNOWN | — | — | — |
| OHLCV | Binance / REST klines | GET /api/v3/klines, GET /fapi/v1/klines | Binance spot and USDⓈ-M futures | Live symbols only; delisted symbols expected to be absent | UNVERIFIED | 1m and coarser, 1000 candles per request | Possible but weight-limited; used for the current month and cross-checks only | Same candle semantics as the archive | Derived from close; a measured available_at needs a live websocket collector, which is out of Phase 0A scope | Yes | UNVERIFIED | Weight-based per IP, reported in X-MBX-USED-WEIGHT-1M; exact weight per klines call still unmeasured | 0 | 0, but paid in time and ban risk | UNVERIFIED | Same as the archive for closed candles | Delisted symbols expected to be unreachable, which is why the archive leads | UNKNOWN | — | — | — |
| OHLCV | Binance / Public data archive | data.binance.vision /data/{spot\|futures/um}/monthly/klines/{SYMBOL}/1m/*.zip | Binance spot and USDⓈ-M futures | Expected to include symbols no longer trading, which is the whole reason this row exists | Not documented by Binance; must be measured per symbol. Third-party example shows 2020 files for a delisted pair | 1m and coarser | Yes — whole-month zipped CSV per symbol | Candle open/close from the exchange. DOCUMENTED: spot timestamps are in MICROSECONDS from 2025-01-01 onward, milliseconds before. Detected per file, never assumed | Derived from candle close, not measured. Archive publication lag is a property of our access path, not of when the market knew the value | Yes, recorded by the pipeline | DOCUMENTED: every zip ships a sibling .CHECKSUM (sha256), so a silent revision is detectable | Static file host; no documented per-IP limit. Untested from our side | 0 | 0 plus bandwidth | UNVERIFIED — internal research use expected to be fine, redistribution is not assumed | Expected good for closed candles; unusable for intra-candle availability questions | No candle is emitted for a minute with no trades; absence is not loss; Delisting and listing dates are not stated by the archive, only implied by file coverage; Coverage start per symbol is not documented anywhere and can only be measured; Header row presence in the CSV is not documented; detected per file | UNKNOWN | https://github.com/binance/binance-public-data — URL layout, per-file .CHECKSUM (sha256), 12-column kline layout, all confirmed to match this implementation | — | — |
| OHLCV | Bybit / v5 market kline | GET /v5/market/kline?category=spot\|linear&interval=1 | Bybit spot, linear and inverse | Live symbols | Not documented by Bybit. Community threads exist specifically about finding the earliest candle, which suggests it is not discoverable without probing. Max 1000 rows per request (documented) | 1m and coarser, newest-first, 1000 per page | Depends entirely on the depth probe | Candle start only; close time is derived by us | Derived from close | Yes | UNVERIFIED | Per-IP limits, exact values UNVERIFIED | 0 | 0 | UNVERIFIED | Conditional on depth | No trade count and no taker-side split; those columns stay null rather than being invented | UNKNOWN | — | — | — |
| On-chain / Wallet | TBD / TBD | — | Multi-chain | — | — | — | — | Block time is not availability time; confirmation depth matters | — | — | Reorgs are real revisions and must be modelled | — | — | — | — | This is where the bitemporal model actually earns its keep | Not investigated in Phase 0A by design | UNKNOWN | — | — | — |
| Order Book / L2 | Binance / Depth snapshots / diff stream | websocket depth, REST depth snapshot | Binance | All | None from the exchange | Snapshot / diff | No | Update ids and timestamps | Live only | Yes | n/a | Heavy | 0 live, storage dominates | Paid vendors only | UNVERIFIED | Live good, historical absent | Stage C is forward-collection or paid; this is a schedule fact, not a detail | UNKNOWN | — | — | — |
| Social Attention | TBD / TBD | — | — | — | Historical social data is usually the hardest and most expensive to obtain point-in-time | — | — | Post time is not indexing time | — | — | Deleted posts and retroactive bot filtering both change history | — | — | — | Scraping raises terms-of-service exposure that market data does not | — | Not investigated in Phase 0A | UNKNOWN | — | — | — |
| Trade Flow / Microstructure | Binance / Kline taker-side columns | Included in every kline row | Binance | Same as klines | Same as klines | 1m | Already collected, free, in the same payload | Same as the candle | Same as the candle | Yes | Same as klines | None additional | 0 | 0 | Same as klines | Same as klines | Whether trade_count and the taker split belong to Stage A or Stage C is an open decision; they are stored and flagged, not used | UNKNOWN | — | — | — |
| Trades | Binance / Public data archive (aggTrades / trades) | data.binance.vision /data/spot/monthly/aggTrades/... | Binance | Expected to match the kline archive | UNVERIFIED | Individual trades | Yes, at one to two orders of magnitude more volume than klines | Per-trade timestamps | Derived from trade time | Yes | UNVERIFIED | Static file host | 0 | 0 in money, large in storage — multi-TB at full universe scale | UNVERIFIED | Expected good | Not fetched in Phase 0A; storage cost must be planned before Stage C | UNKNOWN | — | — | — |
| Whale / Smart Money | TBD / TBD | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | Not investigated in Phase 0A | UNKNOWN | — | — | — |

## Verification plan for rows still UNKNOWN

### Derivatives / OI — Binance Futures open interest

- Confirm the 30-day limit against the live endpoint
- Then use the archive metrics dataset below instead of this endpoint for any historical work

### Derivatives / OI — Binance Public data archive — futures metrics

- FIRST PRIORITY of the network-enabled run: confirm the metrics dataset exists, measure its per-symbol coverage start, and confirm the column layout
- If it is deep, Stage B becomes a historical experiment rather than a months-long forward collection. If it is not, Stage B is blocked on time or money

### Exchange Flows — TBD TBD

- For any candidate vendor, ask specifically whether label assignment history is retrievable. If not, the domain is not PIT-usable

### Fundamental / Tokenomics — TBD TBD

- Deferred to Stage E

### Funding / Basis — Binance Funding rate history

- Confirm depth and pagination for one perpetual

### Liquidations — Binance Force order stream

- Confirm no historical endpoint exists, then decide between forward collection and a paid vendor

### Listing / Delisting Metadata — Binance exchangeInfo

- Collect one snapshot of each endpoint and confirm which fields actually arrive
- Decide and document the approximation rule for spot listing dates (first archive candle)
- Start the daily snapshot cadence immediately; every day not collected is history we cannot recover

### Listing / Delisting Metadata — Bybit v5 instruments-info

- Collect one snapshot per category and measure how many rows carry a usable launchTime

### News / Catalysts — TBD TBD

- Deferred to Stage E

### OHLCV — Binance Public data archive

- REMAINING: fetch one monthly file for a liquid symbol and confirm the published CHECKSUM matches
- REMAINING: walk the S3-style listing to measure per-symbol coverage start and end
- REMAINING: fetch a delisted symbol and confirm it is served (documentation implies yes; a third-party example uses ADABKRW, a delisted pair)
- REMAINING: read the terms of use and record redistribution rights verbatim

### OHLCV — Binance REST klines

- Measure the actual weight cost of a 1000-candle request from the response headers
- Compare one overlapping day against the archive and quantify any disagreement
- Confirm behaviour for a delisted symbol (expected: an error, handled cleanly)

### OHLCV — Bybit v5 market kline

- Run `prepump probe-bybit-depth` and record the measured 1-minute depth as evidence
- If depth is insufficient, evaluate the public trade-level archive and the cost of aggregating candles from it
- Cross-check one overlapping day against Binance for the same token

### On-chain / Wallet — TBD TBD

- Deferred to Stage D. Do not collect anything here until the OHLCV path is verified

### Order Book / L2 — Binance Depth snapshots / diff stream

- Price at least two vendors for historical L2 before Stage C is scheduled

### Social Attention — TBD TBD

- Deferred to Stage E. Legal review before any collection, not after

### Trade Flow / Microstructure — Binance Kline taker-side columns

- Get a decision on the Stage A column policy (docs/decisions/0002) before Stage A features are built

### Trades — Binance Public data archive (aggTrades / trades)

- Measure the compressed size of one month for one liquid symbol and extrapolate before committing

### Whale / Smart Money — TBD TBD

- Deferred to Stage D, after Exchange Flows label semantics are settled

