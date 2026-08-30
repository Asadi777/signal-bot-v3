# Phase 0A — Research Bootstrap

Point-in-time-correct collection of public crypto market data, for the
feasibility work described in `../docs/phase0a_plan_v1.md`.

This is **not** the product. It builds no model, labels no pump, places no
order, and holds no credentials. It answers one question: can we produce
research datasets that are reproducible and honest about what the sources
actually support?

## Quick start

```bash
cd research
make setup                # venv + editable install
make test                 # unit and offline end-to-end tests, no network
make selftest             # proves idempotency, resume and determinism offline
```

`make selftest` generates a synthetic archive that mirrors the Binance public
data layout, runs the real collector against it over `file://`, runs it a
second time, and asserts that nothing was rewritten and every content checksum
is unchanged.

## Offline sample run

```bash
make backfill-offline     # synthetic universe, no network at all
make quality              # writes data/reports/data_quality.{md,json}
make matrix               # validates and renders the Data Source Matrix
```

A committed slice of the output lives in `sample/`, including the data-quality
report and one run manifest.

## Network-enabled run

Everything above runs offline. The live Phase 0A run needs egress to
`data.binance.vision`, `api.binance.com`, `fapi.binance.com` and
`api.bybit.com`:

```bash
make live-metadata        # snapshot exchange metadata — start this cadence now
make live-backfill        # real smoke universe from the public archive
make live-verify          # integration tests + Bybit 1m depth probe
```

`make live-verify` is what turns Data Source Matrix rows from `UNKNOWN` into
evidence. Until it runs, every row stays `UNKNOWN` by design; the validator
refuses to let a row claim `AVAILABLE` without all seven evidence items.

## Commands

| Command | Purpose |
|---|---|
| `prepump version` | version metadata stamped onto every output |
| `prepump schema-info` | the normalized contract and the Stage A column policy |
| `prepump make-fixtures` | generate the synthetic offline archive |
| `prepump backfill-ohlcv` | 1-minute OHLCV backfill, resumable and idempotent |
| `prepump collect-metadata` | exchange symbol/market snapshots |
| `prepump quality-report` | data-quality report over the normalized dataset |
| `prepump validate-matrix` | enforce the matrix evidence rules |
| `prepump render-matrix` | render the matrix to the template layout |
| `prepump probe-bybit-depth` | measure Bybit's real 1-minute history depth |
| `prepump selftest-offline` | the acceptance run, offline |

## Output layout

```
data/raw/<source>/<market_type>/<symbol>/<interval>/<file>      bytes as served
data/raw/.../<file>.meta.json                                   url, sha256, what was detected
data/normalized/ohlcv_1m/exchange=/market_type=/symbol=/date=/  Parquet + _partition.json
data/reports/                                                   derived, regenerable
data/manifests/<run_id>.json                                    what each run did
```

`data/` is git-ignored. Nothing fetched from an exchange is ever committed.

## What the three timestamps mean

| Field | Question it answers |
|---|---|
| `event_time` | when the candle belongs to (its open, UTC) |
| `available_at` | earliest defensible time the system could have used the finalized candle |
| `ingested_at` | when this pipeline actually stored it |

The eligibility rule is `available_at <= T`, never `event_time <= T`. For
archive data `available_at` is *derived* from the candle close, and every row
says so in `available_at_method` with an explicit
`available_at_uncertainty_ms`. A derived value must never be readable as a
measured one.

## Rules the code enforces, not just documents

- **No gap filling.** A missing minute stays missing and is classified. On an
  illiquid pair, "no candle" usually means "nobody traded", not "data lost".
- **No trading surface.** A test fails if any module gains an order or
  withdrawal endpoint.
- **No secrets.** Every source is public; a test fails if credential-shaped
  names appear in the package.
- **No unearned status.** The matrix validator rejects `AVAILABLE` without all
  seven evidence items, a reviewer and a date.
- **Atomic partitions.** A crash leaves a partition either complete or absent,
  never half-written; incomplete partitions are rewritten, not trusted.

## Relationship to `signal_bot.py`

The v3.3 bot at the repository root is untouched and keeps running as it did.
It is also a candidate for one of the heuristic baselines Master Spec §57.4
requires Stage A to beat.
