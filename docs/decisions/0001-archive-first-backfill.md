# 0001 — Archive-first backfill

**Status:** Proposed (implemented in Phase 0A, reversible)
**Date:** 2026-08-30
**Context:** Kickoff §D asks for a REST proof of concept with resumable pagination.

## Decision

Use the Binance public data archive (`data.binance.vision`) as the primary
backfill path. Keep the REST klines endpoint for the current, not-yet-published
month and for independent cross-checks.

## Why

| | Archive files | REST klines |
|---|---|---|
| throughput | one month per request | 1000 candles per request |
| ban risk | static file host | weight-limited, ban-prone |
| delisted symbols | retained | generally unreachable |
| verifiability | published SHA256 per file | none |

The decisive point is the third row. Master Spec §65.3 forbids treating
today's listed symbols as the historical universe. The archive is the only
free path we know of that still serves symbols which no longer trade, so
without it the survivorship rule is unenforceable rather than merely
inconvenient.

## Consequences

- REST pagination is still implemented and tested, because it is needed for the
  current month and because the cross-check is a deliverable (§F).
- The archive's own coverage is unverified until a network-enabled run; if it
  turns out not to retain delisted symbols, the universe policy has to change
  and this decision should be revisited.
- Resume trusts the revision recorded at fetch time, so upstream revisions are
  only detected by a periodic `--no-resume` run. That is a deliberate trade:
  re-verifying every month on every run would give up the reason for choosing
  the archive.
