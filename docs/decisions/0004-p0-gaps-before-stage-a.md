# 0004 — Three P0 decisions missing from the Master Specification

**Status:** OPEN — blocks Stage A, does not block Phase 0A
**Date:** 2026-08-30

Recorded here so they are not lost between documents. None of these prevents
data collection; all three prevent Stage A results from meaning anything.

## 1. Prediction sampling policy

The spec never says what a prediction is issued *on*: every minute for every
symbol? every 15 minutes? only after the not-started gate passes? Without it,
`Precision@K`, `false alerts per day` and AUPRC are not comparable between
runs, and with 500 symbols across 1440 minutes the base rate is small enough
that almost any model looks excellent.

Needs: a cadence, a deduplication rule for overlapping windows, and an explicit
statement that evaluation is event-level rather than row-level.

## 2. Feature embargo

The 15-minute rule constrains when a *signal* may be issued. It says nothing
about where a *feature window* may end. If features run up to the onset, the
model learns that the move has already started and Stage A's lift is an
artefact.

Proposed rule: feature windows end at `onset − 15min`, and walk-forward splits
purge and embargo around each event. Worth promoting to `ARCHITECTURAL RULE`.

Cheap diagnostic that settles it: run Stage A at feature cutoffs of −5, −15,
−30 and −60 minutes and watch how much lift survives distance from onset.

## 3. Alert budget

§48.4 acknowledges the tension between recall-first and a small number of
signals but does not resolve it. Without an approximate budget (say 10–30
formal alerts per day) there is no K, so `Precision@K` is undefined and the
Gate decision has nothing to weigh against recall.

Proposed: fix an order-of-magnitude budget now and make the headline KPI
`Recall @ fixed alert budget`.

## Related, smaller, also open

- Which price the "+50% in 24h" label is measured on (high or close, last or
  mark) and a minimum traded-volume condition so untradeable wicks cannot
  become positives.
- Event deduplication at token level rather than pair level, since one token
  listed on two venues otherwise produces two "pumps".
- A fee and slippage model for paper trading; without one, P&L on small caps is
  fiction.
