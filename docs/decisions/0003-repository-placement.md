# 0003 — Research code lives in `research/` inside this repository

**Status:** Accepted (reversible)
**Date:** 2026-08-30

## Decision

The Phase 0A package lives in `research/` in the `signal-bot-v3` repository,
with its own `pyproject.toml`, dependencies and test suite. The existing
`signal_bot.py` (v3.3) is untouched.

## Why

One repository keeps the specification documents, the decision log, the
existing heuristic bot and the research pipeline in one history. The research
package shares nothing with the bot at runtime — separate dependencies,
separate entry points — so the coupling is organizational, not technical.

The bot is also not dead weight: its scoring rules are a ready-made candidate
for the heuristic baselines Master Spec §57.4 requires Stage A to be compared
against.

## Reversal

Moving `research/` into its own repository is a directory move plus a new
remote. Nothing in the package depends on its location.
