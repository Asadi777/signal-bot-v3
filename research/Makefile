# Reproducible entry points. Every target here is safe to re-run.
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
FIXTURES := data/fixtures/vision

.PHONY: help setup test test-all test-integration selftest fixtures backfill-offline quality matrix sample clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  %-20s %s\n", $$1, $$2}'

setup: ## Create the venv and install the package with dev extras
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test: ## Unit and offline end-to-end tests (no network)
	$(BIN)/python -m pytest

test-integration: ## Live-network tests that turn matrix rows into evidence
	$(BIN)/python -m pytest -m integration

test-all: test test-integration ## Everything

selftest: ## Prove idempotency, resume and determinism offline
	$(BIN)/prepump selftest-offline

fixtures: ## Generate the synthetic offline archive
	$(BIN)/prepump make-fixtures --out $(FIXTURES) --months 2024-01

backfill-offline: fixtures ## Backfill the synthetic universe with no network
	$(BIN)/prepump backfill-ohlcv \
		--universe config/smoke_universe_offline.yaml \
		--start 2024-01 --end 2024-01 \
		--base-url "file://$(CURDIR)/$(FIXTURES)"

quality: ## Data-quality report over the normalized dataset
	$(BIN)/prepump quality-report

matrix: ## Validate the Data Source Matrix and render it to docs/
	$(BIN)/prepump validate-matrix
	$(BIN)/prepump render-matrix --out ../docs/data_source_matrix_v1.md

sample: backfill-offline quality ## Regenerate the committed sample dataset
	$(PY) tools/refresh_sample.py

clean: ## Remove generated data (never touches raw archives you fetched on purpose)
	rm -rf data/normalized data/reports data/manifests data/selftest

# --- network-enabled Phase 0A run (requires egress to the exchange hosts) ---
.PHONY: live-metadata live-backfill live-verify

live-metadata: ## Snapshot exchange metadata (start this cadence as early as possible)
	$(BIN)/prepump collect-metadata --exchange all

live-backfill: ## Backfill the real smoke universe from the public archive
	$(BIN)/prepump backfill-ohlcv --universe config/smoke_universe_v1.yaml --start 2024-01 --end 2024-03

live-verify: ## Run the evidence-gathering integration tests and the Bybit depth probe
	$(BIN)/python -m pytest -m integration
	$(BIN)/prepump probe-bybit-depth --symbol BTCUSDT
