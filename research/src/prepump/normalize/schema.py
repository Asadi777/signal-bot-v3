"""The normalized OHLCV contract (``ohlcv_1m.v1``).

Three timestamps, three different questions:

``event_time``       when the candle *belongs* to (its open, UTC).
``available_at``     the earliest defensible time the system could have used
                     the finalized candle. For a closed candle that is its
                     close time plus a documented source delay.
``ingested_at``      when this pipeline actually stored it.

The eligibility rule of Master Spec §61.1 is ``available_at <= T``, never
``event_time <= T``. ``available_at_method`` and ``available_at_uncertainty_ms``
exist so that an approximation can never be mistaken for a measurement.
"""

from __future__ import annotations

from enum import StrEnum

import pyarrow as pa

TS = pa.timestamp("ms", tz="UTC")


class AvailabilityMethod(StrEnum):
    DERIVED_FROM_CLOSE = "DERIVED_FROM_CLOSE"
    MEASURED = "MEASURED"
    APPROXIMATED = "APPROXIMATED"


class DQFlag(StrEnum):
    OHLC_VIOLATION = "OHLC_VIOLATION"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    ZERO_VOLUME_WITH_RANGE = "ZERO_VOLUME_WITH_RANGE"
    CLOSE_TIME_MISMATCH = "CLOSE_TIME_MISMATCH"
    NOT_MINUTE_ALIGNED = "NOT_MINUTE_ALIGNED"
    QUOTE_VOLUME_MISSING = "QUOTE_VOLUME_MISSING"
    INFERRED_SYMBOL_SPLIT = "INFERRED_SYMBOL_SPLIT"


OHLCV_SCHEMA = pa.schema(
    [
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("market_type", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("symbol_canonical", pa.string(), nullable=False),
        pa.field("base_asset", pa.string(), nullable=False),
        pa.field("quote_asset", pa.string(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
        pa.field("event_time", TS, nullable=False),
        pa.field("event_time_close", TS, nullable=False),
        pa.field("available_at", TS, nullable=False),
        pa.field("available_at_method", pa.string(), nullable=False),
        pa.field("available_at_uncertainty_ms", pa.int32(), nullable=False),
        pa.field("ingested_at", TS, nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("base_volume", pa.float64(), nullable=False),
        pa.field("quote_volume", pa.float64(), nullable=True),
        pa.field("trade_count", pa.int64(), nullable=True),
        pa.field("taker_buy_base", pa.float64(), nullable=True),
        pa.field("taker_buy_quote", pa.float64(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("collector_version", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("dq_flags", pa.list_(pa.string()), nullable=False),
    ]
)

PRIMARY_KEY = ("exchange", "market_type", "symbol", "interval", "event_time")

# Columns that arrive free in the same candle payload but are arguably
# microstructure rather than plain OHLCV. Master Spec §57.1 forbids mixing
# microstructure into the Stage A baseline, and the spec does not say which
# side of that line these fall on. They are therefore always stored and
# flagged not-allowed until Majid decides. See docs/decisions/0002.
STAGE_A_DISALLOWED_COLUMNS = ("trade_count", "taker_buy_base", "taker_buy_quote")


class SchemaValidationError(ValueError):
    pass


def validate_table(table: pa.Table) -> None:
    """Reject anything that does not match the contract exactly."""
    actual = {f.name: f for f in table.schema}
    expected = {f.name: f for f in OHLCV_SCHEMA}

    missing = sorted(set(expected) - set(actual))
    if missing:
        raise SchemaValidationError(f"missing columns: {missing}")
    unknown = sorted(set(actual) - set(expected))
    if unknown:
        raise SchemaValidationError(f"unknown columns: {unknown}")

    for name, field in expected.items():
        if not actual[name].type.equals(field.type):
            raise SchemaValidationError(f"column {name!r} has type {actual[name].type}, expected {field.type}")
        if not field.nullable and table.column(name).null_count:
            raise SchemaValidationError(f"column {name!r} is non-nullable but contains nulls")

    for name in ("event_time", "event_time_close", "available_at", "ingested_at"):
        tz = table.schema.field(name).type.tz
        if tz != "UTC":
            raise SchemaValidationError(f"column {name!r} must be UTC, got tz={tz!r}")
