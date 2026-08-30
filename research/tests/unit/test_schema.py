import pyarrow as pa
import pytest

from prepump.normalize.schema import OHLCV_SCHEMA, SchemaValidationError, validate_table
from tests.helpers import make_table


def test_valid_table_passes():
    validate_table(make_table())


def test_missing_column_is_rejected():
    table = make_table()
    trimmed = table.drop_columns(["quote_volume"])
    with pytest.raises(SchemaValidationError, match="missing columns"):
        validate_table(trimmed)


def test_unknown_column_is_rejected():
    table = make_table()
    extended = table.append_column("surprise", pa.array(["x"] * table.num_rows))
    with pytest.raises(SchemaValidationError, match="unknown columns"):
        validate_table(extended)


def test_wrong_type_is_rejected():
    table = make_table()
    index = table.schema.get_field_index("close")
    broken = table.set_column(index, "close", pa.array([str(v) for v in table.column("close").to_pylist()]))
    with pytest.raises(SchemaValidationError, match="type"):
        validate_table(broken)


def test_null_in_non_nullable_column_is_rejected():
    table = make_table()
    index = table.schema.get_field_index("symbol")
    values = table.column("symbol").to_pylist()
    values[0] = None
    broken = table.set_column(index, OHLCV_SCHEMA.field("symbol").with_nullable(True), pa.array(values))
    with pytest.raises(SchemaValidationError, match="non-nullable"):
        validate_table(broken)


def test_non_utc_timestamp_is_rejected():
    table = make_table()
    index = table.schema.get_field_index("event_time")
    local = table.column("event_time").cast(pa.timestamp("ms", tz="Asia/Tehran"))
    broken = table.set_column(index, pa.field("event_time", pa.timestamp("ms", tz="Asia/Tehran")), local)
    with pytest.raises(SchemaValidationError, match="UTC"):
        validate_table(broken)
