"""Version metadata stamped onto every record, manifest and report.

These are independent of the package version on purpose: a collector can be
fixed without changing the data contract, and the schema can change without
touching collection logic. Master Spec v5 requires both to be reconstructible
from any stored observation.
"""

# Bump when the normalized output contract changes in any way that a consumer
# could observe (column added/removed/retyped, semantics of a field changed).
SCHEMA_VERSION = "ohlcv_1m.v1"

# Bump when collection or normalization logic changes in a way that could
# alter the produced values for identical source input.
COLLECTOR_VERSION = "0.1.0"

# Schema version for the exchange metadata snapshots.
METADATA_SCHEMA_VERSION = "exchange_metadata.v1"

# Schema version for the machine-readable Data Source Matrix.
MATRIX_SCHEMA_VERSION = "data_source_matrix.v1"
