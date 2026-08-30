"""Phase 0A research bootstrap for the crypto pre-pump project.

Scope guard: this package collects and normalizes public market data for
research. It contains no order placement, withdrawal or account endpoint and
must never gain one (enforced by tests/unit/test_scope_guard.py).
"""

from prepump.version import COLLECTOR_VERSION, SCHEMA_VERSION

__all__ = ["COLLECTOR_VERSION", "SCHEMA_VERSION"]
