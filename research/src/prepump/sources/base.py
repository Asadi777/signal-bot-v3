"""Source abstraction.

A source knows three things: how to enumerate the fetch units covering a time
range, how to turn one fetched unit into raw klines, and what its timestamps
mean. Everything else (rate limiting, retries, writing, quality) is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from prepump.normalize.ohlcv import RawKline
from prepump.normalize.schema import AvailabilityMethod
from prepump.normalize.symbols import SymbolSpec


@dataclass(frozen=True)
class FetchUnit:
    """One retrievable chunk: an archive file, or one paginated REST window."""

    key: str
    url: str
    params: dict | None = None
    label: str = ""


@dataclass(frozen=True)
class SourceDescriptor:
    name: str
    exchange: str
    availability_method: AvailabilityMethod
    availability_delta_ms: int
    availability_uncertainty_ms: int
    notes: str = ""


@dataclass
class FetchResult:
    klines: list[RawKline]
    source_revision: str
    raw_bytes: bytes | None = None
    raw_filename: str | None = None
    meta: dict = field(default_factory=dict)


class OhlcvSource(Protocol):
    descriptor: SourceDescriptor

    def supports(self, spec: SymbolSpec) -> bool: ...

    def plan(self, spec: SymbolSpec, start_ms: int, end_ms: int) -> list[FetchUnit]: ...

    def fetch(self, spec: SymbolSpec, unit: FetchUnit) -> FetchResult: ...


class SourceUnavailable(Exception):
    """The source cannot serve this symbol/range at all (documented as a gap)."""
