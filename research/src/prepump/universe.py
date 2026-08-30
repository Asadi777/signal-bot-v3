"""Provisional smoke-test universe.

This is a pipeline test fixture, not a research universe. Master Spec §65.3
forbids treating today's listed symbols as the historical universe, so the
config carries its own limitations and any consumer that loads it gets them
back and is expected to keep them attached to whatever it produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from prepump.normalize.symbols import SymbolSpec, make_spec


@dataclass(frozen=True)
class UniverseEntry:
    spec: SymbolSpec
    liquidity_tier: str
    expected_status: str
    rationale: str


@dataclass(frozen=True)
class Universe:
    version: str
    status: str
    limitations: list[str]
    entries: list[UniverseEntry]

    @property
    def specs(self) -> list[SymbolSpec]:
        return [entry.spec for entry in self.entries]

    def filter(self, exchange: str | None = None, market_type: str | None = None) -> list[UniverseEntry]:
        return [
            entry
            for entry in self.entries
            if (exchange is None or entry.spec.exchange == exchange)
            and (market_type is None or entry.spec.market_type == market_type)
        ]


def load_universe(path: Path) -> Universe:
    payload = yaml.safe_load(Path(path).read_text())
    if not payload or "symbols" not in payload:
        raise ValueError(f"universe file {path} has no 'symbols' section")
    entries = []
    for item in payload["symbols"]:
        spec = make_spec(
            item["exchange"],
            item["market_type"],
            item["symbol"],
            item.get("base_asset"),
            item.get("quote_asset"),
            note=item.get("rationale", ""),
        )
        entries.append(
            UniverseEntry(
                spec=spec,
                liquidity_tier=item.get("liquidity_tier", "unknown"),
                expected_status=item.get("expected_status", "unknown"),
                rationale=item.get("rationale", ""),
            )
        )
    return Universe(
        version=payload.get("version", "unversioned"),
        status=payload.get("status", "PROVISIONAL"),
        limitations=payload.get("limitations", []),
        entries=entries,
    )
