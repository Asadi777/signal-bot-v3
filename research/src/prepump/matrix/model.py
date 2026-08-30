"""Machine-readable Data Source Matrix (Kickoff §B, Master Spec §65).

The matrix is the project's evidence ledger. Its whole value is that a status
cannot be nicer than the evidence behind it, so the evidence block is part of
the schema rather than a convention, and the validator refuses an optimistic
row rather than trusting the author.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Status(StrEnum):
    AVAILABLE = "AVAILABLE"
    LIMITED = "LIMITED"
    BLOCKED = "BLOCKED"
    NEEDS_NEGOTIATION = "NEEDS_NEGOTIATION"
    UNKNOWN = "UNKNOWN"
    REJECTED = "REJECTED"


class Domain(StrEnum):
    OHLCV = "OHLCV"
    TRADES = "Trades"
    DERIVATIVES_OI = "Derivatives / OI"
    FUNDING_BASIS = "Funding / Basis"
    LIQUIDATIONS = "Liquidations"
    ORDER_BOOK_L2 = "Order Book / L2"
    TRADE_FLOW = "Trade Flow / Microstructure"
    ONCHAIN_WALLET = "On-chain / Wallet"
    EXCHANGE_FLOWS = "Exchange Flows"
    WHALE_SMART_MONEY = "Whale / Smart Money"
    SOCIAL_ATTENTION = "Social Attention"
    NEWS_CATALYSTS = "News / Catalysts"
    FUNDAMENTAL_TOKENOMICS = "Fundamental / Tokenomics"
    LISTING_METADATA = "Listing / Delisting Metadata"


class Evidence(BaseModel):
    """The seven items the template requires before a source may be AVAILABLE."""

    official_docs: str = ""
    verified_sample: str = ""
    historical_depth_check: str = ""
    rate_limit_test: str = ""
    timestamp_semantics: str = ""
    cost_and_licensing: str = ""
    known_gaps_recorded: str = ""

    def missing(self) -> list[str]:
        return [name for name, value in self.model_dump().items() if not str(value).strip()]


class SourceRow(BaseModel):
    domain: Domain
    provider: str
    product: str = ""
    endpoint: str = ""
    exchanges_chains: str = ""
    symbol_coverage: str = ""
    historical_start: str = ""
    granularity: str = ""
    raw_backfill: str = ""
    event_time_quality: str = ""
    available_at_reliability: str = ""
    ingested_at_capturable: str = ""
    revisions_backfills: str = ""
    rate_limits: str = ""
    monthly_cost: str = ""
    bulk_backfill_cost: str = ""
    licensing_redistribution: str = ""
    pit_suitability: str = ""
    known_gaps: list[str] = Field(default_factory=list)
    status: Status = Status.UNKNOWN
    evidence: Evidence = Field(default_factory=Evidence)
    verification_plan: list[str] = Field(default_factory=list)
    notes: str = ""
    reviewed_at: str = ""
    reviewer: str = ""

    @model_validator(mode="after")
    def _no_placeholder_provider(self):
        if not self.provider.strip():
            raise ValueError("provider is required on every row")
        return self

    @property
    def key(self) -> str:
        return f"{self.domain}|{self.provider}|{self.product}|{self.endpoint}"


class Matrix(BaseModel):
    schema_version: str
    title: str = "Crypto Pre-Pump Project — Data Source Matrix"
    generated_note: str = ""
    rows: list[SourceRow] = Field(default_factory=list)
