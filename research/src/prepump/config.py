"""Typed, fail-fast configuration.

Anything that changes the produced data is part of the config hash recorded in
the run manifest, so a dataset can always be traced back to the settings that
produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PREPUMP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = Path("./data")
    log_level: str = "INFO"

    # Outbound politeness. The archive-first strategy keeps REST volume tiny;
    # these are the hard ceilings that protect us from an IP ban.
    rate_limit_rps: float = Field(default=4.0, gt=0, le=50)
    max_retries: int = Field(default=5, ge=0, le=10)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_max_seconds: float = Field(default=60.0, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    binance_vision_base_url: str = "https://data.binance.vision"
    binance_spot_rest_base_url: str = "https://api.binance.com"
    binance_futures_rest_base_url: str = "https://fapi.binance.com"
    bybit_rest_base_url: str = "https://api.bybit.com"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return v

    def config_hash(self) -> str:
        """Stable hash of the settings that can influence produced data."""
        material = {
            "binance_vision_base_url": self.binance_vision_base_url,
            "binance_spot_rest_base_url": self.binance_spot_rest_base_url,
            "binance_futures_rest_base_url": self.binance_futures_rest_base_url,
            "bybit_rest_base_url": self.bybit_rest_base_url,
        }
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def normalized_dir(self) -> Path:
        return self.data_root / "normalized"

    @property
    def reports_dir(self) -> Path:
        return self.data_root / "reports"

    @property
    def manifests_dir(self) -> Path:
        return self.data_root / "manifests"


def load_settings(**overrides) -> Settings:
    return Settings(**overrides)
