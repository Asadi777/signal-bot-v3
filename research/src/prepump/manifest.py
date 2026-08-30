"""Run manifests.

Every command writes exactly one manifest describing what it did: versions,
config hash, which partitions were written or skipped, which raw artifacts were
fetched (with their checksums), the outbound-request statistics and every
error. Without this a dataset is not reproducible, only re-derivable by luck.
"""

from __future__ import annotations

import json
import platform
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from prepump.io.paths import ensure_dir
from prepump.timeutils import utc_now
from prepump.version import COLLECTOR_VERSION, SCHEMA_VERSION


def new_run_id() -> str:
    return f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


@dataclass
class RunManifest:
    run_id: str
    command: str
    config_hash: str
    started_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="milliseconds"))
    finished_at: str | None = None
    status: str = "RUNNING"
    parameters: dict = field(default_factory=dict)
    partitions_written: list[dict] = field(default_factory=list)
    partitions_skipped: list[dict] = field(default_factory=list)
    raw_artifacts: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    http_stats: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)

    @classmethod
    def start(cls, command: str, config_hash: str, parameters: dict | None = None) -> "RunManifest":
        return cls(
            run_id=new_run_id(),
            command=command,
            config_hash=config_hash,
            parameters=parameters or {},
        )

    def bump(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def record_error(self, where: str, error: BaseException | str) -> None:
        self.errors.append(
            {
                "where": where,
                "error": f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error),
                "at": utc_now().isoformat(timespec="milliseconds"),
            }
        )

    def record_gap(self, **payload) -> None:
        self.gaps.append(payload)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "command": self.command,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "schema_version": SCHEMA_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "config_hash": self.config_hash,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "parameters": self.parameters,
            "counters": self.counters,
            "http_stats": self.http_stats,
            "partitions_written": self.partitions_written,
            "partitions_skipped": self.partitions_skipped,
            "raw_artifacts": self.raw_artifacts,
            "gaps": self.gaps,
            "errors": self.errors,
        }

    def finish(self, manifests_dir: Path, status: str = "SUCCESS") -> Path:
        self.status = "FAILED" if self.errors and status == "SUCCESS" else status
        self.finished_at = utc_now().isoformat(timespec="milliseconds")
        ensure_dir(manifests_dir)
        path = manifests_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        return path
