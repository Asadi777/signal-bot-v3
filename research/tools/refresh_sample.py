"""Copy a small slice of the offline dataset into the committed sample.

The sample exists so a reviewer can see the shape of the output without
running anything, and so schema changes show up as a diff. It is synthetic by
construction, which is also why committing it raises no redistribution
question at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "normalized" / "ohlcv_1m"
TARGET = ROOT / "sample" / "normalized" / "ohlcv_1m"
KEEP_DATES = ("date=2024-01-01", "date=2024-01-02")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit("run `make backfill-offline` first")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    copied = 0
    for sidecar in sorted(SOURCE.rglob("_partition.json")):
        if sidecar.parent.name not in KEEP_DATES:
            continue
        destination = TARGET / sidecar.parent.relative_to(SOURCE)
        destination.mkdir(parents=True, exist_ok=True)
        for item in sidecar.parent.iterdir():
            shutil.copy2(item, destination / item.name)
        copied += 1

    reports = ROOT / "sample" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for name in ("data_quality.md", "data_quality.json"):
        source = ROOT / "data" / "reports" / name
        if source.exists():
            shutil.copy2(source, reports / name)

    manifests = sorted((ROOT / "data" / "manifests").glob("*.json"))
    if manifests:
        target_dir = ROOT / "sample" / "manifests"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifests[-1], target_dir / "example_backfill_manifest.json")

    print(f"copied {copied} partitions to {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
