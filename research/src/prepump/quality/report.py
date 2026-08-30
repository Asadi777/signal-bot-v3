"""Data-quality report generation (Kickoff §F).

Two outputs from one computation: JSON for machines and later diffing, and
Markdown for a human reviewer. Every number is traceable to a partition, and
the report states its own dataset provenance so a report can never be read
against the wrong dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from prepump.io.paths import PARTITION_FILE, PARTITION_SIDECAR, ensure_dir
from prepump.quality.checks import SymbolQuality, analyze
from prepump.timeutils import ms_to_dt, utc_now
from prepump.version import COLLECTOR_VERSION, SCHEMA_VERSION


def dataset_footprint(normalized_root: Path, dataset: str = "ohlcv_1m") -> dict:
    base = normalized_root / dataset
    partitions = sorted(base.rglob(PARTITION_SIDECAR)) if base.exists() else []
    total_bytes = 0
    sources: dict[str, int] = {}
    for sidecar in partitions:
        data_file = sidecar.parent / PARTITION_FILE
        if data_file.exists():
            total_bytes += data_file.stat().st_size
        try:
            meta = json.loads(sidecar.read_text())
            sources[meta.get("source", "unknown")] = sources.get(meta.get("source", "unknown"), 0) + 1
        except json.JSONDecodeError:
            continue
    return {
        "dataset": dataset,
        "partitions": len(partitions),
        "parquet_bytes": total_bytes,
        "parquet_mib": round(total_bytes / (1024 * 1024), 3),
        "partitions_by_source": dict(sorted(sources.items())),
    }


def build_report(
    table: pa.Table,
    normalized_root: Path,
    *,
    dataset: str = "ohlcv_1m",
    run_id: str | None = None,
    cross_source: list[dict] | None = None,
    notes: list[str] | None = None,
) -> dict:
    per_symbol: list[SymbolQuality] = analyze(table)
    footprint = dataset_footprint(normalized_root, dataset)

    totals = {
        "rows": sum(item.rows for item in per_symbol),
        "symbols": len(per_symbol),
        "duplicate_rows": sum(item.duplicate_rows for item in per_symbol),
        "out_of_order_rows": sum(item.out_of_order_rows for item in per_symbol),
        "missing_minutes": sum(item.missing_minutes for item in per_symbol),
        "expected_minutes": sum(item.expected_minutes for item in per_symbol),
        "zero_volume_minutes": sum(item.zero_volume_minutes for item in per_symbol),
    }
    flags: dict[str, int] = {}
    for item in per_symbol:
        for flag, count in item.dq_flag_counts.items():
            flags[flag] = flags.get(flag, 0) + count

    return {
        "generated_at": utc_now().isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "run_id": run_id,
        "footprint": footprint,
        "totals": totals,
        "dq_flag_totals": dict(sorted(flags.items())),
        "per_symbol": [item.to_dict() for item in per_symbol],
        "cross_source": cross_source or [],
        "notes": notes or [],
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Data Quality Report")
    add("")
    add(f"- Generated at: `{report['generated_at']}`")
    add(f"- Schema version: `{report['schema_version']}`")
    add(f"- Collector version: `{report['collector_version']}`")
    if report.get("run_id"):
        add(f"- Run id: `{report['run_id']}`")
    footprint = report["footprint"]
    add(f"- Dataset: `{footprint['dataset']}` — {footprint['partitions']} partitions, {footprint['parquet_mib']} MiB")
    add(f"- Partitions by source: `{footprint['partitions_by_source']}`")
    add("")

    totals = report["totals"]
    add("## Totals")
    add("")
    add("| Metric | Value |")
    add("|---|---:|")
    for key in (
        "symbols",
        "rows",
        "expected_minutes",
        "missing_minutes",
        "duplicate_rows",
        "out_of_order_rows",
        "zero_volume_minutes",
    ):
        add(f"| {key.replace('_', ' ')} | {totals[key]:,} |")
    add("")

    if report["dq_flag_totals"]:
        add("## Row-level flags")
        add("")
        add("| Flag | Rows |")
        add("|---|---:|")
        for flag, count in report["dq_flag_totals"].items():
            add(f"| `{flag}` | {count:,} |")
        add("")

    add("## Per symbol")
    add("")
    add("| Symbol | Market | Rows | Range (UTC) | Missing min | Missing % | Dupes | Out of order | Longest gap (min) |")
    add("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for item in report["per_symbol"]:
        first = ms_to_dt(item["first_event_time_ms"]).strftime("%Y-%m-%d %H:%M") if item["first_event_time_ms"] else "-"
        last = ms_to_dt(item["last_event_time_ms"]).strftime("%Y-%m-%d %H:%M") if item["last_event_time_ms"] else "-"
        add(
            f"| `{item['symbol']}` | {item['market_type']} | {item['rows']:,} | {first} → {last} | "
            f"{item['missing_minutes']:,} | {item['missing_pct']} | {item['duplicate_rows']} | "
            f"{item['out_of_order_rows']} | {item['longest_gap_minutes']} |"
        )
    add("")

    add("## Gap classification")
    add("")
    add("A missing minute is not automatically a defect: an exchange emits no candle for a minute with no trades.")
    add("Gaps are therefore classified by run length and never filled.")
    add("")
    add("| Symbol | Gap classification | Occurrences |")
    add("|---|---|---:|")
    any_gap = False
    for item in report["per_symbol"]:
        for label, count in (item["gap_runs"] or {}).items():
            any_gap = True
            add(f"| `{item['symbol']}` | `{label}` | {count:,} |")
    if not any_gap:
        add("| — | no gaps detected | 0 |")
    add("")

    examples = [(item["symbol"], ex) for item in report["per_symbol"] for ex in item["gap_examples"]]
    if examples:
        add("### Gaps needing investigation (first 10)")
        add("")
        add("| Symbol | After | Missing minutes | Classification |")
        add("|---|---|---:|---|")
        for symbol, example in examples[:10]:
            add(f"| `{symbol}` | {example['after']} | {example['missing_minutes']} | `{example['classification']}` |")
        add("")

    if report["cross_source"]:
        add("## Cross-source comparison")
        add("")
        add("| Symbol | Left | Right | Overlap | Only left | Only right | Mismatching |")
        add("|---|---|---|---:|---:|---:|---:|")
        for entry in report["cross_source"]:
            add(
                f"| `{entry.get('symbol', '?')}` | {entry.get('left_source', '?')} | {entry.get('right_source', '?')} | "
                f"{entry.get('overlapping_minutes', 0):,} | {entry.get('only_in_left', 0):,} | "
                f"{entry.get('only_in_right', 0):,} | {entry.get('mismatching_minutes', 0):,} |"
            )
        add("")
    else:
        add("## Cross-source comparison")
        add("")
        add("Not run in this report. Requires live access to a second endpoint for the same symbol and range.")
        add("")

    if report["notes"]:
        add("## Notes and known limitations")
        add("")
        for note in report["notes"]:
            add(f"- {note}")
        add("")
    return "\n".join(lines) + "\n"


def write_report(report: dict, reports_dir: Path, stem: str = "data_quality") -> tuple[Path, Path]:
    ensure_dir(reports_dir)
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path
