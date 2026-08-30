"""Render the matrix to the Markdown layout of the template."""

from __future__ import annotations

from prepump.matrix.model import Matrix, SourceRow

COLUMNS: list[tuple[str, str]] = [
    ("Domain", "domain"),
    ("Provider / Product", "_provider_product"),
    ("Endpoint / Dataset", "endpoint"),
    ("Exchanges / Chains", "exchanges_chains"),
    ("Symbol Coverage", "symbol_coverage"),
    ("Historical Start", "historical_start"),
    ("Granularity", "granularity"),
    ("Raw Backfill", "raw_backfill"),
    ("event_time Quality", "event_time_quality"),
    ("available_at / Latency Reliability", "available_at_reliability"),
    ("ingested_at Capturable", "ingested_at_capturable"),
    ("Revisions / Backfills", "revisions_backfills"),
    ("Rate Limits", "rate_limits"),
    ("Monthly Cost", "monthly_cost"),
    ("Bulk Backfill Cost", "bulk_backfill_cost"),
    ("Licensing / Redistribution", "licensing_redistribution"),
    ("PIT Suitability", "pit_suitability"),
    ("Known Gaps", "_known_gaps"),
    ("Status", "status"),
    ("Evidence / Docs", "_evidence"),
    ("Reviewed At", "reviewed_at"),
    ("Reviewer", "reviewer"),
]


def _cell(row: SourceRow, attr: str) -> str:
    if attr == "_provider_product":
        value = " / ".join(part for part in (row.provider, row.product) if part)
    elif attr == "_known_gaps":
        value = "; ".join(row.known_gaps)
    elif attr == "_evidence":
        value = row.evidence.official_docs or ""
    else:
        value = str(getattr(row, attr) or "")
    return value.replace("|", "\\|").replace("\n", " ").strip() or "—"


def render(matrix: Matrix) -> str:
    lines = [f"# {matrix.title}", ""]
    if matrix.generated_note:
        lines += [f"> {matrix.generated_note}", ""]
    lines.append("| " + " | ".join(name for name, _ in COLUMNS) + " |")
    lines.append("|" + "|".join("---" for _ in COLUMNS) + "|")
    for row in sorted(matrix.rows, key=lambda r: (r.domain, r.provider, r.endpoint)):
        lines.append("| " + " | ".join(_cell(row, attr) for _, attr in COLUMNS) + " |")
    lines += ["", "## Verification plan for rows still UNKNOWN", ""]
    pending = [row for row in matrix.rows if row.verification_plan]
    if not pending:
        lines.append("None.")
    for row in sorted(pending, key=lambda r: (r.domain, r.provider)):
        lines.append(f"### {row.domain} — {row.provider} {row.product}".rstrip())
        lines.append("")
        for step in row.verification_plan:
            lines.append(f"- {step}")
        lines.append("")
    return "\n".join(lines) + "\n"
