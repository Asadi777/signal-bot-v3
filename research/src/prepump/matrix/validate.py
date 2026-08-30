"""Matrix validation rules.

The rule that matters: ``AVAILABLE`` requires all seven evidence items plus a
named reviewer and a review date. ``LIMITED`` is also a positive claim, so it
requires documentation, a verified sample, and recorded gaps. Everything else
only has to name who looked and when.
"""

from __future__ import annotations

from dataclasses import dataclass

from prepump.matrix.model import Matrix, SourceRow, Status

FULL_EVIDENCE_STATUSES = {Status.AVAILABLE}
PARTIAL_EVIDENCE_STATUSES = {Status.LIMITED}
PARTIAL_REQUIRED = ("official_docs", "verified_sample", "known_gaps_recorded")
REVIEW_REQUIRED_STATUSES = set(Status) - {Status.UNKNOWN}


@dataclass
class Violation:
    row: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.row}: {self.detail}"


def validate_row(row: SourceRow) -> list[Violation]:
    problems: list[Violation] = []

    if row.status in FULL_EVIDENCE_STATUSES:
        missing = row.evidence.missing()
        if missing:
            problems.append(
                Violation(row.key, "EVIDENCE_REQUIRED", f"status AVAILABLE needs all evidence; missing: {missing}")
            )
        for field_name in ("historical_start", "granularity", "rate_limits", "licensing_redistribution"):
            if not str(getattr(row, field_name)).strip():
                problems.append(
                    Violation(row.key, "FIELD_REQUIRED", f"status AVAILABLE needs a value for {field_name!r}")
                )
        if not row.known_gaps:
            problems.append(
                Violation(
                    row.key,
                    "GAPS_REQUIRED",
                    "status AVAILABLE needs known_gaps recorded (write 'none found' explicitly if that is the finding)",
                )
            )

    if row.status in PARTIAL_EVIDENCE_STATUSES:
        missing = [name for name in PARTIAL_REQUIRED if not str(getattr(row.evidence, name)).strip()]
        if missing:
            problems.append(
                Violation(row.key, "EVIDENCE_REQUIRED", f"status LIMITED needs evidence; missing: {missing}")
            )

    if row.status in REVIEW_REQUIRED_STATUSES:
        if not row.reviewer.strip():
            problems.append(Violation(row.key, "REVIEWER_REQUIRED", f"status {row.status} needs a reviewer"))
        if not row.reviewed_at.strip():
            problems.append(Violation(row.key, "REVIEW_DATE_REQUIRED", f"status {row.status} needs reviewed_at"))

    if row.status is Status.UNKNOWN and not row.verification_plan:
        problems.append(
            Violation(row.key, "PLAN_REQUIRED", "status UNKNOWN needs a verification_plan so the gap is actionable")
        )
    return problems


def validate_matrix(matrix: Matrix) -> list[Violation]:
    problems: list[Violation] = []
    seen: set[str] = set()
    for row in matrix.rows:
        if row.key in seen:
            problems.append(Violation(row.key, "DUPLICATE_ROW", "the same provider/endpoint appears twice"))
        seen.add(row.key)
        problems.extend(validate_row(row))
    return problems
