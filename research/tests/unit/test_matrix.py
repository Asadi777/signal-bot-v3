from pathlib import Path

import yaml

from prepump.matrix.model import Domain, Evidence, Matrix, SourceRow, Status
from prepump.matrix.render import render
from prepump.matrix.validate import validate_matrix, validate_row

MATRIX_PATH = Path(__file__).resolve().parents[2] / "config" / "data_source_matrix_v1.yaml"

FULL_EVIDENCE = Evidence(
    official_docs="https://example.test/docs",
    verified_sample="data/raw/sample.zip sha256:abc",
    historical_depth_check="first candle 2019-09-01, checked 2026-08-30",
    rate_limit_test="60 files in 4 minutes, no throttling observed",
    timestamp_semantics="open_time is candle start, ms, UTC",
    cost_and_licensing="free; redistribution not permitted",
    known_gaps_recorded="no candle emitted for zero-trade minutes",
)


def available_row(**overrides) -> SourceRow:
    payload = dict(
        domain=Domain.OHLCV,
        provider="Example",
        product="Archive",
        endpoint="/x",
        historical_start="2019-09",
        granularity="1m",
        rate_limits="none observed",
        licensing_redistribution="free, no redistribution",
        known_gaps=["zero-trade minutes emit no candle"],
        status=Status.AVAILABLE,
        evidence=FULL_EVIDENCE,
        reviewer="Majid",
        reviewed_at="2026-08-30",
    )
    payload.update(overrides)
    return SourceRow(**payload)


def test_available_row_with_full_evidence_passes():
    assert validate_row(available_row()) == []


def test_available_without_evidence_is_rejected():
    problems = validate_row(available_row(evidence=Evidence()))
    assert any(p.rule == "EVIDENCE_REQUIRED" for p in problems)


def test_each_missing_evidence_item_blocks_available():
    for field in Evidence().model_dump():
        evidence = FULL_EVIDENCE.model_copy(update={field: ""})
        problems = validate_row(available_row(evidence=evidence))
        assert any(field in p.detail for p in problems), f"{field} must be required for AVAILABLE"


def test_available_requires_gaps_to_be_recorded_explicitly():
    problems = validate_row(available_row(known_gaps=[]))
    assert any(p.rule == "GAPS_REQUIRED" for p in problems)


def test_available_requires_the_descriptive_fields():
    problems = validate_row(available_row(rate_limits=""))
    assert any(p.rule == "FIELD_REQUIRED" for p in problems)


def test_reviewer_and_date_are_required_for_any_claim():
    problems = validate_row(available_row(reviewer="", reviewed_at=""))
    assert {p.rule for p in problems} >= {"REVIEWER_REQUIRED", "REVIEW_DATE_REQUIRED"}


def test_limited_needs_docs_sample_and_gaps():
    row = SourceRow(
        domain=Domain.TRADES,
        provider="Example",
        status=Status.LIMITED,
        reviewer="Majid",
        reviewed_at="2026-08-30",
    )
    problems = validate_row(row)
    assert any(p.rule == "EVIDENCE_REQUIRED" for p in problems)


def test_unknown_row_must_carry_a_verification_plan():
    row = SourceRow(domain=Domain.TRADES, provider="Example", status=Status.UNKNOWN)
    assert any(p.rule == "PLAN_REQUIRED" for p in validate_row(row))
    row.verification_plan = ["fetch one sample"]
    assert validate_row(row) == []


def test_duplicate_rows_are_flagged():
    row = SourceRow(domain=Domain.TRADES, provider="Example", status=Status.UNKNOWN, verification_plan=["x"])
    matrix = Matrix(schema_version="data_source_matrix.v1", rows=[row, row.model_copy()])
    assert any(p.rule == "DUPLICATE_ROW" for p in validate_matrix(matrix))


def test_shipped_matrix_is_valid_and_claims_nothing_unverified():
    matrix = Matrix.model_validate(yaml.safe_load(MATRIX_PATH.read_text()))
    assert validate_matrix(matrix) == []
    # Nothing may be promoted above UNKNOWN until evidence exists.
    assert {row.status for row in matrix.rows} == {Status.UNKNOWN}


def test_render_produces_the_template_columns():
    matrix = Matrix.model_validate(yaml.safe_load(MATRIX_PATH.read_text()))
    text = render(matrix)
    assert "| Domain |" in text and "| Status |" in text and "Reviewer |" in text
    assert text.count("\n|") >= len(matrix.rows)
