"""Structural guards.

Phase 0A is research collection. Two properties are worth enforcing in code
rather than trusting to review: the package must contain no trading surface,
and it must not invent data to fill holes.
"""

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "prepump"

FORBIDDEN_TRADING_TOKENS = (
    "/api/v3/order",
    "/fapi/v1/order",
    "/v5/order/",
    "sapi/v1/capital/withdraw",
    "create_order",
    "place_order",
    "cancel_order",
    "withdraw(",
)

FORBIDDEN_IMPUTATION_TOKENS = (
    ".fillna(",
    ".interpolate(",
    "ffill(",
    "bfill(",
)


def python_sources():
    return [p for p in SRC.rglob("*.py")]


def test_no_trading_or_withdrawal_surface():
    offenders = []
    for path in python_sources():
        text = path.read_text()
        for token in FORBIDDEN_TRADING_TOKENS:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, f"research package must contain no trading surface: {offenders}"


def test_no_gap_filling():
    offenders = []
    for path in python_sources():
        text = path.read_text()
        for token in FORBIDDEN_IMPUTATION_TOKENS:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, f"missing data must be reported, never imputed: {offenders}"


def test_no_credentials_are_read_from_the_environment():
    # Every Phase 0A source is public. If this ever fails, a secret has entered
    # a phase that was designed not to need one.
    banned = ("API_KEY", "API_SECRET", "SECRET_KEY", "PRIVATE_KEY")
    offenders = [
        f"{path.name}: {token}"
        for path in python_sources()
        for token in banned
        if token in path.read_text()
    ]
    assert not offenders, offenders
