"""Data model for a single practice question + validation.

A question is a plain dict on disk (JSON). This module documents the schema,
validates it, and provides small helpers. Keeping it dict-based (instead of a
rigid class) makes it trivial to add the 1000 incoming questions by hand or by
script without fighting a schema.
"""

from __future__ import annotations

# Allowed values
QUESTION_TYPES = {"calculation", "lookup", "theory"}
BLOCKS = {"A", "B", "C", "D", "E"}

# Every question MUST have these keys.
REQUIRED_KEYS = ("id", "question", "answer")

# Full set of recognised keys (others are kept but warned about).
KNOWN_KEYS = {
    "id",            # unique string, e.g. "S04-001"
    "exam_set",      # which set this came from, e.g. "section4-practice" or "mock01"
    "section",       # CEC section number (int) or None
    "block",         # exam blueprint block "A".."E" or None
    "topic",         # short topic tag, e.g. "ampacity"
    "type",          # one of QUESTION_TYPES
    "question",      # the question text
    "options",       # list[str] like ["A) ...", "B) ..."] or None for open answer
    "answer",        # canonical correct answer (text) — always present
    "answer_key",    # correct option letter "A".."D" for multiple choice, else None
    "references",    # list[str], e.g. ["Rule 4-004 1)a)", "Table 2"]
    "solution_steps",# list[str], the step-by-step teaching for how to find the answer
    "code_edition",  # "2024" by default
    "verified",      # bool — has a human/AI checked it against the 2024 code?
    "needs_review",  # bool — flagged (missing answer key, OCR noise, 2021 vs 2024, ...)
    "source",        # free-text provenance
}


def validate_question(q: dict) -> list[str]:
    """Return a list of human-readable problems. Empty list == valid."""
    problems: list[str] = []
    if not isinstance(q, dict):
        return ["question is not an object"]

    for key in REQUIRED_KEYS:
        if not q.get(key):
            problems.append(f"missing required field '{key}'")

    qtype = q.get("type")
    if qtype is not None and qtype not in QUESTION_TYPES:
        problems.append(f"type '{qtype}' not in {sorted(QUESTION_TYPES)}")

    block = q.get("block")
    if block is not None and block not in BLOCKS:
        problems.append(f"block '{block}' not in {sorted(BLOCKS)}")

    options = q.get("options")
    if options is not None:
        if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
            problems.append("options must be a list of strings")

    key = q.get("answer_key")
    if key is not None:
        if not options:
            problems.append("answer_key set but no options provided")
        elif key.upper() not in [letter_of(o) for o in options if letter_of(o)]:
            problems.append(f"answer_key '{key}' does not match any option letter")

    for listkey in ("references", "solution_steps"):
        val = q.get(listkey)
        if val is not None and (not isinstance(val, list) or not all(isinstance(x, str) for x in val)):
            problems.append(f"{listkey} must be a list of strings")

    return problems


def letter_of(option: str) -> str | None:
    """Extract the leading option letter from a string like 'A) 300A' -> 'A'."""
    s = option.strip()
    if len(s) >= 2 and s[0].isalpha() and s[1] in ").:-":
        return s[0].upper()
    return None


def is_multiple_choice(q: dict) -> bool:
    return bool(q.get("options"))


def unknown_keys(q: dict) -> set[str]:
    return set(q.keys()) - KNOWN_KEYS
