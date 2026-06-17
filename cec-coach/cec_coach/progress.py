"""Per-question progress tracking, stored as JSON next to the data.

Used to power weak-area drilling and spaced repetition: questions you get
wrong (or have never seen) are prioritised.
"""

from __future__ import annotations

import json
import pathlib
import time

PROGRESS_FILE = pathlib.Path(__file__).resolve().parent.parent / "data" / "progress.json"


def load() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save(data: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def record(data: dict, qid: str, correct: bool) -> None:
    rec = data.setdefault(qid, {"seen": 0, "correct": 0, "wrong": 0, "last": None})
    rec["seen"] += 1
    rec["correct" if correct else "wrong"] += 1
    rec["last"] = time.strftime("%Y-%m-%d %H:%M:%S")


def weakness(data: dict, qid: str) -> float:
    """Higher score = needs practice more. Unseen questions score highest."""
    rec = data.get(qid)
    if not rec or rec["seen"] == 0:
        return 1.0
    return rec["wrong"] / rec["seen"]


def summary(data: dict) -> dict:
    seen = sum(1 for r in data.values() if r["seen"] > 0)
    total_attempts = sum(r["seen"] for r in data.values())
    total_correct = sum(r["correct"] for r in data.values())
    return {
        "questions_seen": seen,
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "accuracy": (total_correct / total_attempts) if total_attempts else 0.0,
    }
