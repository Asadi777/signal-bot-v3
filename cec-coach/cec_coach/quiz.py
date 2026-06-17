"""Question selection for the different practice modes."""

from __future__ import annotations

import random

from . import progress as progress_mod

# Exam blueprint default weights (share of a 100-question exam) by block.
# Used when building a weighted mock exam if no explicit exam_set is chosen.
DEFAULT_BLOCK_WEIGHTS = {"A": 11, "B": 29, "C": 30, "D": 20, "E": 10}


def by_section(questions: list[dict], section: int) -> list[dict]:
    return [q for q in questions if q.get("section") == section]


def by_set(questions: list[dict], exam_set: str) -> list[dict]:
    return [q for q in questions if q.get("exam_set") == exam_set]


def sections_available(questions: list[dict]) -> list[int]:
    return sorted({q["section"] for q in questions if isinstance(q.get("section"), int)})


def sets_available(questions: list[dict]) -> list[str]:
    return sorted({q["exam_set"] for q in questions if q.get("exam_set")})


def weak_first(questions: list[dict], prog: dict, limit: int | None = None) -> list[dict]:
    """Order questions so the ones you struggle with (or have never seen) come first."""
    ordered = sorted(
        questions,
        key=lambda q: (-progress_mod.weakness(prog, q["id"]), random.random()),
    )
    return ordered[:limit] if limit else ordered


def weighted_mock(questions: list[dict], n: int = 100,
                  weights: dict | None = None, seed: int | None = None) -> list[dict]:
    """Build an n-question exam roughly matching the blueprint block weights.

    Falls back gracefully when a block is short on questions.
    """
    weights = weights or DEFAULT_BLOCK_WEIGHTS
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    for q in questions:
        pools.setdefault(q.get("block") or "?", []).append(q)

    total_weight = sum(weights.get(b, 0) for b in weights)
    picked: list[dict] = []
    for block, w in weights.items():
        want = round(n * w / total_weight) if total_weight else 0
        pool = pools.get(block, [])
        rng.shuffle(pool)
        picked.extend(pool[:want])

    # Top up (or trim) to exactly n from whatever is left.
    if len(picked) < n:
        chosen_ids = {q["id"] for q in picked}
        leftovers = [q for q in questions if q["id"] not in chosen_ids]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: n - len(picked)])
    rng.shuffle(picked)
    return picked[:n]


def shuffled(questions: list[dict], seed: int | None = None) -> list[dict]:
    qs = list(questions)
    random.Random(seed).shuffle(qs)
    return qs
