"""Loading, validating, importing and merging question files.

Question files live as JSON under data/questions/. Each file is either a list
of question objects, or an object {"exam_set": "...", "questions": [...]}.

To add new questions later (e.g. the 10 mock exams), drop files into
data/incoming/ and call import_incoming(): they are validated, de-duplicated by
id, and moved into data/questions/.
"""

from __future__ import annotations

import json
import pathlib
import shutil

from . import models

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
QUESTIONS_DIR = DATA / "questions"
INCOMING_DIR = DATA / "incoming"
BLUEPRINT_FILE = DATA / "blueprint.json"


def _coerce(raw, source_name: str) -> list[dict]:
    """Accept either a bare list or {"exam_set", "questions"} and return a list."""
    if isinstance(raw, dict) and "questions" in raw:
        default_set = raw.get("exam_set")
        items = raw["questions"]
        if default_set:
            for q in items:
                q.setdefault("exam_set", default_set)
        return items
    if isinstance(raw, list):
        return raw
    raise ValueError(f"{source_name}: top level must be a list or have a 'questions' key")


def load_file(path: pathlib.Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = _coerce(raw, path.name)
    for q in items:
        q.setdefault("source", path.name)
        q.setdefault("code_edition", "2024")
    return items


def load_all(directory: pathlib.Path = QUESTIONS_DIR) -> tuple[list[dict], list[str]]:
    """Load every question. Returns (questions, warnings). De-dupes by id."""
    questions: list[dict] = []
    warnings: list[str] = []
    seen: dict[str, str] = {}  # id -> source file

    for path in sorted(directory.glob("*.json")):
        try:
            items = load_file(path)
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"SKIPPED {path.name}: {exc}")
            continue
        for q in items:
            problems = models.validate_question(q)
            qid = q.get("id", "<no-id>")
            if problems:
                warnings.append(f"{path.name} [{qid}]: " + "; ".join(problems))
                if not q.get("id") or not q.get("question") or not q.get("answer"):
                    continue  # too broken to use
            if qid in seen:
                warnings.append(f"{path.name} [{qid}]: duplicate id (already in {seen[qid]}) — skipped")
                continue
            seen[qid] = path.name
            questions.append(q)
    return questions, warnings


def load_blueprint() -> dict:
    if BLUEPRINT_FILE.exists():
        return json.loads(BLUEPRINT_FILE.read_text(encoding="utf-8"))
    return {}


def import_incoming(apply: bool = True) -> dict:
    """Validate files in data/incoming/. If apply, move good files into questions/.

    Returns a report dict with per-file results.
    """
    existing, _ = load_all()
    existing_ids = {q["id"] for q in existing}
    report = {"files": [], "added": 0, "skipped": 0}

    for path in sorted(INCOMING_DIR.glob("*.json")):
        entry = {"file": path.name, "ok": 0, "problems": [], "new_ids": [], "dupes": []}
        try:
            items = load_file(path)
        except (ValueError, json.JSONDecodeError) as exc:
            entry["problems"].append(f"unreadable: {exc}")
            report["files"].append(entry)
            continue

        file_ok = True
        for q in items:
            probs = models.validate_question(q)
            qid = q.get("id", "<no-id>")
            if probs:
                entry["problems"].append(f"[{qid}] " + "; ".join(probs))
                file_ok = False
            elif qid in existing_ids or qid in entry["new_ids"]:
                entry["dupes"].append(qid)
            else:
                entry["ok"] += 1
                entry["new_ids"].append(qid)

        if file_ok and apply and entry["ok"] > 0 and not entry["problems"]:
            dest = QUESTIONS_DIR / path.name
            if dest.exists():
                dest = QUESTIONS_DIR / f"{path.stem}_imported{path.suffix}"
            shutil.move(str(path), str(dest))
            existing_ids.update(entry["new_ids"])
            report["added"] += entry["ok"]
        else:
            report["skipped"] += entry["ok"]
        report["files"].append(entry)

    return report


def stats(questions: list[dict]) -> dict:
    by_section: dict = {}
    by_block: dict = {}
    by_set: dict = {}
    needs_review = 0
    for q in questions:
        by_section[q.get("section")] = by_section.get(q.get("section"), 0) + 1
        by_block[q.get("block")] = by_block.get(q.get("block"), 0) + 1
        by_set[q.get("exam_set")] = by_set.get(q.get("exam_set"), 0) + 1
        if q.get("needs_review"):
            needs_review += 1
    return {
        "total": len(questions),
        "by_section": by_section,
        "by_block": by_block,
        "by_set": by_set,
        "needs_review": needs_review,
    }
