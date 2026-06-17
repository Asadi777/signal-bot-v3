#!/usr/bin/env python3
"""Bundle the question bank + blueprint into web/questions.js so the web app
works as a plain static site (no server, no fetch/CORS issues — opens by URL
on GitHub Pages OR by double-clicking index.html, and works offline).

Run this whenever you add questions:  python3 build_web.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
QUESTIONS_DIR = ROOT / "data" / "questions"
BLUEPRINT = ROOT / "data" / "blueprint.json"
OUT = ROOT / "web" / "questions.js"


def load_all():
    questions = []
    seen = set()
    for path in sorted(QUESTIONS_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw
        default_set = raw.get("exam_set") if isinstance(raw, dict) else None
        for q in items:
            if default_set:
                q.setdefault("exam_set", default_set)
            qid = q.get("id")
            if not qid or qid in seen:
                continue
            seen.add(qid)
            questions.append(q)
    return questions


def main():
    questions = load_all()
    blueprint = json.loads(BLUEPRINT.read_text(encoding="utf-8")) if BLUEPRINT.exists() else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {"questions": questions, "blueprint": blueprint}
    OUT.write_text(
        "window.CEC_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT} with {len(questions)} questions.")


if __name__ == "__main__":
    main()
