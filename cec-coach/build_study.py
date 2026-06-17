#!/usr/bin/env python3
"""Merge data/study/group_*.json into web/study.js (window.CEC_STUDY).
Run after editing study content:  python3 build_study.py
"""
import json, glob, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "web" / "study.js"


def order_key(s):
    sec = s.get("section")
    if sec == "occ":
        return 1000
    try:
        return int(sec)
    except (TypeError, ValueError):
        return 999


def main():
    sections = []
    for f in sorted(glob.glob(str(ROOT / "data" / "study" / "group_*.json"))):
        sections.extend(json.loads(pathlib.Path(f).read_text(encoding="utf-8")))
    sections.sort(key=order_key)
    data = {"sections": sections}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.CEC_STUDY = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    kw = sum(len(s.get("keywords", [])) for s in sections)
    print(f"Wrote {OUT}: {len(sections)} sections, {kw} keyword rows.")


if __name__ == "__main__":
    main()
