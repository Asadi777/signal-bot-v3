"""Interactive command-line coach.

Run from the project root:  python main.py
"""

from __future__ import annotations

import textwrap
import time

from . import models, progress as progress_mod, quiz, store

LINE = "─" * 64


def _wrap(text: str, indent: str = "") -> str:
    out = []
    for para in str(text).splitlines() or [""]:
        out.append(textwrap.fill(para, width=64,
                                 initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out)


def _input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def ask_question(q: dict, prog: dict, *, reveal_pause: bool = True) -> bool | None:
    """Present one question, grade it, teach the answer. Returns True/False/None(skip)."""
    print("\n" + LINE)
    sec = q.get("section")
    tag = f"Section {sec}" if sec else (q.get("topic") or "")
    print(f"[{q.get('id','?')}]  {tag}   ({q.get('type','?')})")
    print(_wrap(q["question"]))

    mc = models.is_multiple_choice(q)
    if mc:
        print()
        for opt in q["options"]:
            print(_wrap(opt, indent="   "))

    if mc and q.get("answer_key"):
        ans = _input("\nYour answer (A/B/C/D, 's' skip, 'q' quit): ").upper()
        if ans in ("Q", "QUIT"):
            return "quit"
        if ans in ("S", "SKIP", ""):
            correct = None
        else:
            correct = (ans == q["answer_key"].upper())
    else:
        # Open-answer: think, then reveal and self-grade.
        cmd = _input("\nThink it through, then press Enter to reveal ('s' skip, 'q' quit): ").lower()
        if cmd in ("q", "quit"):
            return "quit"
        if cmd in ("s", "skip"):
            return None

    # ----- reveal / teach -----
    print("\n" + "·" * 64)
    if mc and q.get("answer_key"):
        if correct is True:
            print("✅  Correct!")
        elif correct is False:
            print(f"❌  Not quite. Correct answer: {q['answer_key']}")
        print(f"Answer:  {q['answer']}")
    else:
        print(f"Answer:  {q['answer']}")

    if q.get("references"):
        print("\nWhere to find it in the Code:")
        for r in q["references"]:
            print(f"   • {r}")

    if q.get("solution_steps"):
        print("\nStep-by-step:")
        for i, step in enumerate(q["solution_steps"], 1):
            print(_wrap(f"{i}. {step}", indent="   "))

    if q.get("needs_review"):
        print("\n⚠️  Flagged for review (answer/edition to be verified against CEC 2024).")

    if not (mc and q.get("answer_key")):
        sg = _input("\nDid you get it right? (y/n, Enter to skip grading): ").lower()
        correct = True if sg.startswith("y") else False if sg.startswith("n") else None

    if correct is not None:
        progress_mod.record(prog, q["id"], correct)
    if reveal_pause:
        _input("\nPress Enter for the next question… ")
    return correct


def run_session(questions: list[dict], prog: dict, *, title: str, timed_minutes: int | None = None):
    if not questions:
        print("No questions available for this selection yet.")
        return
    print(f"\n=== {title} — {len(questions)} questions ===")
    if timed_minutes:
        print(f"⏱  Time limit: {timed_minutes} minutes "
              f"(~{timed_minutes*60/len(questions):.0f}s per question). Good luck!")
    start = time.time()
    right = wrong = skipped = 0
    for idx, q in enumerate(questions, 1):
        if timed_minutes and (time.time() - start) > timed_minutes * 60:
            print("\n⏱  Time is up!")
            break
        print(f"\n({idx}/{len(questions)})", end="")
        result = ask_question(q, prog)
        if result == "quit":
            break
        if result is True:
            right += 1
        elif result is False:
            wrong += 1
        else:
            skipped += 1
    progress_mod.save(prog)
    graded = right + wrong
    print("\n" + LINE)
    print(f"Session done: {right} right, {wrong} wrong, {skipped} skipped.")
    if graded:
        pct = 100 * right / graded
        print(f"Score on graded questions: {pct:.0f}%  "
              f"({'PASS ✅ (≥70%)' if pct >= 70 else 'below 70% — keep practising'})")
    print(f"Time: {(time.time()-start)/60:.1f} min")


# --------------------------------------------------------------------------- menus

def _choose_section(questions, prog):
    secs = quiz.sections_available(questions)
    if not secs:
        print("No sectioned questions loaded yet.")
        return
    counts = {s: len(quiz.by_section(questions, s)) for s in secs}
    print("\nAvailable sections:")
    for s in secs:
        print(f"   {s:>3}  ({counts[s]} q)")
    raw = _input("Section number (or 'q'): ")
    if not raw.isdigit():
        return
    sel = int(raw)
    qs = quiz.weak_first(quiz.by_section(questions, sel), prog)
    run_session(qs, prog, title=f"Section {sel} practice")


def _choose_mock(questions, prog):
    sets = quiz.sets_available(questions)
    print("\nMock-exam options:")
    print("   0  Weighted 100-question exam (built from the whole library)")
    for i, s in enumerate(sets, 1):
        print(f"   {i}  {s}  ({len(quiz.by_set(questions, s))} q)")
    raw = _input("Choose (or 'q'): ")
    if not raw.isdigit():
        return
    sel = int(raw)
    timed = _input("Timed? minutes (Enter for 300 = your 5-hour allowance, '0' untimed): ")
    minutes = 300 if timed == "" else (None if timed == "0" else int(timed) if timed.isdigit() else 300)
    if sel == 0:
        qs = quiz.weighted_mock(questions, n=100)
        run_session(qs, prog, title="Weighted mock exam", timed_minutes=minutes)
    elif 1 <= sel <= len(sets):
        qs = quiz.shuffled(quiz.by_set(questions, sets[sel - 1]))
        run_session(qs, prog, title=sets[sel - 1], timed_minutes=minutes)


def _show_progress(questions, prog):
    s = progress_mod.summary(prog)
    print("\n" + LINE)
    print(f"Questions attempted: {s['questions_seen']} / {len(questions)} in library")
    print(f"Total attempts:      {s['total_attempts']}")
    print(f"Overall accuracy:    {s['accuracy']*100:.0f}%")
    weak = [q for q in questions if 0 < prog.get(q['id'], {}).get('seen', 0)
            and progress_mod.weakness(prog, q['id']) >= 0.5]
    if weak:
        print(f"Weak questions (≥50% wrong): {len(weak)}  — use 'Weak-area drill' to fix them.")


def _show_library(questions):
    st = store.stats(questions)
    print("\n" + LINE)
    print(f"Library: {st['total']} questions   ({st['needs_review']} need review)")
    print("By block:", {k: v for k, v in sorted(st["by_block"].items(), key=lambda x: str(x[0]))})
    print("By section:", {k: v for k, v in sorted(st["by_section"].items(), key=lambda x: (x[0] is None, x[0]))})


def main():
    questions, warnings = store.load_all()
    print(LINE)
    print(" CEC Coach — BC Construction Electrician (Red Seal / CofQ), CEC 2024")
    print(LINE)
    print(f"Loaded {len(questions)} questions.", end="")
    print(f"  ({len(warnings)} warnings — run option 6 to see)" if warnings else "")
    prog = progress_mod.load()

    menu = {
        "1": "Practice by section",
        "2": "Mock exam (timed, weighted or a full set)",
        "3": "Weak-area drill",
        "4": "My progress",
        "5": "Import new questions (data/incoming/)",
        "6": "Library stats / warnings",
        "0": "Quit",
    }
    while True:
        print("\n" + LINE)
        for k, v in menu.items():
            print(f"  {k}. {v}")
        choice = _input("> ")
        if choice == "1":
            _choose_section(questions, prog)
        elif choice == "2":
            _choose_mock(questions, prog)
        elif choice == "3":
            qs = quiz.weak_first(questions, prog, limit=20)
            run_session(qs, prog, title="Weak-area drill (20)")
        elif choice == "4":
            _show_progress(questions, prog)
        elif choice == "5":
            report = store.import_incoming(apply=True)
            print(f"\nImported {report['added']} new questions, skipped {report['skipped']}.")
            for f in report["files"]:
                print(f"  {f['file']}: +{f['ok']} ok, {len(f['dupes'])} dupes, "
                      f"{len(f['problems'])} problems")
                for p in f["problems"][:10]:
                    print(f"      - {p}")
            questions, warnings = store.load_all()
            print(f"Library now has {len(questions)} questions.")
        elif choice == "6":
            _show_library(questions)
            if warnings:
                print(f"\n{len(warnings)} warning(s):")
                for w in warnings[:40]:
                    print("  -", w)
        elif choice in ("0", "q", "quit"):
            print("Good luck on the exam! 👷⚡")
            break
