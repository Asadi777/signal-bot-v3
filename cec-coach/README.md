# CEC Coach ⚡

A personal, offline practice + teaching app for the **BC Construction
Electrician (Red Seal / Certificate of Qualification) challenge exam**, based on
the **Canadian Electrical Code 2024 (CSA C22.1:24)**.

The exam is **open book** (the test centre gives you a clean, un-tabbed code
book), 100 multiple-choice questions, 70% to pass, ~4 hours (5 hours with a
language accommodation). So the skills that actually pass it are **fast code
navigation, the right method, and time management** — that's what this app drills.

> Standalone project. Pure Python 3 standard library — **no internet, no API
> key, no extra installs.** It does not touch anything else in this repository.

## Run it

```bash
cd cec-coach
python3 main.py
```

Menu:
1. **Practice by section** (weak questions first)
2. **Mock exam** — a full saved set, or a 100-question exam auto-weighted to the
   real exam blueprint (Block A 11 / B 29 / C 30 / D 20 / E 10). Timed to your
   5-hour allowance by default.
3. **Weak-area drill** — the questions you keep getting wrong
4. **My progress**
5. **Import new questions** (from `data/incoming/`)
6. **Library stats / warnings**

For each question it grades you, then **teaches**: the answer, *where to find it
in the Code* (Rule / Table), and a step-by-step method.

## Add more questions (the 10 × 100-question mock exams)

No code changes needed — drop a JSON file in `data/incoming/` and run
`python3 main.py import`. See **[docs/ADD_QUESTIONS.md](docs/ADD_QUESTIONS.md)**.

## Status (Phase 1)

- ✅ Engine: section practice, weighted/timed mock exams, weak-area drilling,
  progress tracking, importer with validation + de-duplication.
- ✅ Data format + import workflow ready for the incoming 1000 questions.
- 🚧 Question bank: seeded with Section 4 as the worked example. The rest of the
  collected bank (Sections 2, 6, 8, 10, 12, 14, 16, 26, 28 + occupational
  skills, ~360 questions) is being digitised next.
- 🚧 Some answers in the source notes reference the **2021** code; flagged
  `needs_review` to verify against **CEC 2024**.

## Layout

```
cec-coach/
  main.py                 entry point (interactive | import | stats)
  cec_coach/              engine (models, store, quiz, progress, cli)
  data/
    blueprint.json        exam weighting + facts
    questions/*.json      the question bank
    incoming/             drop new question files here, then `import`
    progress.json         your stats (created on first run)
  docs/ADD_QUESTIONS.md   how to add questions
```
