# CEC Coach — Project State & "How to Continue"

**Read this first in any new session.** Claude does not remember past sessions;
this repo IS the memory. This file tells you (or a future Claude) exactly where
things stand and how to add the upcoming class sessions and ~1000 more questions.

Branch: `claude/amazing-franklin-ojwgk7`

---

## What this is
An offline practice + teaching app for the **BC Construction Electrician
(Red Seal / Certificate of Qualification) exam**, on **CEC 2024**. The student
has ~6 weeks, gets a **clean (un-tabbed) code book** at the test centre, 100
MC questions, 70% to pass, 5 hours. Run it: `cd cec-coach && python3 main.py`.

## Current inventory (what's already in knowledge)
- **`data/questions/`** — 365 real practice questions, digitised + verified
  twice. ~123 still `needs_review` (depend on exact 2024 table values).
- **`docs/class-notes/`** — full detailed notes for **7 class sessions**
  (session-1 … session-7) + **HIGH-YIELD-EXAM-MAP.md**.
- **`docs/teacher_notes.md`** — condensed teaching from the sessions.
- **`docs/exam-research/`** — 5 web-research reports (experiences, strategies,
  mistakes, resources, YouTube).
- **`docs/reference-materials/`** — voltage-drop 2024 method (from a video),
  precision conductor sizing, teacher images (B-group, A4, A5).
- **`docs/SIX-WEEK-STRATEGY.md`** — the week-by-week study plan.
- **`tools/persian_transcribe_colab.ipynb`** — Colab notebook (Whisper
  large-v3) that transcribes Persian **audio AND video** in batch.

## Pending / TODO
- ~123 `needs_review` questions need exact CEC 2024 table values verified.
- 3 teacher images (A1, A2, A3) couldn't be auto-read — the Drive download tool
  **truncates** their base64. Fix: OCR them via Google Docs, or describe them.
- `code-quick-reference.pdf` is 206 scanned pages (no text) — needs OCR.

---

## ➕ How to add the next ~1000 questions (no code changes)
1. Put each exam as a JSON file (see **`docs/ADD_QUESTIONS.md`** for the exact
   format — id, question, options, answer, answer_key, section, block,
   references, solution_steps) into **`data/incoming/`**.
2. Run **`python3 main.py import`** — it validates, de-dupes by `id`, and moves
   good files into `data/questions/`.
3. Tip: prefix ids per source (e.g. `M01-001` for mock 1) so they stay unique.
   Use `exam_set` to keep each 100-question mock together (full-exam mode).

## ➕ How to add the next class sessions (audio/video)
1. Student records the class → drops the file in the Google Drive folder.
2. Run **`tools/persian_transcribe_colab.ipynb`** in Google Colab (T4 GPU,
   `MEDIA_DIR` = the Drive folder) → transcripts land in a `transcripts/`
   subfolder (handles audio *and* video; skips already-done files).
3. In a Claude session: read the new transcript (it's Persian; large files get
   saved locally after a token-limit error — read 100% in chunks via
   `json.load(...)["fileContent"]`). Then **append** a detailed
   `docs/class-notes/session-N.md` matching the existing style (sections:
   High-yield/traps, Content-in-order with all numbers + worked examples,
   Mnemonics), update `docs/teacher_notes.md`, and refresh
   `docs/class-notes/HIGH-YIELD-EXAM-MAP.md`.
4. Use new session content to clear `needs_review` flags it resolves.

## 🔁 How to resume in a brand-new session
1. `git checkout claude/amazing-franklin-ojwgk7 && git pull`.
2. Read this file + `docs/SIX-WEEK-STRATEGY.md` + `docs/class-notes/HIGH-YIELD-EXAM-MAP.md`.
3. Ask the user what's new (new transcripts? new question files?) and follow the
   "How to add" steps above. Commit + push every change.

## Drive locations (as of last session)
- Code Book + materials folder: `Code Book 2024` (id `1gTYf2bDFH6_tNPbA3eqdI_EbyUiJjQIa`)
- Transcripts subfolder: id `1TRoPb9Rcdd9R8LnnwBAi0DVfAoMHjPM2`
