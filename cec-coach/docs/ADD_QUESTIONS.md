# How to add more questions (e.g. the 10 × 100-question mock exams)

You do **not** need to change any code to add questions. You only add data files.

## 1. The format

Each question is a small JSON object. A file is either a plain list of these
objects, or an object with an `exam_set` name and a `questions` list:

```json
{
  "exam_set": "mock01",
  "questions": [
    {
      "id": "M01-001",
      "section": 8,
      "block": "B",
      "type": "calculation",
      "question": "A single dwelling of 190 m² has a 16 kW range ... Calculate the demand in amps.",
      "options": ["A) 100 A", "B) 127.5 A", "C) 150 A", "D) 200 A"],
      "answer": "B) 127.5 A",
      "answer_key": "B",
      "references": ["Rule 8-200", "Table 14"],
      "solution_steps": ["Basic load ...", "Add range ...", "Convert to amps ..."]
    }
  ]
}
```

### Field reference
| field | required | notes |
|-------|----------|-------|
| `id` | ✅ | unique string. Tip: prefix per set, e.g. `M01-001` |
| `question` | ✅ | the question text |
| `answer` | ✅ | the correct answer in words (always include) |
| `options` | optional | list like `["A) ...", "B) ..."]`. Omit for open-answer |
| `answer_key` | optional | the correct letter `"A".."D"` (only with options → auto-grading) |
| `section` | optional | CEC section number |
| `block` | optional | `"A".."E"` (exam blueprint block — drives mock-exam weighting) |
| `type` | optional | `calculation` \| `lookup` \| `theory` |
| `topic` | optional | short tag, e.g. `ampacity` |
| `references` | optional | list, e.g. `["Rule 8-200", "Table 14"]` |
| `solution_steps` | optional | the step-by-step teaching, one string per step |
| `needs_review` | optional | `true` if answer/edition still needs verifying against CEC 2024 |

## 2. Import it

1. Put your `mock01.json` (etc.) into `data/incoming/`.
2. Run:
   ```
   python main.py import
   ```
3. The importer validates each question, skips duplicates (by `id`), reports any
   problems, and moves clean files into `data/questions/`.

That's it — the new exam immediately shows up under **Mock exam** and counts
toward section practice and weak-area drilling.

## 3. Don't have step-by-step solutions yet?

Add the question with just `answer` + `references`. The coach still works (it
shows the answer and where to find it). Solutions can be filled in later.
