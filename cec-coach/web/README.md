# CEC Coach — Web App

A single-page web app for the BC Construction Electrician (CEC 2024) exam.
Works on phone and computer, from one link.

## What it does
- **Offline quiz bank** (365 verified questions): Mock Exam (100 Q, timed, exam-weighted),
  Practice by Section, Weak Areas, Quick Mix. Each answer shows the fastest
  *keyword → Rule/Table* path. No internet or key needed for this part.
- **AI Tutor** — ask anything in Persian/English, step-by-step answers.
- **Generate Questions** — AI writes fresh exam-style questions on any topic.
- **PDF · Image · Camera** — attach a PDF or photo (or use the phone camera) and ask.

The AI features call the Claude API **directly from your browser** using *your own*
API key. The key is stored only on your device (localStorage) and sent straight to
Anthropic — there is no server.

## Use it
1. Open `index.html` (locally) or the GitHub Pages link.
2. Tap ⚙️, paste your Anthropic API key (from console.anthropic.com), Save.
3. The quiz works even without a key.

## Rebuild the question bundle after adding questions
```
cd cec-coach
python3 build_web.py      # regenerates web/questions.js from data/questions/*.json
```

## Publish the link (GitHub Pages)
A workflow at `.github/workflows/cec-pages.yml` deploys this folder. In the repo:
**Settings → Pages → Build and deployment → Source: GitHub Actions.** It only
uploads `cec-coach/web/` — it never touches the rest of the repo.
