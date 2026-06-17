# YouTube & Video Walkthroughs for the BC Construction Electrician / Red Seal Exam (CEC 2024)

Research notes on video learning resources for candidates preparing for the BC Construction
Electrician Certificate of Qualification / Interprovincial Red Seal exam, written against the
**Canadian Electrical Code, Part I, 26th Edition (2024)** (CSA C22.1:24).

**Date compiled:** 2026-06-17

## How this was researched (and its limits)

- Sources are **WebSearch result snippets** plus general web pages. Direct `WebFetch` of YouTube
  watch/playlist pages returned **HTTP 403** (YouTube blocks unauthenticated scraping), so I could
  **not** read video descriptions, transcripts, view counts, or comments directly.
- Channel names, video titles, and URLs below come from search-engine titles/snippets. They are
  reliable for *what exists and where it lives*, but the *teaching emphasis* attributed to a video is
  inferred from titles/snippets unless otherwise noted. Anything inferred is flagged.
- **Scope:** Prioritized **Canadian / CEC** content. US-NEC channels (Electrician U, Mike Holt, etc.)
  are explicitly labeled and de-prioritized — their calculations (NEC tables, 310.15, 220.x, box-fill
  in cubic inches) do **not** map cleanly to the CEC and can mislead a Canadian candidate.

---

## Top channels & video resources (Canadian / CEC)

### 1. The Electrical Guide (YouTube channel) — best general CEC navigation + theory
- Channel: https://www.youtube.com/@theelectricalguide
  (also https://www.youtube.com/channel/UCsxCpUBpGeCOyq-2MdPoIZA)
- Playlists page: https://m.youtube.com/@theelectricalguide/playlists
- Canadian Electrical Code playlist: https://www.youtube.com/playlist?list=PLQS5IUr7LE38STR4fVWU1hOgJC6BLjtLY
- Key videos:
  - **"How to use The Canadian Electrical Code book — How to navigate the CEC"** —
    https://www.youtube.com/watch?v=XJKlQiILyYo
  - **"Canadian Electrical Code Introduction and Layout of the Code Book"** —
    https://www.youtube.com/watch?v=s-anJiPQ270
- Who runs it: per search snippet, a Canadian college instructor who teaches electrical at a
  Canadian college, holds a **309A construction & maintenance** license plus a Master's degree, and
  owns an electrical contracting company. Covers code, shop, prints, safety, theory, alternative energy.
- **Why it's the top pick:** It directly teaches *how the book is organized* — the single highest-leverage
  skill for an open-book CEC exam. Good starting point before drilling calculations.

### 2. Section-by-section CEC playlist series (channel behind list `PLA9NP6jZaB1...`)
- **CEC Section 8 (Circuit Loading and Demand Factors)** —
  https://www.youtube.com/playlist?list=PLA9NP6jZaB1P7j6XnNEVVtlEW9YQym1Uv
- **CEC Section 10 (Grounding and Bonding)** —
  https://www.youtube.com/playlist?list=PLA9NP6jZaB1NakAk6VrekNYK2m60uYUUG
- These are organized **by Code section**, which mirrors how exam questions are written (a question
  usually lives in one section). Working a section's playlist alongside reading that section is an
  efficient pairing. *Channel handle not confirmed (YouTube 403); identified by playlist-list ID.*

### 3. Motor-calculation video set — Section 28 walkthroughs
- **"CEC Code (Motor Calculations)" playlist** —
  https://www.youtube.com/playlist?list=PLyL0ResmI4Q0kNfmTYo6tiDNMbtqJd4BW
- **"CEC (Section 10: Grounding & Bonding)" playlist** (same series) —
  https://www.youtube.com/playlist?list=PLyL0ResmI4Q3ejvE0SUe1oJYcguBHqOIL
- Individual Section 28 worked examples:
  - "Section 28 CEC: Motor equipment calculations" — https://www.youtube.com/watch?v=LJaxoRvKp54
  - "Section 28 CEC: Motor Banks calculations" — https://www.youtube.com/watch?v=rRACSzO-7Ok
  - "Section 28 CEC: Specialty motor calculations" — https://www.youtube.com/watch?v=CjEx6tokFJU
  - "Individual Motor Calculations" — https://www.youtube.com/watch?v=qRFglZoNrPI
  - "2018 Canadian Electrical Code — Advanced Level Individual Motor Calculation and MORE" —
    https://www.youtube.com/watch?v=cHG-fVX8Ifw (note: 2018 CEC; verify against 2024 tables)
- **Emphasis (from titles/snippets + CEC structure):** motor work centers on **Section 28**, the FLA
  tables (Tables 44/45 etc. via Rule 28-104), and **Appendix B** worked examples for conductor size,
  overcurrent device size, and overload settings. A single motor vs. a **motor bank/group** (Rule
  28-108, 125% of largest + sum of the rest) is a classic exam distinction these videos call out.

### 4. Single-dwelling load-calculation walkthroughs (Section 8)
- **"Calculated Load Single Dwelling Residential, CEC 8-200, Flowchart to make it Easy!"** —
  https://www.youtube.com/watch?v=bZ_Wma8YmHU (posted Nov 2024, so 2024-CEC current)
- Also: "CEC 8 200 Load Calculation" — https://www.youtube.com/watch?v=obNMQbUYQps ·
  "Single Dwelling Service Calculation" — https://www.youtube.com/watch?v=P6m8Y83Ad8A
- **Emphasis:** a **flowchart for Rule 8-200** that walks the "six elements of load." The 8-200(1)(a)
  method from snippets: 5000 W basic for first 90 m² of living area + 1000 W per additional 90 m²
  (or portion), plus space heating, A/C, the electric range, and other large loads with their demand
  factors. Demand factors (Section 8 generally) are the recurring trip-up these videos target.

### 5. Code-navigation strategy (text, pairs well with video) — for exam emphasis
- **xlr8ed Learning — "CEC Exam Questions: Navigation Strategy"** —
  https://xlr8edlearning.ca/cec-exam-navigation-strategy/ (page itself 403'd; emphasis from snippet)
- **Electrical Exams — "How to Use the Canadian Electrical Code Book — 7 Tips"** —
  https://electricalexams.com/free-membership/using-the-canadian-electrical-code-book-free-tips/
- Both stress the same open-book method (see "Method & emphasis" below).

---

## Method & emphasis the resources converge on

These are the concrete techniques and high-value sections the navigation/strategy sources call out
(from search snippets):

- **Learn the book's hierarchy first.** The CEC is general Sections (0–16) amended by supplementary
  Sections (20+). A question's answer almost always lives in one section; knowing which section saves
  the most time on an open-book, timed exam.
- **Tab/flag the heavy-traffic parts** (if exam rules allow tabs): the **Definitions** (Section 0),
  the **ampacity tables**, conductor/cable selection material, and your most-used wiring methods.
- **Keep a bookmark on the Index.** Index-first lookup is faster than paging by memory for most
  candidates.
- **Use a second bookmark/finger on the originating rule** when a rule points to a table or another
  rule, so you can bounce back without losing your place.
- **Appendix B is the open-book "cheat sheet."** Most complex rules have an Appendix B entry with
  plain-language explanation, diagrams, and **worked examples** — notably the **Section 28 motor
  example** (conductor size, OCPD size, overload setting). Candidates are told to sanity-check against
  the appendix example, then return to the rule to confirm the final value.

### High-value tables/sections named across topics
- **Ampacity & derating:** Table 2 (copper, ≤3 conductors in raceway/cable, 30 °C ambient) and
  Table 4 (aluminum); **Table 5C** bundling correction (4–6 cond. = 80%, 7–24 = 70%, 25–42 = 60%) and
  **Table 5A** ambient-temperature correction. Rule 4-004 ties it together. Classic worked example
  from a snippet: 12 AWG at 20 A in a 3-cond. cable drops to **16 A** when run with 5 other
  current-carrying conductors (20 × 0.80).
- **Conduit fill:** 2024 CEC reworked this — **Table 6 (now greatly expanded)** for max number of
  conductors, **Table 8** for max fill %, **Table 9** for conduit internal cross-sectional area, and
  Tables 6A–6K / 9A–9H for the specific conductor/conduit. The video/calculator method: Table 8 → fill
  %, Table 9 → conduit area, Table 6 → conductor selection.
- **Service/feeder & demand:** **Section 8**, especially **Rule 8-200** (single dwelling) and demand
  factors. Distinguish *calculated* vs *demand* vs *demonstrated* loads.
- **Motors:** **Section 28** + FLA tables + Appendix B example; single motor vs motor group.
- **Grounding & bonding:** **Section 10** (10-500…10-506 bonding conductor characteristics/continuity;
  10-602 bonding conductor in each parallel run). 2024 is the 26th edition with a reorganized Sec. 10.
- **Voltage drop:** typical exam form — size copper to limit drop to a given % (e.g., 2% at 120 V,
  10 A, 25 m one-way). Governed by Rule 8-102 / 4-004 voltage-drop allowance.

## How candidates use video + practice together (recommended workflow)

1. **Watch a navigation video first** (The Electrical Guide #1) to internalize the book's layout —
   do this before any calculation drilling.
2. **Per topic:** watch the section/topic walkthrough (e.g., the 8-200 flowchart video or a Section 28
   motor video) **with the actual Code book open**, pausing to find each rule/table the video cites.
3. **Drill practice questions** (xlr8ed Learning, Electrical Exams / electricalexams.com,
   electricalexam.ca, Dakota Prep guides) and **time the lookups** — the bottleneck on the real exam is
   speed of navigation, not arithmetic.
4. **Cross-check every worked answer against Appendix B**, then return to the governing rule.
5. **Re-do calculations from a blank page** until the table sequence (e.g., conduit fill: Table 8 → 9 →
   6) is automatic.

## Supporting non-video resources (for cross-checking worked examples)
- Dakota Prep Red Seal guides (CEC-2024 updated): Motor Conductor Sizing, Conductor Derating & Sizing,
  Conduit Fill — https://www.dakotaprep.com/ca/red-seal-exam-guides/ (text walkthroughs with worked
  numbers; good companions to the videos).
- xlr8ed Learning Red Seal Construction Electrician prep — https://xlr8edlearning.ca/construction-electrician/
- Electrical Exams (run by electricians) — https://electricalexams.com/ and conduit-fill calculator
  https://electricalexams.com/conduit-fill-calculator-canada/
- electricalexam.ca pre-exam course (CEC expert Sam Mallak) — https://electricalexam.ca/
- BCIT Electrical Red Seal Refresher (TELC 0105), based on 2024 CEC —
  https://www.bcit.ca/courses/electrical-red-seal-refresher-telc-0105/
- SkilledTradesBC — Construction Electrician trade page (official BC exam/eligibility authority) —
  https://skilledtradesbc.ca/electrician-construction
- "Guide to the CE Code, Part 1 — A Road Map" article series (electricalindustry.ca, IAEI Magazine) —
  section-by-section plain-language guides, updated for the 26th (2024) edition.

---

## Uncertain / unverified (could not confirm)

- **Channel identities behind two prolific playlist series.** The section-by-section series
  (`PLA9NP6jZaB1...`) and the motor/grounding series (`PLyL0ResmI4Q...`) are clearly Canadian/CEC and
  appear each to be a single channel, but **I could not confirm the channel names/handles** — YouTube
  returned HTTP 403 to automated fetches, so playlist owner pages weren't readable.
- **Video-level teaching emphasis is partly inferred.** Where a video's emphasis is stated above
  (e.g., "six elements of load," "flowchart for 8-200," "motor banks at 125% of largest"), it is drawn
  from search-snippet text or general CEC structure, **not** from watching the video or reading its
  full description/transcript. Verify by watching.
- **No comments/transcripts reviewed.** As requested, note that video comments and full transcripts
  were not accessible to this research.
- **CEC edition currency per video.** Some videos are tagged 2018 CEC (e.g., watch?v=cHG-fVX8Ifw) or
  undated. Method carries over, but **specific table numbers/values changed in 2024** (Section 10
  reorg, Table 6 expansion, conduit-fill table flow). Confirm any cited table number against the 2024
  book before relying on it.
- **"Best" is subjective.** Rankings reflect topical fit and apparent Canadian-CEC focus, not measured
  popularity or pass-rate data (view counts/ratings were not retrievable).
- **xlr8ed / some pages returned 403**; their navigation-strategy points are summarized from search
  snippets, not the full pages.
