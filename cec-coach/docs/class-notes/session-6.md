# Session 6 — Study Notes
**Course:** BC Construction Electrician Exam Prep — Canadian Electrical Code (CEC) Part I, 2024 (26th Edition, CSA C22.1:24)
**Topic:** Section 12 — Wiring Methods (continued). Underground installations (Table 53), roof decking, the master cable-selection tables (Table 19 + Appendix D / Table D1), parallel conductors, terminations, vertical raceway support, flexible cords (Tables 11A/11B), exposed / inter-building clearances, and the NMD non-metallic-jacketed cable family.

> The instructor's whole pitch this session: **passing the exam is mostly about cable selection and knowing exactly which clearance/cover number applies, and recognizing trick questions.** He repeatedly says "this one is red" = mark it red in your code book = it is a guaranteed exam question. Where a spoken number is garbled in the recording it is flagged **(approx — verify in book)**.

---

## High-yield: important / tested / traps / common mistakes

### Underground cover — Table 53 (Rule 12-012)
- **GUARANTEED EXAM QUESTION (instructor: "red, red, red, this has been an exam question for a long time"):** By how much can you **reduce** the minimum cover of Table 53 when you add **mechanical protection** over the installation? **Answer: 150 mm.** Students who tested 2–3 weeks ago all got this.
- **Trap — "is the number always fixed, or can you reduce it?"** The easy part is reading Table 53. The *real* exam question starts at the reduction. Don't just read the table value; check whether mechanical protection is present.
- **Trap (the famous double-answer sand-bedding question):** Wording matters. The book says screened sand **"75 mm both above AND below"** the conductor. One distractor says "75 mm below, 75 mm above"; another says total. Because the book says **"both above and below," 75 mm applies to each side.** The correct answer is the one that gives **150 mm + cable thickness** (75 above + 75 below + the cable's own diameter). Distractor "C" (just "75 mm above and 75 mm below") is *not wrong but not complete/not the book sentence*, so it loses. Instructor: "Even Canadians get this wrong — it's not a second-language thing."
- **Strategy taught:** When you don't know the right answer, eliminate the **wrong/insufficient** ones. ~7–8 exam questions (especially motor questions) are best solved by elimination. "Not wrong but not sufficient" = treat as wrong.

### Cable selection — Table 19 + Appendix D (Table D1)
- **MOST IMPORTANT job of an electrician (instructor stresses this):** know **what each cable is and where it can be used.** Section 4 taught you how to *size* a conductor; Section 12 / Table 19 tells you *which cable type* is allowed for your conditions.
- **The "three-check" rule for selecting any cable — all three must pass (necessary AND sufficient):** (1) Location moisture — Dry / Damp / Wet; (2) the application — Service / Feeder / Branch / Control / Class 2; (3) the wiring method — Exposed / Concealed / In raceway. If even one fails, you cannot use that cable.
- **Two pieces of info Table 19 does NOT give you → go to Table D1 (Appendix D):** (a) the **voltage rating** of the cable, and (b) **which sizes the cable is manufactured in.** Instructor's complaint: the book never names Table D1 in any rule heading, yet it's heavily tested ("this poor orphan table"). **Mark it / pencil it in next to Table 19.**
- **Key voltage numbers to memorize (from D1):**
  - **NMD90 = 300 V** (house wiring; that's all you need since a house is max 240 V).
  - **AC90 (armoured BX) → up to 2000 V** max (comes in 600 V, 1000 V, up to 2000 V).
  - **TECK90 → full range, up to 5000 V**, and full size range **#14 up to 2000 kcmil**.
- **5000 V project trap:** You CANNOT use AC90 (only to 2000 V) — you must use **TECK90** for 5000 V.
- **Armour boundary for Table 53 cross-reference:** Table 19 pages tell you if a cable is armoured. Page 1 doesn't say "armoured" outright but says "thermoset insulated"; later pages explicitly mark non-metallic vs armoured/metal-sheath. **If asked for cover and not told if armoured: look the named cable up, see if it's armoured, then pick the correct Table 53 column.**
- **Shortcut letters in cable names:** Names starting **"AC"** are usually **A**rmoured **C**able. A **"U"** in the name usually = **U**nderground (can be buried). An **"R"** usually = **R**ubber insulation. (Instructor: "don't take this as a hard formula, but it usually holds.")
- **The two "white-forehead" (most-common) armoured cables to memorize: AC90 (ACWU/"AC90") and TECK90.** AC90 has no jacket version and a jacketed version; TECK is jacketed.

### Cable mis-selection — Dry/Damp/Wet (real worked traps)
- **AC90 location rule:** OK in **Dry** and **Damp**, **NOT in Wet.** Cannot be used Exposed or Concealed — **only in raceway** (per the example cable's row). Common student disaster: installing AC90 on an exterior wall (= Wet) for an A/C condenser — "the worst possible choice."
- **TECK with jacket can be Wet** ("if it has a jacket it can be wet"). TECK is the safe all-purpose pick — slightly more expensive but "no headache."

### Parallel conductors — Rule 12-108
- **RED exam question:** You may **only parallel conductors size 1/0 AWG and larger.** Small conductors are NOT permitted to be paralleled (their tolerance/resistance variation unbalances current → fire).
- **Six conditions for paralleling (a–f), all to keep resistances equal:** must be (1) free of splices / free splice — no joints in the middle; (2) **same size**; (3) **same insulation type** (some insulations trap heat → resistance rises); (4) **same terminations**; (5) **same conductor material** (cannot parallel 400 A aluminum with 400 A copper); (6) **same length** (lengths must match exactly).
- **Why it's dangerous:** unequal length/material/insulation → one leg carries e.g. 30 A while the parallel leg carries 15 A → loss of balance → overheating → fire.

### Terminations — Rule 12-118 (the #10 boundary)
- **Instructor: students often miss this and it IS tested. The #10 AWG boundary is the key.**
- **Size #10 and smaller:** may be connected by **binding-screw terminal** (the screw terminals on receptacles, or studs with lugs).
- **Larger than #10:** must be connected to a **solderless connector/terminal** (solderless conductor connector — a very ordinary lug despite the long name).
- **Exam questions almost always use size #10 itself** (because they know students get the boundary wrong).

### Aluminum terminations — spring washers (conical vs helical)
- Aluminum has a **high coefficient of thermal contraction** → in cold weather it contracts and pulls out of the terminal ("I tightened it, how did it loosen?"). Real-world failure students report.
- **Fix: use a spring washer.** Two types: **conical** and **helical** — both are spring washers (only the shape differs).
- **Trap:** If the question says **conical washer = correct** for aluminum; a flat (non-spring) washer is wrong because it lacks the spring action to take up the contraction. Could be 1–2 exam questions.

### Vertical raceway support — Rule 12-120, Table 21
- Conductors in a **vertical raceway** must be supported at the terminal/top connection and at intervals not exceeding **Table 21** values. (Cable laid on a cable tray / "cinikab"/ladder going up floors must be tied off, e.g. every ~1 m.)
- Table 21 gives max distance vs size (e.g. sizes #14–8 one value; larger sizes shorter). Instructor: the max distances are generous. **New 2024 question bank (~5–6 months old) does test Table 21** — possible exam question.

### Flexible cords — Tables 11A / 11B (and the "extra-hard usage" pattern)
- Table 19 does **NOT** cover equipment wire, portable power cable, flexible cord — those go to **Tables 11A and 11B** (the rule sends you there). 2024 reformatted old Table 11 into 11A/11B (looks like Table 19 now: Dry/Damp/Wet/Exposed/Oil columns), but **technical requirements did not change.**
  - **Table 11A = equipment wire** (page 1).
  - **Table 11B = flexible-cord family** (flexible cord, heater cord, portable power cable, elevator cable, festoon cable). 11B also lists **voltage** and **temperature** directly (no separate D1 needed for these).
- **"Hard usage" vs "Extra-hard usage" — TESTED:**
  - "Not for hard usage" = delicate cord, only where very protected, no mechanical damage possible.
  - "Hard usage" = can tolerate some damage (e.g. a shoulder bumping it).
  - "Extra-hard usage" = can be walked on with a work boot.
- **TWO guaranteed questions whose answer is "extra-hard usage":**
  1. **Section 64 PV / photovoltaic systems** — the cable used between the rooftop solar modules must be **flexible cord, extra-hard usage** (e.g. type **SOW / DW** style). The answer may be hidden in the cable name — recognize which listed cable is extra-hard usage.
  2. **Temporary wiring under construction** (e.g. carpentry, portable cords on site) — must be **extra-hard usage** (boots + wheelbarrow walk over it).
- **Memory hook: if "extra-hard usage" is among the answers, it's almost always the answer.**

### Roof decking — Rule 12-022
- Minor rule but **can be a question.** Roof-decking systems (the corrugated/louvered roof deck seen on house balconies/apartments). If you run cable/raceway through it, **the cable must be concealed — must NOT be run through the visible profile/flutes** (people drill into it to hang flowerpots, etc. and hit the cable).
- **Exam form:** the figure from **Appendix B** with points A, B, C and an arrow; question asks "which is wrong?" The answer relates to the requirement that the run be **concealed** (not in the exposed profile). Picture this figure in your mind. **Mark Appendix B figure red.**

### Exposed wiring & inter-building clearances (Rule 12-300, Rule 12-310, etc.)
- **Title trap — keep the heading in mind:** "Exposed wiring on exterior of buildings and between buildings on the **same premises**." Same premises = one property (e.g. house + detached garage/parking).
- **Span without support:** A run between supports may be up to **4.5 m without an intermediate support**; if longer (e.g. 6 m), you must add a support/pole. Max **unsupported span = 4.5 m** for a single overhead conductor between building/support.
- **Roof clearances (RED / tested):**
  - **2.5 m minimum** above a roof you can **readily walk on** (measured from the roof surface, or from a platform/sloped surface if present — not just floor level).
  - **1 m minimum** above a roof you **cannot walk on** (e.g. steeply sloped/fireguard roof).
  - **Exception:** the AHJ (city) may grant a **deviation** to reduce the 2.5 m, **but not less than 2.0 m.**
  - Conductors must be positioned so a person standing on a **fire escape cannot reach** them.
- **Service-drop support spacing — RED (changed in 2024!):** For neutral-supported / overhead service cables attached to a building (Rule ~12-308 area): the cable shall **not be mounted to any surface**, and clearance not less than **1 m** (and **not less than 50 mm** if it carries an approved mark). Support span must **not exceed 40 mm** in the relevant attached-to-building case. **The "40 mm" was 30 mm (or 35 mm) in the older book — now 40 mm.** Instructor specifically pulled the 2024 book for this. **(approx — the "40 mm span" and "50 mm" figures are read off a noisy recording; verify exact rule/number in the 2024 book.)**

### NMD non-metallic-jacketed cable family — Rules 12-500→12-518 AND the 2024 twins 12-550→12-566
- **Structural trap students must understand:** In CEC 2024 someone **duplicated** the NMD rules. The 12-500 series (12-500…12-518) covers **non-metallic-jacketed cables generally**, and a near-identical 12-550 series (12-550…12-566) was added for **NMD90/NMD-W specifically**. Instructor: "It's basically copy-paste to claim the book changed." **Practical rule he gives: whatever number I tell you in the 500-series, ADD 50 to get the matching 550-series rule, and mark BOTH.** Pairs called out:
  - 12-502 ↔ 12-552 (max voltage)
  - 12-506 ↔ 12-556 (heat clearances)
  - 12-510 ↔ 12-561 (support spacing) *(note: 561, not 560 — read as stated)*
  - 12-514 ↔ 12-564 (concealed/non-concealed mechanical protection)
  - 12-512 ↔ 12-562 (cannot embed these cables)
- **Max voltage:** NMD = **300 V** (per Rule 12-552, matching Table D1). Don't exceed the marked voltage.
- **Heat clearances — RED / tested (instructor: "this came up on my own exam"):** Transfer of heat to the cable shall be minimized by maintaining air clearance:
  - **25 mm** between the cable and a **heating duct / piping**.
  - **50 mm** between the cable and **masonry or concrete chimney** / flue.
  - **150 mm** between the cable and a **chimney/flue cleanout (flue lining/cleanout opening)** — "fluke-lining"; exact meaning uncertain but **always apply 150 mm there.**
- **Support spacing for NMD between boxes (Rule 12-510 ↔ 12-561):**
  - Within **300 mm of every box** (each side of the box) — can be less than 300 mm but not more.
  - At intervals **not more than 1.5 m** along the run.
- **Cannot embed** plain NMD (no armour) in walls/concrete (12-512 ↔ 12-562).
- **Concealed protection — 32 mm rule (RED / common, Rule 12-516 ↔ 12-566):** Where cable runs through **studs or similar members**, the **outer surface of the cable must be at least 32 mm from the edge** of the member. If 32 mm cannot be maintained, use an **approved protector plate** (steel plate, nail-resistant, "approved" — not a homemade can lid). Alternative: a **steel bushing/sleeve** in the hole. (Real-world: drywallers blindly screw/nail and would damage cable; the protector plate stops the screw and warns them something's there.)
- **Non-concealed (attic) protection — Rule ~12-514:** In a non-concealed location like an **attic** where vertical distance between **joists/rafters exceeds 1 m** (walkable), protect the cable with a **running board / guard strip** (two ordinary boards with the cable run between them, or the cable on a board so feet don't land on it).
- **Exposed (surface) wiring — Rule 12-518:** "Like a cook saying 'salt to taste' — nothing specific." Cable used in exposed wiring shall be **adequately protected against mechanical damage**, and where run within **1.5 m of the floor** or otherwise subject to mechanical damage, provide adequate protection. (Just "adequately protected.")

### Concealed vs exposed vs non-concealed — definitions tested
- **Exposed** = surface wiring / visible ("roo-kaar"). **Concealed** = hidden/inside the wall ("too-kaar"), with **no permanent access.** **Non-concealed** = hidden BUT you have permanent access to it (e.g. cable in an **attic** you can enter through a hatch). Attic example: covered (insulation/rock wool) **but accessible** → **non-concealed**, requires the running-board/guard-strip protection where rafter-to-joist clearance > 1 m.

### General exam mindset (instructor coaching)
- Don't panic at an unfamiliar device name. If you learned the *principle*, the device name is irrelevant. Examples: **"mercury thermostat"** — you never studied it by name, but "where do you install a thermostat?" — the location rules are the same; ~3 of the answers are nonsense. **"Hydraulic meter" / "elevator meter"** — it's still a meter; the meter location rules you learned apply regardless of the adjective. **Bonding in a "beauty salon"** — bonding rules don't depend on the room being a salon. Don't let the unfamiliar adjective trip you.
- "If you've learned 6–7 cable names, you'll be fine for normal work" (rule of thumb: knowing ~20% of names lets you handle ~80% of cases). 50–60 cables are listed; you don't need all.

---

## Content taught (in order, full detail)

### 1. Recap & framing
- Resuming **Section 12, around page 109** (student's page numbering differs from instructor's). Section 12 = **wiring methods**: how to run conductors **underground, on the ground, on a wall, inside a wall** — covering all conditions and considerations.
- First topic: **Underground Installation.**

### 2. Underground installation & "minimum cover" (Rule 12-012, Table 53)
- **Direct buried cable** ("Direct Burial Cable") = cable buried directly, with **no conduit/enclosure** around it; OR cable **in a raceway**. **Both cases** "shall be installed to meet the minimum cover requirement of Table 53."
- **Definition of minimum cover (printed under the table):** the distance from the **top surface of your conductor / cable / raceway up to the surface of the ground (grade).** It is NOT the depth of the trench you dug. Example: you dig a trench; for direct burial you put down sand, lay the cable (or raceway) on it; minimum cover = top of cable/raceway → ground surface.
- **What minimum cover depends on:**
  1. **Protection of the cable:** if the cable has **NO protection** (no metal sheath, no armour, not in conduit) → use the **higher** cover column. If it **has protection** (armoured OR in raceway) → use the **lower** column. (Armoured and "in raceway" share the **same numbers**, which is why those two columns are drawn together.)
  2. **Whether vehicles pass over** the location: vehicle traffic → use the higher column; no traffic → the other column.
  3. **Voltage:** the table has columns by voltage. The book now writes **"Extra-low voltage and Low voltage"** for one column and **"High voltage"** for the other. **High voltage is now defined as ≥ 1000 V (was 750 V before)** — they removed the number and just use the words "High voltage" for that column.
- **Simple worked example:** Cable with **no armour**, **no vehicles** over it, voltage 200 V → falls under the "**up to 600 V**" / extra-low+low column. **(approx — instructor's exact spoken cover value was garbled; read the actual Table 53 value: no-mechanical-protection, no-traffic, ≤600 V row. Verify in book.)** This is the "before the trick starts" part.

### 3. Reducing the minimum cover (the exam core) — Table 53 notes
You CAN reduce the Table 53 value if you add extra protection. Three accepted forms of mechanical protection, each then sends you to the **Note** (not back into the table):
- **(a) Treated wooden planking** — pressure-treated wood plank, thickness **at least 38 mm**, and extending **50 mm beyond** the cable on **each side** (left and right). → Note: you may reduce Table 53 by **150 mm** where mechanical protection is placed over the installation.
- **(b) Concrete slab** (precast block) — exactly **50 mm thick**, plus **50 mm extra on each side** beyond the cable edges. → Note: reduce by **150 mm.**
- **(c) Poured concrete** (poured with a wheelbarrow, fills the whole trench width, no 50 mm side extension needed because it fills edge-to-edge) — thickness **at least 50 mm.** → Note: reduce by **150 mm.**
- **Sub-rule structure (2024):** the old single sub-rule for the Table 53 reduction was split — what was Sub-rule 2 became **Sub-rule 3** in your book (a Sub-rule 2 was inserted) but the content didn't change. Sub-rule 3 says: minimum cover may be reduced by **150 mm** where mechanical protection is placed over the installation; "consist of one of the following," and "when in flat form, shall be wide enough to extend at least **50 mm** beyond the cable on each side." Mechanical protection consists of: the **treated planking 38 mm**, the **poured concrete 50 mm**, or the **concrete slab 50 mm**.
- **Common student error (the trap):** Students confuse the numbers — they remember "if poured concrete, you can reduce by 50 mm" (mixing up the 50 mm *thickness* with the *reduction*). **The reduction is always 150 mm; the 50 mm/38 mm are protection dimensions.** Pour 50 mm of concrete → you may reduce **150 mm.** "Don't conflate the two numbers."
- The instructor sends an exam-photo (the dark-burial figure) to the Telegram group and asks the student to delete it after — it clearly shows the geometry so you don't have to memorize raw numbers.

### 4. Direct-burial sand bedding (the double-answer trap)
- Rule text (paraphrased from the book): direct-burial cables **shall be installed so that they run adjacent to each other and do not cross over each other**, with a **layer of screened sand** with **maximum particle size 4.75 mm**, at least **75 mm deep both above and below the conductor.**
- **Why adjacency, no crossing:** unlike cables in conduit (which can overlap), buried cables get **soil pressure** on top; if they cross, the crossing point can be crushed/damaged.
- **The classic trick question (instructor's "treasure" question, from a ~2000-question private bank):** Which is correct? Distractor "C" reads like the book but only says "75 above and 75 below"; the **correct answer D** accounts for **150 mm + cable thickness** (75 + 75 + the cable's own diameter), because the cover/total height must include the cable body too. The discriminating phrase is **"both above and below"** — when you read "both," 75 mm applies to *each* side, totaling 150 mm of sand plus the cable. (Note: instructor finds the question imperfect/"insufficient" but it IS in the bank and the intended answer is D.)
- Method takeaway: **eliminate insufficient/incomplete options;** "not wrong but not sufficient" = treat as wrong (≈7–8 exam Qs, especially motors, solved this way).

### 5. Roof decking — Rule 12-022
- **Roof-decking system** = corrugated/louvered roof deck (seen on house/apartment balconies). To install cable/raceway in it: it must be **concealed** — **do not run through the visible profile**, because occupants drill into the exposed flutes (e.g. to hang plants) and hit the cable.
- **Figure in Appendix B** (mark it red): shows the deck. Exam shows the figure with labels **A / B / C** and asks **which is wrong** (often the answer is the run shown in the exposed profile, "C"). The rule itself says the run must be **concealed**. Memorize the picture.

### 6. Cable selection — Table 19 (Conditions of Use) + Appendix D / Table D1
- **Why many cable types exist:** different installation conditions. Four main (but "not limited to") selection factors the book lists:
  1. **Moisture / wetness** — is the place **Dry, Damp, or Wet**?
  2. **Temperature** where the cable is used.
  3. **Degree of enclosure** — is it fully protected (no rain/oil/snow on it) or out in the environment?
  4. **(Most important) Mechanical protection** — running safely inside a wall vs. exposed where people brush against it.
- **Table 19** = "Conditions of use of insulated conductors and cables." It is **7–8 pages**; lists ~50–60 cable types with their permitted conditions. You only need to recognize ~6–7 names for normal work.
- **Table 19 does NOT cover** equipment wire, portable power cable, flexible cord — those have their **own table** (→ Tables 11A/11B). It DOES cover building wiring, communication cables, etc.
- **Reading the rows:** Page 1 doesn't say "armoured" explicitly but says **"thermoset insulated"** (= non-burning plastic insulation, "nasooz"/fire-resistant). Later pages: page ~4 = **non-metallic** cables (no armour, no metal sheath); page ~5 onward = explicitly **armoured** cables.
- **Communication cables (CMP etc.):** 1–2 questions come from here. Classic: **maximum conductor temperature of CMP** — used to be **60°C** in older sources, but the 2024 book may list a different value (instructor unsure: 60 / 85 / 200 — students report it imprecisely). **(approx — verify CMP max conductor temp in the 2024 book.)** CMP appears on most exams.

### 7. Appendix D / Table D1 — the orphan table
- **Why you need D1 (two things Table 19 lacks):**
  1. **Voltage rating** — e.g. "can I use AC90 on a 5000 V project?" → D1 says AC90 maxes at **2000 V** → NO; use **TECK90** (TECK goes to **5000 V**).
  2. **Available manufactured sizes** — some cables span the whole of Table 2 (#14 → 2000 kcmil); others are limited.
- **Worked size/voltage facts from D1:**
  - **NMD90:** non-metallic-sheathed house wiring (outlets/switches). Cheapest; no armour needed (it's run inside walls, so armour would just waste money). **Voltage = 300 V** (fine: house ≤ 240 V). **Sizes: #14 up to #2** (no need for 2000 kcmil in a house/cafeteria).
  - **AC90:** voltages 600 V, 1000 V, up to **2000 V** max. Armoured. Has both jacketed and non-jacketed versions.
  - **TECK90:** **full range** of Table 2, **#14 → 2000 kcmil**, up to **5000 V.** A single-conductor TECK has only large sizes (6 → 2000); multi-conductor (2+, e.g. 3 or 4) TECK covers the whole table.
  - **ACWU90** (used by builders): armoured AND cheap (cheaper than TECK because lower voltages), comes in small sizes too.
- D1 also marks whether a cable is **armoured** (column), giving a quick way to answer Table 53 cover questions.
- **Decode letters:** "AC" = armoured cable; "U" = underground; "R" = rubber insulation (rubber outer). "Don't treat as a hard formula, but it usually holds — check the spec."
- **Instructor insistence:** D1 is **never named in a rule heading** yet is **heavily tested** (especially voltage questions: "up to what voltage can this cable be used?"). **Pencil "Appendix D / D1" next to Table 19. Mark both tables.**

### 8. The "three-check" cable-selection method — worked example (AC90 / "AR90")
A contractor wants to reuse leftover **AC90** from inventory. Check all three boxes:
1. **Location (Dry/Damp/Wet):** AC90 OK in **Dry** and **Damp**, NOT **Wet.** ✓ for this project (project is dry/damp, no wet areas). Double-check ✓.
   - **Dry** = e.g. inside a house (even a bathroom counts as "dry-family" because moisture is removed by the fan).
   - **Damp** = can be indoor (humid spot like near a pool) or outdoor where direct rain doesn't hit it.
   - **Wet** = indoor very wet OR anywhere outdoors exposed to rain.
2. **Application:** AC90 **cannot be used for Service**, but CAN be used for **Feeder / Branch / Control / Class 2.** Using it for Feeder & Branch → ✓.
3. **Wiring method:** AC90 here is marked **not Exposed, not Concealed — only in Raceway.** If your run needs Exposed/Concealed, AC90 fails. (When a cell has the strike-through/"×", that cable cannot be used that way.)
- **Real-world failure story #1 (A/C unit):** A student installed an A/C, ran **AC90 on the wall outside** to a compressor beside the house. Diagnosis: **worst possible choice** — outside wall = **Wet**, and AC90 can't even be used **Damp** in that configuration. He couldn't even fix it by pulling it through raceway because **AC90 isn't usable in that raceway scenario either**; he had to replace the cable.
- **Real-world story #2 (BX / "B-aks"):** "BX" = market slang for armoured cable. A student bought "BX" for a rooftop A/C, but it turned out to be **AC90** — wrong for a wet/exposed rooftop. If it had been **TECK with a jacket**, jacketed TECK **can be wet** ("damp/wet OK if jacketed"), which is exactly why TECK is used where installers don't want to think hard. **Lesson: cable store sales staff often lack cable-selection knowledge — if you ask for an A/C cable they'll hand you AC90; you must know it's wrong.**
- **AC90 is very common** (stores, libraries — the flexible "shower-hose-looking" conduit-cable seen above restaurant counters) but it must be used in the correct location.

### 9. Parallel conductors — Rule 12-108
- **What "paralleling" means here:** NOT simply running 3 wires from a panel. True paralleling = **two (or more) conductors per phase** acting as one. Per phase you run e.g. 2× A, 2× B, 2× C.
- **Worked example:** A project needed a single **800 kcmil** cable run up 12 floors through one shaft — expensive and hard to source/pull, requiring a hoist. Instructor proposed **two 400 kcmil cables in parallel** instead (code-allowed per Rule 12-108).
- **RED rule:** only **1/0 AWG and larger** may be paralleled. Smaller cables have tolerance/variation you can't control even if lengths match.
- **Conditions a–f (all keep resistances equal):**
  - **Free of splices** (free splice) — no mid-run joints (a joint changes resistance).
  - **Same size.**
  - **Same insulation type** (some insulations retain heat → higher resistance).
  - **Same termination.**
  - **Same conductor material** — cannot mix 400 A aluminum with 400 A copper (different resistance).
  - **Same length** — must match exactly.
- **Danger:** if e.g. lengths differ, one leg may carry **30 A** while the parallel leg carries **15 A** → unbalanced → overheating → **fire.** Paralleling is a precision job; give it only to a careful electrician.

### 10. Terminations — Rule 12-118
- A **termination** = wherever a conductor lands on a terminal (panel breaker lug, receptacle screw, etc.). Never leave a wire dangling; it lands on a terminal.
- Conductor may be **solid or stranded** — doesn't matter for this rule. The boundary is **size #10:**
  - **#10 and smaller:** "shall be permitted to be connected by means of a **binding-screw terminal**" (receptacle-type screw terminals, or studs/lugs).
  - **Larger than #10:** must connect to a **solderless conductor connector/terminal** (ordinary lug; long name, simple device).
- **Exam:** questions usually use **#10 itself** (it's a sub-rule, e.g. "Sub-rule 3"), because students get the boundary wrong.

### 11. Aluminum terminations & spring washers — Tables 5A / 5B
- Aluminum's **high thermal contraction coefficient** → in cold weather it shrinks and **pulls out of the terminal** even when originally tightened (common real-world fault, especially in cold).
- **Solution: spring washer.** Two types in **Tables 5A and 5B**: **conical** and **helical** — both are spring washers; only the visual shape differs. The point is the **spring action** that takes up the contraction. (Made of spring-temper steel.) Likely **1–2 exam questions** distinguishing conical vs helical, both being spring washers.

### 12. Vertical raceway support — Rule 12-120, Table 21
- "It doesn't really fit the section, but it's tested." When a conductor (the Table-2-sized conductor) is placed in a **vertical raceway** (cable brought up multiple floors, laid on a ladder-type tray called "cinikab"), it **shall be supported at the terminal connection and at intervals not exceeding Table 21.**
- **Table 21:** max support distance by size, e.g. **sizes #14–8 → one max distance (m)**; larger conductors → smaller intervals. The maxes are generous (you may support more often). **2024 question bank (~5–6 months old) tests Table 21.**

### 13. Flexible cords — Rule 12-402+, Tables 11A / 11B
- Recall Table 19 excludes equipment wire / portable power cable / flexible cord. The rule (≈12-202 area) sends you to **Tables 11A/11B** (pencil "11A/11B" next to it). Also **Rule 12-406 / 12-102** → "go to Table 11A/11B."
- **2024 reformat:** old **Table 11** → split into **11A** (equipment wire) and **11B** (flexible-cord family). Now formatted like Table 19 (Dry/Damp/Wet/Exposed/Oil columns). **No technical change.** 11A/11B also include **voltage** (e.g. 300 V, 600 V) and **temperature** directly in the table — no separate D1 lookup needed for these.
- **Flexible-cord family (Table 11B):** flexible cord, **heater cord**, equipment wire, **portable power cable**, **elevator cable**, **festoon cable** (the springy coiled cable like a telephone cord, e.g. on garage doors). Names look odd (e.g. **SOW, SOTVW**, etc.).
- **Usage classes:**
  - **"Not for hard usage"** = delicate; only where well-protected, no possibility of mechanical damage.
  - **"Hard usage"** = tolerates limited damage (a shoulder bump).
  - **"Extra-hard usage"** = can be walked on with a safety boot / wheelbarrow over it.
- **Two guaranteed "extra-hard usage" answers:**
  1. **Section 64 PV (photovoltaic / solar):** cable connecting rooftop modules = **flexible cord, extra-hard usage** (e.g. type **SOW/DW**). Answer may be disguised in the cable name.
  2. **Temporary wiring under construction:** portable site cords (carpentry, etc.) = **extra-hard usage** (boots + wheelbarrow).
- **Heuristic: "If 'extra-hard usage' is an answer choice, it's almost always the answer."**

### 14. Exposed wiring on exterior / between buildings — Rule 12-300, 12-308, 12-310
- **Heading:** "Exposed wiring on exterior of buildings, or between buildings on the **same premises**." Same premises = one property (e.g. old houses with a **detached garage** — older garages were built apart due to engine noise/smoke). You may run a cable from a utility pole to the main building's panel, and on to the detached parking/garage (lighting, EV charger, block heater in cold climates of −50/−60).
- **Span without support:** an unsupported single overhead conductor span between building/support may be up to **4.5 m**; longer → add an intermediate support/pole. (e.g. 6 m needs a pole.) **Max unsupported span = 4.5 m.**
- **Reachability:** conductors shall be positioned so a person **standing on a fire escape cannot reach** them.
- **Roof clearances:** **≥ 2.5 m** above a roof a person can **readily walk on** (measured from roof surface, or from a platform if present). **≥ 1 m** above a roof that **cannot be walked on.**
- **Exception (Rule 12-310):** the AHJ may grant a **deviation** permitting **less than 2.5 m but not less than 2.0 m.**
- **Service-cable support (Rule ~12-308, "neutral-supported cables"):** they **shall not be mounted to any surface**; clearance **not less than 1 m**; **not less than 50 mm** if it carries the approved mark. **Support span shall not exceed 40 mm** when attached to the building (if a mobile home / non-fixed structure, etc.). The **40 mm** is the 2024 value — **previously 30 mm or 35 mm** — instructor pulled the new book specifically for this. **(approx — "40 mm span," "1 m," "50 mm" read off noisy audio; confirm exact rule numbers and values in the 2024 book.)**

### 15. NMD non-metallic-jacketed cable — Rules 12-500→12-518 and the 2024 twins 12-550→12-566
- **Definition (Rule 12-500):** non-metallic-jacketed cable = cable with **no armour**, having a **jacket**, non-metallic. Rule 12-500 covers them generally **but states it does NOT apply to NMD90 / NMD-W** — those are covered in the **12-550 series.** In the 2021 book there was only one set; the 2024 book duplicated it. **Rule of thumb: take any 500-series number, add 50 → the 550-series twin. Mark both.**
- **Max voltage (12-502 ↔ 12-552):** don't exceed the marked voltage; NMD = **300 V** (matches D1). Mark 12-552.
- **Heat clearances (12-506 ↔ 12-556) — RED (was on instructor's own exam):** "Transfer of heat to the cable shall be minimized by means of" air clearance:
  - **25 mm** between cable and **heating duct / piping.**
  - **50 mm** between cable and **masonry or concrete chimney** (flue).
  - **150 mm** between cable and **chimney / flue cleanout (flue lining)** — exact term uncertain, but **apply 150 mm.**
- **Support spacing (12-510 ↔ 12-561):** within **300 mm of every box** (each side; may be less, not more), and at intervals **not more than 1.5 m** thereafter.
- **No embedding (12-512 ↔ 12-562):** plain NMD (no armour) **cannot be embedded** in walls/concrete.
- **Three mechanical-protection scenarios (the cable runs through 3 kinds of places):** the book groups them — (1) **non-concealed locations** (Rule 12-514 ↔ 12-564), (2) **concealed installation** (Rule 12-516 ↔ 12-566), (3) **exposed installation** (Rule 12-518). Memorize them as a grouped set; don't study them disjointly. Concealed = inside wall, no permanent access. Exposed = surface. Non-concealed = hidden but accessible (e.g. attic via hatch).
  - **Non-concealed / attic (12-514 ↔ 12-564):** where the **vertical distance between joists and rafters exceeds 1 m** (so a person can walk there, e.g. to service insulation/rock wool), protect the cable from mechanical damage with a **running board / guard strip** (two boards with cable between, or cable on a board so feet don't land on it). Rule text: "...shall be protected from mechanical damage in the form of running board and guard strip... where the vertical distance between the joists/rafters exceeds 1 m."
  - **Concealed (12-516 ↔ 12-566) — 32 mm rule (RED/common):** where cable runs through **studs or similar members**, the **outer surface of the cable shall be kept at least 32 mm from the edge.** If 32 mm can't be met: use an **approved protector plate** (steel, nail-resistant, "approved") or a **steel bushing/sleeve** in the hole. Real-world: drywallers shoot screws/nails blind; the plate stops them and signals a cable is there (story: a driller hit a gas pipe / water pipe behind the wall — that's why we keep clearance and use plates).
  - **Exposed (12-518):** "salt to taste" — cable used exposed shall be **adequately protected against mechanical damage**; where run within **1.5 m of the floor** or subject to mechanical damage, provide adequate protection.

### 16. Wrap-up / exam-mindset coaching (kept because it's study guidance)
- **Don't fear unfamiliar device names** — apply the *principle*: "mercury thermostat" → thermostat location rules; "hydraulic/elevator meter" → meter rules; "bonding in a beauty salon" → bonding rules. The adjective is a distraction.
- **Elimination strategy** for motor/troubleshooting questions ("motor won't start — why?"): many causes are plausible; pick the one that's *uniquely* correct and rule out "not wrong but insufficient" choices.
- **Mark RED everything labeled "red"** — those are repeat exam items, many long-standing.
- **Next sessions:** Section 12 finishes next session (long section). Then Sections 14, 16, 18, 20, 22, 24 in one session (each ~10 min). Sections 26 & 28 take ~3 sessions. ~5 in-person sessions remain; Sections 30+ move to video. Most exam questions come from sections up to 28 (maybe 4–6 questions beyond 28). Near exam time: ~2-hour exam-focused review; up to 30 instruction hours total. **Action item for student: keep doing practice problems from the start; don't fall behind.**

---

## Mnemonics / exact phrasings

- **Three-check cable rule:** Location (Dry/Damp/Wet) ✓ + Application (Service/Feeder/Branch/Control/Class 2) ✓ + Method (Exposed/Concealed/Raceway) ✓ — **all three or it's a no.**
- **"+50 rule" for NMD twins:** 500-series rule number **+ 50** = the matching 2024 550-series rule. Mark both (502↔552, 506↔556, 510↔561, 514↔564, 516↔566, 512↔562).
- **Heat clearances "25 / 50 / 150":** **25 mm** duct/pipe, **50 mm** masonry/concrete chimney, **150 mm** chimney/flue cleanout.
- **NMD support "300 then 1.5":** **300 mm** from each box, then **1.5 m** intervals.
- **"32 from the edge":** outer surface of concealed cable ≥ **32 mm** from the stud edge, else **protector plate**.
- **Cover reduction "= 150":** mechanical protection (38 mm treated plank / 50 mm slab / 50 mm poured concrete) → reduce Table 53 cover by **150 mm**. Don't confuse 150 (reduction) with 38/50 (protection thickness).
- **Sand "75 both, = 150 + cable":** screened sand, max particle **4.75 mm**, **75 mm both above and below** = 150 mm + cable thickness. The word **"both"** is the trap.
- **Roof clearances "2.5 walkable / 1 not / 4.5 span / 2.0 deviation floor":** 2.5 m over walkable roof, 1 m over non-walkable, 4.5 m max unsupported span, deviation may go to 2.0 m but no lower.
- **Voltage ladder:** **NMD90 = 300 V**, **AC90 ≤ 2000 V**, **TECK90 ≤ 5000 V** (and TECK = full size range #14–2000).
- **Parallel "1/0 and up, six-must-match":** only #1/0+, and same size/insulation/termination/material/length + free of splices.
- **Termination "#10 boundary":** ≤#10 → binding-screw terminal; >#10 → solderless connector.
- **Aluminum "spring washer (conical or helical)"** — both are spring washers; flat washer is wrong.
- **Letter decode:** AC = Armoured Cable, U = Underground, R = Rubber. (Heuristic, verify in spec.)
- **"Extra-hard usage = the answer"** whenever it appears (PV Section 64 modules; temporary construction wiring).
- **"Not wrong but not sufficient = wrong."** Eliminate insufficient choices.
- **"If you know the principle, the device name doesn't matter"** (mercury thermostat / hydraulic meter / salon bonding).
- Instructor on the code book: *"This book isn't just for the exam — one day it'll actually serve you on the job."*

> **Noisy-audio flags (verify exact values in the 2024 code book before relying on them):** the "≤600 V / 200 V" worked-example cover value (Table 53); CMP max conductor temperature (60/85/200 °C); the service-cable "40 mm span," "1 m," and "50 mm" figures (Rule ~12-308); and the exact 550-series rule numbers (e.g. 561 vs 560). All concepts above are correct; only these specific numerals were partly garbled in the recording.
