# Session 3 — Study Notes (BC Construction Electrician, CEC 2024)

**Coverage:** Finishes **Section 4 — Conductors / Ampacity** (Neutral-Supported / messenger cables Tables D36A & D36B; flexible cords & equipment wire Table 12; portable power cable & DLO cable Table 12; conductor identification by colour; Rule 4-006 terminal-temperature limitation) and **begins Section 6 — Services & Service Equipment** (number of supply/consumer services, service mast, drip loop, heights & clearances, minimum #10 Cu / #8 Al, service-equipment location, meter).

> Note on numbers: this is a transcribed Persian class, so a few spoken figures are noisy. Where the spoken number disagreed with the instructor's own self-correction, the corrected value is used and flagged "(approx/verify)". Always confirm against the actual CEC table/rule in the exam book.

---

## High-yield: important / tested / traps / common mistakes

### Section 4 — Ampacity wrap-up
- **TWO different insulation temperatures in ONE conduit (the "Question 3" exam problem):** When two conductors of different temperature ratings share a raceway, **base your calculation on the LOWER-rated conductor's temperature column.** This is literally an exam question and also the definition they may ask you to write. The instructor stressed this is a real exam item.
  - **Trap:** Do NOT add the two conductors' ampacities together. Each conductor's allowable current is found **separately**; they are independent.
- **Terminal-temperature limitation (Rule 4-006) — the single most emphasized rule of the session.** "Memorize Rule 4-006." The instructor says experienced engineers get this wrong; expect **2–3 exam questions** tied to it.
  - **Trap (the classic error):** When asked to size a conductor for equipment with a marked terminal temperature, students wrongly jump to the 90 °C column. **You must use the marked terminal temperature column**, NOT the conductor's higher insulation rating.
  - You MAY use a higher-rated insulation (e.g. RW90 / "R90" in place of TW75 / "TWU75"), but the **size is still based on the terminal temperature column.** Higher insulation is allowed, but it does not let you go to a smaller size.
- **Rule 4-006, terminal temperature NOT marked (Table 2 header note):** If the equipment terminal temperature is **not marked**, use **60 °C** for equipment rated **100 A or less / #1 AWG and smaller**, and **75 °C** for equipment rated **over 100 A / larger than #1 AWG.**
  - **Trap:** When equipment is exactly at the 100 A boundary, students wrongly pick 75 °C. Because the rule says "more than," at exactly 100 A you use **60 °C.**
- **Neutral-Supported cable tables are REVERSED vs Tables 1–4.** Tables 1–4 list copper first; **Table D36A = ALUMINUM**, **Table D36B = COPPER.** Reason: overhead/messenger conductors are usually aluminum (lighter), so aluminum gets priority. The instructor admitted he repeatedly made this mistake — flag it.
- **Table 12 (flexible cord / equipment wire) has its own correction factors built in / referenced** — read its notes; don't auto-apply Table 5C blindly. (Details below.)
- **Neutral / identified conductor insulation temperature:** The neutral conductor's insulation temperature **shall not be less than** that of the ungrounded conductors. You can't undersize/under-rate the neutral relative to the hot conductors.

### Section 4 — Conductor colour identification (frequently tested as pure colour questions)
- **Grounding & bonding:** green, OR green with one or more yellow stripes.
- **Single-phase, 2-wire:** black, OR white (white only **when identified**, i.e. when a neutral/return is required).
- **Single-phase, 3-wire (e.g. a dryer):** the two hots = **red & black**, plus **white** when identified is required.
- **3-phase (order matters):** **red = Phase A, black = Phase B, blue = Phase C**, plus **white** when a neutral is required.
- **Trap:** 3-phase neutral text says "when a neutral is required," whereas single-phase says "when identified." Know the distinction (identified = current returns through it).
- **Exam style:** they often just give colour combos (e.g. "red-black-blue" vs "blue-black-red") and ask which order is correct.

### Section 6 — Services
- **Rule 6-102 (number of supply services): maximum = ONE.** "Two or more supply services of the same voltage shall not be run to any building." If asked the maximum number of supply services, the answer is **one** — the exceptions don't change the default answer.
  - Exceptions (don't pick these as the "max" answer): separate supply for a **fire pump**; **industrial establishments / large complex structures** (e.g. a big mall with multiple entrances).
- **Rule 6-104 (number of consumer services): max = FOUR**, "unless a deviation is obtained." Apparent contradiction with 6-102 is resolved because consumer services branch from the one supply; large multi-unit buildings need a **deviation** (permission to depart from the Code, with calculations).
- **Minimum service conductor size: #10 Cu / #8 Al** (Rule 6-302 Subrule 4). **Trap:** students think #10 is too small for a house — it's a **minimum**, and not every service is a house (e.g. a small kiosk/store).
- **Service mast must be metal**, minimum **63 (2½") trade-size rigid steel conduit** (Rule 6-112 Subrule 5 & 6).
- **Drip loop required**, leave **minimum 750 mm** of extra conductor at the service head (Rule 4-302 Subrule 3).
- **"Embedded" universal fact:** anywhere the Code says *embedded*, it means **embedded in not less than 50 mm of concrete or masonry.** If you're unsure of an answer and "embedded 50 mm" appears as a choice, it's the right answer ~90 %+ of the time (instructor's exam tip).
- **Service-equipment location — where it is PROHIBITED** (Rule 6-206 / consumer service equipment): NOT in a **coal bin**, **clothes closet**, **bathroom**, NOT where ambient normally exceeds **30 °C**, NOT in **dangerous/hazardous locations**, and NOT where **headroom is less than 2 m.** (Service headroom = 2 m, vs 2 m × 2 m working space required elsewhere for exposed live parts / motor control.)
- **Consumer service conductors shall be located OUTSIDE** as far as practicable; if brought inside, must be **embedded in ≥ 50 mm concrete or masonry** (mechanical protection, because the conductor has no overcurrent protection until it reaches the panel).
- **Connection point:** for **overhead** = at the service head/where utility splices the drip loop; for **underground** = **at the meter** (utility may NOT splice in the run, in the conduit, or underground).

### Exam logistics mentioned (low-yield but useful)
- ~**65–70 %** of exam questions come from the **code book** (some quotes say 30–35 % only — instructor corrects this to ~70 %); ~30–35 % from "knowledge/skill."
- Plan to **skip ~10 hard/novel questions**; you can still pass. Some questions are trivial (e.g. two 3 V batteries in series vs parallel to get 6 V).
- "Next size up" rule: when the exact ampacity isn't in a table, **always go to the next larger size.**

---

## Content taught (in order, full detail)

### 1. Worked Example — "Question 3": two conductors, two temperatures, one conduit
**Problem:** *"What is the maximum ampacity of each conductor"* when placed in a **single rigid steel conduit** containing:
- **3 × size #4 (RW90 / "R90"-type, 90 °C insulation)**, AND
- **3 × size #1 (75 °C insulation)**.
Total = **6 conductors**, two different temperature ratings.

**Key concept first (taught as a definition):** *A conductor rated for a given temperature can carry the current that would heat it to that temperature.* So:
- "Size #4 at 90 °C" means: it can carry the 90 °C-column ampacity and only then reach 90 °C.
- Equivalently: if you pass the 90 °C-column current through it, it heats to 90 °C.

**The constraint:** You can put two different-temperature conductors in one conduit, BUT the heat is shared. The hotter conductor would damage the cooler-rated one. The #1 conductor is only rated 75 °C; if the #4 (90 °C) ran at its full 90 °C current, the shared heat would push the #1 above 75 °C and degrade it. Therefore **everything must be based on the lower temperature (75 °C).**

**Step-by-step (Table 2 — copper in conduit):**
1. **Size #4 conductor:** Its normal 90 °C ampacity (Table 2) ≈ **95 A**. But we are forced to the **75 °C column**, where size #4 = **85 A.** (Instructor first said 65 A reading the wrong row, then self-corrected to **85 A** — use 85 A. The 65 A was a misread; verify in book.)
2. **Size #1 conductor:** It is already 75 °C-rated. Size #1 at 75 °C = **130 A.**
3. **Number-of-conductors correction:** 6 current-carrying conductors → **Table 5C factor = 0.80** (for 4–6 conductors).
4. Apply 0.80 to **each conductor separately** (do NOT add them):
   - #4: 0.80 × 85 = **68 A**
   - #1: 0.80 × 130 = **104 A**

**Answer:** #4 conductor = **68 A**, #1 conductor = **104 A.** (Instructor noted this matches/derives the definition statement on the exam, and may appear as "Question 31" in their question set.)

**Takeaway phrasing:** "Whenever you have two different temperatures, base it on the lower one."

---

### 2. Neutral-Supported (messenger) cables — Tables D36A & D36B (Rule references: Section 4; "C4G 204 / C4G 205" spoken — the appendix/diagram and table area)

**What they are:** Overhead conductors where the **bare neutral doubles as the mechanical support (messenger).** The lineman wraps the bare neutral around the pole insulator/clamp and tensions it (instructor's field story: pulling/tensioning with a "chain block," leaving a "belly"/sag for summer/winter expansion). Because the neutral does **both** the neutral job and the support job, the cable is called **Neutral-Supported (NS).**

**Do NOT use Tables 1–4 for these** — you must use **Table D36A and D36B.**
- **Table D36A = ALUMINUM** ampacity for NS cables.
- **Table D36B = COPPER.**
- **Reversed order vs Tables 1–4** (which are copper-first). Reason: overhead conductors are usually aluminum (lighter). Flag this trap.

**Naming / construction:** These start with **"NS"** (= Neutral Supported). They have a **75 °C** type and a **90 °C** type (two temperature ratings).
- **Duplex** = 2 conductors (one hot + one neutral) — e.g. a small store needing only single-phase 120.
- **Triplex** = 3 conductors (two hots + one neutral) — typical **house** (120/240).
- **Quadruplex (Quadplex)** = 4 conductors (three phases + one neutral) — for a **3-phase** run.

**How to use (very easy — ~20 % of exam is just table lookups):** They tell you it's an NS cable, give the system (e.g. 120/240 = "3-wire / triplex"), give the load (e.g. 200 A), and you read the size directly. Example: NS75 cable, 200 A, 120/240 → pick the size from D36A (aluminum).

**Correction factors for D36A/B (in Note 2):**
- **Ambient temperature** correction is required: for **30 °C → ×1.0**, **35 °C → ×0.94**, **40 °C → ×0.88** (spoken "9400" and "8800" = 0.94 and 0.88). "Respectively" multiply for each ambient.
  - Example: a 210 A value listed is for 30 °C (×1.0 already); for 35 °C multiply by 0.94, for 40 °C by 0.88.
- **Number-of-conductors correction is NOT separately applied** — the table already did it. The table itself gives, e.g., size #6 at **80 A** for 2–3 conductors and **70 A** for 4 conductors, so you do not go back to a separate count-correction table. (Mark **Subrule 5** — it can be a question; "Rule 4-204 Subrule 5" spoken.)

---

### 3. How much of the exam comes from the book (mid-class aside)
- Roughly **65–75 % (~70 %)** of questions come straight from the code book; the rest (~30–35 %, ~30 questions) from "knowledge/skill" material already sent in the group. Of those, ~10 are trivial; ~10 are genuinely hard and can be skipped.
- Anecdote: a student who failed at 68 % missed a **bonding** question that was in **Table 16, Section 10.** Lesson: know the book well enough to "have 65 in your pocket" before the skill/knowledge questions.

---

### 4. Terminal-Temperature Limitation — Rule 4-006 (heavily emphasized)

**Setup question:** "I need to run a conductor rated **25 A** to a transformer/equipment. What size?"
- Table 2 (copper in conduit), depends on the column:
  - **60 °C column → size #10**
  - **75 °C column → size #10**
  - **90 °C column → size #14**
- So the answer **depends on temperature.** Which column? That's what Rule 4-006 settles.

**The physical rule:** Equipment has two terminals (two screws). The rule: **the conductor must not overheat the equipment, and the equipment must not overheat the conductor.** So you must respect the **marked terminal temperature** of the equipment.

**Worked example A (terminal temp marked = 75 °C, using RW90/aluminum or copper "R90"):**
- You WANT to use 90 °C insulation (e.g. instead of TWU75). Allowed? **Yes** — higher insulation is fine.
- BUT if you sized off the 90 °C column you'd get **size #14**, which permits the conductor to reach 90 °C — and that would push the **75 °C terminal** above its limit. **Not allowed.**
- **Correct:** size off the **75 °C terminal column** → keep **size #12** (not #14). You may still use 90 °C-rated insulation, but the size stays based on 75 °C so neither the cable nor the terminal exceeds 75 °C.
- **Exam trap:** "Equipment terminal = 75 °C, I want to use R90 cable, what size?" — students who didn't take this class go to the 90 °C column. **Wrong.** Use the terminal temperature (the lower of the two), consistent with the "use the lower temperature" rule.

**Worked example B (transformer needing a ~69 A / "60 A-class" conductor, 75 °C terminals, want copper R90):**
- Want to use 90 °C cable → would go to Table 2. Use the **75 °C column**, NOT the 90 °C column.
- "I don't care about the 90 °C rating — someone may want higher insulation (90, 100, even 200 °C); fine — but the **basis for the size is 75 °C.**"

**If terminal temperature is NOT marked** → apply **Table 2 header note** (the rule on the conductor table): read from the start of the line *"where the maximum terminal temperature of equipment is not marked"*:
- Use **60 °C** if equipment is **100 A or less** (≤ #1 AWG).
- Use **75 °C** if equipment is **over 100 A** (> #1 AWG).
- **Trap at exactly 100 A:** because it says "more than," at the boundary use **60 °C**. (Instructor: a student read it three times with his son and still chose 75 — wrong.)
- In example B with terminals unmarked, the basis becomes **60 °C**, and the aluminum-vs-90 °C distraction is irrelevant — the 90 °C/aluminum column "is never relevant" once terminal temperature governs.

**Memorize:** "Equipment is marked with maximum terminal temperature → minimum conductor size is selected based on the ampacity at that terminal temperature (per the Table 1/2 columns), then apply any correction factors." Expect 2–3 exam questions on this.

---

### 5. Flexible cords & Equipment wire — Table 12

**Definitions:**
- **Flexible cord** = ordinary very-flexible wire/cable. Examples: the trailing cable feeding an elevator car (moves up/down); Christmas-tree light string.
- **Equipment wire** = same family (a subset of flexible cord) — the wire **attached to a piece of equipment** (the cord hanging off a fridge, TV, or motor that you plug in).

**Table 12 covers BOTH** (one part for flexible cord, one part for equipment wire) — mark Table 12 for each. No separate marking needed; figures in the table are **ampacities in amps.**

**Examples / notes:**
- Christmas-tree light pulling **2 A** → **size #20** (Table 12 has sizes #14 down to **#27**).
- **Size #27** carries only **0.5 A** — used for **tinsel cord** (very thin decorative string lights, like LEDs blinking on/off).
- Recall Tables 1–4 smallest size was #14; Table 12 goes much smaller (#20, #27).
- **Next-size-up reminder:** when the needed ampacity isn't listed, go to the next larger size (ties back to the earlier "25 A not listed → took 35" idea).

**Correction factors (given within Table 12 itself):**
- **2–3 conductors:** use Table 12 value as-is (100 %).
- **4–6 conductors:** **80 %** of Table 12.
- **7–24 conductors:** a further reduced percentage — these match **Table 5C** percentages (the table just restates them instead of saying "if more than 3 conductors, use Table 5C").

---

### 6. Neutral / Identified conductor — concept and Rule

**Why "neutral":** With two hots and one shared neutral, if the two hot currents are equal, they cancel and the neutral carries ~0 ("neutral = stays out of it"). Realistically a small imbalance returns: e.g. one hot draws 4 A, the other 3 A → **1 A returns on the neutral.** Because little current flows, you can sometimes rate the neutral differently — but you should **NOT** rate the neutral insulation lower than the hots (no benefit; you'd just have to upsize it to protect it). Rule wording: *"The insulation of the neutral conductor shall have a temperature rating not less than that of the ungrounded conductors."* ("Ungrounded" = the hot conductors; the system neutral is the conductor that is grounded.)

**Identified vs neutral (terminology):**
- If you have **2 hots + 1 neutral** and the hots balance → neutral ≈ 0 → it's a true **neutral** ("the housewife who stays home").
- If you have **1 hot + 1 return** carrying the **full** current (e.g. 10 A out → 10 A back), that return is **NOT** a neutral — it's an **identified conductor** (it works just as hard). Casual jobsite talk calls everything "neutral," but technically distinguish them.
- This feeds **conductor identification by colour** (Section 4 — Identification of insulated conductors), covered in the colour list at the top.

---

### 7. Portable power cable & DLO cable — Table 12

**Portable power cable:**
- **"Portable"** = movable (like a trailing/dragged cable). **"Power"** = higher voltage / power circuit (not just any lighting load).
- Field story: feeding a **generator → portable power cable → temporary supply** to a water-treatment plant section during a 2-day shutdown.
- **Ampacity from Table 12.** The table lists by number of insulated conductors (1, 2, 3, 4, 5, 6) and by voltage (e.g. a **2000 V** column). Look up size for the given current/voltage. Simple lookup. Rare on exam but know it.
- Example: a 3-conductor portable power cable, 2000 V column, ~? A → read the size (just answer at the level of the practice questions sent).

**DLO cable:**
- Name "**DLO**" (Diesel Locomotive — originally for diesel locomotives). A flexible, good-quality, sectioned cable; especially good for **cable tray** wiring. Cheap and useful; instructor uses it in trays.
- **Ampacity from Table 12** — it has a **size column** and an **ampacity column.** Simple lookup, rarely tested but good to know.

**→ End of Section 4.**

---

### 8. Section 6 — Services & Service Equipment (begins)

**Definitions:**
- **Service** = the part where power enters the building (the conductors the utility, e.g. BC Hydro, brings in).
- **Service equipment** = the gear in that path (meter, main breaker/panel, etc.).
- This section is taught with photos because dimensions/clearances are meaningless without them.

**Rule 6-102 — Number of supply services permitted:**
- Default **maximum = ONE**: *"Two or more supply services of the same voltage shall not be run to any building."*
- Exceptions: separate supply for a **fire pump** (NOT for houses — for high-rise apartments/malls; fire-pump panel is mounted **upside-down** in a safe place so it can be shut off fast and is unaffected by a building fire); and **industrial establishments / large complex structures** (e.g. a large mall with multiple entrances may take service at 2–3 points).
- Exam answer for "maximum permitted" = **one** (ignore the exceptions for that question).

**Rule 6-104 — Number of consumer services permitted:**
- *"The number of consumer services shall not exceed four"* — **max = 4**, **unless a deviation is obtained.**
- **Deviation** = official permission (from the AHJ/municipality, with calculations) to depart from the Code; needed for buildings with many units (10, 20, 30…). The Code itself authorizes the deviation here.

**Rule 6-112 — Support / attachment of overhead supply or consumer service** (taught with photos):

Two ways power enters:
- **Overhead service installation** (aerial wires) — common in **older areas** (undergrounding used to be costly; less urban conduit/duct existed). Vulnerable to storms (instructor's outage anecdotes).
- **Underground service installation** — common in **newer areas** (~last 5–20 yrs), much cheaper to maintain.

**Connection point (where the utility hands off power):**
- **Overhead:** at the service head; you (owner/electrician) bring the conductor up through the conduit and **leave it hanging in a loop**; after inspection passes, the utility splices its cable to yours at the connection point.
- **Underground:** connection point is **at the meter**; utility may **not** splice mid-run, in the conduit, or underground.

**Drip loop:**
- The "U"-shaped loop at the service head, made for water to **drip off** (so rain/snow/melt doesn't track into the building and rot/damage the structure). Even waterproof heads degrade over years.
- You must **leave extra conductor** to form it. **Minimum length 750 mm** of extra conductor at the head (Rule 4-302 Subrule 3), "complete with drip loop."

**Service mast (the conduit sticking up above the roof):**
- The upper conduit that exits the building is the **service mast.** It runs parallel down into the meter.
- **Must be metal.** **Minimum 63 (2½") trade size**, rigid steel conduit (Rule 6-112 Subrule 6).
- **"63" is a trade/thread size, NOT 63 mm** (≈ 2½"). Actual measured size differs (e.g. rigid metal conduit ~62.5; flexible ~60.5). Whenever the class says boxes/conduit are "63 trade size," the real dimension is not literally 63.

---

### 9. Dimensions / clearances (from the photos) — memorize

- **Service mast height to roof / vertical clearance:** **not less than 915 mm** (clearance between roof and supply-service attachment) — but this may be reduced to **600 mm** over the **drip-loop area** (Rule 6-112). The drip loop itself can hang down up to **600 mm** (but **not less than 600 mm** per the spoken correction — verify direction in book).
- **Guying:** if the mast/conduit length above support is **more than 1.5 m**, it **requires a guy wire** (the extra support wire run behind it) (Rule 6-112 Subrule referenced; Appendix B confirms guy/support projection).
- **Photo-3 dimensions (point of attachment of the service head):** Per Rule 6-116 Subrule(s):
  - **Horizontal** distance from the attachment/messenger point to the head: **maximum 600 mm.**
  - The **point of emergence** of the conductor from the **consumer service head** must be **between 150 mm and 300 mm above** (min 150, max 300) the attachment, AND **maximum 600 mm horizontally** from the support/attachment of the overhead service conductor or cable.
- **Extra conductor at the meter:** when cutting/landing conductors at the meter, leave at least **450 mm** of extra (per Rule 4-302 Subrule 1(c): service conductor not less than **450 mm** spare).
- **Conduit (down to meter) minimum size:** must be **not less than size 21 (¾")** trade size — it carries cable weight (this is a minimum; bigger is fine; don't undersize for a kiosk just because the load is small).
- **Mounting bolt/structure for the mast on a wood wall:** must mount to a backing structure (can't bolt to drywall). The wood backing/ladder must be **not less than 38 × 38 mm** (≥ 38 mm in any dimension) (Rule 6-112 Subrule 7).
- **Service conductor height above ground (overhead, to the building):** ranges **3.5 m to 9 m** depending on what's below:
  - **Pedestrian-only:** minimum **3.5 m.**
  - **Over a residential driveway / where vehicles pass:** **~5 m** (spoken; verify — instructor said "5").
  - **Street / highway:** higher (**≥ 5 m**, larger values for roads).
  - **Maximum 9 m** — because the utility's bucket/lift truck can only reach ~9 m to service it.

**Section 6 Rule 6-200/6-206 — Window/door/balcony clearance (Photo 7):**
- Service conductors near a **window, door, balcony, porch, or terrace** must keep **1 m clearance** on **left, right, and below** (and beside). Two reasons: (1) someone opening the window can't reach it; (2) firefighters breaking the glass to enter aren't endangered by the connections. (Above is generally not where it's mounted; spoken "1 m" applies to side and below.)

**Rule 6-206 — Consumer service equipment LOCATION (prohibited places):**
- Shall **NOT** be located in: a **coal bin** (coal dust is conductive/can arc), a **clothes closet**, a **bathroom**, a location where ambient normally exceeds **30 °C**, a **dangerous/hazardous location** (e.g. near an oil/ship-fuel installation), or where **headroom is less than 2 m.**
- Working space: earlier (Section 2) you learned **2 m × 2 m** for motor control / **exposed live parts**; for **service** equipment the requirement is **2 m headroom** (because service equipment isn't exposed live parts — slightly relaxed). Know the difference but it's minor.
- **Locate as close as possible** to the point of entry: the unprotected service conductors (no fuse/breaker yet) shouldn't be dragged across the building. Keep the panel "as close as possible" to where the conductors enter (e.g. right behind the wall in the basement). If a permitted spot isn't available, you may move it but the inspector then requires a disconnect switch there (so a fault/fire can be cut off).
- The **switch (disconnect)** is **often eliminated** when a cable runs straight to the main panel, but in some cases it's required (e.g. when service equipment had to be relocated).

**Rule 6-206 — Consumer service CONDUCTOR location:**
- Whether raceway or cable, locate **outside the building** as far as practicable (no overcurrent protection yet).
- If brought inside: must be **embedded in not less than 50 mm of concrete or masonry** (mechanical protection in lieu of electrical protection).
- **Universal "embedded = 50 mm" fact:** the Code uses "embedded" in ~12 places, and **every time it means 50 mm** of concrete/masonry. Exam tip: if unsure and "embedded 50 mm" is an option, it's right ~90 %+ of the time. (Example: an aluminum consumer-service-conductor question where the trick was that, if not outside, it must be embedded 50 mm — the aluminum anti-oxidant/waterproof details were a distraction.)

**Minimum conductor size (Rule 6-302 Subrule 4):**
- Consumer service conductor **shall not be less than #10 copper / #8 aluminum.** It's a **minimum**; not necessarily for a house (could be a small store). The utility, not you, often installs it, but you must know the minimum.
- The smaller conduit referenced earlier: **nominal size 21** (¾").

**Meter / meter equipment:**
- The **meter is part of the service equipment**; same location rules apply (restated in Rule 6-206: not in coal bin, closet, bathroom, high-ambient, hazardous, low-headroom — "and similar," meaning anything comparable).
- Leave **450 mm** extra conductor for the meter (Rule 4-302 Subrule 1(c): service/spare conductor not less than 450 mm).
- The **750 mm** minimum-length drip-loop conductor lives in the **wiring methods** rule **4-302 Subrule 3, line 3** ("minimum length 750 mm").

**→ End of Section 6. Next session starts at Section 8.**

---

## Mnemonics / exact phrasings

- **Two temperatures, one conduit:** *"Whenever you have two different temperatures, base it on the LOWER one."* (Find each conductor's ampacity **separately** — never add them.)
- **Rule 4-006 / terminal temperature:** *"Equipment marked with a maximum terminal temperature → minimum conductor size selected on the basis of the ampacity at that terminal temperature."* You may use higher-rated insulation, but **the size is still based on the terminal temperature.** (Memorize Rule 4-006 verbatim — 2–3 questions.)
- **Unmarked terminal temperature:** *"60 °C if 100 A or less; 75 °C if over 100 A."* At exactly 100 A → **60 °C** ("more than" excludes the boundary).
- **NS tables order:** *"D36A is ALUMINUM, D36B is COPPER — backwards from Tables 1–4."* (Aluminum first because overhead messengers are usually aluminum.)
- **Cord families:** *"Flexible cord = anything flexible (elevator cable, Christmas lights); equipment wire = the cord hanging off equipment — both in Table 12."*
- **Neutral vs identified:** *"Neutral = the one that stays home (currents cancel, ~0 A). If the full current returns on it, it's not a neutral — it's an identified conductor."*
- **Colours:** Ground = green / green-yellow. 1-phase 3-wire (dryer) = red + black (+ white when identified). 3-phase = **red-A, black-B, blue-C** (+ white when neutral required).
- **"Embedded" = 50 mm always.** *"Every place the Code says embedded, it means 50 mm of concrete or masonry — if you're stuck and that's an answer, it's right ~90 % of the time."*
- **Supply vs consumer services:** *"Supply services: maximum ONE. Consumer services: maximum FOUR (unless deviation)."*
- **Service basics:** *"Service mast = metal, minimum 63 trade size. Drip loop = leave 750 mm. Meter = leave 450 mm. Minimum service conductor = #10 Cu / #8 Al."*
- **"63 is a trade/thread size, not 63 mm"** (≈ 2½").
- **Heights:** *"3.5 m pedestrian, up to 9 m max (utility lift truck reach); 1 m clearance around windows/doors/balconies."*
- **Next size up:** *"If the exact value isn't in the table, always take the next larger size."*
- **Study advice (instructor):** *"Book your exam 2 weeks to a maximum of 1 month after the course — long enough to drill practice questions, but the material is perishable, so never more than a month."*
