# Session 2-2 — Study Notes (BC Construction Electrician, CEC 2024)

> Scope of this class: finishing **Section 2** (working space / enclosures — Tables 56 & 65, gas-clearance Rule 2-308, Appendix B), then starting **Section 4** (conductor ampacity — selecting Tables 1–4, correction factors from Table 5A ambient and Table 5C grouping, plus two worked exercises).
> The instructor speaks Persian and pronounces the English code terms phonetically; those terms have been interpreted/normalized into proper CEC English below. Where a spoken number is garbled or "approximate," it is flagged **(approx — verify in book)**.

---

## High-yield: important / tested / traps / common mistakes

- **TRAP — 2 m vs. 2 m, two different rules (Section 2 vs. Section 6).** The "minimum height ≥ 2 m" requirement you meet for **motor control / exposed live parts** in Section 2 is NOT the same as the "service equipment height" rule in **Section 6**. The instructor stressed: *do not mix them up.* One is for motor control where live parts are exposed; the other is for the service. The exam deliberately separates these — read which one the question is about.

- **Breaker faces being exposed is acceptable for a residential panel.** A homeowner brushing/dusting the panel face, exposing the breaker handles, is not a hazard — only the breaker handles are outside; the cover protects the live parts behind. (Context for why the working-space/cover rules are written the way they are.)

- **Transformer working space — the 50 kVA threshold (TESTED).** Working space applies to transformers **rated GREATER than 50 kVA**. At exactly 50 kVA it is NOT mandatory; at 51 kVA (anything *above* 50) it IS required. **Minimum horizontal working space = 1 m.** Trap: "exactly 50 kVA" answer = no requirement; "51 kVA" = requirement applies.
  - The 1 m clearance only needs to be on the side where the conductors **enter and exit** the transformer (the live/wiring side). The other sides can be tight to the wall. Example: a transformer fed by 3 conductors + neutral on the supply side and 3 conductors leaving — you provide the 1 m only on the wiring/termination side.

- **Gas clearance — Rule 2-308 lives in the book AND in Appendix B (TESTED, and easy to miss).** The clearance numbers for electrical equipment near a **gas relief/vent (regulator)** are NOT printed in the body of Rule 2-308 — the body shows nothing. You must go to **Appendix B** for the actual distances. *Mark/flag Rule 2-308 in Appendix B in advance* so you can find it under exam time.
  - **Natural gas (lighter than air, rises):** minimum **1 m** clearance from electrical equipment to the gas relief/vent.
  - **Propane (LP, heavier than air, pools at ground level — more dangerous):** larger clearance, **3 m**.
  - There is also a **0.3 m** value (per Appendix B) that applies **if a specific code/listing mark is present on the regulator** — i.e., if the regulator carries the qualifying listing, the distance may be reduced to 0.3 m. Know all three: **1 m / 3 m / 0.3 m**.
  - Why electrical equipment matters here: switches, thermostats, contactors, A/C units, etc. are a **source of ignition** (they spark). "Ignition" = the spark (same word as a car's ignition/starter spark). Ordinary switches all spark.

- **Enclosure type selection — only TWO tables in Section 2 (TESTED).** Section 2 has exactly **two tables: Table 56 and Table 65.**
  - **Table 56 = working space.**
  - **Table 65 = enclosures.**
  - Mnemonic the instructor gave: *each section, write at the top how many tables it has.* Section 2 = "2 tables: 56 and 65." Easy to memorize.

- **Reading Table 65 (enclosure selection) — match enclosure TYPE to the environmental condition (TESTED, "hardest question they can ask").**
  - The left column of Table 65 = the **environmental condition** where you'll install the enclosure (e.g., dripping water, splashing water, corrosive, etc.).
  - **Dripping water** = water drips on it from above (e.g., a greenhouse where condensation/spray falls). You cannot use Type 1 here.
  - **Splashing water** (condition group incl. types where water is sprayed at it — e.g., a car wash) requires specific types.
  - **Corrosive** (e.g., a car wash where corrosive shampoo/chemicals are present): you need a corrosion-resistant type. Worked example given: a panel that is **Type 4X** and **6P** is needed for splashing + corrosive (4X = corrosion-resistant; 6P = submersion-rated).
  - **Enclosures fall into 3 indoor/outdoor categories:** (1) **indoor only**, (2) **outdoor only**, (3) **indoor + outdoor** (outdoor-rated ones can also be used indoors). Exam trap: they give you 4 enclosure types and ask which can go outdoors / which is indoor-only — *you should know all of them.* They may invert the question ("which of these is indoor-only?").
  - **6P / submersible types**: rated to be lowered (e.g., on a chain) into water and pulled back out — fully submersion-capable.

- **2-400 (enclosures).** An "enclosure" is any housing for electrical equipment — can be as small as the palm of your hand or as large as a big cabinet/panel. A panel is an enclosure; a motor frame = "motor enclosure"; a transformer body = transformer enclosure. **What's inside and how much current it carries does NOT matter** for enclosure selection — what matters is **WHERE (the environment) you install it.** Enclosures are graded/classified (specified) by Table 65.

- **Section 4 mindset (instructor's framing).** Section 4 is the **most conceptual ("mafhumi") section** of the book, and a **"mother" section** — almost everything ties back to conductor size and ampacity. It is **not hard** — instructor rates the academic level as "elementary, around grade 5"; it's all about distances/relationships. *Don't panic if it feels hard at first* — even the instructor needed several attempts the first time. Learn ONE table thoroughly and the rest follow the same pattern.

- **Section 4 core skill (TESTED both directions).** Section 4 gives you the relationship between **current (ampacity) and conductor size**:
  - Given **current** (e.g., a 20 A motor) → find the required **conductor size**.
  - Given **conductor size** → find its **ampacity**.
  - It's a table lookup that works both ways.

- **Which ampacity table to use — the selection procedure (HIGH-YIELD, tested).** Decide using two questions: **(A) installation method** — Free Air vs. in a raceway/cable ("Raceway or Cable"); and **(B) conductor material** — Copper vs. Aluminum.
  - **Free Air + Copper → Table 1**
  - **Free Air + Aluminum → Table 3**
  - **Raceway or Cable + Copper → Table 2**
  - **Raceway or Cable + Aluminum → Table 4**
  - Mnemonic: **Copper = Tables 1 & 2; Aluminum = Tables 3 & 4.** (Free Air = odd-paired 1/3; Raceway-or-Cable = 2/4.)
  - **Trap:** In real residential/construction work ~90% is Raceway/Cable and ~10% Free Air, and it's mostly copper — so Tables 2 and 4 dominate daily work. **But the EXAM will deliberately ask the less-common combinations** (e.g., Free-Air Aluminum → Table 3) precisely because you use them least. Learn all four.

- **"Free Air" definition (TRAP — easy to misjudge).** Free Air = a conductor that is supported only at points (poles/standoffs) but along its run is **not touching anything 360° around** — like overhead street wires. If a wire runs **through a conduit / along a wall / into a motor**, it is NOT free air → it's the "Raceway or Cable" case. Learn what Free Air is, and everything else is the other case.

- **Tables 1 & 3 are rated for a SINGLE conductor (TESTED).** Tables 1 and 3 (Free Air) give ampacity for a **single conductor**, because free-air runs are normally single conductors. If you place **more than 3 conductors** together, mutual heating reduces ampacity and you must apply a **derating / correction factor**. (Iran's code had no such rule; the Canadian code does.)

- **"Not more than three" rule (TESTED phrasing).** The published table ampacities assume **not more than three** current-carrying conductors. The book wording: *"not more than three insulated conductors."* So for **1, 2, or 3** conductors → **no grouping derate** (factor = 1). The grouping **correction factor applies only ABOVE three** conductors.
  - Example pattern (illustrative numbers): a #14 conductor ≈ **15 A**; with 2 ≈ 24 A... 3 ≈ 23 A... 4 ≈ 20 A — *these are illustrative, not exact; use the actual table.* **(approx — verify in book.)**

- **TWO independent correction factors, both heat-driven (HIGH-YIELD).**
  - **Grouping (number of conductors) → Table 5C.** Heat transferred cable-to-cable.
  - **Ambient temperature → Table 5A.** Heat from the surroundings. All four ampacity tables are based on **30 °C ambient**; above 30 °C apply the Table 5A correction.
  - When both apply, **multiply both factors** onto the base ampacity. **Order does not matter** — "whether Khaje-Ali or Ali-Khaje," multiplication is commutative. Don't let anyone tell you a required multiplication order exists.

- **Table 5A reading trap — round to the NEXT (higher) temperature row, which gives the smaller factor.** If the given ambient (e.g., 36 °C, 38 °C, 39 °C) is between rows, **go to the next-higher row's column**, NOT the lower one — picking the lower temp would overstate capacity. (Instructor: for 36 / 38 / 39 °C you use the 40 °C column, "take the four[ty]".)

- **Grouping correction (Table 5C) for Raceway/Cable tables (Tables 2 & 4).** For **1–3** conductors: factor = **1** (no derate). Then it steps down: e.g., **4–6 → ×0.8**, **7–9 → ×0.7**, and continues decreasing as the count rises (read exact values from Table 5C).

- **Free-Air grouping derate ONLY applies when conductors are CLOSE (the "if/spacing" condition) — major exam trap.** For free-air single conductors, the grouping derate applies **only if** two or more single conductors are spaced **less than 25% of the largest cable diameter** apart.
  - Worked logic: if your largest cable is 4 inches, 25% of 4 = **1 inch**; if the conductors are spaced **less than 1 inch** apart → apply the derate. If spaced ≥ that → no derate.
  - **Trap:** the rule contains the word **"where"** (a conditional). On the exam, **if the question does NOT state the spacing/closeness condition, you do NOT apply the grouping derate** — answer stays at the base ampacity. Don't invent a derate the question didn't trigger.
  - Contrast: conductors in a **raceway or a multi-conductor cable** are *necessarily* bundled/touching, so their mutual heating always counts — derate applies once you exceed three.

---

## Content taught (in order, full detail)

### 1. Working space recap — exposed live parts vs. service (Section 2 vs. Section 6)
- A homeowner brushing the front of a residential panel and exposing the breaker handles is fine; only the breaker handles sit outside the cover — the live parts stay protected.
- **Why a distinction was drawn:** later, in **Section 6 (Services)**, service equipment must be installed where its height is **not less than 2 m** (i.e., the height/mounting rule for service equipment). That rule lives in Section 6.
- The "2 m" type requirement encountered earlier (motor control with **exposed live parts**) is a **different** rule from the Section 6 service rule. **Do not confuse them** — the exam keeps them clearly separated.

### 2. Transformer working space
- **Rule:** working space is required for transformers rated **above 50 kVA** (i.e., > 50; a 51 kVA unit qualifies, exactly 50 does not).
- **Minimum horizontal working space = 1 m.**
- The 1 m is only required on the side where conductors **enter and leave** (the wiring/termination side, where you can be exposed to the wires). Other faces may sit close to the wall.
- Worked illustration: a transformer with **3 phase conductors + neutral** on the source side and **3 conductors** leaving — provide the 1 m clearance on the conductor entry/exit side. The cable can run tight against the wall elsewhere.

### 3. Rule 2-308 — clearance from electrical equipment to gas relief / regulator vents
- **Subject:** distance from electrical equipment to building features that have a **gas relief / vent** (e.g., the gas **regulator** near a house entrance, roughly 1 m off the ground).
- Gas regulators typically have a **small leak** and a **vent**; walking past you can smell gas. Newer regulators leak far less (being replaced), but a leak program still exists.
- An **air conditioner** is used as the example of equipment you might install on a house: some of its electrical parts (switch, thermostat, contactor) are a **source of ignition** — they spark.
- **"Source of ignition" / "ignition"** = a spark. Same word as a car's ignition (the first spark that starts the motor). Ordinary switches all spark; only special switches are spark-free.
- **Clearance values:**
  - **Natural gas: minimum 1 m.**
  - **Propane: minimum 3 m.** (Spoken "3 metr.")
  - **Why propane is worse:** natural gas is **light → rises** when it leaks (like the gas in BBQ cylinders that vents upward); **propane (LP)** is **heavier than air → pools/spreads at ground level**, so it accumulates and is more dangerous.
  - Real-world example: a residential parking garage sign reading "propane-vehicle entrance — keep 1 m clearance and [direction]."
  - **Special reduced value: 0.3 m** — Appendix B defines a condition where, **if a particular code/listing mark is present on the regulator**, the clearance may be reduced to **0.3 m**.
- **Critical point about WHERE the numbers are:** in the **body** of Rule 2-308 there is essentially **nothing written** — no 1 m, no 3 m. The actual distances are in **Appendix B**. Action item the instructor gave the class: **flag/mark Rule 2-308 in your Appendix B** (write a note in the rule body pointing to Appendix B), and locate Rule 2-308 there in advance. (Page location differs by book edition/printing.)

### 4. Rule 2-400 / Table 65 — enclosures
- **Definition of "enclosure":** any housing for electrical equipment. Size-independent — from palm-sized to a large cabinet. A panel = enclosure; a **motor frame = motor enclosure**; a **transformer body = enclosure**.
- **What does NOT matter:** the contents and the current/wattage inside.
- **What DOES matter:** **WHERE** (the environment) you install it.
- Enclosures are **classified/graded (specified) by Table 65.**
- **Section 2 has only two tables:**
  - **Table 56 → working space.**
  - **Table 65 → enclosures.**
- **How to read Table 65:**
  - **Left column = environmental condition** of the install location.
  - **Dripping water** = water dripping from above (greenhouse example — go inspect; condensation/spray drips on it → cannot use **Type 1**; only the types not "struck out" for that condition).
  - **Splashing water** = water sprayed onto it (car-wash example) → only certain types qualify (the "last four" group of types in that row, per instructor).
  - **Corrosive** = corrosive chemicals present (car-wash shampoo example) → need a corrosion-resistant type. **Worked answer for splashing + corrosive: Type 4X and 6P** (4X = corrosion-resistant, 6P = submersible).
- **Three indoor/outdoor categories of enclosure type:**
  1. **Indoor only**
  2. **Outdoor only**
  3. **Indoor + outdoor** (outdoor-rated can also be used indoors)
- **6P / submersible:** can be lowered into water (e.g., on a chain) and pulled back out — full submersion.
- **Exam question styles to expect:** "which of these 4 types can be used outdoors?", "which is indoor-only?", "which can be mounted [in a wet/sprayed location]?", combined splashing+corrosive scenarios. Know all four types both ways.

*(Break taken — ~10 minutes — to stay focused before Section 4.)*

---

### 5. Section 4 — Conductor ampacity (start)

**Framing.** Section 4 is the most **conceptual** ("mafhumi") section and a **"mother" section**: nearly everything downstream needs a conductor **size** and **type**. (Type is covered later, in **Section 12**.) Section 4 is rated by the instructor as low-difficulty ("grade-5 level — distances and relationships"). A short 3–4 minute video was sent to the class (watch the beginning and end together).

**What Section 4 does (both directions):**
- Given **current/ampacity** → find conductor **size** (example: a motor rated **20 A** → find the wire/cable size).
- Given conductor **size** → find its **ampacity**.

**The many tables vs. the four you use daily.**
- There are many ampacity/conductor-size tables, but **Tables 1, 2, 3, 4** are your everyday tables (especially 2 and the cable case).
- There is also a set **D8, D9, D10, D11** — the instructor **does not teach these** (used to, but stopped, ~1 year, because they're **not asked on the exam** and are very specialized).
- **Difference between Tables 1–4 (above-ground) and D8–D11 (underground):**
  - Big cables run **underground (direct buried)** → use **D8 / D9 / D10 / D11.**
  - Conductors **above ground level** → use **Tables 1 / 2 / 3 / 4.**
  - (This "above ground vs. underground" is the dividing line.)

**Table-selection procedure (the core decision tree).**
1. **Installation method:**
   - **Free Air** — conductor runs in open air, not touching anything 360° around it along its length (street/overhead wires on poles). Supported only at points.
   - **Raceway or Cable** ("raceway / cable") — conductor runs **through a conduit, along a wall, or as part of a cable** into equipment (e.g., a wire from a conduit on a wall into a motor). Anything that is **not** Free Air falls here.
2. **Conductor material:** **Copper** or **Aluminum.**
3. **Pick the table:**

| Installation | Copper | Aluminum |
|---|---|---|
| **Free Air** | **Table 1** | **Table 3** |
| **Raceway or Cable** | **Table 2** | **Table 4** |

- **Copper = Tables 1 & 2; Aluminum = Tables 3 & 4.**
- Daily work ≈ **90% Raceway/Cable, 10% Free Air**, mostly **copper** → Tables 2 & 4 dominate. But the **exam will ask the rare combinations** (e.g., Free-Air Aluminum, Table 3). Learn all four.

**Each table's base assumptions (printed on every table):**
- **Tables 1 & 3 (Free Air):** ampacity is for a **single conductor** (free-air runs are normally single).
- All four tables are **based on 30 °C ambient temperature.**
- All four assume **not more than three** current-carrying conductors ("not more than three insulated conductors").

**Why correction factors exist:** heat. Two sources:
- **Heat from cable to adjacent cable** (grouping) → derate when **more than 3** conductors.
- **Heat from the environment** (ambient) → derate when ambient **> 30 °C** (e.g., near a furnace, a hot room).

**The two correction factors:**
- **Grouping / number-of-conductors factor → Table 5C.** Applies **above 3** conductors. (1, 2, or 3 → factor = 1, "not mandatory" / no derate.)
- **Ambient temperature factor → Table 5A.** Applies when ambient **> 30 °C** (e.g., 35 °C, 40 °C).
- **Both can apply at once → multiply both onto the base ampacity. Order is irrelevant (multiplication commutes).**
- Illustrative grouping factors mentioned (NOT exact — read Table 5C): single = full value; 2 → ×0.9; 4 → ×0.8; etc. **(approx — verify in book.)**

**How to physically read a Table 1/2/3/4 row (simple model the instructor gave):**
- Pick conductor **size** (e.g., #14) → read across to the **insulation-temperature column** that matches the conductor's insulation rating:
  - cheap insulation rated **60 °C** → e.g., **15 A**
  - better insulation **75 °C** → e.g., **20 A**
  - good insulation **90 °C** → e.g., **25 A**
  - *(these are the #14 illustrative trio; the cross-section/area is the same — only the insulation temp rating changes the ampacity.)*
- **Industrial vs. construction:** 60/75/90 are all used; **residential/construction work mostly uses 75 °C and 90 °C.**
- **Insulation temperature** (the column) ≠ **ambient temperature** (the room). Don't confuse the two. Inside the table you work with the **insulation temperature** column; if ambient > 30 °C you then go to **Table 5A** to get the ambient factor and multiply it back in.
- **Reading conductor designations:** the letter codes encode the insulation. Example: **"R"** = rubber/plastic (non-burning plastic); a number after it shows the temperature rating. **"TW 75"** → that insulation is rated **75 °C**. So the **name itself often tells you the temperature rating** (e.g., an "R90"-type name implies 90 °C even if the question doesn't restate it).

---

### 6. Worked Exercise — Section 4, Question 1 (single conductor, Free Air, Copper)

**Given:** An **aluminum-sheathed cable**, single **copper conductor**, run in **Free Air**, conductor size **2/0** ("two-aught" = 2/0 = two zeros), insulation/cable rated **90 °C**.

**Step 1 — which table?**
- It's **Free Air** → Table 1 or Table 3.
- Conductor is **Copper** → **Table 1.** (Table 3 would be aluminum conductor.)
- *Note:* the "aluminum sheath" is just the jacket; the **conductor** is copper, so material = copper → Table 1. (The Free-Air-vs-not and the table-pairing came from the video diagram, not the textbook.)

**Step 2 — read Table 1.**
- Find size **2/0**.
- Use the **90 °C** column (the cable is rated 90 °C; even if the question hadn't said so, the cable's name implies 90 °C).
- **Answer: ampacity = 300 A.**
- *Trap the instructor lived through:* if you accidentally use the **60 °C column** for 2/0 you get **195 A**; if you slip into the **aluminum table** or the **wrong size row** you get a wrong number (e.g., ~30-series). It took the instructor multiple tries the first time. The textbook's answer key (in the Section 4 exercises) confirms **300 A**.

**Step 3 — apply correction factors to the 300 A result.**
- **Ambient (Table 5A):** Table 5A applies to **all four tables (1–4)**, for ambient **above 30 °C** (e.g., 35 °C, 40 °C), based on the conductor's insulation rating.
  - Re-read: cable is **R90 / 90 °C-rated**; in the **best case (≤ 30 °C ambient)** it carries the full **300 A** **(approx — the worked base used 300 A here; the read value was 300 A).**
  - If ambient becomes **35 °C**: enter Table 5A at the **35 °C** row for a 90 °C conductor; the factor (illustrative) **× 0.96** → **300 × 0.96 ≈ 288 A** = the new maximum. So at 35 °C you can no longer push 300 A; max ≈ **288 A.** **(0.96/288 approx — verify Table 5A.)**
- **Grouping (Table 5C) — Free Air, Table 1 only here:**
  - Base (single) = **300 A.**
  - **2 conductors → × 0.9 → ≈ 270 A** (spoken "207"; correct arithmetic 300 × 0.9 = 270 — **flagged: spoken figure garbled**).
  - **4 conductors → × 0.8 → ≈ 240 A.**
  - **Reminder (TRAP):** for Free Air this grouping derate applies **only if** the conductors are spaced **< 25% of the largest cable diameter** apart (the "where" condition). If the question doesn't give that spacing condition, **do not derate** — answer stays **300 A**.

---

### 7. Worked Exercise — Section 4, Question 4 (grouping derate, Raceway/Cable, Copper)

**Given:** **25 conductors**, size **#4**, **copper** (the "25 kapar"), insulation **R90 / 90 °C**, run in a **raceway** (it's a raceway/cable install — RW/cable, R90 type).

**Step 1 — which table?**
- **Raceway/Cable** + **Copper** → **Table 2.** (Tables 1 & 2 = copper; 3 & 4 = aluminum. Raceway → 2 or 4; copper → 2.)

**Step 2 — base ampacity from Table 2.**
- Size **#4**, **90 °C** column → **base ampacity = 95 A.**
- (95 A is valid for **1, 2, or 3** conductors — no grouping derate up to three.)

**Step 3 — grouping correction (Table 5C) for 25 conductors.**
- For **25** conductors the factor = **0.6.**
- **95 × 0.6 = 57 A.**
- **Answer: corrected ampacity = 57 A** (matches the exercise answer key for the #4 case).

**Grouping factors for raceway/cable (Tables 2 & 4) — pattern stated:**
- **1–3 conductors → factor 1** (no derate).
- **4–6 conductors → × 0.8.**
- **7–9 conductors → × 0.7.**
- continues stepping down as the count rises (read exact rows from Table 5C; 25 → 0.6).

---

### 8. Homework / what to solve (Section 4 exercises)
- **Solve:** Section 4 **Questions 1 and 4** (the two done in class) plus the similar ones — instructor listed question numbers: **1, 4, 9, 11, 13, 23, 34** (some numbers spoken quickly — **approx, confirm the list in the workbook**).
- **Skip / self-review only:** any question involving **direct-buried / underground** cable that uses **D8/D9/D10/D11** — these were not taught and **don't appear on the exam**; just be aware they exist (review notes if you want, but don't study them).
- Also: review/solve **all of Section 2's** questions.
- Section starting point next time: continue with Section 4, **Question 3** ("Question 3 is itself a lesson").
- Class time note: today's session length was discussed as **2 / 3 / 5 hours**, trimmed to about **4 hours 45 minutes** (logistics, not exam content).

> The transcript's final portion is off-topic personal/TV-show conversation and contains no exam material — excluded.

---

## Mnemonics / exact phrasings

- **"Each section: write at the top how many tables it has."** Section 2 → **"2 tables: 56 and 65."**
  - **56 = working space. 65 = enclosures.**
- **"Table 56 working space, Table 65 enclosure"** — only two tables in Section 2.
- **Table selection:** **Copper → Tables 1 & 2. Aluminum → Tables 3 & 4.** **Free Air → 1 or 3. Raceway/Cable → 2 or 4.**
  - **Free Air + Copper = 1; Free Air + Aluminum = 3; Raceway/Cable + Copper = 2; Raceway/Cable + Aluminum = 4.**
- **"Free Air = a wire not touching anything 360° around it along its run"** (like street/overhead wires). Anything else = Raceway or Cable.
- **"Not more than three insulated conductors"** = the table's base assumption; grouping derate starts **above 3**.
- **"All tables are based on 30 °C ambient"** → above 30 °C use **Table 5A**.
- **Two correction factors, both from heat:** **grouping → Table 5C** (cable-to-cable heat, >3 conductors); **ambient → Table 5A** (environment heat, >30 °C).
- **"Order doesn't matter — whether Khaje-Ali or Ali-Khaje"** (cheh Ali-Khaje cheh Khaje-Ali) → multiply the correction factors in any order; multiplication commutes.
- **Free-Air grouping "where" trap:** derate only **where** spacing **< 25% of the largest cable diameter**; if the question omits the spacing condition, **do not derate**.
- **Round ambient UP:** for 36/38/39 °C, use the **40 °C** column of Table 5A ("take the forty") — never round down.
- **Transformer:** **> 50 kVA → 1 m** horizontal working space; **= 50 kVA → none.**
- **Gas clearance (Rule 2-308, Appendix B):** **Natural gas 1 m · Propane 3 m · 0.3 m if listed/coded regulator.** Natural gas rises (light); propane pools low (heavy → more dangerous).
- **"R" = rubber/plastic insulation; the number = temperature rating** (e.g., TW75 = 75 °C, R90 = 90 °C). The name tells you the temp.
- **"Mark Rule 2-308 in Appendix B in advance"** — the rule body is blank; the numbers are in Appendix B.
- **Section 4 is the "mother section" and the most conceptual — but only grade-5 level. Learn one table thoroughly; the rest follow the same pattern.**
