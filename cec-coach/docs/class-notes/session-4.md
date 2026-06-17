# Session 4 — Study Notes (BC Construction Electrician, CEC 2024)

**Topic of this session:** Section 8 — Load Calculations (in depth), plus the supporting ideas that feed into it: voltage drop, the interlock rule (8-106), basic vs. special loads, determination of area, range/cooking-unit demand, heating demand, Table 14 basic-load watts/m², show windows, branch-circuit outlet limits, EV/parking and block-heater loads, and a few Section 2 / Section 4 review traps.

> These notes are written to teach the material, not summarize it. Every numeric rule, every worked example, and every "trap" the instructor flagged is preserved. Where the audio was garbled, the number is marked **(approx — verify in book)**.

---

## How to study (study strategy the instructor insisted on)

- Up to **Section 28**: your ONLY job is to **understand the material**, section by section. Do **not** practice exam questions yet, do not worry about time, do not think about the exam at all. Just learn each section deeply.
- **From Section 28 onward**: now you switch to exam mode — mixed questions, 500-at-a-time practice, learning to *identify which section a question belongs to* (the real exam never tells you "this is a Section 4 question").
- Real exam logistics he mentioned: you get a good amount of time (he framed **5 hours as comfortable, 4 hours as tight**) — students rarely run out of time. **(approx — these are his recollections, confirm your own exam's official duration.)**
- Exam-day tactic: **do the fast questions first.** Questions are not equal weight — some take under 30–60 seconds, some (long load calcs) can take ~10 minutes. Don't surrender a booklet early; one student handed back with ~20 unanswered and failed at 65, when ~5 easy ones would have passed him (passing referenced as **~69/70%**).
- Read the **whole code book 3–4 times.** First read you won't be able to "file" things mentally; by the 2nd/3rd read you build the mental map of "what lives in which section."

---

## High-yield: important / tested / traps / common mistakes

### Section 2 / general orientation
- **Section 2 ("General Rules") is the hardest to navigate** because its title is generic — you MUST memorize what topics live inside it (Working Space, Enclosures, Disconnection, Voltage rating/marking, etc.). For most *other* sections the name tells you where to look (motors → motor section; airports; high voltage; cinema/film environments), so you don't memorize their contents — you just need to know the section exists.
- The instructor wants you to specifically memorize the contents of **Section 2, Section 12 (wiring methods), and Section 26.** Everything else is "findable."
- **Grounding/bonding appears in many sections** (Section 10 general, plus airports, Section 36 high voltage, film/TV Section ~66 with zig-zag cable layout, etc.). Treat grounding as a recurring sub-topic in every section — newer exam questions exploit this (e.g., grounding conductor *length* in mobile film-production environments).
- Table reference traps: **grounding = Table 43, bonding = Table 16.** Table numbers do NOT line up with section numbers (e.g., the Section 8 basic-load occupancy table is **Table 14**, not "Table 8-anything"). You must memorize a few of these.

### Section 4 review (Table 56 — working space) — **classic trap**
- **Table 56 left column uses LINE-TO-GROUND voltage**, not line-to-line. This is the #1 trap on these questions.
- Relationship: **V(line-to-line) = √3 × V(line-to-ground)** for three-phase; the **smaller number is always line-to-ground**, the **larger is always line-to-line.**
- **Single-phase:** when you're given ONE voltage it is the **line-to-line** value, and the line-to-ground relationship is **1:2** (factor of 2), NOT √3.
- **Trap — "is it already converted?"** If the problem states **14,189 V** or **8.6 kV** (i.e., already a line-to-ground figure they computed for you), do NOT divide by √3 again. Students who memorize "always divide by √3" get this wrong. Only convert when you're given the line-to-line value.
- **Trap — three-phase with only one voltage given:** the problem MUST tell you it's three-phase; if it gives one voltage and doesn't say single- or three-phase, the question is technically broken — but if it says three-phase, take the given line-to-line value, divide by √3, then enter Table 56.
- **120/240 V single-phase:** to enter Table 56 you use the **line-to-ground = 120** (the smaller), not 240.

### Section 8 — biggest exam area & where the 2024 changes hit
- **The single biggest change from CEC 2021 → 2024 is voltage drop.** The *wording* changed and, more importantly, the **calculation method in Appendix D changed completely.** Old method = 3 tables + 3 formulas; new method = much simpler, basically **one formula with 4 inputs** (the exam gives you 2, you look up 2 fixed values from a table). Make sure you know which code year your exam uses.
- **Don't memorize the W/m² numbers** — know *where* they live so you can look them up fast. But the instructor noted that after enough practice, 99% of students end up memorizing them anyway.

### Range / cooking-unit demand — **the most common mistake**
- **Two different base values depending on what you're calculating:**
  - **Service or feeder (dwelling):** range demand base is **6 kW** (then +40% of anything above 12 kW).
  - **Branch circuit feeding the range in a dwelling unit (8-200/8-300 branch rule):** base is **8 kW** (then +40% of anything above 12 kW) — bigger, because the conductor serves only that range with no diversity.
- **Students constantly get a 2 kW discrepancy** between service answer and branch answer and panic — the difference is exactly the **6 kW vs 8 kW** base. This is intentional on the exam.
- **Commercial/industrial cooking (restaurant, bakery, etc.):** range/cooking units are taken at **100% of rating** (not less than nameplate), NOT the dwelling diversity. A 15 kW bakery oven = 15 kW. Don't accidentally apply the dwelling demand factor commercially.

### Loads over 1500 W (dryer, tank water heater, etc.)
- **WITH an electric range present → take 25% of each such load** (method A). **WITHOUT a range → method B applies** (different treatment).
- Trap: dryer and **storage (tank) water heater** are the typical >1500 W loads. **Tankless water heaters are 100%** (no diversity). **Steamers = 100%. EV charger (vehicle) = 100%.**

### Heating vs. AC — interlock (8-106)
- **8-106 interlock rule:** "Where interlock(s) are installed to prevent simultaneous operation of [electric heating and air conditioning], use ONLY the LARGER of the two in the calculation." Exact code phrase to recognize: *"…whichever is greater shall be used in this calculation."* The smaller becomes 0.
- **Trap:** without interlock you must **add** heating + AC. The problem may give the interlock as a *number* (15 kW heat vs 10 kW AC → use 15) OR only describe it in words.
- **Professional note (hospitals etc.):** even if not stated, heat and AC are interlocked in hospitals — but you only apply that on the design side, not as an exam assumption unless told. For a house it might NOT be interlocked.

### Heating demand staging
- **Electric heating: first 10 kW at 100%, remainder at 75%** — but ONLY when the system requires a thermostat per room / per heated area (e.g., **baseboard heaters with separate thermostats per area**).
- **Trap:** if it's a **furnace (electric furnace)** — one thermostat heats the whole house — take **100%**, the 10kW/75% staging does NOT apply.
- **Trap:** **gas furnace load is NOT calculated** at all (we only count electrical heating). Don't confuse electric vs gas furnace.

### Determination of area
- **Commercial (e.g., a bank with 100 m² basement + 100 m² main = 200 m²): count 100% of all floors.**
- **Residential single-dwelling living area: ground floor 100%, floors above ground 100%, but BASEMENT at 75%** (basements have lower basic load — pot lights, storage, small bar fridge, etc.).

### Show windows (Rule 8-202)
- Show window (store display window / "show window") load: **not less than 650 W per linear metre of show window.** Common easy exam question: "show window load = 550 / 650 / 750 / 850 W per metre?" → **650.** They make it harder by giving a length ("3 m show window → load?" → 3 × 650 = 1950 W).

### Branch-circuit outlet limit (Section 26 preview)
- **Maximum 12 outlets on a 15 A circuit** (residential general-purpose). Memorize **12.**
- The deeper rule (why): an outlet must be able to supply ~**1 A continuous** average; you size by continuous current. 15 A breaker, non-continuous/unmarked → 80% → **12 A continuous → 12 outlets.**

### Service / feeder minimum sizes
- **Single dwelling > 80 m²: minimum 100 A service.** (Method B shortcut: **24,000 W ÷ 240 V = 100 A**.)
- **≤ 80 m² (smaller dwelling): minimum 60 A.** (Method B: **14,400 W ÷ 240 V = 60 A**.) **(approx — audio said "14,000…/14,400"; the standard CEC figure is 14,400 W → confirm.)**
- **Apartment dwelling unit / suite minimum: 60 A.** (A rented basement suite is calculated like an apartment, min 60 A.)
- These minimums apply to **service and feeder only — never to the branch circuit** (branch is sized by its actual load).

### Table 14 (basic load by occupancy) — **new exam item flagged**
- Table 14 gives **basic load in W/m² by occupancy type** (industrial, church, garage, warehouse, cinema, **armouries**, bank, salon, club, lounge, etc.).
- **NEW exam question the instructor flagged: "Armoury / armouries"** (military storage building) — students fail it because they don't know the word and can't find it in the table. Know it's in Table 14.
- **Trap:** Table 14 values are **BASIC LOAD ONLY.** Do NOT assume they include water heater, range, heating, etc. Add special loads separately. The instructor wishes the table said "basic load only" on it.
- **Trap (demand factor):** for **service and feeder** you take the Table 14 result at **100%**; the demand factor only kicks in for the **branch** treatment. (He worked a 100 m² restaurant at 30 W/m² = 3000 W, then noted 3500 for the branch case — see worked example.) **(approx — restaurant W/m² value 30 was used in his example; confirm the exact Table 14 value for your occupancy.)**

### Operating room / hospital (Section ~24, patient-care areas)
- **Operating rooms / high-intensity areas: 20 W/m² for general/basic load PLUS 100 W extra for the operating-room (I-intensity) lighting.**
- **TRAP (the "unfair" question that was pulled from the exam):** the **100 W is added ONCE for all operating rooms combined / per the rule's intent — you do NOT add 100 W separately to each operating room** so that each becomes 120 W. The instructor explicitly said this trick is *not explained in the book*, it caught everyone, the question was so unfair it was withdrawn after mass complaints — but **know it can come back.** He teaches this only to the apprenticeship/inspection-track students because of the subtlety. **(approx — the precise wording of "once vs per-room" was emphasized verbally; verify the exact rule text in the patient-care-area section.)**

---

## Content taught (in order, full detail)

### 1. Mindset / how sections are organized (Section 2 emphasis)
- The book is taught **section by section up to Section 28**, then exam-style mixed practice.
- **Section 2** is titled "General Rules" — generic name, so you must *memorize its contents*: Working Space (multiple marked items), Enclosures, Disconnection, Voltage/Conductor rating, etc. He marked ~20+ items in Section 2 that roll up to ~5–6 headings.
- For sections with descriptive names (motors, services, lighting, airports, high voltage, cinema), you don't memorize contents — the name tells you where to search. Example: "All lighting questions → there is only ONE lighting section." Students waste time asking "which section is lighting in?"
- **Grounding recurs everywhere.** You learn grounding generally (Section 10), then it reappears with extra rules in: airports, **film/TV production (Section ~66)** — cable must be laid out **zig-zag**, and grounding-conductor **length** is limited so cars don't run over it and people don't trip — and **high voltage (Section 36).** Treat grounding as a cross-cutting topic.
- Patient-care areas live in **Section ~24** (receptacle heights in hospitals, etc.) — you just need to know the section exists; you don't memorize the numbers, you look them up.
- **Cathodic protection** "is near the end" of the book and "always appears on the exam."

### 2. Section 4 review — Table 56 working space (worked dialogue)
- Table 56: input **voltage (line-to-ground)** in the left column → read out the **minimum working-space clearance.**
- Core relationships re-taught:
  - **Three-phase:** V(L-L) = √3 × V(L-G). Larger = L-L, smaller = L-G.
  - **Single-phase:** the one given voltage is L-L; L-G relationship is **1:2.**
- **Worked example given (Q10-type):** problem states **4 kV, single-phase, line-to-line.**
  - Single-phase ⇒ given value is line-to-line.
  - To enter Table 56 need line-to-ground ⇒ **4000 ÷ √3 ≈ 2300 V** (he rounded to "2300", said "2.3").
  - Enter Table 56 at ~2300 V ⇒ working space ≈ **1.2 m.**
  - **Wrong answer the exam plants:** if a student forgot to convert (or used the wrong factor) they'd land on **1.5 m** — that's the decoy. He noted the exam deliberately seeds the wrong answer so a student who half-remembers picks it.
- **Already-converted example:** if the problem gives **14,189 V** (i.e., 14.4 kV line-to-line already divided), you put **14,189** straight into the table — do NOT divide again. Same logic if they hand you **8.6 kV** directly.
- Takeaway he wanted memorized from day one: **(a)** smaller number is line-to-ground; **(b)** ratio 1:2 ⇒ single-phase, ratio √3 ⇒ three-phase (works both directions).

### 3. Why Section 8 / load calc exists
- To get a new service you submit a **load calculation** to the authority (BC Hydro / city). They use it to size the service. Houses: they'll grant 200 A fairly readily but still want the calc; a commercial place (pizza shop with multiple AC units, fridges, etc.) you must actually compute.
- Output of a load calc = the **demand**, governed by **demand factors.** "Demand factor" ≈ synonym for "load calculation." The instructor wishes the section were named "Load Calculation."
- BC context aside: ~80% of BC's power is hydro (BC Hydro), so utilities *prefer* you use electricity over gas. (Context only.)

### 4. Two load categories
- **Basic load:** always present everywhere = **lighting + receptacles.** You do NOT count lamps/receptacles one-by-one; **basic load is computed from AREA (m²).** Every occupancy has lighting + receptacles, so it's universal.
- **Special load:** loads that may or may not exist in a given building — **electric range, dryer, water heater, EV charger, electric heating** (vs gas), AC, sauna, etc.
- **Total = basic load + special loads**, but special loads are NOT blindly summed — **demand factors and diversity** apply (you'd never run range + water heater + EV charger + everything at once, so codes discount).

### 5. Voltage drop (taught here even though it logically belongs with branch/Section 4)
- Concept: source (pole transformer at the street) → conductor run → your load. The longer the run, the more the voltage **drops** (e.g., 125 V at the street → 122 V or 120 V at the last house).
- Why it matters: undervoltage makes equipment malfunction or burn out. **Motor example:** P = V × I (constant power). If V drops, **I rises** to keep power constant; higher current overheats conductors not sized for it; the motor's winding insulation/lacquer melts → turns short → **motor burns out.** Conversely overvoltage (he mentioned a spike to ~190 V) is also damaging.
- You can't control the utility voltage, but you **control voltage drop** by **conductor size** (and run length). VD is inversely related to conductor cross-section.
- **Field example:** installing an EV charger in a parking spot far from the panel — first sized cable by ampacity, then checked VD, found VD exceeded limit, **bumped up one conductor size, rechecked, passed.**
- **The limits (memorize):**
  - **Branch circuit: ≤ 3%.**
  - **Feeder: ≤ 3%.**
  - **From the point of supply (utility) all the way to the load (total): ≤ 5%.**
- **Critical interpretation / trap:** because branch + feeder must together stay within the 5% total, **they are NOT independently allowed 3% each.** If the branch already used 3%, the feeder may only use **2%** so the sum ≤ 5%. You must watch BOTH the individual 3% limits AND the combined 5% limit. (Most relevant in commercial/industrial/multi-feeder layouts; in a simple house with one panel it rarely binds.)
- **2024 change reminder:** VD wording changed and the Appendix D method was simplified (old: 3 tables + 3 formulas; new: ~1 formula, 4 inputs, 2 looked up). He made separate 2021 and 2024 videos and posted a multi-part voltage-drop video (Parts 1–4) covering every question type.

### 6. Interlock rule (8-106) — heating vs AC
- Exact phrase to recognize: **"Where interlock(s) are installed to prevent simultaneous operation of …"**
- Thermostat with COLD / HEAT positions ⇒ you physically can't run heater and AC at the same time ⇒ they're interlocked.
- **Rule: in the calculation use ONLY the larger of {heating, AC}; the smaller = 0.**
- **Worked numbers:** heating = **15 kW**, AC = **10 kW**, interlocked ⇒ use **15 kW** (drop the 10).
- Exam may give it as words only ("AC and heating are interlocked, which do we count?") → answer: the larger. Code phrase: *"…whichever is greater shall be used in this calculation."*
- If **NOT** interlocked ⇒ **add** both.
- Hospitals: practically always interlocked (design-side note).

### 7. Number of overcurrent-device spaces required in a dwelling panel
- Rule references the number of breaker spaces a panelboard must provide based on service size (he paraphrased an older rule: e.g., **service ≤ 100 A → provide spaces for at least ~24 breakers; > 100 A → ~30**). **(approx — these counts were a verbal recollection; verify the current 8-108/panel-space rule.)**
- Key conceptual takeaways (these ARE the exam points):
  - The code does **not** dictate how many breakers *you* install for specific appliances — it dictates **minimum spare/space capacity.**
  - **Single dwelling:** at original installation, leave **at least 4 spare spaces** for future use.
  - **One of those spares must be capable of a 2-pole (double) overcurrent device** (240 V loads: dryer, range, sauna are 2-pole, mechanically tied so both poles trip together).
  - **Apartment with a 240 V load (e.g., sauna):** leave **at least 2 additional spaces**, and **at least one must be a 2-pole.** "At least" — if you leave 4, still at least one must be 2-pole-capable.
- "Single dwelling = house." "Dwelling (non-single)" includes hotels, apartments, etc.

### 8. Determination of area (8-110)
- **Commercial (bank example):** 100 m² basement + 100 m² main floor ⇒ **200 m²** (every floor counted 100%).
- **Residential single dwelling:** living area = **ground floor 100% + above-ground floors 100% + basement at 75%.** So 100 m² each on basement/main/upper "real" 300 m² ⇒ counts as **250 m²** for living area. Reason: basements have lower load density (pot lights, storage, beverage fridge). Heating in the basement is a *special* load, separate from this basic-load area math.

### 9. Single-dwelling basic load formula (8-200)
- **Rule:** **5000 W for the first 90 m²**, then **+1000 W for each additional 90 m² (or portion thereof).**
- Examples:
  - 90 m² ⇒ **5000 W.**
  - 95 m² ⇒ next portion starts ⇒ **6000 W.**
  - 200 m² ⇒ **7000 W** (matches the "200 m² → 7000 W" figure he mentioned earlier).
- Then: **AC at 100%** (he flagged AC is taken at 100% here / "50%" was misspoken — **verify**, but he then corrected the operative rule below). **Air-conditioning** is added; **heating** uses the staging rule (Section 8 / 62 cross-reference).
- **(approx — there was a garbled stretch ("50%", "Section 62", "motors") around AC %; the clean, repeated rules are the 5000+1000/90 basic-load and the heating staging in item 11. Verify the AC percentage in 8-200.)**

### 10. Single-dwelling Method A vs Method B (minimum service)
- **Method A:** itemize — compute basic load, then add each special load (AC, water heater, EV charger, etc.) with their demand factors. (Detailed, used when you need the real number.)
- **Method B (shortcut for service/feeder of a single dwelling = house or townhouse):**
  - If **floor area > 80 m²** ⇒ **24,000 W minimum** (24,000 ÷ 240 = **100 A**).
  - If **floor area ≤ 80 m²** ⇒ **14,400 W** (÷240 = **60 A**). **(approx — he said "14,000…"; CEC figure is 14,400 → verify.)**
  - **You take the LARGER of Method A (the real calc) and Method B (the minimum).** Example: a 100 m² house that calculates to **23,000 W** must be bumped to **24,000 W**; but if it calculates to **25,000 W** you keep **25,000 W.** Method B is effectively a floor/minimum.
- **Method B applies to SERVICE and FEEDER ONLY — never to a branch circuit.** Branches are sized by their actual connected load.

### 11. Electric heating demand (single dwelling)
- System with **per-room / per-heated-area thermostats** (e.g., **baseboard heaters**, each area its own thermostat):
  - **First 10 kW at 100%, remainder at 75%.**
  - Worked: **8 kW ⇒ 8 kW** (all of it, under 10). **10 kW ⇒ 10 kW.** **20 kW ⇒ 10 kW + 75% of 10 kW = 10 + 7.5 = 17.5 kW** (he said "10 + 7500 = 17,500").
- **Furnace (electric furnace):** one thermostat, whole house ⇒ **100%** (no staging).
- **Gas furnace:** **not counted** (no electrical heating load).
- "Heated area" can be several baseboard heaters but **one thermostat** controlling them — exam may say "thermostat control in each room" (staging applies) OR "baseboard heater" (recognize it's the staging rule).

### 12. Electric range / cooking-unit demand
- **Single dwelling — service/feeder (8-300):**
  - **Base 6 kW**, regardless of nameplate up to 12 kW. So a range of **6, 8, 10, or 12 kW all count as 6 kW** for the service.
  - **Plus 40% of any amount over 12 kW.**
  - Worked: **16 kW range ⇒ 6 kW + 40% × (16−12)=40% × 4 = 1.6 ⇒ 6 + 1.6 = 7.6 kW.**
- **Single dwelling — BRANCH circuit feeding the range (dwelling unit):**
  - Code text: *"Conductor of branch circuit supplying range in dwelling unit"* ⇒ **base 8 kW** (the 6 becomes 8), still **+40% over 12 kW.**
  - Worked: **16 kW range, branch ⇒ 8 kW + 40% × 4 = 8 + 1.6 = 9.6 kW.**
  - Reason: the branch conductor serves only this range — no diversity with other loads — so it's sized stronger.
- **The classic confusion:** service answer **7.6 kW** vs branch answer **9.6 kW** — the **2 kW gap is exactly the 6→8 kW base change.** A student got stuck on this; that's the whole explanation.
- **Commercial / industrial cooking (8-210):** "range or cooking units installed in commercial/industrial" ⇒ demand **not less than the rating → effectively 100%.** Example: **15 kW oven in a bakery ⇒ 15 kW** (it may be given as a branch, but no diversity reduction). Don't apply dwelling demand factors here.

### 13. Loads over 1500 W (8-200, the "additional loads" group)
- A grouped rule covers miscellaneous loads **rated over 1500 W** not individually listed (e.g., **dryer**, **storage/tank water heater**).
- **Method A (range PRESENT):** **take 25% of EACH such load.** Example: dryer 4 kW + tank water heater 4 kW = 8 kW total; **25% × 8000 = 2000 W (2 kW).** (Or 25% of each individually — same result.)
- **Method B (NO range):** different treatment applies.
- **100% loads (no diversity):** **tankless (instantaneous) water heater = 100%**, **steamer = 100%**, **EV charger (vehicle) = 100%.** (Tankless turns on the instant you open the tap, so no storage diversity.)

### 14. Worked full single-dwelling example (Q3-type, 360 m²)
- Total area **360 m².** Basic load via 5000 + 1000/90:
  - First 90 m² ⇒ **5000 W.**
  - Remaining 360 − 90 = **270 m²** ⇒ 270 / 90 = **3 more 90-m² blocks** ⇒ 3 × 1000 = **3000 W.**
  - **Basic load = 5000 + 3000 = 8000 W.** (In the audio he and the student fumble the arithmetic — the correct count is 3 additional blocks, basic load **8000 W.**)
- Then add range (6 kW base + 40% over 12), heating (10kW/75% staging), >1500 W loads at 25% (range present), etc., and finally compare to the **24,000 W** Method-B minimum. (This is the long, ~5-line word-problem type — under 30 s once practiced.)

### 15. Restaurant / commercial example using Table 14
- Table 14 = **basic load W/m² by occupancy.** Mark it red; it's for "other-than-dwelling" occupancies.
- **Restaurant 100 m² @ 30 W/m² ⇒ basic load = 3000 W** (for service/feeder, 100%). For the **branch** treatment he cited **3500 W** (demand factor differs). **(approx — confirm exact Table 14 value & the branch number.)**
- Reminder: Table 14 is **basic load ONLY** — add range/water heater/heating separately.

### 16. Worked single-phase commercial service example (storage occupancy)
- Building **16 m × 10 m = 160 m².**
- **Storage @ 5 W/m² (Table 14) ⇒ 160 × 5 = 800 W** lighting/basic load.
- For **service**, storage basic load is taken at **70%**: **800 × 0.70 = 560 W.** **(approx — he stated the basic load → 560 after a 70% factor; the running total below uses figures from the dialogue, verify each in-book.)**
- He then folds in heating at the **heating factor (75% on the remainder / "100% then 75%")** vs non-heating at the basic %. The dialogue carried a **15 kW** heating element (counted) into the total.
- Final running total quoted: divide total by **240 V** to get amps; he read **"189.83"** type result ⇒ rounds up to next standard service. **(approx — the intermediate sums in this example were spoken quickly and partly garbled; treat the METHOD as the lesson, recompute the numbers from the book: area→Table 14 basic load → apply service %, add heating at staging, add special loads, ÷240 → amps → round up to standard size.)**
- Lesson he stressed: **you should solve this WITHOUT opening the book** except to confirm the one Table 14 value — the table must be in your head from practice.

### 17. Show windows (8-202)
- **≥ 650 W per linear metre** of show window.
- Question forms: "what value per metre?" → 650; or "3 m show window load?" → 3 × 650 = **1950 W.**

### 18. Branch range/cooking conductor note (penthouse example) tying it together
- A penthouse with a **16 kW range**, asked for the **branch-circuit** demand ⇒ apply the **8 kW base** rule ⇒ **8 + 40%×4 = 9.6 kW** (not the 7.6 kW service value). This is exactly the "2 kW off" student confusion resolved.

### 19. Maximum outlets per circuit (Section 26 preview, Rule 26-712 area)
- **Max 12 outlets on a 15 A general-purpose circuit.**
- Derivation taught: each outlet ≈ **1 A continuous.** Size by **continuous** current.
  - **15 A, unmarked / non-continuous ⇒ ×80% = 12 A continuous ⇒ 12 outlets.**
  - 15 A breaker **marked for continuous** (100%) ⇒ 15 outlets (rare).
  - **20 A, 80% ⇒ 16 A ⇒ 16 outlets**; 20 A @ 100% ⇒ 20 outlets.
- "Outlet" = one duplex receptacle counts as one outlet (he treated a duplex as a single outlet for this count).
- Practical framing: when replacing/adding to a panel you compute how many outlets each breaker can carry; you don't put one breaker per receptacle. Dedicated branches required for fridge, dishwasher, dryer, **microwave** (its own branch, nothing else on it), etc. — details come in Section 26.

### 20. Block heaters & EV/parking loads (8-106 / 8-200 parking, Tables 8-1 / 8-2 style)
- **Block heater:** an element keeping engine oil from freezing in extreme cold (−18 °C and below; −40/−50 in the Prairies — Saskatoon, Calgary, Winnipeg). Relevant for parking-lot load calcs.
- **Parking spaces, two categories:**
  - **NOT restricted or controlled** (typical mall lot, anyone can plug in) → use the **first table** (Table 8-1 style). This is the common case.
  - **Restricted or controlled** → use the **second table** (Table 8-2 style).
- **Max branch circuit current per parking circuit: 15 A or 20 A** (the rule states one of these).
- **Worked parking-lot calc (200 stalls):**
  - **First 60 stalls @ 1200 W each.** (He computed "3000 ÷ 1200"-type framing while choosing 15 A; the operative per-stall figure used was **1200 W**.) **(approx — first-tier per-stall watts stated as 1200; verify the exact Table value for your code.)**
  - **Remaining stalls (200 − 60 = 140) @ ~800 W each** (he said 800 for "above 60").
  - Sum first-60 tier + remaining-140 tier = total parking load.
  - "Per stall = per space" — one allowance per parking space/stall.
  - **(approx — the 1200 W / 800 W / 60-stall breakpoint were spoken quickly; confirm against the parking-load table in Section 8. The METHOD is the lesson: first N stalls at the higher per-stall watt, remainder at the lower per-stall watt, then sum.)**

### 21. Operating room / hospital basic + intensity load
- **Patient-care / operating areas: 20 W/m² basic load** PLUS **100 W for the I-intensity (operating-room) load.**
- **Trap (the withdrawn "unfair" question):** the **100 W is NOT added per operating room** to make each 120 W; it's applied per the rule's intent (not multiplied across every OR). Caught everyone; question pulled after complaints; **may return.** Not in the book's plain text — a known trick.

### 22. Occupancies he flagged as exam-likely (Table 14)
- **Armoury / armouries** (NEW, students fail it — it IS in Table 14).
- Hotel/motel basic-load questions.
- "Other-than-dwelling occupancy" definition: an *occupancy* = a place/use type; the term "occupancy" itself you must learn — Table 14 lists the qualifying occupancies (industrial, church, garage, warehouse, cinema, armoury, bank, salon, club, lounge, etc.).
- School and hospital: **basic-load** questions can appear; full apartment/hotel detailed calcs are NOT taught (only basic-load-style questions appear).

---

## Mnemonics / exact phrasings

- **"Smaller is line-to-ground, larger is line-to-line."** (Table 56 input.)
- **"Ratio 1:2 ⇒ single-phase; ratio √3 ⇒ three-phase"** — works in both directions.
- **"Single-phase gives you ONE voltage, and that one is line-to-LINE."** Don't re-divide a number that's already line-to-ground.
- **Voltage drop: "3% branch, 3% feeder, 5% total — and 3+3 ≠ 5, so if the branch ate 3%, the feeder gets only 2%."**
- **Interlock (8-106): "whichever is greater shall be used in this calculation."** Larger of heat/AC; smaller = 0. No interlock ⇒ add them.
- **Heating staging: "first 10 kW at 100%, the rest at 75%"** — baseboard/per-room thermostat. Furnace = 100%. Gas furnace = nothing.
- **Range bases: "SIX for service, EIGHT for the branch; plus 40% over 12."** (6 kW service / 8 kW branch.)
- **Commercial cooking: "not less than the rating" = 100%.**
- **Basic load single dwelling: "5000 for the first 90 m², +1000 each additional 90 (or part)."**
- **Method B minimums: ">80 m² → 24,000 W → 100 A; ≤80 m² → 14,400 W → 60 A; take the larger of A and B."**
- **Area: "main & upper 100%, basement 75%" (residential); "all floors 100%" (commercial).**
- **>1500 W loads WITH a range: "25% of each."** Tankless / steamer / EV charger = 100%.
- **Show window: "650 watts per metre, minimum."**
- **"12 outlets on a 15-amp circuit"** (15 A × 80% = 12 A ⇒ 1 A/outlet ⇒ 12).
- **Operating room: "20 W/m² plus 100 W — and the 100 is NOT per room."**
- **"Basic load = lighting + receptacles, from AREA. Special load = it's here in one building, gone in another."**
- **Tables don't match section numbers:** grounding = **Table 43**, bonding = **Table 16**, occupancy basic load = **Table 14.**
- Study order: **"Understand everything up to Section 28; only AFTER 28 do you practice for the exam."**

---

### Numbers flagged approximate (recompute from the code book before relying on them)
- Table 56 worked answers (1.2 m / 1.5 m decoy) — verify against your Table 56.
- 60 A / 14,400 W small-dwelling minimum (audio said "14,000").
- AC percentage in the single-dwelling formula (garbled "50%").
- Panel breaker-space counts (24 / 30) — verify the current panel-space rule.
- Restaurant 30 W/m² (and 3000 → 3500 branch figure).
- Storage example running totals (800 → 560 at 70%, the 15 kW heating fold-in, "189.83" amps).
- Parking-lot tier values (1200 W first 60 stalls, 800 W remainder, 60-stall breakpoint, 15/20 A).
- Operating-room "100 W once vs per-room" exact wording.

*Source: Session 4 ("جلسه ۴") Persian class transcript, 72,602 characters, read in full. Off-topic chatter (motivational anecdotes about other students, personal stories) excluded; all technical content retained.*
