# Six-Week Study Strategy — BC Construction Electrician (Red Seal / C of Q), CEC 2024

**Goal:** Pass the 100-question, 70%-to-pass exam on the first try — aim for a real margin (target ≥85%, not a 71% squeaker; remember 68 and 69 are fails).

## The exam you are actually beating
- **100 MC questions, 5 hours guaranteed, 70% pass.** Open-book but the supplied code book is **CLEAN — no tabs, no highlights, no notes.** Your edge is *navigation speed and knowing where things live*, not tabs.
- **Blocks are graded separately. Blocks B + C ≈ 60 questions and DECIDE pass/fail.** Aim ≥90% there. Block D (Section 28 motors) is the hardest — drill it for damage control, not perfection. Blocks A/E are lighter free marks.
- **~20% of questions are pure table lookups** (easy if you know the table), **~6–7 come from Appendix B**, and the **Index** can save 4–5 questions you'd otherwise never locate.
- **~90% of questions match the teacher's bank**, and this app's **365 practice questions are weighted to the blueprint.** Treat the app sets as the closest available proxy for the real bank — run them to mastery, not just once.
- The exam is **not full of trick questions** — but it *is* full of **modifier-word traps** ("except / unless / provided that / not exceeding / both") and **distractors printed to catch a wrong method** (e.g. the 1.5 m working-space answer for someone who forgot ÷√3). Read the exact wording.

## How each week is built (the daily template — ~2 to 2.5 h/day, ~6 days/week)
Every study day runs the same four-block loop. Do them in this order:

1. **Code-navigation drills — 20 min.** Run the "fast navigation routine" (below) on 8–10 scenarios from that week's sections. Time yourself: target landing on the governing rule/table in **under 60–90 seconds**. This is the single highest-leverage habit because the book is clean.
2. **Topic study WITH the book open — 45–60 min.** Read the week's rules *in the actual code book* (not just these notes), follow every cross-reference and Appendix B note, and copy the week's "memorize-cold" numbers onto a paper cheat-map (study only — you can't bring it). Watch a YouTube animation for any concept you can't picture (bonding, armour, drip loop).
3. **Timed practice — 40–50 min.** Run the assigned app set(s) **timed at ~2 min/question with the book open**, exactly like exam conditions. Flag-and-move; never sink 6 minutes into one item.
4. **Review every wrong answer — 20–30 min.** For each miss, find the rule in the book, read *why* each distractor is wrong, and add the trap to a running "trap log." Re-quiz misses the next morning as a 5-minute warm-up.

> App sets referenced by their files in `cec-coach/data/questions/` (365 Q total): `occupational_skills` (63), `section02_general` (18), `section04_conductors` (34), `section06_services` (12), `section08_loads` (27), `section10_grounding` (14), `section12_wiring` (74), `section14_protection` (11), `section16_class` (15), `section26_equipment` (53), `section28_motors` (44).

---

## Week 1 — Foundations: navigation, definitions, Section 2, and the ampacity engine (Section 4)
**Why first:** Section 4 is the "mother" section and the most *conceptual* in the book — an early Section-4 error corrupts every downstream calculation. Section 0/2 definitions are free marks that also control how later rules read. Navigation is the meta-skill you'll use all six weeks.

**Focus topics:**
- Book architecture: 6 parts; even-numbered Sections 0–86; Sections 0–16 general, 18–86 specific (a specific section can override the general). Front-matter **metric/Imperial conversion** (lux vs foot-candle) and **trade-size → inch** table.
- **Section 0 definitions:** ampacity; conductor/insulation/armour/jacket; AWG vs kcmil; bonding vs grounding; **branch circuit / feeder / service**; concealed/exposed; **conduit vs EMT (EMT is NOT a conduit)**; raceway; dwelling unit; emergency lighting (NBC vs CEC); GFCI Class A.
- **Section 2:** voltage classes (ELV/LV/HV); **Rule 2-130 dwelling 150 V-to-ground**; equipment marking (L-L vs L-G, √3); Rule 2-030 deviation/special permission; **Table 56 working space (uses LINE-TO-GROUND voltage)**; **Table 65 enclosures (4X/6P)**; Rule 2-328 gas-clearance (Appendix B).
- **Section 4:** the 4-table selection grid (T1 free-air Cu, T2 raceway Cu, T3 free-air Al, T4 raceway Al); temperature columns (60/75/90 °C); correction factors — **grouping Table 5C** (>3 conductors) and **ambient Table 5A** (>30 °C, round temp UP); **Rule 4-006 terminal-temperature limitation**.

**Traps to master this week:**
- Voltage-class boundaries are inclusive: **30 V = Extra-Low**, **1000 V = Low** ("not exceeding X" includes X).
- **Table 56 uses line-to-ground** — convert first (÷√3 for 3-φ; single-phase value given is already L-L → 120/240 device uses 120). The exam prints 1.5 m as the trap for forgetting √3.
- **Rule 4-006:** size on the **terminal's marked temperature** column, not the cable's higher insulation rating. Unmarked default: 60 °C if ≤100 A, 75 °C if >100 A — **at exactly 100 A use 60 °C**.
- **Two temperatures in one conduit:** rate each conductor on the **lower** temp, compute **separately, never add.**
- The exam loves the *rare* table combo (free-air aluminum = T3); learn all four, the method is identical.
- Free-air grouping only applies if spacing < 25% of the largest diameter; **if spacing isn't stated, don't derate.**
- "EMT is NOT a conduit"; "ungrounded conductor = the hot."

**App sets:** `occupational_skills` (63 — these are Block A free marks; clear them now while light), `section02_general` (18), `section04_conductors` (34). Run each timed, then re-run every miss.

**Milestone:** Score **≥80% on `section04_conductors` and `section02_general`**, and demonstrate the navigation routine landing on any Section 2/4 table in **<90 s**. Build the first page of your paper cheat-map.

---

## Week 2 — Loads & demand: Section 8 (the most math-intensive block)
**Why now:** Section 8 is Block B, high-frequency, and the single most calculation-heavy topic. It also holds the **#1 2021→2024 change (voltage drop method).** Connected-load vs calculated-load is a top documented failure cause.

**Focus topics:**
- **Basic load (8-110):** 5000 W first 90 m² + 1000 W per additional 90 m² (round partials up). Area: commercial = all floors 100%; single-dwelling **basement = 75%**.
- **Calculated load / minimum service (8-200(1)(b)):** >80 m² → **24 000 W / 100 A**; ≤80 m² → **14 400 W / 60 A**; take the larger (service/feeder only, not branch).
- **Range demand:** SERVICE = **6 kW** base; **dwelling BRANCH = 8 kW** base; both + 40% over 12 kW. (16 kW → 7.6 kW service vs 9.6 kW branch.) Commercial cooking (8-210) = 100%.
- **Heating** 100% first 10 kW + 75% remainder (per-room thermostats); electric furnace 100%; **gas furnace not counted**; A/C 100%.
- **Interlock (8-106):** use the **larger** of heat/AC; not interlocked → sum both.
- Loads >1500 W with a range present → **25%**; tankless water heater = 100%.
- **Voltage drop:** ≤3% branch, ≤3% feeder, **≤5% total** (if branch uses 3%, feeder ≤2%). The 2024 method is one formula (2 given + 2 looked-up).
- Table 14 W/m² by occupancy ("armoury", restaurant ~30, warehouse 5) = **basic load ONLY**; show window **650 W/m**; **12 outlets max on a 15 A circuit** (80% × 1 A/outlet).

**Traps to master this week:**
- **Connected load ≠ calculated load** — the question lists every appliance wattage to bait you into summing them. Apply 8-200 demand factors.
- **Range: 6 kW on the service, 8 kW on its own branch** — the penthouse-range 2000 W trap.
- Table 14 gives basic load only — don't multiply area and walk away; special loads still add on.
- **Operating room = 20 W/m² + 100 W** (the 100 W is additive, not per room) — withdrawn "unfair" question that may return.
- Voltage-drop coordination: 3% + 3% is capped by the 5% total.

**App sets:** `section08_loads` (27). Run timed twice; re-derive each worked example by hand without the answer first.

**Milestone:** Run a full single-dwelling service calc (basic + range + heating + >1500 W loads ÷ 240 V) **in under 5 minutes with only one table open**, and score **≥85% on `section08_loads`.**

---

## Week 3 — Grounding & Bonding: Section 10 (clean, well-organized, no excuse to lose marks)
**Why now:** Block B, frequently tested, and the teacher calls it the cleanest section — "if you don't score here there's no excuse." The exam deliberately mixes grounding vs bonding scenarios.

**Focus topics:**
- Three electrode classes: **manufactured** (rod ≥2, ~3 m apart, interconnected; plate ≥0.2 m², ≥600 mm deep), **field-assembled** (bare Cu ≥6 m, ≥600 mm deep), **in-situ**.
- **Concrete-encased electrode = bottom 50 mm of footing, ≥600 mm below grade** (red item).
- **Grounding-electrode conductor = #6 Cu / #4 Al** minimum (10-114). **Electrode conductor sized from Table 43, keyed to SERVICE-CONDUCTOR AMPACITY** (never the breaker) — 200 A / 4/0 service → **size 2**.
- Insulated grounding conductor in a raceway — **bare allowed only if ≤15 m AND ≤ two 90° bends (=180° total).**
- Ground the **identified (neutral)** on a 2-wire system, the **common (centre-tap)** on a 3-wire. Impedance-device conductor **#12 Cu / #10 Al** (10-318).
- Bonding: continuity; **metal raceway/sheath/armour bonded at BOTH ENDS** (explicit in 2024); locking fittings. **Bonding jumper = Table 16, sized by ampacity OR overcurrent device** (Table 43 is ampacity only). Equipotential/occupational bonding of non-electrical metal (water/gas pipe, conductive raised floor) = #6 Cu / #4 Al.

**Traps to master this week:**
- **Table 43 (electrode) is keyed to service-conductor ampacity; Table 16 (bonding) lets you use ampacity OR the OCPD** — don't swap them.
- The **180° bends** trap: watch the word "offset," add the degrees; combos just over 180° are non-compliant.
- **Impedance-device conductor #10 aluminum** is the commonly-asked answer (students wrongly pick #8).
- "Raised floor if conductive → bond it" (disguised as a hair-salon/aluminum-floor question — don't panic). Read whether they ask copper (#6) or aluminum (#4).
- Four-rod requirement is the **HV / Section 36** rule, not Section 10's two-rod rule.

**App sets:** `section10_grounding` (14). Small set — run it three times and aim for 100%; supplement with the trap log.

**Milestone:** Score **≥90% on `section10_grounding`** and recite cold: 6/4, 12/10, Table 43-vs-16, bottom-50-mm, both-ends.

---

## Week 4 — Wiring methods: Section 12, part 1 (the single biggest source of questions)
**Why now:** Section 12 is Block C, ~12 of ~30 wiring questions, and one of only three sections you must know by *content* (with 2 and 26). It's long, so it spans two weeks. This week = underground, cable selection, parallel/terminations, NMD.

**Focus topics:**
- Underground **Table 53 minimum cover** ("cover" = top of cable to grade, not trench depth); unarmoured/non-vehicular/240 V → 600 mm; HV → 1000 mm. **Reduction = 150 mm** with added mechanical protection.
- Direct-buried **sand bedding 75 mm BOTH above and below** (max particle 4.75 mm); run adjacent, don't cross.
- Roof-decking (12-022): cable must be **visible, not in the profile/flutes.**
- **Table 19** picks the cable TYPE (Section 4 only sized it): dry/damp/wet + temperature + exposure + mechanical, all AND. **AC90** = dry/(damp), NOT wet, raceway only, not service. **TECK90** = go-anywhere. **Table D1** = voltage rating & sizes (NMD90 = 300 V; AC90 ≤2000 V; 5000 V job → TECK90).
- **Parallel conductors (12-108): only #1/0 and larger**, six "same" conditions, no splices.
- **Terminations (12-118): #10 and smaller → binding screw; larger → solderless lug** (questions sit on #10). Aluminum → spring (conical/Belleville or helical) washer.
- **NMD (12-500s, twin 12-550s):** 300 V; heat clearances **25 mm (duct) / 50 mm (masonry chimney) / 150 mm (flue)**; support within **300 mm of a box then ≤1.5 m**; not embedded; **concealed through framing ≥32 mm from edge or protector plate.**

**Traps to master this week:**
- **"Both" in the sand-bedding rule** is the stolen word — "75 mm both above and below" is canonical.
- Cover reduction is a flat **150 mm** (plank 38 mm / concrete 50 mm / overhang 50 mm are the *protection* numbers — distractors swap 38/50/35).
- **AC90 on an outdoor wall or rooftop A/C (wet) is the worst possible choice** — AC90 is never wet.
- Termination/parallel questions **almost always sit on size #10 / #1/0** — the boundary.
- **32 mm or a protector plate** for concealed cable through a stud.

**App sets:** Begin `section12_wiring` (74 — the biggest set). This week run the **underground / cable-selection / NMD half** timed; carry the rest into Week 5.

**Milestone:** Score **≥80% on the first ~half of `section12_wiring`**, and from Table 19 + Table D1 correctly approve/reject AC90, NMD90, TECK90 for a given dry/damp/wet + voltage scenario.

---

## Week 5 — Wiring methods part 2 (raceways/boxes) + Section 14 protection + Section 6 services + Section 26/16
**Why now:** Finish Section 12's raceway machine, then sweep the remaining steady-count Block B/C lookup sections. These are where you bank easy marks to fund the motor block.

**Focus topics:**
- **Conduit fill machine — Tables 6 + 8 + 9:** Table 6 → conductor area (sum), Table 8 → fill % (**1→53%, 2→31%, 3+→40%** — memorize 40%; not-lead-sheathed = first row), Table 9 → trade size (9G/9H = 40% column shortcut). **Max 200 conductors (12-910). Smallest size = 16. No splice in a raceway (12-902). Max 4×90° = 360° bends (12-936).**
- Raceway "type" template (repeats per raceway): max conductors 200, bonding continuity (non-metallic → separate bond conductor), temperature limit, expansion joint. **PVC 75 °C / RTRC 100 °C / ENT 75 °C** (PVC & ENT: conductors >75 °C OK but read the **90 °C column** of Table 2/4). **PVC expansion joint when ΔL >45 mm; ΔL = length × ΔT × 0.0520.** Wireway 20% (40% signal/control). Surface raceway max 300 V. Solar raceway max conductor 1 AWG. **Reduction in size (12-2210): re-protect the smaller conductor unless reduced run ≤15 m.** Cable tray clearances 300/300/150/600 mm, bond span ≤15 m. Armoured-cable bend radius 6× (inner edge). **FCC under-carpet: prohibited in dwellings/outdoors; carpet squares ≤750 mm + release adhesive.**
- **Section 14 protection:** 14-100 where OCP is required; 14-104 OCP ≤ ampacity, next-size-up permitted (Table 13) ≤800 A; **small-conductor caps #14→15 A, #12→20 A, #10→30 A.**
- **Section 6 services:** max **1** supply / **4** consumer services; min conductor #10 Cu / #8 Al; mast metal, **min 63 trade size**; drip loop / 750 mm free conductor; **embedded = 50 mm** (the 90%-right fallback); service equipment NOT in coal bin/closet/bathroom; as close as possible to entry.
- **Section 26 (equipment/receptacles), Section 16 (Class 1/2):** receptacle wiring (split = break the tab, two phases), fence ≤1.8 m (26-304, via index "F"). These are lookup-heavy free marks.

**Traps to master this week:**
- **"200" is the answer** for max conductors in a raceway (students pick 10/12).
- **ENT carries the 75 °C limit more often than PVC** — read which raceway.
- Calculator entry of **0.0520** — don't drop the leading zero/decimal.
- Bends: add actual degrees ("equivalent of four 90°"); 3×90 + 3×30 = 360 = max.
- **Embedded = 50 mm**: when stuck and an answer says "embedded in 50 mm of concrete," it's right ~90% of the time.

**App sets:** Finish `section12_wiring` (74), then `section14_protection` (11), `section06_services` (12), `section26_equipment` (53), `section16_class` (15). Heavy week — prioritize 12 and 26 by question volume.

**Milestone:** Complete a conduit-fill calc (6→8→9) end-to-end, score **≥85% on `section12_wiring`** cumulatively and **≥80% on `section26_equipment`**, and have all Block A/B/C app sets cleared at least once.

---

## Week 6 — Section 28 motors (Block D damage control) + full timed mocks + polish
**Why last:** Motors are the hardest block and the most-failed. You don't need 100% here — you need enough (~80%) plus a locked-down Blocks B+C to carry you. The back half of the week is full timed mocks under clean-book conditions.

**Focus topics (Mon–Wed):**
- **The core motor distinction:** **conductors & branch OCP use Table 44 (3-φ) / Table 45 (1-φ) FLA**, sized at **≥125% of FLA (28-106). Overload uses NAMEPLATE current** — 125% if SF <1.15/unknown, 115% if SF ≥1.15 (28-300/306). **Never use Table 44 for overloads; never use nameplate for conductors.**
- Branch short-circuit/ground-fault protection up to **250%** of FLA (time-delay fuse / inverse-time breaker). Feeder OCP (28-200) = largest single motor branch OCP + sum of other FLAs.
- Overcurrent (protects wires) vs overload (protects the motor) — both installed on a motor. DC motors = **Table D2**, not the big book's tables (memorize; the small book isn't supplied).
- **Eliminate-the-wrong-answer** method: ~7–8 questions (especially open-ended "why won't the motor start?") are won by knocking out impossible options.

**Traps to master this week:**
- **Nameplate FLA vs Table 44 FLA** — the named "fail point." Overloads = nameplate; conductors/branch OCP = Table 44/45.
- FLA-vs-FLC wording; service-factor 115% vs 125% boundary.

**App set:** `section28_motors` (44). Run it Mon, review every miss Tue, re-run misses Wed.

### Week-6 timed-mock schedule (Thu–exam eve)
Assemble mocks by pulling a blueprint-weighted spread across the app sets (favour Blocks B+C). Strict conditions: **clean code book only, your cheat-map closed, ~2 min/question, 3-pass method, full silence, phone away.**

- **Thursday — Mock 1 (60 Q, ~2 h, all sections):** find your weak blocks. Review every wrong answer the same evening; log the trap.
- **Friday — Targeted repair (½ day):** re-study only the topics you missed in Mock 1; re-quiz those specific app sets.
- **Saturday — Mock 2 (100 Q, full 5 h simulated):** full dress rehearsal, run the 3-pass plan and the navigation routine for real. Evening: review.
- **Sunday — Mock 3 (100 Q) OR a focused 60-Q retest of remaining weak spots**, plus re-run your entire trap log. Confirm pace: easy/recall items in <60 s, calcs in the time you banked.
- **Exam eve:** NO new material. Skim the memorize-cold list and the trap log only. Pack: original code book, government ID, exam confirmation. Sleep.

**Milestone:** Two full 100-question mocks at **≥80% overall with Blocks B+C ≥90%**, motor set **≥75%**, and every memorize-cold number recalled without the book.

---

## Fast code-navigation routine (drill this daily — your edge on a clean book)
Run this exact chain on every lookup until it's muscle memory; target **<60–90 s** to the answer:

1. **INDEX first (back of book).** Look up the **exact term the question uses**. Multiple hits → read each candidate's **Scope** to pick the right section. (Index alone saves ~4–5 questions.)
2. **SECTION 0 definitions.** Check any term that controls the answer (dwelling unit, continuous load, ampacity, ungrounded, identified) — the definition can flip the rule.
3. **The rule — read the WHOLE rule.** All subrules and every **modifier word: except / unless / provided that / not exceeding / both / equivalent.** Follow every cross-reference before deciding.
4. **APPENDIX B.** If the rule is dense or the number isn't in the body (e.g. gas clearances 2-328, cover-reduction picture, expansion-joint formula), the worked note/figure is here — ~6–7 questions live in Appendix B.
5. **TABLES.** ~20% of the exam is table lookups. Know *which* table and the column logic; apply correction factors in the order the rule specifies. Don't memorize the body of a table — memorize where it lives and how to read it.

Also pre-memorize *where* your highest-frequency tables sit so you flip there blind: Tables 1–4 + 5A/5C (Section 4), Table 14 (Section 8), Tables 16 & 43 (Section 10), Tables 6/8/9 + 19 + 53 (Section 12), Table 13 (Section 14), Tables 44/45 (Section 28), Table 56/65 (Section 2), and the front-matter trade-size conversion.

---

## One-page "memorize-cold numbers" (recall without the book)
**Pass/format:** 70% pass; 100 Q; 5 h; Blocks B+C ≈ 60 Q decide it.

**Voltage classes / phase:** ELV ≤30 V (30 inclusive); LV >30–1000 V (1000 inclusive); HV >1000 V. 1:2 ratio = single-phase (120/240); ×√3 (≈1.732) = three-phase (120/208, 347/600). Bigger number = L-L, smaller = L-G.

**Section 2:** working space front 1 m (Table 56, by L-G voltage); headroom 2.2 m; transformer working space 1 m only >50 kVA; dwelling 150 V-to-ground (exception >250 kVA + resident electrician → 600/347).

**Section 4:** #14 Cu = 15/20/25 A (60/75/90 °C). T1 free-air Cu, T2 raceway Cu, T3 free-air Al, T4 raceway Al. Ambient base 30 °C (Table 5A, round up); grouping base 3 (Table 5C). Rule 4-006: terminal temp sets size; unmarked 60 °C ≤100 A, 75 °C >100 A; exactly 100 A → 60 °C.

**Section 8:** basic 5000 W first 90 m² + 1000 W/90 m²; min service >80 m² → 24 000 W/100 A, ≤80 m² → 14 400 W/60 A; range service 6 kW / dwelling branch 8 kW (+40% over 12 kW); heating 100% first 10 kW + 75% rest; gas furnace not counted; interlock = larger; VD 3% branch / 3% feeder / 5% total; show window 650 W/m; 12 outlets max / 15 A.

**Section 6:** 1 supply / 4 consumer services; min #10 Cu / #8 Al; mast metal min 63 trade size; free conductor 750 mm; embedded = 50 mm.

**Section 10:** grounding/equipotential conductor #6 Cu / #4 Al; impedance device #12 Cu / #10 Al; concrete-encased electrode bottom 50 mm, 600 mm below grade; bare grounding in raceway only if ≤15 m & ≤180° bends; bond metal raceway/armour both ends; Table 43 = service ampacity, Table 16 = ampacity or OCPD.

**Section 12:** max 200 conductors/raceway; smallest size 16; max bends 4×90°=360°; no splice in raceway; fill % 53/31/40 (1/2/3+); cover reduction 150 mm (plank 38 / concrete 50 / overhang 50); sand 75 mm both above & below; NMD 300 V; heat clearances 25/50/150 mm; support 300 mm then ≤1.5 m; concealed 32 mm or plate; parallel only #1/0+; termination boundary #10; PVC/ENT 75 °C (read 90 °C column), RTRC 100 °C; expansion joint >45 mm (ΔL = L×ΔT×0.0520); wireway 20%/40%; surface raceway 300 V; solar raceway max 1 AWG; reduction-in-size exception 15 m; cable tray 300/300/150/600 & bond ≤15 m; armoured bend 6×; FCC carpet squares ≤750 mm.

**Section 14:** OCP ≤ ampacity, next size up (Table 13) ≤800 A; caps #14→15 A, #12→20 A, #10→30 A.

**Section 28:** conductors/branch OCP = Table 44 (3-φ)/45 (1-φ) FLA × 125% (28-106); overload = NAMEPLATE × 125% (SF<1.15) or 115% (SF≥1.15); branch SC/GF up to 250% FLA; feeder = largest motor branch OCP + Σ other FLAs.

**Cross-standard:** emergency lighting / smoke alarms / exit-sign colour = NBC; their **conductor sizing & OCP = CEC.**

---

## Exam-day tactics
**Three-pass time management (5 h, 100 Q):**
- **Pass 1 — sweep (target ~90 min):** answer every definition, single-rule lookup, and obvious table question. **Flag** anything needing a long calc or deep hunt and move on. Never burn 6 minutes on one question early. Bank time on the easy ~half (aim <60 s each) to fund the calculation block.
- **Pass 2 — calculations & deep hunts:** work every flagged item with the time you banked. Use the navigation routine; follow modifier words and cross-references; verify the candidate answer against the actual rule/table, don't trust memory.
- **Pass 3 — review:** revisit every flag and every guess; re-check table columns, unit conversions, and √3 divisions. Confirm you applied 6 kW vs 8 kW range, terminal-temp limits, and connected-vs-calculated load correctly.

**Answer everything — no penalty for guessing.** Never leave a blank at time-out. If you must guess: eliminate the clearly-wrong distractors first (this alone wins ~7–8 questions, especially motor/open-ended items); when truly stuck and "embedded in 50 mm" or "200 conductors" appears, it's the high-probability pick.

**Mechanics:** Read the *exact* wording — the word that flips the answer is usually except/unless/provided/not exceeding/both/equivalent. Convert to line-to-ground *before* entering Table 56. Watch the 0.0520 calculator entry. You have 5 hours and it's generous — don't rush, but watch the clock; budget by pass, not by question.
</content>
</invoke>
