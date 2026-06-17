# 🎯 High-Yield Exam Map — BC Construction Electrician (CEC 2024)

Synthesised from the instructor's 6 recorded classes (see `session-*.md` for full
detail). Purpose: tell you **where to spend energy** to pass 100%. Numbers flagged
*(verify)* were noisy in the audio — confirm against the printed 2024 book.

---

## 0. The big strategy (from Session 1)
- **Pass mark = 70%**, all multiple choice, ~100 questions, **5 hours guaranteed** (a 6th hour is discretionary).
- **Blocks B + C (wiring, raceways, services, devices ≈ 60 questions) DECIDE pass/fail — aim ≥90% there.** Block D (motors) is the hardest; Block A/E are lighter.
- The exam is **open-book but the book is CLEAN (no tabs)** → the real skill is **fast navigation**: the **Index** saves ~4–5 questions, **Tables ≈ 20%** of the exam (easy lookups), **Appendix B ≈ 6–7 questions** ("the note of the day").
- Mindset: *don't memorize answers — know where to find them, and read the exact wording.*

---

## 1. 🔴 "Red-flag" items the teacher says are asked DIRECTLY / often
1. **Bonding definition** — memorize the *complete* sentence ("low-impedance path, permanently established, to connect non-current-carrying metal parts"). Trap = a version with a word dropped.
2. **Overcurrent vs Overload** — overcurrent (fuse/breaker) protects **wires**; overload (thermal) protects the **motor**. Most-confused pair.
3. **Feeder vs Branch-circuit** definitions — feeder = service box → branch OCPD; branch = final OCPD → outlet. Trips up ~90% of electricians.
4. **150 V to ground in a dwelling** — "very common AND very important." Exception: building >250 kVA + resident electrician → 600/347 V.
5. **Rule 4-006 terminal-temperature limitation** — expect 2–3 questions. Size on the **marked terminal temp** column, not the higher insulation rating.

---

## 2. ⚠️ Classic traps (lose-the-mark-if-you-rush)
- **Phase ID by ratio:** 1 : 2 → single-phase (120/240); 1 : √3 → three-phase (120/208). Decides whether you ÷√3 in current calcs. Don't confuse 208 with 240.
- **Voltage-class boundaries are inclusive:** **30 V = Extra-Low** (not Low); **1000 V = Low** (not High). "Not exceeding X" includes X.
- **Unmarked terminal temp:** 60 °C if ≤100 A (≤#1 AWG), 75 °C if >100 A — at *exactly* 100 A use 60 °C.
- **Two temperatures in one conduit:** rate each conductor on the **lower** temp, compute **separately** (never add). e.g. #4 → 68 A, #1 → 104 A.
- **Section 4 table grid:** Free-air Cu = **T1**, Raceway Cu = **T2**, Free-air Al = **T3**, Raceway Al = **T4**. Exam loves the rare combo (free-air aluminum).
- **Two correction factors:** grouping **Table 5C** (only >3 conductors), ambient **Table 5A** (only >30 °C) — multiply both, round ambient temp UP. Free-air grouping only if spacing < 25% of largest diameter; if spacing not stated, **don't derate**.
- **Range demand base:** **service/feeder = 6 kW**, **dwelling branch = 8 kW** (both +40% over 12 kW). 16 kW range → 7.6 kW (service) vs 9.6 kW (branch).
- **Table 56 working space uses LINE-TO-GROUND voltage** — convert first (÷√3 for 3-φ); don't re-divide an already-converted value.
- **Electrode conductor = Table 43 keyed to SERVICE-CONDUCTOR AMPACITY** (never the breaker). 200 A / 4/0 service → **size 2**.
- **Bonding jumper = Table 16, ampacity OR overcurrent device** (different from Table 43, which is ampacity only).
- **Bare grounding conductor in raceway** only if **≤15 m AND ≤ two 90° bends (≈180° total)** — watch "offset" wording, add the degrees.
- **Sand bedding = 75 mm BOTH above and below** (dropped word "both" is the trap).
- **Cover reduction = 150 mm** with mechanical protection — don't confuse with the 38 mm plank / 50 mm concrete thickness.
- **EMT is NOT a conduit** ("other than electrical metallic tubing").
- **Emergency lighting / smoke alarms:** *requirement* is in the **NBC**, *electrical sizing* in the **CEC** — read which the question asks.
- **Operating room load = 20 W/m² + 100 W** (the 100 W is additive, NOT per room) — withdrawn "unfair" question that may return.
- **Gas-meter clearance is in Appendix B**, not the rule body: NG 1 m, propane 3 m, 0.3 m with listed regulator.

---

## 3. Numbers worth memorizing cold
**Section 8 (loads):** basic load 5000 W first 90 m² + 1000 W/additional 90 m²; min service **>80 m² → 24 000 W / 100 A**, **≤80 m² → 14 400 W / 60 A** (take larger); heating 100% first 10 kW + 75% remainder (baseboard), electric furnace 100%, **gas furnace not counted**; interlock (8-106) = larger of heat/AC; voltage drop **3% branch / 3% feeder / 5% total** (the #1 2021→2024 change); show window **650 W/m**; **12 outlets max on a 15 A circuit**.

**Section 6 (services):** max **1** supply service / **4** consumer services; min conductor **#10 Cu / #8 Al**; service mast metal, **min 63 trade size**; drip loop **750 mm**; "embedded = **50 mm**" of concrete (high-probability answer when unsure).

**Section 10 (grounding):** concrete-encased electrode in **bottom 50 mm** of footing, **600 mm** below grade; electrode spacing 3 m *(verify)*; impedance-device conductor **#12 Cu / #10 Al** (10-318).

**Section 12 (wiring):** NMD90 = **300 V**, AC90 ≤ **2000 V**, TECK90 ≤ **5000 V**; parallel only **#1/0+** (12-108); NMD heat clearances **25 / 50 / 150 mm**; concealed cable **32 mm** from stud edge (or protector plate); support **within 300 mm** of a box then **≤1.5 m**; roof clearance **2.5 m** walkable / **1 m** non-walkable / **4.5 m** max span / **2 m** deviation floor; **extra-hard usage** = PV (Sec 64) & temporary construction wiring.

**Section 2:** working-space front **1 m** (Table 56, by L-G voltage); headroom **2.2 m**; transformer working space **1 m only >50 kVA** (exactly 50 = none); enclosures Table 65 (4X + 6P = splash + corrosive).

---

## 4. Where the questions live (study weighting)
| Priority | Sections | Why |
|---|---|---|
| 🔥 Highest | **12** (wiring), **8** (loads), **4** (ampacity), **10** (grounding) | Blocks B+C, ≈60 Q, decide pass/fail |
| High | **2** (general/working space), **6** (services), **26** (equipment), **14** (protection) | Steady question count, many easy lookups |
| Hardest | **28** (motors) | Block D — drill the FLA-vs-FLC overload trap |
| Lighter | **16** (Class 1/2), **0** (definitions) | Fewer Q, but definitions are free marks |

> Bottom line: **own Sections 4, 8, 10, 12 and your book navigation** and you pass.
