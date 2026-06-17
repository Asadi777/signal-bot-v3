# Session 5 — Study Notes (BC Construction Electrician, CEC 2024)

Topic coverage: completes **Section 10 — Grounding & Bonding**, then begins **Section 12 — Wiring Methods**.

These notes are written for the student aiming to pass at 100%. The source class was taught in Persian; mangled/spoken English terms have been interpreted back to their correct CEC terminology. Numbers flagged "(approx — audio noisy)" should be re-confirmed against the actual code book.

---

## High-yield: important / tested / traps / common mistakes

### Section 10 — Grounding electrodes
- **Concrete-encasement 50 mm rule is a flagged exam question (instructor called it "red"/red-hot).** When a grounding electrode (plate, or the 6 m bare field-assembled conductor) is placed in concrete, it must be **within the bottom 50 mm** of the concrete foundation footing, in direct contact with earth, **at not less than 600 mm** below finished grade. The 50 mm is the most-tested number — memorize it exactly.
- **Rod electrode count has no voltage reference in Section 10 — this is a known code quirk the instructor flagged.** Section 10 says "minimum two rods" without stating a voltage; Section 36 (high voltage) requires a minimum of **four** rods. Don't get tripped expecting Section 10 to mention voltage.
- **Electrode-conductor sizing via Table 43 must be keyed to the ampacity of the largest service conductor — NOT to the breaker rating and NOT to the motor/load current.** This is the #1 trap in this topic. Students wrongly read off the main breaker (e.g. 200 A) or off "maximum current." You must take the **ampacity of the service conductor** at its rating temperature (75°C for buildings), then enter Table 43.
- **Worked trap (instructor's own example):** 200 A main breaker, service conductor 4/0. If you wrongly use the 200 A breaker you may land on size 3 (because 200 sits at a range boundary). The correct path: 4/0 copper at 75°C → its **ampacity** → Table 43 → answer is **size 2**. The instructor stressed: "in 90% of cases right vs wrong placement gives the same answer, but a tricky examiner can pick a value right at a range boundary so the wrong method gives size 3." Always look the value up precisely in the table, do not eyeball.
- **Insulated-vs-bare grounding conductor exception — the "equivalent to two 90° bends" wording is a deliberate trap.** Code default: when the grounding conductor runs in the **same raceway** as service conductors it must be **insulated**. Exception (bare/uninsulated permitted): raceway length **≤ 15 m** AND **not more than the equivalent of two 90° bends**. "Equivalent" means total ≈ 180°, so it can be expressed as 4 × 45°, 6 × 30°, or an offset of 3 × 15°, etc. Exam answers list combinations like "2 × 90° + 3 × 15° offset" (that exceeds 180° → fails) or "1 × 90° + 3 × 30°" (= 180° → passes). Add the degrees; the limit is two-90°-equivalent (≈180°).
- **Impedance grounding device conductor size — Rule 10-318 — usually asked about aluminum.** Conductor used with an impedance grounding device: **no smaller than #12 copper or #10 aluminum**. Students answer "#8" by mistake. The aluminum answer (#10) is the one that comes up. (Contrast with the normal grounding conductor minimum of #6 Cu / #4 Al — the impedance device is allowed smaller because the impedance device limits the fault current.)
- **Multiple grounding electrodes in one building: separate by at least 2 m and interconnect them again with the grounding conductor.** "In a building" means the same building (renovation/addition), not a neighbouring building.
- **In-situ / existing-structure electrode** (metal column on a base plate, or a metal water pipe): qualifies as an electrode if it gives the required contact and is buried at least 600 mm below grade. Water pipe also needs ≥ 3 m length in contact (instructor said "3 m" — approx, audio noisy). Modern pipes are non-metallic, so this is rare but still testable.

### Section 10 — Solidly vs impedance grounded; which conductor to ground
- **Two simple, near-guaranteed questions: (1) which conductor do you ground, and (2) impedance-device conductor size.**
- **Solidly grounded system — which conductor to ground:**
  - **DC, 2-wire OR AC, 2-wire:** ground **one conductor** (the identified/neutral conductor — you would never ground the phase/live conductor).
  - **3-wire (AC single-phase 3-wire, or DC 3-wire):** ground the **common conductor** — the centre/mid conductor (the centre tap of the transformer, the point where voltage is zero). Cited as **Rule 10-208(1)(b)** "the mid (neutral) of single-phase 3-wire."
- **Bonding minimums (don't confuse with electrode conductor):** grounding conductor ≥ **#6 Cu / #4 Al** (Rule **10-114**). Impedance device conductor ≥ **#12 Cu / #10 Al** (Rule **10-318**).

### Section 10 — Bonding
- **"Both ends" (bond at both ends) is an exam answer that is NOT spelled out as the phrase in the code — you must know it.** When a metal raceway / metal sheath / cable armour is used, bond it at **both ends** (Rule cited as 10-... (2)(a): "at both ends, where the metal raceway, metal sheath, or cable armour..."). The instructor said this exact question appeared on a student exam two weeks prior and the answer was "both ends," yet the literal phrase is hard to find in the book.
- **Continuity of bonding must be maintained — no breaks.** Where a raceway/cable tray run could open up (e.g. thermal expansion gaps between three 10 m cable-tray sections over a 30 m run), install a **bonding jumper** across the gap so continuity is preserved.
- **Use locknuts (standard locknuts) for bonding fittings** (Rule cited as Table-2 / 10-... (2): "standard locknuts shall be used for bonding"). Screws must be the non-loosening kind ("lock nut" = locking nut), so vibration (e.g. bridges, which are all bolted with locking nuts, not welded) cannot loosen the bond.
- **Bonding-jumper / bonding-conductor size — Table 16 — and you may size by EITHER ampacity OR the overcurrent device.** This is different from the electrode conductor (Table 43), where you are forced to use ampacity only and forbidden from using the OCPD. For the bonding jumper, Table 16 lets you use **either the ampacity of the conductor OR the rating of the overcurrent device.**
  - The jumper/bond can be a **wire** or a **busbar (bar/strip)**.
  - **Worked example:** a 30 A circuit, using a **copper busbar** → required cross-sectional area **3.5 mm²** (read the bar's cross-section from Table 16). If a wire instead, Table 16 gives the AWG/kcmil size. Students mis-answered "4.5" — read the row precisely.
- **Equipotential bonding (Section 10, Rule series ~10-700s, "Bonding of non-electrical equipment") — last page of the code section; very likely tested.**
  - Purpose: bond **non-electrical equipment** so a fault can't put you at a different potential. "Bonding = forming a group" — everything tied to one terminal sits at the same potential, so there is no voltage difference to shock you (like a bird on a wire: both feet at the same potential).
  - **Equipotential bonding conductor minimum size: #6 Cu / #4 Al.** Trap: if the question says "the floor/surface is aluminum," that does NOT automatically mean you answer aluminum size — read whether it asks for the **copper** answer (#6) or the **aluminum** answer (#4). Aluminum → #4; copper → #6.
  - **Reduced size exception:** if the conductor is **concealed AND has adequate mechanical protection**, it may be **#10 Cu / #8 Al**. (Rarely tested, but know it. The reason cables are oversized is often mechanical protection, not fault current.)
  - **Examples of non-electrical equipment to equipotential-bond (these appear as questions):** metal water pipe, metal sewage/waste pipe, metal gas pipe, **raised floor** (conductive raised access floor), conductive metal piping, fences (e.g. a metal fence around equipment). **"Raised floor" was specifically a recent exam question** — a conductive raised access floor (e.g. an aluminum-surfaced floor in a media room or even framed as a "hair salon with aluminum floor" to scare students). Don't be thrown by the setting; the rule is just: conductive raised floor → bond it, conductor ≥ #6 Cu / #4 Al.

### Section 12 — Underground / Table 53
- **Table 53 gives MINIMUM COVER (in millimetres) for direct-buried cable or cable in raceway.** Definition trap: **"cover" = distance from the TOP surface of the cable/raceway up to FINISHED GRADE.** It is NOT the trench depth and NOT measured to the conductor centre.
- **150 mm cover-reduction is a freshly-tested number (two students just got it).** You may reduce the Table 53 cover requirement by **150 mm** where additional mechanical protection is installed in the trench over the underground installation. Allowed protections (one of):
  - **Treated wooden planking** at least **38 mm** thick, extending at least **50 mm** beyond each side of the cable/raceway.
  - **Concrete slab/block** at least **50 mm** thick, extending at least **50 mm** beyond.
  - **Poured concrete** at least **50 mm** thick.
  - **Trap:** all three "extend beyond" values are **50 mm**, but the thicknesses differ — planking **38 mm** vs concrete **50 mm**. Examiners give wrong options like "all 35 mm" or "all 38 mm." Memorize: planking 38, concrete 50; side-extension always 50; reduction always 150.
- **Sand bedding "both above AND below" — the word "both" is the trap, and it is a known two-correct-answer question.** For direct-buried cable (not in raceway), use screened sand (sieved sand, max particle size **4.75 mm**), at least **75 mm deep both above and below** the conductor. The exam strips the word "both": "75 mm above and below" reads as 75 mm total split between top and bottom — that is **wrong**. The code says **75 mm above AND 75 mm below** (so effectively 75 around). A second phrasing using "150 mm for the cable diameter, sand above and below" can also be a correct answer because it counts above + cable + below. Read the wording word-by-word.
- **Direct-buried cables must be laid adjacent / run parallel, must NOT cross over each other** (Rule cited Subrule 4 → your Subrule 5), then covered with the screened-sand layer. Crossing (especially armoured cables with soil between) can damage cable.
- **Table 53 in CEC 2024 dropped the voltage columns.** Older books split by extra-low/low voltage; 2024 merged and changed it: armoured-vs-unarmoured AND in-raceway use the **same number**, differentiated instead by **non-vehicular vs vehicular areas** (vehicle traffic → deeper cover). High voltage rows are all the **same number** regardless of armour. (Instructor: HV armour/no-armour all "1000 mm / 1 m" — approx, audio noisy.)
- **Worked example:** cable, no armour, non-vehicular area, e.g. 120 V → cover **600 mm** (approx — confirm in Table 53).

### Exam-strategy points the instructor hammered
- **The #1 cause of failing is "I think I know it, so I don't open the book."** A capable student failed three times by trusting memory and bubbling answers without checking the code. Always open the code and verify, even for "easy" questions.
- **Tables are where you lose marks.** Every table to the end of the book can be presented so a first glance misleads you. Write each section's relevant table numbers on that section's first page (Section 10: **Table 43** and **Table 16**). Read table ranges precisely.
- **Knowledge fades — re-study before the exam.** Several students "knew it" six months earlier, didn't review, and missed guaranteed questions.
- **Don't chase the exam, chase understanding** — but the instructor will give a set of guaranteed-on-the-exam questions; those plus real understanding pass you.

---

## Content taught (in order, full detail)

### 0. Admin / framing
- Previous session ended at the end of Section 8. This session does Section 10 (grounding & bonding) and starts Section 12.
- A solved photovoltaic-voltage-drop (Section 64 area) problem from Section 10 prep was converted to PowerPoint and will be sent to the student; voltage-drop questions come up.
- The **"small book" (handbook / pocket guide)**: the student doesn't have it. The instructor will send the PDF. Key point: the small book is **not provided at the exam** (questions are drawn from it, but it isn't supplied), whereas the **large code book IS provided at the exam**. So owning the small book isn't essential — the value of having a physical book is to get fast at flipping pages and become familiar before exam day. Roughly 6–10 questions may come from small-book material.
- Grounding & bonding were introduced conceptually back in Section 0 (what they do). This session is about **how to execute** grounding and bonding.

### 1. Grounding vs Bonding are two separate topics
- The instructor teaches **grounding** and **bonding** as two completely separate discussions (in Iran they are lumped as "system earthing"; here they are distinct).
- Grounding/bonding here refers to the run from the panel/load down to earth (electrode) and through the equipment.

### 2. Three classes of grounding electrode (Section 10)
The code divides grounding electrodes into three classes. (The instructor notes there are really four states, but the book groups them as three classes.)

**Class 1 — Manufactured electrode** ("manufactured" = factory-made; you buy it ready and install it). Two forms:
- **(a) Rod type** — a solid round rod ("solid bar," like rebar/round stock) driven into the ground.
  - Rule: **minimum two rods.**
  - Spacing: the rods must be **at least [3 m] apart** (instructor said "samt" → 3 m; approx — confirm).
  - The two rods are interconnected by a wire (grounding conductor) under a clamp, then run to the panel and connected.
  - Length: each must be **fully driven** ("fully left driven") so the top sits flush with grade.
  - Quirk: Section 10 gives no voltage; Section 36 (HV) needs four rods.
  - Practical note: driving rods is hard — the first metre goes easy, the last metre is very hard (needs a hammer drill). Because of this, plates are often used instead.
- **(b) Plate type** — a plate electrode.
  - **Minimum surface area 0.2 m²** (two-tenths of a square metre).
  - The plate is dense/heavy for its size.
  - It must be buried at least **600 mm** below grade (or be concrete-encased, see below).
  - The conductor running from the plate to the panel: **minimum #6 Cu or #4 Al** (this is the electrode/grounding conductor minimum that applies to all three classes).
- (There is also a **chemical** electrode form listed in the code — instructor said you don't need to read/learn it.)

**Class 2 — Field-assembled electrode** (you're in a remote area / no store access, so you assemble it on site):
- Take a **bare copper conductor** (a bare/uninsulated copper wire).
- Size **per Table 43** (the same table used for the electrode conductor).
- Length **not less than 6 m** (instructor sometimes mis-said "6 mm" — it is **6 metres**).
- Buried with at least **600 mm** cover, OR concrete-encased.
- That 6 m bare wire performs the same electrode function as a rod/plate.
- The run-conductor back to the panel is still **min #6 Cu / #4 Al**, but its **size is determined by Table 43** (you cannot just default to #6 Cu / #4 Al — Table 43 may require larger).
- In practice this is usually concrete-encased, because electricians won't dig 6 m of trench by hand.

**Class 3 — In-situ electrode** (you neither buy nor add anything; you use **part of the existing structure / "existing infrastructure"**):
- Use an existing metal element — e.g. a **metal column** sitting on a metal **base plate**; the base plate can serve as the electrode.
- Qualifies if it gives the required contact/surface and is at the required depth, **600 mm** below grade.
- Code example: even a **metal water pipe** can be the electrode if it is in contact for a length of **[3 m]** and at least **600 mm** below grade. (Length value approx — audio.) Problem today: most pipes are non-metallic, so this is rare — but if seen, it still qualifies.
- The run-conductor is again min #6 Cu / #4 Al to the panel.

### 3. The code's own classification (matches the above)
The code lists grounding electrode types as: **manufactured** (rod and plate — full driven, etc.), **field-assembled** (the bare copper conductor), and **in-situ** ("use of an electrode that is part of existing infrastructure/structure"). The instructor praised this section as clean and well-organized: "if you fail here it's your own fault."

Code wording detail for rod (manufactured):
- "In the case of a rod, consists of two rod electrodes, spacing not less than [3 m], interconnected by the grounding conductor, fully driven."
- "In the case of a plate, in direct contact with the earth ('dark contact with external soil'), depth not less than 600 mm."

### 4. Concrete-encasement rule (the 50 mm / 600 mm rule) — heavily tested
When you place the plate electrode (or the 6 m field-assembled bare conductor) in concrete during a foundation pour:
- It must be within the **bottom 50 mm** of the concrete foundation footing.
- That bottom must be in **direct contact with the earth**.
- And **not less than 600 mm** below finished grade.

Reasoning: if the electrode sat in the middle of the concrete, it would have no earth contact. Up to 50 mm of concrete still counts as effectively in contact with earth (a low-resistance path exists); beyond 50 mm it loses the grounding property.

Code wording: "Encasement within the bottom 50 mm of concrete foundation footing in direct contact with the earth at not less than 600 mm [below finished grade]." Both numbers reappear in the code: the **600 mm** to grade, and the **50 mm** to the bottom of the concrete. **The 50 mm is an exam question (flagged red).**

Practical: when a building is being built and the foundation is about to be poured, the electrician tells the concrete crew to let them place the electrode in the footing — saves digging the 600 mm separately because the excavation already exists.

### 5. Electrode-conductor sizing via Table 43 (keyed to service-conductor ampacity)
- The run-conductor (from electrode to panel) min is **#6 Cu / #4 Al**, but **size by Table 43.**
- Table 43 gives conductor sizes **4, 3, 2, 1, 1/0, 2/0, 3/0** keyed to the **ampacity of the largest service conductor.**
  - Ampacity **≤ 165 A → size 4**
  - **166–200 A → size 3**
  - and so on up the table in order.
- **Critical method (worked):** Suppose a panel has a 200 A main breaker fed by a **4/0** service conductor.
  - Service conductors are normally taken at **75°C** for buildings (breakers, lugs, etc. are 75°C-rated). (Note: 2018/2021 books muddled this; the 2015 book had a table stating buildings use 75°C.)
  - You must NOT use the breaker rating, and you must NOT use "maximum current." You use the **ampacity of the service conductor** (4/0 at 75°C).
  - Look up 4/0 ampacity → enter Table 43 → answer **size 2** (NOT size 3, which is what you'd wrongly get from 200).
  - "We never said take maximum current; we said take the ampacity of the service [conductor]."
- 90% of the time, right vs wrong method lands on the same answer; a sharp examiner picks a boundary value so the wrong method gives an adjacent size. Always read the table precisely.

### 6. Insulated-vs-bare grounding conductor in a raceway (≤15 m / two-90°-bend exception)
- The grounding conductor is usually run **bare** in practice (you see a bare wire used as the grounding conductor), but the **code says it must be insulated** when run in the same raceway as the service conductors.
- Reason for insulation: in a bend or over a long run, a bare grounding conductor can scrape/strip the insulation off the adjacent service conductor (or be damaged itself when you pull the service conductor past it).
- **Exception (bare/uninsulated permitted)** — "insulated grounding conductor shall be permitted [to be bare] if":
  - the raceway length is **not more than 15 m**, AND
  - it **does not contain more than the equivalent of two 90° bends.**
- Because electricians typically use a short 1–1.5–2 m conduit with no bends, they fall under the exception and use bare wire.
- **Trap on "equivalent to two 90° bends":** "equivalent" = total ≈ 180°. So acceptable equivalents: **4 × 45°**, **6 × 30°**, etc. Exam options:
  - "2 × 90° bends + 3 × 15° offset" → that's 180° + 45° = exceeds → **fails the exception.**
  - "1 × 90° + 3 × 30°" → 90° + 90° = 180° → **passes.**
  - The word **"offset"** may be used for the small-angle bends. Add the degrees; the cap is the two-90° (≈180°) equivalent.

### 7. Two grounding systems: Solidly grounded vs Impedance grounded
**Solidly grounded** (what we use in homes):
- The grounding conductor runs from panel to electrode and is connected **directly** by a clamp (most common; can also be welded/soldered) — **no device in between.**
- If an appliance (e.g. washing machine) faults to its body and you touch it, current flows through the low-resistance path into the electrode, not through you. If a live wire contacts the body, a very large current flows for an instant and the breaker trips — which is desirable (it removes the hazard). We welcome that current path.

**Impedance grounded:**
- Used where there is sensitive electronic equipment (medical facilities, ICs, transistors). A sudden large fault current can shock/burn the circuitry before the fuse can clear (current spikes faster than the fuse opens, and can damage multiple devices).
- An **impedance device** (a kind of "choke" — a capacitor/coil/reactor) is placed between the grounding conductor and the electrode. It **damps** the wild fault current so the fuse can clear it at a lower level without burning extra devices or shocking the equipment.
- Analogy used: an old kettle/heater turning on dimming the house lights — circuits interact; you can inject a shock into a system through the ground path. The impedance device prevents that.

### 8. Which conductor to ground (Solidly grounded)
- **DC 2-wire OR AC 2-wire:** ground **one conductor** — never the phase; ground the **identified (neutral) conductor.**
- **AC single-phase 3-wire OR DC 3-wire:** ground the **common conductor** — the **centre conductor** (transformer centre tap, where voltage = 0). Cited **Rule 10-208(1)(b)**: ground the mid (neutral) of single-phase 3-wire; the "common conductor" can serve both halves.

### 9. Impedance grounding device conductor size — Rule 10-318
- For a normal grounding conductor we required min #6 Cu / #4 Al. For an **impedance grounding device**, the conductor need **not** be that large, because the impedance device limits the fault current.
- **Rule 10-318:** "Conductor used with [an] impedance grounding device — in no case [smaller than] #12 copper or #10 aluminum."
- The **aluminum** answer (**#10**) is the one that's asked. Common wrong answer: #8. (One student bubbled "8" and failed that question; he knew the work but didn't open the book — failed three times for this habit, passed on the fourth.)

### 10. Bonding (taught from the CEC 2024 book — content changed and is now testable)
- Why 2024: a question used to have **no findable answer** in older books (2011, 2018, 2021); the 2024 book finally states it clearly. The same question still appears on current exams (two students two weeks prior).

**Continuity:**
- Continuity = the bonding system must have **no breaks**.
- Example: a panel feeding a motor through a conduit. The motor's metal body must be bonded. If the raceway is **non-metallic**, you must run a separate **bonding conductor** to the motor (Section 12 repeatedly reminds you of this for non-metallic raceways). If the raceway is **metallic**, the code doesn't require a separate bond wire, but you may add a **jumper** from the raceway to the motor body to be sure of continuity.
- Where a run could open (e.g. three 10 m **cable trays** over a 30 m run, with thermal expansion/contraction gaps in summer/winter): install a **bonding jumper** across the gap so the middle section doesn't end up un-bonded (which would also leave the motor un-bonded). Dangerous otherwise.
- Fittings/screws must be **standard locknuts** ("lock nut" = locking nut) that won't loosen. (Bridges use all-bolted locking nuts, not welds, because of constant flexing — analogy for why bonding hardware must not loosen.)

**Bond at both ends:**
- Where a metal raceway / metal sheath / cable armour is used, **bond at both ends.** (Even an armoured cable: bond its armour, at both ends.)
- Cited wording (Subrule 2(a)): "at both ends, where the metal raceway, metal sheath, or cable armour... at the bonding [means] between the [sheath]..."
- The literal phrase "both ends" is hard to find but **the exam answer is "both ends"** (confirmed appearing on a recent student exam).

### 11. Bonding-jumper / bonding-conductor size — Table 16
- Question form: "what size bonding jumper / bonding conductor?" → **Table 16.**
- The jumper/bond may be a **wire** or a **busbar (bar/strip).**
- **Key difference from Table 43:** for Table 43 you may use **only ampacity** (forbidden to use the overcurrent device). For **Table 16 you may use EITHER the ampacity of the conductor OR the rating of the overcurrent device.**
- The code sentence is awkwardly written ("minimum size of bonding conductor / jumper / bonding conductor...") — an electrician drafted it and an editor patched it; just take it as: find the minimum size for your jumper/bonding conductor from Table 16.
- **Worked example:** 30 A circuit, using a **copper busbar** → cross-section **3.5 mm²** (read the bar's cross-sectional area from Table 16). For a wire it gives the AWG/kcmil. Student wrong answer: "4.5." A "size 12" type answer would also be a distractor. The instructor calls this "a question God gives you to pass you" — very easy if you open the table.

### 12. Equipotential bonding — "Bonding of non-electrical equipment" (Section 10, ~Rule 10-700 series; last page of the section)
- This is **occupational/equipotential bonding of NON-electrical equipment** ("bonding of non-electrical equipment").
- **Concept / worked illustration:** An electric stove/range is bonded → bond runs to the panel → panel runs to the electrode in earth. On a fault to the stove body, current returns through the low-resistance bonding path (not through you), and the breaker trips.
  - **Danger scenario without equipotential bonding:** if you simultaneously touch the faulted stove and a metal water pipe (the pipe itself acts like an electrode), part of the current goes through the pipe to earth and part can go through **you** — unknown proportion — possibly lethal.
  - **Fix:** bond the metal water pipe (and gas pipe) into the bonding system. Now stove, water pipe, gas pipe all tie to **one terminal** → they form a "group" at the **same potential**. Touching the stove and the pipe is now safe because there's **no voltage difference** between them (both at, say, 120 V together, or both at 0). Like a bird standing with both feet on one wire — same potential, no shock.
- "**Equipotential**" = equal potential. Equipotential bonding makes everything the same potential/voltage so you can't be shocked across two surfaces. (Code references **Appendix B** in parentheses for more detail, but the useful explanation is right here in the rule.)
- **Items to equipotential-bond (each a possible exam question):**
  - metal water pipe
  - metal sewage/waste pipe
  - metal gas pipe
  - **raised floor** (conductive raised access floor)
  - conductive metal piping (the "fences / livestock-pen / equipment enclosure" type — "conductive metal parts")
  - metal fences around equipment
- **Raised floor explained (recent exam item):** a raised access floor is used where you must feed power/internet/USB cables up to many workstations (e.g. 20–30 reporters at a press event) from below. If the surface is wood, no shock risk; but if the surface is **conductive material** (e.g. aluminum), it behaves like non-electrical equipment to bond. Wiring runs beneath it could be chewed/damaged and energize the floor. So a conductive raised floor must be **equipotential-bonded.**
  - Exam dressing: "a raised floor in a hair salon with aluminum floor" — ignore the scary setting. Rule: conductive raised floor → bond, conductor min #6 Cu / #4 Al.
- **Equipotential bonding conductor size (final rule of the section):** minimum **#6 Cu / #4 Al.**
  - **Trap:** the question may say "the floor/surface is aluminum" but still ask for the **copper** conductor size. Read which metal is being **sized**: aluminum conductor → **#4**, copper conductor → **#6**.
  - **Reduced-size exception (Subrule 2):** if **concealed AND adequately mechanically protected** → may be **#10 Cu / #8 Al.** (Quick reference: **6 / 4** normal, **10 / 8** if concealed + mechanically protected.) Usually not on the exam, but know it. The reason conductors are oversized is often **mechanical protection**, not fault current — if the run is in a safe/protected location you may go smaller.

> **Section 10 is now complete.** The instructor will send fully solved Section 10 problems. (Class total time tally mentioned: through Section 4 ≈ 4:45; Sections 6/7/8 added ~2:00 → ~6:45; plus today's 145 min — admin only, ignore.)

---

### Section 12 — Wiring Methods (start)

**Organizing principle (how to navigate Section 12 and the whole book):**
- "Wiring Methods" is a long section (may spill into next session). Despite its length it covers **four things**:
  1. **Wiring methods proper** — how you run cable: on a wall, in a wall, underground, overhead — and the rules for each.
  2. **Conductors** — you identify/size the type of conductor here.
  3. **Raceways** — the types of raceways (conduit, etc.).
  4. **Boxes** — boxes (and box fill / "box max").
- If you know which of these four a question is about, you know where in Section 12 to look. ~30 questions come from "wiring methods" overall.

**"Wiring method" as an address (the instructor's key navigation trick):**
- Not all wiring-method questions live in Section 12. **Every section has its own "wiring methods" sub-part.** Use that to jump straight to the answer:
  - **Section 22 (corrosive/wet locations):** for areas with corrosion risk you won't use metal boxes — or if a metal box, it must be **corrosion-resistant**. Find it under Section 22's wiring-methods sub-part.
  - **Fire pump (e.g. Section 32, fire-risk areas):** can't use PVC where there's fire risk → go to fire-pump → its wiring-methods sub-part tells you to use **metal raceway / metallic**, or if non-metallic conduit then **encase in 50 mm** (concrete) of mechanical protection.
  - **Section 64 (renewable energy / photovoltaic):** for module/array wiring → go to Section 64 → photovoltaic → its wiring-methods sub-part → must use **flexible cord of the extra-hard usage type**, or **type RPV[U]** cable. (Some terms approx — audio.)
- Lesson: for "what conductor / raceway / box do I use in environment X," go to environment X's section, open its **wiring methods**, and you have the address — don't waste time scanning the whole section.

**Underground installation (Section 12) — Table 53 (cover):**
- Underground = running cable below grade, either **direct-buried ("direct burial")** or **in a raceway**. Either way, the **minimum cover** comes from **Table 53.**
- **"Minimum cover" defined:** distance from the **top surface** of the cable/raceway up to **finished grade** — NOT the trench depth, NOT to the conductor centre.
- **The weaker/less-protected the cable, the deeper the required cover.** Armoured cable → can be shallower; unarmoured → must be deeper.
- **Reading Table 53:**
  - All values are in **millimetres** (each number is the minimum cover in mm).
  - **No metal shield or armor ("not having a metal shield or armor")** → use one cover column.
  - **Armoured OR in a raceway** → use a different (shared) column — the number is the **same** whether armoured or in a raceway, so those two are drawn together.
  - The remaining split is **non-vehicular areas vs vehicular areas** — where vehicles pass, deeper cover (more traffic load → deeper).
- **2024 change:** Table 53 **dropped the voltage columns.** Older books split extra-low / low voltage; now it's merged. For **high voltage**, all rows are the **same number** regardless of armour (instructor: ~**1000 mm / 1 m** — approx, audio).
- **Worked example:** cable with **no armour**, **non-vehicular area**, ~120 V → cover **600 mm** (approx — confirm in Table 53).

**Cover-reduction by 150 mm (Section 12, Subrule ~3):**
- You may **reduce** the Table 53 minimum cover by **150 mm** where additional mechanical protection is installed **in the trench over the underground installation.** "Where mechanical protection is installed in the trench over the underground installation."
- Two recent student exam questions came from this 150 mm rule.
- The reduction can cover the whole trench or just the stretch that needs it (e.g. only the 4–5 m segment crossing over a water line you want to raise slightly).
- **Mechanical protection = one of the following:**
  - **Treated wooden planking** — thickness at least **38 mm**, and when in **flat form** "shall be wide enough to extend at least **50 mm** beyond" each side of the cable/raceway. (Treated = waterproofed so it won't rot/erode.)
  - **Concrete slab/block** — at least **50 mm** thick, and (flat form) extend at least **50 mm** beyond each side.
  - **Poured concrete** — at least **50 mm** thick.
  - For **flat-form** protections (planking and slab/block) the **50 mm side-extension** applies.
- **Trap:** all "extend beyond" values are **50 mm**, but thickness differs — **planking 38 mm**, **concrete 50 mm**. Examiners give all-same distractors (e.g. all 35, or all 38). A student who hadn't studied with the instructor wrongly said poured-concrete reduces by 50 mm — wrong: the **reduction is 150 mm**; the **50 mm is the concrete thickness condition.** Don't conflate the numbers.

**Sand bedding for direct-buried cable (Section 12, Subrule ~4 → your Subrule 5):**
- For cable installed **direct-buried** (not in raceway):
  - Cables "**shall be installed... run adjacent to each other and do not cross over each other**" — lay them parallel/adjacent, must **not overlap or cross**. (Crossing, especially armoured cables with soil between, damages cable.)
  - Cover with a layer of **screened sand** ("sand screen sand" = sieved/screened sand), **maximum particle size 4.75 mm.**
  - **"At least 75 mm deep both above and below the conductor."**
- **The "both" trap (a known two-correct-answer question):**
  - Wrong reading: "75 mm above and below" → interpreted as 75 mm total. The code says **75 mm above AND 75 mm below** (≈150 mm of sand spanning the cable).
  - Two answers can be correct: (A) "75 mm both above and below," and (B) "150 mm sand for the cable diameter (above + cable + below)" — because that 150 mm counts top sand + cable + bottom sand. A student with strong English flagged that two options were defensible, guessed, and passed at **83%**.
  - Lesson: these are **wording questions, not electrical questions.** One swapped/dropped word ("both") flips correct↔incorrect. Read word-by-word.

> Section 12 continues next session with **conductors** (the class ran ~145 min and stopped just before the conductors sub-topic). Next session: same week, Monday 10:00 AM.

---

## Mnemonics / exact phrasings

- **"50 to the bottom, 600 to the top."** Concrete-encased electrode: within the **bottom 50 mm** of the footing (earth contact), **600 mm** below finished grade. (The 50 mm is the "red" flagged exam number.)
- **"Section 10 = two rods, Section 36 = four rods."** Rod count, no voltage stated in Section 10.
- **Electrode conductor: "Table 43, by SERVICE-conductor AMPACITY — never the breaker, never max current."** (≤165 A → size 4; 166–200 A → size 3; ...). Service conductors at **75°C**.
- **Bare-in-raceway exception: "15 metres and two 90s (≈180° total)."** Equivalents: 4×45, 6×30; watch "offset" wording; add the degrees.
- **"6 / 4 to ground; 12 / 10 for impedance."** Grounding conductor min **#6 Cu / #4 Al** (Rule 10-114). Impedance-device conductor min **#12 Cu / #10 Al** (Rule **10-318**) — aluminum answer **#10**, not #8.
- **Which conductor to ground:** "2-wire → ground the **identified (neutral)**; 3-wire → ground the **common (centre/zero-volt) conductor**." (Rule **10-208(1)(b)**.)
- **"Bond at BOTH ENDS."** Metal raceway / metal sheath / cable armour. (Answer not literally spelled out — memorize.)
- **"Standard locknuts for bonding — like a bridge, all bolted, never loosens."**
- **Bonding jumper: "Table 16 — ampacity OR overcurrent device (your choice)."** (Contrast: Table 43 = ampacity ONLY.) Busbar in mm² or wire in AWG. 30 A Cu bus → **3.5 mm²**.
- **Equipotential bonding: "6 / 4 normally; 10 / 8 if concealed + mechanically protected."** Aluminum → #4, copper → #6 (read which metal is being SIZED, not which surface is aluminum).
- **"Equipotential = equal potential = no voltage difference = no shock (bird on a wire)."**
- **Raised floor:** "conductive raised floor gets bonded — ignore the hair-salon dressing."
- **Cover (Table 53): "cover = TOP of cable to FINISHED GRADE, in millimetres."** Not trench depth.
- **Cover reduction: "minus 150 mm with mechanical protection over it."** Conditions: **planking 38 mm**, **concrete slab/block 50 mm**, **poured concrete 50 mm**; flat-form extends **50 mm** each side. ("38 for wood, 50 for concrete, 50 to the sides, 150 off the cover.")
- **Sand bedding: "75 mm BOTH above AND below; screened sand, max grain 4.75 mm; cables adjacent, never crossing."** The word **"both"** is the trap.
- **Navigation trick: "every section has its own Wiring Methods — that's your address."** (Corrosion → Section 22 metal box must be corrosion-resistant; fire pump → metal raceway or non-metallic encased in 50 mm; PV → Section 64 wiring methods → extra-hard flexible cord / type RPV.)
- **Exam mindset: "I think I know it" is why people fail — OPEN THE BOOK every time. Tables are where marks are lost; read the wording word-by-word.**
