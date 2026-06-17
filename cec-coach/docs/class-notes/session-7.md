# Session 7 — Study Notes
**Course:** BC Construction Electrician Exam Prep — Canadian Electrical Code (CEC) Part I, 2024 (26th Edition, CSA C22.1:24)
**Topic:** Section 12 — Wiring Methods (continued). Armoured cable bending radius (Rule 12-614/616), special sheathed cables (aluminum/copper/MI), **FCC flat-conductor cable / under-carpet wiring**, then the raceway block: general raceway rules, the **conduit-fill machine (Tables 6, 8, 9)** with a full worked example, no-splice-in-raceway (12-902), bends — "4 × 90° = 360° max" (12-936), Table 7 bend radius, rigid/flexible metal conduit (uses, threads Table 40, minimum/maximum size, support spacing), **PVC conduit** (temperature limitation 75 °C, expansion joints with worked example, bonding continuity), RTRC, EMT, ENT, surface raceway, underfloor raceway, **solar raceway**, auxiliary gutter (20 % fill), wireway (20 % / 40 %), busway/splitter + **down-sizing & overcurrent protection (Rule 12-2210 "reduction in size", 15 m exception)**, and **cable tray (Section 12, Table 66)** clearances. Ends by previewing **Section 12 boxes (CC-34 / CC-36)** for next session, then Section 14.

> The teacher's pitch this session: most of these raceway sub-sections are **repetitive "type" questions** — every raceway has the same four recurring sub-headings (max # of conductors → Rule 12-910/12-2010 "200/200"; provision for bonding continuity; temperature limitation; expansion joint). Learn the pattern once and the whole back half of Section 12 collapses. He keeps saying **"mark this red"** = guaranteed exam item, and **"only highlight this, you'll find it in the book during the exam"** = low-probability, just know where it lives. Garbled spoken numbers are flagged **(approx — verify in book)**.

---

## High-yield: what the teacher flags as important / tested / traps / common mistakes

### Armoured-cable bending radius — Rule 12-614 / 12-616
- **TESTED ("this has been on the exam, including this year"):** the **bending radius of armoured cable**. If you bend an armoured cable too tightly the armour **breaks/kinks** at the bend point.
- **Number to know: inner-edge bending radius must be at least 6 × the cable diameter** ("6 inch / 6 times" in the recording — interpret as **6× the cable OD, measured at the *inner edge* of the bend**). "Inner edge" = measured from the inside of the bend.
- **Trap — cable pulled *through* a conduit/tubing:** Normally you do NOT run armoured cable inside conduit (redundant protection); only do it where extreme mechanical damage is likely (e.g. a truck could drive over it). When you DO, the **conduit/tubing bend radius** must be:
  - **not less than 10× the cable diameter for low voltage**, and
  - **not less than 18× the cable diameter for high voltage.**
  - Reason the HV multiplier is larger: higher voltage → thicker insulation → thicker/stiffer cable → needs a gentler bend so it won't jam/be damaged being pulled.
- **The 32 mm rule carries over (Rule 12-616 area):** where armoured cable is run through studs/joists, keep it **at least 32 mm from the nearest edge** (same drywall-screw protection rule taught for NMSC last session). (Recording says "si o do" = 32 mm.)

### Armoured-cable / NMSC support spacing — first support and intervals (Rule 12-618 area)
- **From last week, still in force:** for NMSC between two boxes, first support within **300 mm of the box**, then **intervals not exceeding 1 m (1.5 m)** between supports. *(Recording mixes "1 m" and "1.5 m"; the interval value is read off a noisy recording — verify: NMSC interval is generally 1.5 m, first support 300 mm.)*
- **New wrinkle for armoured cable — the first-support distance scales up with cable size (the connector is self-supporting):** the bigger the armoured cable, the bigger the connector, the more it supports itself, so the first support need not be at 300 mm:
  - up to size **35 (≤ 35) → first support within 300 mm**,
  - **> 35 and ≤ 78 → within 600 mm**,
  - size **78 (tray size) → within 900 mm**.
  - These are **maximums** ("can be less, never more"). (approx — the 35 / 78 size breakpoints and 300/600/900 mm are read off a noisy recording; verify the exact rule/table.)

### Special sheathed cables — aluminum-sheathed / copper-sheathed / mineral-insulated (MI)
- **Low yield — "we've never seen a question, but mark it."** Aluminum-sheathed cable, copper-sheathed cable, mineral-insulated (MI) cable. Just know they exist and where they live in the book in case your exam is the unlucky one. Teacher's standard advice: **prepare for ~5–6 years of question banks**, so know the location even of things you've never seen tested.

### FCC — Flat Conductor Cable / under-carpet wiring (Rule 12-7xx)
- **FCC = Flat Conductor Cable.** Conductors are made **flat** (3/4/5 wires sit side-by-side into a flat profile instead of round) specifically so they can be run **under carpet / floor covering** with minimal bump. This is the **Under-Carpet Wiring System.**
- **THREE separate exam questions come out of FCC (he says "three questions come from this"):**
  1. **Use permitted (where you CAN use it):** dry or **damp** locations, on a surface that is **concrete / ceramic / hard, smooth** flooring.
  2. **Use prohibited (where you CANNOT use it) — this is the common question:** NOT outdoors; NOT where **corrosive materials** are present; NOT in **dwelling units (residential)**; NOT in **hospitals** *except in the office areas* (reason: a patient/child tripping on the bump is a real hazard, so only office areas allowed). Also prohibited where **voltage to ground > 150 V** or **between conductors > 300 V**, and on circuits **over 30 A**.
  3. **The adhesive / carpet-square question (750 mm is the tested number):** If the floor covering (carpet) is the **adhesive (glued-down) type**, you may NOT run FCC under it **unless** the carpet is laid as **carpet squares ≤ 750 mm** on a side, **and** the adhesive is the **release type** (re-stickable / "peel-up" adhesive so the carpet can be lifted for maintenance and re-laid). **750 mm is the exam number.**

### Raceway — general rules
- **Definition:** any channel you pull wire through = a raceway. Each raceway type has its own advantages, disadvantages and **limitations** — temperature limit, voltage limit, conductor-size limit. You pick a raceway by knowing all of them.
- **MAX NUMBER OF CONDUCTORS IN A RACEWAY — guaranteed exam trap (Rule 12-910):** the maximum is **200 conductors** (he hammers this: students are used to seeing "5, 6, 10" and think 200 is "a crazy alien number," so they pick 10 or 12 — **the answer is 200, "not more than 200"**). This general "200" reappears under every individual raceway ("how many conductors can this raceway hold? → Rule 12-910 → 200"). **Keep Rule 12-910 in mind; it's reused constantly.** (Later in the session he also cites it as 12-2010 / "12-910" interchangeably for individual raceways — same 200 figure.)

### Conduit fill — the "three-tables-together" machine: Tables 6, 8, 9 (Rule 12-9xx)
- **Learn Tables 6, 8 and 9 together** — they are one machine for sizing a conduit when you know the conductors going in (sizes/types may be mixed).
- **Procedure (memorize):**
  1. **Table 6** (runs Table 6A … 6K) → get the **cross-sectional area (mm²) of each conductor**, by size/insulation; **sum** them = total conductor csa.
  2. **Table 8** → gives the **maximum fill percentage** you're allowed to use (53 %, 31 %, 30 %, 40 % … depending on number of conductors). For conduit the fill % is **variable by number of conductors** (unlike receptacle boxes = fixed 40 %, or some others 20 %).
  3. **Table 9** (runs Table 9A … 9J/H) → with your required area at the correct fill %, read down the column for your conduit type to get the **conduit trade size**.
- **Table 8 fill percentages — TESTED:** **one conductor → 53 %; two conductors → 31 %; three or more (3, 4, … 200) → 40 %.** Teacher: "**Memorize the 40 %** — for 3-or-more conductors don't even open Table 8, just use 40 %, because the exam almost always gives you more than two." (approx — confirm the 53/31/40 figures against Table 8; he also lists "30 %" among Table 8 numbers, which is the lead-sheathed-cable row.)
- **Table 8 lead-sheathed trap:** Table 8 has **two rows** — the **first/top row = "not lead sheathed"** (the ordinary cable everyone buys), the **second row = lead-sheathed (sheath of lead/سرب)**, which is a rare special-order cable. **If the question does NOT say "lead sheathed," use the FIRST (not-lead-sheathed) row.** Lead-sheathed cable isn't sold off the shelf — special order, "2–3 months to deliver."
- **Table 9 is pre-computed for you — shortcut:** Table 9 already breaks out columns by fill %, so you do **not** have to scale to 100 %:
  - **9A / 9B → 100 % column,** 9C / 9D → 53 %, 9E / 9F → 31 %, **9G / 9H → 40 %.** (Lettered pairs continue because there are many conduit types.)
  - So for a 3+-conductor job you can take your **summed conductor area** straight to the **40 % table (9G/9H)** and read the conduit — no ratio math needed.
- **Trade size ("tre size") concept — TESTED elsewhere:** the named size (e.g. "16") is a **trade/nominal** size, NOT the exact internal diameter. Table 9A shows e.g. rigid metal conduit "16" = ID 16 mm, flexible metal "15" = ID ~15.8, PVC "14" = ID ~15.7 — but **all are called size 16**. When buying you ask for "size 16," not the exact mm. This is why Table 9 has many columns (areas differ slightly per conduit type even at the "same" trade size).

### Minimum conduit size — Rule 12-9xx ("smallest is 16")
- **TESTED:** "What is the smallest conduit/tubing size?" The code states **no conduit shall have an internal (trade) diameter less than 16**, *except size 12 is permitted for **flexible** metal conduit and **liquid-tight** flexible conduit only* (and even those cannot exceed 1¼" / size used). **The book's official answer to "smallest size" is 16** — answer **16**, even though a 12 exists as an exception for flexibles. (He dislikes the wording but says give the book answer.)

### No splices in a raceway — Rule 12-902 (12-920 area)
- **RED / answer hard "NO":** *"There shall be **no joint or splice** in the conductors or cable within the raceway."* If a question asks "can you splice inside a raceway?" → **NO.**
- **Exception (don't get tricked the other way):** in a **busway, wireway, cable tray** (anything where the conductors stay **accessible** for maintenance, e.g. a cable tray you lay cable into) you **CAN** have a joint/splice, because you can still reach it. The prohibition is about splices that get **buried/inaccessible** inside a closed conduit. Answer "no" firmly for ordinary conduit, but watch for the accessible-raceway exception.

### Bends — Rule 12-936 "4 × 90° = 360° maximum"
- **VERY IMPORTANT, recurring:** between the point where you push the wire in and where you pull it out, you may have **"not more than the equivalent of four 90° bends"** = **360° total maximum.** The keyword is **"equivalent."**
- **The classic trap:** the question gives bends that are NOT all 90° — e.g. **three 90° bends + three 30° bends.** Don't blindly answer "4." Add the degrees: 3×90 + 3×30 = 270 + 90 = **360° → that's the max, allowed but nothing more.** "If you read it once you nail it; if you didn't, you freeze." Add up actual degrees vs. 360.
- **Table 7 — bend radius for raceway:** instead of a flat "6×" rule (as armoured cable used), Section 12 gives a **table (Table 7)** of minimum bend radius **by conduit/tube trade size.** (He notes if you back-calc it ranges ~5.5× to 6.5× the size, but **use Table 7**, don't use a single multiplier for conduit.)

### Rigid & flexible metal conduit (Rule 12-1000 area)
- **Use (mark, then read yourself):** permitted in **wood-frame** buildings AND **concrete** buildings; rigid metal in wet/threaded situations.
- **Threaded where moisture is present:** where moisture can enter, rigid metal conduit fittings must be **threaded** (so water can't get in at the connector). A normal connector clamps the conduit with a screw but isn't watertight at the entry; a threaded fitting (with thread sealant, like an old plumbing pipe) keeps water/moisture out.
- **Minimum size:** *"no conduit having an internal diameter less than 16,"* **except size 12** (and not larger than 1¼" for the flexible exception). Smaller flexibles can twist and snag wire, so the upper limit is capped too.
- **Threads — Table 40 (now "red"; was yellow):** tested questions on thread specs. **14 threads per 25.4 mm (per inch)** for size 16. Thread **length** varies: minimum **16.[xx] mm**, maximum **19.[xx] mm** for size 16 (recording garbled — verify exact thread-length values in Table 40). (approx — verify the per-size thread length numbers.)
- **Thread engagement — good to know, not usually tested:** at least **3 complete threads** must engage to be sure the joint is tight/sealed (in **gasoline/gas/explosive** areas it's **4½ turns**, but normal case **3 turns**).
- **Max spacing of conduit support (Rule 12-1010 area):** between two boxes the support spacing depends on whether it's rigid or flexible. For **rigid metal conduit**: max spacing **not more than 3 m** *(recording says "med"/"3 med" — interpret as 3 m for the largest sizes)*. He gives a size-banded set: **size 16 & 21 → not more than 1.5 m; size 27 & 35 → not more than 2 m (3 m?); size 41 → not more than 3 m.** Unlike cable, it does **not** require a support right next to the box (the conduit is self-supporting). (approx — verify the exact 1.5/2/3 m bands per Rule 12-1010 / Table.)
- **Mixed-size group support — TESTED-style:** where rigid metal conduits of **mixed sizes** are run/grouped together, the support spacing is based on **the smallest** conduit in the group (e.g. a group of 16 + 27 + 41 → use the **size-16 spacing (1.5 m)** for all). Don't average, don't use the largest.

### PVC conduit (rigid PVC) — Rule 12-1100 area
- **Restriction on use (read, not heavily tested):** **rigid PVC conduit shall NOT be used in contact with thermal insulation.** Reason: thermal insulation traps heat inside the conduit; the trapped heat can soften/**deform** PVC (PVC is bent by heating it with a heat gun, so it's vulnerable to heat).
- **TEMPERATURE LIMITATION — RED, exam item (Rule 12-1102 area):** *Rigid PVC conduit shall **not be used where it will be subject to temperatures above 75 °C**.* **AND** this shall not prevent the use of conductors rated above 75 °C in it — **provided the conductors are not loaded such that the conductor temperature exceeds 90 °C.**
  - Meaning: you may install a 90 °C-rated or even a 105 °C-rated cable inside PVC, **but you must derate/limit the load so the conductor never runs above 90 °C** internally (so the PVC stays ≤ 75 °C at its surface).
  - **THE EXAM QUESTION (and Table 2 / Table 4 trap — "this second one is red, it's on the exam"):** given a conductor rated e.g. 105 °C **in PVC**, the maximum current you may draw is read from the **90 °C column** of Table 2 or Table 4 — **NOT** the 105 °C column — because the conductor must be held to 90 °C inside PVC.
- **PVC-to-metal connection — Rule 12-11xx:** when a PVC raceway must connect to a metal raceway/fitting, you must use a **female threaded PVC adapter** (threaded PVC female adapter) at the metal connection.
- **Max spacing of PVC conduit support:** support spacing **not greater than 750 mm** for sizes 16 & 21, decreasing for larger sizes — same mixed-size rule (base it on the smallest in a group).
- **EXPANSION JOINT — Rule 12-1118 + Appendix B (worked example, TESTED-style):**
  - **When required:** an expansion joint (a fitting giving the conduit flexibility to move/expand-contract without damaging itself or pulling apart at couplings) is needed when the **length change exceeds 45 mm.** Below 45 mm, no expansion joint needed.
  - **How to compute (Appendix B, under Rule 12-1118):** multiply **three numbers**: (1) the **length of the conduit run** (m), (2) the **total temperature change** (°C) — i.e. (max temp + |min temp|), and (3) the **coefficient of thermal expansion** for PVC = **0.0520** (mm per metre per °C).
  - **Worked example given:** 20 m PVC run; min temp **−40 °C (forty)**, max temp **+30 °C (thirty)** → ΔT = **70 °C**; coefficient **0.0520**. ΔL ≈ **20 × 70 × 0.0520 ≈ 72.8 ≈ 773 (recording)** — *the lecture's spoken result is "773"; recompute: 20 × 70 × 0.0520 = 72.8 mm. The "773" is a recording/transcription artifact — **use ≈ 72.8 mm**.* Since **72.8 mm > 45 mm**, an expansion joint **is required.**
  - **Calculator-error warning (common mistake):** when entering **0.0520**, students drop the **leading 0** and forget the **decimal place** — and 0.0520 itself has a trailing/internal place that gets lost. Enter it carefully. Teacher: he **memorized 0.0520** to avoid hunting Appendix B; you may memorize it too.
  - **Reverse exam question (follow-up):** "Given the previous data, what is the maximum conduit **length** that needs NO expansion joint?" → set ΔL = **45 mm**, solve for length X: **X = 45 / (70 × 0.0520)**. (Lecture's spoken answer "20.3 / 30-ish m" is garbled; compute: 45 / (70 × 0.0520) = 45 / 3.64 ≈ **12.36 m**. Use the formula, not the spoken number.)

### Provision for bonding continuity — recurring rule for every non-metallic raceway
- **The "repeating" sub-section:** every raceway has a **"Provision for bonding continuity."**
- **Section 10 link:** if the raceway is **metal**, you do NOT need a separate bonding conductor (you *may* add one, it's fine). If the raceway is **non-metallic (PVC, RTRC, HDPE/ENT, etc.)**, you **MUST run a separate bonding conductor** inside. Expect this as a one-line answer ("a separate bonding conductor") for each non-metallic raceway.

### RTRC, EMT, ENT, surface/underfloor/solar raceway, auxiliary gutter
- **RTRC (reinforced thermosetting resin conduit):** non-metallic but **higher temperature limit than PVC** — *shall not be used where subject to temperatures above **100 °C*** (PVC's limit was 75 °C; **RTRC = 100 °C** is the tested contrast). Expansion joint = **same 45 mm** (Rule 12-1118 equivalent); max conductors = Rule 12-910 (**200**); bonding = one separate bonding conductor.
- **HDPE / ENT pattern:** same template — expansion joint **45 mm**, max conductors **200**, separate bonding conductor.
- **EMT (electrical metallic tubing) — "I would definitely write an exam question on this":** students misuse EMT.
  - **Use permitted:** exposed, concealed, **wet locations**, outdoors. (A student once failed thinking EMT could NOT go in wet — it CAN.)
  - **Restriction on use:** NOT where subject to **severe mechanical damage**, NOT with **corrosive materials**, and **cannot be directly buried** in earth.
  - **Min/max tubing size:** minimum = **16**; max conductors = **200** (Rule 12-910).
- **ENT — Electrical Non-Metallic Tubing — RED:** the **75 °C temperature limitation actually applies more to ENT than to PVC** (same wording: not used above 75 °C; conductors above 75 °C OK but their ampacity capped at 90 °C). **The temperature-limit exam question is usually about ENT, not PVC. Mark it red.** ENT = Electrical Non-metallic Tubing; must be **surface raceway**.
- **Surface raceway — Rule 12-1600 area:** has a temperature limitation; you may **not fill more than 4000 [?]** (recording garbled — likely a conductor count/area cap). **New point — VOLTAGE limit (Rule 12-1638, "first time I'm telling you this"):** a surface raceway is the **first raceway with a max voltage limit** — *shall not be used where the voltage exceeds **300 V** unless it is **marked** for higher.* **The 300 V is now an exam question** (it wasn't asked before; students recently reported it). Mark it.
- **Underfloor raceway:** "nothing special to highlight," just mark it in case.
- **Solar raceway — Rule 12-18xx (new point, "max conductor size"):** **maximum conductor size = 1 AWG** (whether copper or aluminum, bare or insulated — doesn't matter; you **cannot** put conductors larger than **1 AWG** in solar raceway). Bonding = one separate bonding conductor (non-metallic).
- **Auxiliary gutter — Rule 12-2000 area (new limit):** *you may fill only **20 %** of the auxiliary gutter's cross-section.* (Worked-example style in the practice set.)

### Wireway — Rule 12-19xx (the 20 % vs 40 % trap)
- **TESTED trap (students confuse the two; "I cleaned this up in the practice set"):** wireway has **two different fill percentages**:
  - **20 %** = for **ordinary power conductors** (the general case): *"each wireway / each compartment of a divided wireway shall contain not more than **200** insulated conductors, and the aggregate cross-section shall not exceed **20 %** of the interior cross-section."*
  - **40 %** = **only** where the wireway contains **signal and control conductors** (control circuits) — then up to **40 %.**
- **Conductor-size limit for wireway:** conductors must be **larger than 500 kcmil if copper**, or **larger than 750 kcmil if aluminum** (size floor for what goes in a wireway). (approx — verify direction/threshold in the rule.)

### Busway / splitter + down-sizing & overcurrent — Rule 12-2210 "reduction in size"
- **Busway vs splitter:** both **tap off** a feed. A **splitter** is short (like a box). A **busway** uses internal **bus bars** and can be long (20–40 m), tapping power to multiple motors along its length. As you move down a busway feeding successively smaller loads, the **bus bar may be reduced in size** (less current downstream).
- **RED — "reduction in size," Rule 12-2210 — "this is all red, it's all exam":**
  - When you **reduce the size** of a busway/conductor (down-size it), the **overcurrent protection** must be sized for the **new (smaller) conductor**, NOT the original. (A 500 A breaker won't protect a 50 A bus section.)
  - **EXCEPTION — the famous 15 m number:** you do **NOT** have to re-protect at the reduction **if the reduced (smaller) portion does not exceed 15 m in length.** *"...unless [the reduced portion] does not exceed 15 m."* **The 15 m is an exam question.** Same principle applies to cables (tap rules).

### Cable tray — Section 12, Table 66 (Rule 12-22xx area)
- **Why cable tray:** easy to add/remove/repair conductors; but you must be able to **stand beside it and reach in**, so the code defines **clearances** around the tray.
- **Clearances (mark in code order — TESTED, "they changed this question recently"):**
  - **Two cable trays mounted one above the other:** if the tray depth/diameter is **≥ 50 mm**, vertical spacing between them must be **≥ 300 mm.**
  - **Single cable tray to a ceiling / heating duct / heating equipment above:** **≥ 300 mm.**
  - **Where only a short/incidental obstruction crosses over** (a crossing beam, a water pipe, etc.): **≥ 150 mm.**
  - **Side clearance:** **600 mm** working space on the access side (so a person's hand/body can reach); if two trays are placed side-by-side spanning **> 1 m**, you need **600 mm on each side** (left and right). "Logical — so a person can reach."
- **Cable-tray bonding extension — NEW exam question (students found it; "the question changed this time"):** when a cable tray gets **long**, a single bonding jumper isn't enough — you must **extend/repeat the bonding** along the tray. *Metal cable tray must be bonded such that the bonding span does **not exceed 15 m**.* (Found under the tray's **"provision for bonding"** sub-section.) Answer choices were 10 / 15 / 5 / 25 m → **15 m.** (approx — verify the 15 m bonding interval for cable tray.)
- Detailed cable-tray installation = **Table 66** (he skips reading it line-by-line; "do the Section 12 exercises, ~76–77 questions, and you won't need more practice").

### Boxes — Section 12 (CC-34 / CC-36) — PREVIEWED, taught next session
- **Boxes = 3 (maybe 4) guaranteed exam questions:** "one is definitely a **calculation**, two are easy definitions." The teacher spends most box time on the **last two rules, CC-34 and CC-36** (box-fill calculations). Boxes are deferred to **Session 8** because they're long and calculation-heavy. After boxes → **Section 14.**

---

## Content taught (in order, full detail)

**0. Where we are / housekeeping.** Picking up at **Rule 12-564** area, having finished **12-500s** and the **non-metallic sheathed cable (NMSC)** + Table 19 material last week. "We're now doing the various **cable types** inside Wiring Methods — the cable you choose affects the whole install." Last week = NMSC; this week starts **armoured cable**, then FCC, then the **raceway** family. Good news announced: the teacher's team has **finally consolidated the scattered real exam questions** into clean practice sets (questions were previously sent piecemeal); **from Section 28 onward** students now get full exam-style practice questions reconstructed from ~5–6 recent test-takers' reports.

**1. Armoured cable — bending radius (Rule 12-614 / 12-616).**
- Sensitivity #1 of armoured cable = **bend radius**; tested (including this year).
- Bend the cable too tight → armour **breaks/kinks** at the apex. You can do a sharp 90° (small radius) or a gentle 90° (large radius); armoured cable needs the **large** radius.
- **Rule:** inner-edge bend radius ≥ **6× cable diameter** ("inner edge" = measured from the inside of the bend / inner part of the cable).
- **Armoured cable pulled through conduit/tubing** (uncommon; only where mechanical damage is likely, e.g. truck traffic): the **conduit/tube** bend radius must be **≥ 10× cable dia (low voltage)** and **≥ 18× cable dia (high voltage)**. HV factor larger because higher voltage → thicker insulation → stiffer cable.
- **32 mm edge clearance** (Rule 12-616) still applies where armoured cable runs through studs/joists (drywall-screw protection), same as taught for NMSC.

**2. Support spacing recap + armoured-cable size scaling (Rule 12-618 area).**
- Recap (NMSC, last week): first support within **300 mm** of each box; **intervals not over 1 m (1.5 m)** between.
- New: armoured cable first-support distance **scales with size** because the connector self-supports: ≤35 → 300 mm; >35 ≤78 → 600 mm; 78 (tray size) → 900 mm; all are **maximums.** (approx — verify breakpoints.)

**3. Special sheathed cables.** Aluminum-sheathed, copper-sheathed, mineral-insulated (MI). **No questions seen historically — just mark them.** General advice: be ready for 5–6 years of question banks; know locations even of untested items.

**4. FCC — Flat Conductor Cable / under-carpet wiring.**
- Definition: flat-profile conductors (3/4/5 wires side-by-side) → low bump → run **under carpet**. = Under-Carpet Wiring System.
- **Three exam questions:** (a) **use permitted** — dry/damp, on concrete/ceramic/hard-smooth floor; (b) **use prohibited** — outdoors, corrosive materials, **dwelling units**, **hospitals except office areas**, **>150 V to ground**, **>300 V between conductors**, **>30 A** circuits; (c) **adhesive/carpet-square** — under glued-down (adhesive) floor covering only if **carpet squares ≤ 750 mm** AND **release-type (re-stickable) adhesive**, so carpet lifts for maintenance and re-lays. **750 mm** is the number.

**5. Raceway — general.** Definition (any channel for wire). Each raceway has temp/voltage/size limitations. **Max conductors in a raceway = 200 (Rule 12-910)** — the "200 trap" (students pick 10/12; correct = 200). 12-910 is reused for every raceway.

**6. Conduit fill — Tables 6, 8, 9 (full worked example).**
- **Tables 6+8+9 = one machine.** Step 1 Table 6 → conductor csa, sum them. Step 2 Table 8 → max fill % (53 % one cond, 31 % two cond, 40 % three-or-more; lead-sheathed uses the 2nd row, ~30 %). Step 3 Table 9 → conduit trade size at that fill.
- **Worked example given in class:** *Ten size-6 + ... conductors of type TWN75 / T90 nylon, plus four size-4 — find required Rigid Metal Conduit size.*
  - From **Table 6K** (TWN75 / T90 nylon page): **size 6 → 327 mm²** each? — class reads: **size 6, qty 10 → 327 mm²**; **size 4, qty 10 → 532 mm²**. (Table 6 lists pre-multiplied areas for 1, 5, 10, … conductors, so you read the bundle directly.)
  - **Sum: 327 + 532 = 859 mm²** = total conductor csa, which represents **40 %** of the conduit's allowed capacity.
  - **Scale to 100 %:** 40 % corresponds to ×2.5 → 100 %. So **859 × 2.5 = 2147.5 mm² ≈ 2147** = required **100 %** conduit area.
  - **Table 9A (100 % column, Rigid Metal Conduit):** look up ~2147 → conduit **trade size 53.** Answer = **size 53.**
  - **Shortcut shown:** instead of scaling, take **859 mm²** directly to **Table 9G/H (40 % column, RMC)** → same answer **size 53.** No ratio math needed because Table 9 pre-computes the 40 %.
- **Trade-size note (Table 9A):** the named size (e.g. "16") is **nominal/trade**, not exact ID (RMC "16" ID 16.0, FMC "15" ID 15.8, PVC "14" ID 15.7 — all called **size 16**). When buying, say "size 16."
- **Minimum size = 16, except 12** for flexible/liquid-tight only; book answer to "smallest conduit size" = **16.**
- Practice: **Section 12 exercises, ~76–77 questions; question bank covers 2012, 2015, 2018, 2020, 2021, 2024 — all reflected in these Section-12 exercises.**

**7. Course-strategy aside (mostly meta, exam-relevant bits kept).**
- The two highest-yield **blocks** are **Block B and Block C** (not "sections"): Block C = wiring. "Master B and C and you've basically insured a pass." Section 12 is the single biggest source of questions (he estimates ~12 of ~30 wiring questions), but wiring is spread across the whole book (fire alarm, etc.).
- Study method: read the book well → do each section's own exercises → watch the solution videos → then do the **~1000-question** master set **2–3 times** (10 mock exams), to **understand** (not memorize) so you handle the **trick rewordings** (e.g. taught "two 90° bends" but exam gives "one 90° + three 30°").
- Do your **review/Q&A weeks early**, not the night before; leave the **last ~2 weeks** for polishing. Don't quit work for a week — steady ~6–10 h/week alongside work is enough if you study correctly.

**8. No splice in raceway — Rule 12-902.** *"There shall be no joint or splice in the conductors or cable within the raceway."* Answer **NO**. Exception: accessible raceways (busway, wireway, cable tray) **may** have splices because they stay reachable for maintenance; the ban targets **inaccessible/buried** splices.

**9. Bends — Rule 12-936 + Table 7.** Max **equivalent of four 90° bends = 360°** between pull points. Keyword **"equivalent."** Trap: sum actual degrees (3×90 + 3×30 = 360 = max). **Table 7** gives bend radius by conduit trade size (≈5.5×–6.5×; use the table, not a single multiplier).

**10. Rigid & flexible metal conduit (Rule 12-1000s).**
- Use: wood-frame and concrete; rigid metal threaded where moisture present (threaded fittings + sealant keep water out; a clamp connector isn't watertight).
- **Min size 16, except 12** (flexibles, ≤1¼"). **Threads Table 40 (now red): 14 threads per 25.4 mm; thread length size-16 min ~16.x mm, max ~19.x mm** (verify). **Thread engagement ≥ 3 complete threads** (4½ in gas/explosive areas).
- **Support spacing (12-1010):** rigid metal — banded by size (16/21 → 1.5 m; 27/35 → 2 m; 41 → 3 m, approx — verify). No support needed right at the box (self-supporting). **Mixed-size group → base spacing on the smallest conduit.**

**11. PVC conduit (12-1100s).**
- Restriction: **not in contact with thermal insulation** (trapped heat deforms PVC).
- **Temperature limitation (RED):** not used above **75 °C**; conductors rated >75 °C are OK **provided conductor temp stays ≤ 90 °C** (so derate from the 90 °C column of Table 2/4 — **this is the red exam item**).
- **PVC→metal:** use a **female threaded PVC adapter.**
- **Support spacing:** ≤ **750 mm** for sizes 16 & 21 (smaller for larger; mixed group → smallest).
- **Expansion joint (12-1118 + Appendix B):** required when length change **> 45 mm**. ΔL = run length (m) × ΔT (°C) × **0.0520**. Example: 20 m, −40 → +30 (ΔT 70), 0.0520 → **20×70×0.0520 = 72.8 mm > 45 → joint required.** Reverse: solve length for ΔL = 45 → **45/(70×0.0520) ≈ 12.36 m.** Watch the calculator entry of **0.0520.**
- **Provision for bonding continuity:** non-metallic → **one separate bonding conductor** required.

**12. RTRC.** Non-metallic, **temp limit 100 °C** (vs PVC 75 °C — the tested contrast). Expansion joint 45 mm (12-1118); max conductors 200 (12-910); separate bonding conductor.

**13. HDPE / ENT and the repeating template.** Same four recurring sub-questions per raceway: **max conductors (12-910 → 200)**, **provision for bonding (separate conductor if non-metallic)**, **temperature limitation**, **expansion joint (45 mm)**. ENT = Electrical Non-metallic Tubing; **its 75 °C temp limit is the one most often tested — mark red.** ENT must be **surface raceway**.

**14. EMT.** Use permitted: exposed, concealed, **wet**, outdoors. Restriction: no severe mechanical damage, no corrosive materials, **no direct burial.** Min size 16, max conductors 200. ("I'd write an exam question on EMT.")

**15. Surface raceway (12-1600s).** Temperature limitation; fill cap (~"4000" garbled). **NEW — voltage limit Rule 12-1638: max 300 V unless marked higher** — now an exam question (300 V).

**16. Underfloor raceway.** Nothing special; just mark.

**17. Solar raceway (12-18xx).** **Max conductor size = 1 AWG** (Cu or Al, bare or insulated). Separate bonding conductor.

**18. Auxiliary gutter (12-2000s).** Fill only **20 %** of cross-section.

**19. Wireway (12-19xx).** **20 %** fill for ordinary power conductors (≤200 conductors); **40 %** only for **signal/control** conductors. Conductor-size floor: >500 kcmil Cu / >750 kcmil Al (verify). Students confuse 20 % vs 40 % — cleaned up in practice set.

**20. Busway/splitter + reduction in size (12-2210).** Busway taps multiple motors over long runs; bus bar may down-size as load drops. **Reduction in size = RED:** overcurrent protection must match the **reduced** conductor, **except** if the reduced portion **≤ 15 m** (the 15 m exam number). Same as cable tap rules.

**21. Cable tray (Table 66, 12-22xx).** Clearances: two trays stacked, depth ≥50 mm → ≥300 mm apart; single tray to ceiling/heating duct/equipment → ≥300 mm; short crossing obstruction → ≥150 mm; side working space 600 mm (600 mm each side if grouped >1 m). **Bonding extension: bond span ≤ 15 m for long metal trays** (new question; answer 15 m among 10/15/5/25). Detailed install = Table 66 (skipped; do the exercises).

**22. Wrap / preview.** Boxes deferred to next session (CC-34, CC-36; ~3–4 questions incl. one calculation). Section 12 ends at Rule **12-3000s**; then **Section 14**. Off-topic personal/admin chatter at the end (sending the student the 2024 PDF, the "small book" / abridged code that covers **DC and single-phase motors** which the big book/Section 28 skips and which is **not provided in the exam** — must be memorized; pep talk on the student's odds) is excluded as non-content.

---

## Mnemonics / exact phrasings

- **"Mark it red"** = guaranteed/active exam question. **"Just highlight it / you'll find it in the book"** = low-probability, only know the location.
- **Bend radius:** armoured cable **inner-edge ≥ 6×** OD. Through conduit: **10× (LV) / 18× (HV).** Higher voltage → thicker insulation → bigger multiplier.
- **"No joint or splice in the conductors or cable within the raceway."** Answer **NO** (except accessible raceways — busway/wireway/cable tray).
- **Bends: "not more than the equivalent of four 90° bends" = 360° max.** Watch the word **"equivalent"** — add up actual degrees (e.g. 3×90 + 3×30 = 360).
- **"200" is the answer.** Max conductors in a raceway = **200** (Rule 12-910) — not 10, not 12. "Students think 200 is an alien number; it's the answer."
- **Conduit fill %: 1 → 53 %, 2 → 31 %, 3-or-more → 40 %.** "Memorize the 40 %; don't open Table 8 for 3+ conductors."
- **Table 8 lead trap:** no "lead sheathed" said → use the **first (not-lead-sheathed) row.**
- **Smallest conduit = 16** (book answer), **except 12** for flexible / liquid-tight only.
- **Trade size ≠ exact ID:** "size 16" is nominal; ask the store for "size 16," not "14 mm ID 57."
- **PVC = 75 °C, RTRC = 100 °C.** PVC: conductors >75 °C OK but hold to **90 °C** → read the **90 °C column** of Table 2/4. **ENT carries the same 75 °C limit and is the more-tested one.**
- **Expansion joint when ΔL > 45 mm.** ΔL = **length × ΔT × 0.0520** (PVC coefficient). "Watch the 0.0520 on your calculator — don't drop the leading zero or the decimal."
- **Reduction in size (12-2210): re-protect the smaller conductor — UNLESS the reduced run ≤ 15 m.**
- **Wireway: 20 % normal, 40 % only for signal/control.**
- **Solar raceway max conductor = 1 AWG.** Auxiliary gutter fill = **20 %.** Surface raceway max **300 V** unless marked.
- **Cable tray clearances: 300 / 300 / 150 / 600 mm**, and **bond span ≤ 15 m** for long trays.
- **"Equivalent" appears twice** — in the 4×90° bend rule and in support spacing: always read it as "add up / total," not "count of items."
- **Study-method line:** "You can't pass by memorizing — questions are tricks; you must understand the *why*." Example: "Sara has 2 oranges" after you practiced "Dara has 2 apples" — same math, different words; memorizers freeze.
