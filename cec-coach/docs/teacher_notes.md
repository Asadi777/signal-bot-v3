# Teacher Notes — CEC 2024 / BC Construction Electrician (Red Seal) Coaching

Extracted from three Persian-language class transcripts. English rendering of the instructor's teaching. Tags: **[teacher]** = the instructor's own assertion/method; **[standard CEC]** = matches the published code; **[transcript unclear]** = transcription noise, interpret with care.

Sessions covered:
- **Session 1** — exam strategy/logistics, book structure, Section 0 definitions (ampacity, armour/jacket/insulation/conductor, AWG/kcmil, bonding & grounding, branch circuit / feeder / service, concealed/exposed, conduit vs EMT, dwelling unit, emergency lighting, GFCI Class A).
- **Session 1-2** — overcurrent vs overload, receptacles (duplex/single/split), 120/240 split-phase & neutral current, 3-phase wye line-to-line vs line-to-ground & √3, transformer basics, voltage classes (ELV/LV/HV), Section 2 general rules, deviation/special permission (Rule 2-030), equipment marking line-to-line/line-to-ground, dwelling 150 V-to-ground rule (2-130 area), enclosure types (Table 65), working space (Table 56), gas appliance clearances (Appendix B / Rule 2-328).
- **Session 2-2** — review of working space/enclosures, then Section 4: conductor ampacity, Tables 1–4 selection logic, correction factors (grouping Table 5C, ambient temperature Table 5A), worked examples.

---

## 1. Exam strategy & logistics

- **Pass mark is 70%.** [teacher] "70 itself is a pass." All questions are multiple-choice. A 69 is a fail (he cites students who scored 69 and 68 — "68 failed"). He says one question can flip your result, so he does not compromise / "doesn't go easy on any block."
- **Five blocks: A, B, C, D, E.** Each block has a fixed number of questions and is **graded separately** ("block-i ro joda gune behet midan"). You can fail individual blocks and still pass overall if the heavily-weighted blocks pull you up.
- **Block weighting / which blocks are decisive:** [teacher]
  - **Block C ("Block 30") = wiring / installation** and **Block B (≈ "20-sixteen", essential/installation skills)** together carry the most questions — he says roughly **60 questions come from these two blocks**, and they are **the deciding blocks**. "If you do these two blocks well, even guessing the rest, you'll pass." He expects students to score **above 90%** on these two; anything lower signals the student didn't study evenly.
  - **Essential skills block** (tools/equipment use, safety, scaffolding/guardrail support, rigging) ≈ a handful of questions (he mentions ~"11 to 20" item ranges) — low concern, learn once and move on.
  - **Block D = motors** — the hardest block. Most students from other schools do poorly here. He claims to be a motor/controls specialist (~40+ years, generator-building experience at Ansaldo Italy, with ABB/Siemens etc.) and gets his students **above 80%** on motors.
  - **Signalling & communication ≈ 10 questions**, of which about **half come from the book and half from outside the book**.
- **His question bank / "~90 of 100":** [teacher] He maintains a large, continuously-updated question bank. Students phone him right after their exam reporting which questions appeared. He claims **about 90 of every 100 questions are already in his bank** ("from 100 questions, 90 are ones we already have"). He keeps it updated; if a previously-sent practice set is re-sent updated, it means it "moved/changed."
  - Of the ~20 motor questions, "**seven or eight are from the book**"; out-of-book motor questions are actually the *easy* ones.
  - For signalling/communication's ~10: about 5 from the book, 5 from outside.
- **Where answers live:** Exam questions come "from the book and from the places we mark in the book." He does NOT rely on pre-marking everything; once you're fluent in the book ("when you know where high-voltage bonding is"), you find answers fast even if unmarked.
- **His methodology:** [teacher]
  - Teach **mastery of the code book first**, then problem-solving — *not* drilling Q&A like other schools (he dislikes Q&A-only schools because students never become fluent in the book).
  - Teach **concept first, in Persian**, then solve in English (because the exam is answered in English). Understanding the concept matters more than memorizing.
  - Teaches **definitions deeply even to experienced Iranian electricians** (because Iranian/EU definitions differ) and **very thoroughly to non-electricians**.
  - Recommends **YouTube** (especially cartoon/animation videos) for bonding, grounding, armoured cable, stripping, etc. as homework between sessions.
  - Uses an **image/visual dictionary** approach for English technical terms.
- **Time allowed:** [teacher] Everyone is given **5 hours**; 5 hours is plenty (he says it's even "too much"). **6 hours is sometimes granted but not guaranteed** — don't assume you'll get it. Students have passed coming out in **3 hours 5 minutes**. Most students say 4 hours is too tight; he advises using the full time and watching the clock under exam pressure.
- **Recent results he cites:** Of 9 recent students, **7 passed on the first try**; 2 failed with **68**. He reviews score breakdowns "like a blood test" to find each student's weak spot.
- **Code edition:** Exam is on **CEC 2024**. He owns both 2021 and 2024. **~95% of the book is unchanged** from 2021; changes are mostly trivial number tweaks (his example: a transformer-to-wall clearance "6 mm → 6.5 mm" — *illustrative, not a real value*). He often teaches from the 2021 PDF because the 2024 PDF is locked (search/navigation disabled); he'll switch to 2024 only ~4–5 times in the whole course where a real change exists.
- **Course length:** ~30 hours (he notes it sometimes runs 32–33). He bills/tracks "to-hours"; tells the student not to worry about exact hours and that he'll try to compress to 1–2 sessions/week. Sessions are ~3 hours (some 2 hours).
- **Challenge level note:** [teacher/logistics] For Red Seal challengers, the union recognizes Level-4 status; some companies pay challengers Level-4 wages and stop sending them to school.

### Book structure (he teaches this explicitly as exam-relevant)
The CEC has **six parts**; know the structure cold:
1. **Front matter / preliminaries** — who wrote it, basis, sources. Mostly skip. **Exception: "Metric units" conversion section** — likely an exam question. Two columns: *previously-used (Imperial) units* and *SI units*. The trade works in Imperial (e.g. "18 inches", not mm) but the book is all SI (mm, cm, km). A conversion question example he cites: illumination unit **lux (SI) vs foot-candle (Imperial)** — multiply by the middle conversion factor. (This is the only spot in the front matter that yields questions.)
2. **Sections** — Section 0 through Section 86. They are **even-numbered**, so there are **43 sections** (not 86); **2 high-number sections are also removed/reserved**, and 3 sections total are removed — effectively **~40 sections** are studied. Each section covers one topic (e.g. Section 74 = airports, Section 66 = amusement / film-set / mobile / radio-TV wiring environments). Sections do **not** carry equal weight — some take a whole 3-hour session, ten others fit in one session.
3. **Tables** — "Tables" (one word). **~20% of the book's volume AND ~20% of exam questions come from tables.** He frames this as *good news*: table questions are "lookup" questions — find the spec/number in the right table.
4. **Diagrams** — **not** schematic/technical circuit diagrams; they are *pictorial* (e.g. Diagram 1 shows the physical appearance of Canadian receptacles — "more than 550 of them").
5. **Appendices A through L/M.** Only **Appendix B** matters for the Red Seal exam. Appendix B is *not a test/exam itself* — it gives **explanatory notes** on rules in the sections, and **about 6–7 exam questions come from Appendix B**. Rule of thumb: if you can't find something in the main section, check Appendix B. **Appendix D** also holds some tables (he criticizes this as disorganized — e.g. cable tables, **Table D2 = DC motors**, while 3-phase motors are in **Table 44**, single-phase in **Table 45**).
6. **Index** (back of book) — alphabetical list of technical terms → rule numbers. **Very valuable under exam pressure**; can save 3–5 questions when you can't guess which section a topic lives in. Example he gives: **fence height = Rule 26-304**, "shall not exceed 1.8 m" — found via the index under "F". Caveat: not 100% — e.g. "**chain-link fabric**" (chain-link fence mesh) he once couldn't find because it's filed under fencing, not its own entry.

### Administrative & scope notes
- He **skips the Administrative section** at the front of Section 2 ("not your job, not on the exam, poorly defined").
- **Section 0 (definitions)** is normally not taught by other schools (they assume a 4-year-trained challenger knows them), but he teaches it because definitions are exam-heavy and differ from Iranian/other conventions.
- The CEC is a **law/code book, not a textbook** — written in mixed legal + technical language by an electrical author then edited by a non-electrical editor, which makes it hard. Even Canadian-raised / Filipino students struggle to parse it. He restates rules in plain language.

---

## 2. Concepts taught, by CEC section / topic

### Section 0 — Definitions

**Object of the code [standard CEC]:** establishment of **safety standards for the installation and maintenance of electrical equipment**; "consideration has been given to the prevention of fire and shock hazard." [teacher] He notes ~**20% of Canadian fires are electrical**, so the code's purpose is reducing that. The object statement can itself be a question.

**Ampacity** — from *amp* + *capacity* = current capacity of a conductor. **The maximum current a conductor is permitted to carry continuously.** Example: "ampacity of this conductor is 15 A" → you may pass at most 15 A; above that the conductor is damaged. Used constantly (sizing a cable for a motor = pick a conductor whose ampacity covers the motor's current).

**Conductor / Insulation / Armour / Jacket** — the four layers of a cable:
- **Conductor** = the metallic current-carrying wire (usually copper or aluminum).
- **Insulation** = the "shirt" on the conductor; prevents contact/short between the conductor and the metallic armour. (Translated as "āyegh".)
- **Armour** = the steel wires/wrap around the cable ("zereh-dār" / armoured cable in Iran) = **mechanical protection** (e.g. if someone hits it with a pick). Used where protection is needed (e.g. telecom cables). Look it up as an image ("armour for electric cable").
- **Jacket** = the **non-metallic covering** = final outer sheath providing **environmental protection** (against moisture, acidic media, corrosion/rust). Armour = metallic covering; jacket = non-metallic covering.
- An armoured cable **may or may not have a jacket**: **jacketed** vs **non-jacketed ("unjacketed")**. Example cable type he names: **AC90 ("ای سین آی دی" = AC90)** — search "armoured cable" on YouTube to learn it. Another type is the flexible "shower-hose-like" interlocked armour you can unwind by twisting opposite to its lay (common in restaurants).

**AWG vs kcmil** — two units for conductor size; a weak point for students. [teacher] He calls the dual-unit system a **weakness of the North American standard.**
- **Smaller AWG number = larger conductor = more current.** Sequence: 14, 12, ... 2, 1, then **1/0, 2/0, 3/0, 4/0** (written "one-aught… four-aught", and with slashes), then jumps to **250 (kcmil)**.
- Practical sizing: **#14** for lighting (low current), **#12** for kitchen/receptacle circuits, **#8** for dryer/range (high current).
- Up to **107 mm²** → expressed in **AWG**; from **127 mm² up to 2000** → expressed in **kcmil** (he reads "kcmil"). The two unit ranges don't overlap, so they're not confused. kcmil also has an inch/circular-mil cross-sectional basis (look up later).
- [teacher] Note: the AWG numbers (and the aught numbers) are *not* the cross-sectional area — they're a coded scale; the area must be read from the table.

**Bonding vs Grounding** — taught via a washing-machine example.
- **Grounding:** The utility creates the neutral/return at the street by driving a metal rod (**electrode**) into the earth near the transformer; earth = the zero-potential point ("voltage to ground = 0"). Two physics rules he states: **(1) electricity always wants to go to earth**, and **(2) electricity takes the path of least resistance** — "the foundation of grounding and bonding." From the panel, a green wire runs to the electrode; that whole assembly (electrode + its wire + everything bonded there) is **grounding**.
- **Bonding:** If a hot wire loosens and energizes the metal frame of an appliance (washer, fridge, dryer), a person touching it while standing on the ground gets shocked because current wants to reach earth through them. A **green bonding conductor** connects all the **metal (non-current-carrying) parts** back to that grounded point, giving current a low-resistance path; this effectively connects the energized frame to neutral, so the breaker trips instantly. The wires from the panel that go out to bond the washer/fridge/dryer frames = **bonding** ("you bonded them together").
- **Definition of bonding [standard CEC, paraphrased by teacher]:** a **permanent, low-impedance path** (he says he'd have written "low-resistance" but the code uses "impedance") created/joined to connect together the **non-current-carrying metal parts** of equipment. *This definition is a frequent exam question* (he marks it red). Exam trick: they give the definition with a key word removed (e.g. drop "low-impedance," or change "non-current-carrying") and you must spot the complete/correct version. Bond only the **non-current-carrying metal parts** — not parts that are meant to carry current.

**Branch circuit / Feeder / Service** — taught via a house with a main panel + two sub-panels (two tenants).
- **Service** = the incoming supply conductors entering the building and the **service equipment** (the main panel itself). "Service conductor" / "service equipment."
- **Branch circuit** = the wiring **after the final overcurrent device** — there is **no further overcurrent device (fuse/breaker) beyond it**. [standard CEC, paraphrased] "that portion of wiring between the **final overcurrent device** protecting the circuit and the **outlet(s)**" (outlet can be a receptacle, lighting, washer, fridge receptacle, etc.).
- **Feeder** = "any portion of the electrical circuit **between the service box and the branch-circuit overcurrent device**." [standard CEC, paraphrased] Boundary: from the breaker in the service box to the breaker in the sub-panel. [teacher] **~90% of electricians don't know "feeder"** — it's the conductor feeding a sub-panel (sub-panel of a house, the panel of a workshop in a factory, each store's panel in a mall, each apartment's panel). The thick cables run inside a building (e.g. to the range) are loosely called feeders by tradespeople, but precisely a feeder feeds a *panel*.

**Concealed vs Exposed** (used for wiring):
- **Concealed** = "covered" — wiring run before drywall, made **permanently inaccessible** by the structure/finished building (Iranian "tu-kār" / in-wall).
- **Exposed** = on the surface, accessible (Iranian "ru-kār"). ("Expose" like an exhibition — visible/on display.)
- **Bare conductor** = a wire with **no insulation** (sim-e lokht).

**Conduit (and Raceway)**:
- **Raceway** = a **general/generic term** — any **channel designed to hold/contain conductors** (a "wireway" / any enclosed conduit). You can't buy "a raceway" at a store; it's a category, not a part. Includes **conduit** (rigid, flexible, metal or non-metal), **EMT**, **underfloor raceways**, etc.
- **Conduit** = a raceway with **circular cross-section** (like a pipe / tube). Types: **flexible metal conduit** (bendable by hand), **rigid metal conduit** (must be bent with a bender), **rigid PVC conduit** (non-metallic).
- **EMT (Electrical Metallic Tubing)** — [teacher, emphasized as a common confusion] **EMT is NOT a conduit.** The definition explicitly *excludes* EMT from "conduit." EMT is metal, round, and used for electrical wiring (so it *looks* like rigid metal conduit), but a **tube** has **thin wall** (low mechanical strength) → that's EMT's **weak point**. EMT's advantages: **cheaper (~1/3 the price), lighter, easier to install/cut**. Practice: use **conduit only where mechanical protection is needed** (e.g. a short run in a parking garage where a car/bike could hit it); run **EMT for the rest of the route**. He stresses tradespeople often don't know the difference because supervisors just hand them a buy-list.
- **ENT (Electrical Non-metallic Tubing)** = a non-metallic round tubing (the flexible corrugated "elephant-trunk" type); also not a conduit.

**Dwelling unit** — two meanings, context-dependent:
- **General meaning** = residential places generally: house, townhouse, condo, hotel ("one or more rooms..."). When a rule says "**other than a dwelling unit**" it means **non-residential** (commercial / industrial).
- **Specific:** **Single dwelling unit** = a **detached** type (house / townhouse). Plain **dwelling unit** (when contrasted with "single dwelling unit") then means **apartment / similar / row-link** type. [teacher] This single-vs-plain distinction is learned from experience, not stated outright in the book.

**Emergency lighting** [standard CEC + cross-reference]:
- Purpose serves **both**: occupant **safe egress** AND **safe operation/fire-fighting** facilitation during an emergency. (Battery-backed exit lights stay on when power/breaker is lost; lights along the escape corridor.)
- **Cross-standard exam trap:** emergency lighting, smoke detectors, exit-sign **colours**, etc. are **provisioned in the NBC (National Building Code)**, **not** the CEC. Exam may ask "in which standard is emergency lighting / a smoke detector / exit-sign colour provisioned?" — answer **NBC**, not CEC. *But* if the question asks about the **cable size / breaker / overcurrent** for the emergency lighting, that **IS in the CEC** (CEC handles conductor sizing & protection; NBC handles where/what device, colours, detector placement). Don't reflexively answer "CEC."

**GFCI Class A (Ground-Fault Circuit Interrupter)** [standard CEC + physics]:
- = **Ground Fault Circuit Interrupter**; interrupts the circuit to the load on a ground fault. There is also a **Class A** designation (no Class B in use); the "Class A" label distinguishes it from older types.
- **Used wherever shock risk exists due to contact with water:** bathroom sink/vanity receptacles (razor/hair-dryer), kitchen counter receptacles, outdoor receptacles, near tub/shower, etc. — **all over the building** anywhere water contact is possible. ("All outdoor receptacles" — note he says this broadly.)
- It can be a **GFCI receptacle** (with test/reset buttons) **or a GFCI breaker** in the panel. A GFCI breaker has an **extra return wire (a pigtail / neutral-return)** that ordinary breakers lack — that's how to recognize it.
- **Operating principle:** current out on the hot must equal current back on the neutral. If hot = 5 A but neutral returns only 4 A, **1 A is leaking** (e.g. through water to ground / through the metal sink) — a ground fault. The GFCI senses this imbalance and trips.
- **Trip threshold [teacher / standard]:** trips at a small leakage; he states the design figure as **trips at not less than ~4–6 mA** — "6 mA" is the reference, with the lower **~4 mA** as the **tolerance** so it won't trip on trivial/false faults. (Standard CEC Class A figure is 5 mA nominal; his "4–6 mA tolerance band" framing.)

---

### Section 2 — General Rules (overview & specific rules)

[teacher] Section 2 is where **rules** (numbered) begin; Section 0 was only definitions. Rule numbering: first digit = section number (rules in Section 4 start with "4-", Section 20 with "20-"), then a sequence number; **rules are usually even-numbered** (odd numbers left as spares for inserting new rules later). Rules have **subrules** denoted (1), (2), (3)... He skips the Administrative subsection. He stops saying the literal rule number's "(1)" each time ("ghānun-e do safsi…" = rule "2-0xx(1)").

**Overcurrent vs Overload** — a core distinction students confuse:
- **Overcurrent** — same definition worldwide; **primary job = protect the conductors/cables (wires)**. Installed as either a **fuse** or a **breaker**. A short circuit ("ettesāl-e kutāh") creates a very large current that rapidly heats/melts the wire and can start a fire; overcurrent protection cuts it off.
- **Fuse vs Breaker:** A **fuse melts** (a thin wire sized to carry e.g. 20 A without melting; "this fuse is 20 A" = carries 20 A without blowing) — must be **replaced** after it blows. A **breaker is reusable / resettable** — just switch it back on (e.g. when a coffee-maker + air-fryer trip the kitchen-counter breaker).
- **Overload** — **primary job = protect the MOTOR**, NOT the wires. Taught with an elevator/motor analogy:
  - A motor **draws current in proportion to its load** (an elevator rated for 10 people might draw 30 A full-load; with 2 people only ~5–6 A). **Calculations are done at full-load current (FLC)** — i.e. at the 30 A.
  - On **start-up**, a motor draws a large **inrush/starting current** for a very short time (fraction of a second, e.g. "half a second"), which is **not precisely calculable** and varies by motor (a 30 A motor's inrush could be 100–150 A).
  - Because of inrush, you **cannot** size the motor's fuse at the running current. [teacher] For a **30 A motor you might install a 100 A fuse** so the brief inrush doesn't blow it (he says **Section 28** explains the real motor protection multipliers — *he references "Section 28" / motors*).
  - **Problem:** a 100 A fuse won't detect a genuine overload — e.g. the elevator overfilled drawing 40 A — because 40 A < 100 A. So you add a separate device for **overload** (the extra/excess load).
  - **Overload relay = thermal (bimetal).** Two dissimilar metals with different thermal-expansion coefficients heat and bend, opening the circuit. It's deliberately thermal so it **ignores the brief inrush** (no time to heat up during a fraction of a second), but trips on **sustained** excess current. It has an **adjustment screw** (settable current and **time** — inverse-time); you can set it to tolerate e.g. 40 A for up to 1 minute then trip.
  - **Moral / exam takeaway he states explicitly:**
    - **Overload protects the motor.**
    - **Overcurrent protects the cables and wires.**
    - You install **both** on a motor.

**Receptacles** [Section 26 area + definitions]:
- **Receptacle** = what Iran calls "priz" (outlet). Types:
  - **Duplex** = the common two-outlet receptacle in rooms/bedrooms.
  - **Single** = one outlet, used where there's a single dedicated load (e.g. garage-door opener, dryer) — the bigger single ones for dryer etc.
  - **Split** = a duplex whose two outlets are **separated** (split). A **Canadian standard for homes**, common on **kitchen counters**.
- **Normal duplex wiring:**
  - The receptacle has a **brass/gold-coloured screw** on one side that connects to the **smallest slot** = the **hot** ("most dangerous wire") — typically the **black** wire.
  - The **silver/nickel screw** side connects to the other slot = the **neutral/white** wire (low hazard).
  - The **largest slot / green screw** = **bonding** (green wire to the panel).
  - The two top brass screws are joined by an internal **bridge/tab** (a "platin"), and likewise the two neutral screws — so one black wire and one white wire feed both outlets through the tabs. Standard receptacles are **15 A** ("ponze amp").
- **Kitchen counter** receptacles are often made **20 A** (stronger, with **#12 wire**) because that's where breakers trip most (e.g. air-fryer + coffee-maker together).
- **Split receptacle** = you **break the tab** (the middle connecting bridge) so the two outlets are fed by **two different hot wires** from **two different breakers**, so loads don't all trip one breaker.
  - **Danger / exam point:** the two hots on a split are on **different phases** with a **voltage difference (240 V) between them**. If you fail to break the tab (or connect the two hots) you create a short — "very dangerous." The middle tab is **designed to be broken** for this purpose.
  - The two breakers feeding a split must be **adjacent and on different buses** — called a **two-pole (tied) breaker**; in a panel the bus plates alternate phase, so two adjacent breakers automatically come off **different phases**. They have a **handle tie / pin** so if one trips, the other trips with it (electrically one trips, the tie mechanically forces the other).

**120/240 V split-phase residential system & neutral current** [Section 4/8 area, taught here]:
- A house gets **two hots + one neutral**. Each hot is **120 V to neutral**; the two hots are **180° out of phase**, so **hot-to-hot = 240 V**.
  - Voltmeter hot-to-neutral → **120 V**; hot-to-hot → **240 V**.
- **Why both 120 and 240 are brought in:** to serve both small loads and large loads. Small loads (fridge, TV, lighting) need only **one hot + neutral = 120 V**. Large loads — EV charger (30 A+), dryer, electric range, baseboard heater — are **240 V**, taking **one wire from each hot** (no neutral needed for a pure-240 device like an EV charger → run 2 conductors; but range/dryer need **2 hots + neutral = 3 conductors** because parts of them run at 120 V).
  - Examples: a **dryer's motor and timer/lamp run at 120 V**, the **heating element at 240 V**; an electric **range** likewise (clock/oven-light 120 V, heating 240 V). That's why range/dryer use **3 wires** (2 hot + neutral), while a pure-240 V load (e.g. EV charger) uses **2 wires** (bonding aside).
  - [teacher cultural note] An Iranian 240 V resistive samovar works fine here on 240 V; **motor** appliances may burn out because of the **frequency difference (Iran 50 Hz vs Canada 60 Hz)** — motors here spin ~10 rev/s faster.
- **Neutral current rule (the key worked concept):** the neutral carries the **difference** (imbalance) of the two hot legs, not their sum.
  - If both legs draw **100 A** and are **180° out of phase**, neutral current = **0 A** ("they cancel; each plays the role of phase and neutral for the other").
  - If one leg = 100 A and the other = 80 A, neutral carries **20 A** (the difference). The **worst case** is one leg loaded and the other zero → neutral = the full leg current → so **the neutral is sized the same as the hots** in dwellings. [teacher] Don't double the neutral for two hots; the worst case is only equal-to-one-leg.
  - **This answers a student's renovation question:** when running a circuit with **two hots**, in a *dwelling receptacle* circuit you still use a **same-size neutral** (because one leg could pass 15 A while the other passes nothing). **[teacher]** The *utility*, however, by experience sizes the **incoming service neutral SMALLER** because total household imbalance maxes out around **30%** (e.g. if one leg is 100 A the other is ~70 A in worst typical case; normally e.g. 40 vs 35 → only ~5 A on neutral). A nationwide copper saving; that's why the utility's service neutral conductor is visibly thinner. (He distinguishes **branch receptacle wiring [same-size neutral]** from **utility service drop [reduced neutral]**.)

**3-phase wye: line-to-line vs line-to-ground & √3** [Section 2/equipment marking + transformer]:
- Iran: single-phase 220 V, three-phase 380 V. Canada labels phases **A, B, C** (Iran R/S/T); the three lines are **L1, L2, L3**.
- **Two things to keep in mind [teacher, emphasized repeatedly]:**
  1. The **larger** of two stated voltages is always the **line-to-line** voltage; the **smaller** is **line-to-ground (line-to-neutral)**.
  2. Their relationship is **√3** (≈ **1.732**). Line-to-ground × √3 = line-to-line, everywhere in the world. (Because it's a **vector** sum, not arithmetic — adding two 120° -displaced windings gives ×√3, not ×2.)
- Examples: **120 V × √3 ≈ 208 V** (so a **120/208 V** wye system); **347 V × √3 ≈ 600 V** (so **347/600 V**). In a marking like "**208/120**" or "**600/347**," the bigger number is L-L, smaller is L-G.
- **Exam application [teacher]:** Given a circuit is "**120/208**" → it's **3-phase** (ratio √3). Given "**120/240**" → it's **single-phase** (ratio 1:2). A question may give the two voltages and ask single- or three-phase; deduce from the ratio (1:2 = single-phase; √3 = three-phase). He notes the exam usually tells you, "but I wouldn't" — a real electrician should know. This matters for **current calc**: single-phase I = P/V; three-phase I = P/(V·√3) — the **√3 divisor** depends on knowing it's three-phase.
- **120/240 has no √3 relationship** (240 ≠ 120·√3 ≈ 208), so 120/240 is **not** three-phase — it's the **split-phase single-phase** system. Iran has no native 120/240 system (only 120 single-phase or 208 three-phase equivalents).

**Transformer basics** [Section 8 area, taught here]:
- A **three-phase transformer = three single-phase transformers** ("rule": a 3-phase transformer is built from 3 single-phase units), each with a **primary** and a **secondary** winding.
- For the residential supply, the secondary is wound for **240 V**, **center-tapped** so the midpoint is the **0 (neutral)** point; **120 V** appears from center to either end → that's how **120/240** is created. The winding is linear: the more turns from the tap, the higher the voltage (tap at various points to get 30 V, 60 V, 120 V, etc. — he recalls hand-winding transformers in trade school).
- The **center-tap / 0 point is grounded** — connected to earth (the utility electrode at the street near the transformer). The grounded neutral becomes the supply neutral. The house then receives **L1, L2 (two hots) + neutral**, giving **120 V** (hot-neutral) or **240 V** (hot-hot).
- [teacher] You may **not** call 120/240 "two-phase" — it's **one phase** (single-phase, split). A customer who said "240, so it's two-phase" was wrong; "two-phase" is a misused term.

**Voltage classes (ELV / LV / HV)** [Section 0 definitions, Rule 2 area]:
- **Extra-Low Voltage (ELV)** [AC]: any voltage **not exceeding 30 V** — i.e. **30 V is included** in ELV.
- **Low Voltage (LV):** **above 30 V and not exceeding 1000 V** — i.e. **1000 V is included** in LV.
- **High Voltage (HV):** **above 1000 V**.
- **Exam trap [teacher]:** questions target the **boundary values**. Classic: "**Is 30 V Low Voltage or Extra-Low Voltage?**" → **Extra-Low Voltage** (because the rule says "not exceeding 30 V" — 30 is the inclusive top of ELV). Students wrongly pick LV. Be clear on inclusive boundaries beforehand. (He also notes "low voltage" is colloquially misused for 24 V control wiring, which is technically ELV by this definition.)

**Deviation / Special permission — Rule 2-030** [standard CEC]:
- **Deviation** = "departure/anhirāf" from the code. You can deviate from the code **only where the code itself permits it** (the code tells you the rule **"...unless special permission..."**). You may **NOT** deviate on your own anywhere you like — otherwise everyone would self-deviate.
- **Rule 2-030 (paraphrased reading):** "In any case where deviation is **necessary**, **special permission shall be obtained ... before proceeding with the work** ... and the special permission **shall apply only to the particular installation for which it is given**."
- [teacher] Who gives it depends on the work: for **city/municipal work**, the **inspection authority/city**; for **refinery/petrochemical** work, the **client's** authority. Two real cautions he gives:
  1. A permission obtained for **last year's project cannot be reused** this year — it applies **only** to that specific installation.
  2. If you can't do something per code, you get permission **first**, *before* starting.
- [teacher] **Deviations are only in the technical part** of the code (technical = where engineering judgement applies), not the administrative part.

**Equipment marking — line-to-line / line-to-ground** [Rule 2-104 area]:
- [standard CEC, paraphrased] **Electrical equipment shall be marked** with its **line-to-line and line-to-ground voltage** (some single-phase devices show only 120 V). Example: a dryer/range may be marked **"125/250"** (i.e. it's rated up to 125 L-G / 250 L-L). [teacher] He explains the utility provides slightly higher voltage at the nearest house ("**240/120** at the first house, dropping with distance" — because longer runs / farther from source lower the voltage) which is why ratings carry a margin, and both single- and three-phase versions are marked accordingly.

**Dwelling voltage-to-ground 150 V rule** [Rule 2-130 area — exam-important]:
- [standard CEC, emphasized as "this is itself an exam question, common and important"] In a **dwelling unit**, the **voltage-to-ground shall not exceed 150 V**.
- Reasoning he gives: 240 V line-to-line → 120 V line-to-ground (well under 150). But a check: if it were "still in the hot range" the 347/600 system would give 347 V to ground — far above 150 — so it's barred for dwellings.
- **Exam trap:** students wrongly answer "**120**" or "**125**" because that's what's marked, but the **rule's stated limit is 150 V to ground**.
- **Exceptions (the rule says "**except**"):** [teacher reading]
  - For an **apartment / similar building** whose **demand exceeds 250 kVA**, higher voltage-to-ground is permitted **provided a qualified resident electrician** is on staff (someone who responds to electrical faults — not merely living there), **and the higher voltage shall not exceed 347/600** (i.e. ≤ 347 V to ground / 600 V line-to-line). (He notes single-family homes never hit this.)
- **600 V (347/600) usage in apartments [standard CEC]:** You may **bring 347/600 V into an apartment building** only because a **transformer steps it down to 120/240** for the dwellings. This does **not** mean occupants use 600 V directly — household appliances are all low-voltage (e.g. fridge ~120 V, TV, lamps). However, **347/600 V may be used directly for FIXED equipment in a dwelling** when it serves **central** systems: **central heating, central hot-water, central air-conditioning** (located in the building's utility room, not the suite) — it would be wasteful to add an extra transformer for central equipment that can use 600 V directly. The **150 V-to-ground limit** is the governing red-line for the suite itself.

---

### Section 2 — Working space, enclosures, gas-appliance clearances

**Motor-control vs Service working-space heights** (don't confuse) [teacher]:
- **Motor control** that is **exposed live parts** — its working/clearance heights are set in the motor section. (He warns students not to confuse the two-metre figures.)
- **Service equipment** (Section 6) must be located where its **height is not less than [~2 m]** (low enough to operate). A homeowner can reach up to reset a breaker without being shocked because only the breaker handle is exposed. — He stresses these are *separate*, clearly-distinguished exam items.

**Transformer working space** [standard CEC]:
- For transformers **rated greater than 50 kVA**, the **minimum horizontal working space is 1 m** ("for transformer rated greater than 50 kVA, minimum horizontal space = 1 m").
- For **50 kVA and below**, the 1 m is **not mandatory**; **51 kVA and above** it's required (boundary: above 50). The 1 m applies on the side where the **conductors enter/exit** (where you'd service the cables) — e.g. a 3-phase transformer with neutral = 4 conductors in / 3 out; keep the 1 m clear only on that working side (other sides may touch the wall).

**Rule 2-328 — clearance of electrical equipment from gas pressure-relief devices** [Appendix B, exam-relevant]:
- Near a home's entrance there's a gas **regulator** with a **vent** that can have a small **leak/leakage**; nearby **electrical equipment** (switches, A/C units, thermostats) is a **source of ignition** (spark — "ignition" like a car's ignition). Switches normally spark; sparkless switches exist as the exception.
- **Required clearances [standard CEC, via Appendix B]:**
  - **Natural gas:** at least **1 m** from the relief vent (he also cites **3 m** as the table figure for one case — see below).
  - **Propane:** more dangerous than natural gas because **propane is heavier than air** and **pools at ground level** (natural gas is lighter, **rises** and disperses) — so propane needs **greater** clearance.
  - **The actual numbers are NOT in the section text** — they're in **Appendix B** (he tells students to mark Appendix B for Rule 2-328). His read of the Appendix B figures: **natural gas 3 m, propane 3 m**, with a **0.3 m** figure tied to a specific listed/approved regulator condition ("if the regulator is listed/approved, the distance can be reduced to 0.3 m"). *(Numbers per his Appendix-B reading; treat the listing-reduction as the key concept.)*
- **Method takeaway:** if a clearance isn't found in the main rule, go to **Appendix B**.

**Enclosures — Rule 2-400 & Table 65** [standard CEC]:
- **Enclosure** = any housing that holds electrical equipment — from palm-sized boxes to large cabinets/panelboards, motor frames ("motor enclosure"), transformer bodies. What's *inside* and its current don't matter for selection; **where you install it (the environment) governs the type.**
- Enclosures are **rated/graded ("Type" numbers)** by environmental condition. **Table 65** lists this (left column = environmental condition; the body = which enclosure **Types** are permitted).
- Section 2 has **only two tables: Table 56 and Table 65.** [teacher mnemonic] **Table 56 = working space; Table 65 = enclosures.**
- Environmental-condition examples and the worked logic:
  - **Dripping** (drops falling, e.g. a greenhouse with condensation overhead) → cannot use **Type 1**; use the types not "struck out" (he refers to the table's struck-through/allowed marks). Several types qualify (e.g. **1, 2, 4, 5, 6** for some conditions).
  - **Splashing water** (e.g. installing a panel/motor in a **car wash**) → use the allowed types (the "last four" in his read).
  - **Corrosive** (e.g. the car wash also has corrosive shampoo) → use a **corrosion-resistant** type, e.g. **Type 4X** (and **6P**). He calls "**4X + corrosive shampoo present**" about **the hardest version** of this question.
- Enclosure types split into **three groups:** **Indoor only**, **Outdoor only**, and **Indoor & Outdoor** (those usable outdoors are also usable indoors). Exam may give a scenario (outdoor / wet / corrosive) and ask which **Type** is acceptable; know all four common ones. He notes practice exercises were provided in Section 2 for this.

---

### Section 4 — Conductor ampacity (the most "conceptual" section)

[teacher] Section 4 is the **"mother" section** and the **most conceptual** of the book — you must understand it, not just memorize. (He jokingly rates its required academic level as ~5th-grade — mostly distances/lookups in other sections, but Section 4 needs real understanding.) It answers two reciprocal questions: **given a current → find conductor size**, and **given a conductor size → find its ampacity**, all via tables.

**Which table to use — the selection procedure [teacher's flowchart]:**
Two binary choices decide the table:
1. **Installation method:** **Free air** vs **in a raceway / cable** ("raceway or cable").
2. **Conductor material:** **Copper** vs **Aluminum**.

| Condition | Table |
|---|---|
| **Free air + Copper** | **Table 1** |
| **Free air + Aluminum** | **Table 3** |
| **Raceway/cable + Copper** | **Table 2** |
| **Raceway/cable + Aluminum** | **Table 4** |

(So: Tables **1 & 2 = copper**; Tables **3 & 4 = aluminum**. Tables **1 & 3 = free air** (single conductor); Tables **2 & 4 = raceway/cable**.)
- These four (Tables 1–4) are the **everyday tables**, used for **conductors above ground**. He says **~90% of real work is the raceway/cable case (Table 2/4 area), ~10% free air**, and mostly copper — but the **exam will deliberately ask the less-common combinations** (e.g. free-air aluminum), so learn all four; the method is identical.
- **Underground** large cables use **Tables D8, D9, D10, D11** (in Appendix D) — he does **not** teach these (no exam questions from them), the dividing line being above-ground (Tables 1–4) vs underground (D8–D11).

**Definition of "free air" [teacher]:** a conductor that is **not in contact with anything for its full length / 360° around** (e.g. overhead street wires on poles — supported only at points, free along the run). Anything else (in conduit on a wall, running to a motor through conduit) is **not** free air. Free air is the single recognizable case; everything else is the "other" case. Free-air tables (1 & 3) are tabulated for a **single conductor** because free-air runs are typically single conductors.

**Reading the table — temperature column [worked]:**
- A given size (e.g. **#14**) has different ampacities by **insulation temperature rating** (the conductor's temperature column, not ambient):
  - **60 °C** insulation, #14 → **15 A** [standard CEC]
  - **75 °C** insulation, #14 → **20 A**
  - **90 °C** insulation, #14 → **25 A**
- Industrial/construction work mostly uses **75 °C and 90 °C** insulations.
- Insulation type codes encode this: e.g. **"R"** = rubber/plastic, **"RW90"** ≈ 90 °C wet-rated; **"TW75"** = 75 °C; **"T90 Nylon"** = 90 °C. The number in the name is the insulation temperature.
- Worked size example: **#2/0 ("two-aught")**, **90 °C**, **Table 1 (free-air copper)** → **195 A** at 90 °C (he initially mis-navigates to Table 2 / wrong column, illustrating the common error of reading the wrong table/column). He stresses table reading is error-prone at first ("it took me a month to get one of these right when I came back to it") — don't panic.

**Correction factors (two kinds, BOTH due to heat):** [standard CEC + teacher]
1. **Grouping / number-of-conductors** (more than the table's base) — **Table 5C** (he says "5C"). Tabulated ampacities assume **not more than the base number of conductors** (free air base = the single/few; for raceway, **"not more than three" insulated conductors** at full value). Beyond that, heat from cable-to-cable derates ampacity.
   - **Free air (Table 1):** base value is for **single conductor**. If **2 conductors** → ×**0.9**; **3** → smaller; **4** → smaller; e.g. (his worked numbers) free-air #?? 300 A → 2 conductors → ×0.9 = **270 A**; 4 conductors → ×0.8 = **240 A**. (Exact factors from Table 5C; "above four/three conductors" begins the derate.)
   - **Raceway/cable (Tables 2 & 4):** **1–3 conductors → factor 1 (no derate)**; **4–6 → ×0.8**; **7–24 → ×0.7**; and so on down the table. (His worked example: 25 conductors → factor **0.6**.)
   - [standard CEC nuance] For **free-air SINGLE conductors**, grouping only applies if **2–4 single conductors are spaced closer than 25% of the largest cable diameter**. Example: a 4″ and a 2″ cable → 25% of 4″ = 1″; if spacing **< 1″**, apply the grouping derate; if the exam doesn't state the spacing/"if" condition, **do not** apply it. (He stresses: the word "if/where" governs — if the spacing condition isn't given, no derate.)
2. **Ambient temperature** (above the base **30 °C**) — **Table 5A** (he says "5A"). All tabulated ampacities assume **30 °C ambient**. Above 30 °C (e.g. 35 °C, 40 °C near a furnace), ampacity is derated by a Table-5A factor based on the conductor's **insulation rating**.
   - **Table 5A "round up to next-higher temp" rule:** if your ambient isn't a listed value, **go to the next-higher listed temperature** (don't round down — that would overstate capacity). E.g. **36 °C, 38 °C, or 39 °C → use the 40 °C row.**

**Applying both factors:** if a case has both (e.g. 10 conductors at 40 °C ambient), **multiply the base ampacity by both correction factors** (grouping × ambient). [teacher] **Order doesn't matter** (multiplication is commutative — "whether it's Khājeh-Ali or Ali-Khājeh"). Example he gives: base 15 A × 0.9 × 0.8.

**Worked examples he runs:**
- **Free-air copper #2/0, 90 °C → Table 1 → 195 A** (then with grouping: 2 → 270/×0.9 etc. on a different base).
- **Ambient derate example:** an **RW90 cable** normally **350 A** at ≤30 °C; at **35 °C ambient (90 °C insulation column)** apply Table-5A factor (he reads ≈ **0.96**) → max ≈ **338 A** ("288/338"-area number) — *the worked value is approximate from his table read*; the **concept** is: above 30 °C you can't pass the full rated current.
- **Section 4, Question 1 (his answer):** Aluminum-sheathed cable, **free air**, copper conductor → free-air + copper = **Table 1**; size **2/0** at **90 °C** → **195 A**; book answer **195 A**.
- **Section 4, Question 4 (his answer):** **25 conductors**, size **#4**, **RW90**, copper, in a **raceway** (EMT) → raceway+copper = **Table 2**; #4 at 90 °C = **95 A** (value for 1–3 conductors); **25 conductors → factor 0.6** → 95 × 0.6 = **57 A**.

**Aluminum vs copper / single vs grouped (table header reminders he repeats):**
- Free-air tables (1 & 3) values are **per single conductor**.
- Raceway tables (2 & 4) note **"not more than three insulated conductors"** at the base value; **≥4** → grouping factor; both also **"based on ambient temperature 30 °C"** → above 30 °C apply ambient factor too.

---

## 3. Answers / clarifications this audio provides for the question bank

Specific exam-answer facts the teacher states (use to fill bank gaps):

- **Pass mark = 70%** (70 itself passes; 69 and 68 fail). All questions multiple-choice.
- **Object of the CEC** = establish **safety standards for installation and maintenance of electrical equipment**; prevention of **fire and shock hazard**.
- **ELV boundary: 30 V is INCLUSIVE → 30 V is Extra-Low Voltage** (not Low Voltage). ELV = "not exceeding 30 V."
- **LV = above 30 V up to and including 1000 V**; **HV = above 1000 V** (1000 V is inclusive in LV).
- **Dwelling voltage-to-ground limit = 150 V** (Rule 2-130 area) — not 120/125. Exception: apartment/similar > 250 kVA demand with a qualified resident electrician may go higher, **not exceeding 347/600**.
- **347 V × √3 ≈ 600 V**, **120 V × √3 ≈ 208 V**; in a "x/y" marking the **larger = line-to-line**, **smaller = line-to-ground**.
- **Ratio test:** 1:2 voltage pair = **single-phase** (e.g. 120/240); **√3** pair = **three-phase** (e.g. 120/208, 347/600). **120/240 is single-phase (split), NOT two-phase, NOT three-phase.**
- **Bonding** = **permanent, low-impedance path** connecting **non-current-carrying metal parts** (exact-wording / fill-the-blank question — answer must include "low-impedance," "permanent," "non-current-carrying").
- **Overcurrent protects conductors/wires; Overload protects the motor.** Fuse = melts/one-shot; breaker = reusable/resettable.
- **GFCI = Ground Fault Circuit Interrupter; Class A**; trips on hot-vs-neutral current imbalance; reference trip ~**5 mA** (he frames "≈4–6 mA," 6 mA with ~4 mA tolerance). GFCI breaker has an extra neutral-return (pigtail) wire.
- **Feeder** = portion of circuit **between the service box and the branch-circuit overcurrent device** (boundary: service-box breaker → sub-panel breaker).
- **Branch circuit** = wiring **after the final overcurrent device** (no further fuse/breaker beyond it) to the outlet.
- **Service** = incoming supply + **service equipment** (main panel). Service equipment height: **not less than ~2 m** (Section 6).
- **EMT is NOT a conduit** (definition excludes EMT). Conduit = round-cross-section raceway. **Raceway** is a generic category, not a purchasable item.
- **Emergency lighting / smoke detectors / exit-sign colours are provisioned in the NBC (National Building Code), not the CEC.** BUT **conductor size / overcurrent for emergency lighting is in the CEC.**
- **Transformer working space (horizontal) = 1 m for > 50 kVA** (required at 51 kVA+; not mandatory at ≤50 kVA), on the conductor entry/exit side.
- **Section 2 has exactly two tables: Table 56 (working space) and Table 65 (enclosures).**
- **Enclosure Type 4X** (and **6P**) = corrosion-resistant choice (e.g. car wash with corrosive shampoo). Type 1 not allowed for dripping/wet.
- **Fence height ≤ 1.8 m → Rule 26-304** (found via index "F"; "chain-link fabric" is filed under fencing).
- **Ampacity #14 copper:** 60 °C = **15 A**, 75 °C = **20 A**, 90 °C = **25 A**.
- **Section 4 table selection:** Free-air copper = **Table 1**; raceway/cable copper = **Table 2**; free-air aluminum = **Table 3**; raceway/cable aluminum = **Table 4**.
- **#2/0 copper, free air, 90 °C = 195 A** (Table 1).
- **#4 copper, RW90, raceway, 90 °C = 95 A** base; with **25 conductors (×0.6) = 57 A**.
- **Correction factors:** base ambient = **30 °C** (Table 5A); raceway grouping base = **3 conductors** (Table 5C); 1–3 conductors factor = 1, 4–6 = 0.8, etc. Apply **both** factors by multiplication, **order irrelevant**. **Round ambient UP** to the next listed temperature.
- **Neutral in a 120/240 dwelling branch circuit is sized the same as the hots** (worst case = full one-leg current); the **utility service neutral is reduced** (typical imbalance ≤ ~30%).
- **Metric/Imperial conversion** question type exists (e.g. **lux vs foot-candle**), found in the front-matter metric-units table.
- **~6–7 exam questions come from Appendix B**; ~**20% of questions are table lookups**; **Block B + Block C ≈ 60 questions** and decide the pass.

---

## 4. Memorable mnemonics / phrasings he uses

- **"Overcurrent protects the cables/wires; Overload protects the motor."** (His explicit "moral of the story.")
- **"Bigger voltage is always line-to-line; smaller voltage is always line-to-ground; their relationship is √3."** (Repeated as a "keep in mind.")
- **"√3 ≈ 1.732"** → line-to-ground × √3 = line-to-line; "1:2 means single-phase, √3 means three-phase."
- **Vector, not arithmetic:** two 120°-displaced 120 V windings give ×√3 (≈208), not ×2 (≈240) — "because it's a vector sum."
- **Table-number mnemonic:** *"Section 2 has only two tables — **56** and **65**; 56 = working space, 65 = enclosures"* (the digits are reverses of each other, "easy to remember").
- **Table-selection grid:** *"1 & 2 are copper, 3 & 4 are aluminum; 1 & 3 are free air, 2 & 4 are raceway/cable."*
- **"Whether it's Khājeh-Ali or Ali-Khājeh"** — order of multiplying correction factors doesn't matter.
- **"The exam is like a blood test"** — reading a student's score breakdown reveals exactly where they're weak.
- **Free air = "not touching anything 360° around for its whole length"** (overhead street wire on poles).
- **"EMT is not a conduit — it's a thin-walled tube; its weak point is the thin wall, its strong point is it's ~⅓ the price, lighter, easier."**
- **Propane vs natural gas:** *propane is heavier than air and pools at the ground → more dangerous → needs greater clearance; natural gas is lighter and rises.*
- **Neutral cancels:** *"if both legs are 100 A and 180° apart, the neutral carries 0 — each acts as phase and neutral for the other."*
- **Receptacle screws:** *brass/gold screw → smallest slot → hot ("most dangerous wire", black); silver screw → neutral (white); biggest slot/green → bonding.* **Split = break the middle tab.**
- **"Get fluent in the book first, then solve problems"** — his core teaching philosophy vs Q&A-drilling schools.
- **Use cartoon/animation YouTube videos** for bonding/grounding/armoured cable — "I understand animations better myself."
- Two physics axioms for grounding/bonding: **"electricity always wants to go to earth"** and **"electricity always takes the path of least resistance."**

---

## Session 3 (جلسه ۳)

Continuation of the same coaching series. This session **finishes Section 4** (the items not covered in Session 2-2 — Section-4 Question 3, neutral-supported cables, flexible cords / equipment wire, portable power cable & DLO, conductor colour identification, and the critical **termination/terminal-temperature limitation rule**) and then **starts Section 6 (Service & Service Equipment)**, taught largely from photos. The recording closes with re-exam logistics and personal chatter (excluded here). Tags as before: **[teacher]** = his own assertion/method; **[standard CEC]** = matches the published code; numbers flagged *approx.* are transcription-noisy.

### Concepts taught

#### Section 4 — finishing conductor ampacity

**Section 4, Question 3 — two different insulation temperatures in one conduit [teacher, worked]:**
- Setup (his reading): *"What is the maximum ampacity of each conductor"* when conductors of **two different temperature ratings** share **one rigid steel conduit** — three **#4** conductors and three **#1** conductors, **six total**, copper, in raceway → **Table 2**.
- **Key conceptual rule he drills:** when a raceway contains conductors of **two different temperature ratings, you must base everything on the LOWER-rated conductor** — otherwise the cooler-rated cable's insulation overheats and degrades. ("If the lower one is rated 75 °C and you let the run reach 90 °C, after about a week the 75 °C insulation is damaged.") He praises the student for deriving this independently, says *"this is itself the exam question / exam definition."*
- Worked numbers (Table 2, copper, raceway):
  - **#4 at 75 °C → 85 A** (he first mis-states 65 A, then corrects to **85 A** by re-checking the table — flagged as his own live correction).
  - **#1 at 75 °C → 130 A** (he uses 75 °C for *both* because the lower-rated conductor governs).
  - **6 conductors → grouping factor 0.8** (Table 5C).
  - **#4: 85 × 0.8 = 68 A**; **#1: 130 × 0.8 = 104 A.**
  - **Each conductor is reported separately** — do NOT sum them. Final answer: **68 A and 104 A.** [teacher]
- Earlier in the worked talk he tied this to a #4 at the three columns: #4 at **60 °C / 75 °C** → the table value steps down as the column temperature drops; the takeaway sentence he wants memorized: *"If I want it not to exceed 75 °C, I can only pass [the 75 °C-column ampacity] through it; and conversely, if I pass that current it heats only to 75 °C."*

**Neutral-supported cables — Tables D36A / D36B [teacher]:**
- **Neutral-supported (messenger) overhead cables** are the building-feed aerial wires where the **bare neutral conductor doubles as the mechanical support (messenger)** — wrapped around the standoff/insulator, clamped, and the two/three hots hang off it. ("It does both the neutral job and the support job.")
- These are **NOT** sized from Tables 1–4 — they have their **own tables: Table D36A and Table D36B** (in Appendix D).
- **Order is REVERSED vs Tables 1–4 [teacher, flagged as a common error]:** Tables 1–4 list **copper first, aluminum second**; but **Table D36A = ALUMINUM**, **Table D36B = COPPER**. Reason: aerial/messenger cables are usually aluminum (lighter for overhead spans), so aluminum is listed first. He admits he personally kept getting this wrong (looking in the copper table by habit).
- Cable naming: prefix **"NS"** = Neutral-Supported. Two temperature columns (**75 °C and 90 °C**). Conductor count drives the ampacity column:
  - **Duplex** = 2 wires (one hot + neutral) — e.g. a small corner store needing only one phase.
  - **Triplex** = 3 wires (two hots + neutral) — the normal house feed (120/240).
  - **Quadruplex** = 4 wires (three phases + neutral) — three-phase service.
- **Worked lookup [teacher]:** NS75 aluminum, neutral-supported, for a **200 A** house → read the table to the size that covers 200 A (he points to "size 2/0" area). *(Exact size approximate from his table read.)* He repeats this is the "20% of the exam that is easy table lookups."
- **Correction factors for D36A/B:** the table itself prints the **ambient-temperature** factors in its note (he reads ≈ **1.0 at 30 °C, ~0.94 at 35 °C, ~0.88 at 40 °C** — *factors approx. from transcript*). **There is NO separate grouping/number-of-conductors factor** for these cables — the table is already built for the duplex/triplex/quad case, so don't apply Table 5C on top. [teacher] He notes this in contrast to Tables 1–4 where you add grouping.

**Flexible cords & equipment wire — Table 12 [teacher]:**
- **Cord** = ordinary flexible wire; **flexible cord** = very bendable wire used where movement happens (elevator-car trailing cable that travels up/down, Christmas-tree light strings).
- **Equipment wire** = same family — the attached lead/pigtail that hangs off a purchased appliance (fridge, TV, motor) that you plug into the board.
- **Ampacity for BOTH flexible cord AND equipment wire = Table 12** (one table, split into a flexible-cord section and an equipment-wire section). Mark Table 12 for both. [teacher]
- Smallest sizes go **much smaller than #14**: Table 12 includes **#18, #20, even #27**.
  - Christmas-tree string ~**2 A → about size 20**.
  - **Tinsel cord** (very thin decorative LED-blink light wire) → **size 27, ~0.5 A only**; not even allowed for general 120 V use.
- **Grouping factors are printed in Table 12's own note** (1–3 cond. = full value; 4–6 → ×0.8; 7–24 → next factor — "the same percentages as Table 5C"). [teacher]
- Rounding reminder he repeats: when the exact ampacity isn't a listed size, **go up to the next-larger size** (as in the earlier 25 A → size that covers it example).

**Portable power cable & DLO cable — Table 12-area / Appendix [teacher]:**
- **Portable power cable** = movable ("portable") **high-voltage/power** cable, e.g. a generator feed temporarily run to a building during a power-equipment outage (his water-treatment-plant project example). "Power" here means **high voltage / power-level**, not just any energized wire. Ampacity from its dedicated table; look up by insulated-conductor count (1 through 6) and voltage column (e.g. a 2000 V column). Rare exam item — *"I haven't seen a question from it, but know it exists."* [teacher]
- **DLO cable** ("Diesel Locomotive" cable) — a flexible, durable cable he uses in the trade (especially **cable tray** wiring); the code even cites it for cable-tray use. Originally for diesel locomotives. Ampacity from a simple two-column table (size vs ampacity). Very rare exam item. [teacher]

**Conductor colour identification — Rule 4-036 area [standard CEC, paraphrased by teacher]:**
- **"Identified conductor"** ≠ casually "neutral." Precisely, an **identified conductor is the white/grey conductor that is INTENDED to carry the return current** (in a 2-wire single-phase circuit the return on the white wire equals the current on the black). The book uses **"identified," not "neutral,"** in the colour rule because a true *neutral* only carries the *imbalance* (in a 3-wire system), whereas an identified conductor may carry the full return. [teacher: "neutral means it's resting — the difference of the two legs; if all current returns on it, it's working, so it's an *identified* conductor, not a neutral."]
- **"Ungrounded conductor"** = the **hot** conductors (the ones NOT grounded). The neutral/identified conductor is the one that gets grounded. Exam wording trap: "ungrounded" = hot.
- **Bonding/grounding conductor colour:** **green, OR green with one or more yellow stripes.** [standard CEC]
- **Single-phase / DC, 2-wire:** hot = **black** (or red); identified (white) = **white**, used **"when identified"** (i.e. when a return/neutral is needed). A pure-240 V 2-wire load (EV charger) needs **no white** — two hots only.
- **Single-phase, 3-wire (e.g. dryer/range):** two hots **black + red**, plus **white** when a neutral is required (because the 120 V parts — motor, timer, oven light — return on it while the 240 V element does not). Count "how many wires" excludes bonding: **3-wire = black, red, white.**
- **Three-phase colour order (matters in the field):** **Phase A = red, Phase B = black, Phase C = blue**, in that order, plus **white** if a neutral is needed. It says **"if neutral required," NOT "if identified required"** — because a true 3-phase neutral only carries imbalance (it rests when balanced). [teacher contrasts this wording deliberately with the single-phase "identified" wording.]
  - A/B/C labelling is **arbitrary/by convention** — "in Canada the phase rotation swaps 60 times a second" — you just agree which is A, B, C.
- **Exam form [teacher]:** usually pure colour-order multiple choice — *"which is correct: red-black-blue / blue-black-red / blue-red-black?"* Field importance: lighting & receptacles run **black + white**; dryer runs **black-red-white**. Mis-wiring white (assumed neutral) when it is actually energized is dangerous.

**TERMINATION / TERMINAL-TEMPERATURE LIMITATION — Rule 4-006 [standard CEC; teacher flags as one of the most important & most-failed rules]:**
- He tells students they may not memorize most of the book, **but Rule 4-006 they MUST memorize** — "you'll get 2–3 questions tied to this and it's critical on the job; even experienced engineers get it wrong." Mark it.
- **The core principle:** equipment terminals (the two screws/lugs you land the cable on) have a **maximum terminal temperature**; the conductor must be sized so that **neither the cable overheats the terminal nor the terminal overheats the cable.** Therefore you must **size the conductor based on the TERMINAL'S temperature column — not the cable's own higher insulation rating.**
- **Worked example [teacher]:** a transformer/equipment terminal marked **75 °C**, you want to use **armoured copper cable** in a raceway (Table 2). Even though the cable's insulation may be rated 90 °C, you must read the **75 °C column** (because the terminal is the limit). You **may** use a 90 °C-rated cable, but its **size is still set by the 75 °C ampacity** — you do NOT get to read the 90 °C column and pick the smaller wire. Picking the 90 °C size (e.g. size 14 for a 25 A job) is the classic mistake; the right answer uses the 75 °C-column size (e.g. size 12). [teacher]
- **When the terminal temperature is NOT marked — Rule 4-006 default (the part everyone misreads):** read the rule's "where the maximum terminal temperature of equipment is not marked" clause:
  - **Use 60 °C** if the equipment is rated **100 A or less** (he renders the threshold as "100 amp" — *the transcription is noisy here; standard CEC threshold is 100 A / #1 AWG*).
  - **Use 75 °C** if the equipment is rated **above that threshold.**
  - **Exam trap he highlights:** if the equipment is **exactly at the threshold (100 A)**, you take **60 °C** (because the rule says "*not more than* / *and below*" → the boundary value falls in the 60 °C group). Students wrongly pick 75 °C. He stresses he has had to repeat this three times to students who still got it wrong.
- **Restated takeaway:** *"Higher cable insulation rating lets you USE the cable, but the BASE for sizing is always the terminal temperature (or the unmarked default)."* The armour/insulation rating is "never the deciding factor" for the terminal-limited size.

#### Section 6 — Service & Service Equipment

**Definitions [standard CEC, paraphrased]:**
- **Service** = the portion where power **enters the building** — the incoming supply from BC Hydro (or other utility).
- **Service equipment** = the equipment in that path: the **meter**, the main panel/board, disconnect, etc.
- Taught heavily with photos ("this section is meaningless without pictures — the book describes distances between things you can't even visualize from text").

**Number of supply services — Rule 6-102 [standard CEC]:**
- **Maximum number of supply services to a building = ONE.** Code wording: *"two or more supply services of the same voltage shall not be run to any building."*
- **Exceptions (don't get tricked):** a **separate supply is permitted for a fire pump** (so that a fire burning the building's main service doesn't kill the fire pump — fire pumps in high-rises / malls have their own service, own panel, in a safe protected location, breaker run "reverse/locked" so it's hard to accidentally cut). Also exceptions for **industrial establishments and other large/complex structures** (a big mall may legitimately take service at two or three points). [teacher: "if the question asks the *maximum number of services permitted*, the answer is **one** — the others are exceptions."]

**Number of consumer services — Rule 6-104 [standard CEC]:**
- *"Number of consumer services shall not exceed FOUR"* — one supply service may feed **up to four** consumers. [teacher]
- Apparent contradiction (one service in, but four consumers off it) resolved by reading both rules together.
- For buildings with **many units** (10, 20, 30 suites) the rule says **"unless..."** → you obtain a **deviation/special permission** (with calculations submitted to the authority). The code itself authorizes the deviation here. [teacher ties back to Session-1/2 deviation teaching: "deviation is allowed exactly where the code says it's allowed."]

**Overhead vs underground service [standard CEC + teacher]:**
- Two ways power enters: **overhead service installation** (aerial wires) or **underground service installation** (buried cable). Older areas = overhead; newer areas (last ~5–20 yrs) = underground (lower maintenance, survives storms, cheaper long-term).
- **Connection point** = where the utility hands off:
  - **Overhead:** the owner/electrician hangs a length of conductor out of the conduit; the inspector approves; then the utility splices on at that **connection point**.
  - **Underground:** the connection point is **inside the meter** — no splicing allowed mid-run, underground or in the conduit.
- Overhead service drops are aerial **neutral-supported cables** → sized from **Tables D36A/D36B** (ties back to earlier in the session).

**Service mast & drip loop — Rule 6-112 (+ Appendix B) [standard CEC; many exam questions]:**
- **Service mast** = the conduit projecting above the roof that the overhead service lands on; its **upper portion must be METAL** ("the part above is the mast; it can be PVC or metal lower, but the mast itself is metal"). [standard CEC]
- **Service mast conduit minimum size = 63 trade size (= 2.5", rigid steel conduit).** He clarifies **"63" is the THREAD/TRADE size, not 63 mm** — the actual measured diameter differs (e.g. ~62.5 mm rigid, ~60.5 mm flexible); the trade just calls it "63" or "2.5". *(Standard CEC service-mast minimum is rigid metal conduit of this trade size.)*
- **Drip loop** = the **U-shaped loop** left in the service conductors so rain/snow water drips off the bottom of the loop instead of tracking into the building (waterproofing degrades over years). Required.
- **Lengths/clearances he states [standard CEC, some approx. from transcript]:**
  - **Minimum free conductor length left for the utility at the head = 750 mm**, complete with drip loop. (Rule 6-302 sub-rule 3 — *"minimum length 750 mm."*)
  - **Drip loop can hang down up to 600 mm** (and not less). 
  - **Clearance between roof and the supply-service attachment ≥ 915 mm** (Appendix-B note on Rule 6-112), reducible to **600 mm** for the drip-loop portion in the listed case.
  - At the **consumer service head**, the conductor emergence point must be **150 mm to 300 mm above** and **maximum 600 mm horizontally** from the attachment/insulator of the overhead service conductor (Rule 6-116 sub-rule(?) — *he cites this off the photo*). The standoff/insulator itself must sit within that 150–300 mm vertical band and ≤ 600 mm horizontal.
  - **Service-drop height above ground = 3.5 m to 9 m** depending on what's below: **3.5 m** over pedestrian-only areas, **~5 m** over driveways/where vehicles pass, up to **9 m max** (utility bucket-truck reach limit). *(Mid-range values approx.)*
  - If the mast/run **exceeds 1.5 m**, it needs a **guy wire** (back-stay) for support; and the mast's projection above its support must **not exceed 1 m** without guying (Appendix-B note: *"maximum unguyed projection."*).
  - Mounting to a wood structure: the backing block/ladder **shall be not less than 38 mm × 38 mm** ("not less than 38 mm in any dimension").
  - **Window/door/porch/balcony clearance = 1 m** on left, right, and below (a person opening a window mustn't reach it; firefighters breaking a window must be clear of it). The rule does **not** specify above. (Rule 6-112 sub-rule 4.)

**Consumer service conductor minimum size — Rule 6-208 (sub-rule 4) [standard CEC]:**
- **Consumer service conductor: minimum #10 copper / #8 aluminum.** [teacher: students think "#10 is tiny for a house" — but the rule says *minimum*, and it's not necessarily for a house; a small corner-store service can legitimately be that small. "It says minimum — go bigger if you want, nobody cares."]
- **Meter wire allowance:** leave **≥ 450 mm** of extra conductor at the meter (Rule 6-?? sub-rule, *"service conductor not less than 450 mm"* for the meter connection).
- **Conduit feeding the meter: minimum trade size 21** (must bear the cable weight; tradespeople think this is too small, but it's a *minimum*).

**Consumer service equipment location — Rule 6-206 (sub-rule c) [standard CEC]:**
- The service equipment **shall be located as close as possible to the point of entry** of the service conductors (because the un-protected service conductors — no fuse/breaker ahead of them yet — should travel the **shortest possible distance inside**).
- **Shall NOT be located in:** [standard CEC list he reads]
  - a **coal bin** (coal dust is conductive/combustible),
  - a **clothes closet**,
  - a **bathroom**,
  - any room whose **ambient temperature is normally above ~30 °C** ("storeroom that runs hot"),
  - **hazardous / dangerous locations** (e.g. near oil on a ship, flammable areas),
  - any space with **headroom less than ~2 m**, and **similar** locations.
- **Headroom note [teacher]:** service-equipment headroom **2 m** is enough because the parts are **not exposed live** — contrast with **Section 2's motor-control / exposed-live-parts** working space, which needs the larger figure. "Different rule, but don't confuse the two." He says the difference exists but "doesn't matter much" for pass/fail.

**Unprotected service conductor inside a building — embed 50 mm [standard CEC; teacher's exam tip]:**
- A consumer service conductor (which has **no overcurrent protection** ahead of it) **shall be located outside** the building as far as practicable. **Unless** it is **embedded in not less than 50 mm of concrete or masonry** — that **mechanical protection** substitutes for being outdoors.
- **General "embedded = 50 mm" rule of thumb [teacher]:** the word **"embedded"** appears in ~12 places in the code, and **every time, the figure is 50 mm.** Exam shortcut: *"if a question stumps you and one of the answers says 'embedded in 50 mm of concrete,' it's right about 90% of the time."* (His own anecdote: an aluminum-service-conductor question that looked like it was about aluminum oxidation was actually answered by "embedded 50 mm.")

### Answer clarifications for the question bank

New specific exam-answer facts the teacher states in Session 3:

- **Two different conductor temperature ratings in one raceway → size EVERYTHING on the LOWER rating.** (Worked: #4 & #1 copper, six conductors, → use 75 °C column for both.)
- **Section 4 Q3 final answer:** **#4 → 68 A** and **#1 → 104 A** (Table 2 at 75 °C: 85 A and 130 A, each × 0.8 grouping for 6 conductors, **reported separately, not summed**).
- **#4 copper, raceway, 75 °C = 85 A; #1 copper, raceway, 75 °C = 130 A** (Table 2). *(85 A is his corrected value.)*
- **Neutral-supported cable ampacity → Tables D36A / D36B**, NOT Tables 1–4. **D36A = aluminum, D36B = copper** (order reversed vs the usual copper-first convention).
- **Duplex = 2 wires, Triplex = 3 wires, Quadruplex = 4 wires** (for neutral-supported/messenger cable selection). House service = **triplex**.
- **Flexible cord AND equipment wire ampacity → Table 12** (single table, two sections). Smallest sizes include **#18, #20, #27**; **tinsel cord = #27 ≈ 0.5 A.**
- **Portable power cable** and **DLO cable** each have their own ampacity tables (rare exam items).
- **Bonding/grounding conductor colour = green OR green-with-one-or-more-yellow-stripes.**
- **"Ungrounded conductor" = the HOT conductor(s)** (the neutral/identified conductor is the grounded one).
- **"Identified conductor"** (white) = intended to carry return current; distinct from a true **neutral** (which carries only imbalance). Colour rule says **"black/red hot, white identified."**
- **Three-phase colour order: A = red, B = black, C = blue**, plus **white** if a **neutral** (not "identified") is required.
- **Rule 4-006 (terminal/termination temperature):** size the conductor on the **terminal's marked temperature** (or unmarked default), **not** the cable's higher insulation column.
- **Unmarked terminal default: 60 °C if equipment ≤ ~100 A; 75 °C if above.** At exactly **100 A → use 60 °C** (boundary falls in the lower group). *(Threshold value transcription-noisy; standard CEC is 100 A / #1 AWG.)*
- **Maximum supply services to a building = ONE** (Rule 6-102). Exceptions: **fire pump**, industrial/large complex structures.
- **Maximum consumer services off one supply = FOUR** (Rule 6-104); more units → deviation/special permission with calculations.
- **Overhead service-drop height = 3.5 m (pedestrian) to 9 m (max);** ~5 m over driveways/vehicle areas. *(Mid-values approx.)*
- **Service mast minimum = 63 trade size (2.5") rigid steel conduit; upper portion must be metal.** "63" is the **thread/trade size, not millimetres.**
- **Minimum free conductor at service head = 750 mm** (with drip loop); **drip loop hangs up to 600 mm.**
- **Roof-to-service-attachment clearance ≥ 915 mm** (reducible to **600 mm** for the drip-loop portion, per Appendix B).
- **Service head conductor emergence: 150–300 mm above and ≤ 600 mm horizontally** from the overhead-conductor attachment point.
- **Mast/run > 1.5 m → requires a guy wire;** unguyed mast projection above support **≤ 1 m.**
- **Wood backing block/ladder for the mast: not less than 38 mm in any dimension.**
- **Service-conductor clearance from windows/doors/porches/balconies = 1 m** (left, right, below; above not specified).
- **Consumer service conductor minimum size = #10 copper / #8 aluminum** (Rule 6-208). It is a **minimum** — not "too small for a house."
- **Leave ≥ 450 mm of conductor at the meter;** meter-feed conduit **minimum trade size 21.**
- **Service equipment location (Rule 6-206):** **as close as possible to point of entry**, and **NOT** in a **coal bin, clothes closet, bathroom, room normally > ~30 °C, hazardous location, or where headroom < ~2 m** (or similar). Service-equipment headroom **2 m** (vs the larger Section-2 figure for exposed-live motor controls).
- **Unprotected consumer service conductor inside a building must be embedded in ≥ 50 mm of concrete/masonry.** General tip: wherever the code says **"embedded," the figure is 50 mm** (≈12 occurrences) — a high-probability correct answer.
- **Re-exam logistics [teacher/logistics]:** if you fail, you may **re-book immediately the same day** (no mandatory waiting period in practice — earliest open slot is usually ~1.5 months out anyway); you **cannot** book two attempts at once — the result must post first. Advice: book the exam **2 weeks to 1 month after the course ends**, never longer ("the material is volatile/perishable"), using that gap purely for practice questions.

### Mnemonics / phrasings

- **"With two temperatures in one conduit, the lower rating rules everything."** (Section 4 Q3 takeaway.)
- **"The cable's higher insulation rating lets you USE it, but the TERMINAL temperature sets the size."** (Rule 4-006.)
- **"At exactly 100 A with an unmarked terminal, take 60 °C, not 75."** (Boundary-value trap.)
- **"D36A is aluminum, D36B is copper — backwards from the usual, because aerial cable is aluminum."**
- **"Neutral means resting; if it carries the full return it's *working* — so it's an *identified* conductor, not a neutral."**
- **"A, B, C is just a convention — in Canada the phase order swaps 60 times a second."**
- **"Red, black, blue — A, B, C."** (Three-phase colour order.)
- **"Drip loop — the U so the water drips off instead of running into the house."**
- **"63 is the thread size, not 63 millimetres."** (Service mast / conduit trade sizing.)
- **"Service equipment: as close as possible to where the power enters — that bare conductor has no breaker in front of it."**
- **"Wherever the book says *embedded*, it's 50 mm — and that's the right answer about 90% of the time."**
- **"One service in, up to four consumers out."** (Rules 6-102 / 6-104.)
- **"Maximum services to a building? One. Everything else is an exception."**
- **"Book the re-take within two weeks to a month — never longer — because the material is perishable."**
