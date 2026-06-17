# Session 1 — Part 2 (جلسه ۱-۲) — Study Notes

**Course:** BC Construction Electrician exam prep — Canadian Electrical Code (CEC) 2024
**Coverage:** Section 0 definitions (overcurrent vs overload, jacket, live/exposed, GFCI/AFCI, raceway/receptacle types), receptacles (duplex / single / split), 120/240 V split‑phase systems & neutral current, 3‑phase line‑to‑line vs line‑to‑ground and √3, transformers, and the start of Section 2 general rules (deviation / special permission, 150 V to ground in dwellings, voltage classes ELV/LV/HV).

> These notes are written to capture **everything exam‑relevant** from the recording in full, not as a summary. Persian audio transcription mangled English technical terms; they have been interpreted to the correct CEC terminology. Where a spoken number was noisy/ambiguous it is flagged **(approx.)**.

---

## High-yield: what the teacher flags as important / tested / traps / common mistakes

- **Overcurrent vs Overload — the #1 confusion the teacher sees.** Students from Iran (and most students) can explain *overcurrent* but, when asked to define *overload*, just repeat the overcurrent definition ("when extra current flows"). **These are NOT the same.** Know both cold:
  - **Overcurrent protection (fuse/breaker) protects the CONDUCTORS / wires / cables.**
  - **Overload protection protects the MOTOR (equipment), NOT the wires.**
  - If asked "what is the main job of an overload device?" the answer is **protect the motor**, not the cable.
- **The CEC's *definition* of overcurrent is, in the teacher's opinion, misleading / poorly worded** — he even emailed about it. The real-world definition of overcurrent is universal worldwide (same in Iran, Germany, Canada, anywhere). Don't get tripped up by the book wording; understand the concept: overcurrent protection disconnects when excess current (especially a short‑circuit / fault current) flows, to protect the conductor from burning.
- **Section 0 is DEFINITIONS only** — the teacher stresses there are no "rules" yet; just learn the precise definitions. Section 0 entries are listed **alphabetically (A–Z)**. Rules with numbers begin in Section 2.
- **Voltage‑class boundary questions are a favourite exam trap.** Exams love to test the *edge* values:
  - **Is 30 V Low Voltage or Extra‑Low Voltage?** → It is **Extra‑Low Voltage** (the "not exceeding 30 V" wording **includes** 30 V itself). A common wrong answer is "Low Voltage."
  - **Is 1000 V Low Voltage or High Voltage?** → **1000 V is included in Low Voltage** ("not exceeding 1000 V"). Above 1000 V is High Voltage.
  - Trap mechanism: students argue the boundary value; read the inclusive wording carefully ("not exceeding X" includes X).
- **150 V to ground limit in dwellings — explicitly called an exam question, "very common AND very important."**
  - In a **dwelling unit**, the voltage **to ground shall not exceed 150 V**.
  - **Trap:** students see "120/240 V" marked on equipment and wrongly answer "120" or "125." The relevant value is the **line‑to‑ground** voltage. 240 V is line‑to‑line, so line‑to‑ground = 240/√3 ≈ **she said ~150** in the split‑phase context (see note below) — the limit is **150 V to ground**. Watch the L‑L vs L‑G distinction.
  - There is an **exception ("except")**: higher voltage IS permitted in an **apartment / similar building** whose demand exceeds **250 kVA**, provided a **resident electrician** is on site (someone who responds to electrical faults — not necessarily lives there). For higher voltage in that case, it **shall not exceed 600 V / 347 V** (i.e., line‑to‑line 600, line‑to‑ground 347).
- **√3 relationship is a load-bearing exam concept.** From the L‑L and L‑G voltage pair you can determine single‑phase vs three‑phase:
  - **Ratio 1 : 2 → single‑phase** (e.g., 120/240).
  - **Ratio 1 : √3 (≈1.732) → three‑phase** (e.g., 120/208, 277/480, 347/600).
  - **Exam can give you a circuit (e.g., "120/240" or "120/208") and ask single‑ or three‑phase.** You must answer from the ratio. The teacher notes a "fair" exam usually tells you, "but if I wrote the question, I wouldn't tell you" — so be ready to derive it.
  - **Why it matters for current calc:** single‑phase uses `I = P / V`; three‑phase divides by **√3**. If you can't tell the phase, you can't pick the formula. 120/240 ⇒ single‑phase (no √3); 120/208 ⇒ three‑phase (use √3).
- **Common student error: 120/208 ≈ 120/240?** No. **208 ≠ 240.** 208 comes from 120 × √3 (three‑phase). 240 comes from 120 × 2 (single‑phase split). Don't "round" 208 up to 240 — they're fundamentally different systems.
- **Neutral sizing trap (split‑phase 120/240).** Students think: "two hot legs, so the neutral must carry double / be sized double." **WRONG.** The neutral carries only the **difference** (imbalance) of the two legs. Worst case the neutral equals one leg's full current (e.g., 100 A on one leg, 0 A on the other ⇒ 100 A on neutral). So in **receptacle** wiring the neutral is the **same size** as the hots. (Utility service drops are an exception — see content section.)
- **Split receptacle "break the tab" warning (kitchen counters).** When you split a duplex you connect **two different hot legs** that have a **voltage difference** (240 V between them). You **must break the connecting tab** on the hot side, otherwise you create a short‑circuit between the two hots — "very dangerous." A real failure mode: someone splits the receptacle but **forgets to break the tab**, so the breaker trips immediately.
- **Two‑pole breaker requirement for splits / 240 V.** The two breakers feeding a split (or a 240 V load) must be **adjacent**, on **opposite phases** (left/right alternate on the panel), and **mechanically tied (two‑pole)** with a handle tie / pin so that if one trips, the other trips with it. (Electrical trip on one pole alone won't mechanically drag the other unless tied.)
- **"Two‑phase" is a wrong term to avoid.** 120/240 split‑phase is **single‑phase** (one phase, center‑tapped → two hots 180° apart + neutral). Do **not** call it "two‑phase." The teacher recounts being misunderstood on a job by saying "two‑phase"; correct usage is single‑phase 120/240.
- **Deviation / Special Permission (Section 2) — practical exam/job point.**
  - **Deviation = departure from the Code.** You may deviate **only** where the Code *itself* allows it, and you must obtain **Special Permission** from the authority (inspector / AHJ).
  - **Special permission shall be obtained BEFORE proceeding with the work.**
  - **Special permission applies ONLY to the particular installation for which it was given** — not transferable to another project or another year. (Teacher's anecdote: a guy tried to reuse last year's deviation on this year's job — not valid.)
  - You cannot self‑authorize a deviation just because the Code is silent or because you'd prefer to — otherwise everyone would deviate constantly.
- **637 V (600/347) usage in dwellings — limited and conditional.** You may bring 600/347 V into an apartment building **only** to feed **fixed equipment** that is **central / building service** (central heating, central hot water, central air conditioning located in utility/mechanical rooms) — not into the individual dwelling units for occupant use. Occupant equipment runs at low voltage (fridge, lamps, TV ~120 V). The 150 V‑to‑ground limit still governs what reaches the occupant.
- **Aluminum conductor restriction note.** Some installations (e.g., hospitals) **prohibit aluminum conductors** by Code even though aluminum is cheaper; designers write conductor requirements on the drawings ("conductor selection per the rules"). Don't substitute aluminum to save money where the Code/spec forbids it.
- **Section 2 = General Rules.** Each CEC section has a specific scope (motors, pools, hospitals, film sets, etc.); **Section 2 is the general rules** that apply broadly and is not tightly ordered. The administrative front matter (very first part) is **skipped** by the teacher — not on the exam, not job‑relevant.
- **Rule numbering convention (Section 2 onward):** The **first digit is the section number** (Section 4 rules start with 4, Section 20 rules start with 20, etc.), then a sequential rule number; **rule numbers are usually even** (gaps left so new rules can be inserted later — don't be surprised by an occasional odd number). Rules also have **sub‑rules** (e.g., "Rule 2‑XYZ, sub‑rule (1)" → teacher will just say "sub 1").
- **Frequency difference Iran vs Canada (practical caution, may appear as background):** Iran 50 Hz, Canada **60 Hz**. Motors spin faster in Canada (≈10 more revolutions/second), so motor‑type appliances brought from Iran may burn out. Pure resistive/element loads (e.g., a samovar) are fine at 240 V because they only heat.

---

## Content taught (in order, full detail)

### 1. Live / Exposed / Energized (continuing Section 0 definitions)

- A part can be simultaneously **bare, live, and exposed**.
- **Live part** = an **energized conductive component** — the actual point that is "bare"/uncovered and carrying voltage. (Spoken: "live part یعنی توش برهنست" = the bare/exposed energized point.)
- **Exposed** ties into **GFCI** requirements — in areas needing GFCI (Class‑A GFCI), the rule is mandatory.

### 2. GFCI and the GFCI/AFCI breaker (definitions / hardware orientation)

- **GFCI requirements:** in designated areas, **Class‑A GFCI** is mandatory. It can be implemented as a **receptacle (reset on the plug)** or as a **breaker**.
- **GFCI breaker has an extra return ("pigtail") wire.**
  - **Ordinary (normal) breakers** do not have this extra wire — current simply flows in one side and back out the other.
  - **A GFCI breaker has a white return/neutral pigtail** because it needs to **measure the current on the single sensed conductor** to detect a ground‑fault imbalance.
  - **AFCI** also exists and works similarly (a return wire to sense). Teacher will cover AFCI in detail later.
- **Fuse vs Breaker (the two forms of overcurrent device):**
  - **Fuse — burns/melts (sacrificial).** Inside is a **thin wire** sized to a rated current; on a short circuit it melts and must be **replaced**. Example: **"this fuse is 15 A"** means it can carry 15 A without melting (i.e., pass up to its rating).
  - **Breaker — reusable / resettable.** When it trips (e.g., kitchen counter overload from running coffee maker + air fryer together), you just **flip the switch back on**.
  - (Teacher anecdote, illustrative: people in Iran bypassing a fuse with a nail so it wouldn't blow — dangerous, do not do; the fuse exists to protect.)

### 3. Overcurrent (overcurrent protection)

- **Purpose: protect the CABLES/wires.**
- **Why every load needs it — short‑circuit reasoning:** If the two conductors feeding, say, a lamp short together, a **very large fault current** flows. That fault current is high and **rapidly heats and burns the conductor**, igniting it (fire risk). Overcurrent protection disconnects when this excess current flows.
- **Two forms:** **fuse** or **breaker** (as above). Industrial environments have additional **special overcurrent devices** that are not the topic of this construction course (newer technology, not common in general installs) — set aside for now.
- The definition of overcurrent is the **same worldwide**; understand the function rather than memorizing the book's (in his view) confusing wording.

### 4. Overload (overload protection) — worked motor example

- **Purpose: protect the MOTOR**, not the wires.

**Motor behaviour basics (the elevator analogy):**
- A motor draws current **in proportion to the load** placed on it.
- Example: an elevator rated to lift **10 people** draws **30 A (approx.)** at full load.
- If only **2 people** ride, it draws far less — e.g., **~5–6 A (approx.)** instead of 30 A.
- **Motor calculations are done at FULL LOAD** ("current at full load"). So we size for the 10‑person / **30 A** case.
- A motor *can* draw more than rated if overloaded (e.g., if you could cram many more people in, it might pull **40 A or more** — "the motor doesn't speak the language," it just keeps drawing more current as load increases).

**Starting (inrush) current:**
- At the **instant of starting**, a motor draws a large **inrush current** that then **drops down to its running value** (e.g., back to ~30 A).
- **Inrush is not precisely calculable in general** — it depends on motor type and size. For a 30 A motor it is **not** simply "50 or 60 A"; for one motor it might be **100 A**, for another **150 A (approx.)**.
- Assume the inrush event is brief — picture it lasting **~half a second / ~½ s (approx.)** just to build intuition.

**Why a fuse alone is not enough (sizing problem):**
- You **cannot** put a **30 A fuse** on a 30 A motor — the **starting inrush would blow it** every start.
- So you might use a **larger fuse (e.g., 100 A)** to ride through inrush. (Teacher: **Section 28** covers motor protection; you may legally put, e.g., a **100 A fuse on a 30 A motor** for starting/various conditions — "the rule of thumb is roughly that.")
- **But now the fuse won't protect against overload:** if running load rises to **40 A** (overload), a **100 A fuse never notices** — it only reacts above 100 A. So an **overload device** is required to protect the motor at currents between rated and the fuse rating.

**The overload device — how it works (thermal / bimetal):**
- The overload uses a **thermal (heat‑based) mechanism**: a **bimetallic** element — **two dissimilar metals with different thermal expansion coefficients** bonded together. As current heats it, the two metals expand unequally, the strip **bends**, and at a set point it **trips/disconnects**.
- **Why bimetal/thermal (and why it ignores inrush):** Heat builds with time. A brief inrush (e.g., 150 A for an instant) doesn't generate enough sustained heat to bend the strip, so the overload **does not** trip on starting inrush. Only **sustained** over‑current heats it enough to trip.
- **Adjustable:** overloads have a **setting screw** and are **time‑adjustable / inverse‑time** ("inverse time" — higher current trips faster, lower current trips slower). Standard time curves are marked on them.
  - Example of the time/current trade: a motor might tolerate **40 A for up to ~1 minute (approx.)** depending on type. You can set the overload to **trip after ~1 minute** at that current, but not trip immediately.
- **Inverse‑time meaning:** the **larger** the overcurrent, the **shorter** the time to trip (and vice‑versa).

**Moral / summary (teacher's "ethical conclusion"):**
- **Overload protects the motor**; it does **not** protect the wires.
- A heavy sustained over‑current could still **overheat and ignite the wires** if only an overload were present — so you install **both**: an **overload (for the motor)** AND an **overcurrent device / breaker (for the cables and wires)**.
- **Bottom line:** **Overload → protects motors.** **Overcurrent → protects cables and wires.**

### 5. Jacket (definition)

- **Jacket** can be **metallic** or **non‑metallic**.
- A **non‑metallic covering** (mentioned earlier) provides **mechanical and environmental protection**.
- **Mechanical protection** — protects the cable from physical damage.
- **Environmental protection** — e.g., if a cable runs in a **wet location or where acidic/corrosive materials** are present, it would corrode; the jacket provides **environmental protection** against that.

### 6. Raceway (definition)

- **Raceway** = any **channel designed/defined to hold (enclose) conductors**.
- Includes **enclosed conduit**:
  - **Conduit** can be **rigid** or **flexible**, **metallic** or **non‑metallic**.
  - Includes **EMT** (electrical metallic tubing).
  - Includes **underfloor raceways**.
- (Details of each type to be taught later — here just learn the **word "raceway"** and what it encompasses.)

### 7. Receptacle (definition) and receptacle types

- **Receptacle** = what in Iran is called "priz" (the outlet you plug into). The Code term is **receptacle**.
- **Types:**
  - **Duplex receptacle** — the common two‑gang household outlet in rooms/homes. Most rooms use duplex receptacles.
  - **Single receptacle** — one outlet for a single dedicated load (e.g., a **garage door opener**, a single appliance, a dryer). Where only one device is served, a single receptacle is used (you *may* use a duplex there too, but single is typical for dedicated loads). Larger dedicated outlets (e.g., dryer) are **single**.
  - **Split receptacle** — a duplex that has been **separated (split)** so the **top and bottom halves are on different circuits/hot legs**. (Standard in Canadian homes, especially kitchen counters.)

**Receptacle wiring — terminals (standard duplex):**
- **Brass/gold (yellow) screw** → connects to the **smallest slot** → this is the **hot (live) conductor** (smallest slot = most dangerous conductor). On the right side.
- **Silver screw** → connects to the **neutral (white)** conductor (lower hazard) → the **other (larger) slot**.
- **Green (grounding/bonding) screw** → connects to the **bonding** system; the **largest slot/round pin**. When wiring a receptacle you bring a **green wire** from the panel to the green screw to bond it.
- **Internal tab/plate ("platin"):** the two brass screws are joined internally by a **connecting tab (plate)**, and likewise the two silver screws. So for a **normal (non‑split)** receptacle you only need to land **one black (hot)** wire on one brass screw and **one white (neutral)** wire on one silver screw — the internal tab feeds the second screw automatically.

**Normal receptacle:**
- One **black** (hot) to a brass screw, one **white** (neutral) to a silver screw, **green** to ground; internal tabs carry power to the second outlet. **This is the standard for general room/area receptacles.**
- **Standard general receptacles are 15 A** ("we usually run these at 15 A").

**Kitchen counter receptacles:**
- Kitchen counters are where **breakers trip most often** (e.g., running an **air fryer** and a **coffee maker** together).
- Two ways to make the kitchen counter stronger:
  1. Use a **20 A** receptacle/circuit with **12‑gauge (12 AWG) wire** ("stronger"), or
  2. Use a **split receptacle** (separate the two halves onto different breakers) so the load is divided and the breaker doesn't trip.

**Split receptacle wiring (detail + danger):**
- To split, you **break the internal connecting tab** between the two **hot (brass)** screws, then land **two different hot legs** — one black hot to the top brass screw, another hot to the bottom brass screw — fed from **two different breakers**.
- Because the **two hots are on different phases**, there is a **voltage difference (240 V)** between them. **If you do NOT break the tab**, the two hots are connected → **short circuit** → "very dangerous."
- The split is **designed so that** if you break the center tab, the top and bottom operate independently.
- **Two breakers feeding a split must be adjacent, on opposite phases, two‑pole (handle‑tied/pinned)** so they trip together. On the panel, breaker phases **alternate** (one position is phase A, next is phase B, etc.); two **adjacent** breakers are therefore on **two different phases**.
- **The tie pin:** when one pole trips electrically, the **mechanical tie** drags the other pole off too. Without the tie, one pole can trip while the other stays live.

### 8. 120/240 V single‑phase (split‑phase) residential system

- A Canadian home is fed by **two hots + one neutral** (three wires).
- **Each hot to neutral = 120 V (approx. "120/125").** The top hot is +120 V relative to neutral; the bottom hot is −120 V (opposite direction / 180° apart) relative to neutral.
- **Hot‑to‑hot (the two hots together) = 240 V**, because the two are **180° out of phase**, so their voltages **add** in magnitude.
- **Voltmeter readings:**
  - Hot ↔ neutral → **120 V**.
  - Hot ↔ hot (across the two hots) → **240 V**.
- **Why both voltages exist in a home:** so you can serve both small loads and large loads.
  - **120 V loads (one hot + neutral):** fridge, TV, lighting — only need one hot and the neutral.
  - **240 V loads (both hots):** EV charger (30 A and up), dryer (often ≥30 A), electric range/stove, baseboard heaters, hot water — high‑draw "element"/heating loads use 240 V.
- **Wiring count for loads:**
  - A **240 V‑only** load (e.g., EV charger): bring **2 wires** (two hots) — plus a bonding wire which we don't count toward the supply.
  - A **range/dryer** that needs both 120 V and 240 V: bring **3 wires** (two hots + neutral) — plus bond.
- **Within one appliance both voltages can be used.** Example: a **dryer** — the **motor runs on 120 V**, while the **heating element runs on 240 V**. An **electric range** — clock/light/receptacle parts run on **120 V**, the **heating burners/elements** run on **240 V**.
- **Bringing appliances from Iran:** A **resistive samovar** rated 240 V works fine here (Iran has 220/240‑class). **Motor appliances are NOT recommended** because (a) voltage differs and (b) **frequency differs: Iran 50 Hz vs Canada 60 Hz** — motors spin faster here (≈10 more rev/s) and may burn out. A samovar (pure element) is safe.
- **"Two‑phase" is wrong terminology.** 120/240 is **single‑phase** (one phase center‑tapped). You may have "120 and 240" but that does **not** make it "two‑phase." (Anecdote about being misunderstood on the job.)

### 9. Neutral current in 120/240 split‑phase (worked example)

- **Setup:** 100 A available on the top leg; the house draws 100 A on the top hot. **How much returns on the neutral?**
- **Answer when both legs are balanced (equal loads): neutral current ≈ 0 A.**
  - Because the two hots are **180° apart**, equal currents cancel in the neutral. When both are 100 A, the neutral carries ~0.
- **Why have a neutral at all then?** Because loads are rarely perfectly balanced. The two halves of the house usually draw **different** currents, and the **difference flows on the neutral**.
- **Imbalance example:** top leg 100 A, bottom leg 80 A → neutral carries the **difference = 20 A**.
- **Worst case for neutral current:** one leg fully loaded, the other off → e.g., **100 A on one hot, 0 A on the other ⇒ 100 A on the neutral.**
  - So the **maximum** the neutral ever carries equals **one full leg's current (100 A)** — never more.
- **Consequence for sizing (receptacles):** Because the worst case is one full leg, the **neutral is the SAME size as the hots** in receptacle/branch wiring. Students wrongly assume "two hots ⇒ double the neutral" — **not true.** ("15 A goes out and 15 A comes back.")
- **Utility service exception:** The **utility** (power company) **may** run a **smaller neutral** on the service drop. Why? The utility knows that across a whole dwelling the **imbalance maxes out around 30%** in practice (worst realistic case ~70 A return if one leg is 100 A; typically far less — e.g., 40 A vs 35 A leaves only ~5 A on neutral). Because not all big loads (dryer, range) run at once and lights may be off, the **utility's neutral is noticeably smaller** to save copper at national scale. **BUT inside the dwelling, for receptacles, always size the neutral the same as the hots** ("we size them equal").
- **Reconciling the two:** The worst theoretical case (100 A on neutral) is why branch‑circuit neutrals match the hots; the utility's statistical 30% imbalance is why the **service** neutral can be reduced.

### 10. Three‑phase systems, line‑to‑line vs line‑to‑ground, and √3

**Iran reference:** single‑phase 220 V; three‑phase 380 V (220/380 system).

**Three‑phase wye (star) layout:**
- Three lines come in: **L1, L2, L3** (in Canada the phases are named **A, B, C**; in Iran R/S/T).
- **Line‑to‑line voltage** (between any two lines, e.g., L1–L2, L2–L3) = **380 V** in the Iran example ("three‑phase means this").
- The **center point** is the **neutral (N)** — the point of (near) zero potential that can be **connected to earth/ground**.
- **Line‑to‑ground (line‑to‑neutral) voltage** = the smaller value, **220 V** in the Iran example.

**The two rules to KEEP IN MIND (teacher emphasizes "keep this in your mind"):**
1. **The LARGER voltage is always Line‑to‑Line.**
2. **The SMALLER voltage is always Line‑to‑Ground (line‑to‑neutral).**
3. **Their relationship is √3.** Larger = smaller × √3.

**√3 worked examples:**
- **220 × √3 = 220 × 1.732 ≈ 380** (Iran three‑phase). √3 ≈ **1.732** ("one‑point‑seven‑three‑two").
- **Canada 120 × √3 ≈ 208** → the **120/208 V** three‑phase system. (She notes 208 is "again the smaller one.")
- **277 × √3 ≈ 480**, **347 × √3 ≈ 600** (other Canadian three‑phase systems implied by the 600/347 discussion later).

**Why √3 and not 2 (vector addition):**
- When you add two phase voltages that are **120° apart** (not 180°), the **vector sum is LESS than double**. If they were **in‑phase (0°)** you'd multiply by 2 (→ 120+120 = 240). Because three‑phase legs are **120° apart**, you multiply by **√3 (≈1.732)**, not by 2.
- Contrast with split‑phase: 120/240 hots are **180° apart**, so they add to **240 (×2)** — that's why 120/240 is **single‑phase**, ratio **1:2**.

**Phase identification rule (exam‑relevant):**
- **Ratio 1:2 → single‑phase** (120/240).
- **Ratio 1:√3 → three‑phase** (120/208).
- This is how you decide whether to divide by √3 in a current calculation. (See High‑yield.)

**Note on 120/240 in Canada vs Iran:**
- The **120/240** split system **does not exist in Iran** (Iran is either 220 single‑phase or 380 three‑phase). Canada's everyday residential supply is **120/240** to ~99% of homes (three‑phase service to homes is rare — only special cases like a home with multiple EV chargers, a pool, etc., might get three‑phase).

### 11. Transformers (how 120/240 is produced; intro for Section 8 later)

- **A three‑phase transformer is built from THREE single‑phase transformers.** ("When we say a three‑phase transformer, it means it's made of three single‑phase transformers" — stated as a rule.)
- Each transformer has a **primary** and a **secondary** winding.
- **Producing 120/240 (single‑phase, center‑tapped secondary):**
  - The secondary is wound for **240 V**. If you place a **center tap** in the middle of that secondary winding, you split it into **two 120 V halves**.
  - The center point = the **zero/neutral** point. From neutral up = 120 V; the full winding = 240 V.
  - **The number of turns is directly proportional to the voltage** ("turns ratio"). Tapping partway up the winding gives a proportional voltage — e.g., tap for 30 V, 60 V, 90 V, up to 120 V (and the reverse).
  - (Teacher recalls hand‑winding transformers in trade school to get any desired voltage by counting turns.)
- **Grounding the system / where the neutral is made:**
  - The **zero point (neutral)** of the secondary is connected to **ground**. The utility creates the **ground (earth) connection at the transformer on your street** — they drive a **ground electrode** ~1 m (approx.) into the earth, attach a wire, and that becomes the **neutral (zero) conductor**.
  - From the street transformer, **three wires** run to your home: **L1, one more hot (L2), and the neutral.** (i.e., two hots + neutral, as in §8.)
  - Use **two of these for 120 V** (one hot + neutral) or **both hots for 240 V**, as described.
- **Section 8 (later):** voltage **drops** along the run — the first house near the transformer may see **240/250**, farther houses see less, because longer distance from the source lowers voltage. Equipment is therefore marked with the **range** (line‑to‑line and the single‑phase value, and the three‑phase value if applicable).

---

### 12. Voltage classes (still Section 0 definitions) — ELV / LV / HV

The teacher gives the **AC** thresholds only:

- **Extra‑Low Voltage (ELV):** any voltage **not exceeding 30 V** — i.e., **30 V is included** (≤ 30 V). 
- **Low Voltage (LV):** **greater than 30 V and not exceeding 1000 V** — i.e., **1000 V is included** (> 30 V, ≤ 1000 V).
- **High Voltage (HV):** **above 1000 V.**

**Boundary‑value exam traps (re‑emphasized):**
- **30 V → Extra‑Low Voltage** (not "Low Voltage" — students get this wrong). Because the cutoff "not exceeding 30 V" **includes** 30.
- **1000 V → Low Voltage** (included), not High Voltage. Above 1000 V is High Voltage.
- Exams favour these edge values. Have the inclusive definitions firm so you don't argue yourself out of the right answer.

**Field note on misuse of "low voltage":** People casually call **24 V** work "low voltage." Technically, by the Code definition, low voltage spans up to 1000 V — the term is widely misused on the job. Know the **Code** definition for the exam.

---

### 13. Transition into Section 2 — General Rules

- **Section 0 (definitions) is finished; Section 2 begins the actual numbered rules.**
- **The CEC is a book of rules/law, not a textbook.** It opens with an **Administrative** part which the teacher **skips** (not examined, not job‑relevant, poorly written).
- **Each section has a defined scope:** motors, pools, hospitals, film/broadcast locations, etc. **Section 2 = General Rules** — broad items, **not tightly ordered**.
- **Rule numbering:** first digit = section number (Section 4 → 4‑xxx; Section 20 → 20‑xxx); then a sequential number; **numbers are usually even** (gaps reserved for future inserts; an odd number occasionally appears). Rules have **sub‑rules** (teacher will say "sub 1," etc.). Rules are written "2‑XYZ"; he'll stop saying the leading "2."

### 14. Section 2 — Deviation & Special Permission

- **Deviation** literally means **departure / divergence** — a departure from the Code.
- You may **only** deviate where the Code permits it. Some rules **allow no deviation**; others **do** allow it, and the Code **tells you where** and grants the permission in that very place ("do X, X, X — unless you go get a deviation").
- Sometimes the Code doesn't explicitly say, but the rule is **worded such that deviation is possible** — you must recognize this.
- **You CANNOT self‑authorize a deviation** — neither when the Code is silent nor just because you prefer to. Otherwise everyone would deviate at will.
- **Exact Code language (teacher reads it):**
  - "**In any case where it is necessary to deviate** ... **special permission shall be obtained before proceeding with the work.**"
  - "**This special permission shall apply only to the particular installation for which it is given.**"
- **Practical points:**
  - Get **special permission BEFORE starting** the work (especially for **renovation** work).
  - Permission is **non‑transferable** — only for the **specific installation** it was granted for; **not** reusable on another project or a different year (anecdote: a guy tried to reuse last year's deviation; invalid — an inspector arriving would not accept it).
  - If you don't know, **say "I don't know"** to the inspector — don't bluff.

### 15. Section 2 — Equipment ratings & line‑to‑line / line‑to‑ground marking (Rule 2‑104 area)

- **Rule 2‑104 (approx.):** electrical equipment used in Canada **shall be marked / shall have a rating** — and the marking includes **line‑to‑line AND line‑to‑ground voltage**.
- Equipment commonly shows its voltage on the nameplate. Some equipment is **120 V only** (so just one value); larger items show a pair.
- **Examples of nameplate pairs:**
  - Dryer / range (technical/spec sheet): **120/240** → line‑to‑line **240**, line‑to‑ground **120**.
  - Teacher also references **"125/250"** style markings (i.e., the 120/240 class shown as 125/250).
  - **600/347** style for the higher‑voltage three‑phase service (line‑to‑line 600, line‑to‑ground 347).
- This reinforces §10: **larger = L‑L, smaller = L‑G, ratio √3 for three‑phase.**

### 16. Section 2 — 150 V to ground in dwellings (Rule on dwelling‑unit voltage)

- **Rule (dwelling units):** In a **dwelling unit / dwelling unit similar building**, the voltage **to ground shall NOT exceed 150 V.**
- The current Code raised the wording so that the limit is expressed around the **125/250 → 150 V‑to‑ground** ceiling. **You may NOT have voltage to ground higher than 150 V** in a dwelling.
- **Why 150 and not "120/240"?** Because the limit is on the **line‑to‑ground** voltage. For 240 V line‑to‑line, the relevant to‑ground value is below 150, so it's allowed. (Teacher walks students through that 240 V L‑L corresponds to the permitted to‑ground figure; the controlling number is **150 V to ground**.)
- **This is explicitly a "very common AND very important" exam question.** Trap: seeing "120/240" or "125" on equipment and answering 120/125 instead of recognizing the **150 V to ground** rule.

**Exception — apartments / similar buildings with high demand:**
- **Higher voltage IS permitted** (the rule has an **"except"**) when:
  - The building is an **apartment / similar (commercial) building**, **AND**
  - Its **demand exceeds 250 kVA**, **AND**
  - A **resident electrician** is provided — meaning someone who **responds to electrical problems** when called (not necessarily someone who lives there; the point is fault response, "won't say 'I can't come, I'm at a wedding'").
- In that exception, the higher voltage **shall not exceed 600 V / 347 V** (line‑to‑line 600, line‑to‑ground 347).
- **kVA vs kW note:** the demand is stated in **kVA** (volt‑amperes), not watts. They are related but **not identical**; the rule uses **250 kVA**.

### 17. Section 2 — 600/347 V (637 V) use in dwellings: limited to central fixed equipment

- Canada uses **many transformers** (more than Iran did) — e.g., a single pizza shop with ovens, A/C, and freezers will get its **own transformer** stepping down to its needs.
- **You do NOT run 600/347 V (spoken "637") directly to occupant equipment.** Bringing 600/347 into an apartment means putting a **transformer** that converts it down to **120/240** for the units.
- **600/347 V may be used directly only for fixed equipment that is CENTRAL / building‑service**, located in **utility/mechanical rooms**, e.g.:
  - **Central heating system**,
  - **Central hot‑water heater**,
  - **Central air‑conditioning** (the building's central A/C, not in‑unit).
- Rationale: it's wasteful to add an extra transformer for the central equipment when high‑voltage‑rated central gear can use 600/347 directly. But **occupant** equipment (fridge ~120 V, lamps, TV) is **low voltage** — occupants never get 600/347 V.
- **The controlling limit remains 150 V to ground** for what reaches the occupant/dwelling. ("The red line is the 150 V.")
- **Code phrasing (paraphrased):** "...permitted to be used in a dwelling unit **for fixed equipment**..." for central heating / hot water / air conditioning supplied at the higher voltage.

### 18. Next up (teacher's roadmap)

- After this, the class would move to **Rule 2‑30 / 2‑20 area** ("2.30," "2.20") — flagged but the session ends here with off‑topic scheduling chatter (excluded).
- Practice with the **provided past exam questions** was recommended so students learn the exact question style; deeper treatment of these rules continues in the **Section 2** lessons, with motor protection in **Section 28** and transformer/voltage‑drop detail in **Section 8**.

---

## Mnemonics / exact phrasings

- **"Overcurrent protects CABLES/WIRES; Overload protects the MOTOR."** (The teacher's core distinction — memorize this exact split.)
- **"Fuse burns (replace it); Breaker is reusable (reset it)."**
- **"The bigger voltage is always Line‑to‑Line; the smaller voltage is always Line‑to‑Ground; their relationship is √3."** (Stated verbatim as the two things to "keep in mind.")
- **Ratio test:** **"1 : 2 = single‑phase; 1 : √3 = three‑phase."** (e.g., 120/240 single‑phase; 120/208 three‑phase.)
- **√3 ≈ 1.732** ("one‑point‑seven‑three‑two"). 120 × √3 ≈ 208; 277 × √3 ≈ 480; 347 × √3 ≈ 600; 220 × √3 ≈ 380.
- **"In a dwelling unit, voltage to ground shall not exceed 150 V."** (Exception: apartment/similar building > 250 kVA with a resident electrician → up to 600/347 V.)
- **"Special permission shall be obtained BEFORE proceeding with the work, and applies ONLY to the particular installation for which it is given."** (Deviation rule, near‑verbatim Code language.)
- **Boundary inclusivity:** **"30 V is Extra‑Low Voltage; 1000 V is Low Voltage."** ("Not exceeding X" **includes** X.)
- **Split receptacle:** **"Break the tab"** (between the two hots) — forgetting it = dead short across two phases (240 V).
- **Neutral myth‑buster:** **"Two hots do NOT mean a double‑sized neutral."** In a balanced split‑phase the neutral carries ~0; worst case it carries one leg's full current; in branch wiring size the neutral equal to the hots.
- **"Don't call 120/240 two‑phase — it's single‑phase."**
- **Rule numbering:** **"First digit = the section; rule numbers are usually even (gaps left for future rules)."**
- **Frequency:** **"Iran 50 Hz, Canada 60 Hz"** — motor appliances from Iran may burn out; resistive/element loads are fine.
