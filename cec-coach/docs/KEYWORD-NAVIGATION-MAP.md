# 🔑 Keyword → Table/Rule Navigation Map (CEC 2024)

The open-book skill that passes the exam: **see a keyword → jump straight to the
right Rule/Table.** Built from the teacher's classes + the question bank. Table
numbers are CEC 2024 — confirm the exact number in your own book, but the
keyword→concept mapping is solid.

> Golden first move: if a **defined term** confuses you → **Section 0** first.
> If a keyword could be in two places → check the **Index** for the exact word.

---

## ⚡ Section 4 — Conductors / Ampacity (high-yield)
| When you see… | Jump to | Note |
|---|---|---|
| "ampacity" / "max current a conductor can carry" | **Table 1–4** | pick the table by the next two keywords ↓ |
| "in **free air**" + copper / aluminum | **Table 1 (Cu) / Table 3 (Al)** | |
| "in a **raceway / conduit / cable**" + Cu / Al | **Table 2 (Cu) / Table 4 (Al)** | start in the **90°C** column |
| "**X conductors**" (more than 3) | **Table 5C** | grouping derate (4–6=0.80, 7–24=0.70, 25–42=0.60) |
| "**ambient** … °C" (above 30°C) | **Table 5A** | temperature correction |
| "**terminated at** / terminal **marked** X°C" | **Rule 4-006** | use the X°C column (the cable's "90" is bait) |
| "**continuous** load" | **Rule 8-104** | conductor & breaker at **125%** |
| "**neutral-supported** / NS cable / overhead service" | **Table 36B (Cu) / 36A (Al)** | |
| "**flexible cord** / **equipment wire** / portable" | **Table 12** (cords: 11A/11B) | |
| "**underground**, direct-buried, **spaced**" | **Rule 4-004 1)d)**, Diagrams **D8/D10/D11** | |

## ⚡ Section 8 — Loads & Demand
| When you see… | Jump to | Note |
|---|---|---|
| "**single dwelling** … calculate demand/service" | **Rule 8-200** | basic 5000W/first 90m² + 1000W/extra 90m² |
| "**apartment** / multiple dwelling" | **Rule 8-202** | |
| "basic load **W/m²** by occupancy" (store, theatre, armoury…) | **Table 14** | |
| "**show window**" | **650 W/m** | |
| "electric **range**" | **Rule 8-200** (service=6kW) / **8-300** (branch=8kW) | +40% over 12kW |
| "**voltage drop**" | **Rule 8-102 + Table D3** | 3% branch / 3% feeder / **5% total** |
| "how many **receptacles** on a circuit" | **max 12 on a 15A** circuit | |
| "**parking** spaces / EV charger load" | **Rule 8-400** | |

## ⚡ Section 10 — Grounding & Bonding
| When you see… | Jump to | Note |
|---|---|---|
| "**grounding electrode conductor** / size of grounding conductor" | **Table 43** | keyed to **service-conductor ampacity** (NOT the breaker) |
| "**bonding** conductor / bonding **jumper**" | **Table 16** | sized by **ampacity OR overcurrent device** |
| "**field-assembled** grounding electrode" | **Rule 10-102/10-104** | bare Cu, bottom 50mm of footing, 600mm deep |
| "**equipotential** bonding" (pool, raised floor, pipe) | **Rule 10-406** | **#6 Cu / #4 Al** |

## ⚡ Section 12 — Wiring Methods (biggest question source)
| When you see… | Jump to | Note |
|---|---|---|
| "**box fill** / how many conductors in a box / box volume" | **Rule 12-3036 + Table 23** | bond/ground = 1 total, no separate volume |
| "**conduit fill** / minimum conduit size" | **Rule 12-910 + Tables 6/8/9** | fill 1→53%, 2→31%, 3+→40% |
| "**bend radius**" | **Table 7** | |
| "number of **bends**" | **Rule 12-936** | max 4 × 90° = **360°** |
| "**support** / secured / spacing" | **Rule 12-510/560** (300mm then ≤1.5m); **Table 21** (vertical) | |
| "minimum **cover** / buried depth" | **Table 53** | reduce 150mm with mechanical protection |
| "NMD90 / NMSC **distance from stud**" | **32 mm** (Rule 12-516) | else protector plate (Fig 12-30) |
| "**cable type** selection / dry-damp-wet" | **Table 19 + Table D1** | NMD90=300V, AC90≤2000V, TECK90≤5000V |
| "**parallel** conductors" | **Rule 12-108** | only #1/0 and larger |

## ⚡ Section 14 — Protection
| When you see… | Jump to | Note |
|---|---|---|
| "**overcurrent** / breaker or fuse size for a conductor" | **Rule 14-104 + Table 13** | ≤ conductor ampacity / next standard size |
| "is **ground-fault protection required**?" | **Rule 14-102** | only if **≥1000A AND >150V to ground** |

## ⚡ Section 26 — Equipment
| When you see… | Jump to | Note |
|---|---|---|
| "**receptacle spacing** in a dwelling" | **Rule 26-712** | no point on wall >1.8m from a receptacle |
| "**GFCI** required?" | **Rule 26-700-ish** | sinks, outdoors ≤2.5m grade, kitchen counter |
| "**transformer** overcurrent / primary fuse" | **Rule 26-250/26-254** | |
| "**capacitor** conductor size" | **Rule 26-210** | ≥ **135%** of rated current |
| "panelboard **handle height**" | **1.7 m** max | |

## ⚡ Section 28 — Motors (hardest block)
| When you see… | Jump to | Note |
|---|---|---|
| "motor **full-load current / FLC**" (by hp & voltage) | **Table 44** (3-ph) / **Table 45** (1-ph) | NOT the nameplate |
| "motor **branch conductor** size" | **Rule 28-106** | **125% of FLC** (Table 44/45) |
| "**overload** size" | **Rule 28-306** | from **nameplate FLA** × 1.25 (SF≥1.15) or 1.15 |
| "max **fuse/breaker** for a motor branch" | **Table 29** | time-delay 175/225%, non-time-delay 300%, inst-trip 1300% |
| "**disconnect** distance" | within **9 m** / in sight | |

## ⚡ Section 2 / 0 — General & Definitions
| When you see… | Jump to | Note |
|---|---|---|
| "**working space** / clearance in front of" | **Rule 2-308 + Table 56** | Table 56 uses **line-to-ground** voltage |
| "**voltage to ground** in a dwelling" | **Rule 2-110** | **150 V** (exception >250kVA + resident electrician) |
| "**enclosure type** (3R, 4X…)" | **Table 65** | |
| "**deviation** / special permission" | **Rule 2-030** | before work, that installation only |
| any unfamiliar **defined term** | **Section 0** | start here |

---

### How to drill this
For every practice question, before opening the book say out loud: **"keyword = ___ → table/rule = ___."** When that mapping is automatic, you'll land on the answer in 10–20 seconds.
