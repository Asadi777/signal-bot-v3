# Voltage Drop — CEC 2024 method (teacher video)

From the instructor's video "Voltage Drop calculation, 2024 version" (transcript
of *WhatsApp Video 2026-06-16 at 9.26.46 PM*). The teacher calls voltage drop
**the single biggest change from the 2021 → 2024 code** and a guaranteed exam topic.

## Why it changed
- **2021:** 2 formulas + 4 tables (complex).
- **2024:** **ONE formula + TWO tables** (Table D3 for **K**, and the **F**-factor table). Much simpler — "solvable in under a minute."

## The limits (Rule 8-102)
- **Branch circuit: max 3%** (from the last overcurrent device to the load).
- **Feeder: max 3%.**
- **Service-to-load total: max 5%.**
- Trap: 3% + 3% ≠ 5% → if the branch uses the full 3%, the feeder may only use **2%** so the total stays ≤ 5%.

## Why control it
If VD isn't controlled it stresses transformers/generators and equipment — e.g. a fridge motor with low voltage draws **extra current → windings overheat → motor can fail**.

## The formula (Appendix D, Table D3)
```
VD = (K × I × L × F) / 1000
```
Rearranged for the two question types:
- Find max length:  **L ≤ (1000 × VD) / (K × F × I)**
- Find min size:    **K ≤ (1000 × VD) / (I × L × F)**  → then read Table D3 for a size with K ≤ that.

Where:
- **VD** = volts dropped allowed = (system voltage × allowed %). e.g. 150 V source → 135 V load = 15 V drop.
- **K** = from **Table D3** (ohm/km). Depends on conductor size, material, temperature, and power-factor/installation.
- **I** = load current (given).
- **L** = one-way cable length in **metres**.
- **F** = system voltage-drop factor (from the F table).
- **1000** = fixed constant, converts km → m (K is in ohm/**km**, L is in **m**).

### Reading Table D3 for K — defaults & corrections
- **Two material columns:** copper vs aluminum.
- **Power factor / installation:**
  - PF = 1 (100%) **or** DC → cable-vs-raceway makes **no difference**.
  - PF = 0.9 or 0.8 → it DOES matter: use the **cable** column or the **raceway** column as stated.
- **Temperature:** table base is **75 °C**. If the conductor is **60 °C → multiply K × 0.95**; if **90 °C → multiply K × 1.35** (per the note at the bottom of Table D3, part 3). Default 75 °C if not stated.
- **Defaults when the question doesn't specify:** PF = 1, copper, 75 °C.

### The F factor (its own table)
- **Default F = 2** in almost all cases (DC, single-phase, and most 3-phase).
- **F = 1.73 (√3)** ONLY for **3-phase AC** that is either **3-wire (line-to-line, no grounded conductor)** or **4-wire with a grounded conductor**.
- If you have no info → F = 2.

## Worked example (from the video)
> *What is the maximum distance for conductor size #3, 480 V, load 50 A, max 3% voltage drop?*

1. VD = 480 × 3% = **14.4 V**.
2. K: no PF/material/temp given → defaults PF 1, copper, 75 °C, size #3 → Table D3 **K = 0.7913 ohm/km** *(approx — verify in Table D3)*.
3. F: nothing special stated → **F = 2**.
4. L ≤ (1000 × 14.4) / (0.7913 × 2 × 50) = 14400 / 79.13 ≈ **181 m**.

→ Maximum distance ≈ **181 m**.

## How this helps our question bank
This resolves/clarifies the flagged Section 8 voltage-drop items — e.g. **S08-023** (#4 AWG via the 2% limit), **S08-025 / S08-027** (extra-low-voltage max distance), **S08-026** (min voltage at a 600 V HVAC unit). Use this exact 2024 method (single formula + Table D3 + F table) when writing their solution steps.

*Transcription note:* the Persian audio garbled a few terms (رسوی = raceway, وی‌دی = VD, سی‌ست = 347). The method and the worked answer (181 m) are sound; confirm the exact K value against the printed Table D3.
