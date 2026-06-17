# Precision Conductor Sizing — Voltage-Drop-Limited Method (CEC)

> **Source:** `Precision_Conductor_Sizing.pdf` (Google Drive fileId `1mSRkZjnZW6_TAdER2prlaP6lF9G-qKEk`)
> **Document type:** Text-layer PDF (NotebookLM-generated study/case-study handout). The body text is in **Persian/Farsi** with embedded English technical terms, table names, and all numeric values intact. Read in full — no scanned/image-only pages.
> **Scope note (IMPORTANT):** Despite the broad "conductor sizing" title, this document teaches **only the voltage-drop-limited sizing method** using the CEC voltage-drop formula and the **K-value table (Table D3, copper column)**. It does **not** cover ampacity selection from Table 2/4, ambient correction (Table 5A), grouping correction (Table 5C), termination temperature (Rule 4-006), or continuous-load 125% (Rule 8-104). Those topics are flagged as *not present* in the "Gaps vs. the standard method" section below.

---

## 1. What this document teaches

A **4-step engineering approach** to find the **smallest standard conductor size** whose voltage drop over a given run length stays **below the allowable percentage** (here, 2%). This is the *voltage-drop branch* of conductor sizing — the size is driven by the design voltage-drop ceiling, not by ampacity.

The worked example uses a **solid copper core** conductor (IEC 60228 Class 1, Cu-ETP), XLPE insulation, PVC sheath. (Those construction details are descriptive of the illustration; the sizing math uses the standard CEC copper K-values.)

---

## 2. Given design inputs (worked example)

| Quantity | Symbol | Value |
|---|---|---|
| Load / design current | I | 30 A |
| One-way run length | L | 60 m |
| Source voltage (single-phase) | V | 240 V (single phase, line-to-line) |
| Maximum allowable voltage drop | — | 2 % |

**Goal:** Find the smallest standard cable size that keeps voltage drop under 2% over this 60 m run.

---

## 3. The 4-step method

### Step 1 — Convert the percentage limit to actual volts

The allowable VD percentage must be turned into a real voltage figure before it can be used in the formula.

```
VD = V × (allowable %)
VD = 240 V × 0.02 = 4.8 V
```

**Maximum allowable voltage drop (VD) = 4.8 volts.**

### Step 2 — Determine the system factor `f`

The structure of the distribution system sets the multiplier `f` in the voltage-drop formula.

- For a **single-phase, line-to-line** load (fed by two conductors): **`f = 2`**
- Stated source: **Note 2 of CEC Table D3** (the table's accompanying notes define `f` per system type).

> Standard CEC practice: `f = 2` for single-phase, and `f ≈ 1.732 (√3)` for three-phase. This document only exercises the single-phase case (`f = 2`).

### Step 3 — Rearrange the formula to solve for the maximum allowable K

When the conductor size is the unknown, the voltage-drop formula is rearranged to find the **ceiling value of K** (the per-table voltage-drop coefficient, in volts per amp-metre-style units).

```
        VD × 1000
K  ≤  ───────────────
         I × L × f
```

Plugging in:

```
        4.8 × 1000        4800
K  ≤  ───────────────  =  ──────  =  1.333
        30 × 60 × 2        3600
```

**Key result: K ≤ 1.333**

### Step 4 — Evaluation matrix: pick the conductor from Table D3 (copper column)

Goal: find the **first (smallest) standard size** whose tabulated K value does **not exceed** the allowable ceiling of **1.333**.

| Size (AWG) | Tabulated K (copper) | Status vs. limit 1.333 |
|---|---|---|
| #8 | 2.54 | Rejected — voltage drop far too high (K > 1.333) |
| #6 | 1.59 | Rejected — still above the limit (K > 1.333) |
| **#4** | **1.51** *(see flag below)* | **First compliant / selected size** |
| #3 | 0.792 | Compliant but oversized (more than needed) |

> **⚠ Data-fidelity flag:** In the source table, #4 AWG is labeled the "first compliant" / selected size, yet its printed K value reads **1.51**, which is numerically **greater** than the 1.333 ceiling. This is an internal inconsistency in the source PDF (likely an OCR/transcription error in the NotebookLM export — a real Table D3 copper K for #4 AWG is roughly **1.0**, comfortably under 1.333, while #6 ≈ 1.6 and #8 ≈ 2.5). The document's **logic and final selection (#4 AWG) are correct**; only the printed K=1.51 figure for #4 appears garbled. Treat the #4 K value as unreliable as printed.

---

## 4. Conclusion of the worked example

**Optimal engineering selection: #4 AWG copper conductor.**

Recap of design inputs:
- Design run length: 60 m
- Load current: 30 A
- Voltage-drop ceiling: 4.8 V (2% of 240 V), giving K ≤ 1.333

**#4 AWG** is the first and most cost-effective standard size that reliably keeps voltage drop below the 2% limit on this run and complies with CEC requirements.

**Stated correct answer: (b).** (This is a multiple-choice exam item; option (b) corresponds to #4 AWG.)

---

## 5. The method as a reusable decision flow

```
1. VD_volts = V_source × allowable_percent
2. f = 2  (single-phase) | f = √3 ≈ 1.732 (three-phase)   ← from Table D3 notes
3. K_max = (VD_volts × 1000) / (I × L × f)
4. From Table D3 (correct metal column: Cu or Al),
   scan sizes smallest → largest;
   choose the FIRST size whose tabulated K ≤ K_max.
```

Rule of thumb the document reinforces: a **larger conductor has a smaller K**. You want the smallest conductor (largest acceptable K) whose K still falls at or under your computed K_max — going larger (smaller K, e.g. #3) is compliant but wasteful.

---

## 6. Rules / tables actually referenced

| Reference | How it's used in this document |
|---|---|
| **CEC voltage-drop formula** `K ≤ (VD×1000)/(I·L·f)` | Core sizing equation (Step 3). |
| **Table D3 (copper column)** | Source of tabulated K values per size; selection table (Step 4). |
| **Table D3, Note 2** | Defines the system factor `f` (= 2 for single-phase line-to-line). |
| **2% voltage-drop design limit** | Used as the allowable-VD design ceiling (Step 1). The 3% / 5% CEC objectives are not discussed. |

---

## 7. Gaps vs. the standard CEC conductor-sizing method (NOT covered here)

The full CEC sizing workflow normally requires you to size for **ampacity first**, then **check** voltage drop. This document covers **only the voltage-drop check** and omits the ampacity side entirely. The following items from the standard method are **absent** from the source PDF and should be sourced elsewhere:

- **Rule 4-004 / Table 2 & Table 4 ampacity** — minimum conductor ampacity for the load. *(Not present.)*
- **Table 5A — ambient temperature correction factors.** *(Not present.)*
- **Table 5C — conductor grouping / bundling derating (>3 current-carrying conductors).** *(Not present.)*
- **Rule 4-006 — termination temperature limits (60/75/90 °C).** *(Not present.)*
- **Rule 8-104 — continuous-load 125% / continuous vs. non-continuous loading.** *(Not present.)*

**Where it reinforces the standard:** the voltage-drop formula, the `f` factor convention (2 for 1-φ, √3 for 3-φ), the K-value table approach (Table D3), and the "choose smallest compliant size" decision logic all match standard CEC practice. **Where it differs:** it treats voltage drop as the *sole* sizing driver for the example and skips ampacity/correction-factor selection, so it should be used as a *voltage-drop module*, not a complete sizing procedure.

---

## 8. Reading / quality flags

- **Language:** Source body is Persian/Farsi; technical terms, table references, and all numbers are in English/digits and were fully legible.
- **Not scanned:** A real text layer was returned (no OCR-of-image fallback needed).
- **Garbled value:** The printed **K = 1.51 for #4 AWG** in the Step-4 table contradicts the K ≤ 1.333 ceiling under which #4 is selected (see §3 flag). Logic and final answer are sound; that one cell's number is unreliable.
- **Illustration callouts** (e.g., `Ø25.00 mm`, `825.00 mm`, `B 40.00 mm`, `120 ISOMETRIC VIEW`, `A = 490.87 mm²`) are diagram dimension labels for the conductor cross-section graphic and are not part of the sizing calculation.
