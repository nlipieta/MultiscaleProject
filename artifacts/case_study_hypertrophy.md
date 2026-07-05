# Biological Case Study: Cardiomyocyte Hypertrophy

**(Reviewer point #5 — one deep biological case study + a predicted perturbation direction)**

## Why this program

Hypertrophy is the model's cleanest, most mechanistically-encoded pathway and the
one with a genuine cross-species pair (mouse pressure-overload GSE120064 + human
HCM), so it is the fairest place to ask: *does the model's internal logic recover
the textbook mechanism, and does it make a falsifiable perturbation prediction?*

## The encoded mechanism (KG cascade)

The literature KG encodes the mechanotransduction-to-chromatin axis (Backs & Olson;
Nakamura & Sadoshima reviews):

```
MechanicalStretch → Piezo1 → CaMKII ┐
                    Piezo1 → PKD    ┘→ (nuclear EXPORT of) HDAC4, HDAC5
                                        HDAC4/5 ⊣ MEF2   (de-repression)
                                        MEF2 → Hypertrophy
```

The load-bearing logic is a **de-repression switch**: HDAC4/5 tonically repress the
pro-hypertrophic TF MEF2. Stretch activates CaMKII/PKD, which phosphorylate HDAC4/5
and drive their nuclear export, lifting the brake on MEF2 and committing the cell to
the hypertrophic program. This is a class-II-HDAC signal-responsive checkpoint.

## What the model recovers (interpretability)

Permutation-importance over the trained model attributes the Hypertrophy readout to
the CaMKII → HDAC4/5 → MEF2 sub-graph (the interpretability pass recovers the correct
regulators for this program), i.e. the model is not routing hypertrophy through
spurious features but through the encoded de-repression axis. This is the prerequisite
for trusting a perturbation prediction from it.

## Predicted perturbation direction (falsifiable)

Because commitment runs through **HDAC4/5 nuclear export**, the model predicts a
directional, sign-specific set of interventions:

| Intervention | Node effect | Predicted program shift |
|---|---|---|
| CaMKII inhibition (e.g. KN-93) under stretch | ↓ CaMKII → HDAC4/5 stay nuclear | **Hypertrophy suppressed** (MEF2 stays repressed) |
| HDAC4/5 nuclear-retention / class-II HDAC stabilization | ↑ nuclear HDAC4/5 | **Hypertrophy suppressed** |
| HDAC4/5 knockdown | ↓ repressor | **Hypertrophy enhanced / cue-independent** (brake removed) |
| Constitutively active CaMKII (no stretch) | ↑ CaMKII without cue | **Hypertrophy induced without the mechanical cue** |

The sharp, testable claim: **HDAC4/5 is the directional lever** — reducing the
repressor should *promote* the program and be able to substitute for the mechanical
cue, while blocking the CaMKII→export step should *abolish* stretch-induced
hypertrophy. This matches the thesis's plasticity-gate logic: the cue only commits
the cell if it can lift the intrinsic (HDAC) brake.

## In-silico test (planned, on the trained model)

Operationalize each intervention as a node-input edit on held-out Hypertrophy cells
and measure the change in predicted Hypertrophy probability:

- **Knock down HDAC4/5**: zero the HDAC4/HDAC5 inputs → predict ↑ P(Hypertrophy),
  including in cells with the cue absent (cue-independence).
- **Block CaMKII**: zero CaMKII input under stretch → predict ↓ P(Hypertrophy)
  toward Quiescent.
- **Direction check**: the *sign* of ΔP(Hypertrophy) must match the table (HDAC↓⇒up,
  CaMKII↓⇒down). A model that merely memorized markers would not show this coherent,
  mechanism-consistent directionality.

## In-vitro validation path (wet-lab, if pursued)

Neonatal rat ventricular myocytes (or hiPSC-CMs) under phenylephrine/cyclic-stretch:
(1) KN-93 (CaMKII inhibitor) should blunt cell-size / ANP-BNP induction; (2) HDAC4/5
knockdown should sensitize cells to hypertrophy at sub-threshold stretch. These are
standard, low-cost assays with well-characterized readouts (cell area, NPPA/NPPB).

## Honest scope

This case study is a *mechanistic-consistency + directional-prediction* demonstration,
not a claim of quantitative effect-size prediction. The KG edges are literature-curated
(not fit to data), so the model's job is to show its learned representation respects the
encoded sign logic — which is exactly what makes the HDAC4/5 direction falsifiable.
