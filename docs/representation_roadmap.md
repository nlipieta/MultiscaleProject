# Representation-Widening Roadmap for the Attractor Model

Goal: make the signed-GRN a **robust, generalizable** multistable attractor model of cell fate — the
substrate for "given components + initial condition, what states emerge, and which perturbations move the
system between them." Current status (2026-07-15): the machinery is built and works on clean synthetic
data, but on real data it (1) has **no asymptotic multistability** (collapses to one attractor), (2) shows
**fragile operating-horizon separation** (0.96 on full data, collapses on a batch subset), and (3) validates
perturbations only in **direction**, not rank/magnitude. This roadmap attacks the root cause.

## Diagnosis: why it collapses on real data

The dynamics runs over ~300 KG nodes, of which only **~116 carry expression signal** in the multiome. The
~15–20 genes that actually distinguish erythroid from myeloid are **diluted** across mostly-uninformative
nodes, so fate prototypes sit only **0.04 RMS/node apart**. That weak fate signal never engages the
positive-feedback toggle (GATA1↔PU1), so the vector field keeps a single sink and the contractive dynamics
erases the 0.04 gap. On synthetic data (fates 0.2–0.9 apart) the toggle engages and two basins form — so
**the failure is representational, not a bug in the circuit.** (Hypothesis; Phase 0 tests it.)

## Guiding tension: interpretability vs capacity

The literature-KG structure is what makes this model interpretable and thesis-relevant. Widening to
thousands of data-driven features would separate fates easily but discard that structure. The roadmap
navigates this by **concentrating and expanding signal while keeping a named, signed network** — not by
dropping to an opaque latent model (that is only the Phase-1b fallback).

## Phase 0 — De-risk (cheap, days). Is it representation or formulation?

- **P0.1 Focused subnetwork.** Run the dynamics on the informative hematopoietic subgraph (fate
  regulators + effectors + the toggle), not all 300 nodes. If fates separate and multistability holds
  *robustly across seeds*, representation dilution is confirmed as the lever. (Implemented:
  `--subnet hemato` in train_multistable / shift_analysis.)
- **P0.2 Feature informativeness weighting.** Down-weight low-variance / non-discriminative nodes in the
  prototype metric and flow target.
- **P0.3 Robustness.** Multi-seed + leave-batches-out stability of the separation (the fragility must be
  fixed, not just average accuracy).
- **GATE:** subnetwork engages robust multistability → representation is the lever, go to Phase 1.
  Still collapses/fragile → the *dynamical formulation* is the blocker → Phase 1b (energy-based rethink).

## Phase 1 — Widen the representation (weeks)

- **P1.1 ATAC-informed + data-inferred edges.** Run SCENIC/GRNBoost2 on the multiome RNA for TF→target
  edges and add ATAC peak→gene links (Cicero/ArchR-style) for enhancer regulation → a denser,
  data-grounded, still-named network. This directly leverages the matched multiome (the epigenetic thesis).
- **P1.2 Signs + provenance.** Literature signs where known; sign inferred edges by co-expression /
  accessibility direction; tag every edge's source (literature / TRRUST / inferred) as we already do.
- **P1.3 Re-evaluate** multistability + robustness on the widened network.

## Phase 2 — Generalize (weeks)

- More fates (lymphoid, the full hematopoietic tree), not just Ery/Myeloid.
- Multiple datasets / cross-donor held-out evaluation — the real test of "generalizable".
- Report BALANCED accuracy + per-fate sensitivity + multi-seed variance always (never one-sided).

## Phase 3 — Validate transitions (the product claim)

- Only if robust multistability holds: validate predicted transitions against perturbation atlases with a
  readout above the noise floor (trajectory/velocity or within-dataset perturbation, not cross-dataset
  basin membership, which we showed sits at the noise floor).

## Honest risk register

- Even widened, the dynamical formulation may not yield robust multistability — Phase 0 gates this before
  the expensive Phase 1.
- Data-inferred edges dilute interpretability purity — mitigate with provenance tags + a literature core.
- The interpretability↔capacity tension may force a choice; Phase 1b (latent dynamics) is the capacity
  fallback but sacrifices node-level interpretability.
- Validation resolution caps quantitative claims regardless of the model — magnitude/rank may stay
  direction-only until better perturbation readouts exist.

## What is already banked (do not re-litigate)

§3.3 structure-improves-ranking (AUPRC +0.255, p=0.0001, marker-controlled); interpretable regulator
recovery; perturbation *direction* validated vs Replogle. These stand independent of this roadmap.
