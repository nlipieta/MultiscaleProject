# Modeling framework — emergence + perturbation (predicting experimental outcomes)

**The model's purpose (N. Lytell, 2026-07-14).** Not snapshot classification. Two questions, answered
through the multiscale framework, aimed at predicting *experimental outcomes*:

- **Q1 — EMERGENCE.** Given **components** (KG nodes), **interactions** (edges), **inputs** (cue +
  measured expression/accessibility), and an **initial condition** (starting node state) → *what
  attractor state(s) emerge?*
- **Q2 — PERTURBATION.** Which edits to components / interactions / inputs **move the system from one
  emergent state to another?**

## Why this framing is correct (and dissolves the "inert mechanisms" problem)
`ResistanceToggle` already *is* a recurrent graph dynamical system: initialize node states → propagate
the interactions (message passing) → settle into an attractor. The mechanisms that read as **inert
under static classification are the core of this framing**:
- **resistance** = depth of the attractor basin (how stable an emergent state is),
- **plasticity** = whether a perturbation can cross between basins,
- **attractor** = the emergent state itself.
They were inert only because a *snapshot classifier* has no emergence or transition to reward. Judged
on emergence + perturbation, they are the model. The reframe re-scores existing work correctly; it is
not a restart. Seeds already in the repo: `perturb.py` (Q2 in-silico), `temporal.py`/`trajectory_atac.py`
(Q1 emergence), the plasticity/hysteresis simulation.

## Operational definitions
- **Q1 emergence:** init node hidden states from the cell's initial condition (expression/accessibility);
  inject the cue transiently; run T message-passing steps over the interactions; read the emergent
  program attractor(s) — soft/graded, possibly several coexisting (multiple attractors per cell).
- **Q2 perturbation:** edit a node input / interaction / cue → re-run the dynamics → read the new
  emergent state; report ΔP(state) per intervention and which cross a basin boundary (state flip).

## Evaluation — validate against EXPERIMENTAL OUTCOMES (the whole point)
- **Q1 ← (initial → emergent) data = TIME COURSES.** Have: EMT (0d→7d, GSE147405), erythroid
  (HSC→early→late-Ery, GSE207308 SHARE-seq). Test: from the initial condition, does the model's
  emergent attractor match the observed endpoint / graded trajectory? **Buildable now.**
- **Q2 ← (perturbation → outcome) EXPERIMENTS = control vs KO/drug/cue single-cell.** THE GAP. The
  in-silico perturbation is currently **unvalidated** (the HDAC4/5 prediction is honestly flagged as an
  unconfirmed wet-lab prediction). To *predict experimental outcomes* we must validate against real
  perturbation experiments **where the perturbed target is a node we also perturb in-silico**.

## Perturbation-validation data (scout 2026-07-14, GEO/repository-verified)
- **Erythropoiesis — Replogle genome-scale Perturb-seq (K562 CRISPRi)** — Figshare+ 20029387 (processed
  AnnData) / SRA PRJNA831566 / gwps.wi.mit.edu. Genetic KD of **GATA1, LMO2** (named erythroid
  regulators) + TAL1/KLF1 among ~9,867 targets; ~2.5M cells, non-targeting controls, direct guide
  capture. **The clean Q2 validation:** perturb GATA1/TAL1 in-silico ↔ real KD outcome. Pairs with the
  erythroid SHARE-seq multiome (emergence) → **Erythropoiesis = the one program with BOTH Q1 and Q2
  data.** Caveat: Figshare+ may gate large files; confirm GSE mirror before citing.
- **EMT — GSE147405** (already in our data): TGFβ/EGF/TNF ligand induction time course + kinase-inhibitor
  screens, untreated controls, ~104k cells. Cue/inhibitor perturbations (not TF-KO) → Q2 for EMT.
- **Reprogramming — GSE115943** (Schiebinger Waddington-OT): OSKM Dox induction MEF→iPSC, 18-day course,
  controls, ~250k cells → Q2 for Pluripotency/reprogramming.
- **Hypertrophy (HDAC4/5, CaMKII) — CONFIRMED GAP: no public single-cell perturbation dataset exists.**
  The cardiac HDAC/CaMKII literature is all bulk/ChIP/phenotype; TAC snRNA sets are disease-state, not
  the molecular perturbation. So the flagship HDAC4/5-de-repression prediction stays a genuine,
  falsifiable WET-LAB prediction — it cannot be validated in-silico against existing data. State it so.

Net: Q2 is validatable now for **Erythropoiesis (Replogle, genetic), EMT (GSE147405, ligand/inhibitor),
Reprogramming (GSE115943, OSKM)**; **Hypertrophy is not** (data gap, honestly a prediction only).

## Honest guardrail
Until Q2 is validated on held-out perturbation experiments, the model **generates falsifiable
perturbation predictions** — it does **not yet "predict experimental outcomes."** That claim is earned
*per cascade* as validation data lands (Erythropoiesis via Replogle first; Hypertrophy pending real
HDAC/CaMKII data).

## Project shape (supersedes the classification-first framing)
- **Supporting:** the RNA structure/ranking result — a sanity check that the graph carries real signal.
- **Core (this framework):** EMERGENCE (validated on trajectories) + PERTURBATION (validated on real
  perturbation experiments). This is the thesis deliverable, and the axis where the dynamical mechanisms
  are load-bearing rather than inert.
