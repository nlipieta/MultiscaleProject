# WLD v4 multi-study foundation contract

## Purpose

WLD v4 keeps the original scientific framework but changes how it is trained.
It first learns reusable, context-dependent kinetics from many single-cell
studies.  Only after that pretraining stage is frozen may a sealed tissue or
study be used to assess generalization or attractor behavior.

This repository contains the architecture and training contract.  It does not
contain a completed biological foundation-model checkpoint.

## The original framework, made operational

1. **Epigenetic landscape — where chromatin is open.** ATAC measurements form
   the chromatin state. Peak-to-gene contacts and TF motifs localize the gates
   through which TF activity can affect RNA.
2. **Protein and metabolic state — what signals and molecules are present.**
   Measured protein/metabolic features project only through declared biological
   maps. Missing assays are masked; they are not imputed and mislabeled as
   measurements.
3. **Circuit architecture — which interactions are supported.** Evidence fixes
   the sparse set of allowed edges and their reference orientation. It does not
   freeze their cell-specific strength.
4. **Hybrid latent state.** A structured neural encoder estimates named signal,
   TF and chromatin starting states. A neural ODE evolves signal, TF, chromatin
   and RNA variables. There is no neural residual vector field and no direct
   latent-to-future-RNA decoder.
5. **Leakage protection.** Study, donor, barcode, cell-type, cluster,
   pseudotime, integrated embeddings and future/outcome variables cannot enter
   the encoder. Study and donor identifiers are used only to create splits.

## What is shared and what varies

The evidence-supported topology is shared. Within that topology the following
are context-conditioned for every observation:

- supported interaction gains;
- basal production rates;
- signal, TF and RNA decay rates;
- chromatin relaxation timescales;
- initial signal, TF, chromatin and (when measured) RNA state.

Context adapters start at a population value but are trainable. A shrinkage
penalty discourages gratuitous study-specific behavior without freezing it.
The context code is built from measured modalities and declared numeric
biological covariates, not study/donor identity.

Reference interaction signs are hard in v4.0. If a later evidence table
explicitly supports context-dependent sign reversal, that needs a separately
audited bidirectional-edge contract; it must not be inferred by quietly adding
an unrestricted neural edge.

## Pretraining and assessment phases

### Phase A — cross-modal foundation pretraining

Use joint RNA/ATAC data across tissues, species and assay technologies. Add
measured protein or metabolic data when present. Modality dropout teaches the
model to operate when supplementary measurements are unavailable. Missingness
masks are always supplied.

This phase learns initialization and cross-modal mappings. It does not establish
temporal dynamics by itself.

### Phase B — temporal and perturbational dynamics

Use real time courses, longitudinal samples and perturbation experiments. When
the same cells are not followed through time, compare predicted and future
populations with distributional objectives. Never manufacture cell pairs from
pseudotime or optimal transport and describe them as observations.

### Phase C — validation-only development

Hold out whole studies and donors before feature selection or prior compilation.
Select checkpoints and hyperparameters using only these validation studies.

### Phase D — sealed assessment

Open the test studies once. Report supported-circuit, no-TF-circuit and
degree/support-preserving sign-shuffle controls. Attractor claims additionally
require converged fixed points, stable Jacobian eigenvalues, reproducible basin
return and relevant perturbational behavior. Predictive performance alone is
not evidence of an attractor.

## Minimum diversity gate before assessment

Do not assess a general foundation model until training includes:

- multiple tissues and at least three assay technologies;
- multiple donors in each major human tissue cohort;
- both cross-sectional multiome and genuine temporal/perturbation studies;
- at least one protein-augmented cohort;
- explicit held-out studies, not only random held-out cells;
- a frozen feature/prior build made only from training studies.

The candidate registry is in `wld_v4_study_registry.json`. Each entry still
needs a license review, exact donor/group audit, genome-build adapter and checksums
before it becomes a trainable manifest.

## What the smoke test establishes

`run_wld_v4_foundation_smoke.py` checks topology, gradients into contextual
kinetics, missing-modality masks, whole-study/donor splitting, distributional
training and scientific controls on synthetic data. It deliberately leaves the
synthetic test study sealed. Passing it is a software-contract result only.
