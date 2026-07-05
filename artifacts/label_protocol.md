# Label Provenance & Sensitivity Protocol

**(Reviewer point #3 — airtight, transparent, sensitivity-tested labels)**

Every training label is the *biological program* assigned to a cell. This document
states, per dataset, exactly where that label comes from and how strong the claim
is, so no label is a hidden threshold on the model's own input genes.

## Label-type taxonomy

We grade each dataset's labels by how directly the individual cell's state was
measured:

- **A — Deposited per-cell annotation.** The original authors assigned a cell-type
  / substate call per cell (whole-transcriptome clustering). Strongest; the label
  is independent of our KG input genes.
- **B — Per-cell annotation × donor/disease field.** Per-cell type is deposited,
  but the program call also uses the donor's disease status (e.g. HCM vs normal).
- **C — Condition / timepoint proxy.** The label is the experimental arm (treated
  vs baseline, or timepoint), applied to all qualifying cells. Weaker: it assumes
  the arm induced the program, not verified cell-by-cell.
- **D — Derived by us.** We re-clustered / marker-scored because no per-cell label
  was deposited. Flagged explicitly; most exposed to circularity.

## Per-dataset protocol

| Dataset | Program / Quiescent | Organism · assay | Label source | Rule | Type |
|---|---|---|---|---|---|
| GSE172380 | ADM / Quiescent | mouse · scRNA | authors' `celltypeLabel` | Ductal-like/MucinDuctal→ADM; Acinar→Quiescent | **A** |
| GSE188819 | ADM / Quiescent | mouse · scRNA | authors' `annotated_clusters` | Ductal→ADM; Acinar→Quiescent | **A** |
| GSE113049 | Regeneration / Quiescent | mouse · scRNA | authors' `cell_type` substate | Injured AEC2→Regeneration; Naive AEC1/2→Quiescent | **A** |
| GSE143437 | Regeneration / Quiescent | mouse · scRNA | `cell_annotation` (lineage) × `injury` | MuSC/progenitor lineage: Day0→Quiescent; post-injury→Regeneration | **A×C** |
| GSE254185 | Fibrosis / Quiescent | human · snRNA | authors' `celltype_l2` | Myofibroblast→Fibrosis; Fibroblast→Quiescent | **A** |
| GSE135893 | Fibrosis / Quiescent | human · scRNA (10x) | authors' cell-type | Myofibroblast→Fibrosis; Fibroblast→Quiescent | **A** |
| DCM atlas | Fibrosis / Quiescent | human · snRNA | `disease` field | cardiomyopathy fibroblast→Fibrosis; normal→Quiescent | **B** |
| EB atlas | Pluripotency / Quiescent | human · scRNA | authors' `cell_type` | pluripotent stem cell→Pluripotency; differentiated lineages→Quiescent | **A** |
| GSE168776 | MyogenicDiff / Quiescent | mouse · scRNA (Split-seq) | `sample` (nuclei type) | MT_nuclei→MyogenicDiff; MB→Quiescent | **A** |
| GSE149451 | MyogenicDiff / Quiescent | human · scRNA | **re-clustered + marker-scored** | myogenic cluster→MyogenicDiff | **D** |
| E-MTAB-9702 | InnateMemory / Quiescent | human · SORT-seq | priming arm | β-glucan-primed→InnateMemory; RPMI→Quiescent (T2 LPS) | **C** |
| GSE184241 | InnateMemory / Quiescent | human · SORT-seq | priming arm | BCG-primed→InnateMemory; control→Quiescent | **C** |
| HCM | Hypertrophy / Quiescent | human · snRNA | `disease` field | hypertrophic cardiomyopathy→Hypertrophy; normal→Quiescent | **B** |
| GSE120064 | Hypertrophy / Quiescent | mouse · scRNA | `condition` (among CM) | TAC→Hypertrophy; 0w baseline→Quiescent | **C** |
| GSE147405 | EMT / Quiescent | human · scRNA | `Time` | 0d→Quiescent; later timepoints→EMT | **C** |
| GSE115301 | Senescence / Quiescent | human · scRNA | `Condition2` | RIS→Senescence; Growing→Quiescent | **C** |
| Neuronal organoid | NeuronalDiff / Quiescent | human · scRNA | authors' `cell_type` | cortical/Cajal-Retzius neuron→NeuronalDiff; progenitor/NRP→Quiescent | **A** |
| Craniofacial | Osteogenesis / Quiescent | human · snRNA | authors' `cell_type` | osteoblast→Osteogenesis; mesenchymal cell→Quiescent | **A** |
| GSE21608 | MyogenicDiff/Pluripotency | mouse · microarray | sample title | Mullen 2011 arm labels (bulk anchors, n≈6) | **C** |

**Summary:** 10 of 18 sources are type-A (deposited per-cell annotation), 2 type-B
(per-cell × disease), 5 type-C (condition proxy), 1 type-D (derived). The headline
programs with ≥2 independent sources include at least one type-A source each, except
Hypertrophy (B human + C mouse) and InnateMemory (C + C) — noted as the two programs
whose labels are condition/disease-anchored rather than per-cell-annotated.

## Cue handling (the leakage fix)

Cues (LPS, MechanicalStretch, TGFβ, …) are applied **uniformly per dataset**, not
gated by the outcome. An earlier version set the cue only on treated/diseased cells,
which for type-B/C datasets made the cue a perfect proxy for the label (e.g.
`MechanicalStretch=1 ⇔ Hypertrophy`). That leak inflated cross-species Hypertrophy to
0.98; after the fix the cue carries no label information and the result rests on real
expression routing. **Cue is uniform going forward.**

## Sensitivity tests already applied

1. **Marker-shortcut control** (`no_markers`, `lineage_only`): program-proximal
   readout genes (Sox9, mTORC1, Autophagy) are zeroed so a label cannot be recovered
   from its own downstream marker. All headline numbers are reported `no_markers`.
2. **Grouped split** (`--group-split`): whole datasets are held out per fold, so a
   model cannot exploit a dataset-specific labeling convention or batch signature —
   it must transfer the program definition across sources. This is the honest test
   for the type-C condition-proxy labels in particular.
3. **Majority/linear/non-linear baselines** (reviewer #2): majority, logistic
   regression, random forest and gradient boosting are run on the identical masked
   features and folds, so any KG-GNN gain is over strong label-agnostic learners.

## Known label limitations (stated, not hidden)

- Type-C labels (Hypertrophy-mouse, EMT, Senescence, InnateMemory, muscle-Regen
  injury axis) assume the experimental arm induced the program uniformly; individual
  non-responder cells in a treated arm are mislabeled as program-positive.
- GSE149451 (type-D) is our own clustering; most exposed to circularity — treated as
  a secondary MyogenicDiff source, not a primary claim.
- EMT and Senescence currently have a single source each; second sources
  (MCF10A TGFβ dose GSE213753; WI-38 RS/IR/ETO GSE226225) are queued to convert
  their type-C axis into a cross-source claim.
