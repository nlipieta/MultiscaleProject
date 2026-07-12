# Matched-multiome (RNA+ATAC) data plan — the epigenetic-landscape study

**Goal.** The thesis is fundamentally epigenetic: the chromatin landscape is the substrate for
resistance (attractor stability), plasticity (openness), and memory. On RNA-only data the epigenetic
layer is unobserved, so the mechanisms can't be fairly tested. This plan accumulates **same-cell
RNA+ATAC** (10x Multiome / SHARE-seq) so the mechanisms run on the substrate they model, and — the
non-redundant target — tests **what ATAC predicts that RNA can't** (chromatin potential, latent/poised
programs, plasticity-gated transitions), not redundant co-prediction of state.

## Hard constraints
- **Same experiment only** (RNA+ATAC same cells or same study) — no cross-study grafting.
- **Experimental labels only** (author clustering / experimental variable) — NO imputed/projected labels.
- **No imputing ATAC** for RNA-only programs (imputed ATAC = another model's output; violates the rule).
  => the epigenetic-landscape study is DEEP on the multiome subset, not broad across all 22 programs.

## Datasets (verified; SHARE-seq scout 2026-07-10 + Multiome)
| Program(s) | Accession | Assay | Status | Labels |
|---|---|---|---|---|
| Erythropoiesis (+ commitment trajectory) | **GSE207308** | SHARE-seqV2 BMMC | **INGESTED** (bmmc_shareseq.csv; 12,347 cells; early/late-Ery + HSC/MPP; stage 0/1/2) | tier-2 author clusters |
| Erythropoiesis (POC 2nd source) | GSE194122 | 10x Multiome BMMC | ingested (bmmc_multiome.csv; binary Ery/Quiescent) | tier-2 |
| Pluripotency + directed differentiation | **GSE217215** | SHARE-seq hESC | NEXT (highest breadth; human; tier-1 TF+timepoint) | tier-1 |
| Exhaustion (CD8 T) | GSE244184 (SHARE-seq arm only) | SHARE-seq | queued | tier-1 drug/timepoint |
| NeuronalDiff | GSE306582 | SHARE-seq visual cortex | queued | tier-1 age/rearing |

**Not recoverable / gaps:** Megakaryopoiesis (no committed-MK label in any dataset found — only MEP).
~16 of 22 programs have NO public matched multiome (ADM, Hypertrophy, Fibrosis, EMT, Senescence,
Adipo, Myo, Osteo, EndMT, Regen, Trophoblast, Tfh/Treg/GC, InnateMemory, MacrophageActivation) →
stay RNA-only; the epigenetic-landscape claims are demonstrated on the multiome subset only.

## Loaders
- SHARE-seq (separate barcode-paired RNA h5 + ATAC MatrixMarket + metadata): `ingest_gse207308_shareseq.py`
  is the reusable template (RNA custom-h5 CSC + peak→gene-activity + celltype/stage from metadata).
- 10x Multiome (single h5ad, feature_types GEX/ATAC): `ingest_gse194122_multiome.py` (backed-mode).
- Each new dataset = one loader in that mold; verify mapped genes/cells > 0 before trusting.

## Tests (what ATAC adds beyond RNA)
- **Chromatin potential** (`scripts/chromatin_potential.py`): does the ATAC landscape forecast erythroid
  commitment EARLIER/better than RNA (ATAC-only vs RNA-only vs both, stratified by commitment stage;
  descriptive accessibility-vs-expression lead-lag). The predictive-priority signal.
- **Plasticity = f(epigenetic landscape)** (`plasticity_source=atac`): per-program accessibility +
  global openness + cross-program diversity → plasticity; test on the trajectory axis.
- **Latent/poised programs** (future): accessible-but-not-expressed programs = the model's room for
  undiscovered biology.

## Strategic framing (two results, not one)
- **A — broad RNA structure result** (22 programs): regulatory graph improves program RANKING,
  significant, controlled. Solid now; the real-data headline. Does NOT depend on ATAC.
- **B — deep epigenetic-landscape study** (multiome subset): chromatin potential / plasticity /
  latent programs. Ambitious, thesis-true, data-limited to the ~4-6 multiome programs above.
Keep them distinct; B is the frontier, A is banked.
