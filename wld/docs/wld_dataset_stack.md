# WLD biological dataset stack

## Decision

The first real-data build should use a **human skeletal-muscle exercise stack**,
not one assay that is expected to measure every layer well.

1. **Dynamic observations — GSE240061.** Human vastus lateralis nuclei were
   assayed by 10x Multiome before and 3.5 hours after endurance exercise, with
   exercise and time-matched resting participants. RNA and ATAC are measured in
   the same nuclei. Subjects, not cells, are the split unit.
2. **Tissue 3D scaffold — GSE126100.** Promoter-capture Hi-C in human primary
   skeletal-muscle cells supplies enhancer/promoter-to-gene topology. This is a
   more stable tissue-matched scaffold than transferring sparse mouse-embryo
   single-cell contacts into human muscle.
3. **Binding feasibility — JASPAR CORE vertebrate motifs.** Motif hits are
   localized to the retained ATAC peaks.
4. **Validated circuit — CollecTRI.** Signed TF-to-gene relations provide the
   TF-to-gene and TF-to-TF signs. A TF-to-TF edge exists only when the target TF
   gene is supported by the same signed relation.
5. **Cue and protein path — OmniPath.** Exercise or genuinely measured
   metabolic/protein cues enter through prespecified cue-to-signal,
   signal-to-signal, and signal-to-TF edges.

These layers implement the required causal intersection:

```text
measured time-zero cue -> signed signaling/PPI path -> TF
                                                   |
time-zero open peak -> localized TF motif ----------+
          |
          +-> tissue-matched Hi-C contact -> target gene
                                      ^
                                      |
                         signed validated TF-gene edge
```

The scaffold defines what interactions are possible. Measured ATAC and cues
define the starting condition. Future RNA and ATAC are targets and never enter
the initial state.

## Why this is the first build

GSE240061 contains 37,154 nuclei from human skeletal muscle, four exercise
participants sampled before and after the intervention, and two time-matched
resting participants. It offers genuine physical time and biological-group
holdouts. The 3.5-hour endpoint is a transient response, so every transition is
declared `terminal: false`; the dataset can test dynamics but cannot by itself
establish an attractor.

GSE126100 reports promoter-capture Hi-C in the same tissue and includes control,
palmitate, and TNF-alpha conditions. Its processed interaction table is small
enough to use as a reproducible external prior. Genome-build compatibility must
be verified before peak overlap; coordinates must be lifted once, recorded, and
then frozen before the held-out subjects are evaluated.

## Ranked supplementary datasets

| Priority | Dataset | Modalities and role | Use in WLD | Do not use it for |
|---|---|---|---|---|
| 1 | [GSE240061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240061) | Human muscle RNA+ATAC, pre/post exercise and rest | Primary grouped transient benchmark | Declaring the 3.5 h response an attractor |
| 2 | [GSE126100](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126100) | Human muscle promoter-capture Hi-C | Fixed tissue peak-to-gene topology | Pretending bulk contacts are cell-specific measurements |
| 3 | [GSE223917](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223917) (HiRES) | Same-cell Hi-C+RNA in mouse embryos | Calibrate contact reliability or pretrain a contact-to-regulation module | Copying mouse embryo edges into the human muscle graph; treating inferred pseudotime as observed transitions |
| 4 | [GSE133688](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133688) (scNMT-seq) | Same-cell RNA, accessibility, and DNA methylation across mouse gastrulation | Epigenetic pretraining and testing a slow methylation gate | Main human-muscle OOD benchmark |
| 5 | [GSE284047](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE284047) | Multiome differentiation data used by MultiVeloVAE | Secondary velocity/differentiation comparison | Replacing measured time with a learned latent time |
| 6 | [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013) (TEA-seq) | RNA+ATAC+surface protein | Pretraining or testing protein-conditioned state inference | An attractor benchmark without a temporal intervention design |

HiRES is the strongest true single-cell Hi-C option in this list, but tissue and
species match matter more than the word “single-cell” when compiling the human
muscle topology. Its contacts are therefore auxiliary knowledge, not direct
edges in the primary graph.

## Metabolic and protein supplementation

A supplemental measurement is admitted only when its subject/sample and time
are known. Its handling is fixed as follows:

- same subject and initial time: include as a measured cue;
- sample- or subject-level measurement: repeat across that unit's cells and
  retain `measurement_level` provenance;
- absent value: set its cue mask to zero and never impute it;
- unmatched external cohort: use for pathway selection or external validation,
  not as a cell-level input;
- RNA-derived pathway score: report as an outcome or ablation, never feed it
  back as “metabolic state” in the core model.

MoTrPAC exercise metabolomics/proteomics can inform pathway selection and
external validation. It should become a WLD cue only if a release can be matched
to the same biological unit and time as the multiome observation. Exercise
assignment itself is a legitimate experiment-level cue, not a metabolite.

## Leakage-safe split for GSE240061

The exact subject identifiers must be read from the released metadata. The
prespecified structure is:

- training: two exercise subjects and one resting subject;
- validation: one exercise subject;
- sealed test: one exercise subject and one resting subject.

All cells and both time points from a subject stay in one split. The builder
performs the split before RNA-gene ranking or ATAC-peak ranking. Cell labels,
clusters, UMAP, RNA pseudotime, CREMA results inferred on all subjects, and
future measurements are not model inputs. Author-provided circuits inferred
from the full cohort are comparison data only; a prior for the held-out test
must be external or refit on training subjects.

## Canonical evidence files

`build_wld_muscle_exercise_dataset.py` expects matrices in Matrix Market
`features x cells` orientation and five small, reviewable tables:

```text
peak_gene_links.tsv: peak_id, gene, score
motif_hits.tsv:      peak_id, tf, score
tf_gene_edges.tsv:   source, target, sign, score
signaling_edges.tsv: source, target, source_type, target_type, sign, score
tf_peak_effects.tsv: tf, peak_id, sign, score                         # optional
```

`source_type -> target_type` is restricted to `cue -> signal`,
`signal -> signal`, or `signal -> tf`. A supplemental cue without a complete
path to a retained TF is excluded and recorded rather than concatenated as an
unused feature.

Example build after exporting the multiome and canonical priors:

```bash
python build_wld_muscle_exercise_dataset.py \
  --rna-mtx export/rna.mtx.gz \
  --atac-mtx export/atac.mtx.gz \
  --genes export/genes.tsv \
  --peaks export/peaks.tsv \
  --barcodes export/barcodes.tsv \
  --metadata export/metadata.tsv \
  --peak-gene-links priors/peak_gene_links.tsv \
  --motif-hits priors/motif_hits.tsv \
  --tf-gene-edges priors/tf_gene_edges.tsv \
  --signaling-edges priors/signaling_edges.tsv \
  --split-json split.json \
  --output wld_gse240061

python wld_temporal_training.py validate --data wld_gse240061
```

The core build omits time-zero RNA. `--include-initial-rna` is an explicit
ablation only. The build report records every selected feature, excluded cue,
edge count, group split, and claim boundary.

## Required biological comparisons

Before evaluating the sealed subjects, prespecify:

1. true circuit versus no circuit;
2. signed degree-matched circuit control;
3. tissue Hi-C links versus distance-matched or promoter-only links;
4. localized motif gate versus gene-level accessibility;
5. shuffled held-out ATAC within valid subject/time blocks;
6. exercise cue shuffled only within a scientifically valid randomization
   block;
7. ATAC-only core versus ATAC plus each genuinely matched metabolic/protein
   cue.

The primary result is per-subject future-distribution prediction. An attractor
claim remains out of scope until a later dataset supplies relaxed terminal
states, perturb-and-return evidence, and prospective basin-crossing tests.
