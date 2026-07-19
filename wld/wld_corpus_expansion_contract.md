# WLD training-corpus expansion contract

This layer expands the data available to the WLD foundation model before any
new biological assessment. It leaves the completed Phase B checkpoint and its
report untouched.

## What is added

- Human GRCh38 SHARE-seq from seven bone-marrow donors (`GSE216464`).
- Human GRCh38 SHARE-seq of antigen-specific T-cell exhaustion from four
  donors, two time points, and DMSO/iberdomide conditions (`GSE244184`).
- Mouse mm10 same-cell SHARE-seq from brain, late-anagen skin, and lung
  (`GSE140203`).

The human and mouse studies are not placed in one feature atlas. Every atlas is
keyed by species, genome build, and a content hash of its source bundle
manifests. Adding a study therefore creates a new immutable atlas snapshot
instead of changing the feature space beneath an old checkpoint.

## What varies rather than being globally frozen

Raw chromatin accessibility, RNA abundance, and available protein abundance
remain observation-level measurements. Donor, age, sex, tissue, time,
treatment, condition, sample, and batch are retained in locked source metadata
as potential context. A later training stage may use legitimate context only
through fold-local encodings and context-conditioned parameter heads, so basal
activity, degradation, thresholds, accessibility gates, and supported circuit
edge magnitudes may vary by cell, tissue, and subject.

No state label, cell type, cluster, pseudotime, barcode, embedding, study ID, or
donor ID is appended to the encoder input. Labels may be used only for audits
after representation learning.

## Pairing rule

RNA and ATAC observations become exact cell pairs only when the depositor
provided a shared identifier or an explicit ATAC-to-RNA barcode translation.
Expression similarity, nearest neighbors, optimal transport, labels, and
embeddings are never used to manufacture cell pairs. If deposited identifiers
do not support pairing, the cohort remains an unpaired population and cannot
enter exact-pair contrastive training.

## Staged sources

`GSE217215` hESC differentiation is valuable but remains staged: its deposited
ATAC H5AD and per-sample CSV TAR require a schema probe that proves the raw
RNA-to-ATAC barcode relation. `GSE207308` remains staged because its peaks are
hg19 and require an audited liftover before any GRCh38 merge. Staging is not a
negative biological judgment; it prevents silent coordinate or pairing errors.

## Non-claims

Corpus expansion does not fit ODE kinetics, evaluate held-out biology, establish
attractor basins, or open the sealed `GSE183273` and `GSE214546` studies. Those
steps occur only after diverse pretraining and parameter-conditioning stages
pass their own validation contracts.
