# WLD v4 Phase A real-data contract

## What this phase does

Phase A builds the multi-study raw-count foundation needed before biological
pretraining. It downloads public processed matrices, locks their SHA-256
checksums, preserves modality-specific barcodes, writes sparse cohort bundles,
builds shared feature atlases from training studies only, and projects each
cohort into the appropriate species/build-specific feature space.

It does **not** fit dynamics, open a sealed test study, or report biological
performance.

## Active public cohorts

- GSE158013 / GSM5123951: joint RNA/ATAC plus ADT in human PBMCs.
- GSE126074: RNA/ATAC SNARE-seq from neonatal and adult mouse brain.
- GSE233046: combined 10x-style RNA/ATAC matrices from stimulated and
  unstimulated human immune samples.
- GSE240061: the existing raw RNA/ATAC muscle export, used only as validation
  when its export directory is supplied.

GSE214546 and GSE183273 are named sealed-test candidates. The development
runner refuses to ingest a cohort marked `sealed_test`.

## Scientific boundaries

1. Study and donor partitions are defined before a feature atlas is built.
2. Raw counts remain raw; integrated assays and cell-state labels are excluded.
3. Different modality barcode sets remain unpaired populations unless their
   barcodes are exactly equal in the same order.
4. Human and mouse atlases are separate. Cross-species parameter sharing will
   require an explicit ortholog map rather than matching gene symbols by chance.
5. Different genome builds cannot share a peak atlas before explicit liftover.
6. Shared genes, proteins and fixed genomic bins are selected only by training-
   study prevalence with lexical tie-breaking—not expression, labels or held-out
   data.
7. A first successful download creates `source.lock.json`; all later runs verify
   URL, byte count and SHA-256 against that immutable lock.

## Output layout

```text
wld_phase_a/
  raw/<cohort>/source.lock.json
  bundles/<cohort>/bundle_manifest.json
  bundles/<cohort>/modalities/<modality>/counts.csr.npz
  training_atlas/<species_build>/atlas_manifest.json
  harmonized/<species_build>/<cohort>/bundle_manifest.json
  phase_a_ingestion_report.json
```

The harmonized bundles are inputs to cross-modal snapshot pretraining. A
separate prior-compilation phase must intersect the human/mouse feature atlases
with motif, contact and signed circuit evidence before WLD v4 can be trained.
