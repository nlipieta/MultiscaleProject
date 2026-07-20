# WLD expanded-corpus representation-pretraining contract

This stage continues the existing WLD Phase B human representation checkpoint
using both the original Phase A development cohorts and the newly constructed
human GRCh38 expansion cohorts.

## Pairing-aware objectives

- Exact RNA/ATAC pairs use cell-level cross-modal contrastive and prior-TF
  consistency losses.
- Unpaired RNA/ATAC observations are sampled independently and use only latent
  distribution moments and covariance alignment.
- Expression similarity, labels, embeddings, pseudotime, nearest neighbors,
  optimal transport and row order never manufacture cell pairs.

## Biological context and circuit boundary

The structured encoder and context network learn from measured accessibility,
RNA and available protein. Study, donor, barcode, tissue label, cell type and
cluster are partition/audit metadata rather than encoder tensors. The learned
context is the input that can later modulate supported circuit gains, basal
production, decay, thresholds and chromatin time scales.

Those kinetic quantities are not globally fixed, but cross-sectional snapshots
do not identify their temporal values. Therefore the circuit-field parameters
remain `requires_grad=True` for longitudinal/perturbational fine-tuning while
this snapshot stage optimizes only the identifiable representation modules.
The report hashes the representation before and after training to prove that it
changed, hashes the circuit field to prove that it did not, and verifies that
the snapshot optimizer contains no circuit-field parameter. This is a
stage-specific identifiability rule, not a claim that those biological rates
are constant across cells, subjects or tissues.

## Species and feature spaces

New human GRCh38 cohorts are projected into the immutable Phase A training
atlas before use with human motif/contact/signed-circuit priors. Mouse mm10
cohorts remain in their own atlas and are listed as staged; they require their
own motif, contact, signed-regulatory and signaling prior compilation. Human
priors are never applied to mouse measurements.

## Validation and non-claims

The pre-existing Phase A whole-study validation split remains unchanged.
Expansion cohorts are training-only. GSE183273 and GSE214546 remain sealed and
are neither downloaded nor evaluated. This stage produces no test metric,
temporal-dynamics estimate, attractor basin or attractor-state claim.
