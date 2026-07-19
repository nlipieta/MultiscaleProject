# WLD Phase B contract: biological priors and snapshot pretraining

## What Phase B now does

Phase B converts the repaired, training-only Phase A atlas into a bounded WLD
subnetwork and begins real multi-study representation pretraining. The prior
compiler requires the intersection of:

1. Phase A genes and 2 kb chromatin bins selected without held-out studies;
2. peak-to-gene contact/promoter evidence;
3. localized TF motif evidence;
4. signed TF-to-gene regulation;
5. signed signaling paths when available.

An edge cannot be created solely because it improves training loss. Unsupported
edges remain absent. Evidence signs remain fixed. Within the supported graph,
the WLD architecture retains context-conditioned edge gains, production, decay
and chromatin time scales, so their effective values can vary between cells,
subjects and tissues.

## What is trained from snapshots

Phase A human cohorts with exactly matching RNA/ATAC barcodes are used for
cross-modal representation learning. An ATAC-only view and an RNA-only view of
the same measured cell must agree in the context latent space and in their
prior-projected TF activities. Protein is used only when its measured feature
can be explicitly mapped to a supported signaling node. Missing protein and
metabolic assays remain missing.

Study, donor, barcode, cluster, cell-type and integrated coordinates are not
model tensors. Study identifiers are used only to enforce whole-study train and
validation partitions.

## Why ODE kinetics are not trained in this phase

Cross-sectional snapshots do not identify a temporal vector field. Phase B does
not turn cell similarity, pseudotime or nearest-neighbor matching into fabricated
future cell pairs. It trains the multimodal representation while leaving kinetic
estimation for longitudinal or perturbational data. The variable kinetic
architecture remains present and trainable in that later stage.

## Evidence boundaries

- Motif occupancy says that binding is physically possible; it does not prove
  TF activation or whether chromatin opens or closes.
- Peak-to-gene evidence is topology/confidence support, not a claim that 3D
  contact strength is invariant across tissues.
- TF-to-peak signed effects remain empty until perturbational chromatin evidence
  establishes opening versus closing.
- Human GRCh38 priors are never applied to mouse cohorts. Mouse pretraining
  requires its own genome, signed circuit and contact compilation.
- The current Phase A cohort set is a real beginning, not yet the requested
  large pan-tissue foundation corpus.

## Restart and sealing

Prior artifacts and every training epoch are written to Google Drive. A rerun
resumes the same configuration. GSE183273 and GSE214546 remain sealed, and no
attractor or biological-performance claim is produced in Phase B.

