# WLD v4 pretrained temporal-fine-tuning contract

This stage continues the expanded-corpus WLD v4 checkpoint on the measured
GSE240061 pre-to-3.5-hour exercise experiment. It is a transient temporal
development experiment, not an attractor benchmark.

## Inputs

- The encoder receives measured time-zero ATAC and the declared exercise cue.
- Measured time-zero RNA initializes the RNA component of the ODE state. It is
  not passed into the encoder or context network.
- Protein and metabolic inputs remain explicitly missing because GSE240061 did
  not measure them.
- Subject IDs, cell IDs, cell-type labels, integrated embeddings, pseudotime,
  future measurements, and target-state labels never enter the encoder.

## Population alignment

Pre and post nuclei are destructive observations. Source and target cells are
sampled independently within each subject. The objective aligns future RNA and
ATAC distributions and never manufactures cell pairs using expression,
optimal transport, embeddings, labels, or nearest neighbors.

## Parameter policy

The hard evidence topology remains fixed. The structured representation,
context network, circuit gains, production rates, decay rates, thresholds,
Hill coefficients, and chromatin time scales all remain trainable. Distinct
learning rates protect the pretrained representation without declaring
cell-, tissue-, or subject-dependent biological variables constant.

## Development split and controls

- Temporal training subjects: E, G, N.
- Checkpoint-selection subject: I.
- J and L are not evaluated in this stage.
- Mandatory conditions: supported circuit, no TF circuit, and a sign shuffle
  on the same biologically supported topology.
- A persistence baseline compares the initial population directly with the
  future population.

The upstream snapshot checkpoint used the full GSE240061 bundle for
representation validation. J/L are therefore excluded here but cannot be
claimed as pristine test subjects. Clean assessment requires an external
temporal or perturbational study that remains sealed from every upstream
selection step.

## Claim boundary

The 3.5-hour endpoint is transient. Terminal velocity is reported but is not
penalized toward zero. No fixed-point, basin, or attractor claim is allowed in
this stage, regardless of validation performance.
