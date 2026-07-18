# WLD temporal-training data contract

`wld_temporal_training.py` is the biological-training entry point for the
hard-constrained WLD v3 vector field. It does not use the public PBMC snapshot,
RNA-derived pseudotime, nearest-neighbor early/late matching, cell labels, or
target-state labels. Those inputs cannot identify temporal dynamics.

## Experimental unit and split

Each transition declares a biological `group_id`, a measured positive
`horizon`, and an optional `terminal` flag. A group is the highest unit that
could share biological or technical information, such as a donor, animal,
organoid, clone, perturbation replicate, or independently executed experiment.

The manifest must assign every group to exactly one of `train`, `validation`,
or `test`. WLD selects checkpoints using validation groups and evaluates test
groups only after selection. Preprocessing, feature selection, normalization
parameters, peak-to-gene links, and every control prior must be fit without
validation or test groups. `priors_fit_groups` records that provenance and must
be a nonempty subset of the training groups.

If a donor contributes multiple time points, conditions, or perturbations, all
of them remain in the same split. Splitting cells from one donor across train
and test is leakage.

## Directory layout

```text
temporal_cohort/
├── manifest.json
├── observations.npz
├── priors.npz
└── signed_permuted_priors.npz   # optional prespecified control
```

The loader uses `allow_pickle=False`. String arrays therefore need a fixed
NumPy Unicode dtype such as `U64`, not an object dtype.

### `manifest.json`

```json
{
  "schema_version": 1,
  "alignment_mode": "distribution",
  "initial_feature_names": ["ATAC_peaks", "external_cue"],
  "priors_fit_groups": ["donor_01", "donor_02"],
  "split_groups": {
    "train": ["donor_01", "donor_02"],
    "validation": ["donor_03"],
    "test": ["donor_04"]
  },
  "transitions": [
    {
      "transition_id": "donor_01_stim_6h",
      "group_id": "donor_01",
      "horizon": 0.25,
      "terminal": false
    }
  ],
  "control_prior_archives": {
    "signed_degree_permuted": "signed_permuted_priors.npz"
  }
}
```

Every transition ID in the manifest must occur in both the initial and target
arrays. `horizon` uses the model's declared time unit. If physical hours are
rescaled, the scaling rule must be selected using training groups and then
frozen. The integration step count is a numerical parameter, not an observed
time interval.

`terminal: true` adds a terminal-velocity penalty. It should be declared only
when the experiment is designed to sample a relaxed endpoint; it is not a cell
type or attractor label.

`control_prior_archives` is optional. Each archive must have the same model
dimensions as `priors.npz` and must itself be compiled using training groups
only. Use it for prespecified controls such as a signed degree-preserving graph
permutation. The built-in `no_circuit` condition removes TF-to-TF edges.

### `observations.npz`

Required arrays:

| Array | Shape | Meaning |
|---|---:|---|
| `initial_atac` | initial cells × peaks | Time-zero accessibility in `[0, 1]` |
| `initial_cues` | initial cells × cues | Time-zero measured nonnegative cues |
| `initial_transition` | initial cells | Transition ID for each initial cell |
| `target_rna` | target cells × genes | Nonnegative future RNA measurement |
| `target_transition` | target cells | Transition ID for each target cell |

Optional arrays:

| Array | Shape | When it is valid |
|---|---:|---|
| `initial_rna` | initial cells × genes | RNA measured at the declared initial time; its use is recorded explicitly |
| `target_atac` | target cells × peaks | Accessibility measured at the future time |
| `target_derivative` | target cells × full state | A directly observed derivative for paired data |
| `initial_pair_id` | initial cells | Required with paired alignment |
| `target_pair_id` | target cells | Required with paired alignment |

The full-state derivative order is signaling nodes, TFs, peaks, then genes.

## Alignment modes

`distribution` is the default for destructive single-cell assays. It never
pretends that a cell destroyed at time zero is the ancestor of a particular
future cell. The objective compares the simulated and observed log-RNA
populations using sliced Wasserstein distance, gene-wise mean and variance
losses, optional future-ATAC alignment, an optional terminal-velocity loss, and
edge-gain regularization.

`paired` is allowed only when explicit pair or lineage IDs establish a real
one-to-one correspondence within every transition. Pair IDs must be unique and
identical across the initial and target samples. Sorting by pseudotime,
nearest-neighbor matching, optimal transport, shared cluster labels, or matching
on future RNA does not create a valid pair ID.

## Prior archive

`priors.npz` contains the hard circuit scaffold expected by
`MultiscaleCircuitPriors`:

- `peak_to_gene`
- `peak_tf_motif`
- `tf_gene_support`
- `circuit_tf_tf`
- `tf_gene_index`
- `signal_signal`
- `signal_tf`
- `cue_signal`
- `tf_peak_effect`

Zeros remove parameters from the model. Nonzero signs are fixed; training can
adjust only the magnitudes on supplied edges. The model has no dense neural
residual capable of bypassing this scaffold.

## Commands

Validate a prepared cohort before allocating a GPU:

```bash
python wld_temporal_training.py validate --data /path/to/temporal_cohort
```

Run the prespecified circuit and no-circuit conditions:

```bash
python wld_temporal_training.py benchmark \
  --data /path/to/temporal_cohort \
  --output /path/to/results \
  --conditions true_circuit,no_circuit \
  --epochs 100 --steps 12
```

If the manifest supplies a `signed_degree_permuted` archive, include it in the
same prespecified run:

```bash
python wld_temporal_training.py benchmark \
  --data /path/to/temporal_cohort \
  --output /path/to/results \
  --conditions true_circuit,no_circuit,signed_degree_permuted
```

The output contains one validation-selected checkpoint per condition and
`wld_temporal_results.json`, including per-transition and per-group held-out
metrics, the training-target mean baseline, and control differences. For test
transitions prespecified as terminal, it also refines a fixed-point candidate,
reports the vector-field residual, evaluates the full Jacobian when the state
dimension is within the configured limit, and runs perturb-and-return basin
diagnostics. It never prints an unconditional success claim.

## What the result can establish

Beating the train-mean, no-circuit, and signed degree-preserving controls on
held-out biological groups supports temporal predictive value and dependence on
the supplied circuit. It does not by itself establish an attractor. An attractor
claim additionally requires held-out fixed-point residuals, negative real parts
of Jacobian eigenvalues, recovery from multiple perturbations within a basin,
and prospective intervention outcomes. Those analyses must use biological
measurements not used to select the model.
