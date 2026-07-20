# WLD v3: Circuit-first temporal dynamics

## Scope

WLD v3 is the recommended temporal model for testing whether a validated
regulatory circuit explains latent cell-state dynamics. It is intentionally
separate from the single-snapshot PBMC reconstruction runner. The public PBMC
sample can test cross-modal state reconstruction, but it cannot identify a
vector field, fixed point, basin, transition, or attractor.

A chromatin toggle-switch or synthetic cell-engineering demonstration is not
part of this path. Such a toggle is a possible downstream application after
the biological dynamics have been identified and falsified; it is not evidence
that the model learned an endogenous attractor.

## Causal state and hard prior contract

The v3 state is

\[
x(t) = [s(t), z(t), a(t), m(t)],
\]

where `s` is signaling-protein activity, `z` is core TF activity, `a` is
peak-level chromatin accessibility, and `m` is RNA abundance. The causal flow
is

```text
measured extracellular/tissue cue
                |
                v
       signed signaling/PPI graph
                |
                v
        signed core TF circuit <----> slow chromatin state
                |                           |
                +--- TF x motif x enhancer--+
                                |
                                v
                         RNA production
```

The graph is not passed to a general neural network. Each supplied edge creates
exactly one positive-magnitude kinetic parameter; an absent edge creates no
parameter. The prior fixes the activating or repressive sign. Positive Hill
thresholds and coefficients govern regulatory occupancy, and positive decay
rates make production and degradation explicit. The only trainable node-level
terms are basal production, decay, and the chromatin timescale.

Prior compilation rejects internally inconsistent edges: each TF-to-gene edge
must have localized peak evidence, each TF-to-peak effect must have localized
binding evidence, and every TF-to-TF circuit edge must agree in presence and
sign with regulation of the target TF's own gene.

Cell-specific TF-to-gene regulation requires the intersection of three pieces
of evidence:

1. the enhancer or promoter is open in that cell;
2. the TF has a localized motif or occupancy call in that peak; and
3. that peak is linked to the target gene and the TF-gene relation is supported.

This is the enhancer-level causal chain used by
[SCENIC+](https://www.nature.com/articles/s41592-023-01938-4), rather than a
random dense ATAC-to-gene projection. Signed Hill dynamics follow the same
mechanistic direction as the prior-guided neural ODE in
[PHOENIX](https://link.springer.com/article/10.1186/s13059-024-03264-0).
Perturbations enter the vector field explicitly, following the experimental
logic of [CellBox](https://www.sciencedirect.com/science/article/pii/S2405471220304646).

## Required input evidence

| Model component | Required evidence | Examples | Failure mode if absent |
|---|---|---|---|
| Initial chromatin | measured peak-level ATAC normalized to [0, 1] | paired scATAC/multiome | landscape is not observed |
| TF binding feasibility | localized motif or occupancy | motif scan, ChIP/CUT&Tag | openness is mistaken for binding |
| Enhancer-to-gene link | defensible cis-regulatory link | co-accessibility, ABC, validated promoter link | peaks are mapped to arbitrary genes |
| Core circuit | signed, confidence-weighted TF relations | CollecTRI plus experiment-specific evidence | circuit sign/topology is not identifiable |
| Signaling layer | signed cue-to-protein and protein-to-TF edges | perturbation-aware pathway/PPI prior | tissue cue cannot enter mechanistically |
| Time | true sampling interval or derivative information | time course, metabolic labeling, velocity | snapshot reconstruction is mislabeled dynamics |
| Falsification | held-out intervention and biological group | donor, replicate, perturbation | model memorizes identity/state proxies |

All preprocessing, feature selection, peak-to-gene inference, and prior
filtering must be fitted on training groups only. RNA target values, cell-type
labels, clusters, pseudotime, target-state labels, and future measurements
must not enter the initial encoder. Initial RNA is allowed only when it is a
real time-zero measurement in a future-state prediction experiment.

## Training path

1. Split by donor, replicate, or independent experiment before any
   data-dependent feature or prior selection.
2. Compile the signed signaling, TF-circuit, TF-to-peak, TF-to-gene, motif, and
   peak-to-gene tensors from training evidence.
3. Initialize chromatin from time-zero ATAC, TF activity from localized motif
   accessibility, and signaling from measured cues. Do not use a dense
   cell-identity encoder.
4. Integrate the hard-constrained ODE over the measured interval. Fit RNA and,
   when available, ATAC and derivative/velocity observations. Apply a terminal
   velocity penalty only to plateaus defined by the experimental design.
5. Select hyperparameters using held-out training groups. Freeze the complete
   analysis before evaluating the test donor/replicate/experiment.

Interventions use non-negative activity or edge scales: zero represents an
inhibition/knockout and values above one represent increased activity. Negative
scales are rejected because they would reverse a validated edge sign.

## Required comparisons

The circuit interpretation is supported only if the full model consistently
beats all relevant controls on held-out groups:

- mean and regularized linear prediction baselines;
- a matched model with the signed graph degree-preservingly permuted;
- chromatin profiles shuffled across held-out cells;
- cues or interventions shuffled within valid experimental blocks;
- the same mechanistic model with circuit edges disabled;
- the same model with enhancer gates disabled or replaced by nonlocalized
  gene-level accessibility.

The primary temporal metrics should be defined before test evaluation and
reported per donor/replicate, not only after flattening all cells and genes.

## Attractor audit

A low prediction loss does not establish an attractor. For each candidate
steady state, v3 exposes four separate checks:

1. refine a fixed point without reference to held-out target labels;
2. verify a small full-vector-field residual;
3. compute the full-state Jacobian and require negative real parts for local
   asymptotic stability; and
4. perturb many nearby initial conditions and measure return to the same state.

The strongest evidence is prospective: interventions predicted to leave,
return to, or cross a basin boundary must reproduce those outcomes in held-out
experiments. If the real temporal data do not support these tests, the result
is a constrained transition model, not an attractor model.

## Repository implementation

- `wld_circuit_dynamics_v3.py` implements the hard-sparse multiscale vector
  field, RK4 integration, explicit circuit interventions, temporal loss, fixed
  point refinement, Jacobian spectra, basin-return tests, and the signed
  degree-preserving negative control.
- `run_wld_v3_validation.py` checks the architecture contract and diagnostic
  plumbing on small neutral systems. It does not claim biological validation
  and deliberately contains no toggle-switch benchmark.
- `run_wld_pbmc_colab.py` remains the single-snapshot ATAC-to-RNA benchmark and
  never invokes the v3 temporal model.
