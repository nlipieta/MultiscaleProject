# WLD v5.6 null-aware mechanistic chromatin-response contract

## Scientific scope

WLD v5.6 is a development-stage model of **single-endpoint, population-level
chromatin response to named perturbations**. It asks whether a measured control
state, a named regulator intervention, open-chromatin context, curated
regulator/TF and protein-complex evidence, and train-derived accessibility
modules can predict a held-out target-minus-control accessibility response.

It is not yet a biological or clinical digital twin. It is not continuously
synchronized to one physical counterpart, has no subject-specific update loop,
and has not demonstrated counterfactual validity for an individual. A single
post-perturbation endpoint does not identify kinetic time, fixed points, basin
boundaries, multistability, or attractors. Those claims remain false regardless
of predictive performance in this phase.

## Why v5.5 did not qualify

The completed v5.5 development run produced a useful negative result:

- persistence-minus-true SWD was **-0.000208** (95% target interval
  **-0.000405 to -0.000061**), so persistence was better;
- frozen-route-removal-minus-true was also **-0.000208**, so removing the fitted
  mechanistic paths did not hurt and fitted path reliance was false;
- retrained topology-shuffle-minus-true was **+0.000243** (95% target interval
  **+0.000001 to +0.000365**), a small topology-specific result that did not
  rescue prediction; and
- useful perturbation prediction, digital-twin status, and attractor status all
  remained false.

This is not evidence that circuit architecture is irrelevant. It shows that the
v5.5 optimization contract did not make the non-null mechanistic response a
competitive solution. Its multistage route gains began far enough from
persistence to create avoidable initial movement, while multiplicative gains
could still leave a fitted route functionally negligible. Moreover, L2 applied
to inverse-softplus ``raw_*`` coordinates can move them toward zero while the
*realized* positive gain moves toward ``softplus(0)``, so raw-parameter shrinkage
is not a reliable mechanistic-effect penalty. Checkpoint selection was dominated
by full-state distributional agreement; in a sparse perturbation response, that
objective can reward the large unchanged background and favor near-control
predictions without learning the target-specific change.

The v5.6 changes below are therefore optimization and null-calibration repairs,
not permission to reinterpret v5.5 as a success.

## Null-aware model requirements

1. **Near-persistence initialization.** The initialized intervention response
   must be small relative to the observed training response and full ATAC state,
   while retaining finite, nonzero gradients for every supported route gate.
   Initialization must not hard-freeze any quantity that can vary by cell,
   tissue, subject, condition, or study.

2. **Learnable mechanistic gates.** TF and complex-route gates start small but
   differentiable. A synthetic supported response must move the appropriate
   gate measurably away from its initialization. Unsupported edges cannot be
   created by a gate or neural context adapter.

3. **Realized-effect regularization.** Complexity control acts on bounded
   effective branch gates and the actual predicted endpoint displacement, not
   on inverse-link/raw coordinates. The response-focused training loss—not an
   artificial minimum-effect constraint—must provide the evidence for learning
   away from persistence. Regularization may not use validation values or any
   sealed-test outcome, and it may not force a nonzero effect when the data do
   not support one.

4. **Exact null behavior.** Zero intervention must return the measured control
   state bit-for-bit. A frozen evaluation with all mechanistic routes set to
   zero must also return exact persistence. Neural context, recovery dynamics,
   encoder outputs, and numerical integration may not move the state in either
   case.

5. **Route normalization.** Each named regulator's TF and complex drive is
   normalized by its fixed supported path mass/footprint before learnable gains
   act. A hub cannot obtain a larger initial effect solely because it has more
   annotated TFs, complexes, modules, or bins. Normalization is computed from
   evidence topology, never from validation or test response values.

6. **State-dependent parameters remain trainable.** Accessibility gates,
   context-dependent gains/rates, and measured modality contributions may vary
   across cells and cohorts. Evidence support is fixed; biological activity is
   not. Context can modulate a supported route but cannot add a new edge or a
   direct context-to-peak residual decoder.

7. **Device correctness.** Model parameters, registered evidence buffers,
   intervention tensors, full-bin control states, route normalizers, losses and
   gradients must remain on the selected device. The synthetic contract runs on
   CPU and on CUDA when CUDA is available.

## Response-focused development objective

Model fitting and checkpoint selection must explicitly score the perturbation
response

`mean(predicted perturbed ATAC) - mean(matched control ATAC)`

against the observed target-minus-control response. Response NRMSE and response
direction/cosine carry the primary selection weight. Full-state sliced
Wasserstein distance and the literal persistence comparator remain mandatory:
a checkpoint cannot qualify merely by matching response direction while making
the complete cell-state distribution worse than persistence.

This emphasis follows the caution raised by the Systema evaluation framework:
control-referenced aggregate scores can absorb systematic experimental or
dataset variation and should not automatically be read as perturbation-specific
biological prediction (Systema, DOI
[10.1038/s41587-025-02777-8](https://doi.org/10.1038/s41587-025-02777-8)).
WLD therefore reports both target-minus-control response metrics and full-state
metrics, keeps screen/cohort grouping explicit, and does not use cell identity
or target-state proxies in the encoder.

A screen-matched **perturbed-mean baseline** is compiled from training targets
only: within each screen it averages training target-minus-NTC pseudobulk
responses, then applies that generic shift to reused validation controls. The
biological model must beat this baseline as well as literal persistence. This
separates named-route prediction from merely learning that perturbations tend
to cause some average response. Validation and sealed-test values are never
used to construct the baseline.

## Topology-control contract

- At least **10** independently seeded, end-to-end matched topology controls
  are fitted with the same capacity, optimizer, target roster and checkpoint
  rule as the biological graph.
- TF and complex route profiles are permuted jointly so the null preserves the
  distribution of full genomic footprint, fixed complex-route signs, and signed
  and absolute path mass.
- Permutations occur **within the immutable whole-target split**. Train,
  validation and test each retain their own exact support/footprint/mass
  distribution. Only target names and split labels may be read for this step;
  sealed-test ATAC observations are never loaded.
- Each permutation has no fixed regulator label, differs from the biological
  topology, and has an immutable seed, permutation digest and topology digest.
- This control tests assignment of named regulators to mechanistic route
  profiles. It does not prove that the curated downstream evidence is complete
  or uniquely correct.

## Data separation and leakage boundaries

- Whole perturbation targets, subjects and studies are split before feature,
  module or parameter selection.
- Complex accessibility modules and any response-informed normalization are
  built from training targets only.
- All v5.5 validation targets have already been inspected. v5.6 may reuse them
  for disclosed architecture development and checkpoint selection, but it must
  label them ``previously used``, compute no untouched-audit interval or
  p-value, and make no confirmatory claim from them.
- Sealed test matrices are not materialized during development. Split labels
  and target names may be used to preserve control balance, but no test count,
  accessibility value, outcome statistic or label-derived phenotype enters
  training, selection, normalization, regularization or calibration.
- Guide/target identity enters only as a named post-encoder intervention. Cell
  type, pseudotime, target state, integrated identity labels and direct RNA-count
  proxies are forbidden encoder inputs.

## Development gates and future qualification

The reused-development report must keep these questions separate and label all
comparisons descriptive. It may decide whether to freeze a new confirmatory
plan, but it may not compute a confidence interval, p-value, or claim from the
previously inspected v5.5 validation targets:

1. **Non-null predictive utility:** the biological model descriptively beats
   literal persistence and the training-only perturbed-mean baseline on
   full-state SWD, has response NRMSE below persistence (`< 1`), and has
   positive response cosine.
2. **Fitted route reliance:** frozen removal of all routes descriptively
   worsens prediction.
3. **Topology specificity:** the biological topology beats the prespecified
   ensemble of at least ten matched, retrained controls; all controls use the
   same target and seed grid.
4. **Future confirmation plan:** only if all descriptive development gates
   pass may a new plan be frozen. Target-level intervals and any p-value then
   wait for a newly frozen, untouched target/study evaluation. The reused v5.6
   development set cannot be relabeled as calibration or audit data after the
   fact.

Failure of any item is reported as a negative or inconclusive result, not
hidden by averaging. Passing all four supports only the phrase
"mechanistically constrained transient chromatin-response model." It still
does not support a digital-twin or attractor claim.

## Required synthetic checks

Before real-data development, the smoke suite must demonstrate:

- small but nonzero initialized movement and nonzero gate gradients;
- learning away from the near-persistence initialization on a supported
  response fixture;
- realized-effect regularization that is zero when all routes are frozen,
  positive for a nonzero fitted effect, differentiable through the effective
  gates/output, and never implemented as L2 on inverse-link coordinates;
- exact persistence under zero intervention and frozen all-route removal;
- route-mass normalization under deliberately unequal graph degrees;
- CPU and, when available, CUDA parameter/buffer/forward/backward placement;
- checkpoint selection that prefers a response-correct model over a
  persistence-like full-state model while still enforcing full-state metrics;
- a screen-matched perturbed-mean baseline built exclusively from training
  targets, with explicit false flags for validation/test use during
  construction;
- at least ten distinct split-stratified matched topology controls; and
- explicit false flags for sealed-test evaluation, digital-twin status and
  attractor status.
