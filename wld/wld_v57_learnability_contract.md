# WLD v5.7 response-learnability contract

## Purpose and boundary

WLD v5.7 is a **response-only diagnostic ladder** run after the completed v5.6
model failed its practical-effect audit. It asks whether GSE161002 contains a
reproducible target-minus-control accessibility response and, if so, whether
that response is low-rank and predictable from the frozen biological routes.
It diagnoses measurement, representation, prior, or optimization failure
before another WLD model is trained.

This stage may fit prespecified linear and small static nonlinear diagnostic
baselines. It does not initialize, update, select, or evaluate a new WLD
checkpoint. Historical WLD results are comparators only. Full-state SWD,
cell-distribution prediction, new ODE dynamics, kinetic time, fixed points,
basins, multistability, recovery, digital-twin status, and attractors are all
deferred.

The 16 v5.3 test targets, muscle subjects J/L, and external studies remain
sealed. The 16 v5.5/v5.6 validation targets have already been inspected and
remain disclosed reused-development data. Nothing in v5.7 is confirmatory.

## Durable inputs and output

The Colab runner verifies these artifacts under `MyDrive/WLD_Backup`:

- `wld_phase_b/priors/homo_sapiens_grch38`;
- `wld_v53_crispr_sciatac_ingestion/bundle`;
- `wld_v55_chromatin_twin_r3/tf_routes`;
- `wld_v55_chromatin_twin_r3/complex_modules`; and
- `wld_v56_nullaware_r2/practical_effect_audit/` followed by
  `wld_v56_practical_effect_audit.json`.

The output is
`wld_v57_response_learnability/wld_v57_response_learnability_report.json`.
The v5.6 audit must retain an immutable source, a failed corrected practical
gate, and false flags for test opening, inference, digital-twin status, and
attractor status.

## Leakage-safe response construction

Responses are screen-matched target-minus-NTC accessibility pseudobulks. Target
and NTC cells on opposite split halves are disjoint. Complete perturbation
targets—not cells—are the model-selection and evaluation units; all screens for
one target stay in one fold. Targets receive equal weight after within-target
screen aggregation.

Feature bins and routes remain those compiled without test outcomes. Target
name, guide, barcode, cell type, cluster, integrated embedding, future state,
and validation/test response are forbidden predictors. A target may select its
fixed route profile after encoding; it may not become a one-hot shortcut.

Ranks, normalization, response bases, hyperparameters, and model parameters are
selected using training targets only. Permuting reused-development responses
must not change a selected transform or hyperparameter. Unsupported targets
receive exact zero-response predictions and remain in all-target summaries.

## Prespecified ladder

### 1. Split-half reliability

Use at least 50 deterministic disjoint cell splits and 200 screen-matched
NTC-versus-NTC permutations. A target requires at least 128 target cells and
256 matched NTC cells. Target and control response halves therefore contain at
least 64 cells each, while the null can use four disjoint 64-cell NTC groups.
The null averages two NTC-minus-NTC differences, matching the variance of the
observed `(split A + split B) / 2` response statistic; multi-screen null vectors
are averaged before their RMS is scored. Underpowered targets count as not
demonstrated rather than being omitted.

A target is reproducible when median split-half response cosine is at least
`0.20`, at least 80% of split cosines are positive, and response RMS exceeds
the matched NTC null 95th percentile. The cohort advances with at least eight
route-supported targets and at least 50% of all route-supported targets passing
in both the training and reused-development partitions.
The existing `0.002` NRMSE denominator floor is numerical protection, not
evidence of detectability.

### 2. Low-rank measurement ceiling

For reproducible training targets, learn response bases from split A with the
evaluated target left out. Select rank from `1, 2, 4, 8, 16`, capped at one
quarter of the available training-target count. Project the held target's split
A response into that basis and predict split B. Because this uses the same
target's split-A response, it is an oracle compressibility ceiling—not an
unseen-target predictor.

The ceiling passes when rank is at most 16, median response NRMSE is at most
`0.80`, median response cosine is at least `0.30`, at least 60% of reproducible
targets have NRMSE below `0.90`, and the basis captures at least 50% of the
squared-error improvement available from raw split A over zero. Report zero
response, other-training-target screen mean, raw split A, and bin-permuted
comparators on identical split-B outcomes.

### 3. Route-linear whole-target generalization

Build an unsupervised embedding from the frozen regulator-to-TF support profile
and regulator-to-complex membership profile, plus a response basis from
training responses only. Nested whole-target ridge regression maps these route
coordinates to response coordinates. Reused-development targets are scored
only after training-only selection. In this diagnostic, `route_supported`
means that this upstream named-node profile is nonzero; it does not claim that
a downstream TF/motif or complex/module path reaches every scored response bin.
Failure can therefore identify an incomplete target-to-route or
route-to-response mapping rather than disproving the underlying biology.

The route-linear screen passes only when mean response NRMSE is at most `0.95`,
median response cosine is at least `0.10`, response-space squared error improves
by at least 1% over zero response, at least 60% of reproducible route-supported
targets beat zero response, and true routes beat both the screen-matched
training perturbed-mean baseline and the 95th percentile of at least 20 matched
route shuffles. The ridge and kernel-ridge solutions are deterministic; the
prespecified seeds govern target folds, cell splits, and topology-null draws,
not redundant optimizer initializations.

### 4. Static nonlinear whole-target generalization

Use the same route embedding, response basis, folds, cell draws, seeds, and
scores as route-linear. The only extra capacity is a prespecified static map,
such as RBF kernel ridge or one hidden layer of width at most 32. Tuning stays
inside training-target folds.

The nonlinear gate requires mean/median response NRMSE at most `0.90/0.95`,
mean/median response cosine at least `0.20/0.10`, at least 1% response-space
squared-error improvement over zero, and improvement on at least 60% of
reproducible route-supported targets. It must beat the generic perturbed mean,
the route-shuffle ensemble, and route-linear response NRMSE by at least 2%,
unless route-linear fails and the nonlinear model independently passes every
stronger absolute gate.

Report sensitivity at cosine `0.10/0.20/0.30` and NRMSE
`0.95/0.90/0.80` without changing the preregistered decision.

## Baseline and shuffle fairness

Zero response, training-only screen mean, raw split half, route-linear, static
nonlinear, and shuffled routes use identical target rosters, matched control
pools, train-selected bins, response scaling, seeds, and target-equal
aggregation. The generic mean is built separately by screen from training
targets only.

Each route shuffle moves the **whole upstream TF/complex route profile** as one
unit within the immutable train or reused-development stratum. It preserves the
exact profile values, nonzero footprint, signs, and absolute mass of every moved
row. It does not pretend to preserve downstream response-bin path mass that is
not part of this v5.7 feature matrix. Shuffling may read target names and split
labels only; it must never load, inspect, summarize, or transform a sealed-test
ATAC value. True and shuffled models use identical capacity and nested
selection. Report all targets, the reproducible subset, the route-supported
subset, and their intersection without silent filtering.

## Deterministic diagnosis

The report emits one primary class plus applicable flags:

- `MEASUREMENT_LIMITED`: split-half reliability fails;
- `STABLE_HIGH_RANK_OR_LATENT_BOTTLENECK`: reliability passes but the ceiling
  fails;
- `ROUTE_PRIOR_OR_TARGET_MAPPING_INSUFFICIENT`: the ceiling passes but route
  predictors do not generalize;
- `TARGET_NONSPECIFIC_SCREEN_RESPONSE`: the generic screen response is
  competitive with or better than route models;
- `LINEAR_ROUTE_SIGNAL`: route-linear passes, so an ODE is unnecessary here;
- `STATIC_NONLINEAR_LEARNABLE`: linear fails but static nonlinear passes;
- `WHOLE_TARGET_SHIFT_OR_OVERFIT`: nested training performance does not survive
  reused-development targets;
- `TOPOLOGY_NONSPECIFIC`: biological routes do not beat matched shuffles; and
- `WLD_OPTIMIZATION_OR_PROPAGATION_FAILURE`: a simpler response model passes
  while historical WLD does not.

Every result also carries `TRANSIENT_RESPONSE_ONLY_NO_DYNAMICS`. A response-only
pass justifies designing a later, separately frozen full-state or temporal
experiment; it does not retroactively validate dynamics.

## Required tests and claims

The smoke suite covers null response, reproducible low-rank linear response,
stable high-rank response, nonlinear response, route-independent response,
screen confounding, target-equal weighting, disjoint target/NTC halves,
unsupported-target zero response, route-shuffle invariants, training-only
selection, sealed-test guards, and deterministic seeded control output. The
completed v5.6 near-persistence result must remain classified as not learned.

The final report states `development_only=true` and
`historical_wld_results_only=true`; it states `fresh_wld_training=false`,
`test_values_materialized=false`, `test_targets_evaluated=false`,
`confirmatory_inference=false`, `digital_twin_claim=false`, and
`attractor_claim=false`.
