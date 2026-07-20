# WLD v5.5 mechanistic chromatin-twin contract

## Purpose and claim boundary

WLD v5.5 is a **mechanistic single-cell chromatin-response digital-model
prototype**. It predicts how an observed accessibility state changes after a
named perturbation while forcing the prediction through biological routes. It
extends v5.4 with a second route for chromatin-regulatory protein complexes and
with uncertainty-aware, whole-target validation.

The term *digital twin* is an engineering direction, not a present biological
claim. A true subject-level twin would require repeated measurements from the
same individual, ongoing data assimilation and recalibration, prospective
intervention validation, and a declared clinical or experimental use. The
current data contain different cells sampled at an endpoint, so v5.5 is neither
a continuously synchronized patient twin nor evidence of an attractor, fixed
point, basin, or kinetic time scale.

## Exact information flow

1. A control cell's full measured ATAC vector initializes the chromatin state.
   The subset aligned to the pretrained foundation atlas enters the context
   encoder; the full peak vector remains the state evolved by the model.
2. Optional measured protein, metabolite or environmental cues may enter only
   through declared named-node maps and explicit missingness masks. An absent
   assay is not imputed and presented as a measurement.
3. Guide identity is excluded from the encoder. After state encoding, it
   selects one named regulator intervention.
4. The intervention can reach peaks through two auditable branches:
   `regulator -> supported signaling/TF interaction -> motif-supported peak`,
   and `regulator -> curated protein complex -> training-derived accessibility
   module -> supported peak`.
5. The two mechanistic drives are summed. Context-conditioned opening, closing
   and recovery rates evolve the observed peak state with the numerical ODE
   integrator.
6. Neural context may modulate gains and rates on existing routes. It has no
   guide-conditioned or context-only decoder to future peaks and cannot create
   an unsupported edge.

This modular, intervention-driven design follows the useful precedents of an
integrated whole-cell model ([Karr *et al.*, Cell 2012](https://doi.org/10.1016/j.cell.2012.05.044))
and explicit perturbational dynamics in
[CellBox](https://www.sciencedirect.com/science/article/pii/S2405471220304646).
The staged specification, calibration and validation boundary follows the
[immune-system digital-twin roadmap](https://doi.org/10.1038/s41746-022-00610-z).
These references motivate the contract; they do not validate WLD.

## Shared evidence versus biological variation

The model freezes evidence that should be shared: genome build and coordinates,
named-node vocabulary, whole-target split, curated interaction support,
TF-to-peak motif support, protein-complex membership, feature mappings and
provenance hashes. Complexes come from experimentally curated resources such as
[CORUM](https://pmc.ncbi.nlm.nih.gov/articles/PMC9825459/) and are represented as
physical entities distinct from their components, consistent with the
[Reactome data model](https://reactome.org/documentation/data-model/).

It does **not** freeze quantities expected to differ across cells, tissues or
subjects. The observed starting state, supported-route gains, opening and
closing rates, recovery strength, and relative TF/complex contribution are
context conditioned and trainable. Population parameters provide shrinkage,
not forced equality. Context must be inferred from measured biology and
declared numeric cues; tissue, study, donor, cell-type and target labels cannot
serve as encoder lookup keys.

Reference membership is not proof that a complex is active in every cell.
Accessibility and measured context gate its realized contribution. Unsupported
targets remain unsupported rather than receiving a dense neural shortcut.

## Training-only accessibility modules

Complex-to-peak effects are empirical priors compiled strictly inside the
training partition:

1. For each training target, compare its perturbed cells with same-screen
   controls as unpaired populations.
2. Bootstrap cells within target and control groups, compute pseudobulk peak
   responses, and retain effects that pass prespecified magnitude and sign-
   stability thresholds.
3. Aggregate stable responses only across perturbed members of each curated
   complex. Robust signed loadings define that complex's accessibility module.
4. Freeze the compiled module, source hashes, thresholds, training targets and
   random seeds before validation.

Validation and test targets cannot influence module selection, signs, peak
loadings, feature selection or thresholds. A complex with no training-supported
response route is unavailable at evaluation, even if its membership is known.
This distinguishes curated physical topology from perturbational evidence about
direction and magnitude.

## Leakage and falsification controls

- Split complete perturbation targets, subjects and studies before any learned
  feature or module construction; keep sealed tests unopened during development.
- Exclude guide, target, barcode, donor, study, condition, cell-type, cluster,
  pseudotime, integrated embedding and future/outcome measurements from the
  encoder.
- Treat control and endpoint cells as unpaired populations; do not manufacture
  cell trajectories by nearest neighbors, optimal transport or pseudotime.
- Report persistence, TF-only, complex-only, both-routes-zero, frozen branch
  removal, and degree/support-preserving membership or sign shuffles.
- Use identical evaluation cells and random projections for each frozen
  comparison. No minimum route effect is forced.

## Uncertainty and reporting

Train multiple prespecified seeds and retain seed-level results. Treat a
perturbation target—not a cell—as the inferential sampling unit: fixed,
prespecified cell subsamples vary across seeds to expose sampling/algorithmic
sensitivity, then confidence intervals bootstrap target-level seed averages.
This is not presented as a nested biological cell bootstrap. Report confidence
intervals for SWD improvement over persistence, response NRMSE, response cosine
and frozen-route effects. Report ensemble spread, route coverage and unsupported
targets rather than hiding them in an aggregate. Fit interval widths on a
disjoint calibration-target block and assess coverage on a third untouched
audit-target block; these remain development uncertainty, not clinical
probabilities.

## Validation ladder

1. **Software contract:** topology masks, gradients, interventions, numerical
   integration, split enforcement and synthetic recovery tests.
2. **Internal perturbation development:** unseen whole targets, persistence and
   capacity-matched shuffled/retrained controls, plus frozen-route ablations.
3. **External study generalization:** frozen model and priors on an unopened
   study, with assay/genome adapters fixed without outcome access.
4. **Subject/tissue adaptation:** calibrate only permitted context-dependent
   parameters on predeclared adaptation data, then assess unseen interventions.
5. **Longitudinal twin assessment:** repeated same-subject observations,
   prospective intervention predictions and documented update/calibration
   behavior.
6. **Attractor assessment:** only genuine temporal/release data can support
   fixed-point residuals, stable Jacobian spectra, basin return and perturbation
   recovery claims.

Passing an earlier rung does not imply a later one. v5.5 may support a transient
chromatin-response statement only if it beats persistence and retrained route
controls and its fitted mechanistic branches matter under frozen ablation.
