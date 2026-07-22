# WLD v6.0 atlas-conditioned virtual-tissue contract

## Purpose and present claim boundary

WLD v6.0 is a **software and data-contract prototype** for an
atlas-conditioned mechanistic virtual tissue. It is intended to combine a
cross-study regulatory atlas with specimen observations, anatomical context,
declared cues and graph-constrained dynamics. The first Colab run validates
the architecture with synthetic fixtures and audits public study metadata. It
does not download assay matrices, train on biological measurements, evaluate a
held-out study, or establish biological predictive power.

"Digital twin" remains a target application rather than a result. A
specimen-specific digital twin would require repeated measurements from the
same animal, declared data-assimilation and recalibration rules, uncertainty
calibration and prospective intervention assessment. Likewise, regeneration
time points alone do not establish fixed points, basin return or attractors.
Those terms remain prohibited until the corresponding experiments pass.

## Scientific representation

The model retains the original WLD premise but makes every realized mechanism
context dependent:

```text
candidate regulatory support
  x local chromatin accessibility
  x regulator/protein availability
  x enhancer-promoter contact evidence
  x local cue and spatial context
= active regulatory interaction in this specimen at this time
```

The candidate graph is an auditable support scaffold assembled from genome
coordinates, sequence motifs, enhancer-promoter evidence, validated regulatory
interactions, curated complexes and named signaling routes. A candidate edge
is not assumed active in every cell and has zero realized effect until explicit
contextual route evidence is supplied. The active graph must change when measured
chromatin, protein, cue, spatial, donor or other declared biological context
changes.

Shared support and variable quantities are kept distinct. Species, genome
build, feature identity, coordinate mappings, curated memberships, named-node
identity, source hashes and split rosters may be frozen. Accessibility,
regulator abundance, contact strength, signaling gain, kinetic rates,
cell-neighborhood effects, metabolism and subject-specific deviations may not
be frozen merely because a shared atlas exists. Population parameters provide
shrinkage, not forced equality across cells, tissues or subjects.

The dynamic state may contain separate chromatin, regulator, transcriptional,
signal and optional metabolic compartments. Separate compartment kinetics are
required; a single unconstrained latent vector cannot silently stand in for all
of them. The v6.0 scaffold does not implement explicit transcriptional or
chromatin delay terms, so no learned-delay claim is permitted. Neural
components may infer compact context and bounded edge/rate
modulations, but the support mask is applied after every such modulation. No
neural decoder may create an unsupported edge or a direct future-state path.

## Hierarchical context and identity leakage

The active graph may be decomposed conceptually as

```text
shared support + lineage deviation + tissue deviation
               + state deviation + subject deviation
```

but every nonzero deviation must be carried by a provenance-backed biological
measurement or inference. A raw batch-length tensor, donor lookup table or
subject identifier is not a biological context measurement and must be
rejected. Bare identity, lineage, cell-type, cluster, tissue, study, donor,
subject, condition,
barcode, target, guide, pseudotime and integrated-embedding labels are
forbidden encoder inputs or lookup keys. They may be retained outside the
encoder for grouping, splitting, auditing and evaluation.

Declared exogenous cues and spatial coordinates may enter the model only when
their provenance is explicit and their measurement time is at or before the
prediction origin. A categorical cell label renamed as a cue is still leakage.
Future measurements, post-intervention outcomes and quantities
computed from a sealed evaluation partition are always forbidden.

All learned feature selection, normalization, context maps, atlases and
hyperparameters are fit inside training folds after studies, specimens and
applicable time points have been separated. Whole-study and whole-subject
generalization is the minimum meaningful evaluation unit.

## Observation provenance and uncertainty

Every biological value carries exactly one provenance state:

- `observed`: directly measured for the current specimen and assay;
- `reference_transferred`: transferred from a declared reference atlas;
- `model_inferred`: inferred from other observations by a declared model; or
- `unknown`: unavailable and not filled.

`reference_transferred` and `model_inferred` values require a finite,
non-negative uncertainty representation and source or method lineage. They may
never be relabeled `observed`. An `unknown` value remains masked; zero is not a
missing-value encoding. Observed values may retain technical uncertainty, but
the raw observation lineage must remain intact. Every tensor declares source
feature names and source lineage. Required mechanistic gates cannot silently
consume an unknown value: they fail closed or use a separately declared model
with retained inferred provenance. A measured physical zero remains known and
is distinguishable from an unknown masked entry throughout simulation.

Every provenance tensor also declares a finite physical measurement time.
It also declares an explicit source role, source partition and source accession
set. Development-time model boundaries accept only development, training,
reference-atlas, model-inference or synthetic-fixture roles. Any sealed,
test, validation, holdout or evaluation partition—and specifically accession
`GSE315993`—is rejected in context gates, initial states and learned residual
lineage, even if the numeric tensor and timestamp would otherwise be valid.
Temporal sealing applies uniformly to cues, accessibility, TF/protein
availability, contact evidence, spatial coordinates and adjacency, hierarchical
edge effects, metabolic observations and node-kinetic modulators. Any value
measured after the prediction origin is rejected; a non-cue feature cannot
avoid this rule by bypassing the causal-feature declaration table.

The same seal applies to the public initial ODE state. Chromatin, signaling,
TF, RNA and (when present) metabolic state must each enter through an explicit
`ProvenanceTensor` whose values match the named state and whose measurement
time is at or before the prediction origin. Raw-array `TwinState` values are
permitted only for internal RK stages and derivative results; public
`derivative`, `rollout` and `diagnostics` boundaries reject them. Thus a future
RNA or ATAC matrix cannot be relabeled as an unprovenanced time-zero state.

Spatial extrapolation is distributional. If a cell lacks measured position,
the system may infer a distribution over position from an explicitly allowed
reference, but cannot assign a single coordinate and present it as measured.
The same rule applies to inferred cues, cell-cell communication and metabolic
state. Predictions must propagate or otherwise report the uncertainty created
by these inferred inputs.

## Pairing, space and time

Modalities are exactly paired only when the deposited data provide a common
observation identifier or an explicit barcode crosswalk. Row order, expression
similarity, cell labels, nearest neighbors, optimal transport and pseudotime do
not establish exact pairing. When identifiers do not support a match, the
modalities remain unpaired population observations and use distributional
objectives. Pairing evidence is a structured record containing its mode,
evidence type, deposited evidence description and explicit false flags for
fabrication, expression similarity and label matching. A declared exact mode
is not counted as verified until the deposited barcode/spot relation has been
schema-checked. In particular, an unverified GSE315993 spatial relation remains
sealed and contributes no verified exact-pairing count.

Physical time and deposited anatomical coordinates are distinct from
pseudotime and embeddings. Spatial graphs must record their coordinate system,
construction rule, source lineage and uncertainty. Raw adjacency arrays are
not accepted as causal inputs. Neighbor messages pass only through a named,
support-masked and provenance-backed signal route; a spatial neural network may
not become an unrestricted cell-identity channel.

## Graph-constrained hybrid dynamics

The derivative is the sum of auditable supported components, for example:

```text
d chromatin / dt = supported opening - supported closing + recovery
d regulator / dt = supported circuit drive + bounded supported residual
d RNA       / dt = supported enhancer/gene drive - degradation
d cue       / dt = declared external input + supported neighbor exchange
```

The residual is bounded and projected through the biological support graph. It
may modulate existing edges or rates; it cannot add an output-space vector,
decode a cell label, or bypass the graph. With all supported routes removed,
the derivative is exactly zero and numerical integration returns the measured
initial state (persistence). This is tested at boundary values as well as in
the interior. Exact persistence cannot be disabled by configuration. Required
chromatin, signaling, TF and RNA production/decay rates may vary by context
only through declared provenance-backed kinetic modulators; population defaults
are shrinkage anchors rather than evidence that every specimen has one rate.

Only a validated `TwinContext` may be supplied to the public derivative and
rollout interfaces. A caller-constructed realized-context object cannot bypass
provenance, leakage, unknown-value or spatial-route validation.

The reference integrator is RK4 with finite positive step size. Every state
component preserves batch and feature dimensions and must remain finite for
the validated numerical fixture. Passing the RK4 software check is not
evidence that a biological time scale has been learned.

## Axolotl virtual-tissue source roles

The source registry separates evidence roles before any data access:

- regulatory-atlas training sources;
- regeneration development sources;
- spatial or temporal reference sources permitted for transfer;
- validation-only sources; and
- sealed external test sources.

Each entry declares accession, species, genome build or assembly status,
tissue, modalities, physical time coverage, spatial availability, pairing
status, source URLs, intended role and allowed access level. Live audits may
fetch accession-level metadata only. Matrix, cell, coordinate and outcome URLs
are never fetched by the metadata audit. Study identifiers are globally unique
across development, reference-atlas and sealed roles. Redirect destinations
are revalidated before following and again before reading response bodies: an
allowlisted host does not make a supplementary matrix,
archive, image or other measurement-value path safe.

The development registry retains the deposited 41,376-cell, eight-stage
PRJNA589484 upper-arm regeneration schedule exactly: homeostatic 0 hours,
then 3 hours and 1, 3, 7, 14, 22 and 33 days post-amputation, corresponding to
homeostatic, trauma, wound-healing, early-bud, mid-bud, late-bud, palette and
re-differentiated stages. It cannot be silently collapsed into a four-stage
0/3/7/21-dpa design.

`GSE315993` is sealed for external spatial testing. Its accession-level
metadata may be checked, but its values, supplementary matrices, cell/spot
records and derived outcomes may not be downloaded, opened, hashed, cached or
used in feature construction during development. A source cannot appear in
both development and sealed roles, and a sealed role cannot be weakened by a
restart or command-line flag.

Cross-species transfer requires an explicit orthology map and retains species
and assembly provenance. Human or mouse support may propose candidate axolotl
edges; it cannot be presented as axolotl observation or validation. Ambiguous
orthologs stay ambiguous or are excluded under a prespecified rule.

## Falsification and evaluation ladder

1. **Software contract:** context-dependent active edges, provenance and
   uncertainty enforcement, leakage rejection, pairing rules, graph-only
   information flow, exact persistence ablation and numerical integration.
2. **Metadata-only source audit:** validate registry roles and accessibility
   without downloading assay values; keep `GSE315993` sealed.
3. **Atlas pretraining:** learn shared regulatory grammar with entire studies
   and donors held out; compare against sequence-only, accessibility-only and
   label-free baselines.
4. **Regeneration development:** assimilate allowed axolotl RNA, chromatin,
   physical time, space and cues; compare with persistence, nearest physical
   time, generic wound response and graph shuffles.
5. **External spatial assessment:** freeze the plan before opening the sealed
   study and score held-out regions, time points and animals.
6. **Prospective twin assessment:** calibrate to one animal using predeclared
   observations, predict a later measurement or intervention, and document
   update behavior and uncertainty.
7. **Attractor assessment:** use release/recovery or sufficiently dense
   longitudinal perturbation data to test fixed points, Jacobian stability,
   basin return and convergence.

Passing one rung does not imply the next. The initial v6.0 run reports only a
software and metadata contract. It must state, in machine-readable output:

```text
model_trained = false
assay_values_downloaded = false
sealed_values_fetched = false
sealed_study_evaluated = false
biological_prediction_claim = false
digital_twin_claim = false
attractor_claim = false
```

## Initial Colab output

The restart-safe launcher downloads an immutable, hash-verified implementation,
runs the synthetic contract, performs only the permitted live metadata audit
and writes a durable report under Google Drive. It must fail closed if source
hashes, registry schema, claims, access roles or the `GSE315993` seal differ
from this contract. No checkpoint or biological model metric is produced.
