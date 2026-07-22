# WLD v6.1 real axolotl measurement-corpus contract

## Purpose and claim boundary

WLD v6.1 is the first value-level data layer for the atlas-conditioned
axolotl virtual-tissue program. It downloads and verifies **unsealed public
measurements**, preserves their deposited structure, assigns biological groups
before any learned transformation, and compiles fold-specific registries for a
later model. It does not train WLD, estimate dynamics, evaluate a held-out
study, or establish a biological prediction.

The build has two deliberately separate stages:

```text
immutable raw ingestion
  -> source locks, deposited sparse measurements, observation metadata,
     pairing evidence and measurement provenance

split-aware fold compilation
  -> biological-group partitions, train-fit feature registries,
     harmonization records and model-consumption manifests
```

Raw ingestion is evidence preservation, not data fitting. Fold compilation is
the first stage permitted to learn a vocabulary or transformation, and it is
rerun independently for every development fold. A corpus bundle is never
converted directly into a `TwinContext`; the v6.0 causal, temporal and
provenance validators remain the boundary for later model consumption.

## Development measurements and roles

Every source has one immutable role in the v6.1 measurement registry. Roles do
not become weaker when the runner resumes or receives a command-line option.

- `GSE121737` supplies unpaired-population single-cell RNA measurements across
  intact, contralateral, wound-healing, early-bud and medium-bud limb samples.
  Deposited sample prefixes—not cells or inferred clusters—are the biological
  split units. RNA is supervision or representation data, never an initial
  encoder shortcut to a future RNA outcome.
- `GSE182296` supplies measured Am580-versus-DMSO RNA perturbation responses for
  retinoic-acid cue development. Treatment, dose and harvest time are retained
  as observed experimental context outside the encoder identity channel.
- `GSE184948` supplies Tig1, mutant and control RNA perturbation responses at
  deposited physical times. Perturbation groups and biological replicates are
  preserved rather than pooled into synthetic trajectories.
- `GSE272731` supplies talarozole/DMSO and proximal/distal RNA response data as
  an unsealed whole-study development check. It is not used to select features,
  fit normalization or tune a model when assigned to validation.
- `GSE243225` supplies one deposited 5-dpa Visium specimen as a spatial
  reference. Spot expression and coordinates are exact relations only after
  the deposited barcode crosswalk passes schema checks. One section is not
  split into train and validation spots and is not treated as biological
  replication.
- `GSE217591`, `GSE217592` and `GSE217593` are registered
  chromatin/transcriptional atlas stages. Their CUT&Tag, ATAC and RNA values are
  acquired only by an explicit high-volume stage with the same raw-ingestion
  and group-first rules. Bulk accessibility remains a bulk observation and is
  never relabeled cell-specific accessibility.
- `PRJNA682840`, `PRJNA589484` and `GSE106269` remain candidate atlas,
  temporal or lineage sources and are not part of this frozen registry.
  Admission requires a reviewed registry update with exact deposited files and
  biological split units. Destructive samples would remain unpaired populations;
  physical time, anatomy and reporter evidence do not create longitudinal cells.
- `GSE315993` is the sealed external spatial study. Accession metadata may be
  named, but measurement matrices, spot records, coordinates, images and
  derived values are inaccessible throughout v6.1.

Additional public studies may be admitted only by a reviewed registry change
that declares their experimental unit, biological replicate, pool structure,
species, assembly, assay, deposited files, context semantics, pairing status,
quality limitations and role. Discovery at runtime cannot silently expand the
corpus.

## Biological groups precede features

Studies, specimens, animals, deposited biological replicates and applicable
perturbation targets are partitioned before any feature statistic is computed.
Cells, nuclei and spots from one biological specimen always remain in one
partition. A pooled sample is one experimental unit unless a depositor-backed
crosswalk proves separable biological units. Technical libraries from the same
unit stay together.

Partitions are deterministic, namespaced by accession and written to a locked
split manifest. Validation values may be parsed and stored, but they cannot
affect feature membership, rank, normalization, vocabulary order, assembly
mapping, hyperparameters or stopping decisions. Changing only validation
values must leave the train-fit feature-registry content hash unchanged.

Feature selection is fit from training groups only, using declared
measurement-independent eligibility rules and train-only prevalence or count
statistics with deterministic lexical tie-breaking. Labels such as cell type,
cluster, inferred state, pseudotime, condition outcome, response class and
integrated embedding are forbidden selectors. Each feature registry records
its fit-group roster, split-manifest hash, source-assembly namespace, rule and
content hash.

## Measurement preservation and harmonization

Deposited count matrices remain sparse, non-negative and unnormalized in the
raw bundles. The corpus layer does not log-normalize, smooth, denoise, impute,
integrate or PCA-transform them. Every modality keeps its own feature table and
observation identifiers. RNA is marked `supervision_only` unless a later
causal contract explicitly permits a time-zero RNA measurement.

Assembly and feature namespaces are explicit. Am_2.2 transcript identifiers,
AmexG_v6/AmbMex60DD annotations, `GCA_002915635.2`,
`GCA_002915635.3`, and any other deposited build are not merged by symbol
coincidence. A mapping may be added only with a versioned crosswalk, mapping
method and one-to-many policy. Ambiguous mappings are quarantined or retained
as ambiguous; they are never resolved by guessing. Cross-species transfer
requires an explicit orthology record and remains `reference_transferred`, not
an axolotl observation.

Physical regeneration time, anatomical position, perturbation, dose and
spatial coordinate are separate context fields. Pseudotime and low-dimensional
embeddings cannot replace physical time or position. Deposited cues are
observed context only for the specimen and time actually assayed. Reasonable
spatial, cue, metabolic or regulatory extrapolations may be compiled later as
inferences, but never written into the raw measurement bundle.

## Pairing and temporal semantics

Pairing has structured evidence and one of the following meanings:

- `exact`: deposited shared identifiers or a deposited crosswalk establish the
  same cell, nucleus or spot after schema and cardinality checks;
- `unpaired_population`: modalities or time points describe biological
  populations without observation-level correspondence; or
- `single_modality_not_applicable`: no cross-modal relation is asserted.

Row order, expression similarity, nearest neighbors, optimal transport,
labels, clusters and pseudotime never establish exact pairing. An expected
Visium spot relation remains unverified until the deposited expression and
coordinate barcode sets are checked. Destructive regeneration time points are
never relabeled longitudinal cell trajectories. Population-only relations may
support distributional objectives later, but cannot enter exact-pair losses.

## Provenance, missingness and uncertainty

Every stored measurement or contextual value records source accession, source
file, observation group, modality, measurement time, assembly when applicable,
and one provenance state:

- `observed`: directly deposited for this experimental unit;
- `reference_transferred`: transferred from a named atlas or mapping;
- `model_inferred`: generated by a declared inference procedure; or
- `unknown`: unavailable and masked.

Transferred and inferred values require method lineage plus a finite,
non-negative uncertainty representation. They may not be promoted to
`observed`. A measured zero is known and distinct from unknown; missing values
are masked rather than zero-filled. Spatial position, cue exposure, protein or
metabolic state is observed only when deposited for that unit. An inferred
coordinate, cue, contact, cell-cell signal or metabolic value retains its
inferred provenance and uncertainty through every downstream manifest.

## Leakage and future-information prohibition

Bare study, specimen, donor, barcode, condition, perturbation target, guide,
cell type, lineage, tissue, cluster, state, pseudotime and integrated-embedding
identifiers remain available for splitting and audit only. They are not encoder
features, learned identity embeddings or lookup-table keys. Renaming a state
label as a cue or atlas context does not make it causal input.

Initial/context inputs may contain only measurements taken at or before the
declared prediction origin. A future RNA, accessibility, morphology, spatial,
protein or outcome value cannot enter the initial state, context, feature
registry or learned reference atlas. Test, holdout and validation partitions
cannot contribute statistics to a development transformation. RNA response
matrices are targets or evaluation observations, not direct proxies for the
state being predicted.

## Sealed-source access boundary

`GSE315993` is refused before URL resolution, redirect following, response-body
access, hashing, caching or file creation. The refusal applies to plain,
case-varied, percent-encoded and multiply encoded accession text; query
parameters; path components; archive members; filenames; attachment headers;
and redirect destinations. An allowlisted host does not weaken the seal.

No v6.1 registry entry contains a GSE315993 measurement URL. A download request
or redirect that resolves to a sealed accession fails before the body is read,
and the output tree must remain byte-for-byte free of sealed values. Restarting
cannot weaken the rule. The runner has no `--unseal`, `--test` or equivalent
development flag.

## Restart, integrity and immutability

Downloads use a temporary partial path and an atomic final rename. A completed
file is accepted only after its byte count, gzip/container readability where
applicable and SHA-256 digest pass. The first successful acquisition creates an
immutable source lock containing the requested URL, final resolved URL, byte
count and locally computed digest. A locally computed digest is explicitly
labeled as such and is not represented as a publisher-supplied checksum.

On resume, complete files are reverified against their locks and reused.
Partial files may resume only when the server and recorded offset support it.
Hash, length, URL or schema disagreement is a hard failure: corrupt or changed
content is never silently accepted, overwritten or blended with a prior build.
Extraction and fold compilation write temporary outputs and publish manifests
atomically only after validation. Source locks, split manifests and feature
registries are content-addressed so a rerun is reproducible and auditable.

## Required outputs

The durable corpus layout is:

```text
wld_v61_axolotl_corpus/
  registry/source_registry.lock.json
  access_ledger.json
  raw/<source>/source.lock.json
  raw/<source>/<deposited files>
  bundles/<cohort>/bundle_manifest.json
  bundles/<cohort>/observations.tsv.gz
  bundles/<cohort>/modalities/<modality>/features.tsv.gz
  bundles/<cohort>/modalities/<modality>/counts.csr.npz
  bundles/<cohort>/pairing.json
  bundles/<cohort>/provenance.json
  folds/<fold>/split_manifest.json
  folds/<fold>/feature_registry.json
  folds/<fold>/harmonization_manifest.json
  staged/chromatin_acquisition_plan.json
  wld_v61_axolotl_corpus_report.json
```

Equivalent content-addressed subpaths are allowed, but the report must link
every output to its source lock, split and registry hash. It must also state
which sources were downloaded, staged, refused or left sealed; which exact
pairings were schema verified; and which groups fit each feature registry.

## Explicit non-claims

Completing v6.1 means that real unsealed measurements were ingested under a
leakage-resistant, restart-safe contract. It does **not** mean that the studies
are sufficiently powered, that measurements from distinct specimens are
longitudinally paired, that bulk chromatin is cell specific, that a circuit is
causal, or that missing spatial, cue, protein, contact or metabolic context has
been observed.

The machine-readable report must retain:

```text
development_assay_values_downloaded = true
sealed_values_fetched = false
sealed_study_evaluated = false
model_trained = false
biological_prediction_claim = false
digital_twin_claim = false
attractor_claim = false
```

No later stage may open the sealed study merely because ingestion completed.
A model, predictive, digital-twin or attractor claim requires its own
predeclared training, calibration and falsification contract.
