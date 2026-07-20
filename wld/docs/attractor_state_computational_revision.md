# Computational Framework Revision

## Prior-Constrained Inference of Latent Cell State and Attractor Dynamics

### Recommended insertion point

Insert this section after **Context-Dependent Cellular Response** and before **Tissue-level Convergence Towards a Stable Physiological State**. It converts the manuscript's conceptual attractor framework into a falsifiable computational proposal.

### Replacement manuscript section

#### From epigenetic landscape to an identifiable latent state

An attractor-state model should not treat a low-dimensional embedding as a biological state merely because it separates annotated cell types. Instead, the latent state should be constrained by three experimentally interpretable layers. First, chromatin accessibility defines the regions of the regulatory genome that are available for interaction. Second, sequence motifs, occupancy assays, and protein-DNA interaction evidence constrain which transcription factors can bind within those accessible regions and which genes those regions may regulate. Third, perturbationally or experimentally validated circuit interactions constrain how transcription factors activate, repress, or reinforce one another. Together, these layers define a hybrid scaffold in which the learned state is not an arbitrary summary of expression, but a compact representation of regulatory activity that is consistent with what is open, what can bind, and which interactions are supported.

Let \(a_i\) denote the accessibility vector for cell \(i\), \(P\) a peak-to-gene linkage matrix, \(M\) a transcription-factor-by-gene motif or occupancy matrix, and \(C\) a directed transcription-factor circuit. Gene-linked accessibility is

\[
g_i = a_i P,
\]

and the binding-feasibility tensor is

\[
B_i = M \odot g_i,
\]

where gene accessibility is broadcast over transcription factors. \(B_i\) therefore represents regulatory interactions that are simultaneously supported by a binding prior and accessible in that cell. The initial latent state is inferred as

\[
z_i(0)=E_\phi(g_i, B_i, u_i),
\]

where \(u_i\) contains measured upstream cues, such as perturbation, ligand concentration, matrix stiffness, or bioelectric condition. The encoder is intentionally restricted from receiving RNA expression, cell-type annotations, cluster identities, pseudotime, terminal-state labels, or other direct proxies for the state it is expected to infer.

#### Hybrid mechanistic and neural dynamics

The latent state is evolved using a continuous vector field containing a validated circuit term and a bounded neural residual:

\[
\frac{dz}{dt} =
\tanh\left[z(W_C \odot C)\right]
-\lambda\odot z
+\alpha f_\theta(z,u).
\]

The masked circuit term represents known interactions, the positive decay term prevents unconstrained growth, and the neural residual represents missing or context-dependent regulation. Restricting the residual's scale makes it possible to test whether predictive performance depends on established regulatory architecture or on unconstrained capacity. A decoder maps the evolving transcription-factor-aligned state into gene expression through the accessibility-conditioned binding scaffold:

\[
\hat{x}(t)=\operatorname{softplus}\left(
\sum_k z_k(t)[W_D\odot B_i]_{k,:}+b
\right).
\]

This structure makes three distinct claims testable: accessible chromatin limits regulatory opportunity; binding evidence limits transcription-factor-to-target compatibility; and the circuit graph limits the mechanistic component of latent dynamics.

#### Operational definition of an attractor

A predicted state is not an attractor simply because cells cluster near it. A candidate fixed point \(z^*\) must satisfy

\[
\left\|f(z^*,u)\right\| \approx 0.
\]

Local stability is assessed using the Jacobian \(J=\partial f/\partial z\vert_{z^*}\). The state is locally asymptotically stable only if every eigenvalue of \(J\) has a negative real component. Attractor strength may then be summarized by convergence rate, basin size under biologically plausible perturbations, and the amount of input required to cross into an alternative basin. These quantities connect the manuscript's concepts of robustness and plasticity to measurable dynamical properties rather than visual features of an embedding.

#### What the data can identify

Paired RNA and ATAC measurements from a single time point can test whether accessibility and regulatory priors recover contemporaneous transcriptional state. They cannot, by themselves, identify a temporal vector field or demonstrate convergence to an attractor. Dynamical claims require time-resolved multiome data, lineage tracing, metabolic labeling, controlled perturbations, or another design that constrains direction and transition. Pseudotime and RNA-nearest-neighbor matching may support exploratory hypotheses but should not be treated as observed cell transitions.

The transcriptome is used as supervision or an external validation target, not as an input to the latent-state encoder. This distinction is essential: if present or future RNA, RNA-derived pseudotime, cell labels, or outcome-derived matching enters the encoder or pairing procedure, the model can recover cell identity without learning the proposed epigenetic mechanism.

#### Leakage-resistant validation

Donors, experiments, or perturbation replicates must be assigned to training, validation, and test partitions before feature selection, normalization parameter fitting, dimensionality reduction, graph construction, transport, smoothing, or hyperparameter selection. Transformations learned from data are fitted on the training partition and applied unchanged to held-out groups. No test outcome may determine a cell pairing, terminal-state label, trajectory, feature set, or prior edge.

Evaluation should distinguish state reconstruction from dynamics. State reconstruction is assessed with held-out per-cell and per-gene correlations, calibrated error, and variance explained relative to training-mean, ridge, and unconstrained neural baselines. Dynamics are assessed only when transitions are experimentally supported, using future-state error, displacement cosine similarity, perturbation-response prediction, endpoint fixed-point residual, and stability. An area under the precision-recall curve must name its prediction target and report positive-class prevalence; otherwise it is not interpretable. Global flattened Pearson correlation should be supplementary because it can be dominated by gene-level mean differences.

#### Required ablations

The mechanistic interpretation depends on prospective ablations:

1. Remove the circuit mask while matching parameter count.
2. Randomly degree-preserve and shuffle the circuit graph.
3. Remove motif/occupancy constraints.
4. Remove accessibility gating.
5. Remove the neural residual.
6. Vary the residual scale to measure reliance on the prior scaffold.
7. Compare against no-change, training-mean, ridge, and unconstrained multilayer-perceptron baselines.

If the biologically constrained model generalizes across held-out donors or experiments and its inferred fixed points are stable under perturbation, then the results support the narrower claim that latent regulatory dynamics are partly identifiable from accessibility-conditioned binding and validated circuit architecture. They do not by themselves establish a universal Waddington landscape.

## Replacement conclusion paragraphs

The framework proposed here yields a concrete computational hypothesis: latent cellular state can be represented as transcription-factor-aligned activity constrained by accessible chromatin, feasible binding, and validated regulatory circuits, while context-dependent effects are captured by a bounded learned residual. Attractor states are defined dynamically as stable fixed points of this hybrid vector field, rather than as clusters in a visualization. Robustness corresponds to local stability and basin geometry, whereas plasticity corresponds to the ease with which physiological input moves the system between basins.

Testing this hypothesis requires data that separate state inference from temporal dynamics. Snapshot multiome measurements can evaluate whether chromatin state anticipates transcriptional programs, but time-resolved or perturbational measurements are required to infer direction, convergence, and causal transitions. Leakage-resistant donor- or experiment-level holdouts, together with circuit, motif, accessibility, and neural-residual ablations, are therefore central to evaluating whether the model has learned transferable regulatory structure rather than direct proxies for cell identity.

## How this changes the previous WLD prototype

| Previous component | Problem | Revision |
|---|---|---|
| RNA and ATAC concatenated in encoder | RNA is a direct state proxy | Encoder uses ATAC-derived accessibility and measured upstream cues only |
| Random peak-to-gene projection | No biological meaning and not learned | Supply genomic peak-to-gene links and motif/occupancy support |
| Sequence-length-one attention | Cannot perform meaningful attention | Removed; TF-aligned latent variables are modeled directly |
| One-step delta prediction | Does not define a dynamical system or attractor | Integrate an explicit ODE and retain the full latent path |
| Motif-absent edges initialized to 0.25 | Destroys prior sparsity | Unsupported TF-gene edges remain exactly zero |
| PCA smoothing fitted before splitting | Leaks test outcome structure | Split by donor/experiment first; fit all transforms on training only |
| RNA-derived pseudotime and Hungarian pairing | Outcome determines transition targets | Use experimentally supported time/lineage/perturbation transitions |
| Flattened Pearson as primary metric | Can reward gene means rather than cell-specific prediction | Report per-cell, per-gene, transition, calibration, and stability metrics |
| “Attractor” inferred from endpoint prediction | No fixed-point or stability test | Require low vector-field norm and negative-real-part Jacobian eigenvalues |

## Minimum experimental design

| Data structure | Evidence and evaluation |
|---|---|
| At least two donors or independent experiments, with an entire group held out | Time, lineage, metabolic-labeling, or perturbation information for any dynamical claim |
| Paired or appropriately integrated chromatin and transcriptomic measurements | Predeclared endpoint and AUPRC labels, with prevalence reported |
| Peak-to-gene links compiled without test outcomes | Baselines and ablations trained under identical partitions and preprocessing |
| TF motif or occupancy evidence at linked regulatory elements | A directed, signed regulatory circuit with evidence provenance |

## Scope of the supplied implementation

`wld_attractor_model_v2.py` supplies the architecture, RK4 integration, fixed-point stability diagnostic, leakage-safe group splitter, and an explicit leakage audit. It intentionally does not manufacture trajectories from the 10x PBMC snapshot. Dataset-specific ingestion must provide the three prior matrices and a donor- or experiment-level split.

### Required tensors

- `atac`: cells by peaks, normalized without fitting on held-out groups.
- `peak_to_gene`: peaks by genes, derived from genomic linkage evidence.
- `motif_tf_gene`: transcription factors by genes, indicating motif or occupancy support at linked regulatory elements.
- `circuit_tf_tf`: directed transcription-factor circuit with edge provenance.
- `cues`: optional measured perturbation or physiological inputs; never inferred state labels.
- `rna_target`: RNA used only as supervision or evaluation, not as an encoder input.

### Training and audit sequence

1. Hold out donors, experiments, or perturbation replicates before any learned preprocessing.
2. Compile prior matrices without consulting held-out outcomes.
3. Train state reconstruction first; enable trajectory and terminal-state losses only when the experimental design supports them.
4. Compare all baselines and ablations under the identical split and preprocessing objects.
5. Locate candidate fixed points, calculate Jacobian eigenvalues, and test recovery after biologically plausible perturbations.
6. Report failures and unstable equilibria rather than relabeling every endpoint as an attractor.
