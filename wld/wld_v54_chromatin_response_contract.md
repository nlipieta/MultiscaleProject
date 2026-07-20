# WLD v5.4 graph-routed chromatin response contract

## v5.4.1 response calibration

The initial v5.4 development run exposed a numerical degeneracy: supported
response amplitudes were initialized effectively at zero, so absolute-state
distribution loss selected persistence. It also evaluated frozen route removal
on a different random validation draw, confounding that effect.

v5.4.1 uses nondegenerate but bounded supported-path initialization, adds an
unpaired training-target pseudobulk response objective, and selects checkpoints
with a prespecified combination of relative SWD and response NRMSE. Frozen and
unmodified evaluations use identical cells and SWD projections. The topology,
whole-target split, leakage boundary, sealed tests, and claim scope are
unchanged.

## Scientific scope

GSE161002 is a single-endpoint CRISPR-sciATAC screen. It can supervise a
perturbational accessibility response, but it cannot identify a continuous-time
kinetic scale, a fixed point, a basin of attraction, or return after release.
WLD v5.4 therefore makes no attractor claim.

## Permitted information flow

1. A control cell's measured ATAC profile is encoded by the broadly pretrained
   WLD foundation representation.
2. Guide/target identity is excluded from that encoder and selects one named
   regulator node only after encoding.
3. The intervention may reach a modeled TF only through a frozen direct or
   two-hop interaction route compiled from the durable OmniPath source.
4. A TF may reach a peak only where the frozen JASPAR motif evidence supports
   that TF-to-peak edge.
5. CRISPR-sciATAC training may learn opening/closing direction and strength on
   those supported paths because perturbational chromatin evidence, unlike a
   motif alone, is evidence about response direction.
6. Neural context may modulate supported gains and opening, closing and recovery
   rates. It has no direct decoder to peaks and cannot create an unsupported
   edge.

Candidate regulator membership is not edge evidence. Targets without an
interaction route are reported as unsupported, not given a dense guide decoder.

## Variation versus frozen evidence

Frozen: named-node identity, genome build, feature split, interaction topology,
motif topology, evidence provenance, and whole-target split.

Context conditioned and trainable: cell state, supported-edge gain,
opening/closing direction learned from training perturbations, accessibility
opening and closing rates, and recovery rate. Tissue, subject, condition, study,
cell type, target label and integrated cluster identity are not encoder inputs.

## Training and evaluation

- Control and perturbed cells are unpaired population observations; no cell
  matching or pseudotime pairing is fabricated.
- Feature selection was completed using v5.3 training cells only.
- Train, validation and test are split by complete perturbation target.
- Validation chooses checkpoints using entire unseen targets.
- Persistence, frozen-zero routes, frozen degree-preserving route shuffles and
  independently retrained degree-preserving route shuffles are all reported.
- The 16 test targets, muscle subjects J/L and external sealed studies remain
  unopened by development.

A route-specific transient-response statement requires all of the following:
the true model beats persistence, beats retrained degree-preserving route
controls, and worsens when its fitted routes are removed without retraining.
Even then, the result is not evidence of attractor dynamics.
