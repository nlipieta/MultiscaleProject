# Thesis-faithful architecture — staged signal→attractor pipeline

Supersedes the interim "transition-resistance" framing (`resistance_architecture.md`). That doc
introduced a global per-node state-retention term (`r·h`) to fake persistence because the KG has
no autoregulatory topology. This design removes that and models the information flow the thesis
actually writes.

## Task framing
We predict **how a signal modulates a cell's attractor landscape**, under the ground truth that
**multiple attractor states coexist within a cell**. The single per-cell program label in the data
is only the *currently-dominant* basin, not a claim of mutual exclusivity. The object of interest
is the *shift* a cue induces across a landscape of simultaneously-held attractors — which is what
the temporal-emergence and perturbation analyses measure directly.

## Information flow (the pipeline)
```
signal → signaling processing → integration + circuit regulation → epigenetic landscape → attractor
   ▲                                                                        ▲
   └─ extrinsic cue (transient, shallow)                                    └─ plastic signals reshape
      + intrinsic expression (persistent, deep)                               the landscape (open/poise)
```
Realized over the KG's existing node-type layers:

| stage | node types | role |
|---|---|---|
| signal | `cue`, `modality` | extrinsic inputs (bioelectric, mechanical, morphogen, inflammatory) |
| signaling processing | `signaling` | transduction (Piezo1, CaMKII, PKD, IKKβ, mTORC1…) |
| integration + circuit regulation | `tf`, `integration` | GRNs/CRCs, feedback, antagonism, switch-like circuits |
| epigenetic landscape | `modifier`, `mark` (+ **latent poised nodes**) | chromatin state — **defines the attractor**, stores memory |
| attractor state | `program` | co-existing basins; non-exclusive readout |
| plastic signals | `plasticity` | reshape the epigenetic landscape toward poised/pluripotent |

### 1. Linear feed-forward flow, no reinjection (DONE)
Expression is injected ONCE as the source and flows forward along edges (message passing = the
signaling cascade). No per-step re-injection of the raw signal. `alpha_memory`/`mem_inj` removed.

### 2. Memory localized to the epigenetic layer (replaces global resistance)
Persistence across steps is a learnable **per-node autoregulation gate** `g∈[0,1]`, but **eligible
only for epigenetic (`modifier`/`mark`) and circuit (`tf`/`integration`) nodes** — signal/signaling/
cue nodes are forced transient (`g≈0`, they flow through). This makes "deep (persistent) vs shallow
(transient) processing" structural, and localizes memory where the thesis stores it (poised marks,
CRC feedback). The global `r·h` retention term is removed.

    h_next = φ( Σ_r A_r · Wᵣ·h  +  cue_t )  +  g ⊙ (1 − plast_effect) ⊙ h    # g masked by node type

### 3. Plasticity integrated into the pipeline
Plastic signals act on the epigenetic layer: they lower the autoregulation gate on chromatin/latent
nodes (destabilize the current attractor) AND open access to poised/latent nodes (below). Not a
scalar multiplier on everything — a signal that reshapes the landscape toward pluripotency.

### 4. Attractor readout FROM the epigenetic landscape
Program activations are read from the **chromatin/mark layer state** (pooled epigenetic context +
each program node's own state), NOT directly off TFs — encoding the thesis claim that "attractor
states are operationally defined by their chromatin/epigenetic landscape."

### 5. Multiple co-existing attractors (non-WTA)
Program outputs are **independent per-program activations (sigmoid), trained with per-program binary
cross-entropy** against the annotated program as the positive. This lets several attractors be active
at once and never forces mutual exclusivity, while requiring NO labels the data doesn't have
(single-positive BCE). prog-AUPRC, grouped-CV, structure-isolation controls all carry over unchanged.
[decision (a); (b) full multi-label supervision is deferred — data has no multi-attractor ground truth.]

### 6. Latent poised nodes — room for undiscovered biology (NEW, first-class)
`K` latent nodes in the epigenetic layer with **learnable connectivity** to/from TF and program
nodes, representing the thesis's "latent cellular programs" / poised (H3K4me1) enhancers — capacity
for programs/signaling not yet in the literature graph (e.g. peptidergic). They:
- are **plasticity-gated**: only accessible as plasticity opens the landscape (high plasticity →
  latent nodes participate; low → they stay poised/silent), matching plasticity→pluripotency.
- are **ablatable** (`K=0` → current behavior).

Honest caveat carried in the design: latent capacity can only surface biology that leaves a
**fingerprint in the training data**. Transcriptome-only input cannot learn a signaling layer the
measurement never captured; this is a DATA limit (addressed by Multiome/scATAC widening what the
model sees), not an architecture limit.

## Controls (does each new piece capture real structure or just add capacity?)
- **latent-node control**: run with latent nodes (i) ON, (ii) OFF (`K=0`), (iii) **scrambled** —
  learned latent connectivity frozen at random. If scrambled ≈ ON, the latent nodes are adding
  capacity, not structure; report all three. (Mirrors the `scramble_edges` structure control.)
- **epigenetic-readout control**: read attractor from epigenetic layer vs directly from TFs — does
  routing through chromatin help or is it decorative?
- **structure-isolation** (`no_edges`) and grouped-CV / paired-Wilcoxon leakage controls preserved.
- **autoregulation localization check**: report learned `g` by node type — persistence should
  concentrate in epigenetic/CRC nodes if the thesis mapping holds.

## Blast radius (implementation)
- `resistance.py`: rewrite forward — staged flow, node-type-masked autoregulation gate, epigenetic
  readout, sigmoid multi-attractor head, latent nodes + plasticity gate. Rename → thesis model.
- loss: cross-entropy → per-program BCE (single-positive) in `dynamics.py::train`.
- `baselines.py`/`temporal.py`/`ablate.py`/`eval_lopo.py`: new flags (`--latent-nodes K`,
  `--latent scramble`, `--epigenetic-readout on/off`); ablation table gains latent + readout knobs.
- metrics: prog-AUPRC / recall unchanged (already per-program); argmax still reported as top-1.
- manuscript: methods §2.1 rewritten to the staged pipeline; results need a re-run (all converged
  numbers were the resistance model).
```
Nothing is trained on synthetic/oracle labels; single-positive BCE uses only the real annotated program.
```
