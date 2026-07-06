# Resistance-gated / competence-gated KG-GNN — architecture & evaluation plan

**Reframe:** intrinsic memory is *not* a re-injected default signal; it defines **transition
resistance** — the barrier a cue must overcome to leave the current attractor. Plasticity
**lowers** that barrier; it does not amplify the cue. The model asks: *given this cell's
current attractor and resistance landscape, which regulatory subgraph is accessible, and is
the cue strong enough to move the cell into another basin?*

## Core update rule (replaces `h = GNN(h + memory + cue)`)
```
candidate      = KG_GNN(h, cue_t, graph)                     # where the cue could push
base_resist    = sigmoid(W_resist([chromatin, lineage, program_state]))   # inertia of current basin
plast_effect   = sigmoid(W_plast([plasticity/stress nodes, plasticity_input]))
resistance     = base_resist * (1 - plast_effect)            # plasticity LOWERS the barrier
h_next         = resistance * h + (1 - resistance) * candidate + alpha_memory * mem_inj
```
`alpha_memory` is learnable/configurable (default small), **not** hard-coded to 1. Readout uses a
**soft/delayed** attractor (graded), not hard WTA.

## Eight components (main architecture = 1–6; 7–8 optional/diagnostic)
1. **Learnable memory reinjection** `alpha_memory ∈ {zero, low, learned, full}` — memory is no
   longer forcibly re-added each round.
2. **Resistance/inertia module** — `resistance∈[0,1]` from lineage TFs (memory_nodes), chromatin
   nodes (modifier+mark), current program state. High = stay; low = movable.
3. **Plasticity = barrier-lowering** — `resistance *= (1 - plast_effect)`; also opens edge
   accessibility; lowers attractor sharpness during high plasticity.
4. **Context-gated subgraph** — `active_edge = base_edge * accessibility_gate(cell_state)`; the
   same cue is interpreted by competence (epithelial+TGFβ→EMT subgraph, etc.).
5. **Soft/delayed attractor** — `logits += attractor_strength(commitment) * feedback(logits)`;
   strength ramps with commitment. Preserves graded probabilities for temporal analysis.
6. **Signed/de-repression** — separate activation vs inhibition messages; a de-repression motif
   `inhibit(inhibitor)→activate(target)` (fixes HDAC4/5→MEF2→hypertrophy).
7. **Circuit-competition diagnostics (analysis layer, not a hard assumption)** — regulon
   competition, residual-origin vs target activity, context subgraph accessibility, fate-
   probability splitting, attractor-resistance, perturbation-defined shifts. Direct antagonism
   is TOGGLEABLE, only where literature supports it (PU.1/GATA1-like); not assumed.
8. **Leakage controls preserved** — grouped CV (whole datasets out), uniform cue per dataset,
   marker-in & marker-masked, edge-removed, paired folds/seeds, paired Wilcoxon.

## Forward pass (pseudocode, minimal-first = steps without context-gate/signed)
```
h = embed(nodes) + in_proj(x0)                       # init
mem_inj = x0 on intrinsic nodes;  cue_inj = x0 on cue nodes
for t in range(steps):
    cue_t = cue_inj * cue_decay**t                   # transient
    if plasticity_mode in (amplify, both): cue_t *= plasticity_input
    msg   = self_lin(h) + RGCN(h, adjacency[, signed][, context_gate])
    cand  = GRU(msg + cue_t, h)                       # candidate basin
    chrom = mean(h[chromatin_nodes]); lin = mean(h[lineage_nodes]); prog = mean(h[program_nodes])
    base_r = sigmoid(W_resist([chrom, lin, prog]))
    plast  = sigmoid(W_plast([mean(h[plasticity_nodes]), plasticity_input]))
    resist = base_r * (1 - plast) if plasticity_mode in (lower_resistance, both) else base_r
    if not use_resistance: resist = 0                 # ablation -> pure candidate (no inertia)
    h = resist * h + (1 - resist) * cand + sigmoid(alpha_memory) * mem_inj
logits = readout(h) [+ hybrid_skip(x0)]
logits = soft_attractor(logits, strength_schedule, commitment)   # graded, not 0/1
```

## Config flags (all ablation-ready; leakage controls unchanged)
```
--arch            toggle | resistance                     # model family
--alpha-memory    zero | low | learned | full             # (1)
--resistance      on | off                                # (2)
--plasticity-mode amplify | lower_resistance | both | none # (3)
--context-gate    off | on | scrambled                    # (4)
--attractor       none | hard_wta | soft | delayed_soft | learned   # (5)
--signed          off | on | on_derepression              # (6)
--competition     off | regulon,residual,fate,resistance,perturb    # (7) diagnostics
# preserved: --group-split --mask {none,no_markers} --structure-test --seeds --save-folds
```

## Evaluation plan (variants A–F)
| id | variant | flags |
|----|---------|-------|
| A | current baseline | `--arch toggle --attractor hard_wta` |
| B | reduced reinjection | `--arch resistance --alpha-memory low/learned --resistance off` |
| C | resistance-gated | `--arch resistance --resistance on --plasticity-mode lower_resistance --attractor soft` |
| D | context-gated graph | C `+ --context-gate on` |
| E | signed/de-repression | C `+ --signed on_derepression` |
| F | full | `--arch resistance --resistance on --plasticity-mode both --context-gate on --attractor delayed_soft --signed on_derepression` |

Each variant also run with `--structure-test` (adds `_noedges` twin) and `--mask none/no_markers`.

**Metrics:** grouped macro-AUPRC · balanced acc · program recall · macro-F1 · EMT temporal
Spearman · perturbation directionality (HDAC4/5 + CaMKII signs) · calibration (ECE, if easy).

**Key tests (each a paired Wilcoxon vs the relevant control):**
1. resistance-gating vs current on macro-AUPRC
2. resistance-gating on top-1 without sacrificing AUPRC
3. context-gate vs no-edges (does gating widen the structure benefit?)
4. soft attractor vs WTA on EMT temporal Spearman (graded preserved?)
5. signed/de-repression fixes HDAC4/5 knockdown sign
6. plasticity-as-barrier-lowering vs plasticity-as-cue-amplification (temporal + AUPRC)
7. competition diagnostics interpretable without hard antagonism

## Expected result table template
```
variant            AUPRC↑   bal-acc↑  prog-rec↑  F1↑   EMT-ρ↑  perturb-sign↑  vs-noedges(paired p)
A current                                                                     
B reduced-reinj                                                               
C resistance                                                                  
D +context-gate                                                               
E +signed/derep                                                               
F full                                                                        
```
Fill from `--save-folds` + paired tests. Bold the best per column; report p vs A and vs `_noedges`.

## Failure modes to watch
- **Resistance collapses to 0 or 1** (degenerate) → the update becomes pure-candidate or frozen.
  Mitigate: init `W_resist` near 0 (resist≈0.5), monitor mean resistance per epoch; add a mild
  entropy/variance regularizer if it saturates.
- **alpha_memory learns to full** → we're back to the old model; report the learned value.
- **Soft attractor too weak** → no commitment, low top-1; too strong → temporal flattening returns.
  Sweep strength; the delayed schedule should decouple these.
- **Context-gate as covert dataset detector** → could re-introduce leakage. Guard: gate only from
  *node-state* (lineage/chromatin), never dataset/tissue IDs; verify `scrambled` gate ablation
  destroys the benefit (if scrambled ties real, the gate is memorizing, not routing).
- **Signed/de-repression instability** → inhibitory messages can blow up; clamp/normalize, and
  verify on the 2-hop HDAC4/5 case before trusting.
- **More knobs → overfitting on a small pool** → keep grouped CV + paired tests; a variant only
  "wins" if it beats controls on held-out datasets, not in-distribution.

## Minimal-first (implement now, compute-light)
`resistance.py :: ResistanceToggle` with flags `--alpha-memory`, `--resistance`,
`--plasticity-mode {amplify,lower_resistance,both}`, `--attractor {soft,delayed_soft,hard_wta}`;
keep `--structure-test`, `--mask`, `--group-split`, `--save-folds`. Defer context-gate (4),
signed/de-repression (6), competition diagnostics (7) to phase 2.
```
Run order: C vs A (does resistance help AUPRC + top-1), then the plasticity-mode sweep (test 6),
then soft vs WTA on EMT temporal (test 4). Those three answer the central hypothesis cheaply.
```
