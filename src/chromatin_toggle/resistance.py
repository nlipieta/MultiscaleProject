"""Resistance-gated / competence-gated KG-GNN (minimal-first).

Reframes intrinsic memory as TRANSITION RESISTANCE (the barrier a cue must overcome to
leave the current attractor) rather than a re-injected default. Plasticity LOWERS the
barrier; it does not amplify the cue. Update rule:

    candidate = KG_GNN(h, cue_t, graph)
    resistance = base_resist(lineage, chromatin, program state) * (1 - plasticity_effect)
    h_next     = resistance * h + (1 - resistance) * candidate

Intrinsic identity is a PERSISTENT, DEEPLY-PROCESSED signal: the cell's expression is injected
ONCE into the initial node state and then processed across all message-passing steps, carried
forward by the resistance gate (memory = inertia). It is NOT re-injected each round -- forcibly
re-adding the raw signal every step would short-circuit the processing depth and contradict the
thesis's intrinsic(deep)/extrinsic(shallow) asymmetry. The extrinsic cue is the opposite: a
transient, shallowly-processed signal injected with decay each step.

Readout uses a SOFT / delayed attractor (graded), not hard winner-take-all, so temporal
gradients are preserved. Forward signature: forward(x0, plasticity)
-> logits [B, n_classes], so it drops into the existing train/predict/eval harnesses.

Phase-2 hooks (context-gated subgraph, signed/de-repression, competition diagnostics) are
specified in docs/resistance_architecture.md and NOT implemented here (minimal-first).
"""
from __future__ import annotations

import torch
import torch.nn as nn

CHROMATIN_TYPES = {"modifier", "mark"}


class ResistanceToggle(nn.Module):
    def __init__(self, kg, hidden=64, steps=6, resistance=True,
                 plasticity_mode="lower_resistance", attractor="soft", hybrid=True,
                 cue_decay=0.6, attr_iters=3, attr_strength=0.15,
                 node_ann=None, context_dim=0, use_atac=False, sparse_adj=False,
                 plasticity_source="const"):
        super().__init__()
        assert plasticity_source in ("const", "atac", "intrinsic")
        # sparse_adj: aggregate via sparse mm over the (~0.04%-dense) adjacency instead of a
        # dense [N,N]x[B,N,H] einsum. Numerically identical; big compute cut on GPU where the
        # dense einsum is the bottleneck. Opt-in until benchmarked per device.
        self.sparse_adj = sparse_adj
        self._Asp = None                                  # lazily-built per-device sparse adjacency
        assert plasticity_mode in ("amplify", "lower_resistance", "both", "none")
        assert attractor in ("none", "hard_wta", "soft", "delayed_soft", "learned")
        self.N, self.steps, self.hidden = kg.num_nodes, steps, hidden
        self.use_resistance = resistance
        self.plasticity_mode = plasticity_mode
        self.attractor = attractor
        self.cue_decay = cue_decay
        self.attr_iters, self.attr_strength = attr_iters, attr_strength
        # use_atac: node input becomes [expression, chromatin-accessibility] (2 channels)
        # from paired 10x Multiome; default False keeps the RNA-only model byte-identical.
        self.use_atac = use_atac

        # --- shared GNN machinery ---
        self.id_emb = nn.Embedding(kg.num_nodes, hidden)
        self.type_emb = nn.Embedding(kg.num_types, hidden)
        self.in_proj = nn.Linear(2 if use_atac else 1, hidden)
        self.rel_lin = nn.ModuleList([nn.Linear(hidden, hidden, bias=False)
                                      for _ in range(kg.num_relations)])
        self.self_lin = nn.Linear(hidden, hidden, bias=False)
        self.gru = nn.GRUCell(hidden, hidden)
        self.prog_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.quiescent_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        # --- resistance / plasticity heads ---
        # base resistance from [chromatin, lineage, current-program] pooled states
        self.W_resist = nn.Linear(3 * hidden, 1)
        # plasticity effect from [plasticity-node state, plasticity input scalar]
        self.W_plast = nn.Linear(hidden + 1, 1)
        nn.init.zeros_(self.W_resist.weight); nn.init.zeros_(self.W_resist.bias)   # start resist~0.5
        nn.init.zeros_(self.W_plast.weight); nn.init.zeros_(self.W_plast.bias)

        # plasticity as a FUNCTION OF ATAC ACCESSIBILITY (thesis: epigenetic openness across the
        # target pathway AND adjacent/alternative networks = plasticity). Per program, pool ATAC
        # accessibility over its network genes (regulators + markers), row-normalized -> per-cell
        # program-accessibility profile [B,P]; an MLP maps that to a per-cell plasticity in [0,1]
        # (high when many programs are open = poised/multi-potent). Requires use_atac. No cue, no leak.
        self.plasticity_source = plasticity_source
        if plasticity_source == "atac":
            P, N = len(kg.program_index), kg.num_nodes
            prog_pos = {ni: r for r, ni in enumerate(kg.program_index)}
            memb = torch.zeros(P, N)
            for s, _rel, d, _w in kg.edges:                        # s (regulator/marker) -> d (program)
                if d in prog_pos:
                    memb[prog_pos[d], s] = 1.0
            memb = memb / memb.sum(1, keepdim=True).clamp(min=1.0)  # mean accessibility per program network
            self.register_buffer("prog_membership", memb)          # [P, N]
            # plasticity = f(epigenetic landscape): per-program accessibility profile (pathway +
            # adjacent networks) + GLOBAL chromatin openness + accessibility DIVERSITY across programs
            # (multi-lineage priming = the poised/plastic state). +2 = [openness, diversity].
            self.plast_from_atac = nn.Sequential(nn.Linear(P + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        if plasticity_source == "intrinsic":
            # RNA-only analog: plasticity = chromatin OPENNESS proxied from the chromatin-modifier
            # node states (varies per cell; wakes the gate without a cue or ATAC). Distinct head so
            # it isn't tied to the resistance gate's identity barrier.
            self.plast_intrinsic = nn.Linear(hidden, 1)

        self.hybrid = hybrid
        if hybrid:
            self.skip = nn.Linear(kg.num_nodes, len(kg.program_index) + 1)
        if attractor == "learned":
            self.attr_param = nn.Parameter(torch.tensor(-1.5))

        # --- optional annotation layers (ported from the initial formulation; ablatable by omission) ---
        self.ann_proj = None
        if node_ann is not None:                          # per-node gene role + pathway terms
            self.register_buffer("node_ann", torch.as_tensor(node_ann, dtype=torch.float32))
            self.ann_proj = nn.Linear(self.node_ann.size(1), hidden, bias=False)
        self.ctx_proj = nn.Linear(context_dim, hidden, bias=False) if context_dim > 0 else None

        # --- buffers: structure, masks, node-subset indices ---
        self.register_buffer("node_ids", torch.arange(kg.num_nodes))
        self.register_buffer("node_types",
            torch.tensor([kg.type_index[t] for t in kg.node_type], dtype=torch.long))
        self.register_buffer("adjacency", kg.structural_adjacency())
        self.register_buffer("program_index", torch.tensor(kg.program_index, dtype=torch.long))
        cue = torch.tensor([t == "cue" for t in kg.node_type])
        prog = torch.zeros(kg.num_nodes, dtype=torch.bool); prog[kg.program_index] = True
        self.register_buffer("cue_mask", cue.float().view(1, -1, 1))
        self.register_buffer("intrinsic_mask", (~cue & ~prog).float().view(1, -1, 1))
        self.register_buffer("lineage_idx",
            torch.tensor([kg.node_index[n] for n in kg.memory_nodes if n in kg.node_index], dtype=torch.long))
        self.register_buffer("chromatin_idx",
            torch.tensor([i for i, t in enumerate(kg.node_type) if t in CHROMATIN_TYPES], dtype=torch.long))
        plast = [i for i, t in enumerate(kg.node_type) if t == "plasticity"]
        self.register_buffer("plasticity_idx", torch.tensor(plast, dtype=torch.long))

    def _pool(self, h, idx):
        return h[:, idx, :].mean(1) if idx.numel() else h.new_zeros(h.size(0), self.hidden)

    def _atac_plasticity(self, acc):
        """Plasticity from the epigenetic landscape (ATAC). acc: [B,N] accessibility -> [B,1] in [0,1].
        Features: per-program accessibility profile (pathway + adjacent networks) + global chromatin
        openness + cross-program accessibility diversity (multi-lineage priming = poised/plastic)."""
        prog_acc = acc @ self.prog_membership.t()                  # [B,P]
        openness = acc.mean(dim=1, keepdim=True)                   # [B,1]
        q = prog_acc.clamp(min=0)
        q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-6)
        diversity = -(q * (q + 1e-9).log()).sum(dim=1, keepdim=True)   # [B,1]
        return torch.sigmoid(self.plast_from_atac(torch.cat([prog_acc, openness, diversity], dim=-1)))

    def _sparse_list(self):
        """Per-relation sparse adjacency, cached per device (rebuilt if device changes).
        Assumes adjacency is fixed after the first forward (true for all normal + ablation
        usage, where edge edits happen right after construction)."""
        if self._Asp is None or self._Asp[0].device != self.adjacency.device:
            self._Asp = [self.adjacency[r].to_sparse() for r in range(self.adjacency.size(0))]
        return self._Asp

    def _rgcn(self, h):                       # relation aggregation, memory-light: O(B,N,H)
        msg = self.self_lin(h)
        if self.sparse_adj:                   # sparse mm over the ~306 real edges (vs dense N^2)
            B, N, H = h.shape
            Asp = self._sparse_list()
            for r, lin in enumerate(self.rel_lin):
                lh = lin(h).permute(1, 0, 2).reshape(N, B * H)     # [N, B*H]
                msg = msg + torch.sparse.mm(Asp[r], lh).reshape(N, B, H).permute(1, 0, 2)
            return msg
        # dense fallback: loop over relations (not a fused [R,B,N,H] einsum) so big batches
        # don't OOM; at large batch the per-relation launch overhead is amortized.
        for r, lin in enumerate(self.rel_lin):
            msg = msg + torch.einsum("ds,bsh->bdh", self.adjacency[r], lin(h))
        return msg

    def _soft_attractor(self, logits):
        if self.attractor == "none":
            return logits
        base = 0.5 if self.attractor == "hard_wta" else self.attr_strength
        for k in range(self.attr_iters):
            if self.attractor == "delayed_soft":
                s = base * (k + 1) / self.attr_iters               # ramps with commitment
            elif self.attractor == "learned":
                s = torch.nn.functional.softplus(self.attr_param)
            else:
                s = base
            a = torch.softmax(logits, dim=-1)
            logits = logits + s * (a - a.mean(dim=-1, keepdim=True))
        return logits

    def forward(self, x0, plasticity=1.0, cue_window=None, context=None, atac=None):
        """cue_window: if set, the extrinsic cue is injected only for steps t < cue_window,
        then withdrawn (hysteresis/persistence test). context: optional [B, context_dim]
        experiment-metadata vector conditioning the graph. atac: optional [B, N] chromatin
        accessibility per node (paired 10x Multiome); only used when use_atac=True."""
        B = x0.size(0)
        if not torch.is_tensor(plasticity):
            plasticity = torch.full((B, 1), float(plasticity), device=x0.device)
        else:
            plasticity = plasticity.view(B, 1).to(x0.device)
        base_node = self.id_emb(self.node_ids) + self.type_emb(self.node_types)
        if self.ann_proj is not None:                             # gene-annotation features
            base_node = base_node + self.ann_proj(self.node_ann)
        base = base_node.unsqueeze(0)
        if self.use_atac:                                         # 2-channel [expression, accessibility]
            acc = atac if atac is not None else torch.zeros_like(x0)
            xin = self.in_proj(torch.stack([x0, acc], dim=-1))    # [B,N,2] -> [B,N,H]
        else:
            xin = self.in_proj(x0.unsqueeze(-1))                  # [B,N,1] -> [B,N,H]
        intrinsic_inj = xin * self.intrinsic_mask                 # persistent intrinsic state
        cue_inj = xin * self.cue_mask                             # transient extrinsic cue
        # inject the cell's expression ONCE, at init: intrinsic identity is the deeply-processed
        # starting attractor; the graph + resistance carry it forward (memory = inertia, not
        # a re-added drive). The cue is injected transiently each step (below).
        h = base.expand(B, -1, -1).contiguous() + intrinsic_inj
        if self.ctx_proj is not None and context is not None:     # experiment context
            h = h + self.ctx_proj(context).unsqueeze(1)

        plast_atac = None
        if self.plasticity_source == "atac":                      # plasticity from the epigenetic landscape
            acc = atac if atac is not None else torch.zeros(B, self.N, device=x0.device)
            plast_atac = self._atac_plasticity(acc)               # [B,1] per-cell plasticity

        for t in range(self.steps):
            cue_t = cue_inj * (self.cue_decay ** t)
            if self.plasticity_mode in ("amplify", "both"):
                cue_t = cue_t * plasticity.view(B, 1, 1)
            if cue_window is not None and t >= cue_window:        # cue withdrawn (hysteresis)
                cue_t = cue_t * 0.0
            cand = self.gru((self._rgcn(h) + cue_t).reshape(B * self.N, -1),
                            h.reshape(B * self.N, -1)).reshape(B, self.N, -1)
            if self.use_resistance:
                feat = torch.cat([self._pool(h, self.chromatin_idx),
                                  self._pool(h, self.lineage_idx),
                                  self._pool(h, self.program_index)], dim=-1)
                base_r = torch.sigmoid(self.W_resist(feat))                       # [B,1]
                if plast_atac is not None:                        # ATAC-derived plasticity (thesis)
                    plast_eff = plast_atac
                elif self.plasticity_source == "intrinsic":       # chromatin-openness proxy (RNA-only)
                    plast_eff = torch.sigmoid(self.plast_intrinsic(self._pool(h, self.chromatin_idx)))
                else:                                             # legacy: plasticity-node state + scalar
                    plast_eff = torch.sigmoid(self.W_plast(
                        torch.cat([self._pool(h, self.plasticity_idx), plasticity], dim=-1)))
                resist = base_r * (1 - plast_eff) if self.plasticity_mode in ("lower_resistance", "both") else base_r
            else:
                resist = torch.zeros(B, 1, device=x0.device)                     # ablation: pure candidate
            r = resist.view(B, 1, 1)
            h = r * h + (1 - r) * cand                            # memory persists via resistance, no re-injection

        prog_logits = self.prog_head(h[:, self.program_index, :]).squeeze(-1)
        quiescent = self.quiescent_head(h.mean(dim=1))
        logits = torch.cat([prog_logits, quiescent], dim=-1)
        if self.hybrid:
            logits = logits + self.skip(x0)
        return self._soft_attractor(logits)

    @torch.no_grad()
    def mean_resistance(self, x0, plasticity=1.0):
        """Diagnostic: mean transition resistance at the first step (0=movable, 1=frozen)."""
        B = x0.size(0)
        base = (self.id_emb(self.node_ids) + self.type_emb(self.node_types)).unsqueeze(0)
        xin = self.in_proj(x0.unsqueeze(-1))                      # match forward's init injection
        h = base.expand(B, -1, -1).contiguous() + xin * self.intrinsic_mask
        feat = torch.cat([self._pool(h, self.chromatin_idx), self._pool(h, self.lineage_idx),
                          self._pool(h, self.program_index)], dim=-1)
        return torch.sigmoid(self.W_resist(feat)).mean().item()
