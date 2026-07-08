"""Resistance-gated / competence-gated KG-GNN (minimal-first).

Reframes intrinsic memory as TRANSITION RESISTANCE (the barrier a cue must overcome to
leave the current attractor) rather than a re-injected default. Plasticity LOWERS the
barrier; it does not amplify the cue. Update rule:

    candidate = KG_GNN(h, cue_t, graph)
    resistance = base_resist(lineage, chromatin, program state) * (1 - plasticity_effect)
    h_next     = resistance * h + (1 - resistance) * candidate + alpha_memory * mem_inj

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
    def __init__(self, kg, hidden=64, steps=6, alpha_memory="learned", resistance=True,
                 plasticity_mode="lower_resistance", attractor="soft", hybrid=True,
                 cue_decay=0.6, attr_iters=3, attr_strength=0.15,
                 node_ann=None, context_dim=0):
        super().__init__()
        assert alpha_memory in ("zero", "low", "learned", "full")
        assert plasticity_mode in ("amplify", "lower_resistance", "both", "none")
        assert attractor in ("none", "hard_wta", "soft", "delayed_soft", "learned")
        self.N, self.steps, self.hidden = kg.num_nodes, steps, hidden
        self.use_resistance = resistance
        self.plasticity_mode = plasticity_mode
        self.attractor = attractor
        self.cue_decay = cue_decay
        self.attr_iters, self.attr_strength = attr_iters, attr_strength

        # --- shared GNN machinery ---
        self.id_emb = nn.Embedding(kg.num_nodes, hidden)
        self.type_emb = nn.Embedding(kg.num_types, hidden)
        self.in_proj = nn.Linear(1, hidden)
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

        # --- learnable/configurable memory reinjection alpha ---
        self._alpha_mode = alpha_memory
        if alpha_memory == "learned":
            self.alpha_param = nn.Parameter(torch.tensor(-2.2))   # sigmoid(-2.2)~0.1
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

    def _alpha(self):
        return {"zero": 0.0, "low": 0.1, "full": 1.0}.get(
            self._alpha_mode, None) if self._alpha_mode != "learned" else torch.sigmoid(self.alpha_param)

    def _pool(self, h, idx):
        return h[:, idx, :].mean(1) if idx.numel() else h.new_zeros(h.size(0), self.hidden)

    def _rgcn(self, h):                       # relation aggregation, memory-light: O(B,N,H)
        # loop over relations (not a fused [R,B,N,H] einsum) so big batches don't OOM;
        # at large batch the per-relation launch overhead is amortized over few batches.
        msg = self.self_lin(h)
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

    def forward(self, x0, plasticity=1.0, cue_window=None, context=None):
        """cue_window: if set, the extrinsic cue is injected only for steps t < cue_window,
        then withdrawn (hysteresis/persistence test). context: optional [B, context_dim]
        experiment-metadata vector conditioning the graph."""
        B = x0.size(0)
        if not torch.is_tensor(plasticity):
            plasticity = torch.full((B, 1), float(plasticity), device=x0.device)
        else:
            plasticity = plasticity.view(B, 1).to(x0.device)
        base_node = self.id_emb(self.node_ids) + self.type_emb(self.node_types)
        if self.ann_proj is not None:                             # gene-annotation features
            base_node = base_node + self.ann_proj(self.node_ann)
        base = base_node.unsqueeze(0)
        xin = self.in_proj(x0.unsqueeze(-1))
        mem_inj = xin * self.intrinsic_mask
        cue_inj = xin * self.cue_mask
        h = base.expand(B, -1, -1).contiguous()
        if self.ctx_proj is not None and context is not None:     # experiment context
            h = h + self.ctx_proj(context).unsqueeze(1)
        alpha = self._alpha()

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
                plast_eff = torch.sigmoid(self.W_plast(
                    torch.cat([self._pool(h, self.plasticity_idx), plasticity], dim=-1)))
                resist = base_r * (1 - plast_eff) if self.plasticity_mode in ("lower_resistance", "both") else base_r
            else:
                resist = torch.zeros(B, 1, device=x0.device)                     # ablation: pure candidate
            r = resist.view(B, 1, 1)
            h = r * h + (1 - r) * cand + alpha * mem_inj

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
        h = base.expand(B, -1, -1).contiguous()
        feat = torch.cat([self._pool(h, self.chromatin_idx), self._pool(h, self.lineage_idx),
                          self._pool(h, self.program_index)], dim=-1)
        return torch.sigmoid(self.W_resist(feat)).mean().item()
