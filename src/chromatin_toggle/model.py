"""Relational, temporal Graph Neural Network over the literature KG.

Design mirrors the report's Section 5 architecture:
  * nodes = KG entities (cues, signaling, TFs, modifiers, marks, programs)
  * each node embedding = identity + type embedding, plus an injected initial
    activation (the observed cue + intrinsic memory -- ALL the model sees)
  * T rounds of relation-typed message passing "over simulated time", with a
    GRU cell updating node hidden states each round (R-GCN x recurrent update)
  * readout on the response-program nodes -> phenotype logits (+ a Quiescent
    class from global pooling)

The model sees only the graph STRUCTURE (binary typed adjacency) plus the
observed initial state (cue + intrinsic memory). It never sees the oracle's
node biases OR its signed edge weights -- edge sign and magnitude are learned
per relation via rel_lin. It must learn signal x memory -> phenotype itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .kg import KnowledgeGraph


class ToggleGNN(nn.Module):
    def __init__(self, kg: KnowledgeGraph, hidden: int = 64, steps: int = 6):
        super().__init__()
        self.N = kg.num_nodes
        self.steps = steps
        self.hidden = hidden

        self.id_emb = nn.Embedding(kg.num_nodes, hidden)
        self.type_emb = nn.Embedding(kg.num_types, hidden)
        self.in_proj = nn.Linear(1, hidden)

        # one linear transform per relation (R-GCN style) + a self transform
        self.rel_lin = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(kg.num_relations)]
        )
        self.self_lin = nn.Linear(hidden, hidden, bias=False)
        self.gru = nn.GRUCell(hidden, hidden)

        self.prog_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.quiescent_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

        # static graph tensors as buffers (move with .to(device))
        self.register_buffer("node_ids", torch.arange(kg.num_nodes))
        self.register_buffer(
            "node_types",
            torch.tensor([kg.type_index[t] for t in kg.node_type], dtype=torch.long),
        )
        # STRUCTURE only (binary). The oracle's signed weights are deliberately
        # withheld so the model can't shortcut to the label-generating function.
        self.register_buffer("adjacency", kg.structural_adjacency())  # [R, N, N] 0/1
        self.register_buffer(
            "program_index", torch.tensor(kg.program_index, dtype=torch.long)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """x0: [B, N] observed initial activations -> logits [B, num_programs+1]."""
        B = x0.size(0)
        base = self.id_emb(self.node_ids) + self.type_emb(self.node_types)  # [N, H]
        h = base.unsqueeze(0).expand(B, -1, -1) + self.in_proj(x0.unsqueeze(-1))
        h = h.contiguous()  # [B, N, H]

        for _ in range(self.steps):
            msg = self.self_lin(h)  # [B, N, H]
            for r in range(self.adjacency.size(0)):
                A_r = self.adjacency[r]           # [N, N] (dst, src)
                transformed = self.rel_lin[r](h)  # [B, N, H]
                # aggregate over sources: msg[dst] += sum_src A_r[dst,src]*W_r(h[src])
                msg = msg + torch.einsum("ds,bsh->bdh", A_r, transformed)
            h = self.gru(
                msg.reshape(B * self.N, -1), h.reshape(B * self.N, -1)
            ).reshape(B, self.N, -1)

        prog_h = h[:, self.program_index, :]                  # [B, P, H]
        prog_logits = self.prog_head(prog_h).squeeze(-1)      # [B, P]
        quiescent_logit = self.quiescent_head(h.mean(dim=1))  # [B, 1]
        return torch.cat([prog_logits, quiescent_logit], dim=-1)
