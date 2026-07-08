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


class KGEncoder(nn.Module):
    """Backbone shared by the classifier: node embeddings + T rounds of
    relation-typed message passing over the KG. Returns final node states."""

    def __init__(self, kg: KnowledgeGraph, hidden: int = 64, steps: int = 6):
        super().__init__()
        self.N = kg.num_nodes
        self.steps = steps
        self.id_emb = nn.Embedding(kg.num_nodes, hidden)
        self.type_emb = nn.Embedding(kg.num_types, hidden)
        self.in_proj = nn.Linear(1, hidden)
        self.rel_lin = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(kg.num_relations)]
        )
        self.self_lin = nn.Linear(hidden, hidden, bias=False)
        self.gru = nn.GRUCell(hidden, hidden)
        self.register_buffer("node_ids", torch.arange(kg.num_nodes))
        self.register_buffer(
            "node_types",
            torch.tensor([kg.type_index[t] for t in kg.node_type], dtype=torch.long),
        )
        self.register_buffer("adjacency", kg.dense_adjacency())

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        base = self.id_emb(self.node_ids) + self.type_emb(self.node_types)
        h = (base.unsqueeze(0).expand(B, -1, -1)
             + self.in_proj(x0.unsqueeze(-1))).contiguous()
        for _ in range(self.steps):
            msg = self.self_lin(h)
            for r in range(self.adjacency.size(0)):
                msg = msg + torch.einsum("ds,bsh->bdh", self.adjacency[r], self.rel_lin[r](h))
            h = self.gru(msg.reshape(B * self.N, -1),
                         h.reshape(B * self.N, -1)).reshape(B, self.N, -1)
        return h


class KGClassifier(nn.Module):
    """Generic classifier: KG message passing over per-cell node activations ->
    mean-pool -> linear head over `n_classes`. Used for the real-label benchmark
    (arbitrary labels, e.g. cell type or disease)."""

    def __init__(self, kg: KnowledgeGraph, n_classes: int, hidden: int = 64, steps: int = 6):
        super().__init__()
        self.encoder = KGEncoder(kg, hidden=hidden, steps=steps)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x0)          # [B, N, H]
        return self.head(h.mean(dim=1))
