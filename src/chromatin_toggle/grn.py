"""Signed-GRN dynamical model: node activities evolve on the KG's signed edges to a fixed point.
Attractors = programs = the minima of a (Waddington) potential landscape. See scripts/grn_dynamics.py
(train + perturbation) and scripts/landscape_viz.py (landscape visualization on the fixed points).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .oracle import QUIESCENT, all_classes


class GRNDynamics(nn.Module):
    def __init__(self, kg, steps=15, eta=0.3):
        super().__init__()
        self.N, self.steps, self.eta = kg.num_nodes, steps, eta
        self.classes = all_classes(kg)
        src, dst, sign = [], [], []
        for s, rel, d, _w in kg.edges:
            src.append(s); dst.append(d); sign.append(kg.relation_sign[rel])
        self.register_buffer("src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.tensor(dst, dtype=torch.long))
        self.register_buffer("sign", torch.tensor(sign, dtype=torch.float32))
        self.mag = nn.Parameter(torch.full((len(src),), 0.5))     # softplus -> positive magnitude/edge
        self.bias = nn.Parameter(torch.zeros(self.N))
        self.in_scale = nn.Parameter(torch.tensor(1.0))
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.q_logit = nn.Parameter(torch.zeros(1))
        cls_node = [kg.node_index[c] if (c != QUIESCENT and c in kg.node_index) else -1
                    for c in self.classes]
        self.register_buffer("cls_node", torch.tensor(cls_node, dtype=torch.long))
        self.q_index = self.classes.index(QUIESCENT) if QUIESCENT in self.classes else -1

    def _step(self, x):
        m = torch.nn.functional.softplus(self.mag) * self.sign    # [E] signed weights
        agg = x.new_zeros(x.shape)
        agg.index_add_(1, self.dst, x[:, self.src] * m)           # incoming signed messages
        return (1 - self.eta) * x + self.eta * torch.tanh(agg + self.bias)

    def settle(self, x0, clamp_idx=None, n_steps=None):
        """Evolve to the fixed point; returns the settled activity state [B, N]."""
        steps = n_steps if n_steps is not None else self.steps
        x = x0 * self.in_scale
        if clamp_idx is not None:
            x = x.clone(); x[:, clamp_idx] = 0.0
        for _ in range(steps):
            x = self._step(x)
            if clamp_idx is not None:
                x[:, clamp_idx] = 0.0
        return x

    def forward(self, x0, clamp_idx=None, n_steps=None):
        x = self.settle(x0, clamp_idx=clamp_idx, n_steps=n_steps)
        B = x.size(0)
        logits = x.new_full((B, len(self.classes)), 0.0)
        prog = self.cls_node >= 0
        logits[:, prog] = x[:, self.cls_node[prog]] * self.temp
        if self.q_index >= 0:
            logits[:, self.q_index] = self.q_logit
        return logits
