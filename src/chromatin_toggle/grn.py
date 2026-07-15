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


class MultistableGRN(GRNDynamics):
    """Signed-GRN trained to be MULTISTABLE: each stored program is a stable fixed-point attractor and
    cells flow to their program's basin. Keeps the fixed literature signs + learnable magnitudes/bias
    (the mechanism/interpretability), but adds learnable per-program prototypes and a training objective
    the classifier objective never supplied. Attractors then become a genuine property of the flow.

    stored_classes: the program labels to store as attractors (typically the classes present in the data).
    proto_init: [C, N] initial prototype states (e.g. per-program mean cell state); defaults to zeros.
    """

    def __init__(self, kg, stored_classes, proto_init=None, steps=15, eta=0.3):
        super().__init__(kg, steps=steps, eta=eta)
        self.stored = list(stored_classes)
        C = len(self.stored)
        self.proto = nn.Parameter(proto_init.clone() if proto_init is not None
                                  else torch.zeros(C, self.N))
        # anchor: each attractor must stay near its class's data mean, or the optimizer collapses all
        # prototypes onto one shared stable point (a degenerate solution to flow/fp/basin).
        self.register_buffer("proto_anchor", proto_init.clone() if proto_init is not None
                             else torch.zeros(C, self.N))
        # map full-class index -> stored position (-1 if that class is not stored)
        full2stored = [-1] * len(self.classes)
        for j, c in enumerate(self.stored):
            full2stored[self.classes.index(c)] = j
        self.register_buffer("full2stored", torch.tensor(full2stored, dtype=torch.long))

    def fp_loss(self):
        """Prototypes must be fixed points of the dynamics: ||step(p) - p||^2."""
        return ((self._step(self.proto) - self.proto) ** 2).mean()

    def anchor_loss(self):
        """Keep each attractor near its class's data mean (anti-collapse)."""
        return ((self.proto - self.proto_anchor) ** 2).mean()

    def basin_loss(self, eps=0.1, k=4):
        """Each prototype must be a STABLE attractor: perturb it and require re-settling to RETURN.
        This is what turns a fixed point into an attracting basin (local stability); without it a
        prototype can be a saddle cells flow past -> monostable collapse. Trains stability directly
        through the dynamics, no eigendecomposition."""
        p = self.proto.repeat(k, 1)
        xs = self.settle(p + eps * torch.randn_like(p))
        return ((xs - p) ** 2).mean()

    def flow_loss(self, x0, y_full):
        """Cells must settle to their program's prototype: ||settle(x) - proto[stored(y)]||^2."""
        xs = self.settle(x0)
        tgt = self.proto[self.full2stored[y_full]]
        return ((xs - tgt) ** 2).mean()

    @torch.no_grad()
    def assign(self, x0, clamp_idx=None, n_steps=None):
        """Settle (optionally with a node clamped throughout = a held knockdown) then assign each cell to
        the nearest prototype; returns stored-position indices."""
        xs = self.settle(x0, clamp_idx=clamp_idx, n_steps=n_steps)
        d = torch.cdist(xs, self.proto)          # [B, C]
        return d.argmin(1)

    @torch.no_grad()
    def basin_prob(self, x0, clamp_idx=None, n_steps=None, temp=0.5):
        """Graded basin membership: softmax over -distance to each prototype after settling. More
        sensitive to PARTIAL destabilization than the hard nearest-prototype assignment."""
        xs = self.settle(x0, clamp_idx=clamp_idx, n_steps=n_steps)
        d = torch.cdist(xs, self.proto)          # [B, C]
        return torch.softmax(-d / temp, dim=1)
