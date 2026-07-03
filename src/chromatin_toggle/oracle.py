"""Mechanistic KG-propagation oracle.

This is a signed threshold/Boolean-relaxation network over the literature KG
(the "continuous Boolean simulation" style described in Section 1 of the report).
It propagates an initial activation through the mechanistic edges and reads out
the winning response program.

ROLE: it GENERATES bootstrap training labels and provides a human-readable
mechanistic trace. It is NOT the predictive model -- the GNN in model.py must
LEARN this mapping from (cue + memory) alone, without seeing the biases/weights.

For real scientific use, replace oracle-generated labels with experimentally
measured phenotype labels (e.g. Perturb-seq) via the CSV interface.
"""
from __future__ import annotations

import torch

from .kg import KnowledgeGraph

QUIESCENT = "Quiescent"


def _input_mask(kg: KnowledgeGraph) -> torch.Tensor:
    """Exogenous nodes (cues + intrinsic-memory nodes) are held at their input
    value throughout propagation -- including when that value is 0 (absent).
    Everything else is computed by the network."""
    mask = torch.zeros(kg.num_nodes, dtype=torch.bool)
    for i, t in enumerate(kg.node_type):
        if t == "cue":
            mask[i] = True
    for name in kg.memory_nodes:
        mask[kg.node_index[name]] = True
    return mask


def propagate(
    kg: KnowledgeGraph,
    x0: torch.Tensor,
    steps: int = 20,
    gain: float = 6.0,
) -> torch.Tensor:
    """Relax the network to a fixed point. Cue + memory inputs are clamped.

    Returns final activation vector [N] in [0, 1].
    """
    A = kg.dense_adjacency().sum(0)          # [N, N] combined signed weights
    bias = kg.node_bias                      # [N]
    clamp_mask = _input_mask(kg)             # pin exogenous inputs to x0
    a = x0.clone()
    for _ in range(steps):
        drive = bias + A @ a                 # [N]
        a_new = torch.sigmoid(gain * drive)
        a_new[clamp_mask] = x0[clamp_mask]
        a = a_new
    return a


def label_from_activation(
    kg: KnowledgeGraph, a: torch.Tensor, threshold: float = 0.5
) -> str:
    """Argmax over program nodes; QUIESCENT if none crosses threshold."""
    prog = a[torch.tensor(kg.program_index)]
    top = int(torch.argmax(prog))
    if float(prog[top]) < threshold:
        return QUIESCENT
    return kg.program_nodes[top]


def oracle_label(kg: KnowledgeGraph, x0: torch.Tensor, **kw) -> str:
    return label_from_activation(kg, propagate(kg, x0, **kw))


def trace(
    kg: KnowledgeGraph,
    x0: torch.Tensor,
    baseline: torch.Tensor | None = None,
    top_k: int = 8,
    **kw,
):
    """Return (label, [(node, activation, delta), ...]) for mechanistic
    explanation. When `baseline` (same memory, no cue) is given, nodes are ranked
    by how much the cue MOVED them -- isolating the cue-driven cascade and hiding
    always-on basal repressors."""
    a = propagate(kg, x0, **kw)
    label = label_from_activation(kg, a)
    if baseline is not None:
        b = propagate(kg, baseline, **kw)
        moved = [
            (kg.node_ids[i], float(a[i]), float(a[i] - b[i]))
            for i in range(kg.num_nodes)
        ]
        moved = [m for m in moved if abs(m[2]) > 0.2]
        moved.sort(key=lambda t: abs(t[2]), reverse=True)
        return label, moved[:top_k]
    ranked = sorted(
        ((kg.node_ids[i], float(a[i]), 0.0) for i in range(kg.num_nodes)),
        key=lambda t: t[1],
        reverse=True,
    )
    return label, [(n, v, d) for n, v, d in ranked if v > 0.5][:top_k]


def all_classes(kg: KnowledgeGraph) -> list[str]:
    return list(kg.program_nodes) + [QUIESCENT]
