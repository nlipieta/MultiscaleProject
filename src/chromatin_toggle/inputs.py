"""Turn a (cell context, extrinsic cue) pair into an initial node-activation
vector -- the ONLY thing the GNN observes. This is the model's input contract.

For real data, bypass this and supply a per-node activation vector directly
(see dataset.load_csv and the README's data-interface section).
"""
from __future__ import annotations

import torch

from .kg import KnowledgeGraph

LEVELS = {"off": 0.0, "low": 0.34, "med": 0.67, "high": 1.0}


def build_input(
    kg: KnowledgeGraph,
    on_nodes: list[str] | None = None,
    cue: str | None = None,
    level: str | float = "high",
) -> torch.Tensor:
    """Return an [N] activation vector: memory nodes = 1, cue node = level, else 0."""
    x = torch.zeros(kg.num_nodes)
    for name in on_nodes or []:
        x[kg.node_index[name]] = 1.0
    if cue is not None:
        lvl = LEVELS[level] if isinstance(level, str) else float(level)
        x[kg.node_index[cue]] = lvl
    return x


def row_input(
    kg: KnowledgeGraph,
    node_values: dict[str, float],
    cue: str | None = None,
    level: str | float = "high",
) -> torch.Tensor:
    """Build an input from a per-node activation dict (e.g. real expression-
    grounded memory from CELLxGENE), then overlay the extrinsic cue."""
    x = torch.zeros(kg.num_nodes)
    for name, val in node_values.items():
        if name in kg.node_index:
            x[kg.node_index[name]] = float(val)
    if cue is not None:
        lvl = LEVELS[level] if isinstance(level, str) else float(level)
        x[kg.node_index[cue]] = lvl
    return x


def context_input(
    kg: KnowledgeGraph,
    contexts: dict[str, list[str]],
    context: str,
    cue: str | None,
    level: str | float = "high",
) -> torch.Tensor:
    if context not in contexts:
        raise KeyError(f"unknown context {context!r}; known: {sorted(contexts)}")
    return build_input(kg, contexts[context], cue, level)
