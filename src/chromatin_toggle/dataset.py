"""Build training data.

Two paths:

1. `build_bootstrap(...)`  -- WIRING HARNESS. Enumerates (context x cue x level)
   combinations, labels each with the mechanistic oracle, and adds noisy
   replicas. This lets the whole pipeline train + predict end-to-end with zero
   external data. It reproduces literature-encoded mechanisms; it is NOT an
   independent scientific result.

2. `load_csv(path)`       -- REAL DATA. One row per observation. Columns = every
   KG node name (initial activation in [0,1], e.g. scaled scRNA/scATAC pseudobulk
   for that cell/condition) plus a `label` column naming the observed program.
   This is the hook for Perturb-seq / real single-cell phenotype labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .inputs import LEVELS, build_input
from .kg import KnowledgeGraph
from .oracle import all_classes, oracle_label


def build_bootstrap(
    kg: KnowledgeGraph,
    contexts: dict[str, list[str]],
    cues: list[str],
    levels: tuple[str, ...] = ("low", "med", "high"),
    replicas: int = 8,
    noise: float = 0.08,
    seed: int = 0,
    exclude: set[tuple[str, str | None, str | None]] | None = None,
):
    """Return (X [M,N] float, y [M] long, groups [M] long, classes list[str]).

    `exclude` is a set of (context, cue, level) base combinations to omit
    entirely -- used to hold the literature anchors out of training so the
    anchor scorecard is a true generalization test, not memorization.

    `groups` labels each row with the index of its base combination so callers
    can split by combination (all replicas of a combo stay on one side); a
    random per-row split would leak because replicas are near-duplicates.
    """
    classes = all_classes(kg)
    class_index = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)
    exclude = exclude or set()

    base: list[tuple[torch.Tensor, str]] = []
    # every context with no cue -> baseline
    for ctx, on in contexts.items():
        if (ctx, None, None) in exclude:
            continue
        x = build_input(kg, on, None)
        base.append((x, oracle_label(kg, x)))
    # every context x cue x level
    for ctx, on in contexts.items():
        for cue in cues:
            for lvl in levels:
                if (ctx, cue, lvl) in exclude:
                    continue
                x = build_input(kg, on, cue, lvl)
                base.append((x, oracle_label(kg, x)))

    X, y, groups = [], [], []
    for gid, (x, lab) in enumerate(base):
        for r in range(replicas):
            xn = x.clone()
            if r > 0 and noise > 0:
                # jitter only the observed (nonzero) inputs; keep >= 0
                mask = xn > 0
                jitter = torch.tensor(
                    rng.normal(0, noise, size=xn.shape), dtype=xn.dtype
                )
                xn[mask] = (xn[mask] + jitter[mask]).clamp(0.0, 1.0)
            X.append(xn)
            y.append(class_index[lab])
            groups.append(gid)

    return (
        torch.stack(X),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(groups, dtype=torch.long),
        classes,
    )


def load_csv(kg: KnowledgeGraph, path: str | Path, classes: list[str]):
    """Load real observations. Missing node columns default to 0."""
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column")
    class_index = {c: i for i, c in enumerate(classes)}
    X = torch.zeros(len(df), kg.num_nodes)
    for name, j in kg.node_index.items():
        if name in df.columns:
            X[:, j] = torch.tensor(df[name].to_numpy(), dtype=torch.float32)
    y = torch.tensor([class_index[l] for l in df["label"]], dtype=torch.long)
    return X, y


def split(X, y, val_frac: float = 0.2, seed: int = 0, groups=None):
    """Split into (train, val).

    If `groups` is given, split by group so all rows of a group land on the same
    side (prevents near-duplicate replicas leaking across the split). Otherwise
    a plain per-row split (fine for independent real observations).
    """
    g = torch.Generator().manual_seed(seed)
    if groups is None:
        perm = torch.randperm(X.size(0), generator=g)
        n_val = int(X.size(0) * val_frac)
        vi, ti = perm[:n_val], perm[n_val:]
        return (X[ti], y[ti]), (X[vi], y[vi])

    uniq = torch.unique(groups)
    gperm = uniq[torch.randperm(uniq.numel(), generator=g)]
    n_val_g = max(1, int(uniq.numel() * val_frac))
    val_groups = set(gperm[:n_val_g].tolist())
    is_val = torch.tensor([int(gid) in val_groups for gid in groups])
    vi = is_val.nonzero(as_tuple=True)[0]
    ti = (~is_val).nonzero(as_tuple=True)[0]
    return (X[ti], y[ti]), (X[vi], y[vi])
