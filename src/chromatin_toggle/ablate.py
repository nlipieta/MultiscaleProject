"""Deeper mechanism + structure ablation table (reviewer item #4).

Removes one load-bearing piece of the multiscale model at a time and re-measures
program recall on a FIXED split (same split across ablations, so the delta is
attributable to the removed piece, not to fold noise). Three families:

  MECHANISM (thesis inductive biases; model flags)
    full                -- everything on
    -asymmetric         -- symmetric integration (no persistent-strong intrinsic /
                           transient-weak decaying extrinsic distinction)
    -plasticity_gate    -- cue gain no longer scaled by the plasticity input
    -attractor(WTA)     -- no winner-take-all sharpening among program logits

  STRUCTURE (what the literature graph contributes; adjacency edits)
    scramble_edges      -- permute the source axis (degree kept, wiring destroyed)
    no_edges            -- zero adjacency (pure per-node readout, graph removed)
    collapse_relations  -- merge all typed relations into one (removes relation typing)

  NODES (which biological inputs matter; input masking by node type)
    -intrinsic_memory   -- zero the lineage/identity memory-node inputs
    -chromatin_nodes    -- zero chromatin modifier + mark node inputs
    -tf_nodes           -- zero transcription-factor node inputs

NOTE on "remove edge signs": N/A for this model. The GNN sees only BINARY
structural_adjacency() -- edge sign/magnitude are never exposed (it learns them
per relation via rel_lin). So there is no sign to remove; the closest structural
knockouts are scramble_edges / collapse_relations / no_edges above.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .device import pick_device
from .dynamics import ToggleDynamics, _load, _mask_input, train, class_weights, predict, predict_proba
from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes

CHROMATIN_TYPES = {"modifier", "mark"}


def _split(y, groups, val_frac, seed, group_split):
    if group_split and groups is not None:
        uniq = sorted(set(groups)); rng = np.random.default_rng(seed); rng.shuffle(uniq)
        nval = max(1, int(len(uniq) * val_frac)); vset = set(uniq[:nval])
        isval = np.array([g in vset for g in groups])
        return np.where(~isval)[0], np.where(isval)[0]
    perm = np.random.default_rng(seed).permutation(len(y))
    k = int(len(y) * val_frac)
    return perm[k:], perm[:k]


def _node_mask_input(X, kg, drop):
    """Zero the input columns for a set of node indices (input-level knockout)."""
    X = X.clone(); X[:, list(drop)] = 0.0
    return X


def _metrics(pred, proba, y, n_classes, prog_cols):
    from sklearn.metrics import average_precision_score
    pred, y = np.asarray(pred), np.asarray(y)
    acc = float((pred == y).mean())
    bal = float(np.mean([ (pred[y==c]==c).mean() for c in range(n_classes) if (y==c).any() ]))
    prog = float(np.mean([ (pred[y==c]==c).mean() for c in prog_cols if (y==c).any() ]))
    aps = [float(average_precision_score((y==c).astype(int), proba[:, c]))
           for c in prog_cols if (y==c).any()] if proba is not None else []
    auprc = float(np.mean(aps)) if aps else float("nan")
    return acc, bal, prog, auprc


def _run_one(name, kg, X, y, tr, te, n_classes, prog_cols, args, w, dev):
    flags = dict(asymmetric=True, plasticity=True, attractor=True, hybrid=True)
    Xtr, Xte = X[tr], X[te]
    adj_override = None
    if name == "-asymmetric":       flags["asymmetric"] = False
    elif name == "-plasticity_gate": flags["plasticity"] = False
    elif name == "-attractor(WTA)":  flags["attractor"] = False
    elif name == "-hybrid_residual": flags["hybrid"] = False
    elif name in ("scramble_edges", "no_edges", "collapse_relations"):
        adj = kg.structural_adjacency().clone()
        if name == "scramble_edges":
            g = torch.Generator().manual_seed(args.seed + 7)
            perm = torch.randperm(kg.num_nodes, generator=g)
            adj = adj[:, :, perm].contiguous()
        elif name == "no_edges":
            adj = torch.zeros_like(adj)
        elif name == "collapse_relations":
            merged = (adj.sum(0) != 0).to(adj.dtype)
            adj = torch.zeros_like(adj); adj[0] = merged
        adj_override = adj
    elif name == "-intrinsic_memory":
        drop = {kg.node_index[n] for n in kg.memory_nodes if n in kg.node_index}
        Xtr = _node_mask_input(Xtr, kg, drop); Xte = _node_mask_input(Xte, kg, drop)
    elif name == "-chromatin_nodes":
        drop = {i for i, t in enumerate(kg.node_type) if t in CHROMATIN_TYPES}
        Xtr = _node_mask_input(Xtr, kg, drop); Xte = _node_mask_input(Xte, kg, drop)
    elif name == "-tf_nodes":
        drop = {i for i, t in enumerate(kg.node_type) if t == "tf"}
        Xtr = _node_mask_input(Xtr, kg, drop); Xte = _node_mask_input(Xte, kg, drop)

    torch.manual_seed(args.seed)
    m = ToggleDynamics(kg, hidden=args.hidden, steps=args.steps, **flags).to(dev)
    if adj_override is not None:
        m.adjacency.copy_(adj_override.to(dev))
    train(m, Xtr, y[tr], args.epochs, 256, 1e-3, args.seed, weights=w)
    pred = predict(m, Xte).numpy()
    proba = predict_proba(m, Xte).numpy()
    return _metrics(pred, proba, y[te].numpy(), n_classes, prog_cols)


def main():
    ap = argparse.ArgumentParser(description="Mechanism/structure/node ablation table")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"], default="no_markers")
    ap.add_argument("--group-split", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="auto", help="cpu / cuda / mps / auto")
    args = ap.parse_args()
    dev = pick_device(args.device)

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    X = _mask_input(X, kg, args.mask)
    groups = df["dataset"].to_numpy() if "dataset" in df.columns else None
    if args.subsample and X.size(0) > args.subsample:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:args.subsample]
        X, y = X[idx], y[idx]
        if groups is not None:
            groups = groups[idx.numpy()]
    n_classes = len(classes)
    prog_cols = [classes.index(c) for c in classes if c != QUIESCENT]
    tr, te = _split(y, groups, args.val_frac, args.seed, args.group_split)
    w = class_weights(y[tr], n_classes) if args.class_weight else None

    order = ["full", "-hybrid_residual", "-asymmetric", "-plasticity_gate", "-attractor(WTA)",
             "scramble_edges", "no_edges", "collapse_relations",
             "-intrinsic_memory", "-chromatin_nodes", "-tf_nodes"]
    print(f"Ablation | data={Path(args.data).name} n={X.size(0)} mask={args.mask} "
          f"group_split={args.group_split} class_weight={args.class_weight} "
          f"seed={args.seed} steps={args.steps} hidden={args.hidden}")
    print(f"(same fixed split across all ablations; delta vs full attributes the mechanism)\n")

    results = {}
    full_prog = full_auprc = None
    for name in order:
        a, b, p, ap = _run_one(name, kg, X, y, tr, te, n_classes, prog_cols, args, w, dev)
        if name == "full":
            full_prog, full_auprc = p, ap
        results[name] = (a, b, p, ap)
        print(f"  {name:<20} acc {a:.3f}  bal {b:.3f}  prog-recall {p:.3f}  AUPRC {ap:.3f}  "
              f"(ΔAUPRC {ap-full_auprc:+.3f}, Δprog {p-full_prog:+.3f})")

    print(f"\n{'ablation':<20}{'acc':>9}{'bal':>9}{'prog-rec':>10}{'AUPRC':>9}"
          f"{'ΔAUPRC':>10}{'Δprog':>9}")
    print("-" * 76)
    for name in order:
        a, b, p, ap = results[name]
        da = "" if name == "full" else f"{ap-full_auprc:+.3f}"
        dp = "" if name == "full" else f"{p-full_prog:+.3f}"
        print(f"{name:<20}{a:>9.3f}{b:>9.3f}{p:>10.3f}{ap:>9.3f}{da:>10}{dp:>9}")
    print("\nΔAUPRC is the thesis-relevant metric (program ranking). Most-negative ΔAUPRC = "
          "most load-bearing for ranking; e.g. no_edges should be NEGATIVE (structure helps ranking).")


if __name__ == "__main__":
    main()
