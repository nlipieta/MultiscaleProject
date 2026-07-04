"""Main-result classification + the marker-gene-shortcut control.

Some KG input genes are late-stage READOUTS of the program state (e.g. Sox9 for
ADM) rather than upstream drivers, so including them as inputs while labelling by
cell state is partly circular. This trains the multiscale model under three input
regimes and reports held-out per-program accuracy, so the shortcut is measured
rather than hidden:

  full          -- all node inputs (may exploit marker readouts)
  no_markers    -- program-marker readout genes zeroed (Sox9, mTORC1, Autophagy)
  lineage_only  -- ONLY the theory's legitimate inputs: extrinsic cue + lineage/
                   master-TF intrinsic memory (PU1, Oct4, MyoD, AP1); everything
                   downstream zeroed. The honest test of the thesis's claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .dynamics import ToggleDynamics, _load, train, class_weights
from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes

MARKER_NODES = ["Sox9", "mTORC1", "Autophagy"]  # program-proximal readouts


def _masked(X, kg, mode):
    X = X.clone()
    if mode == "full":
        return X
    if mode == "no_markers":
        for n in MARKER_NODES:
            if n in kg.node_index:
                X[:, kg.node_index[n]] = 0.0
        return X
    if mode == "lineage_only":
        keep = {kg.node_index[n] for n in kg.memory_nodes if n in kg.node_index}
        keep |= {i for i, t in enumerate(kg.node_type) if t == "cue"}
        for j in range(kg.num_nodes):
            if j not in keep:
                X[:, j] = 0.0
        return X
    raise ValueError(mode)


@torch.no_grad()
def _predict(model, X, bs=1024):
    model.eval()
    return torch.cat([model(X[i:i + bs], plasticity=1.0).argmax(-1)
                      for i in range(0, X.size(0), bs)])


def main():
    ap = argparse.ArgumentParser(description="Classification + marker-shortcut control")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--subsample", type=int, default=0, help="cap total cells (0=all)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class-weight", action="store_true", help="inverse-frequency class weights")
    args = ap.parse_args()

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    if args.subsample and X.size(0) > args.subsample:
        g0 = torch.Generator().manual_seed(0)
        keep = torch.randperm(X.size(0), generator=g0)[:args.subsample]
        X, y = X[keep], y[keep]
    n_classes = len(classes)
    prog_names = [c for c in classes if c != QUIESCENT]

    modes = ["full", "no_markers", "lineage_only"]
    print(f"Marker-shortcut control | data={Path(args.data).name} n={X.size(0)} "
          f"seeds={args.seeds}\n")
    results = {m: {"acc": [], "prog": []} for m in modes}
    for s in args.seeds:
        g = torch.Generator().manual_seed(s)
        perm = torch.randperm(X.size(0), generator=g)
        k = int(X.size(0) * args.val_frac)
        va, tr = perm[:k], perm[k:]
        w = class_weights(y[tr], n_classes) if args.class_weight else None
        for m in modes:
            Xm = _masked(X, kg, m)
            torch.manual_seed(s)
            model = ToggleDynamics(kg, hidden=args.hidden, steps=args.steps)
            train(model, Xm[tr], y[tr], args.epochs, 256, args.lr, s, weights=w)
            pred = _predict(model, Xm[va])
            results[m]["acc"].append(float((pred == y[va]).float().mean()))
            # mean recall over the activated (non-Quiescent) programs
            recs = []
            for p in prog_names:
                c = classes.index(p)
                mask = y[va] == c
                if int(mask.sum()):
                    recs.append(float((pred[mask] == c).float().mean()))
            results[m]["prog"].append(float(np.mean(recs)))

    print(f"{'input regime':<16}{'overall acc':>16}{'mean program recall':>22}")
    print("-" * 54)
    for m in modes:
        a, sa = np.mean(results[m]["acc"]), np.std(results[m]["acc"])
        p, sp = np.mean(results[m]["prog"]), np.std(results[m]["prog"])
        print(f"{m:<16}{f'{a:.3f}±{sa:.3f}':>16}{f'{p:.3f}±{sp:.3f}':>22}")
    print("\nA large full -> no_markers drop = reliance on marker readouts (shortcut).")
    print("lineage_only = accuracy using ONLY cue + lineage memory (the thesis inputs).")


if __name__ == "__main__":
    main()
