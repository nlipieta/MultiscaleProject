"""Stratified k-fold cross-validation (publication item 6: generalization).

Splits the data into k class-balanced folds; trains the marker-controlled
multiscale model on k-1 folds and evaluates on the held-out fold, rotating.
Reports per-fold and mean +/- std held-out accuracy and program recall -- a
standard generalization estimate that does not depend on one lucky split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .dynamics import ToggleDynamics, _load, _mask_input, train
from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes


def _stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in torch.unique(y).tolist():
        idx = np.where(y.numpy() == c)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(int(j))
    return [np.array(f) for f in folds]


def main():
    ap = argparse.ArgumentParser(description="Stratified k-fold CV of the multiscale model")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"], default="no_markers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    X = _mask_input(X, kg, args.mask)
    if args.subsample and X.size(0) > args.subsample:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:args.subsample]
        X, y = X[idx], y[idx]
    prog_cols = [classes.index(c) for c in classes if c != QUIESCENT]
    folds = _stratified_folds(y, args.kfolds, args.seed)
    print(f"CV | data={Path(args.data).name} n={X.size(0)} k={args.kfolds} mask={args.mask}\n")

    accs, progs = [], []
    for f in range(args.kfolds):
        te = torch.tensor(folds[f], dtype=torch.long)
        tr = torch.tensor(np.concatenate([folds[i] for i in range(args.kfolds) if i != f]), dtype=torch.long)
        torch.manual_seed(args.seed)
        m = ToggleDynamics(kg, hidden=args.hidden, steps=args.steps)
        train(m, X[tr], y[tr], args.epochs, 256, 1e-3, args.seed)
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(X[te][i:i+1024], plasticity=1.0).argmax(-1)
                              for i in range(0, len(te), 1024)])
        acc = float((pred == y[te]).float().mean())
        recs = [float((pred[y[te] == c] == c).float().mean()) for c in prog_cols
                if int((y[te] == c).sum())]
        accs.append(acc); progs.append(float(np.mean(recs)))
        print(f"  fold {f+1}/{args.kfolds}  acc {acc:.3f}  program-recall {progs[-1]:.3f}")

    print(f"\n{args.kfolds}-fold CV: accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f} | "
          f"program recall {np.mean(progs):.3f} +/- {np.std(progs):.3f}")


if __name__ == "__main__":
    main()
