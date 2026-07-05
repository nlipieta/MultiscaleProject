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

from .device import pick_device
from .dynamics import ToggleDynamics, _load, _mask_input, train, class_weights, predict
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
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--group-split", action="store_true",
                    help="assign whole datasets to folds (removes batch confound; honest generalization)")
    ap.add_argument("--seed", type=int, default=0)
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
    prog_cols = [classes.index(c) for c in classes if c != QUIESCENT]
    if args.group_split:
        if groups is None:
            raise SystemExit("--group-split needs a 'dataset' column (build with chromatin-combine)")
        uniq = sorted(set(groups))
        rng = np.random.default_rng(args.seed); rng.shuffle(uniq)
        ds_fold = {d: i % args.kfolds for i, d in enumerate(uniq)}
        folds = [np.where(np.array([ds_fold[g] for g in groups]) == f)[0] for f in range(args.kfolds)]
    else:
        folds = _stratified_folds(y, args.kfolds, args.seed)
    print(f"CV | data={Path(args.data).name} n={X.size(0)} k={args.kfolds} mask={args.mask} "
          f"group_split={args.group_split}\n")

    accs, progs, bals = [], [], []
    for f in range(args.kfolds):
        te = torch.tensor(folds[f], dtype=torch.long)
        tr = torch.tensor(np.concatenate([folds[i] for i in range(args.kfolds) if i != f]), dtype=torch.long)
        w = class_weights(y[tr], len(classes)) if args.class_weight else None
        torch.manual_seed(args.seed)
        m = ToggleDynamics(kg, hidden=args.hidden, steps=args.steps).to(dev)
        train(m, X[tr], y[tr], args.epochs, 256, 1e-3, args.seed, weights=w)
        pred = predict(m, X[te])
        acc = float((pred == y[te]).float().mean())
        allrec = [float((pred[y[te] == c] == c).float().mean()) for c in range(len(classes))
                  if int((y[te] == c).sum())]
        progr = [float((pred[y[te] == c] == c).float().mean()) for c in prog_cols
                 if int((y[te] == c).sum())]
        accs.append(acc); bals.append(float(np.mean(allrec))); progs.append(float(np.mean(progr)))
        print(f"  fold {f+1}/{args.kfolds}  acc {acc:.3f}  balanced-acc {bals[-1]:.3f}  program-recall {progs[-1]:.3f}")

    print(f"\n{args.kfolds}-fold CV (mask={args.mask}, class_weight={args.class_weight}):")
    print(f"  overall accuracy   {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"  balanced accuracy  {np.mean(bals):.3f} +/- {np.std(bals):.3f}")
    print(f"  program recall     {np.mean(progs):.3f} +/- {np.std(progs):.3f}")


if __name__ == "__main__":
    main()
