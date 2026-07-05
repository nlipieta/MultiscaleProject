"""Baseline comparison (reviewer item #2: expand the baselines).

The KG-GNN's headline number is only meaningful against strong, theory-agnostic
learners on the SAME features and the SAME honest splits. This runs, per fold of
a grouped k-fold CV (whole datasets held out, so no batch leakage):

  * majority   -- DummyClassifier (predicts the most frequent class)
  * logreg     -- multinomial logistic regression (linear baseline)
  * rforest    -- random forest (non-linear, feature-interaction baseline)
  * gboost     -- gradient boosting (strong tabular baseline; sklearn, no xgboost dep)
  * kg_gnn     -- the ToggleDynamics multiscale model over the literature KG

All models see the identical masked node-vector (default no_markers, so the
program-marker shortcut is removed for everyone). Reported per model: overall
accuracy, balanced accuracy (macro recall over all classes) and program recall
(mean recall over the non-Quiescent programs -- the number that is not inflated
by the easy majority Quiescent class).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .dynamics import ToggleDynamics, _load, _mask_input, train, class_weights
from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes


def _grouped_folds(groups, k, seed):
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    ds_fold = {d: i % k for i, d in enumerate(uniq)}
    return [np.where(np.array([ds_fold[g] for g in groups]) == f)[0] for f in range(k)]


def _stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in torch.unique(y).tolist():
        idx = np.where(y.numpy() == c)[0]; rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(int(j))
    return [np.array(f) for f in folds]


def _metrics(pred, y, n_classes, prog_cols):
    pred, y = np.asarray(pred), np.asarray(y)
    acc = float((pred == y).mean())
    recs = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    progr = [float((pred[y == c] == c).mean()) for c in prog_cols if (y == c).any()]
    return acc, float(np.mean(recs)), float(np.mean(progr))


def _fit_sklearn(kind, Xtr, ytr, Xte, class_weight, seed):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    cw = "balanced" if class_weight else None
    if kind == "majority":
        m = DummyClassifier(strategy="most_frequent")
    elif kind == "logreg":
        m = LogisticRegression(max_iter=2000, class_weight=cw, C=1.0)
    elif kind == "rforest":
        m = RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=seed, n_jobs=-1)
    elif kind == "gboost":
        m = GradientBoostingClassifier(random_state=seed)  # no native class_weight
    else:
        raise ValueError(kind)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def _fit_gnn(kg, Xtr, ytr, Xte, n_classes, hidden, steps, epochs, class_weight, seed):
    w = class_weights(ytr, n_classes) if class_weight else None
    torch.manual_seed(seed)
    m = ToggleDynamics(kg, hidden=hidden, steps=steps)
    train(m, Xtr, ytr, epochs, 256, 1e-3, seed, weights=w)
    m.eval()
    with torch.no_grad():
        return torch.cat([m(Xte[i:i+1024], plasticity=1.0).argmax(-1)
                          for i in range(0, Xte.size(0), 1024)]).numpy()


def main():
    ap = argparse.ArgumentParser(description="Baseline comparison vs the KG-GNN")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"], default="no_markers")
    ap.add_argument("--group-split", action="store_true", help="hold out whole datasets")
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25, help="GNN epochs")
    ap.add_argument("--steps", type=int, default=6, help="GNN message-passing rounds")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--no-gnn", action="store_true", help="skip the (slow) KG-GNN")
    ap.add_argument("--models", nargs="*",
                    default=["majority", "logreg", "rforest", "gboost"],
                    help="sklearn baselines to run")
    args = ap.parse_args()

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

    if args.group_split:
        if groups is None:
            raise SystemExit("--group-split needs a 'dataset' column")
        folds = _grouped_folds(groups, args.kfolds, args.seed)
    else:
        folds = _stratified_folds(y, args.kfolds, args.seed)

    model_list = list(args.models) + ([] if args.no_gnn else ["kg_gnn"])
    print(f"Baseline comparison | data={Path(args.data).name} n={X.size(0)} "
          f"k={args.kfolds} mask={args.mask} group_split={args.group_split} "
          f"class_weight={args.class_weight}")
    print(f"classes={n_classes} models={model_list}\n")

    Xnp = X.numpy(); ynp = y.numpy()
    scores = {m: {"acc": [], "bal": [], "prog": []} for m in model_list}
    for f in range(args.kfolds):
        te = folds[f]
        tr = np.concatenate([folds[i] for i in range(args.kfolds) if i != f])
        for m in model_list:
            if m == "kg_gnn":
                pred = _fit_gnn(kg, X[tr], y[tr], X[te], n_classes, args.hidden,
                                args.steps, args.epochs, args.class_weight, args.seed)
            else:
                pred = _fit_sklearn(m, Xnp[tr], ynp[tr], Xnp[te], args.class_weight, args.seed)
            a, b, p = _metrics(pred, ynp[te], n_classes, prog_cols)
            scores[m]["acc"].append(a); scores[m]["bal"].append(b); scores[m]["prog"].append(p)
        print(f"  fold {f+1}/{args.kfolds} done")

    print(f"\n{args.kfolds}-fold CV (mask={args.mask}, class_weight={args.class_weight}):")
    print(f"{'model':<12}{'overall acc':>16}{'balanced acc':>16}{'program recall':>18}")
    print("-" * 62)
    for m in model_list:
        a = f"{np.mean(scores[m]['acc']):.3f}+/-{np.std(scores[m]['acc']):.3f}"
        b = f"{np.mean(scores[m]['bal']):.3f}+/-{np.std(scores[m]['bal']):.3f}"
        p = f"{np.mean(scores[m]['prog']):.3f}+/-{np.std(scores[m]['prog']):.3f}"
        print(f"{m:<12}{a:>16}{b:>16}{p:>18}")


if __name__ == "__main__":
    main()
