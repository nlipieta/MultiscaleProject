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

from .device import pick_device
from .dynamics import (ToggleDynamics, _load, _mask_input, train, class_weights,
                       predict, predict_proba)
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


def _metrics(pred, proba, y, n_classes, prog_cols):
    from sklearn.metrics import average_precision_score, f1_score
    pred, y = np.asarray(pred), np.asarray(y)
    acc = float((pred == y).mean())
    recs = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    progr = [float((pred[y == c] == c).mean()) for c in prog_cols if (y == c).any()]
    # macro-F1 over all classes present in this fold (balances precision + recall)
    f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    # macro-AUPRC (threshold-independent): mean over PROGRAM classes present in fold
    aps = []
    if proba is not None:
        for c in prog_cols:
            if (y == c).any():
                aps.append(float(average_precision_score((y == c).astype(int), proba[:, c])))
    auprc = float(np.mean(aps)) if aps else float("nan")
    return acc, float(np.mean(recs)), float(np.mean(progr)), f1, auprc


def _fit_sklearn(kind, Xtr, ytr, Xte, n_classes, class_weight, seed):
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
    # align predict_proba columns (indexed by m.classes_) to full 0..n_classes-1
    proba = np.zeros((Xte.shape[0], n_classes))
    p = m.predict_proba(Xte)
    for j, c in enumerate(m.classes_):
        proba[:, int(c)] = p[:, j]
    return m.predict(Xte), proba


def _fit_gnn(kg, Xtr, ytr, Xte, n_classes, hidden, steps, epochs, class_weight, seed,
             device, attractor=True, no_edges=False, arch="toggle", rcfg=None, bs=256):
    w = class_weights(ytr, n_classes) if class_weight else None
    torch.manual_seed(seed)
    if arch == "resistance":                       # resistance-gated / competence-gated KG-GNN
        from .resistance import ResistanceToggle
        m = ResistanceToggle(kg, hidden=hidden, steps=steps, **(rcfg or {})).to(device)
    else:
        m = ToggleDynamics(kg, hidden=hidden, steps=steps, attractor=attractor).to(device)
    if no_edges:                                   # markers-without-structure control
        m.adjacency.zero_()
    train(m, Xtr, ytr, epochs, bs, 1e-3, seed, weights=w)
    return predict(m, Xte).numpy(), predict_proba(m, Xte).numpy()


def main():
    ap = argparse.ArgumentParser(description="Baseline comparison vs the KG-GNN")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"], default="no_markers")
    ap.add_argument("--group-split", action="store_true", help="hold out whole datasets")
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="multiple seeds -> pooled seed x fold error bars (overrides --seed)")
    ap.add_argument("--epochs", type=int, default=25, help="GNN epochs")
    ap.add_argument("--steps", type=int, default=6, help="GNN message-passing rounds")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--no-gnn", action="store_true", help="skip the (slow) KG-GNN")
    ap.add_argument("--attractor", choices=["on", "off"], default="on",
                    help="WTA attractor sharpening; 'off' = honest graded classifier (no forced fate)")
    ap.add_argument("--structure-test", action="store_true",
                    help="add kg_gnn_noedges (same model, graph removed) to isolate structure's value")
    ap.add_argument("--arch", choices=["toggle", "resistance"], default="toggle",
                    help="GNN family: toggle (original) or resistance (resistance-gated)")
    ap.add_argument("--alpha-memory", choices=["zero", "low", "learned", "full"], default="learned")
    ap.add_argument("--resistance-gate", choices=["on", "off"], default="on")
    ap.add_argument("--plasticity-mode",
                    choices=["amplify", "lower_resistance", "both", "none"], default="lower_resistance")
    ap.add_argument("--attractor-mode",
                    choices=["none", "hard_wta", "soft", "delayed_soft", "learned"], default="soft")
    ap.add_argument("--device", default="auto", help="cpu / cuda / mps / auto")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="GNN minibatch; BIG (1024-4096) is much faster on GPU (launch-bound model)")
    ap.add_argument("--save-folds", default=None, help="write per-(seed,fold) metrics to this CSV")
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

    if args.group_split and groups is None:
        raise SystemExit("--group-split needs a 'dataset' column")

    gnn_models = [] if args.no_gnn else (["kg_gnn", "kg_gnn_noedges"] if args.structure_test else ["kg_gnn"])
    model_list = list(args.models) + gnn_models
    dev = pick_device(args.device)
    attractor = args.attractor == "on"
    rcfg = dict(alpha_memory=args.alpha_memory, resistance=(args.resistance_gate == "on"),
                plasticity_mode=args.plasticity_mode, attractor=args.attractor_mode)
    seeds = args.seeds if args.seeds else [args.seed]
    n_gene_cols = sum(1 for g in kg.gene_map if g in df.columns)  # gene nodes present in DATA
    print(f"Baseline comparison | data={Path(args.data).name} n={X.size(0)} "
          f"k={args.kfolds} mask={args.mask} group_split={args.group_split} "
          f"class_weight={args.class_weight} device={dev} seeds={seeds}")
    print(f"classes={n_classes} gene-columns-in-data={n_gene_cols} "
          f"(~42=narrow, ~148=WIDE) models={model_list}\n")

    Xnp = X.numpy(); ynp = y.numpy()
    scores = {m: {"acc": [], "bal": [], "prog": [], "f1": [], "auprc": []} for m in model_list}
    for s in seeds:                                   # pool over seeds x folds
        folds = (_grouped_folds(groups, args.kfolds, s) if args.group_split
                 else _stratified_folds(y, args.kfolds, s))
        for f in range(args.kfolds):
            te = folds[f]
            tr = np.concatenate([folds[i] for i in range(args.kfolds) if i != f])
            for m in model_list:
                if m.startswith("kg_gnn"):
                    pred, proba = _fit_gnn(kg, X[tr], y[tr], X[te], n_classes, args.hidden,
                                           args.steps, args.epochs, args.class_weight, s, dev,
                                           attractor=attractor, no_edges=m.endswith("noedges"),
                                           arch=args.arch, rcfg=rcfg, bs=args.batch_size)
                else:
                    pred, proba = _fit_sklearn(m, Xnp[tr], ynp[tr], Xnp[te], n_classes,
                                               args.class_weight, s)
                a, b, p, f1, auprc = _metrics(pred, proba, ynp[te], n_classes, prog_cols)
                scores[m]["acc"].append(a); scores[m]["bal"].append(b); scores[m]["prog"].append(p)
                scores[m]["f1"].append(f1); scores[m]["auprc"].append(auprc)
        print(f"  seed {s} done ({args.kfolds} folds)")

    print(f"\n{args.kfolds}-fold CV x {len(seeds)} seed(s) (mask={args.mask}, "
          f"class_weight={args.class_weight}):")
    hdr = (f"{'model':<12}{'overall acc':>16}{'balanced acc':>16}{'prog recall':>16}"
           f"{'macro-F1':>16}{'prog-AUPRC':>16}")
    print(hdr); print("-" * len(hdr))
    def cell(m, k): return f"{np.mean(scores[m][k]):.3f}+/-{np.std(scores[m][k]):.3f}"
    for m in model_list:
        print(f"{m:<12}{cell(m,'acc'):>16}{cell(m,'bal'):>16}{cell(m,'prog'):>16}"
              f"{cell(m,'f1'):>16}{cell(m,'auprc'):>16}")
    print("\nmacro-F1 = precision+recall balance; prog-AUPRC = threshold-independent, "
          "program classes.")

    # optional: persist per-(seed,fold) values for external paired analysis
    if args.save_folds:
        import csv
        n = len(next(iter(scores.values()))["auprc"])
        with open(args.save_folds, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["model", "idx", "acc", "bal", "prog", "f1", "auprc"])
            for m in model_list:
                for i in range(n):
                    w.writerow([m, i] + [scores[m][k][i] for k in ("acc", "bal", "prog", "f1", "auprc")])
        print(f"per-fold metrics -> {args.save_folds}")

    # paired significance: KG-GNN vs each baseline on the SAME seed x fold splits.
    # scores[m][metric] are aligned by (seed,fold) across models, so a paired
    # (Wilcoxon signed-rank) test is valid and controls for fold difficulty.
    if "kg_gnn" in model_list and len(model_list) > 1:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            wilcoxon = None
        print("\nPaired significance vs KG-GNN (Wilcoxon signed-rank over seed x fold):")
        print(f"{'baseline':<12}{'metric':>10}{'median dP(gnn-base)':>22}{'p-value':>12}")
        print("-" * 56)
        g = scores["kg_gnn"]
        for m in model_list:
            if m == "kg_gnn":
                continue
            for k in ("auprc", "prog"):
                d = np.array(g[k]) - np.array(scores[m][k])
                med = float(np.median(d))
                if wilcoxon is not None and np.any(d != 0):
                    try:
                        p = float(wilcoxon(g[k], scores[m][k]).pvalue)
                    except ValueError:
                        p = float("nan")
                else:
                    p = float("nan")
                print(f"{m:<12}{k:>10}{med:>+22.3f}{p:>12.4f}")
        print("positive median = KG-GNN higher; p<0.05 = the paired difference is significant.")


if __name__ == "__main__":
    main()
