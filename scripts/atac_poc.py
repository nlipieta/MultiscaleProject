"""scATAC proof-of-concept: does a paired chromatin-accessibility channel improve the
resistance-gated model on the BMMC Multiome pool (Erythropoiesis / Megakaryopoiesis / Quiescent)?

Ablation with IDENTICAL model capacity (use_atac=True in both arms; the 2nd input channel is
fed real accessibility vs zeros), stratified k-fold, paired Wilcoxon over folds:
  +ATAC       -- node input [expression, real accessibility]
  RNA-only    -- node input [expression, ZEROS]  (same 2-channel net, accessibility withheld)

Usage:  uv run python scripts/atac_poc.py --data data/bmmc_multiome.csv --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import class_weights
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import QUIESCENT, all_classes
from chromatin_toggle.resistance import ResistanceToggle


def _load(path, kg):
    df = pd.read_csv(path)
    classes = all_classes(kg)
    ci = {c: i for i, c in enumerate(classes)}
    node_cols = list(kg.node_ids)
    X = np.zeros((len(df), len(node_cols)), dtype=np.float32)
    A = np.zeros((len(df), len(node_cols)), dtype=np.float32)
    for j, c in enumerate(node_cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
        ac = f"{c}__atac"
        if ac in df.columns:
            A[:, j] = pd.to_numeric(df[ac], errors="coerce").fillna(0).to_numpy()
    y = np.array([ci[l] for l in df["label"]])
    return torch.tensor(X), torch.tensor(A), torch.tensor(y), classes


def _strat_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(int(j))
    return [np.array(f) for f in folds]


def _train(kg, X, A, y, n_classes, w, dev, epochs, bs, lr, seed, use_real_atac, plast_source="const"):
    torch.manual_seed(seed)
    m = ResistanceToggle(kg, hidden=64, steps=6, use_atac=True, plasticity_source=plast_source).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=w.to(dev))
    X, y = X.to(dev), y.to(dev)
    A = A.to(dev) if use_real_atac else torch.zeros_like(X).to(dev)
    g = torch.Generator().manual_seed(seed); n = X.size(0)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(m(X[idx], atac=A[idx]), y[idx])
            loss.backward(); opt.step()
        sched.step()
    return m


@torch.no_grad()
def _plast_std(m, X, A, dev):
    """Per-cell plasticity spread in the ATAC-plasticity arm (>0 => mechanism ACTIVE, not dormant)."""
    if getattr(m, "plasticity_source", "const") != "atac":
        return float("nan")
    pa = A.to(dev) @ m.prog_membership.t()
    return torch.sigmoid(m.plast_from_atac(pa)).std().item()


@torch.no_grad()
def _eval(m, X, A, y, dev, prog_cols, use_real_atac, bs=1024):
    from sklearn.metrics import average_precision_score
    m.eval()
    A = A if use_real_atac else torch.zeros_like(X)
    proba = torch.cat([torch.softmax(m(X[i:i+bs].to(dev), atac=A[i:i+bs].to(dev)), -1).cpu()
                       for i in range(0, X.size(0), bs)]).numpy()
    pred = proba.argmax(1); yn = y.numpy()
    recs = [float((pred[yn == c] == c).mean()) for c in prog_cols if (yn == c).any()]
    aps = [float(average_precision_score((yn == c).astype(int), proba[:, c]))
           for c in prog_cols if (yn == c).any()]
    return float(np.mean(aps)), float(np.mean(recs))


def main():
    ap = argparse.ArgumentParser(description="scATAC POC: accessibility channel on vs off")
    ap.add_argument("--data", default=str(DATA_DIR / "bmmc_multiome.csv"))
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    kg = load_kg(); dev = pick_device(args.device)
    X, A, y, classes = _load(args.data, kg)
    n_classes = len(classes)
    prog_cols = [classes.index(c) for c in classes if c != QUIESCENT]
    n_atac = int((A.abs().sum(0) > 0).sum())
    print(f"scATAC POC | data={Path(args.data).name} cells={X.size(0)} classes={n_classes} "
          f"| accessibility-bearing nodes={n_atac} | device={dev} seeds={args.seeds}\n")

    # three arms: accessibility withheld / accessibility as INPUT / accessibility routed into PLASTICITY
    arms = [("RNA-only", False, "const"), ("+ATAC", True, "const"), ("+ATAC+plast", True, "atac")]
    scores = {a[0]: {"auprc": [], "prog": []} for a in arms}
    plast_std = []
    for s in args.seeds:
        folds = _strat_folds(y.numpy(), args.kfolds, s)
        for f in range(args.kfolds):
            te = folds[f]; tr = np.concatenate([folds[i] for i in range(args.kfolds) if i != f])
            w = class_weights(y[tr], n_classes)
            for arm, real, ps in arms:
                m = _train(kg, X[tr], A[tr], y[tr], n_classes, w, dev,
                           args.epochs, args.batch_size, args.lr, s, real, plast_source=ps)
                au, pr = _eval(m, X[te], A[te], y[te], dev, prog_cols, real)
                scores[arm]["auprc"].append(au); scores[arm]["prog"].append(pr)
                if ps == "atac":
                    plast_std.append(_plast_std(m, X[te], A[te], dev))
        print(f"  seed {s} done", flush=True)

    print(f"\n{args.kfolds}-fold x {len(args.seeds)} seed(s):")
    for arm, _, _ in arms:
        a, p = scores[arm]["auprc"], scores[arm]["prog"]
        print(f"  {arm:<12} prog-AUPRC {np.mean(a):.3f}+/-{np.std(a):.3f}   prog-recall {np.mean(p):.3f}+/-{np.std(p):.3f}")
    if plast_std:
        print(f"  ATAC-plasticity per-cell std = {np.mean(plast_std):.4f}  (>0 => plasticity gate is now ACTIVE)")
    try:
        from scipy.stats import wilcoxon
        def paired(aarm, barm):
            for k in ("auprc", "prog"):
                da, db = scores[aarm][k], scores[barm][k]
                d = np.array(da) - np.array(db)
                p = float(wilcoxon(da, db).pvalue) if np.any(d) else float("nan")
                print(f"  paired {aarm} vs {barm} [{k}]: median dP={np.median(d):+.3f}  p={p:.4f}")
        paired("+ATAC", "RNA-only")            # does accessibility as an INPUT help?
        paired("+ATAC+plast", "+ATAC")         # does routing ATAC into PLASTICITY (the thesis) add more?
    except ImportError:
        pass
    print("\n+ATAC vs RNA-only = accessibility as an input channel; +ATAC+plast vs +ATAC = the thesis")
    print("mechanism (plasticity driven by chromatin openness across pathway + alternative networks).")


if __name__ == "__main__":
    main()
