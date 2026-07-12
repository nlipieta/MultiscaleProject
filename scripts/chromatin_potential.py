"""Chromatin-potential test: does the ATAC landscape predict erythroid commitment EARLIER / better
than RNA -- the non-redundant, thesis-defining signal (Ma 2020 'chromatin potential': accessibility
of lineage genes precedes their expression, so the epigenetic landscape forecasts fate before RNA).

On GSE207308 SHARE-seq (stage: HSC/MPP=0 -> early-Ery=1 -> late-Ery=2):

 1. DESCRIPTIVE lead-lag (no model): per-cell erythroid ACCESSIBILITY score vs EXPRESSION score
    (mean over the Erythropoiesis network genes), by stage. Chromatin potential => accessibility
    is 'ahead' of expression (early-Ery accessibility approaches late-Ery expression).

 2. MODEL predictive priority: same 2-channel net, three input maskings --
      RNA-only  (accessibility zeroed) | ATAC-only (expression zeroed) | both
    predict P(Erythropoiesis), stratified by stage. Chromatin potential => ATAC-only assigns
    HIGHER P(Ery) to EARLY-Ery cells than RNA-only does (accessibility already 'knows' the fate).

This isolates what ATAC adds BEYOND RNA (predictive priority), not redundant co-prediction of state.

Run:  uv run python scripts/chromatin_potential.py --data data/bmmc_shareseq.csv --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import class_weights
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import all_classes
from chromatin_toggle.resistance import ResistanceToggle


def _load(path, kg):
    df = pd.read_csv(path)
    classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    node_cols = list(kg.node_ids)
    X = np.zeros((len(df), len(node_cols)), np.float32); A = np.zeros_like(X)
    for j, c in enumerate(node_cols):
        if c in df.columns: X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if f"{c}__atac" in df.columns: A[:, j] = pd.to_numeric(df[f"{c}__atac"], errors="coerce").fillna(0)
    y = np.array([ci[l] for l in df["label"]])
    stage = pd.to_numeric(df["timepoint"], errors="coerce").to_numpy()
    return torch.tensor(X), torch.tensor(A), torch.tensor(y), stage, classes


def _strat_folds(y, k, seed):
    rng = np.random.default_rng(seed); folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        for i, j in enumerate(idx): folds[i % k].append(int(j))
    return [np.array(f) for f in folds]


def _mask(X, A, use_x, use_a):
    return (X if use_x else torch.zeros_like(X)), (A if use_a else torch.zeros_like(A))


def _train(kg, X, A, y, dev, epochs, bs, lr, seed, use_x, use_a):
    torch.manual_seed(seed)
    m = ResistanceToggle(kg, hidden=64, steps=6, use_atac=True).to(dev)   # 2-channel; mask inputs per arm
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=class_weights(y, len(all_classes(kg))).to(dev))
    Xt, At = _mask(X, A, use_x, use_a); Xt, At, y = Xt.to(dev), At.to(dev), y.to(dev)
    g = torch.Generator().manual_seed(seed); n = Xt.size(0)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            lossf(m(Xt[idx], atac=At[idx]), y[idx]).backward(); opt.step()
        sched.step()
    return m


@torch.no_grad()
def _p(m, X, A, dev, prog_i, use_x, use_a, bs=2048):
    m.eval(); Xt, At = _mask(X, A, use_x, use_a)
    return torch.cat([torch.softmax(m(Xt[i:i+bs].to(dev), atac=At[i:i+bs].to(dev)), -1)[:, prog_i].cpu()
                      for i in range(0, X.size(0), bs)]).numpy()


def main():
    ap = argparse.ArgumentParser(description="chromatin-potential test (ATAC predictive priority over RNA)")
    ap.add_argument("--data", default=str(DATA_DIR / "bmmc_shareseq.csv"))
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    kg = load_kg(); dev = pick_device(a.device)
    X, A, y, stage, classes = _load(a.data, kg)
    prog_i = classes.index("Erythropoiesis")
    stages = sorted(set(stage))

    # --- 1. descriptive lead-lag: erythroid accessibility vs expression score by stage ---
    memb = np.zeros(kg.num_nodes)                        # Erythropoiesis-network genes (regulators+markers)
    for s, _r, d, _w in kg.edges:
        if d == kg.node_index.get("Erythropoiesis"): memb[s] = 1.0
    ery_nodes = np.where(memb > 0)[0]
    Xn, An = X.numpy(), A.numpy()
    print(f"chromatin-potential | cells={len(stage)} stages={stages} erythroid-network genes={len(ery_nodes)} dev={dev}\n")
    print("1) DESCRIPTIVE lead-lag (erythroid-network score, mean +/- over cells) by stage:")
    print(f"   {'stage':>6}{'n':>7}{'ATAC score':>13}{'RNA score':>12}")
    for st in stages:
        sel = stage == st
        print(f"   {st:>6.0f}{int(sel.sum()):>7}{An[sel][:, ery_nodes].mean():>13.3f}{Xn[sel][:, ery_nodes].mean():>12.3f}")
    print("   chromatin potential => ATAC score rises BEFORE RNA score (accessibility leads).\n")

    # --- 2. model predictive priority: RNA-only vs ATAC-only vs both ---
    arms = [("RNA-only", True, False), ("ATAC-only", False, True), ("both", True, True)]
    print("2) MODEL P(Erythropoiesis) by stage, and AUPRC, per input channel:")
    for arm, ux, ua in arms:
        Pcell = np.zeros(len(stage)); auprc = []
        for s in a.seeds:
            folds = _strat_folds(y.numpy(), a.kfolds, s)
            for f in range(a.kfolds):
                te = folds[f]; tr = np.concatenate([folds[i] for i in range(a.kfolds) if i != f])
                m = _train(kg, X[tr], A[tr], y[tr], dev, a.epochs, a.batch_size, a.lr, s, ux, ua)
                Pcell[te] += _p(m, X[te], A[te], dev, prog_i, ux, ua) / len(a.seeds)
                yb = y[te].numpy()
                auprc.append(average_precision_score((yb == prog_i).astype(int),
                             _p(m, X[te], A[te], dev, prog_i, ux, ua)))
        by_stage = "  ".join(f"s{int(st)}={Pcell[stage==st].mean():.3f}" for st in stages)
        rho, _ = spearmanr(stage, Pcell)
        print(f"   {arm:<10} AUPRC {np.mean(auprc):.3f}  P(Ery) {by_stage}  Spearman(P,stage) {rho:+.3f}")
    print("\nchromatin potential => ATAC-only assigns HIGHER P(Ery) to EARLY-Ery (s1) cells than RNA-only")
    print("(accessibility forecasts commitment before expression). 'both' >= max(single) => complementary.")


if __name__ == "__main__":
    main()
