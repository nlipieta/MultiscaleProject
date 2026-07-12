"""Erythroid-commitment TRAJECTORY test for ATAC-driven plasticity (GSE207308 SHARE-seq BMMC).

The saturated binary POC couldn't judge the plasticity mechanism (AUPRC pinned at 1.0). This tests
it on the RIGHT axis: a TRANSITION. Cells carry a maturation stage (HSC/MPP=0 -> early-Ery=1 ->
late-Ery=2). The non-circular question: among cells ALL labelled Erythropoiesis (early+late), does
predicted P(Erythropoiesis) RISE from early->late (graded commitment beyond the binary label)? And
does routing chromatin accessibility into PLASTICITY improve that graded signal?

3 arms x stratified k-fold, predictions on held-out cells:
  RNA-only     -- accessibility withheld (2nd channel zeroed)
  +ATAC        -- accessibility as an input channel (const plasticity)
  +ATAC+plast  -- accessibility ALSO drives plasticity (the thesis mechanism)
Metric per arm: Spearman(P(Erythro), stage) among ERYTHROID held-out cells (non-circular), plus
all-cells; AUPRC and per-cell plasticity std (mechanism-active check).

Run:  uv run python scripts/trajectory_atac.py --data data/bmmc_shareseq.csv --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import class_weights
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import QUIESCENT, all_classes
from chromatin_toggle.resistance import ResistanceToggle


def _load(path, kg):
    df = pd.read_csv(path)
    classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    node_cols = list(kg.node_ids)
    X = np.zeros((len(df), len(node_cols)), np.float32)
    A = np.zeros((len(df), len(node_cols)), np.float32)
    for j, c in enumerate(node_cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if f"{c}__atac" in df.columns:
            A[:, j] = pd.to_numeric(df[f"{c}__atac"], errors="coerce").fillna(0)
    y = np.array([ci[l] for l in df["label"]])
    stage = pd.to_numeric(df["timepoint"], errors="coerce").to_numpy()
    return torch.tensor(X), torch.tensor(A), torch.tensor(y), stage, classes


def _strat_folds(y, k, seed):
    rng = np.random.default_rng(seed); folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        for i, j in enumerate(idx): folds[i % k].append(int(j))
    return [np.array(f) for f in folds]


def _train(kg, X, A, y, dev, epochs, bs, lr, seed, real, plast):
    torch.manual_seed(seed)
    m = ResistanceToggle(kg, hidden=64, steps=6, use_atac=True, plasticity_source=plast).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights(y, len(all_classes(kg))).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    X, y = X.to(dev), y.to(dev)
    A = (A if real else torch.zeros_like(A)).to(dev)
    g = torch.Generator().manual_seed(seed); n = X.size(0)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            lossf(m(X[idx], atac=A[idx]), y[idx]).backward(); opt.step()
        sched.step()
    return m


@torch.no_grad()
def _p(m, X, A, dev, prog_i, real, bs=2048):
    m.eval(); A = A if real else torch.zeros_like(A)
    return torch.cat([torch.softmax(m(X[i:i+bs].to(dev), atac=A[i:i+bs].to(dev)), -1)[:, prog_i].cpu()
                      for i in range(0, X.size(0), bs)]).numpy()


@torch.no_grad()
def _plast_std(m, A, dev):
    if getattr(m, "plasticity_source", "const") != "atac":
        return float("nan")
    return m._atac_plasticity(A.to(dev)).std().item()


def main():
    ap = argparse.ArgumentParser(description="erythroid-trajectory ATAC-plasticity test")
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
    ery = y.numpy() == prog_i
    from sklearn.metrics import average_precision_score
    print(f"trajectory ATAC test | cells={X.size(0)} erythroid={int(ery.sum())} "
          f"stages={sorted(set(stage))} device={dev}\n")

    arms = [("RNA-only", False, "const"), ("+ATAC", True, "const"), ("+ATAC+plast", True, "atac")]
    for arm, real, ps in arms:
        P = np.zeros(X.size(0)); auprc = []; pstd = []
        for s in a.seeds:
            folds = _strat_folds(y.numpy(), a.kfolds, s)
            for f in range(a.kfolds):
                te = folds[f]; tr = np.concatenate([folds[i] for i in range(a.kfolds) if i != f])
                m = _train(kg, X[tr], A[tr], y[tr], dev, a.epochs, a.batch_size, a.lr, s, real, ps)
                P[te] += _p(m, X[te], A[te], dev, prog_i, real) / len(a.seeds)
                yb = y[te].numpy()
                auprc.append(average_precision_score((yb == prog_i).astype(int), _p(m, X[te], A[te], dev, prog_i, real)))
                if ps == "atac": pstd.append(_plast_std(m, A[te], dev))
        rho_ery, p_ery = spearmanr(stage[ery], P[ery])            # non-circular: within erythroid
        rho_all, p_all = spearmanr(stage, P)
        tail = f"  plast-std {np.mean(pstd):.4f}" if pstd else ""
        print(f"  {arm:<12} AUPRC {np.mean(auprc):.3f}  |  Spearman(P(Ery),stage) "
              f"erythroid-only rho={rho_ery:+.3f} (p={p_ery:.1e})  all rho={rho_all:+.3f}{tail}")
    print("\nerythroid-only rho = graded commitment early->late among cells all labelled Erythropoiesis")
    print("(the non-circular test). Compare +ATAC+plast vs +ATAC: does chromatin-driven plasticity")
    print("sharpen the graded commitment signal? (all-cells rho partly reflects the Quiescent baseline.)")


if __name__ == "__main__":
    main()
