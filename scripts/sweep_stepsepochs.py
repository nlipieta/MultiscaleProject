"""Accuracy-vs-(steps, epochs) sweep: find the cheapest config that doesn't hurt the result.

Runtime = per-step x STEPS x EPOCHS x (folds*seeds). steps=8 / epochs=120 may be past what the
model needs; cutting them is linear speedup at zero method risk -- IF prog-AUPRC and the
structure gap (kg_gnn vs edge-removed) hold. This measures that.

Efficient: for each `steps`, trains ONCE to max(epochs) and evaluates at every epoch checkpoint
(so the epochs axis is free), for both the full model and its edge-removed twin. Grouped split
(whole datasets held out) so it reflects the real generalization regime.

Run:  uv run python scripts/sweep_stepsepochs.py --device cuda --amp
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import _load, _mask_input, class_weights, predict_proba
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import QUIESCENT, all_classes
from chromatin_toggle.resistance import ResistanceToggle


def _grouped_split(groups, val_frac, seed):
    uniq = sorted(set(groups)); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    vset = set(uniq[:max(1, int(len(uniq) * val_frac))])
    isval = np.array([g in vset for g in groups])
    return np.where(~isval)[0], np.where(isval)[0]


def _auprc(model, Xte, yte, prog_cols):
    proba = predict_proba(model, Xte).numpy()
    y = yte.numpy()
    aps = [average_precision_score((y == c).astype(int), proba[:, c])
           for c in prog_cols if (y == c).any()]
    return float(np.mean(aps)) if aps else float("nan")


def train_eval(kg, Xtr, ytr, Xte, yte, hidden, steps, ckpts, bs, lr, seed, dev,
               n_classes, prog_cols, w, no_edges, amp):
    torch.manual_seed(seed)
    m = ResistanceToggle(kg, hidden=hidden, steps=steps).to(dev)
    if no_edges:
        m.adjacency.zero_()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=w.to(dev))
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)
    g = torch.Generator().manual_seed(seed)
    n = Xtr.size(0); stop = (n // bs) * bs if n >= bs else n
    out = {}
    for ep in range(1, max(ckpts) + 1):
        m.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, stop, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = lossf(m(Xtr[idx]), ytr[idx])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if ep in ckpts:
            out[ep] = _auprc(m, Xte, yte, prog_cols)
    return out


def main():
    ap = argparse.ArgumentParser(description="steps x epochs accuracy sweep (structure gap tracked)")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--mask", default="no_markers")
    ap.add_argument("--subsample", type=int, default=6000)
    ap.add_argument("--steps-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--epoch-ckpts", type=int, nargs="+", default=[40, 80, 120])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    a = ap.parse_args()
    dev = pick_device(a.device)

    kg = load_kg()
    X, y, classes, df = _load(a.data, kg)
    X = _mask_input(X, kg, a.mask)
    groups = df["dataset"].to_numpy()
    if a.subsample and X.size(0) > a.subsample:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:a.subsample]
        X, y, groups = X[idx], y[idx], groups[idx.numpy()]
    tr, te = _grouped_split(groups, a.val_frac, a.seed)
    Xtr, ytr = X[tr].to(dev), y[tr].to(dev)
    Xte, yte = X[te].to(dev), y[te]
    n_classes = len(classes)
    prog_cols = [classes.index(c) for c in all_classes(kg) if c != QUIESCENT and c in classes]
    w = class_weights(ytr.cpu(), n_classes)
    print(f"device={dev} n={X.size(0)} (train {len(tr)}/test {len(te)}) mask={a.mask} amp={a.amp}")
    print(f"prog-AUPRC (full model) and structure gap (full - edge-removed), by steps x epoch:\n")

    for steps in a.steps_list:
        t = time.perf_counter()
        full = train_eval(kg, Xtr, ytr, Xte, yte, a.hidden, steps, a.epoch_ckpts, a.bs, a.lr,
                          a.seed, dev, n_classes, prog_cols, w, no_edges=False, amp=a.amp)
        none = train_eval(kg, Xtr, ytr, Xte, yte, a.hidden, steps, a.epoch_ckpts, a.bs, a.lr,
                          a.seed, dev, n_classes, prog_cols, w, no_edges=True, amp=a.amp)
        dt = time.perf_counter() - t
        cells = "  ".join(f"ep{ep}: {full[ep]:.3f} (gap {full[ep]-none[ep]:+.3f})"
                          for ep in a.epoch_ckpts)
        print(f"steps={steps}:  {cells}   [{dt:.0f}s]")
    print("\nPick the smallest (steps, epochs) where AUPRC and the gap match steps=8/ep=120.")


if __name__ == "__main__":
    main()
