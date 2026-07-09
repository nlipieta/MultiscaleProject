"""In-silico perturbation test for the hypertrophy case study (reviewer #5).

Trains the multiscale model on the cross-pathway pool, then edits node inputs on
held-out Hypertrophy cells to test whether the model's learned representation
respects the encoded CaMKII -> HDAC4/5-export -> MEF2 de-repression logic:

  HDAC4/5 knockdown (zero HDAC4,HDAC5 inputs)  -> predict P(Hypertrophy) UP
                                                  (repressor removed; cue-independent)
  CaMKII block      (zero CaMKII input)        -> predict P(Hypertrophy) DOWN
  CaMKII boost      (CaMKII input -> 1)        -> predict P(Hypertrophy) UP

A marker-memorizing model would not show this coherent, sign-correct directionality.
Reports mean ΔP(Hypertrophy) per intervention and whether the SIGN matches prediction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .device import pick_device
from .resistance import ResistanceToggle
from .dynamics import _load, _mask_input, train, class_weights
from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes


@torch.no_grad()
def _phyper(model, X, hyper_i, bs=1024):
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, X.size(0), bs):
        p = torch.softmax(model(X[i:i+bs].to(dev), plasticity=1.0), dim=-1)[:, hyper_i]
        out.append(p.cpu())
    return torch.cat(out)


def _edit(X, kg, zero=(), setto=None):
    X = X.clone()
    for n in zero:
        if n in kg.node_index:
            X[:, kg.node_index[n]] = 0.0
    if setto:
        for n, v in setto.items():
            if n in kg.node_index:
                X[:, kg.node_index[n]] = v
    return X


def main():
    ap = argparse.ArgumentParser(description="Hypertrophy in-silico perturbation test")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"], default="no_markers")
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cpu / cuda / mps / auto")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="GNN minibatch; BIG (1024-4096) is much faster on GPU (launch-bound model)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(reduce-overhead) the GNN training forward (cuda only)")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 mixed precision (tensor cores) -- ~1.9x on T4 (cuda only)")
    args = ap.parse_args()
    dev = pick_device(args.device)

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    X = _mask_input(X, kg, args.mask)
    if args.subsample and X.size(0) > args.subsample:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:args.subsample]
        X, y = X[idx], y[idx]
        df = df.iloc[idx.numpy()].reset_index(drop=True)
    n_classes = len(classes)
    hyper_i = classes.index("Hypertrophy")

    # hold out a Hypertrophy dataset entirely (honest: model never saw these cells)
    groups = df["dataset"].to_numpy()
    hyper_ds = sorted({g for g, lab in zip(groups, [classes[i] for i in y])
                       if lab == "Hypertrophy"})
    held = hyper_ds[0] if hyper_ds else None
    te = np.where(groups == held)[0]
    tr = np.where(groups != held)[0]
    print(f"held-out hypertrophy dataset: {held}  (train n={len(tr)}, test n={len(te)})")

    w = class_weights(y[torch.tensor(tr)], n_classes)
    torch.manual_seed(args.seed)
    m = ResistanceToggle(kg, hidden=args.hidden, steps=args.steps).to(dev)
    train(m, X[tr], y[tr], args.epochs, args.batch_size, 1e-3, args.seed, weights=w,
          compile=args.compile, amp=args.amp)

    Xte = X[torch.tensor(te)]
    yte = y[torch.tensor(te)].numpy()
    hyper_cells = Xte[yte == hyper_i]
    print(f"Hypertrophy test cells: {len(hyper_cells)}\n")

    base = _phyper(m, hyper_cells, hyper_i).mean().item()
    interventions = [
        ("HDAC4/5 knockdown", dict(zero=("HDAC4", "HDAC5")), "UP"),
        ("CaMKII block",      dict(zero=("CaMKII",)),         "DOWN"),
        ("CaMKII boost",      dict(setto={"CaMKII": 1.0}),    "UP"),
        ("PKD block",         dict(zero=("PKD",)),            "DOWN"),
    ]
    print(f"baseline mean P(Hypertrophy) on held-out hypertrophy cells: {base:.3f}\n")
    print(f"{'intervention':<20}{'ΔP(Hyper)':>12}{'predicted':>11}{'sign match':>12}")
    print("-" * 55)
    for name, edit, pred in interventions:
        Xe = _edit(hyper_cells, kg, **edit)
        p = _phyper(m, Xe, hyper_i).mean().item()
        d = p - base
        got = "UP" if d > 0 else "DOWN"
        match = "yes" if got == pred else "NO"
        print(f"{name:<20}{d:>+12.3f}{pred:>11}{match:>12}")
    print("\nSign-correct directionality = the learned representation respects the "
          "encoded\nCaMKII->HDAC4/5->MEF2 de-repression logic (not marker memorization).")


if __name__ == "__main__":
    main()
