"""Data-scaling law (item: is the model data-limited?).

Holds out a FIXED test set, then trains the multiscale model on training subsets
of increasing size N and measures held-out accuracy + mean program recall vs N.
A rising curve => more data helps (justifies collecting more); a plateau => the
bottleneck is elsewhere. Runs marker-controlled (no_markers) by default so the
curve reflects real signal, not the label shortcut.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .device import pick_device
from .dynamics import ToggleDynamics, _load, _mask_input, train
from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes


@torch.no_grad()
def _eval(model, X, y, classes, prog_cols):
    dev = next(model.parameters()).device
    model.eval()
    pred = torch.cat([model(X[i:i + 1024].to(dev), plasticity=1.0).argmax(-1).cpu()
                      for i in range(0, X.size(0), 1024)])
    acc = float((pred == y).float().mean())
    recs = []
    for c in prog_cols:
        m = y == c
        if int(m.sum()):
            recs.append(float((pred[m] == c).float().mean()))
    return acc, float(np.mean(recs)) if recs else 0.0


def main():
    ap = argparse.ArgumentParser(description="Data-scaling law for the multiscale model")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway.csv"))
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[1000, 2000, 4000, 8000, 16000, 24000])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"], default="no_markers")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--device", default="auto", help="cpu / cuda / mps / auto")
    args = ap.parse_args()
    dev = pick_device(args.device)

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    X = _mask_input(X, kg, args.mask)
    prog_cols = [classes.index(c) for c in classes if c != QUIESCENT]
    print(f"Scaling law | data={Path(args.data).name} n={X.size(0)} mask={args.mask} "
          f"seeds={args.seeds}\n")

    rows = {N: {"acc": [], "prog": []} for N in args.sizes}
    for s in args.seeds:
        g = torch.Generator().manual_seed(s)
        perm = torch.randperm(X.size(0), generator=g)
        nte = int(X.size(0) * args.test_frac)
        te, pool = perm[:nte], perm[nte:]                 # FIXED test set per seed
        for N in args.sizes:
            if N > len(pool):
                continue
            sub = pool[torch.randperm(len(pool), generator=g)[:N]]
            torch.manual_seed(s)
            m = ToggleDynamics(kg, hidden=args.hidden, steps=args.steps).to(dev)
            train(m, X[sub], y[sub], args.epochs, 256, 1e-3, s)
            acc, prog = _eval(m, X[te], y[te], classes, prog_cols)
            rows[N]["acc"].append(acc); rows[N]["prog"].append(prog)
            print(f"  seed {s} N={N:6d}  acc {acc:.3f}  program-recall {prog:.3f}")

    print(f"\n{'train N':>10}{'val acc':>16}{'mean program recall':>22}")
    print("-" * 48)
    Ns, accs, progs = [], [], []
    for N in args.sizes:
        if not rows[N]["acc"]:
            continue
        a, p = np.mean(rows[N]["acc"]), np.mean(rows[N]["prog"])
        sa, sp = np.std(rows[N]["acc"]), np.std(rows[N]["prog"])
        Ns.append(N); accs.append(a); progs.append(p)
        print(f"{N:>10}{f'{a:.3f}±{sa:.3f}':>16}{f'{p:.3f}±{sp:.3f}':>22}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(Ns, accs, "o-", label="overall accuracy")
        ax.plot(Ns, progs, "s-", label="mean program recall")
        ax.set_xscale("log"); ax.set_xlabel("training cells (N)"); ax.set_ylabel("held-out performance")
        ax.set_title(f"Data-scaling law (mask={args.mask})"); ax.legend(); ax.grid(alpha=0.3)
        png = DATA_DIR.parent / "artifacts" / "figures" / "scaling_law.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(png, dpi=150)
        print(f"\nSaved scaling curve -> {png}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
