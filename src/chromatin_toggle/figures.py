"""Render the remaining publication figures.

1. main_result_confusion.png -- held-out confusion matrix over the 10 programs
   (marker-controlled), the "model predicts meaningful cell states" figure.
2. baseline_comparison.png   -- cross-species Hypertrophy generalization,
   KG-GNN vs shuffled-KG vs bag-of-genes MLP (3-seed means, from LOPO).
3. representation_control.png -- full / no_markers / lineage_only program recall
   (3-seed means), showing the shortcut is dissolved and cue+memory predicts.

The bar-chart numbers are the 3-seed results measured via chromatin-eval /
chromatin-classify; the confusion matrix is trained fresh here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .resistance import ResistanceToggle
from .dynamics import _load, _mask_input, train, class_weights
from .kg import DATA_DIR, load_kg
from .oracle import all_classes

# 3-seed results measured earlier (mean, std)
LOPO_XSPECIES = {"KG-GNN": (0.977, 0.009), "shuffled-KG": (0.007, 0.009), "gene MLP": (0.057, 0.037)}
REPR_CONTROL = {"full": (0.476, 0.058), "no_markers": (0.441, 0.016), "lineage_only": (0.344, 0.064)}
FIGS = DATA_DIR.parent / "artifacts" / "figures"


def _confusion(data, subN, epochs, steps, seed):
    kg = load_kg()
    X, y, classes, df = _load(data, kg)
    X = _mask_input(X, kg, "no_markers")
    if subN and X.size(0) > subN:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(seed))[:subN]
        X, y = X[idx], y[idx]
    perm = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(seed + 1))
    k = int(X.size(0) * 0.25)
    va, tr = perm[:k], perm[k:]
    m = ResistanceToggle(kg, hidden=64, steps=steps)  # improved config
    train(m, X[tr], y[tr], epochs, 256, 1e-3, seed, weights=class_weights(y[tr], len(classes)))
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(X[va][i:i+1024], plasticity=1.0).argmax(-1) for i in range(0, len(va), 1024)])
    C = np.zeros((len(classes), len(classes)))
    for t, p in zip(y[va].tolist(), pred.tolist()):
        C[t, p] += 1
    Cn = C / C.sum(1, keepdims=True).clip(min=1)
    return Cn, classes, float((pred == y[va]).float().mean())


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Render publication figures")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--subsample", type=int, default=9000)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)

    # 1. confusion matrix (main result)
    C, classes, acc = _confusion(args.data, args.subsample, args.epochs, args.steps, 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Held-out confusion (no_markers), acc={acc:.2f}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if C[i, j] > 0.02:
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if C[i, j] > 0.5 else "black")
    fig.colorbar(im, shrink=0.7, label="row-normalized fraction")
    fig.tight_layout(); fig.savefig(FIGS / "main_result_confusion.png", dpi=150); plt.close(fig)

    # 2 & 3. bar charts from measured 3-seed numbers
    for fname, title, d, ylab in [
        ("baseline_comparison.png", "Cross-species Hypertrophy transfer (activated recall)", LOPO_XSPECIES, "activated recall"),
        ("representation_control.png", "Marker-shortcut control (mean program recall, 3 seeds)", REPR_CONTROL, "mean program recall"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ks = list(d.keys()); means = [d[k][0] for k in ks]; stds = [d[k][1] for k in ks]
        ax.bar(ks, means, yerr=stds, capsize=5, color=["#2c7fb8", "#7fcdbb", "#c7e9b4"])
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9); ax.set_ylim(0, 1)
        for i, m in enumerate(means):
            ax.text(i, m + 0.03, f"{m:.2f}", ha="center", fontsize=9)
        fig.tight_layout(); fig.savefig(FIGS / fname, dpi=150); plt.close(fig)

    print(f"confusion acc {acc:.3f}; saved 3 figures to {FIGS}")
    for f in sorted(FIGS.glob("*.png")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
