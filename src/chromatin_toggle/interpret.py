"""Interpretability (item 4) + biological validation (item 5).

Trains the multiscale model, then computes PERMUTATION IMPORTANCE for every KG
node: shuffle that node's activations across cells and measure the drop in each
program's recall. This yields a node x program importance matrix (which
regulators the model relies on for each program) that is model-agnostic and
needs no gradients. It then cross-checks the top nodes per program against known
biology (expected regulators from the literature) and writes a heatmap + CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .dynamics import ToggleDynamics, _load, train
from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes

# Known key regulators per program (literature) for the biological sanity check.
EXPECTED = {
    "Fibrosis":      ["YAP_TAZ", "TEAD", "Factin", "ARID1A", "SWI_SNF"],
    "Hypertrophy":   ["MEF2", "HDAC4", "HDAC5", "CaMKII", "Piezo1", "PKD"],
    "ADM":           ["Sox9", "mTORC1", "Autophagy", "NFkB"],
    "InnateMemory":  ["NFkB", "PU1"],
    "MyogenicDiff":  ["MyoD", "MEF2", "Smad3"],
    "Pluripotency":  ["Oct4", "Smad3"],
    "Regeneration":  ["AP1", "SLC5A8", "HDAC1_3"],
}


def _recall_per_class(pred, y, n_classes):
    rec = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = (y == c)
        if int(m.sum()):
            rec[c] = float((pred[m] == c).float().mean())
    return rec


@torch.no_grad()
def _predict(model, X, bs=1024):
    model.eval()
    out = [model(X[i:i + bs], plasticity=1.0).argmax(-1) for i in range(0, X.size(0), bs)]
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description="Permutation-importance interpretability + bio validation")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    args = ap.parse_args()

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    n_classes = len(classes)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(X.size(0), generator=g)
    k = int(X.size(0) * args.val_frac)
    va, tr = perm[:k], perm[k:]

    model = ToggleDynamics(kg, hidden=args.hidden, steps=args.steps)
    train(model, X[tr], y[tr], args.epochs, 256, 1e-3, args.seed)

    base_pred = _predict(model, X[va])
    base_rec = _recall_per_class(base_pred, y[va], n_classes)
    base_acc = float((base_pred == y[va]).float().mean())
    print(f"trained on {len(tr)} cells; val acc {base_acc:.3f} on {len(va)} cells\n")

    # permutation importance: shuffle each node's column, measure recall drop
    rng = np.random.default_rng(args.seed)
    node_ids = list(kg.node_ids)
    # only nodes that actually carry input signal are worth permuting
    active = [j for j in range(kg.num_nodes) if float(X[va, j].abs().sum()) > 0]
    imp = np.zeros((len(active), n_classes))       # node x program recall drop
    for r, j in enumerate(active):
        Xp = X[va].clone()
        Xp[:, j] = X[va][torch.tensor(rng.permutation(len(va))), j]
        rec = _recall_per_class(_predict(model, Xp), y[va], n_classes)
        imp[r] = np.nan_to_num(base_rec - rec)

    prog_names = [c for c in classes if c != QUIESCENT]
    prog_cols = [classes.index(p) for p in prog_names]
    M = imp[:, prog_cols]                          # active-node x program

    # top nodes per program + biological validation
    print(f"{'program':<14}{'top regulators (permutation importance)':<52}{'known-biology hits'}")
    print("-" * 90)
    hits_total = exp_total = 0
    for pi, prog in enumerate(prog_names):
        order = np.argsort(-M[:, pi])
        top = [node_ids[active[o]] for o in order[:5]]
        exp = set(EXPECTED.get(prog, []))
        hits = [n for n in top if n in exp]
        if exp:
            hits_total += len(set(top) & exp); exp_total += len(exp)
        print(f"{prog:<14}{', '.join(top):<52}{', '.join(hits) if hits else '-'}")
    if exp_total:
        print(f"\nBiological validation: {hits_total} of top-5 lists intersect the "
              f"literature regulator sets (expected-set size {exp_total}).")

    # save matrix + heatmap
    out_csv = DATA_DIR.parent / "artifacts" / "importance_matrix.csv"
    out_csv.parent.mkdir(exist_ok=True)
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node"] + prog_names)
        for r, j in enumerate(active):
            w.writerow([node_ids[j]] + [f"{v:.4f}" for v in M[r]])
    print(f"\nSaved importance matrix -> {out_csv}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, max(6, len(active) * 0.22)))
        im = ax.imshow(M, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(prog_names))); ax.set_xticklabels(prog_names, rotation=45, ha="right")
        ax.set_yticks(range(len(active))); ax.set_yticklabels([node_ids[j] for j in active], fontsize=6)
        ax.set_title("Permutation importance (recall drop) — node x program")
        fig.colorbar(im, ax=ax, shrink=0.5, label="recall drop when node shuffled")
        fig.tight_layout()
        png = out_csv.parent / "figures" / "importance_heatmap.png"
        png.parent.mkdir(exist_ok=True)
        fig.savefig(png, dpi=150)
        print(f"Saved heatmap -> {png}")
    except Exception as e:
        print(f"(heatmap skipped: {e})")


if __name__ == "__main__":
    main()
