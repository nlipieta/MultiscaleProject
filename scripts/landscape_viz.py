"""Waddington landscape on the signed-GRN's fixed points (emergence visualization, framework Q1).

Trains the GRN dynamical model, settles every cell to its fixed point, embeds the settled states in
2D (PCA), and builds a potential V(z) = -log(density) -- valleys = attractors. Cells are plotted on
the surface coloured by commitment stage (HSC/MPP -> early-Ery -> late-Ery), showing the erythroid
trajectory descending into its basin. This is the landscape's legitimate use: visualizing 'what state
emerges', NOT perturbation magnitude (which is KG-density limited).

Run:  uv run python scripts/landscape_viz.py --data data/bmmc_shareseq.csv --device cuda --out /content/drive/MyDrive/mtoggle/landscape.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import class_weights
from chromatin_toggle.grn import GRNDynamics
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import all_classes

STAGE_COLORS = ["#4477AA", "#CCBB44", "#EE6677"]      # colorblind-safe: HSC / early-Ery / late-Ery
STAGE_NAMES = {0.0: "HSC/MPP", 1.0: "early-Ery", 2.0: "late-Ery"}


def _load(path, kg):
    df = pd.read_csv(path); classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    cols = list(kg.node_ids); X = np.zeros((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    y = np.array([ci[l] for l in df["label"]])
    stage = pd.to_numeric(df["timepoint"], errors="coerce").to_numpy()
    return torch.tensor(X), y, stage


def main():
    ap = argparse.ArgumentParser(description="Waddington landscape on the GRN fixed points")
    ap.add_argument("--data", default=str(DATA_DIR / "bmmc_shareseq.csv"))
    ap.add_argument("--out", default=str(DATA_DIR / "landscape.png"))
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    kg = load_kg(); dev = pick_device(a.device)
    classes = all_classes(kg)
    X, y, stage = _load(a.data, kg)

    torch.manual_seed(a.seed)
    m = GRNDynamics(kg, steps=a.steps, eta=a.eta).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    lossf = nn.CrossEntropyLoss(weight=class_weights(torch.tensor(y), len(classes)).to(dev))
    Xd, yd = X.to(dev), torch.tensor(y).to(dev); n = Xd.size(0)
    g = torch.Generator().manual_seed(a.seed)
    print(f"training GRN (steps={a.steps}) for landscape ...")
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, a.batch_size):
            idx = perm[i:i+a.batch_size]; opt.zero_grad()
            loss = lossf(m(Xd[idx]), yd[idx]); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval()
    with torch.no_grad():
        S = torch.cat([m.settle(Xd[i:i+2048]) for i in range(0, n, 2048)]).cpu().numpy()

    # 2D embedding of the settled states (PCA), oriented so stage increases along z1
    Sc = S - S.mean(0)
    _, _, Vt = np.linalg.svd(Sc, full_matrices=False)
    z = Sc @ Vt[:2].T
    if np.corrcoef(z[:, 0], stage)[0, 1] < 0:
        z[:, 0] *= -1
    # potential V = -log(density): valleys = attractors
    kde = gaussian_kde(z.T)
    Vcells = -np.log(kde(z.T) + 1e-12)
    g1 = np.linspace(z[:, 0].min(), z[:, 0].max(), 80)
    g2 = np.linspace(z[:, 1].min(), z[:, 1].max(), 80)
    G1, G2 = np.meshgrid(g1, g2)
    Vgrid = -np.log(kde(np.vstack([G1.ravel(), G2.ravel()])) + 1e-12).reshape(G1.shape)

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(G1, G2, Vgrid, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True)
    for s in sorted(set(stage)):
        sel = stage == s
        ax.scatter(z[sel, 0], z[sel, 1], Vcells[sel], s=4, alpha=0.5,
                   color=STAGE_COLORS[int(s) % 3], label=STAGE_NAMES.get(s, str(s)))
    ax.set_xlabel("PC1 (commitment)"); ax.set_ylabel("PC2"); ax.set_zlabel("potential V = -log density")
    ax.set_title("GRN fixed-point landscape (erythroid emergence)"); ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=38, azim=-60)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(G1, G2, Vgrid, levels=25, cmap="viridis")
    for s in sorted(set(stage)):
        sel = stage == s
        ax2.scatter(z[sel, 0], z[sel, 1], s=5, alpha=0.6,
                    color=STAGE_COLORS[int(s) % 3], label=STAGE_NAMES.get(s, str(s)))
    ax2.set_xlabel("PC1 (commitment)"); ax2.set_ylabel("PC2")
    ax2.set_title("contour (valleys = attractors); cells by stage"); ax2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(a.out, dpi=140)
    print(f"saved landscape -> {a.out}")
    # honest readout: is the erythroid valley separated by stage?
    for s in sorted(set(stage)):
        sel = stage == s
        print(f"  stage {STAGE_NAMES.get(s, s):<10} mean PC1 {z[sel,0].mean():+.2f}  mean V {Vcells[sel].mean():.2f}")
    print("read: if late-Ery sits at higher PC1 / lower V than HSC, cells descend into the erythroid basin.")


if __name__ == "__main__":
    main()
