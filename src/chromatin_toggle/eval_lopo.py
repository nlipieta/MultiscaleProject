"""Leave-one-pathway-out (LOPO) generalization harness.

Tests whether the KG substrate lets a model trained on some pathways predict a
HELD-OUT pathway it never saw labelled -- the honest test of "the pathways are
intertwined and the shared structure generalizes." For each pathway P: train on
all rows whose `pathway != P`, evaluate on `pathway == P`.

Three models are compared per fold so any generalization can be attributed to
the KG STRUCTURE rather than to the features or the readout:
  * kg_gnn   -- the resistance-gated KG-GNN over the literature KG
  * shuffled -- same GNN with the edge SOURCES permuted (degree preserved, wiring
                destroyed); if kg_gnn doesn't beat this, the KG isn't helping
  * mlp      -- bag-of-genes MLP on the node vector (no graph at all)

Reported per held-out pathway: overall accuracy, and "activated recall" = of the
held-out pathway's NON-Quiescent cells, how often the correct (never-trained)
program is predicted. Activated recall is the zero-shot signal; overall accuracy
is inflated by Quiescent (which the model does see in other pathways).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .kg import DATA_DIR, load_kg
from .resistance import ResistanceToggle
from .oracle import QUIESCENT, all_classes


class BagOfGenesMLP(nn.Module):
    """Theory-agnostic baseline: MLP on the raw node-activation vector."""
    def __init__(self, n_in: int, n_out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def _build(kind: str, kg, n_classes: int, seed: int, hidden: int = 64, steps: int = 6):
    torch.manual_seed(seed)
    if kind == "mlp":
        return BagOfGenesMLP(kg.num_nodes, n_classes, hidden=hidden)
    model = ResistanceToggle(kg, hidden=hidden, steps=steps)
    if kind == "shuffled":
        # permute the SOURCE axis of each relation's adjacency: keeps per-dst
        # in-degree, destroys which biological source feeds which target.
        g = torch.Generator().manual_seed(seed + 1)
        perm = torch.randperm(kg.num_nodes, generator=g)
        model.adjacency = model.adjacency[:, :, perm].contiguous()
    return model


def _train(model, Xtr, ytr, device, epochs, bs, lr, seed):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    n = Xtr.size(0)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _predict(model, X, device, bs=1024):
    model.eval()
    out = []
    for i in range(0, X.size(0), bs):
        out.append(model(X[i:i + bs].to(device)).argmax(-1).cpu())
    return torch.cat(out)


def run(data: str, epochs: int, bs: int, lr: float, seed: int,
        pathways: list[str] | None, device: str,
        hidden: int = 64, steps: int = 6, mask: str = "none") -> None:
    kg = load_kg()
    classes = all_classes(kg)
    cls_idx = {c: i for i, c in enumerate(classes)}
    quiescent_i = cls_idx[QUIESCENT]
    dev = torch.device(device if device != "auto" else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))

    df = pd.read_csv(data)
    if "pathway" not in df.columns:
        raise SystemExit("data needs a 'pathway' column (build with chromatin-combine)")
    node_cols = [c for c in kg.node_ids if c in df.columns]
    X_all = torch.tensor(df[node_cols].to_numpy(), dtype=torch.float32)
    # map onto full node ordering (missing nodes -> 0)
    full = torch.zeros(len(df), kg.num_nodes)
    for c in node_cols:
        full[:, kg.node_index[c]] = X_all[:, node_cols.index(c)]
    # marker-shortcut control (same regimes as classify/dynamics)
    if mask == "no_markers":
        for n in ["Sox9", "mTORC1", "Autophagy"]:
            if n in kg.node_index:
                full[:, kg.node_index[n]] = 0.0
    elif mask == "lineage_only":
        keep = {kg.node_index[n] for n in kg.memory_nodes if n in kg.node_index}
        keep |= {i for i, t in enumerate(kg.node_type) if t == "cue"}
        for j in range(kg.num_nodes):
            if j not in keep:
                full[:, j] = 0.0
    y_all = torch.tensor([cls_idx[l] for l in df["label"]], dtype=torch.long)
    pw_all = df["pathway"].to_numpy()
    print(f"(input mask: {mask})")

    folds = pathways or sorted(set(pw_all))
    kinds = ["kg_gnn", "shuffled", "mlp"]
    print(f"LOPO over {len(folds)} pathways x {len(kinds)} models "
          f"(epochs={epochs}, bs={bs}, lr={lr}, device={dev})\n")
    header = f"{'held-out pathway':<18}{'held program':<14}" + "".join(f"{k:>12}" for k in kinds)
    print(header + "   (overall acc / activated recall)")
    print("-" * len(header))

    for P in folds:
        te = pw_all == P
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        Xtr, ytr = full[tr], y_all[tr]
        Xte, yte = full[te], y_all[te]
        # the held-out pathway's activated (non-Quiescent) program(s)
        act_mask = yte != quiescent_i
        held_progs = sorted({classes[int(i)] for i in yte[act_mask].unique()})
        held_str = "/".join(held_progs) if held_progs else "-"

        cells = []
        for kind in kinds:
            m = _build(kind, kg, len(classes), seed, hidden=hidden, steps=steps)
            m = _train(m, Xtr, ytr, dev, epochs, bs, lr, seed)
            pred = _predict(m, Xte, dev)
            acc = float((pred == yte).float().mean())
            arec = (float((pred[act_mask] == yte[act_mask]).float().mean())
                    if int(act_mask.sum()) else float("nan"))
            cells.append(f"{acc:.2f}/{arec:.2f}")
        print(f"{P:<18}{held_str:<14}" + "".join(f"{c:>12}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-one-pathway-out generalization harness")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pathways", nargs="*", default=None,
                    help="subset of held-out pathways (default: all)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6, help="GNN message-passing rounds")
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"], default="none")
    args = ap.parse_args()
    run(args.data, args.epochs, args.batch_size, args.lr, args.seed,
        args.pathways, args.device, hidden=args.hidden, steps=args.steps, mask=args.mask)


if __name__ == "__main__":
    main()
