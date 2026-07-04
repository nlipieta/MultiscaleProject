"""Thesis-faithful dynamical model + test.

Implements the review's core claim as architecture: cells integrate
  * INTRINSIC identity biases -- stable, persistent, deeply processed -> STRONG:
    re-injected (clamped) at every message-passing step, high gain.
  * EXTRINSIC cues -- variable, transient, superficially processed -> WEAK:
    injected with a decaying schedule (transient) and gated by PLASTICITY.
...to stabilize the program with the strongest net bias (attractor via program
lateral inhibition), where raising PLASTICITY lets a weak transient cue overcome
the entrenched intrinsic bias and flip the stabilized program.

Each mechanism is a flag so it can be ablated against the symmetric baseline:
  asymmetric   -- persistent-strong intrinsic vs transient-weak extrinsic
  plasticity   -- extrinsic gain scales with the plasticity input p in [0,1]
  attractor    -- winner-take-all sharpening among program logits

The DYNAMICAL TEST (`sweep`) does not measure held-out classification; it checks
the thesis's behavioral predictions: at low plasticity the intrinsic default
holds regardless of cue (robustness); above a plasticity threshold the cue flips
the stabilized program (plasticity-gated override).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes


class ToggleDynamics(nn.Module):
    def __init__(self, kg, hidden=32, steps=8, asymmetric=True, plasticity=True,
                 attractor=True, mem_gain=1.0, cue_gain=1.0, cue_decay=0.6,
                 wta_gain=0.5, wta_iters=3):
        super().__init__()
        self.N, self.steps, self.hidden = kg.num_nodes, steps, hidden
        self.asymmetric, self.use_plasticity, self.attractor = asymmetric, plasticity, attractor
        self.mem_gain, self.cue_gain, self.cue_decay = mem_gain, cue_gain, cue_decay
        self.wta_gain, self.wta_iters = wta_gain, wta_iters

        self.id_emb = nn.Embedding(kg.num_nodes, hidden)
        self.type_emb = nn.Embedding(kg.num_types, hidden)
        self.in_proj = nn.Linear(1, hidden)
        self.rel_lin = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(kg.num_relations)])
        self.self_lin = nn.Linear(hidden, hidden, bias=False)
        self.gru = nn.GRUCell(hidden, hidden)
        self.prog_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.quiescent_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        self.register_buffer("node_ids", torch.arange(kg.num_nodes))
        self.register_buffer("node_types",
            torch.tensor([kg.type_index[t] for t in kg.node_type], dtype=torch.long))
        self.register_buffer("adjacency", kg.structural_adjacency())
        self.register_buffer("program_index", torch.tensor(kg.program_index, dtype=torch.long))
        cue = torch.tensor([t == "cue" for t in kg.node_type])
        prog = torch.zeros(kg.num_nodes, dtype=torch.bool)
        prog[kg.program_index] = True
        self.register_buffer("cue_mask", cue.float().view(1, -1, 1))
        self.register_buffer("intrinsic_mask", (~cue & ~prog).float().view(1, -1, 1))

    def forward(self, x0, plasticity=1.0):
        B = x0.size(0)
        if not torch.is_tensor(plasticity):
            plasticity = torch.full((B,), float(plasticity), device=x0.device)
        p = plasticity.view(B, 1, 1)
        base = (self.id_emb(self.node_ids) + self.type_emb(self.node_types)).unsqueeze(0)
        xin = self.in_proj(x0.unsqueeze(-1))                      # [B,N,H]
        mem_inj = xin * self.intrinsic_mask
        cue_inj = xin * self.cue_mask
        h = base.expand(B, -1, -1).contiguous()

        for t in range(self.steps):
            if self.asymmetric:
                g_mem = self.mem_gain                              # persistent, constant
                g_cue = self.cue_gain * (self.cue_decay ** t)      # transient, decaying
            else:
                g_mem = g_cue = 1.0                                # symmetric baseline
            if self.use_plasticity:
                g_cue = g_cue * p                                  # extrinsic gated by plasticity
            inj = g_mem * mem_inj + g_cue * cue_inj

            msg = self.self_lin(h)
            for r in range(self.adjacency.size(0)):
                msg = msg + torch.einsum("ds,bsh->bdh", self.adjacency[r], self.rel_lin[r](h))
            h = self.gru((msg + inj).reshape(B * self.N, -1),
                         h.reshape(B * self.N, -1)).reshape(B, self.N, -1)

        prog_logits = self.prog_head(h[:, self.program_index, :]).squeeze(-1)   # [B,P]
        quiescent = self.quiescent_head(h.mean(dim=1))                          # [B,1]
        logits = torch.cat([prog_logits, quiescent], dim=-1)
        if self.attractor:  # winner-take-all sharpening: settle toward one program
            for _ in range(self.wta_iters):
                a = torch.softmax(logits, dim=-1)
                logits = logits + self.wta_gain * (a - a.mean(dim=-1, keepdim=True))
        return logits


def _load(data, kg):
    df = pd.read_csv(data)
    classes = all_classes(kg)
    ci = {c: i for i, c in enumerate(classes)}
    X = torch.zeros(len(df), kg.num_nodes)
    for c in kg.node_ids:
        if c in df.columns:
            X[:, kg.node_index[c]] = torch.tensor(pd.to_numeric(df[c], errors="coerce")
                                                   .fillna(0).to_numpy(), dtype=torch.float32)
    y = torch.tensor([ci[l] for l in df["label"]], dtype=torch.long)
    return X, y, classes, df


def train(model, X, y, epochs, bs, lr, seed, plasticity_train=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(X.size(0), generator=g)
        for i in range(0, X.size(0), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(X[idx], plasticity=plasticity_train), y[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def sweep(model, X, y, classes, df, levels=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """For each pathway, sweep plasticity and report the predicted-program mix.
    The thesis predicts: low plasticity -> intrinsic default (mostly Quiescent);
    high plasticity -> the cue-driven program appears."""
    model.eval()
    qi = classes.index(QUIESCENT)
    print("\nPLASTICITY SWEEP  (fraction predicting the pathway's activated program "
          "| fraction Quiescent)\n")
    print(f"{'pathway':<18}{'activated program':<16}" + "".join(f"{f'p={l}':>14}" for l in levels))
    print("-" * (34 + 14 * len(levels)))
    for pw in sorted(df["pathway"].unique()):
        mask = (df["pathway"] == pw).to_numpy()
        Xp = X[mask]
        yp = y[mask]
        act = yp[yp != qi]
        if len(act) == 0:
            continue
        prog_i = int(torch.mode(act).values)      # the pathway's dominant activated program
        cells = []
        for lv in levels:
            pred = model(Xp, plasticity=lv).argmax(-1)
            f_act = float((pred == prog_i).float().mean())
            f_q = float((pred == qi).float().mean())
            cells.append(f"{f_act:.2f}|{f_q:.2f}")
        print(f"{pw:<18}{classes[prog_i]:<16}" + "".join(f"{c:>14}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description="Thesis dynamical model: train + plasticity sweep")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_small.csv"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", choices=["none", "asymmetric", "plasticity", "attractor"],
                    default="none", help="turn OFF one mechanism to ablate it")
    args = ap.parse_args()

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    flags = dict(asymmetric=True, plasticity=True, attractor=True)
    if args.ablate != "none":
        flags[args.ablate] = False
    print(f"ToggleDynamics {flags} | data={Path(args.data).name} n={len(df)}")
    model = ToggleDynamics(kg, hidden=args.hidden, steps=args.steps, **flags)
    train(model, X, y, args.epochs, args.batch_size, args.lr, args.seed)
    sweep(model, X, y, classes, df)


if __name__ == "__main__":
    main()
