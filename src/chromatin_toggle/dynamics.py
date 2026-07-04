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
                 wta_gain=0.5, wta_iters=3, node_ann=None, context_dim=0):
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

        # optional annotation layers (each ablatable by omission)
        self.ann_proj = None
        if node_ann is not None:                          # per-node gene role + pathway terms
            self.register_buffer("node_ann", torch.as_tensor(node_ann, dtype=torch.float32))
            self.ann_proj = nn.Linear(self.node_ann.size(1), hidden, bias=False)
        self.ctx_proj = nn.Linear(context_dim, hidden, bias=False) if context_dim > 0 else None

    def forward(self, x0, plasticity=1.0, cue_window=None, context=None):
        """cue_window: if set, the extrinsic cue is injected only for steps
        t < cue_window, then removed (hysteresis test). context: optional
        [B, context_dim] experiment-metadata vector conditioning the graph."""
        B = x0.size(0)
        if not torch.is_tensor(plasticity):
            plasticity = torch.full((B,), float(plasticity), device=x0.device)
        p = plasticity.view(B, 1, 1)
        base = self.id_emb(self.node_ids) + self.type_emb(self.node_types)
        if self.ann_proj is not None:                             # gene-annotation features
            base = base + self.ann_proj(self.node_ann)
        base = base.unsqueeze(0)
        xin = self.in_proj(x0.unsqueeze(-1))                      # [B,N,H]
        mem_inj = xin * self.intrinsic_mask
        cue_inj = xin * self.cue_mask
        h = base.expand(B, -1, -1).contiguous()
        if self.ctx_proj is not None and context is not None:     # experiment context
            h = h + self.ctx_proj(context).unsqueeze(1)

        for t in range(self.steps):
            if self.asymmetric:
                g_mem = self.mem_gain                              # persistent, constant
                g_cue = self.cue_gain * (self.cue_decay ** t)      # transient, decaying
            else:
                g_mem = g_cue = 1.0                                # symmetric baseline
            if self.use_plasticity:
                g_cue = g_cue * p                                  # extrinsic gated by plasticity
            if cue_window is not None and t >= cue_window:
                g_cue = g_cue * 0.0                                # cue withdrawn
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


def class_weights(y, n_classes):
    """Inverse-frequency class weights (normalized) to counter imbalance."""
    counts = torch.bincount(y, minlength=n_classes).float()
    w = 1.0 / counts.clamp(min=1)
    return (w / w.sum() * n_classes)


def train(model, X, y, epochs, bs, lr, seed, plasticity_train=1.0, weights=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=weights)
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


MARKER_NODES = ["Sox9", "mTORC1", "Autophagy"]  # program-proximal readouts


def _mask_input(X, kg, mode):
    """Zero node inputs to control the marker-gene shortcut. none / no_markers /
    lineage_only (keep only cue + lineage-TF memory)."""
    if mode == "none":
        return X
    X = X.clone()
    if mode == "no_markers":
        for n in MARKER_NODES:
            if n in kg.node_index:
                X[:, kg.node_index[n]] = 0.0
    elif mode == "lineage_only":
        keep = {kg.node_index[n] for n in kg.memory_nodes if n in kg.node_index}
        keep |= {i for i, t in enumerate(kg.node_type) if t == "cue"}
        for j in range(kg.num_nodes):
            if j not in keep:
                X[:, j] = 0.0
    return X


def _pathway_programs(y, df, qi):
    """Return {pathway: activated_program_index} for pathways that have one."""
    out = {}
    for pw in sorted(df["pathway"].unique()):
        act = y[(df["pathway"] == pw).to_numpy()]
        act = act[act != qi]
        if len(act):
            out[pw] = int(torch.mode(act).values)
    return out


@torch.no_grad()
def sweep(model, X, df, prog_of, qi, levels):
    """{pathway: [fraction predicting its activated program at each plasticity]}."""
    model.eval()
    res = {}
    for pw, prog_i in prog_of.items():
        Xp = X[(df["pathway"] == pw).to_numpy()]
        res[pw] = [float((model(Xp, plasticity=lv).argmax(-1) == prog_i).float().mean())
                   for lv in levels]
    return res


@torch.no_grad()
def hysteresis(model, X, df, prog_of, window):
    """Fraction predicting the cue program under NEVER / TRANSIENT / SUSTAINED cue.
    TRANSIENT = cue on for `window` steps at high plasticity then withdrawn.
    Persistence (memory) = TRANSIENT stays high after the cue is gone."""
    model.eval()
    res = {}
    for pw, prog_i in prog_of.items():
        Xp = X[(df["pathway"] == pw).to_numpy()]
        never = float((model(Xp, plasticity=0.0).argmax(-1) == prog_i).float().mean())
        trans = float((model(Xp, plasticity=1.0, cue_window=window).argmax(-1) == prog_i).float().mean())
        sust = float((model(Xp, plasticity=1.0).argmax(-1) == prog_i).float().mean())
        res[pw] = (never, trans, sust)
    return res


def _agg(dicts, key):
    """Stack a per-seed list of {pathway: array} into mean/std over seeds."""
    pws = dicts[0].keys()
    return {pw: (np.mean([np.array(d[pw]) for d in dicts], 0),
                 np.std([np.array(d[pw]) for d in dicts], 0)) for pw in pws}


def main():
    ap = argparse.ArgumentParser(description="Thesis dynamical model: plasticity sweep + hysteresis")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway.csv"))
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--window", type=int, default=4, help="hysteresis: cue-on steps")
    ap.add_argument("--ablate", choices=["none", "asymmetric", "plasticity", "attractor"],
                    default="none")
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"],
                    default="none", help="marker-shortcut control on the inputs")
    args = ap.parse_args()

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    X = _mask_input(X, kg, args.mask)
    qi = classes.index(QUIESCENT)
    prog_of = _pathway_programs(y, df, qi)
    flags = dict(asymmetric=True, plasticity=True, attractor=True)
    if args.ablate != "none":
        flags[args.ablate] = False
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    print(f"ToggleDynamics {flags} | data={Path(args.data).name} n={len(df)} "
          f"| seeds={args.seeds} | mask={args.mask}")

    sweeps, hysts = [], []
    for s in args.seeds:
        torch.manual_seed(s)
        m = ToggleDynamics(kg, hidden=args.hidden, steps=args.steps, **flags)
        train(m, X, y, args.epochs, args.batch_size, args.lr, s)
        sweeps.append(sweep(m, X, df, prog_of, qi, levels))
        hysts.append(hysteresis(m, X, df, prog_of, args.window))
        print(f"  seed {s} done")

    sw = _agg(sweeps, None)
    print("\nPLASTICITY SWEEP  (mean fraction predicting cue program, +/- std over "
          f"{len(args.seeds)} seeds)")
    print(f"{'pathway':<18}{'program':<14}" + "".join(f"{f'p={l}':>14}" for l in levels))
    print("-" * (32 + 14 * len(levels)))
    for pw, prog_i in prog_of.items():
        mean, std = sw[pw]
        print(f"{pw:<18}{classes[prog_i]:<14}" +
              "".join(f"{m:.2f}±{sd:.2f}".rjust(14) for m, sd in zip(mean, std)))

    hy = _agg(hysts, None)
    print("\nHYSTERESIS  (mean fraction predicting cue program; TRANSIENT = cue "
          f"withdrawn after {args.window} steps)")
    print(f"{'pathway':<18}{'program':<14}{'NEVER':>14}{'TRANSIENT':>14}{'SUSTAINED':>14}")
    print("-" * 74)
    for pw, prog_i in prog_of.items():
        mean, std = hy[pw]
        n, t, s = mean
        print(f"{pw:<18}{classes[prog_i]:<14}{n:>14.2f}{t:>14.2f}{s:>14.2f}")
    print("\n(persistence/memory = TRANSIENT stays well above NEVER after cue removal)")


if __name__ == "__main__":
    main()
