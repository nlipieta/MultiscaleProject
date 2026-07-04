"""Build the annotation layers and ablate their PREDICTIVE POWER.

Three structured priors (from data/annotations.yaml) are turned into model inputs:
  role     -- per-node one-hot gene regulatory role (pioneer/master/signal/...)
  pathway  -- per-node multi-hot curated pathway/process terms
  context  -- per-sample one-hot experiment metadata (organism/modality/tissue/...)

`role` + `pathway` enter as extra node features; `context` conditions the whole
graph. The ablation trains the multiscale ToggleDynamics with each layer on/off
and reports held-out validation accuracy, so we can see which annotation layer
actually adds predictive power (vs. the no-annotation baseline).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .dynamics import ToggleDynamics, _load
from .kg import DATA_DIR, load_kg


def build_features(kg, df):
    ann = yaml.safe_load((DATA_DIR / "annotations.yaml").read_text())
    N = kg.num_nodes

    # role one-hot
    roles = sorted(set(ann["gene_roles"].values()))
    ri = {r: i for i, r in enumerate(roles)}
    role_feat = np.zeros((N, len(roles)), dtype=np.float32)
    for node, r in ann["gene_roles"].items():
        if node in kg.node_index:
            role_feat[kg.node_index[node], ri[r]] = 1.0

    # pathway-term multi-hot
    terms = sorted({t for tags in ann["pathway_terms"].values() for t in tags})
    ti = {t: i for i, t in enumerate(terms)}
    path_feat = np.zeros((N, len(terms)), dtype=np.float32)
    for node, tags in ann["pathway_terms"].items():
        if node in kg.node_index:
            for t in tags:
                path_feat[kg.node_index[node], ti[t]] = 1.0

    # experiment context: one-hot categorical fields + ordinal level, per pathway
    exp = ann["experiments"]
    cat_fields = ["organism", "assay", "modality", "condition", "tissue", "timepoint"]
    vocab = {f: sorted({str(exp[p][f]) for p in exp}) for f in cat_fields}
    lvl_map = {"none": 0.0, "low": 0.34, "med": 0.67, "high": 1.0}
    def ctx_vec(pw):
        parts = []
        for f in cat_fields:
            oh = [0.0] * len(vocab[f])
            if pw in exp:
                oh[vocab[f].index(str(exp[pw][f]))] = 1.0
            parts.extend(oh)
        parts.append(lvl_map.get(exp.get(pw, {}).get("level", "none"), 0.0))
        return parts
    ctx_dim = sum(len(vocab[f]) for f in cat_fields) + 1
    ctx = np.array([ctx_vec(pw) for pw in df["pathway"]], dtype=np.float32)  # [rows, ctx_dim]

    return role_feat, path_feat, ctx, ctx_dim


def _split(n, val_frac, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    k = int(n * val_frac)
    return perm[k:], perm[:k]


def run_config(kg, X, y, ctx, ctx_dim, node_ann, use_ctx, tr, va,
               epochs, bs, lr, steps, hidden, seed):
    torch.manual_seed(seed)
    model = ToggleDynamics(kg, hidden=hidden, steps=steps,
                           node_ann=node_ann, context_dim=ctx_dim if use_ctx else 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    ctxt = torch.tensor(ctx) if use_ctx else None
    g = torch.Generator().manual_seed(seed + 7)
    for _ in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            c = ctxt[idx] if use_ctx else None
            loss = lossf(model(X[idx], plasticity=1.0, context=c), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        c = ctxt[va] if use_ctx else None
        pred = model(X[va], plasticity=1.0, context=c).argmax(-1)
        return float((pred == y[va]).float().mean())


def main():
    ap = argparse.ArgumentParser(description="Ablate predictive power of annotation layers")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_small.csv"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    role_feat, path_feat, ctx, ctx_dim = build_features(kg, df)
    both = np.concatenate([role_feat, path_feat], axis=1)

    # (name, node_ann, use_ctx)
    configs = [
        ("baseline (none)",  None,       False),
        ("+ role",           role_feat,  False),
        ("+ pathway terms",  path_feat,  False),
        ("+ experiment ctx", None,       True),
        ("+ all three",      both,       True),
    ]
    print(f"Annotation ablation | data={Path(args.data).name} n={len(df)} "
          f"seeds={args.seeds}\n  role_dim={role_feat.shape[1]} "
          f"pathway_dim={path_feat.shape[1]} ctx_dim={ctx_dim}\n")
    print(f"{'config':<20}{'val acc (mean±std)':>22}")
    print("-" * 42)
    for name, ann, use_ctx in configs:
        accs = []
        for s in args.seeds:
            tr, va = _split(len(df), args.val_frac, s)
            accs.append(run_config(kg, X, y, ctx, ctx_dim, ann, use_ctx, tr, va,
                                   args.epochs, args.batch_size, args.lr,
                                   args.steps, args.hidden, s))
        print(f"{name:<20}{f'{np.mean(accs):.3f}±{np.std(accs):.3f}':>22}")


if __name__ == "__main__":
    main()
