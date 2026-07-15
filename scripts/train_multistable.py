"""Train the structure-preserving MULTISTABLE signed-GRN, then verify multistability in the same run.

The classifier-trained GRN was monostable and cells sat far from fixed points (attractor_diagnostic.py).
This trains the SAME signed-KG dynamics with an added objective so that (i) each program is a stable
fixed-point attractor and (ii) cells flow to their program's basin -- turning the substrate into a genuine
attractor model while keeping the fixed literature signs + learnable strengths (mechanism/interpretability).

Objective:  L = flow (cells -> their prototype) + lambda_fp * fixed-point(prototypes) + lambda_ce * CE(readout).

Verification (printed, compares against the classifier-GRN baseline median displacement 0.759 / 1 attractor):
  - prototype fixed-point residual (should be ~0 after training)
  - do cells now sit AT fixed points? (RMS displacement, should be << 0.759)
  - how many attractors now? basin composition + program label + stability per attractor
  - basin assignment accuracy (settle -> nearest prototype vs true label)

Run:  uv run python scripts/train_multistable.py --data data/bmmc_shareseq.csv --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from chromatin_toggle.attractors import build_A, enumerate_attractors, settle_converged
from chromatin_toggle.device import pick_device
from chromatin_toggle.grn import MultistableGRN
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import all_classes


def _load(path, kg):
    df = pd.read_csv(path); classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    cols = list(kg.node_ids); X = np.zeros((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    y = np.array([ci[l] for l in df["label"]])
    return torch.tensor(X), torch.tensor(y), classes


def main():
    ap = argparse.ArgumentParser(description="train + verify the multistable signed-GRN")
    ap.add_argument("--data", default=str(DATA_DIR / "bmmc_shareseq.csv"))
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lambda-fp", type=float, default=1.0)
    ap.add_argument("--lambda-basin", type=float, default=1.0)
    ap.add_argument("--lambda-ce", type=float, default=0.1)
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--eps-rms", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    kg = load_kg(); dev = pick_device(a.device)
    X, y, classes = _load(a.data, kg)
    present = sorted(set(y.tolist()))
    stored = [classes[i] for i in present]
    proto_init = torch.stack([X[y == i].mean(0) for i in present])   # per-program mean state
    print(f"stored program attractors: {stored}")

    torch.manual_seed(a.seed)
    m = MultistableGRN(kg, stored, proto_init=proto_init, steps=a.steps, eta=a.eta).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    ce = nn.CrossEntropyLoss()
    Xd, yd = X.to(dev), y.to(dev); n = Xd.size(0)
    g = torch.Generator().manual_seed(a.seed)
    print(f"training multistable GRN (steps={a.steps}, lambda_fp={a.lambda_fp}, "
          f"lambda_basin={a.lambda_basin}, lambda_ce={a.lambda_ce}) ...")
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n, generator=g).to(dev); tot = np.zeros(4)
        for i in range(0, n, a.batch_size):
            idx = perm[i:i+a.batch_size]; opt.zero_grad()
            lf = m.flow_loss(Xd[idx], yd[idx]); lp = m.fp_loss(); lb = m.basin_loss(); lc = ce(m(Xd[idx]), yd[idx])
            loss = lf + a.lambda_fp * lp + a.lambda_basin * lb + a.lambda_ce * lc
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            tot += [lf.item(), lp.item(), lb.item(), lc.item()]
        if ep % 10 == 0 or ep == a.epochs - 1:
            nb = max(1, n // a.batch_size)
            print(f"  epoch {ep}: flow {tot[0]/nb:.4f}  fp {tot[1]/nb:.4f}  basin {tot[2]/nb:.4f}  "
                  f"ce {tot[3]/nb:.3f}", flush=True)
    m.eval()

    # ---- verification ----
    A, b, in_scale, eta = build_A(m)
    fp_res = ((m._step(m.proto) - m.proto) ** 2).mean(1).sqrt().detach().cpu().numpy()
    print(f"\n[fixed-point] prototype residual RMS/node: "
          f"{', '.join(f'{stored[i]}={v:.3f}' for i, v in enumerate(fp_res))}  (want ~0)")

    rng = np.random.default_rng(a.seed)
    idx = np.concatenate([rng.choice(np.where(y.numpy() == i)[0],
                                     min((y == i).sum().item(), 1500), replace=False) for i in present])
    cell_x0 = Xd[torch.tensor(idx)]
    with torch.no_grad():
        cs, _, _ = settle_converged(A, b, eta, cell_x0 * in_scale)
    disp = ((cs - cell_x0 * in_scale) ** 2).mean(1).sqrt().cpu().numpy()
    print(f"[cells@fixed-points] RMS/node displacement: median {np.median(disp):.3f}  p90 "
          f"{np.percentile(disp,90):.3f}   (classifier-GRN baseline was 0.759 -> want much smaller)")

    lo, hi = (Xd.amin(0) * in_scale), (Xd.amax(0) * in_scale)
    rand = lo + (hi - lo) * torch.rand(a.n_random, m.N, generator=torch.Generator().manual_seed(a.seed)).to(dev)
    all_x0 = torch.cat([cell_x0, rand])
    src = np.array([classes[int(y[i])] for i in idx] + ["random"] * a.n_random)
    labels, cents, rho, _ = enumerate_attractors(m, all_x0, a.eps_rms)
    K = cents.size(0)
    print(f"\n[multistability] {K} distinct attractor(s) at eps_rms={a.eps_rms} "
          f"(classifier-GRN baseline was 1)")
    with torch.no_grad():
        proto = m.proto.cpu()
    for rank, k in enumerate(sorted(range(K), key=lambda k: -(labels == k).sum().item())):
        sel = labels == k
        near = int(torch.cdist(cents[k:k+1], proto).argmin())
        comp = pd.Series(src[sel.numpy()]).value_counts().head(3).to_dict()
        print(f"  attractor {rank}: basin {int(sel.sum())}/{len(labels)} ({100*sel.float().mean():.0f}%)  "
              f"~{stored[near]}  [spectral radius {rho[k]:.3f}]  composition {comp}")

    pred = m.assign(cell_x0).cpu()
    true_local = m.full2stored.cpu()[y[torch.tensor(idx)]]
    acc = (pred == true_local).float().mean().item()
    print(f"\n[basin assignment] settle -> nearest prototype vs true label: acc {acc:.3f}")
    print("READ: >=2 attractors + small displacement + one basin per program + good assignment acc")
    print("      => the structure-preserving multistable GRN works; proceed to shift/separatrix analysis.")


if __name__ == "__main__":
    main()
