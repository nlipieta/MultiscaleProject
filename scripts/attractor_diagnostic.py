"""Attractor diagnostic on the trained signed-GRN: is the learned dynamics a usable attractor model?

Before building a transition/shift analyzer we must know whether the GRN (trained as a CLASSIFIER, not
an energy model) actually behaves like a multistable dynamical system. This script answers three things:

  (1) MULTISTABILITY -- settle many initial conditions to TRUE convergence and cluster the fixed points.
      How many distinct attractors does the system support? What program is active in each? (framework Q1)
  (2) ARE CELLS AT FIXED POINTS -- displacement ||settle(cell) - cell||. If real cells do NOT sit near
      fixed points, the classifier-GRN is not a clean attractor model -> motivates energy-based retraining.
  (3) STABILITY = RESISTANCE -- the Jacobian J = (1-eta)I + eta*diag(1-tanh(z)^2)A at each fixed point.
      Spectral radius < 1 => stable; the eigenvalue nearest 1 is the SOFTEST mode (easiest escape
      direction), and its eigenvector's top nodes are what a shift would push on. Resistance becomes a
      measured basin curvature, not a learned gate.

This is a READ-ONLY diagnostic (no tuning). Its result decides whether we analyze the current GRN or
first retrain the dynamics so data states are true fixed points.

Run:  uv run python scripts/attractor_diagnostic.py --data data/bmmc_shareseq.csv --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import class_weights
from chromatin_toggle.grn import GRNDynamics
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import QUIESCENT, all_classes


def _load(path, kg):
    df = pd.read_csv(path); classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    cols = list(kg.node_ids); X = np.zeros((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    y = np.array([ci[l] for l in df["label"]])
    return torch.tensor(X), y, classes


def build_A(m):
    """Dense signed weighted adjacency A[dst,src] = softplus(mag)*sign, matching GRNDynamics._step."""
    N = m.N
    with torch.no_grad():
        A = torch.zeros(N, N, device=m.mag.device)
        w = torch.nn.functional.softplus(m.mag) * m.sign
        A.index_put_((m.dst, m.src), w, accumulate=True)
        return A, m.bias.detach(), float(m.in_scale.detach()), m.eta


def settle_converged(A, b, eta, x0, tol=1e-5, max_steps=500):
    """Iterate x <- (1-eta)x + eta*tanh(A x + b) to a fixed point; return x*, steps used, converged mask."""
    x = x0
    last = None
    for t in range(max_steps):
        z = x @ A.T + b
        xn = (1 - eta) * x + eta * torch.tanh(z)
        d = (xn - x).abs().amax(dim=1)
        x = xn
        if last is not None and float(d.max()) < tol:
            return x, t + 1, d < tol
        last = d
    return x, max_steps, (last < tol) if last is not None else torch.zeros(x.size(0), dtype=torch.bool)


def jacobian(A, b, eta, xstar):
    z = xstar @ A.T + b
    g = 1.0 - torch.tanh(z) ** 2                 # sech^2, [N]
    return (1 - eta) * torch.eye(A.size(0), device=A.device) + eta * (g[:, None] * A)


def greedy_cluster(states, eps_rms):
    """Cluster settled states by RMS-per-node distance; returns (labels, centroids)."""
    N = states.size(1); cents = []; labels = torch.empty(states.size(0), dtype=torch.long)
    for i in range(states.size(0)):
        s = states[i]
        if cents:
            C = torch.stack(cents)
            d = ((C - s) ** 2).mean(1).sqrt()    # RMS per node
            j = int(d.argmin())
            if float(d[j]) < eps_rms:
                labels[i] = j; continue
        labels[i] = len(cents); cents.append(s.clone())
    return labels, torch.stack(cents)


def main():
    ap = argparse.ArgumentParser(description="attractor diagnostic on the trained signed-GRN")
    ap.add_argument("--data", default=str(DATA_DIR / "bmmc_shareseq.csv"))
    ap.add_argument("--steps", type=int, default=15, help="GRN training rollout depth")
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-cells", type=int, default=2000, help="data cells to settle (stratified)")
    ap.add_argument("--n-random", type=int, default=400, help="random initial conditions to settle")
    ap.add_argument("--eps-rms", type=float, default=0.08, help="attractor clustering tol (RMS/node)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    kg = load_kg(); dev = pick_device(a.device)
    X, y, classes = _load(a.data, kg)

    torch.manual_seed(a.seed)
    m = GRNDynamics(kg, steps=a.steps, eta=a.eta).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    lossf = nn.CrossEntropyLoss(weight=class_weights(torch.tensor(y), len(classes)).to(dev))
    Xd, yd = X.to(dev), torch.tensor(y).to(dev); n = Xd.size(0)
    g = torch.Generator().manual_seed(a.seed)
    print(f"training GRN (steps={a.steps}) ...")
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, a.batch_size):
            idx = perm[i:i+a.batch_size]; opt.zero_grad()
            loss = lossf(m(Xd[idx]), yd[idx]); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval()

    A, b, in_scale, eta = build_A(m)
    # faithfulness check: A-dynamics must equal the model's own _step
    with torch.no_grad():
        xt = Xd[:64] * in_scale
        ref = m._step(xt); mine = (1 - eta) * xt + eta * torch.tanh(xt @ A.T + b)
        assert float((ref - mine).abs().max()) < 1e-4, "A-dynamics != model._step"
    print("A-dynamics matches model._step (faithful).")

    # ---- stratified data-cell inits + random inits ----
    rng = np.random.default_rng(a.seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, min(len(ci), a.n_cells // len(np.unique(y))), replace=False))
    idx = np.array(idx)
    cell_x0 = (Xd[torch.tensor(idx)] * in_scale)
    lo, hi = Xd.amin(0) * in_scale, Xd.amax(0) * in_scale
    rand_x0 = lo + (hi - lo) * torch.rand(a.n_random, m.N, generator=torch.Generator(device="cpu").manual_seed(a.seed)).to(dev)
    with torch.no_grad():
        cell_star, cs_steps, _ = settle_converged(A, b, eta, cell_x0)
        rand_star, rs_steps, _ = settle_converged(A, b, eta, rand_x0)

    # ---- (2) do cells sit near fixed points? ----
    disp = ((cell_star - cell_x0) ** 2).mean(1).sqrt().cpu().numpy()   # RMS/node displacement
    print("\n(2) ARE CELLS AT FIXED POINTS  (RMS/node displacement settle(cell)-cell):")
    print(f"    median {np.median(disp):.3f}  p90 {np.percentile(disp,90):.3f}  max {disp.max():.3f}"
          f"   (converged in ~{cs_steps} steps)")
    print("    small (<~0.1) => cells already near fixed points (good attractor model);")
    print("    large => classifier-GRN is NOT a clean attractor model -> motivates energy-based training.")

    # ---- (1) enumerate attractors ----
    allstar = torch.cat([cell_star, rand_star])
    src = np.array(["cell:" + classes[y[i]] for i in idx] + ["random"] * a.n_random)
    labels, cents = greedy_cluster(allstar, a.eps_rms)
    K = cents.size(0)
    print(f"\n(1) ATTRACTORS: {K} distinct fixed point(s) at eps_rms={a.eps_rms} "
          f"(from {allstar.size(0)} initial conditions)")
    prog_idx = [(classes[i], int(nidx)) for i, nidx in enumerate(m.cls_node.tolist()) if nidx >= 0]
    order = sorted(range(K), key=lambda k: -(labels == k).sum().item())
    for rank, k in enumerate(order):
        sel = labels == k
        cent = cents[k]
        acts = sorted(((cent[nidx].item(), name) for name, nidx in prog_idx), reverse=True)[:4]
        top_prog = ", ".join(f"{nm} {v:+.2f}" for v, nm in acts)
        comp = pd.Series(src[sel.numpy()]).value_counts().head(3).to_dict()
        # stability
        J = jacobian(A, b, eta, cent.to(dev))
        ev = torch.linalg.eigvals(J).abs().cpu().numpy()
        rho = ev.max(); soft = np.sort(ev)[-1]
        stable = "STABLE" if rho < 1.0 + 1e-6 else "UNSTABLE"
        print(f"  attractor {rank}: basin {int(sel.sum())}/{allstar.size(0)} "
              f"({100*sel.float().mean():.0f}%)  [{stable}, spectral radius {rho:.3f}]")
        print(f"      top program activity: {top_prog}")
        print(f"      basin composition: {comp}")
    print("\n(3) STABILITY = RESISTANCE: spectral radius per attractor above (closer to 1 = softer/shallower")
    print("    basin = lower resistance). If everything collapses to 1 attractor, the trained dynamics is")
    print("    NOT multistable as-is -> needs an attractor-training objective before transition analysis.")


if __name__ == "__main__":
    main()
