"""GPU op-level profile: find the TRUE training hot spot by measurement, not guesswork.

Two prior hypotheses (launch-bound -> compile; sparse-einsum) were both refuted by benchmark.
This runs torch.profiler with CUDA timing over a few real training steps and prints the ops
that actually consume GPU time, so we optimize the real bottleneck.

Run:  uv run python scripts/profile_gpu.py --device cuda            # your production path (dense)
      uv run python scripts/profile_gpu.py --device cuda --sparse   # compare
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import _load, _mask_input, class_weights
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.resistance import ResistanceToggle


def _cuda_self(e):                       # attribute name varies across torch versions
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(e, attr):
            return getattr(e, attr)
    return 0.0


def main():
    ap = argparse.ArgumentParser(description="GPU op-level training profile")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--mask", default="no_markers")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sparse", action="store_true")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU on this runtime. Colab: Runtime > Change runtime type > GPU.")
    dev = pick_device(a.device)

    kg = load_kg()
    X, y, classes, _ = _load(a.data, kg)
    X = _mask_input(X, kg, a.mask)
    if X.size(0) > a.n:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:a.n]
        X, y = X[idx], y[idx]
    X, y = X.to(dev), y.to(dev)
    w = class_weights(y, len(classes)).to(dev)
    torch.manual_seed(0)
    m = ResistanceToggle(kg, hidden=a.hidden, steps=a.steps, sparse_adj=a.sparse).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss(weight=w)
    xb, yb = X[:a.bs], y[:a.bs]

    for _ in range(3):                   # warm up (autotune + allocation)
        opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(a.iters):
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()

    ka = list(prof.key_averages())
    tot = sum(_cuda_self(e) for e in ka) or 1.0
    top = sorted(ka, key=lambda e: -_cuda_self(e))[:15]
    print(f"\nsparse={a.sparse} bs={a.bs} steps={a.steps} hidden={a.hidden} iters={a.iters}")
    print(f"total CUDA self-time {tot/1000/a.iters:.1f} ms/step\n")
    print(f"{'op':34}{'CUDA ms/step':>13}{'%':>7}{'calls/step':>12}")
    print("-" * 66)
    for e in top:
        print(f"{e.key[:33]:34}{_cuda_self(e)/1000/a.iters:13.2f}{_cuda_self(e)/tot*100:7.1f}"
              f"{e.count/a.iters:12.1f}")


if __name__ == "__main__":
    main()
