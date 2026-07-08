"""One-fold A/B benchmark: eager vs torch.compile, to measure the REAL speedup on YOUR GPU.

The model is launch-bound (tiny KG, ~15k op-launches/step), so torch.compile(reduce-overhead)
-> CUDA graphs should be a large win. This times one representative fold both ways, separating
the one-time compile WARMUP (paid once per fold) from steady-state per-epoch cost, and extrapolates
to a full run (5 folds x 3 seeds x 120 epochs).

Run on the GPU:  uv run python scripts/bench_compile.py --device cuda
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import _load, _mask_input, class_weights
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.resistance import ResistanceToggle


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def run(compile_mode, X, y, w, kg, dev, hidden, steps, bs, lr, epochs, sparse=False):
    torch.manual_seed(0)
    m = ResistanceToggle(kg, hidden=hidden, steps=steps, sparse_adj=sparse).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=w.to(dev))
    n = X.size(0)
    stop = (n // bs) * bs                              # static shape (CUDA graphs need it)
    fwd = m
    warmup = 0.0
    if compile_mode:
        t = time.perf_counter()
        fwd = torch.compile(m, mode="reduce-overhead", dynamic=False)
        loss = lossf(fwd(X[:bs], plasticity=1.0), y[:bs])  # first call triggers the compile
        loss.backward(); opt.zero_grad(set_to_none=True)
        _sync(dev); warmup = time.perf_counter() - t
    g = torch.Generator().manual_seed(0)
    _sync(dev); t = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, stop, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(fwd(X[idx], plasticity=1.0), y[idx]).backward()
            opt.step()
    _sync(dev); steady = time.perf_counter() - t
    return warmup, steady / epochs


def main():
    ap = argparse.ArgumentParser(description="eager vs torch.compile one-fold benchmark")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--mask", default="no_markers")
    ap.add_argument("--n", type=int, default=6000, help="subsample cells (one-fold-sized)")
    ap.add_argument("--epochs", type=int, default=20, help="timed steady-state epochs")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    # what a real converged run costs, for the extrapolation
    ap.add_argument("--full-folds", type=int, default=5)
    ap.add_argument("--full-seeds", type=int, default=3)
    ap.add_argument("--full-epochs", type=int, default=120)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU on this runtime (torch.cuda.is_available()==False).\n"
            "  Colab: Runtime > Change runtime type > GPU (T4), reconnect, re-run setup, retry.\n"
            "  (check with: !nvidia-smi -L)  -- or pass --device cpu to benchmark on CPU (no CUDA-graph gain).")
    dev = pick_device(args.device)

    kg = load_kg()
    X, y, classes, _ = _load(args.data, kg)
    X = _mask_input(X, kg, args.mask)
    if X.size(0) > args.n:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:args.n]
        X, y = X[idx], y[idx]
    X, y = X.to(dev), y.to(dev)
    w = class_weights(y, len(classes))
    print(f"device={dev}  n={X.size(0)}  hidden={args.hidden} steps={args.steps} bs={args.bs}  "
          f"(timing {args.epochs} steady epochs each mode)\n")

    if dev.type != "cuda":
        print("WARNING: torch.compile(reduce-overhead) targets CUDA graphs; on non-CUDA the compiled\n"
              "path may fall back or not reflect the real (GPU) speedup. Run with --device cuda.\n")

    A = args  # shorthand
    configs = [
        ("dense  eager   ", False, False),
        ("dense  compiled", True,  False),
        ("sparse eager   ", False, True),
        ("sparse compiled", True,  True),
    ]
    res = {}
    for label, comp, sparse in configs:
        warm, ep = run(comp, X, y, w, kg, dev, A.hidden, A.steps, A.bs, A.lr, A.epochs, sparse=sparse)
        res[label] = (warm, ep)
        extra = f"   + {warm:.1f}s warmup/fold" if comp else ""
        print(f"{label}: {ep*1000:8.1f} ms/epoch{extra}")

    base = res["dense  eager   "][1]                   # current production path
    print(f"\nsteady-state speedup vs dense-eager (your current path):")
    for label, _, _ in configs:
        print(f"  {label}: {base/res[label][1]:.1f}x")

    F, S, E = A.full_folds, A.full_seeds, A.full_epochs
    folds = F * S
    print(f"\nfull run ({F}fold x {S}seed x {E}ep = {folds} trainings), end-to-end:")
    best = None
    for label, comp, _ in configs:
        warm, ep = res[label]
        total = (warm + ep * E) * folds if comp else ep * E * folds  # warmup paid per fold
        print(f"  {label}: ~{total/60:6.1f} min   (~{res['dense  eager   '][1]*E*folds/total:.1f}x)")
        if best is None or total < best[1]:
            best = (label, total)
    print(f"\n-> fastest: {best[0].strip()} (~{best[1]/60:.0f} min)")
    if "sparse" in best[0]:
        print("   sparse adjacency is the win -> I'll make --sparse-adj the default for the real runs.")


if __name__ == "__main__":
    main()
