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
from .resistance import ResistanceToggle


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


def train(model, X, y, epochs, bs, lr, seed, plasticity_train=1.0, weights=None,
          weight_decay=0.0, schedule=True, compile=False):
    dev = next(model.parameters()).device          # follow the model's device (CPU/MPS/CUDA)
    X = X.to(dev); y = y.to(dev)                    # pre-move data ONCE (it's small) -> no per-batch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # H2D copies
    # cosine LR anneal over epochs -> cleaner convergence (matters once epochs are large)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if schedule else None
    lossf = nn.CrossEntropyLoss(weight=weights.to(dev) if weights is not None else None)
    g = torch.Generator().manual_seed(seed)
    n = X.size(0)
    # The KG is tiny (N~194, R=8) so training is kernel-LAUNCH-bound on GPU, not
    # compute-bound: thousands of microsecond kernels per step. torch.compile with
    # reduce-overhead captures the step loop into a CUDA graph -> one launch replaces
    # the whole graph (biggest lever for small models). CUDA graphs need a STATIC
    # batch shape, so we drop the last partial minibatch when compiling (loses <bs
    # samples/epoch, unbiased). Only the training forward is compiled; predict/eval
    # call the eager module so their variable shapes don't trigger recompiles.
    fwd, drop_last, compiled = model, False, False
    if compile and dev.type == "cuda" and n >= bs:
        # Compile errors surface on the FIRST call, not at construction -> warm up on
        # one static-shape batch inside try/except and fall back to eager if it fails.
        # NOTE compile is best on SINGLE long runs; in a many-fold loop it recompiles
        # (and pins a CUDA-graph pool) per model -> free it at the end of train().
        try:
            cand = torch.compile(model, mode="reduce-overhead", dynamic=False)
            cand(X[:bs], plasticity=plasticity_train).sum().backward()
            opt.zero_grad()
            fwd, drop_last, compiled = cand, True, True
        except Exception as e:                       # unsupported op / OOM at this batch
            print(f"[train] torch.compile unavailable ({e}); running eager")
            fwd, drop_last, compiled = model, False, False
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()                 # reclaim the failed attempt's memory
    stop = (n // bs) * bs if (drop_last and n >= bs) else n
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, stop, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(fwd(X[idx], plasticity=plasticity_train), y[idx])   # already on device
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
    if dev.type == "cuda":                            # free per-fold memory so it doesn't
        if compiled:                                  # accumulate across a k-fold/seed loop
            try:
                torch.compiler.reset()                # drop this model's CUDA-graph pool
            except Exception:
                pass
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def predict_proba(model, X, bs=1024, plasticity=1.0):
    """Device-agnostic batched softmax probabilities; returns CPU [N, n_classes]."""
    dev = next(model.parameters()).device
    model.eval()
    return torch.cat([torch.softmax(model(X[i:i + bs].to(dev), plasticity=plasticity), -1).cpu()
                      for i in range(0, X.size(0), bs)])


@torch.no_grad()
def predict(model, X, bs=1024, plasticity=1.0):
    """Device-agnostic batched prediction; returns CPU class indices."""
    dev = next(model.parameters()).device
    model.eval()
    return torch.cat([model(X[i:i + bs].to(dev), plasticity=plasticity).argmax(-1).cpu()
                      for i in range(0, X.size(0), bs)])


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
    ap.add_argument("--ablate", choices=["none", "resistance", "plasticity", "attractor"],
                    default="none", help="knock out a resistance-gated mechanism")
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"],
                    default="none", help="marker-shortcut control on the inputs")
    ap.add_argument("--subsample", type=int, default=0, help="subsample cells (0 = all)")
    ap.add_argument("--save", default=None, help="write per-program hysteresis to this YAML")
    args = ap.parse_args()

    kg = load_kg()
    X, y, classes, df = _load(args.data, kg)
    X = _mask_input(X, kg, args.mask)
    if args.subsample and X.size(0) > args.subsample:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:args.subsample]
        X, y = X[idx], y[idx]; df = df.iloc[idx.numpy()].reset_index(drop=True)
    qi = classes.index(QUIESCENT)
    prog_of = _pathway_programs(y, df, qi)
    flags = dict(resistance=True, plasticity_mode="lower_resistance", attractor="soft")
    if args.ablate == "resistance":  flags["resistance"] = False
    elif args.ablate == "plasticity": flags["plasticity_mode"] = "none"
    elif args.ablate == "attractor":  flags["attractor"] = "none"
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    print(f"ResistanceToggle {flags} | data={Path(args.data).name} n={len(df)} "
          f"| seeds={args.seeds} | mask={args.mask}")

    sweeps, hysts = [], []
    for s in args.seeds:
        torch.manual_seed(s)
        m = ResistanceToggle(kg, hidden=args.hidden, steps=args.steps, **flags)
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

    if args.save:
        import yaml
        out = {}
        for pw, prog_i in prog_of.items():
            n, t, s = hy[pw][0]
            out[classes[prog_i]] = {
                "never": round(float(n), 3), "transient": round(float(t), 3),
                "sustained": round(float(s), 3),
                # persistence = program still expressed after the cue is withdrawn;
                # reversibility = how much it drops vs a sustained cue (large drop => reversible)
                "persistence_after_cue_removal": round(float(t), 3),
                "reversibility_drop": round(float(s - t), 3),
                "pathway": pw,
            }
        Path(args.save).write_text(yaml.safe_dump(out, sort_keys=True))
        print(f"saved per-program hysteresis -> {args.save}")


if __name__ == "__main__":
    main()
