"""Temporal-trajectory validation (thesis: graded, time-integrated program emergence).

Uses REAL experimental time as an enrichment for VALIDATION, not as a classification
feature (which would leak, since for time-course datasets the label is defined by
timepoint). Procedure: train a program classifier on the pool, then on a time-course
dataset's cells compute the model's predicted P(program) and track it as a function of
real time since the cue.

The sharp, non-circular test: among cells that are ALL labelled with the program (e.g.
GSE147405 EMT at 8h/1d/3d/7d are all "EMT"), does P(program) RISE with time? A model
that merely recovered the binary label would output a flat-high P(program) across those
timepoints; a rising trajectory means the model captures the GRADED temporal progression
(cells at 8h are less-committed than at 7d) beyond the binary label -- i.e. the thesis's
time-integrated emergence. Reported: mean P(program) per timepoint + Spearman(P, time)
over all cells and over program-labelled cells only (the continuum test).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .device import pick_device
from .dynamics import _load, _mask_input, train, class_weights, predict_proba
from .resistance import ResistanceToggle
from .kg import DATA_DIR, load_kg
from .oracle import QUIESCENT, all_classes


def main():
    ap = argparse.ArgumentParser(description="Temporal-trajectory validation of program emergence")
    ap.add_argument("--pool", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--target", default=str(DATA_DIR / "gse147405_emt.csv"),
                    help="time-course dataset with a 'timepoint' column")
    ap.add_argument("--program", default="EMT")
    ap.add_argument("--mask", choices=["none", "no_markers", "lineage_only"], default="no_markers")
    ap.add_argument("--model", choices=["logreg", "gnn"], default="logreg")
    ap.add_argument("--hold-out-target", action="store_true",
                    help="drop the target dataset from training (honest cross-dataset view)")
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-edges", action="store_true",
                    help="remove the graph (markers-without-structure control) to isolate structure")
    ap.add_argument("--attractor-mode",
                    choices=["none", "hard_wta", "soft", "delayed_soft", "learned"], default="soft")
    ap.add_argument("--plasticity-mode",
                    choices=["amplify", "lower_resistance", "both", "none"], default="lower_resistance")
    ap.add_argument("--alpha-memory", choices=["zero", "low", "learned", "full"], default="learned")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="GNN minibatch; BIG (1024-4096) is much faster on GPU (launch-bound model)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(reduce-overhead) the GNN training forward (cuda only)")
    args = ap.parse_args()

    kg = load_kg()
    classes = all_classes(kg)
    if args.program not in classes:
        raise SystemExit(f"{args.program} not in classes {classes}")
    prog_i = classes.index(args.program)

    # --- target time-course cells ---
    tdf = pd.read_csv(args.target)
    if "timepoint" not in tdf.columns:
        raise SystemExit(f"{args.target} has no 'timepoint' column (re-ingest with time_of)")
    Xt, yt, _, tdf = _load(args.target, kg)
    Xt = _mask_input(Xt, kg, args.mask)
    tp = pd.to_numeric(tdf["timepoint"], errors="coerce").to_numpy()
    keep = ~np.isnan(tp)
    Xt, tp = Xt[torch.tensor(np.where(keep)[0])], tp[keep]
    tlab = tdf["label"].to_numpy()[keep]

    # --- training pool ---
    Xp, yp, _, pdf = _load(args.pool, kg)
    Xp = _mask_input(Xp, kg, args.mask)
    if args.hold_out_target and "dataset" in pdf.columns:
        tgt_stem = Path(args.target).stem
        mask = pdf["dataset"].to_numpy() != tgt_stem
        Xp, yp = Xp[torch.tensor(np.where(mask)[0])], yp[torch.tensor(np.where(mask)[0])]
        print(f"held out '{tgt_stem}' from training ({int((~mask).sum())} rows dropped)")
    if args.subsample and Xp.size(0) > args.subsample:
        idx = torch.randperm(Xp.size(0), generator=torch.Generator().manual_seed(0))[:args.subsample]
        Xp, yp = Xp[idx], yp[idx]

    # --- train + predict P(program) on target cells ---
    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(Xp.numpy(), yp.numpy())
        col = list(m.classes_).index(prog_i)
        p_prog = m.predict_proba(Xt.numpy())[:, col]
    else:
        dev = pick_device(args.device)
        w = class_weights(yp, len(classes))
        torch.manual_seed(args.seed)
        model = ResistanceToggle(kg, hidden=args.hidden, steps=args.steps,
                                 attractor=args.attractor_mode, plasticity_mode=args.plasticity_mode,
                                 alpha_memory=args.alpha_memory).to(dev)
        if args.no_edges:
            model.adjacency.zero_()
        train(model, Xp, yp, args.epochs, args.batch_size, 1e-3, args.seed, weights=w, compile=args.compile)
        p_prog = predict_proba(model, Xt)[:, prog_i].numpy()

    # --- trajectory: mean P(program) per real timepoint ---
    from scipy.stats import spearmanr
    print(f"\nTemporal trajectory | target={Path(args.target).name} program={args.program} "
          f"model={args.model} mask={args.mask} hold_out={args.hold_out_target}\n")
    print(f"{'time (days)':>12}{'n':>7}{'label':>12}{'mean P('+args.program+')':>18}")
    print("-" * 49)
    for t in sorted(set(tp)):
        sel = tp == t
        lab = pd.Series(tlab[sel]).mode().iat[0]
        print(f"{t:>12.3f}{int(sel.sum()):>7}{lab:>12}{p_prog[sel].mean():>18.3f}")

    rho_all, p_all = spearmanr(tp, p_prog)
    prog_sel = tlab == args.program                      # the continuum test (program-only cells)
    if prog_sel.sum() > 2 and len(set(tp[prog_sel])) > 1:
        rho_prog, p_prog_p = spearmanr(tp[prog_sel], p_prog[prog_sel])
    else:
        rho_prog, p_prog_p = float("nan"), float("nan")
    print(f"\nSpearman(P, time), ALL cells:            rho={rho_all:+.3f}  p={p_all:.2e}")
    print(f"Spearman(P, time), {args.program}-labelled only: rho={rho_prog:+.3f}  p={p_prog_p:.2e}")
    print("\nRising P among same-labelled cells = model captures GRADED temporal emergence")
    print("beyond the binary label (thesis: time-integrated program commitment).")
    print("Caveat: for time-course datasets the 0-timepoint IS the Quiescent label, so the")
    print("all-cells correlation partly reflects labelling; the program-only correlation is")
    print("the non-circular test.")


if __name__ == "__main__":
    main()
