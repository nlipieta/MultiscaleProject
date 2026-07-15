"""Q2 validation: does the model's IN-SILICO perturbation match the REAL experimental outcome?

Framework question 2 ("which perturbations move the system between states"), validated against real
CRISPRi knockdowns (Replogle K562). For each erythroid regulator the model encodes (GATA1/TAL1/KLF1/
LMO2):
  - REAL effect      = mean P(Erythropoiesis) on real KD cells  -  on non-targeting controls
  - IN-SILICO effect = mean P(Erythropoiesis) on control cells with that node's input ZEROED  -  controls
A knockdown of an erythroid regulator should LOWER erythroid identity; validation = the two effects
agree in SIGN (and roughly magnitude). Sign agreement on held-out real perturbations = the model's
perturbation predictions reflect reality, not just in-silico self-consistency.

Model is trained on the erythroid multiome (Erythropoiesis vs Quiescent); Replogle (K562) is the
held-out perturbation experiment. Both use KG-node columns, so features align.

Run:  uv run python scripts/perturb_validate.py --train data/bmmc_shareseq.csv --pert data/replogle_k562.csv --device cuda
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
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import all_classes
from chromatin_toggle.resistance import ResistanceToggle


def _load_nodes(path, kg, need_pert=False):
    df = pd.read_csv(path)
    classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    cols = list(kg.node_ids)
    X = np.zeros((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = torch.tensor(X)
    y = torch.tensor([ci[l] for l in df["label"]]) if "label" in df and not need_pert else None
    pert = df["perturbation"].to_numpy() if need_pert else None
    return X, y, pert


@torch.no_grad()
def _p(m, X, dev, prog_i, n_steps=None, bs=2048):
    m.eval()
    return torch.cat([torch.softmax(m(X[i:i+bs].to(dev), n_steps=n_steps), -1)[:, prog_i].cpu()
                      for i in range(0, X.size(0), bs)]).numpy()


def main():
    ap = argparse.ArgumentParser(description="Q2: in-silico vs real perturbation validation")
    ap.add_argument("--train", default=str(DATA_DIR / "bmmc_shareseq.csv"))
    ap.add_argument("--pert", default=str(DATA_DIR / "replogle_k562.csv"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--relax-steps", type=int, default=None,
                    help="also compute in-silico dP with the perturbed state RELAXED for N steps "
                         "(attractor settling / landscape prototype) -- does it grow toward the real dP?")
    a = ap.parse_args()
    kg = load_kg(); dev = pick_device(a.device)
    classes = all_classes(kg); prog_i = classes.index("Erythropoiesis")

    Xtr, ytr, _ = _load_nodes(a.train, kg)
    Xp, _, pert = _load_nodes(a.pert, kg, need_pert=True)
    from chromatin_toggle.dynamics import train
    torch.manual_seed(a.seed)
    m = ResistanceToggle(kg, hidden=a.hidden, steps=a.steps).to(dev)
    w = class_weights(ytr, len(classes))
    print(f"training erythroid model on {Path(a.train).name} (n={Xtr.size(0)}) ...")
    train(m, Xtr, ytr, a.epochs, a.batch_size, 1e-3, a.seed, weights=w, amp=a.amp)

    ctrl = pert == "control"
    P = _p(m, Xp, dev, prog_i)
    p_ctrl = P[ctrl].mean()
    Xc = Xp[torch.tensor(np.where(ctrl)[0])]
    relax = a.relax_steps
    p_ctrl_relax = _p(m, Xc, dev, prog_i, n_steps=relax).mean() if relax else None  # relaxed baseline
    print(f"\ncontrol P(Erythropoiesis) baseline = {p_ctrl:.3f}  (n={int(ctrl.sum())} non-targeting cells)")
    if relax:
        print(f"control baseline @ relax {relax} steps = {p_ctrl_relax:.3f}")
    rlx_col = f"{'in-silico dP@relax'+str(relax):>20}" if relax else ""
    print(f"\n{'target':>8}{'n_KD':>7}{'real dP':>10}{'in-silico dP':>14}{rlx_col}{'sign':>7}")
    print("-" * (51 + (20 if relax else 0)))
    targets = [t for t in ["GATA1", "TAL1", "KLF1", "LMO2"] if (pert == t).any()]
    for t in targets:
        real_dp = P[pert == t].mean() - p_ctrl                      # real KD cells vs control
        node = t if t in kg.node_index else next((n for n, s in kg.gene_map.items()
                                                   if str(s).upper() == t), None)
        if node is None:
            print(f"{t:>8}{int((pert==t).sum()):>7}   (not a KG node — skip in-silico)")
            continue
        Xz = Xc.clone(); Xz[:, kg.node_index[node]] = 0.0           # in-silico KD: zero the node input
        insilico_dp = _p(m, Xz, dev, prog_i).mean() - p_ctrl
        match = "yes" if np.sign(real_dp) == np.sign(insilico_dp) and abs(real_dp) > 0.005 else "—"
        rlx = ""
        if relax:                                                   # perturbed state relaxed to settling
            insilico_relax = _p(m, Xz, dev, prog_i, n_steps=relax).mean() - p_ctrl_relax
            rlx = f"{insilico_relax:>+20.3f}"
        print(f"{t:>8}{int((pert==t).sum()):>7}{real_dp:>+10.3f}{insilico_dp:>+14.3f}{rlx}{match:>7}")
    if relax:
        print(f"\nRELAX = perturbed state run to {relax} steps (attractor settling). If in-silico dP@relax")
        print("grows toward the real dP, letting the perturbation propagate/settle closes the magnitude gap")
        print("-> motivates the explicit potential-landscape model.")
    print("\nBoth negative = KD lowers erythroid identity, in-silico agrees with the real experiment.")
    print("Sign agreement on held-out real knockdowns = Q2 validated (model predicts perturbation")
    print("outcomes, not just self-consistent in-silico edits). Caveat: K562 cell line, cross-dataset;")
    print("interpret the DIRECTION, and check control P(Ery) is sensibly high before trusting deltas.")


if __name__ == "__main__":
    main()
