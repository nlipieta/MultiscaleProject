"""Signed-GRN DYNAMICAL model (the Waddington-landscape realization of the framework).

Fixes the design choices that made perturbations not propagate in the classifier:
 - node ACTIVITIES x_i evolve as a dynamical system (not H-dim hidden states injected-and-held);
 - edges carry FIXED literature SIGNS (ACTIVATES=+1, INHIBITS/EXPORTS=-1) with learned magnitudes
   -> the system can actually propagate activation/inhibition;
 - NO hybrid skip -> the readout depends only on the settled dynamics, so a perturbation must propagate;
 - expression = INITIAL CONDITION that evolves to a fixed point; attractors = programs = landscape
   minima;
 - perturbation (knockdown) = CLAMP the node to 0 throughout evolution and re-settle -> downstream
   targets lose their driver and drop, so the KD propagates (unlike the classifier, which healed it).

Dynamics:  x <- (1-eta)*x + eta*tanh( W_signed @ x + bias ),  W_signed[d,s] = sign(s->d)*softplus(mag).
Readout:   class logits = program-node steady activities (+ a competing Quiescent logit).

Usage (train + Q2 clamp-and-resettle perturbation validation):
  uv run python scripts/grn_dynamics.py --train data/bmmc_shareseq.csv --pert data/replogle_k562.csv --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from chromatin_toggle.device import pick_device
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import QUIESCENT, all_classes

from chromatin_toggle.grn import GRNDynamics  # model lives in the package now

def _load(path, kg, need_pert=False):
    df = pd.read_csv(path); classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    cols = list(kg.node_ids); X = np.zeros((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = torch.tensor(X)
    y = torch.tensor([ci[l] for l in df["label"]]) if "label" in df else None
    pert = df["perturbation"].to_numpy() if need_pert else None
    return X, y, pert


@torch.no_grad()
def _p(m, X, dev, prog_i, clamp_idx=None, bs=2048):
    m.eval()
    return torch.cat([torch.softmax(m(X[i:i+bs].to(dev), clamp_idx=clamp_idx), -1)[:, prog_i].cpu()
                      for i in range(0, X.size(0), bs)]).numpy()


def main():
    ap = argparse.ArgumentParser(description="signed-GRN dynamical model: train + Q2 clamp perturbation")
    ap.add_argument("--train", default=str(DATA_DIR / "bmmc_shareseq.csv"))
    ap.add_argument("--pert", default=str(DATA_DIR / "replogle_k562.csv"))
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    kg = load_kg(); dev = pick_device(a.device)
    classes = all_classes(kg); prog_i = classes.index("Erythropoiesis")

    Xtr, ytr, _ = _load(a.train, kg)
    torch.manual_seed(a.seed)
    m = GRNDynamics(kg, steps=a.steps, eta=a.eta).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    from chromatin_toggle.dynamics import class_weights
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr, len(classes)).to(dev))
    Xtr, ytr = Xtr.to(dev), ytr.to(dev); n = Xtr.size(0)
    g = torch.Generator().manual_seed(a.seed)
    print(f"training signed-GRN dynamical model (steps={a.steps} eta={a.eta}) on {Path(a.train).name} n={n} ...")
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for i in range(0, n, a.batch_size):
            idx = perm[i:i+a.batch_size]; opt.zero_grad()
            loss = lossf(m(Xtr[idx]), ytr[idx]); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep}: loss {tot/max(1,n//a.batch_size):.3f}", flush=True)

    # classification sanity on train
    from sklearn.metrics import average_precision_score
    P_tr = _p(m, Xtr.cpu(), dev, prog_i); ytr_np = ytr.cpu().numpy()
    print(f"\ntrain erythroid AUPRC (sanity the GRN models the state): "
          f"{average_precision_score((ytr_np==prog_i).astype(int), P_tr):.3f}")

    # Q2: clamp-and-resettle perturbation vs real Replogle
    Xp, _, pert = _load(a.pert, kg, need_pert=True)
    ctrl = pert == "control"
    P = _p(m, Xp, dev, prog_i); p_ctrl = P[ctrl].mean()
    Xc = Xp[torch.tensor(np.where(ctrl)[0])]
    print(f"\ncontrol P(Erythropoiesis) = {p_ctrl:.3f}")
    print(f"{'target':>8}{'n_KD':>7}{'real dP':>10}{'GRN clamp dP':>14}{'sign':>7}")
    print("-" * 46)
    for t in ["GATA1", "TAL1", "KLF1", "LMO2"]:
        if not (pert == t).any():
            continue
        real_dp = P[pert == t].mean() - p_ctrl
        node = t if t in kg.node_index else next((nn_ for nn_, s in kg.gene_map.items()
                                                  if str(s).upper() == t), None)
        if node is None:
            print(f"{t:>8}{int((pert==t).sum()):>7}   (not a KG node)"); continue
        clamp_dp = _p(m, Xc, dev, prog_i, clamp_idx=kg.node_index[node]).mean() - p_ctrl
        match = "yes" if np.sign(real_dp) == np.sign(clamp_dp) and abs(real_dp) > 0.005 else "—"
        print(f"{t:>8}{int((pert==t).sum()):>7}{real_dp:>+10.3f}{clamp_dp:>+14.3f}{match:>7}")
    print("\nGRN clamp dP = knockdown propagated through the signed graph to a new fixed point.")
    print("Does it reach the real magnitude (vs the classifier's healed ~-0.03)? That's the test of")
    print("whether the dynamical/landscape design closes the gap. (Also check the train AUPRC first --")
    print("if the GRN can't even model erythroid state, the perturbation numbers aren't meaningful.)")


if __name__ == "__main__":
    main()
