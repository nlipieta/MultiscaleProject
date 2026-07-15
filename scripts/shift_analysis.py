"""What does a shift require? -- in-silico perturbation screen on the metastable multistable GRN,
validated against real CRISPRi knockdowns (framework Q2, honestly scoped to DIRECTION + RANK + FLIP).

The multistable GRN gives metastable fate basins at the regulatory (operating) timescale. A "shift" is a
perturbation that moves cells OUT of their basin toward another fate. This screens every node: clamp it,
re-settle the erythroid cells at the operating horizon, and measure the fraction that LEAVE the erythroid
basin (flip rate). Ranking the nodes = "what a shift requires". We then check the model's predicted
shift-drivers against real Replogle K562 knockdowns:
  - REAL   : erythroid-basin fraction among real KD cells  vs  non-targeting controls
  - IN-SILICO: erythroid-basin fraction among control cells with that node clamped  vs  controls
Validation = SIGN agreement (KD lowers erythroid-basin occupancy) + RANK agreement (which regulators are
most potent). NOT magnitude -- magnitude is topology-governed (established), so we report direction/rank.

Run:  uv run python scripts/shift_analysis.py --data data/cross_pathway_eval.csv \
        --pert data/replogle_k562.csv --classes Erythropoiesis Megakaryopoiesis --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from chromatin_toggle.device import pick_device
from chromatin_toggle.grn import MultistableGRN
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import all_classes


def _load(path, kg, need_pert=False):
    df = pd.read_csv(path); classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    cols = list(kg.node_ids); X = np.zeros((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = torch.tensor(X)
    y = torch.tensor([ci[l] for l in df["label"]]) if "label" in df and not need_pert else None
    pert = df["perturbation"].to_numpy() if need_pert else None
    return X, y, pert


def _node_for(target, kg):
    if target in kg.node_index:
        return target
    return next((n for n, s in kg.gene_map.items() if str(s).upper() == target), None)


def main():
    ap = argparse.ArgumentParser(description="in-silico shift screen on the metastable multistable GRN")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--pert", default=str(DATA_DIR / "replogle_k562.csv"))
    ap.add_argument("--classes", nargs="+", default=["Erythropoiesis", "Megakaryopoiesis"])
    ap.add_argument("--target-class", default="Erythropoiesis", help="basin to shift cells OUT of")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lambda-fp", type=float, default=5.0)
    ap.add_argument("--lambda-basin", type=float, default=5.0)
    ap.add_argument("--lambda-anchor", type=float, default=2.0)
    ap.add_argument("--n-cells", type=int, default=1000, help="erythroid cells to screen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    kg = load_kg(); dev = pick_device(a.device)
    X, y, _ = _load(a.data, kg); classes = all_classes(kg)
    counts = {classes[int(i)]: int((y == i).sum()) for i in torch.unique(y)}
    keep = [classes.index(c) for c in a.classes]
    mask = torch.tensor([int(v) in keep for v in y])
    X, y = X[mask], y[mask]
    if X.size(0) == 0 or any(int((y == i).sum()) == 0 for i in keep):
        avail = ", ".join(f"{k}({v})" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        raise SystemExit(f"ERROR: {a.data} has no cells for {a.classes} "
                         f"(per-class: {[(classes[i], int((y==i).sum())) for i in keep]}).\n"
                         f"Available programs in this file: {avail}\n"
                         f"Pick --classes from two well-populated, well-separated programs above.")
    proto_init = torch.stack([X[y == i].mean(0) for i in keep])

    torch.manual_seed(a.seed)
    m = MultistableGRN(kg, a.classes, proto_init=proto_init, steps=a.steps, eta=a.eta).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3); ce = nn.CrossEntropyLoss()
    Xd, yd = X.to(dev), y.to(dev); n = Xd.size(0); g = torch.Generator().manual_seed(a.seed)
    print(f"training metastable GRN on {a.classes} (n={n}) ...")
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, a.batch_size):
            idx = perm[i:i+a.batch_size]; opt.zero_grad()
            loss = (m.flow_loss(Xd[idx], yd[idx]) + a.lambda_fp * m.fp_loss()
                    + a.lambda_basin * m.basin_loss() + a.lambda_anchor * m.anchor_loss()
                    + 0.1 * ce(m(Xd[idx]), yd[idx]))
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval()

    tgt_local = a.classes.index(a.target_class)
    ery = Xd[yd == keep[a.classes.index(a.target_class)]][:a.n_cells]
    base_frac = float((m.assign(ery) == tgt_local).float().mean())
    print(f"\nbaseline: {100*base_frac:.0f}% of {a.target_class} cells sit in the {a.target_class} basin "
          f"(operating horizon, {a.steps} steps)")

    # ---- in-silico shift screen: clamp each node (held knockdown), measure fraction LEAVING the basin ----
    rows = []
    with torch.no_grad():
        for j in range(m.N):
            asg = m.assign(ery, clamp_idx=j)
            leave = base_frac - float((asg == tgt_local).float().mean())
            rows.append((kg.node_ids[j], leave))
    rows.sort(key=lambda r: -r[1])
    print(f"\nWHAT A SHIFT REQUIRES -- top knockdowns that move {a.target_class} cells out of their basin:")
    print(f"  {'node':>12}  frac-leaving")
    for name, leave in rows[:15]:
        print(f"  {name:>12}  {leave:+.3f}")

    # ---- validate the shift-drivers against real Replogle knockdowns ----
    if Path(a.pert).exists():
        Xp, _, pert = _load(a.pert, kg, need_pert=True)
        Xp = Xp.to(dev); ctrl = pert == "control"
        base_p = float((m.assign(Xp[torch.tensor(np.where(ctrl)[0])]) == tgt_local).float().mean())
        Xc = Xp[torch.tensor(np.where(ctrl)[0])]
        rank = {name: i for i, (name, _) in enumerate(rows)}
        print(f"\nVALIDATION vs Replogle (control erythroid-basin frac {100*base_p:.0f}%):")
        print(f"  {'target':>8}{'real dFrac':>12}{'in-silico dFrac':>16}{'in-silico rank':>16}{'sign':>7}")
        real_rows, insil_rows = [], []
        for t in ["GATA1", "TAL1", "KLF1", "LMO2"]:
            if not (pert == t).any():
                continue
            real_d = float((m.assign(Xp[torch.tensor(np.where(pert == t)[0])]) == tgt_local).float().mean()) - base_p
            node = _node_for(t, kg)
            if node is None:
                print(f"  {t:>8}{real_d:>+12.3f}      (not a KG node)"); continue
            with torch.no_grad():
                insil_d = float((m.assign(Xc, clamp_idx=kg.node_index[node]) == tgt_local).float().mean()) - base_p
            match = "yes" if np.sign(real_d) == np.sign(insil_d) and abs(real_d) > 0.005 else "-"
            print(f"  {t:>8}{real_d:>+12.3f}{insil_d:>+16.3f}{rank.get(node,'-'):>16}{match:>7}")
            real_rows.append(real_d); insil_rows.append(insil_d)
        if len(real_rows) >= 3:
            from scipy.stats import spearmanr
            rho = spearmanr(real_rows, insil_rows).correlation
            print(f"\n  rank agreement (Spearman real vs in-silico dFrac): rho={rho:+.2f} (n={len(real_rows)})")
    print("\nREAD: shift-drivers = knockdowns that empty the target basin; validated by SIGN + RANK vs real")
    print("      KDs (not magnitude). This answers 'which perturbations move the system between states'.")


if __name__ == "__main__":
    main()
