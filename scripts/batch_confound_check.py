"""Is the erythroid-vs-myeloid basin separation biology, or a BATCH artifact?

The fate bifurcation comes from GSE194122, where both fates span all 13 donor/site batches -- so it
SHOULD be batch-clean, but the project has a leakage history, so we verify. Two checks:

  (1) per-batch fate accuracy -- do cells separate by fate WITHIN every batch (uniform = biology)?
  (2) leave-batches-out -- train the metastable model on HALF the batches, then assign cells from the
      HELD-OUT batches (never seen in training). If unseen-batch accuracy ~= seen-batch accuracy, the
      basins track fate, not batch. A big drop on unseen batches would mean batch memorization.

Run:  uv run python scripts/batch_confound_check.py --data data/bmmc_ery_myeloid.csv \
        --classes Erythropoiesis MacrophageActivation --device cuda
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


def _load(path, kg):
    df = pd.read_csv(path); classes = all_classes(kg); ci = {c: i for i, c in enumerate(classes)}
    cols = list(kg.node_ids); X = np.zeros((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            X[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    y = np.array([ci[l] for l in df["label"]])
    batch = df["batch"].astype(str).to_numpy() if "batch" in df else np.array(["na"] * len(df))
    return torch.tensor(X), torch.tensor(y), batch, classes


def _train(m, Xd, yd, epochs, bs, seed, dev):
    # lambdas matched to scripts/train_multistable.py defaults (1/1/1) so this test reflects the same
    # config that separates fates in-sample (mismatched reg previously gave a misleadingly low number).
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3); ce = nn.CrossEntropyLoss()
    n = Xd.size(0); g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = (m.flow_loss(Xd[idx], yd[idx]) + m.fp_loss() + m.basin_loss()
                    + m.anchor_loss() + 0.1 * ce(m(Xd[idx]), yd[idx]))
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval()


def _report(tag, pred, yl, sel):
    p, t = pred[sel], yl[sel]
    sens = [float((p[t == c] == c).float().mean()) if (t == c).any() else float("nan") for c in (0, 1)]
    bal = float(np.nanmean(sens))
    print(f"    {tag:>7}: balanced acc {bal:.3f}  (fate0 sens {sens[0]:.3f}, fate1 sens {sens[1]:.3f}, "
          f"n={int(sel.sum())})")


def main():
    ap = argparse.ArgumentParser(description="batch-confound check for the fate basins")
    ap.add_argument("--data", default=str(DATA_DIR / "bmmc_ery_myeloid.csv"))
    ap.add_argument("--classes", nargs="+", default=["Erythropoiesis", "MacrophageActivation"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    kg = load_kg(); dev = pick_device(a.device)
    X, y, batch, classes = _load(a.data, kg)
    keep = [classes.index(c) for c in a.classes]
    mask = np.array([int(v) in keep for v in y])
    X, y, batch = X[mask], y[mask], batch[mask]
    local = {c: i for i, c in enumerate(keep)}
    yl = torch.tensor([local[int(v)] for v in y])                 # fate index 0/1
    batches = sorted(set(batch))
    print(f"{X.size(0)} cells, fates={a.classes}, {len(batches)} batches")

    # ---- (2) leave-batches-out: train on half the batches, test on the held-out half ----
    train_b = set(batches[::2]); test_b = set(batches[1::2])
    tr = np.array([b in train_b for b in batch]); te = ~tr
    proto_init = torch.stack([X[tr][yl[tr] == i].mean(0) for i in range(len(keep))])
    torch.manual_seed(a.seed)
    m = MultistableGRN(kg, a.classes, proto_init=proto_init).to(dev)
    _train(m, X[tr].to(dev), yl[tr].to(dev), a.epochs, a.batch_size, a.seed, dev)
    with torch.no_grad():
        pred = torch.cat([m.assign(X[i:i+4096].to(dev)).cpu() for i in range(0, X.size(0), 4096)])
    print(f"\n(2) LEAVE-BATCHES-OUT (train on {len(train_b)} batches, test on {len(test_b)} unseen); "
          f"BALANCED acc + per-fate sensitivity:")
    _report("SEEN", pred, yl, tr)
    _report("UNSEEN", pred, yl, te)
    print("    => generalizes (biology) if UNSEEN balanced acc stays high and ~= SEEN; a big drop or a")
    print("       fate sensitivity ->0 on unseen batches = batch/donor dependence, not portable fate.")

    # ---- (1) per-batch fate accuracy (uniform high across batches = biology) ----
    print(f"\n(1) PER-BATCH fate accuracy (uniform => separation is fate, not batch):")
    print(f"    {'batch':>8}{'n':>7}{'%'+a.classes[0][:4]:>8}{'acc':>7}")
    for b in batches:
        sel = batch == b
        if sel.sum() == 0:
            continue
        acc_b = float((pred[torch.tensor(sel)] == yl[torch.tensor(sel)]).float().mean())
        f0 = float((yl[torch.tensor(sel)] == 0).float().mean())
        tag = "" if 0.05 < f0 < 0.95 else "  (single-fate batch)"
        print(f"    {b:>8}{int(sel.sum()):>7}{100*f0:>8.0f}{acc_b:>7.2f}{tag}")
    print("\nREAD: high + uniform per-batch accuracy AND unseen-batch ~= seen-batch => basins track FATE,")
    print("      not batch -> the multistability result is not a batch artifact.")


if __name__ == "__main__":
    main()
