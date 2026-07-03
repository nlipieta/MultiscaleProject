"""Train the ToggleGNN and evaluate it against the literature anchor cases.

Usage:
    chromatin-train                       # bootstrap data, default hyperparams
    chromatin-train --epochs 300 --device mps
    chromatin-train --data my_real.csv    # train on real observations instead
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .dataset import build_bootstrap, load_csv, split
from .device import pick_device
from .inputs import context_input
from .kg import DATA_DIR, load_contexts, load_kg
from .model import ToggleGNN
from .oracle import all_classes

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def evaluate(model, X, y, device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device)).argmax(-1).cpu()
    return float((pred == y).float().mean())


def evaluate_anchors(model, kg, contexts, classes, device):
    cases = yaml.safe_load((DATA_DIR / "literature_cases.yaml").read_text())["cases"]
    rows, correct = [], 0
    model.eval()
    for c in cases:
        x = context_input(kg, contexts, c["context"], c["cue"], c["level"]).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x.to(device)), -1)[0].cpu()
        pred = classes[int(probs.argmax())]
        ok = pred == c["expected"]
        correct += ok
        rows.append((c["context"], c["cue"], c["expected"], pred, float(probs.max()), ok))
    return rows, correct / len(cases)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the chromatin-toggle GNN")
    ap.add_argument("--kg", default=None, help="path to kg.yaml")
    ap.add_argument("--data", default=None, help="real-data CSV (skips bootstrap)")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6, help="GNN message-passing rounds")
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ARTIFACTS / "model.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    kg = load_kg(args.kg)
    contexts = load_contexts()
    classes = all_classes(kg)
    cues = [n for n in kg.node_ids if kg.node_type[kg.node_index[n]] == "cue"]

    if args.data:
        X, y = load_csv(kg, args.data, classes)
        print(f"Loaded {X.size(0)} real observations from {args.data}")
    else:
        X, y, classes = build_bootstrap(
            kg, contexts, cues, replicas=args.replicas,
            noise=args.noise, seed=args.seed,
        )
        print(f"Built {X.size(0)} bootstrap examples "
              f"(context x cue x level, oracle-labeled). WIRING HARNESS.")

    (Xtr, ytr), (Xva, yva) = split(X, y, seed=args.seed)
    model = ToggleGNN(kg, hidden=args.hidden, steps=args.steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    Xtr_d, ytr_d = Xtr.to(device), ytr.to(device)
    print(f"device={device}  train={Xtr.size(0)}  val={Xva.size(0)}  "
          f"nodes={kg.num_nodes}  relations={kg.num_relations}  classes={len(classes)}")

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr_d), ytr_d)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == 1:
            va = evaluate(model, Xva, yva, device)
            print(f"  epoch {ep:4d}  loss {loss.item():.4f}  val_acc {va:.3f}")

    val_acc = evaluate(model, Xva, yva, device)
    rows, anchor_acc = evaluate_anchors(model, kg, contexts, classes, device)

    print(f"\nFinal val accuracy: {val_acc:.3f}")
    print(f"Literature-anchor accuracy: {anchor_acc:.3f}  "
          "(does the learned model reproduce known biology?)")
    print(f"\n{'context':<14}{'cue':<26}{'expected':<14}{'predicted':<14}{'p':>6}  ok")
    print("-" * 82)
    for ctx, cue, exp, pred, p, ok in rows:
        print(f"{ctx:<14}{cue:<26}{exp:<14}{pred:<14}{p:>6.2f}  {'YES' if ok else 'no'}")

    ARTIFACTS.mkdir(exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "hidden": args.hidden,
        "steps": args.steps,
        "kg_path": args.kg or str(DATA_DIR / "kg.yaml"),
    }
    torch.save(ckpt, args.out)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(
        {"val_acc": val_acc, "anchor_acc": anchor_acc,
         "anchors": [{"context": r[0], "cue": r[1], "expected": r[2],
                      "predicted": r[3], "prob": r[4], "correct": bool(r[5])}
                     for r in rows]}, indent=2))
    print(f"\nSaved model -> {args.out}")
    print(f"Saved metrics -> {ARTIFACTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
