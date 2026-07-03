"""Predict the phenotype a cue drives in a given cell context, with a
mechanistic trace.

Usage:
    chromatin-predict --context myoblast --cue TGFbeta --level high
    chromatin-predict --context epithelial --cue MechanicalStiffness
    chromatin-predict --on MyoD,Oct4 --cue TGFbeta          # custom memory set
    chromatin-predict --list                                 # show contexts & cues
"""
from __future__ import annotations

import argparse

import torch

from .device import pick_device
from .inputs import LEVELS, build_input, context_input, row_input
from .kg import load_contexts, load_kg
from .model import ToggleGNN
from .oracle import trace


def load_model(path: str, kg, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ToggleGNN(kg, hidden=ckpt["hidden"], steps=ckpt["steps"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict cell-state bias")
    ap.add_argument("--model", default="artifacts/model.pt")
    ap.add_argument("--context", default=None, help="named cell context")
    ap.add_argument("--real-context", default=None,
                    help="context name from a CELLxGENE-grounded contexts CSV")
    ap.add_argument("--contexts-csv", default="data/cellxgene_contexts.csv",
                    help="CSV built by chromatin-census")
    ap.add_argument("--on", default=None, help="comma-separated ON memory nodes")
    ap.add_argument("--cue", default=None, help="extrinsic cue node")
    ap.add_argument("--level", default="high", help="off|low|med|high or a float")
    ap.add_argument("--kg", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--list", action="store_true", help="list contexts and cues")
    args = ap.parse_args()

    kg = load_kg(args.kg)
    contexts = load_contexts()
    cues = [n for n in kg.node_ids if kg.node_type[kg.node_index[n]] == "cue"]

    if args.list:
        print("Contexts:")
        for c, on in contexts.items():
            print(f"  {c:<14} memory ON: {on}")
        print("\nCues:", ", ".join(cues))
        print("Levels:", ", ".join(LEVELS))
        return

    device = pick_device(args.device)
    model, classes = load_model(args.model, kg, device)

    level = args.level
    try:
        level = float(level)
    except ValueError:
        pass

    if args.real_context is not None:
        import pandas as pd

        df = pd.read_csv(args.contexts_csv)
        match = df[df["context"] == args.real_context]
        if match.empty:
            ap.error(f"{args.real_context!r} not in {args.contexts_csv}; "
                     f"have {list(df['context'])}")
        vals = {c: float(match.iloc[0][c]) for c in df.columns if c != "context"}
        x = row_input(kg, vals, args.cue, level)
        baseline = row_input(kg, vals, None)
        ctx_desc = f"real-context={args.real_context} (CELLxGENE-grounded memory)"
    elif args.on is not None:
        on_nodes = [s.strip() for s in args.on.split(",") if s.strip()]
        x = build_input(kg, on_nodes, args.cue, level)
        baseline = build_input(kg, on_nodes, None)
        ctx_desc = f"memory={on_nodes}"
    elif args.context is not None:
        x = context_input(kg, contexts, args.context, args.cue, level)
        baseline = context_input(kg, contexts, args.context, None)
        ctx_desc = f"context={args.context} (memory {contexts[args.context]})"
    else:
        ap.error("provide --context NAME, --real-context NAME, "
                 "or --on NODE1,NODE2 (or --list)")

    with torch.no_grad():
        probs = torch.softmax(model(x.unsqueeze(0).to(device)), -1)[0].cpu()
    ranked = sorted(zip(classes, probs.tolist()), key=lambda t: t[1], reverse=True)

    print(f"\nInput: {ctx_desc}, cue={args.cue}:{args.level}")
    print("\nPredicted phenotype (GNN):")
    for name, p in ranked[:5]:
        bar = "#" * int(round(p * 30))
        print(f"  {name:<14} {p:6.3f} {bar}")

    label, nodes = trace(kg, x, baseline=baseline)
    print("\nMechanistic trace (nodes the cue moved vs no-cue baseline):")
    print(f"  winning program: {label}")
    for n, v, d in nodes:
        arrow = "up" if d > 0 else "dn"
        print(f"    {n:<20} {v:.2f}  ({arrow} {d:+.2f})")


if __name__ == "__main__":
    main()
