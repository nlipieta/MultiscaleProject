"""Are the resistance/plasticity/attractor mechanisms ACTIVE, or dormant-by-setup?

The static classification benchmark runs with cue=None and plasticity=const, so the thesis's
cue->plasticity->resistance->flip loop is never exercised. This measures, on a trained model,
whether each mechanism actually varies across cells (active) or is degenerate (flat = it CANNOT
contribute, so its ablation being ~0 is tautological, not evidence against the thesis).

Reports, on a held-out batch:
  - base resistance per cell: mean / std / range   (std~0 => flat, not discriminating)
  - plasticity effect per cell: mean / std          (std~0 => provably dormant on this task)
  - learned ||W_resist|| / ||W_plast||              (~0 => never learned away from init)
  - attractor: mean |logit change| it applies       (~0 => attractor doing nothing)

Run:  uv run python scripts/diagnose_mechanisms.py --device cuda --amp
"""
from __future__ import annotations

import argparse

import torch

from chromatin_toggle.device import pick_device
from chromatin_toggle.dynamics import _load, _mask_input, class_weights, train
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.resistance import ResistanceToggle


@torch.no_grad()
def mechanism_report(m, X, plasticity=1.0):
    """Recompute the step-0 mechanism quantities the forward pass uses, per cell."""
    dev = next(m.parameters()).device
    X = X.to(dev)
    B = X.size(0)
    base = (m.id_emb(m.node_ids) + m.type_emb(m.node_types)).unsqueeze(0)
    if m.ann_proj is not None:
        base = base + m.ann_proj(m.node_ann).unsqueeze(0)
    xin = m.in_proj(X.unsqueeze(-1))
    h = base.expand(B, -1, -1).contiguous() + xin * m.intrinsic_mask
    feat = torch.cat([m._pool(h, m.chromatin_idx), m._pool(h, m.lineage_idx),
                      m._pool(h, m.program_index)], dim=-1)
    base_r = torch.sigmoid(m.W_resist(feat)).squeeze(-1)                 # [B]
    plast_in = torch.full((B, 1), float(plasticity), device=dev)
    plast_eff = torch.sigmoid(m.W_plast(torch.cat([m._pool(h, m.plasticity_idx), plast_in], -1))).squeeze(-1)
    # attractor effect: how much the soft attractor moves the logits
    logits = m(X, plasticity=plasticity)
    m_noattr_mode, m.attractor = m.attractor, "none"
    logits_noattr = m(X, plasticity=plasticity)
    m.attractor = m_noattr_mode
    attr_delta = (logits - logits_noattr).abs().mean().item()
    return base_r, plast_eff, attr_delta


def main():
    ap = argparse.ArgumentParser(description="mechanism-activity diagnostic")
    ap.add_argument("--data", default=str(DATA_DIR / "cross_pathway_eval.csv"))
    ap.add_argument("--mask", default="no_markers")
    ap.add_argument("--subsample", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    a = ap.parse_args()
    dev = pick_device(a.device)

    kg = load_kg()
    X, y, classes, _ = _load(a.data, kg)
    X = _mask_input(X, kg, a.mask)
    if a.subsample and X.size(0) > a.subsample:
        idx = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))[:a.subsample]
        X, y = X[idx], y[idx]
    n = X.size(0); ntr = int(n * 0.8)
    w = class_weights(y[:ntr], len(classes))
    torch.manual_seed(a.seed)
    m = ResistanceToggle(kg, hidden=a.hidden, steps=a.steps).to(dev)
    print(f"training (n={ntr}, ep{a.epochs}, amp={a.amp}) ...")
    train(m, X[:ntr], y[:ntr], a.epochs, a.bs, 1e-3, a.seed, weights=w, amp=a.amp)

    base_r, plast_eff, attr_delta = mechanism_report(m, X[ntr:])
    def stat(t): return f"mean {t.mean():.4f}  std {t.std():.4f}  range [{t.min():.4f}, {t.max():.4f}]"
    print("\n=== MECHANISM ACTIVITY (held-out cells) ===")
    print(f"resistance (per-cell barrier):  {stat(base_r)}")
    print(f"plasticity effect (per-cell) :  {stat(plast_eff)}")
    print(f"  -> plasticity std ~0 means it is CONSTANT across cells (dormant on this cue-free task)")
    print(f"attractor mean |logit shift| :  {attr_delta:.4f}   (~0 => attractor inactive)")
    wr = m.W_resist.weight.norm().item(); wp = m.W_plast.weight.norm().item()
    print(f"learned ||W_resist|| {wr:.4f}   ||W_plast|| {wp:.4f}   (started at 0; ~0 => never learned)")
    print("\nread: flat resistance / zero-variance plasticity / ~0 weights => the mechanism is")
    print("dormant-by-SETUP, so its ablation being ~0 is tautological, NOT evidence against the thesis.")
    print("A fair test must EXERCISE the mechanisms (cue on / plasticity varying / a transition target).")


if __name__ == "__main__":
    main()
