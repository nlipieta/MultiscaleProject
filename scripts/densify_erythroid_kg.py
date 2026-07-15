"""Densify the erythroid regulatory cascade in kg.yaml with real, literature-curated TF->target edges.

Motivation: the erythroid subgraph was just Gata1->Erythropoiesis and Klf1->Erythropoiesis -- the master
TFs had ZERO downstream targets, so an in-silico knockdown (clamp) could not propagate and the predicted
perturbation effect was ~3x too small vs real Replogle KDs. This adds the missing TF->effector layer so a
clamp pulls its targets (and their marker->program contributions) down with it.

Edges come from TRRUST v2 (grnpedia.org) -- literature-curated, PMID-backed regulatory relationships --
frozen in data/erythroid_regulatory_edges.tsv (auditable). Restricted to edges where BOTH endpoints are
already KG nodes (erythroid TFs and erythroid marker effectors), so NO re-ingest is needed: the effector
columns already exist in the cached data. Signs: Activation->ACTIVATES, Repression->INHIBITS; TRRUST
'Unknown'-direction edges are resolved to ACTIVATES only when the target is a positive erythroid marker
(sign follows from program membership, flagged `inferred=1`).

Excluded by hand (see message/commit): PU1->Gata1 'Activation' -- contradicts the canonical GATA1<->PU.1
mutual-repression toggle central to the thesis; we keep only the canonical Gata1 -| PU1.

Idempotent: appends only edges not already present. Run locally, commit kg.yaml + the TSV.
  uv run python scripts/densify_erythroid_kg.py
"""
from __future__ import annotations

from pathlib import Path

from chromatin_toggle.kg import DATA_DIR, load_kg

TSV = DATA_DIR / "erythroid_regulatory_edges.tsv"
FEEDBACK = DATA_DIR / "erythroid_feedback_edges.tsv"
KG = DATA_DIR / "kg.yaml"
W = 1.0  # resting edge weight; the GRN/GNN learns the actual magnitude


def main():
    kg = load_kg()
    nodes = set(kg.node_ids)
    # presence check: (src, rel, dst) from the parsed edge tuples (rel is the relation string)
    have_pairs = {(kg.node_ids[s], rel, kg.node_ids[d]) for s, rel, d, w in kg.edges}

    def collect(tsv, feedback=False):
        rows = [l.rstrip("\n").split("\t") for l in tsv.read_text().splitlines()]
        out = []
        for r in rows[1:]:
            if not r or len(r) < 4:
                continue
            if feedback:
                src, rel, dst, note = r[0], r[1], r[2], r[3]
            else:
                src, rel, dst, mode, pmid, inferred = r[:6]
                note = f"TRRUST {mode} PMID:{pmid}" + (" [dir inferred from marker]" if inferred == "1" else "")
            if src not in nodes or dst not in nodes:
                print(f"  SKIP (endpoint not a node): {src} -> {dst}"); continue
            if (src, rel, dst) in have_pairs:
                print(f"  skip (already present): {src} --{rel}--> {dst}"); continue
            out.append((src, rel, dst, note))
        return out

    add = collect(TSV)
    add_fb = collect(FEEDBACK, feedback=True) if FEEDBACK.exists() else []
    if not add and not add_fb:
        print("nothing to add -- kg.yaml already densified.")
        return

    lines = []
    if add:
        lines += ["", "  # --- Erythroid regulatory cascade (TRRUST v2, literature-curated; see",
                  "  #     scripts/densify_erythroid_kg.py + data/erythroid_regulatory_edges.tsv). Adds the",
                  "  #     TF->effector layer so an in-silico knockdown propagates through the graph. ---"]
        lines += [f"  - {{src: {s}, rel: {r}, dst: {d}, w: {W}}}   # {n}" for s, r, d, n in add]
    if add_fb:
        lines += ["", "  # --- Erythroid FEEDBACK circuits (autoregulation + double-negative toggle; see",
                  "  #     data/erythroid_feedback_edges.tsv). Positive feedback = the topological",
                  "  #     requirement for MULTISTABILITY (distinct stable attractors). ---"]
        lines += [f"  - {{src: {s}, rel: {r}, dst: {d}, w: {W}}}   # {n}" for s, r, d, n in add_fb]
    with KG.open("a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"appended {len(add)} regulatory + {len(add_fb)} feedback edges to {KG.name}:")
    for s, r, d, n in add + add_fb:
        print(f"   {s:>7} --{r:<9}--> {d:<8}  ({n[:70]})")


if __name__ == "__main__":
    main()
