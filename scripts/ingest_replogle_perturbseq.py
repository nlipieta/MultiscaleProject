"""Ingest Replogle 2022 K562 Perturb-seq -> the Q2 perturbation-VALIDATION set for Erythropoiesis.

Real CRISPRi knockdowns + non-targeting controls. Keeps EVERY perturbation whose target gene is a KG
node (with enough cells), so the shift model can be validated against as many real knockdowns as the
data provides -- a proper sample size for the potency-RANK test, not just a handful. Used to VALIDATE
the model's in-silico perturbations: does knocking down gene X in-silico shift the model's state the
same way the REAL X-KD cells shifted? (turns Q2 from a hypothesis generator into 'predicts outcomes').

Source (verified public, no paywall): pertpy mirror is the smallest single-cell handle --
  https://exampledata.scverse.org/pertpy/replogle_2022_k562_essential.h5ad   (~1.55 GB)
  (or Figshare+ K562_essential_normalized_singlecell_01.h5ad, 10.7 GB).
obs['gene'] = KD target symbol; non-targeting controls labelled 'non-targeting' (or 'control' on the
pertpy copy -- auto-detected). var index = Ensembl IDs; var['gene_name'] = symbols. GRCh38.

Usage (Colab):  uv run python scripts/ingest_replogle_perturbseq.py --h5ad <path to .h5ad>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chromatin_toggle.kg import DATA_DIR, load_kg


def main():
    ap = argparse.ArgumentParser(description="Replogle K562 Perturb-seq -> Q2 erythroid validation CSV")
    ap.add_argument("--h5ad", required=True, help="path to K562_essential*singlecell*.h5ad")
    ap.add_argument("--out", default=str(DATA_DIR / "replogle_k562.csv"))
    ap.add_argument("--max-control", type=int, default=4000, help="subsample non-targeting controls")
    ap.add_argument("--max-per-target", type=int, default=1000)
    ap.add_argument("--min-per-target", type=int, default=50,
                    help="keep any KG-node perturbation with at least this many cells")
    args = ap.parse_args()
    import anndata as ad

    kg = load_kg()
    node_by_sym = {s.upper(): n for n, s in kg.gene_map.items()}
    want = set(node_by_sym)

    print(f"[replogle] opening {args.h5ad} BACKED ...")
    a = ad.read_h5ad(args.h5ad, backed="r")
    if "gene" not in a.obs:
        raise SystemExit(f"no obs['gene']; obs cols = {list(a.obs.columns)[:20]}")
    g = a.obs["gene"].astype(str)
    vc = g.value_counts()
    ntc = next((c for c in ("non-targeting", "control", "non_targeting", "NTC") if c in set(g)), None)
    if ntc is None:
        raise SystemExit(f"no non-targeting label found; top obs.gene values: {vc.head(10).to_dict()}")
    # keep EVERY perturbation whose target is a KG node and has >= min_per_target cells
    present = sorted([t for t in set(g) if t != ntc and t.upper() in want
                      and int(vc.get(t, 0)) >= args.min_per_target],
                     key=lambda t: -int(vc[t]))
    print(f"[replogle] control label='{ntc}' (n={int(vc.get(ntc,0))}); {len(present)} KG-node targets "
          f">= {args.min_per_target} cells: {present}")
    if not present:
        raise SystemExit(f"no KG-node perturbations with >= {args.min_per_target} cells; "
                         f"sample obs.gene: {list(vc.head(20).index)}")

    rng = np.random.default_rng(0)
    keep_idx, pert = [], []
    for grp, cap in [(ntc, args.max_control)] + [(t, args.max_per_target) for t in present]:
        rows = np.where(g.to_numpy() == grp)[0]
        if len(rows) > cap:
            rows = rng.choice(rows, cap, replace=False)
        keep_idx.extend(rows.tolist())
        pert.extend([("control" if grp == ntc else grp)] * len(rows))
    order = np.argsort(keep_idx); keep_idx = list(np.array(keep_idx)[order]); pert = list(np.array(pert)[order])
    print(f"[replogle] keeping {len(keep_idx)} cells; materializing subset ...")
    sub = a[keep_idx].to_memory()

    # var: map symbols (var['gene_name']) -> KG nodes
    if "gene_name" in sub.var:
        vsym = sub.var["gene_name"].astype(str).str.upper().to_numpy()
    else:
        vsym = np.array([str(x).upper() for x in sub.var_names])
    col_of = {}
    for j, s in enumerate(vsym):
        if s in want and s not in col_of:
            col_of[s] = j
    import scipy.sparse as sp
    X = sub.X.tocsc() if sp.issparse(sub.X) else sp.csc_matrix(sub.X)
    expr = {}
    for sym, j in col_of.items():
        v = np.asarray(X[:, j].todense()).ravel().astype(float)
        v = np.log1p(v)
        lo, hi = v.min(), v.max()
        expr[sym] = np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)
    print(f"[replogle] mapped {len(expr)}/{len(want)} KG genes; per-perturbation counts:")
    for p in ["control"] + present:
        print(f"    {p}: {pert.count(p)}")

    node_cols = list(kg.node_ids)
    header = node_cols + ["label", "dataset", "pathway", "perturbation"]
    lines = [",".join(header)]
    for k in range(len(keep_idx)):
        row = {c: 0.0 for c in node_cols}
        for sym, v in expr.items():
            row[node_by_sym[sym]] = v[k]
        vals = [f"{row[c]}" for c in node_cols]
        lines.append(",".join(vals + ["Erythropoiesis", "replogle_k562", "perturb_validation", pert[k]]))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[replogle] wrote {len(keep_idx)} cells x {len(expr)} genes -> {args.out}")


if __name__ == "__main__":
    main()
