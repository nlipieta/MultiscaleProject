"""Ingest GSE194122 (BMMC 10x Multiome) as a PAIRED expression + chromatin-accessibility
per-cell table for the scATAC proof-of-concept.

The dataset is one AnnData with var['feature_types'] in {'GEX','ATAC'}: ~14k genes +
~110k peaks, same cells. We:
  1. map obs['cell_type'] -> Erythropoiesis / Megakaryopoiesis / Quiescent (drop the rest);
  2. EXPRESSION channel: GEX counts for the KG gene nodes;
  3. ACCESSIBILITY channel: ATAC peaks -> per-gene GENE-ACTIVITY (sum peak counts overlapping
     each KG gene's body +/- 2 kb, using data/gene_coords.tsv, GRCh38);
  4. CP10K+log1p then min-max scale each channel across cells;
  5. write one row/cell with node columns (expression) + '<node>__atac' columns (accessibility)
     + label. RNA-only datasets simply lack the __atac columns -> accessibility 0 downstream.

Big (2.7 GB h5ad); intended to run on Colab. Usage:
    uv run python scripts/ingest_gse194122_multiome.py --h5ad <path or url>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from chromatin_toggle.kg import DATA_DIR, load_kg

PROMOTER = 2000  # bp window added around the gene body for gene-activity


def program_of(ct: str) -> str | None:
    """Map BMMC cell types to the erythroid-vs-myeloid FATE BIFURCATION (the GATA1<->PU.1 toggle).
    Both fates come from the SAME experiment (all 13 batches) so their separation is not confounded
    with batch. BMMC lacks mature megakaryocytes (Ficoll depletes them; only MK/E prog present), so
    the demonstrable fork is erythroid vs myeloid, not erythroid vs Mk."""
    c = str(ct).lower()
    if any(k in c for k in ("erythro", "normoblast", "proerythro")):
        return "Erythropoiesis"
    if "megakaryo" in c or c == "mk" or "mkp" in c:              # mature Mk (absent in BMMC, kept for reuse)
        return "Megakaryopoiesis"
    if any(k in c for k in ("mono", "myeloid", "g/m prog", "gmp")):   # monocyte/granulocyte myeloid fate
        return "MacrophageActivation"                            # = myeloid basin (monocyte/GMP proxy)
    if "hsc" in c or c == "mpp" or "multipotent" in c or "hematopoietic stem" in c:
        return "Quiescent"
    return None  # MK/E prog, DCs, and lymphoid (T/B/NK/ILC/Plasma) -> dropped (not on the Ery/myeloid fork)


def load_coords(path: Path) -> dict[str, tuple[str, int, int]]:
    coords = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            coords[r["symbol"].upper()] = (r["chrom"].replace("chr", ""),
                                           int(r["start"]) - PROMOTER, int(r["end"]) + PROMOTER)
    return coords


def parse_peak(name: str):
    """'chr1:1000-2000' or 'chr1-1000-2000' -> ('1', 1000, 2000); None if unparseable."""
    s = name.replace("chr", "")
    for sep in (":", "-"):
        if sep in s:
            left, _, rest = s.partition(sep)
            a, _, b = rest.replace(":", "-").partition("-")
            try:
                return left, int(a), int(b)
            except ValueError:
                return None
    return None


def gene_activity(atac, peak_names, want_genes, coords):
    """atac: [cells, peaks] sparse counts. Returns {gene: np.array[cells]} = summed accessibility
    over peaks overlapping gene body +/- PROMOTER. want_genes: {symbol_upper}."""
    # peaks by chrom -> (idx, start, end)
    by_chrom: dict[str, list] = {}
    for j, nm in enumerate(peak_names):
        p = parse_peak(nm)
        if p:
            by_chrom.setdefault(p[0], []).append((j, p[1], p[2]))
    atac = atac.tocsc()
    out = {}
    for g in want_genes:
        if g not in coords:
            continue
        chrom, gs, ge = coords[g]
        hits = [j for (j, s, e) in by_chrom.get(chrom, []) if s <= ge and e >= gs]  # overlap
        if hits:
            out[g] = np.asarray(atac[:, hits].sum(axis=1)).ravel()
    return out


def _cp10k_log1p_minmax(mat_by_gene, ncells):
    node_cols = {}
    for g, v in mat_by_gene.items():
        v = np.log1p(v)                      # counts already summed; log1p (per-gene, not per-cell CP10K)
        lo, hi = v.min(), v.max()
        node_cols[g] = np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)
    return node_cols


def main():
    ap = argparse.ArgumentParser(description="Ingest GSE194122 BMMC Multiome (expr + ATAC gene-activity)")
    ap.add_argument("--h5ad", required=True, help="path to (or URL of) the multiome .h5ad")
    ap.add_argument("--out", default=str(DATA_DIR / "bmmc_multiome.csv"))
    ap.add_argument("--coords", default=str(DATA_DIR / "gene_coords.tsv"))
    ap.add_argument("--max-cells", type=int, default=12000, help="subsample cells (memory)")
    args = ap.parse_args()

    import anndata as ad
    kg = load_kg()
    coords = load_coords(Path(args.coords))
    node_by_sym = {s.upper(): n for n, s in kg.gene_map.items()}   # symbol -> KG node
    want = set(node_by_sym)

    # BACKED mode: open with X left ON DISK; obs/var (small) come into memory so we can decide
    # which cells to keep, then materialize ONLY those rows. Avoids loading the full ~70k x 124k
    # matrix just to discard most of it (the expensive step + the usual Colab OOM).
    print(f"[multiome] opening {args.h5ad} BACKED (obs only; X stays on disk) ...")
    a = ad.read_h5ad(args.h5ad, backed="r")
    prog = a.obs["cell_type"].astype(str).map(program_of)         # obs only -> no X read
    idx = np.where(prog.notna().to_numpy())[0]                    # keep Erythro/Mega/HSC cells
    if args.max_cells and idx.size > args.max_cells:
        idx = np.random.default_rng(0).choice(idx, args.max_cells, replace=False)
    idx.sort()
    print(f"[multiome] {idx.size} of {a.n_obs} cells kept; reading only those rows off disk ...")
    a = a[idx].to_memory()                                        # materialize ONLY the kept rows
    prog = prog.iloc[idx]
    ft = a.var["feature_types"].astype(str)
    print(f"[multiome] programs: {prog.value_counts().to_dict()}")

    gex = a[:, (ft == "GEX").to_numpy()]
    atac = a[:, (ft == "ATAC").to_numpy()]
    print(f"[multiome] GEX genes={gex.n_vars}  ATAC peaks={atac.n_vars}")

    # expression channel: GEX symbol -> KG node
    gsym = {str(s).upper(): j for j, s in enumerate(gex.var.get("feature_name", gex.var_names))}
    Xg = gex.X.tocsc() if sp.issparse(gex.X) else sp.csc_matrix(gex.X)
    expr = {}
    for sym in want:
        if sym in gsym:
            expr[sym] = np.asarray(Xg[:, gsym[sym]].todense()).ravel()
    # accessibility channel: ATAC peaks -> gene activity
    Xa = atac.X.tocsr() if sp.issparse(atac.X) else sp.csr_matrix(atac.X)
    acc = gene_activity(Xa, list(atac.var_names), want, coords)
    print(f"[multiome] mapped {len(expr)} expression genes, {len(acc)} accessibility genes -> KG nodes")

    expr_n = _cp10k_log1p_minmax(expr, a.n_obs)
    acc_n = _cp10k_log1p_minmax(acc, a.n_obs)

    node_cols = list(kg.node_ids)
    header = node_cols + [f"{n}__atac" for n in node_cols] + ["label", "dataset", "pathway", "batch"]
    labels = prog.to_numpy()
    batches = a.obs["batch"].astype(str).to_numpy() if "batch" in a.obs else np.array(["na"] * a.n_obs)
    lines = [",".join(header)]
    for i in range(a.n_obs):
        row = {c: 0.0 for c in node_cols}
        arow = {f"{c}__atac": 0.0 for c in node_cols}
        for sym, v in expr_n.items():
            row[node_by_sym[sym]] = v[i]
        for sym, v in acc_n.items():
            arow[f"{node_by_sym[sym]}__atac"] = v[i]
        vals = [f"{row[c]}" for c in node_cols] + [f"{arow[f'{c}__atac']}" for c in node_cols]
        lines.append(",".join(vals + [labels[i], "bmmc_multiome", "hematopoiesis_multiome", batches[i]]))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[multiome] wrote {a.n_obs} cells x ({len(expr_n)} expr + {len(acc_n)} atac) -> {args.out}")


if __name__ == "__main__":
    main()
