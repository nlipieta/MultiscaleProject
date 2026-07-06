"""Ingest T-cell exhaustion (GSE156728, Zheng 2021 pan-cancer Tex atlas) -> per-cell CSV.
New program for the multiscale KG. Per-cancer CD8 count matrices (genes x cells TSV) joined
to GSE156728_metadata.txt.gz by barcode (for 10X CD8 files, metadata cellID == barcode,
matched within cancerType). Labels from meta.cluster, restricted to tumor (loc=='T'):
  *.Tex.*                       -> Exhaustion   (c11.PDCD1 / c12.CXCL13 / c13 / c14.TCF7)
  *.Tn.* / .Tm. / .Tem. / .Temra. -> Quiescent
Genes named in kg.gene_map are CP10K+log1p normalized then min-max scaled across cells.
"""
from __future__ import annotations
import gzip, sys, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from chromatin_toggle.kg import load_kg, DATA_DIR

CACHE = DATA_DIR / "geo_cache"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156728/suppl"
CANCERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BC", "PACA", "MM", "ESCA"]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_DIR / "gse156728_exhaustion.csv"

QUI = (".Tn.", ".Tm.", ".Tem.", ".Temra.")


def _label(cluster: str):
    if ".Tex." in cluster:
        return "Exhaustion"
    if any(t in cluster for t in QUI):
        return "Quiescent"
    return None


def _dl(name):
    p = CACHE / name
    if not p.exists():
        print(f"  downloading {name} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{name}", p)
    return p


def main():
    kg = load_kg()
    want = {sym.upper(): node for node, sym in kg.gene_map.items() if node in kg.node_index}
    node_cols = list(kg.node_ids)

    meta = pd.read_csv(_dl("GSE156728_metadata.txt.gz"), sep="\t")
    meta = meta[(meta["loc"] == "T") & meta["meta.cluster"].str.contains("CD8", na=False)]
    meta["__prog"] = meta["meta.cluster"].map(_label)
    meta = meta[meta["__prog"].notna()]
    cell_prog = dict(zip(meta["cellID"], meta["__prog"]))     # barcode -> program

    header = node_cols + ["label", "dataset", "pathway", "assay"]
    rows = [",".join(header)]
    total_counts = {}
    for c in CANCERS:
        counts = _dl(f"GSE156728_{c}_10X.CD8.counts.txt.gz")
        # stream genes x cells: capture KG-gene rows + per-cell totals
        node_vals, cells, totals = {}, None, None
        for chunk in pd.read_csv(counts, sep="\t", index_col=0, chunksize=4000):
            if cells is None:
                cells = list(chunk.columns); totals = np.zeros(len(cells))
            totals += chunk.to_numpy(float).sum(0)
            up = {str(g).upper(): g for g in chunk.index}
            for sym, node in want.items():
                if sym in up:
                    node_vals[node] = chunk.loc[up[sym]].to_numpy(float)
        safe = np.where(totals > 0, totals, 1.0)
        X = np.zeros((len(cells), len(node_cols)))
        for node, cnt in node_vals.items():
            norm = np.log1p(cnt / safe * 1e4)
            lo, hi = norm.min(), norm.max()
            X[:, node_cols.index(node)] = 0.0 if hi <= lo else (norm - lo) / (hi - lo)
        kept = 0
        for i, bc in enumerate(cells):
            prog = cell_prog.get(bc)
            if prog is None:
                continue
            vals = [f"{X[i, j]}" for j in range(len(node_cols))] + [prog, "gse156728_exhaustion", "exhaustion_tumor", "scRNA"]
            rows.append(",".join(vals)); kept += 1
            total_counts[prog] = total_counts.get(prog, 0) + 1
        print(f"  {c}: {kept} labelled CD8 tumor cells", flush=True)
    OUT.write_text("\n".join(rows) + "\n")
    print(f"\nwrote {OUT}: {sum(total_counts.values())} cells {total_counts}")
    print(f"  mapped {len(want)} gene nodes; cancers={CANCERS}")


if __name__ == "__main__":
    main()
