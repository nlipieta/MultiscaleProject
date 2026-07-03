"""Ingest a scPerturb single-cell perturbation dataset into the model schema.

scPerturb (Peidli et al., Nat Methods 2024) is a harmonized pool of 40+
single-cell perturbation-response datasets, hosted on Zenodo (RNA record
7041849, ATAC 7058382) as AnnData ``.h5ad`` files. This module:

  1. lists / downloads a chosen ``.h5ad`` from the Zenodo record,
  2. groups cells by their perturbation label (``obs`` column),
  3. averages expression of the genes named in ``gene_map`` (kg.yaml) onto the
     KG nodes, scaling each node to [0, 1] across perturbation groups,
  4. writes one row per perturbation group in the model's node-column schema.

HONEST SCOPE. scPerturb's perturbations are genetic (CRISPR KO/KD) and chemical
(drugs) in cell lines -- they do NOT correspond to this model's extrinsic cues
(LPS, TGF-beta, mechanical, bioelectric, caerulein) or its 7 response programs.
So there is no automatic ``label`` mapping: the writer emits a ``perturbation``
column and leaves ``label`` for you to fill (via ``--map`` or by hand). Use this
when repurposing the architecture for general perturbation-response prediction
(cf. CellCap / STATE), not for the 7 curated toggle programs.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from .kg import DATA_DIR, load_kg

RNA_RECORD = "7041849"
_API = "https://zenodo.org/api/records"

# Common obs columns that hold the perturbation identity, best-first.
PERT_COLS = ["perturbation", "perturbation_name", "guide_id", "target_gene",
             "gene", "target", "condition", "treatment"]


def list_files(record: str = RNA_RECORD) -> list[tuple[str, float]]:
    """Return [(filename, size_MB), ...] for a scPerturb Zenodo record."""
    with urllib.request.urlopen(f"{_API}/{record}") as r:
        meta = json.load(r)
    files = [(f["key"], f["size"] / 1e6) for f in meta.get("files", [])]
    return sorted(files, key=lambda t: t[1])


def download(filename: str, dest: Path, record: str = RNA_RECORD) -> Path:
    if not dest.exists():
        url = f"{_API}/{record}/files/{filename}/content"
        print(f"  fetching {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def ingest(h5ad: Path, out: Path, pert_col: str | None = None,
           label_map: dict[str, str] | None = None) -> None:
    import anndata as ad

    kg = load_kg()
    adata = ad.read_h5ad(h5ad)
    print(f"  loaded {h5ad.name}: {adata.n_obs} cells x {adata.n_vars} genes")

    # pick the perturbation column
    col = pert_col or next((c for c in PERT_COLS if c in adata.obs.columns), None)
    if col is None:
        raise SystemExit(f"No perturbation column found; obs has: "
                         f"{list(adata.obs.columns)}. Pass --pert-col.")
    print(f"  grouping by obs['{col}'] ({adata.obs[col].nunique()} groups)")

    # gene symbol -> var index (var_names are usually gene symbols)
    var_names = [str(v).upper() for v in adata.var_names]
    sym_to_var = {}
    for i, s in enumerate(var_names):
        sym_to_var.setdefault(s, i)

    node_cols = list(kg.node_ids)
    node_gene_var = {}  # node -> var index
    for node, sym in kg.gene_map.items():
        vi = sym_to_var.get(sym.upper())
        if vi is not None and node in kg.node_index:
            node_gene_var[node] = vi
    print(f"  mapped {len(node_gene_var)}/{len(kg.gene_map)} gene nodes: "
          f"{', '.join(node_gene_var)}")

    groups = list(adata.obs[col].astype(str).unique())
    X = np.asarray(adata.X.todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X)
    obs_vals = adata.obs[col].astype(str).to_numpy()

    rows = []
    node_matrix = np.zeros((len(groups), len(node_cols)))
    for gi, g in enumerate(groups):
        mask = obs_vals == g
        mean_expr = X[mask].mean(axis=0)
        for node, vi in node_gene_var.items():
            node_matrix[gi, node_cols.index(node)] = float(mean_expr[vi])

    # scale each mapped node column to [0,1] across groups
    for node in node_gene_var:
        j = node_cols.index(node)
        lo, hi = node_matrix[:, j].min(), node_matrix[:, j].max()
        node_matrix[:, j] = 0.0 if hi <= lo else (node_matrix[:, j] - lo) / (hi - lo)

    node_matrix = np.nan_to_num(node_matrix, nan=0.0)
    label_map = label_map or {}
    header = node_cols + ["perturbation", "label"]
    lines = [",".join(header)]
    for gi, g in enumerate(groups):
        vals = [f"{node_matrix[gi, j]}" for j in range(len(node_cols))]
        vals.append(str(g))
        vals.append(str(label_map.get(str(g), "")))  # blank unless --map supplied
        lines.append(",".join(vals))
    out.write_text("\n".join(lines) + "\n")

    n_labelled = sum(1 for g in groups if g in label_map)
    print(f"\nWrote {len(groups)} perturbation-group rows -> {out}")
    print(f"  labelled: {n_labelled}/{len(groups)} "
          f"({'supply --map to set model programs' if n_labelled == 0 else 'from --map'})")
    print("  NOTE: scPerturb perturbations are genetic/chemical and do NOT map"
          "\n  to the model's cues/programs; set the 'label' column yourself"
          "\n  before training, or reframe the task to perturbation-response.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Ingest a scPerturb .h5ad to the model CSV schema")
    ap.add_argument("--list", action="store_true", help="list files in the Zenodo record and exit")
    ap.add_argument("--file", help="scPerturb .h5ad filename (from --list)")
    ap.add_argument("--pert-col", default=None, help="obs column holding the perturbation")
    ap.add_argument("--map", default=None, help="JSON file: {perturbation: program_label}")
    ap.add_argument("--out", default=None, help="output CSV")
    ap.add_argument("--cache", default=None, help="download cache dir")
    args = ap.parse_args()

    if args.list:
        print(f"scPerturb RNA record {RNA_RECORD}:")
        for name, mb in list_files():
            print(f"  {mb:8.1f} MB  {name}")
        return

    if not args.file:
        raise SystemExit("pass --file <name> (see --list) or --list")
    cache = Path(args.cache) if args.cache else DATA_DIR / "scperturb_cache"
    cache.mkdir(parents=True, exist_ok=True)
    h5ad = download(args.file, cache / args.file)
    label_map = json.loads(Path(args.map).read_text()) if args.map else None
    out = Path(args.out) if args.out else DATA_DIR / f"{Path(args.file).stem}.csv"
    ingest(h5ad, out, pert_col=args.pert_col, label_map=label_map)


if __name__ == "__main__":
    main()
