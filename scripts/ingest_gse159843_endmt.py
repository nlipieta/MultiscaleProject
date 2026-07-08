"""Ingest GSE159843 (HUVEC endothelial-to-mesenchymal transition) into the Toggle.

Per-sample Cell Ranger .h5 files inside GSE159843_RAW.tar; the sample IS the label
(no per-cell metadata). IL1b+TGFb2-treated samples -> EndMT, untreated/control -> Quiescent.
NOTE: sample/condition-level label -> use grouped-split (whole dataset held out); do not
treat sample-of-origin as an independent per-cell feature.

    Day0 / Cont_Day3 / Cont_Day7  -> Quiescent (endothelial baseline)
    EndMT_Day3 / EndMT_Day7       -> EndMT

Usage:  uv run python scripts/ingest_gse159843_endmt.py
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

from chromatin_toggle.geo import ingest_h5ad_percell
from chromatin_toggle.kg import DATA_DIR

TAR = DATA_DIR / "geo_cache" / "GSE159843_RAW.tar"
OUT = DATA_DIR / "endmt.csv"
SAMPLES = [
    ("GSM4848247_Day0_filtered_gene_bc_matrices_h5.h5", "Quiescent"),
    ("GSM4848248_Cont_Day3_filtered_gene_bc_matrices_h5.h5", "Quiescent"),
    ("GSM4848249_EndMT_Day3_filtered_gene_bc_matrices_h5.h5", "EndMT"),
    ("GSM4848250_Cont_Day7_filtered_gene_bc_matrices_h5.h5", "Quiescent"),
    ("GSM4848251_EndMT_Day7_filtered_gene_bc_matrices_h5.h5", "EndMT"),
]


def main() -> None:
    if not TAR.exists():
        raise SystemExit(f"{TAR} not found -- download GSE159843_RAW.tar into geo_cache first")
    tmp = Path(tempfile.mkdtemp(prefix="endmt_", dir=DATA_DIR / "geo_cache"))
    try:
        adatas = []
        with tarfile.open(TAR) as tar:
            for member, label in SAMPLES:
                src = tar.extractfile(member)
                if src is None:
                    raise SystemExit(f"member {member} not in {TAR.name}")
                h5 = tmp / member
                with open(h5, "wb") as fh:
                    shutil.copyfileobj(src, fh)
                a = sc.read_10x_h5(h5)                     # var_names = gene symbols
                a.var_names_make_unique()
                tag = member.split("_filtered")[0]
                a.obs["_label"] = label
                a.obs_names = [f"{tag}_{b}" for b in a.obs_names]
                print(f"  {tag}: {a.n_obs} cells x {a.n_vars} genes -> {label}")
                adatas.append(a)
        adata = ad.concat(adatas, join="outer", index_unique=None)
        adata.var_names_make_unique()
        adata.obs.index = pd.Index([str(x) for x in adata.obs.index], dtype=object)
        adata.var.index = pd.Index([str(x) for x in adata.var.index], dtype=object)
        adata.obs["_label"] = adata.obs["_label"].astype(str).astype(object)
        tmp_h5ad = tmp / "endmt_combined.h5ad"
        adata.write_h5ad(tmp_h5ad)
        ingest_h5ad_percell(
            tmp_h5ad, OUT, cell_type_col="_label",
            program_map={"EndMT": "EndMT", "Quiescent": "Quiescent"},
            cue=None,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
