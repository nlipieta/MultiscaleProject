"""Ingest GSE226365 (3T3-L1 adipogenesis, mouse) into the Chromatin Toggle.

The processed data ships as per-timepoint 10x MTX triplets inside GSE226365_RAW.tar
(no per-cell metadata) -- the FILE is the label. This resolves the earlier deferral
(it does NOT need the 1 GB Seurat .Rds). We assemble the two mouse (mm10) samples into
one AnnData tagged by timepoint, then reuse geo.ingest_h5ad_percell:

    mm10 D0 (GSM7073976) -> Quiescent (preadipocyte)
    mm10 D5 (GSM7073977) -> Adipogenesis (adipocyte)

Usage:  uv run python scripts/ingest_gse226365_adipogenesis.py
"""
from __future__ import annotations

import gzip
import shutil
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.io

from chromatin_toggle.geo import ingest_h5ad_percell
from chromatin_toggle.kg import DATA_DIR

TAR = DATA_DIR / "geo_cache" / "GSE226365_RAW.tar"
OUT = DATA_DIR / "adipogenesis.csv"
# (member-prefix, program label). Mouse arm only (cleaner canonical 3T3-L1 line).
SAMPLES = [
    ("GSM7073976_mm10_D0", "Quiescent"),
    ("GSM7073977_mm10_D5", "Adipogenesis"),
]
TRIPLET = {"matrix.mtx.gz": "matrix.mtx.gz",
           "barcodes.tsv.gz": "barcodes.tsv.gz",
           "genes.tsv.gz": "genes.tsv.gz"}


def _read_sample(tar: tarfile.TarFile, prefix: str) -> ad.AnnData:
    """Read a sample's 10x MTX triplet from the tar into an AnnData (cells x genes,
    var index = gene symbols). Manual scipy read -> no scanpy v2/v3 filename quirks."""
    def member(suffix):
        f = tar.extractfile(f"{prefix}_{suffix}")
        if f is None:
            raise SystemExit(f"member {prefix}_{suffix} not in {TAR.name}")
        return f
    M = scipy.io.mmread(gzip.open(member("matrix.mtx.gz"), "rb")).tocsr()   # genes x cells
    with gzip.open(member("genes.tsv.gz"), "rt") as fh:
        # v2 genes.tsv = [ensembl, symbol]; v3 features.tsv = [ensembl, symbol, type].
        # symbol is field [1] in both; DON'T use [-1] (that's the feature type in v3).
        syms = []
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            s = f[1] if len(f) > 1 else f[0]
            for pre in ("mm10_", "hg19_", "GRCh38_", "GRCh38-", "mm10-"):     # mixed-ref prefix
                if s.startswith(pre):
                    s = s[len(pre):]
                    break
            syms.append(s)
    with gzip.open(member("barcodes.tsv.gz"), "rt") as fh:
        bcs = [ln.strip() for ln in fh]
    a = ad.AnnData(X=M.T.tocsr(),                                           # -> cells x genes
                   obs=pd.DataFrame(index=[f"{prefix}_{b}" for b in bcs]),
                   var=pd.DataFrame(index=syms))
    a.var_names_make_unique()
    return a


def main() -> None:
    if not TAR.exists():
        raise SystemExit(f"{TAR} not found -- download GSE226365_RAW.tar into geo_cache first")
    tmp = Path(tempfile.mkdtemp(prefix="adipo_", dir=DATA_DIR / "geo_cache"))
    try:
        adatas = []
        with tarfile.open(TAR) as tar:
            for prefix, label in SAMPLES:
                a = _read_sample(tar, prefix)
                a.obs["_label"] = label
                print(f"  {prefix}: {a.n_obs} cells x {a.n_vars} genes -> {label}")
                adatas.append(a)
        adata = ad.concat(adatas, join="outer", index_unique=None)
        adata.var_names_make_unique()
        # coerce Arrow-backed string indices/columns to plain object str (h5ad-writable)
        adata.obs.index = pd.Index([str(x) for x in adata.obs.index], dtype=object)
        adata.var.index = pd.Index([str(x) for x in adata.var.index], dtype=object)
        adata.obs["_label"] = adata.obs["_label"].astype(str).astype(object)
        tmp_h5ad = tmp / "adipo_combined.h5ad"
        adata.write_h5ad(tmp_h5ad)
        ingest_h5ad_percell(
            tmp_h5ad, OUT, cell_type_col="_label",
            program_map={"Quiescent": "Quiescent", "Adipogenesis": "Adipogenesis"},
            cue=None,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
