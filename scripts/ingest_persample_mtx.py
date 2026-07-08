"""Ingest GEO datasets delivered as PER-SAMPLE 10x MTX triplets in a RAW.tar,
where the sample/GSM IS the label (no per-cell metadata). Generic version of the
adipogenesis loader; add a dataset to CONFIG and run:

    uv run python scripts/ingest_persample_mtx.py <key>

Sample-level labels -> use grouped-split (whole dataset held out) downstream.
"""
from __future__ import annotations

import gzip
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.io

from chromatin_toggle.geo import ingest_h5ad_percell
from chromatin_toggle.kg import DATA_DIR

CACHE = DATA_DIR / "geo_cache"

CONFIG = {
    # GSE161125: mouse BMDM. M0 resting -> Quiescent; M1 (LPS+IFNg classical
    # activation) -> MacrophageActivation. M2 (IL-4) and the co-stim arm dropped
    # to keep a single-cue program.
    "macrophage_gse161125": dict(
        tar="GSE161125_RAW.tar",
        samples=[("GSM4889920_M0", "Quiescent"),
                 ("GSM4889921_M1", "MacrophageActivation")],
        out="macrophage_activation.csv",
    ),
    # GSE158866: mouse liver, partial hepatectomy. PHx 0hr -> Quiescent; PHx 48hr
    # regenerating hepatocytes -> Regeneration (THIRD organ source for the existing
    # Regeneration program, alongside lung and muscle).
    "liver_gse158866": dict(
        tar="GSE158866_RAW.tar",
        samples=[("GSM4812353_PHX_0_10X", "Quiescent"),
                 ("GSM4812354_PHX_48_10X", "Regeneration")],
        out="liver_regen.csv",
    ),
}


def _read_sample(tar: tarfile.TarFile, prefix: str) -> ad.AnnData:
    def member(suffix):
        f = tar.extractfile(f"{prefix}_{suffix}")
        if f is None:
            raise SystemExit(f"member {prefix}_{suffix} not found")
        return f
    M = scipy.io.mmread(gzip.open(member("matrix.mtx.gz"), "rb")).tocsr()   # genes x cells
    with gzip.open(member("genes.tsv.gz"), "rt") as fh:
        syms = []
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            s = f[1] if len(f) > 1 else f[0]                                # symbol col, not type
            for pre in ("mm10_", "hg19_", "GRCh38_", "GRCh38-", "mm10-"):   # mixed-ref prefix
                if s.startswith(pre):
                    s = s[len(pre):]
                    break
            syms.append(s)
    with gzip.open(member("barcodes.tsv.gz"), "rt") as fh:
        bcs = [ln.strip() for ln in fh]
    a = ad.AnnData(X=M.T.tocsr(),
                   obs=pd.DataFrame(index=[f"{prefix}_{b}" for b in bcs]),
                   var=pd.DataFrame(index=syms))
    a.var_names_make_unique()
    return a


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIG:
        raise SystemExit(f"usage: ingest_persample_mtx.py <{'|'.join(CONFIG)}>")
    cfg = CONFIG[sys.argv[1]]
    tar_path = CACHE / cfg["tar"]
    if not tar_path.exists():
        raise SystemExit(f"{tar_path} not found -- download the RAW.tar into geo_cache first")
    out = DATA_DIR / cfg["out"]
    labels = {lab for _, lab in cfg["samples"]}
    program_map = {lab: lab for lab in labels}

    tmp = Path(tempfile.mkdtemp(prefix="psmtx_", dir=CACHE))
    try:
        adatas = []
        with tarfile.open(tar_path) as tar:
            for prefix, label in cfg["samples"]:
                a = _read_sample(tar, prefix)
                a.obs["_label"] = label
                print(f"  {prefix}: {a.n_obs} cells x {a.n_vars} genes -> {label}")
                adatas.append(a)
        adata = ad.concat(adatas, join="outer", index_unique=None)
        adata.var_names_make_unique()
        adata.obs.index = pd.Index([str(x) for x in adata.obs.index], dtype=object)
        adata.var.index = pd.Index([str(x) for x in adata.var.index], dtype=object)
        adata.obs["_label"] = adata.obs["_label"].astype(str).astype(object)
        tmp_h5ad = tmp / "combined.h5ad"
        adata.write_h5ad(tmp_h5ad)
        ingest_h5ad_percell(tmp_h5ad, out, cell_type_col="_label",
                            program_map=program_map, cue=None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
