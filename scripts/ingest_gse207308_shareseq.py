"""Ingest GSE207308 (SHARE-seqV2 BMMC) as a paired expression + ATAC-accessibility table
with an ERYTHROID COMMITMENT TRAJECTORY, for the ATAC-plasticity test.

SHARE-seq = same-cell RNA + ATAC (passes the same-experiment rule). BMMC files:
  RNA : GSM6284350_BMMC.RNA.hg19.gene.bc.matrices.h5  (custom CSC: genes x cells; gene_names, barcodes)
  ATAC: GSM6284346_BMMC.count.matrix.txt.gz (MatrixMarket peaks x cells, 173026 x 78708)
        + GSM6284346_BMMC.peaks.txt.gz  (header 'chr starts end', 1-indexed -> peak coords)
  meta: GSM6284346_BMMC.metadata.txt.gz (atac.bc, rna.bc, ..., celltype, umap1, umap2);
        row i == ATAC matrix column i; rna.bc pairs to the RNA h5 barcode.

Labels (deposited author celltype, tier-2):
  early-Ery, late-Ery -> Erythropoiesis   (stage 1, 2)
  HSC/MPP, LMPP       -> Quiescent         (stage 0)
  everything else dropped (Mono/B/T/NK/DC/MEP/... = other lineages; NO committed megakaryocyte label
  exists in this dataset -> Megakaryopoiesis is NOT recoverable here).
'stage' column = erythroid maturation ordinal (0 HSC -> 1 early-Ery -> 2 late-Ery) for the
trajectory test (written to the 'timepoint' column so chromatin-temporal can read it).

Usage (Colab, tar already downloaded):  uv run python scripts/ingest_gse207308_shareseq.py --tar <path>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import scipy.sparse as sp

from chromatin_toggle.kg import DATA_DIR, load_kg

PROMOTER = 2000
CELLTYPE_MAP = {"early-Ery": "Erythropoiesis", "late-Ery": "Erythropoiesis",
                "HSC/MPP": "Quiescent", "LMPP": "Quiescent"}
STAGE = {"HSC/MPP": 0.0, "LMPP": 0.0, "early-Ery": 1.0, "late-Ery": 2.0}
RNA = "GSM6284350_BMMC.RNA.hg19.gene.bc.matrices.h5"
ATAC_MTX = "GSM6284346_BMMC.count.matrix.txt.gz"
PEAKS = "GSM6284346_BMMC.peaks.txt.gz"
META = "GSM6284346_BMMC.metadata.txt.gz"


def load_coords(path):
    coords = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            coords[r["symbol"].upper()] = (r["chrom"].replace("chr", ""),
                                            int(r["start"]) - PROMOTER, int(r["end"]) + PROMOTER)
    return coords


def gene_activity(atac_csc, peak_coords, want_genes, coords):
    """atac_csc: [peaks, cells] sparse. peak_coords: list of (chrom,start,end) per peak row.
    Returns {gene: np.array[cells]} summed over peaks overlapping gene body +/- PROMOTER."""
    by_chrom = {}
    for j, p in enumerate(peak_coords):
        if p:
            by_chrom.setdefault(p[0], []).append((j, p[1], p[2]))
    atac = atac_csc.tocsr()
    out = {}
    for g in want_genes:
        if g not in coords:
            continue
        chrom, gs, ge = coords[g]
        rows = [j for (j, s, e) in by_chrom.get(chrom, []) if s <= ge and e >= gs]
        if rows:
            out[g] = np.asarray(atac[rows, :].sum(axis=0)).ravel()   # [cells]
    return out


def _minmax_log(mat_by_gene):
    out = {}
    for g, v in mat_by_gene.items():
        v = np.log1p(v)
        lo, hi = v.min(), v.max()
        out[g] = np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)
    return out


def _read_meta(t):
    raw = gzip.decompress(t.extractfile(t.getmember(META)).read()).decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(raw), delimiter="\t"))
    return rows


def _read_peaks(t):
    raw = gzip.decompress(t.extractfile(t.getmember(PEAKS)).read()).decode("utf-8", "replace")
    coords = []
    for i, line in enumerate(raw.splitlines()):
        if i == 0 and "start" in line.lower():
            continue                                    # header 'chr starts end'
        parts = line.split()
        # rows look like: '<idx> chr1 10335 10634'  OR  'chr1 10335 10634'
        cse = parts[-3:]
        try:
            coords.append((cse[0].replace("chr", ""), int(cse[1]), int(cse[2])))
        except (ValueError, IndexError):
            coords.append(None)
    return coords


def _read_rna_h5(path):
    with h5py.File(path, "r") as h:
        grp = h if "data" in h else h[list(h.keys())[0]]
        data, indices, indptr = grp["data"][:], grp["indices"][:], grp["indptr"][:]
        shape = tuple(int(x) for x in grp["shape"][:])
        genes = [g.decode() if isinstance(g, bytes) else str(g) for g in grp["gene_names"][:]]
        bcs = [b.decode() if isinstance(b, bytes) else str(b) for b in grp["barcodes"][:]]
    M = sp.csc_matrix((data, indices, indptr), shape=shape)          # genes x cells
    return M, genes, bcs


def main():
    ap = argparse.ArgumentParser(description="Ingest GSE207308 SHARE-seq BMMC (erythroid trajectory)")
    ap.add_argument("--tar", default=str(DATA_DIR / "geo_cache" / "GSE207308_RAW.tar"))
    ap.add_argument("--out", default=str(DATA_DIR / "bmmc_shareseq.csv"))
    ap.add_argument("--coords", default=str(DATA_DIR / "gene_coords.tsv"))
    args = ap.parse_args()

    kg = load_kg()
    node_by_sym = {s.upper(): n for n, s in kg.gene_map.items()}
    want = set(node_by_sym)
    coords = load_coords(Path(args.coords))
    t = tarfile.open(args.tar)

    meta = _read_meta(t)
    keep = [(i, r) for i, r in enumerate(meta) if r["celltype"] in CELLTYPE_MAP]   # (atac col idx, row)
    prog = [CELLTYPE_MAP[r["celltype"]] for _, r in keep]
    stage = [STAGE[r["celltype"]] for _, r in keep]
    atac_cols = [i for i, _ in keep]
    rna_bcs_kept = [r["rna.bc"] for _, r in keep]
    from collections import Counter
    print(f"[shareseq] {len(meta)} cells; kept {len(keep)} labelled -> {Counter(prog)}")

    # --- ATAC: MatrixMarket peaks x cells -> keep our columns, gene-activity ---
    print("[shareseq] reading ATAC MatrixMarket ...")
    A = scipy.io.mmread(gzip.open(t.extractfile(t.getmember(ATAC_MTX)))).tocsc()   # [peaks, cells]
    A = A[:, atac_cols]
    peak_coords = _read_peaks(t)
    print(f"[shareseq] ATAC {A.shape[0]} peaks x {A.shape[1]} kept cells; peak coords {len(peak_coords)}")
    acc = gene_activity(A, peak_coords, want, coords)

    # --- RNA: match rna.bc -> h5 column ---
    print("[shareseq] reading RNA h5 ...")
    tmp = Path(args.out).parent / "_rna_207308.h5"
    with open(tmp, "wb") as fh:
        fh.write(t.extractfile(t.getmember(RNA)).read())
    M, genes, bcs = _read_rna_h5(tmp)
    tmp.unlink(missing_ok=True)
    bc_idx = {b: j for j, b in enumerate(bcs)}
    matched = sum(1 for b in rna_bcs_kept if b in bc_idx)
    print(f"[shareseq] RNA {M.shape}; rna.bc matched {matched}/{len(rna_bcs_kept)} "
          f"(examples meta={rna_bcs_kept[:2]} h5={bcs[:2]})")
    if matched == 0:
        raise SystemExit("0 rna.bc matched the RNA h5 barcodes -- barcode format mismatch; inspect bcs[:5]")
    gsym = {g.upper(): i for i, g in enumerate(genes)}
    cols = [bc_idx.get(b, -1) for b in rna_bcs_kept]
    expr = {}
    for sym in want:
        gi = gsym.get(sym)
        if gi is None:
            continue
        row = np.asarray(M[gi, :].todense()).ravel()
        expr[sym] = np.array([row[c] if c >= 0 else 0.0 for c in cols])
    print(f"[shareseq] mapped {len(expr)} expression genes, {len(acc)} accessibility genes -> KG nodes")

    expr_n, acc_n = _minmax_log(expr), _minmax_log(acc)
    node_cols = list(kg.node_ids)
    header = node_cols + [f"{n}__atac" for n in node_cols] + ["label", "dataset", "pathway", "timepoint"]
    lines = [",".join(header)]
    for k in range(len(keep)):
        row = {c: 0.0 for c in node_cols}
        arow = {f"{c}__atac": 0.0 for c in node_cols}
        for sym, v in expr_n.items():
            row[node_by_sym[sym]] = v[k]
        for sym, v in acc_n.items():
            arow[f"{node_by_sym[sym]}__atac"] = v[k]
        vals = [f"{row[c]}" for c in node_cols] + [f"{arow[f'{c}__atac']}" for c in node_cols]
        lines.append(",".join(vals + [prog[k], "bmmc_shareseq", "hematopoiesis_shareseq", str(stage[k])]))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[shareseq] wrote {len(keep)} cells x ({len(expr_n)} expr + {len(acc_n)} atac) -> {args.out}")


if __name__ == "__main__":
    main()
