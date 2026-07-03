"""Ingest a GEO expression series into the model's per-node CSV schema.

Bridges REAL public data (NCBI GEO) into the generic CSV interface that
`dataset.load_csv` / `chromatin-train --data` already consume. It:

  1. downloads a GEO ``*_series_matrix.txt.gz`` (sample metadata + a
     probe x sample expression matrix),
  2. downloads the platform ``*.annot.gz`` to map probe IDs -> gene symbols,
  3. averages probe values onto the KG nodes named in ``gene_map`` (kg.yaml),
     min-max scaling each node to [0, 1] across samples,
  4. reads the applied cue and the observed response-program label from each
     sample's metadata via a small, per-dataset ``SampleRule`` table,
  5. writes one row per sample in the model's node-column + ``label`` schema.

It is reusable for any GEO expression series: add a ``SampleRule`` set keyed by
GSE accession. The included rules cover Mullen et al. 2011 (GSE21608), the
TGF-beta cell-identity study underlying pathways 5/6.

IMPORTANT limitations are dataset-specific and printed at ingest time. For
GSE21608 the matrix holds two-color log-ratios (TGF-beta response), not absolute
expression, and n=6 across 3 cell types -- a real-data *proof of ingestion*, not
a training set. Prefer one-color / RNA-seq series (absolute expression, larger
n) for genuine supervised training.
"""
from __future__ import annotations

import gzip
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kg import DATA_DIR, load_kg


def _gse_matrix_url(gse: str) -> str:
    stub = re.sub(r"\d{1,3}$", "nnn", gse)
    return (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}"
            f"/matrix/{gse}_series_matrix.txt.gz")


def _gpl_annot_url(gpl: str) -> str:
    stub = re.sub(r"\d{1,3}$", "nnn", gpl)
    return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{stub}/{gpl}/annot/{gpl}.annot.gz"


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  fetching {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


@dataclass
class SampleRule:
    """How to read one sample's model coordinates from its GEO title.

    match: regex tested against the sample title. cue/level/label/context are the
    model coordinates assigned when it matches (cue=None -> no cue applied).
    """
    match: str
    label: str
    context: str
    cue: str | None = None
    level: float = 1.0


# Per-dataset rules. Sample titles come from !Sample_title in the series matrix.
RULES: dict[str, list[SampleRule]] = {
    # Mullen 2011, GSE21608 (mouse, two-color Agilent GPL4134). Only the myotube
    # arm maps cleanly to a model cue+program; the ES arm used a TGF-beta
    # INHIBITOR (SB431542), so its cue is left absent and it is labelled by
    # lineage identity (Pluripotency) rather than a TGF-beta-driven transition.
    "GSE21608": [
        SampleRule(r"C2C12_myoT.*TGFb", label="MyogenicDiff", context="myoblast",
                   cue="TGFbeta", level=1.0),
        SampleRule(r"ES_cell.*SB431542", label="Pluripotency", context="ESC",
                   cue=None),
        SampleRule(r"38B9.*TGFb", label="Quiescent", context="neutral",
                   cue="TGFbeta", level=1.0),  # pro-B: no program node in model
    ],
    "GSE21610": [],  # ChIP-seq platform partner; not expression -> unmapped
}


def _parse_series_matrix(path: Path):
    """Return (sample_titles, platform_id, probe_ids, value_matrix [P x S])."""
    titles: list[str] = []
    platform = None
    probe_ids: list[str] = []
    rows: list[list[float]] = []
    in_table = False
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Series_platform_id"):
                platform = line.rstrip("\n").split("\t")[1].strip('"')
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
                next_is_header = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                parts = line.rstrip("\n").split("\t")
                if next_is_header:
                    next_is_header = False
                    continue  # column header (ID_REF, GSM...)
                probe_ids.append(parts[0].strip('"'))
                vals = []
                for v in parts[1:]:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(np.nan)
                rows.append(vals)
    return titles, platform, probe_ids, np.array(rows, dtype=float)


def _parse_probe_to_symbol(path: Path) -> dict[str, str]:
    """Map probe ID -> UPPERCASE gene symbol from a GEO .annot.gz table."""
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt") as fh:
        header_cols = None
        sym_col = None
        for line in fh:
            if line.startswith(("#", "^", "!")):
                continue
            parts = line.rstrip("\n").split("\t")
            if header_cols is None:
                header_cols = parts
                for i, c in enumerate(parts):
                    if c.strip().lower() in ("gene symbol", "gene_symbol"):
                        sym_col = i
                        break
                continue
            if sym_col is not None and len(parts) > sym_col:
                sym = parts[sym_col].strip()
                if sym:
                    mapping[parts[0].strip()] = sym.upper()
    return mapping


def ingest(gse: str, out: Path, cache: Path) -> None:
    kg = load_kg()
    rules = RULES.get(gse)
    if not rules:
        raise SystemExit(f"No SampleRule set for {gse}. Add one to geo.RULES.")

    cache.mkdir(parents=True, exist_ok=True)
    mat = _download(_gse_matrix_url(gse), cache / f"{gse}_series_matrix.txt.gz")
    titles, platform, probe_ids, values = _parse_series_matrix(mat)
    annot = _download(_gpl_annot_url(platform), cache / f"{platform}.annot.gz")
    probe2sym = _parse_probe_to_symbol(annot)

    # gene symbol (upper) -> list of matrix row indices
    sym_rows: dict[str, list[int]] = {}
    for i, pid in enumerate(probe_ids):
        sym = probe2sym.get(pid)
        if sym:
            sym_rows.setdefault(sym, []).append(i)

    # Assemble a value per KG gene node per sample (mean over its probes).
    node_cols = list(kg.node_ids)
    S = len(titles)
    X = np.zeros((S, len(node_cols)), dtype=float)
    mapped_nodes = []
    for node, sym in kg.gene_map.items():
        if node not in kg.node_index:
            continue
        rows_i = sym_rows.get(sym.upper())
        if not rows_i:
            continue
        col = node_cols.index(node)
        X[:, col] = np.nanmean(values[rows_i, :], axis=0)
        mapped_nodes.append(node)

    # min-max scale each mapped node column to [0,1] across samples
    for node in mapped_nodes:
        j = node_cols.index(node)
        lo, hi = np.nanmin(X[:, j]), np.nanmax(X[:, j])
        X[:, j] = 0.0 if hi <= lo else (X[:, j] - lo) / (hi - lo)
    X = np.nan_to_num(X, nan=0.0)

    # assign cue + label from sample titles via rules; overlay cue level
    kept = []
    for s, title in enumerate(titles):
        rule = next((r for r in rules if re.search(r.match, title)), None)
        if rule is None:
            print(f"  (skip unmatched sample: {title})")
            continue
        row = {node_cols[j]: float(X[s, j]) for j in range(len(node_cols))}
        if rule.cue and rule.cue in kg.node_index:
            row[rule.cue] = float(rule.level)
        row["label"] = rule.label
        kept.append((title, rule, row))

    header = node_cols + ["label"]
    lines = [",".join(header)]
    for _, _, row in kept:
        lines.append(",".join(f"{row[c]}" if c != "label" else row["label"]
                              for c in header))
    out.write_text("\n".join(lines) + "\n")

    print(f"\nIngested {gse} ({platform}, {S} samples) -> {out}")
    print(f"  mapped {len(mapped_nodes)}/{len(kg.gene_map)} gene nodes: "
          f"{', '.join(mapped_nodes)}")
    print(f"  wrote {len(kept)} labelled rows:")
    for title, rule, _ in kept:
        cue = f"{rule.cue}={rule.level}" if rule.cue else "no-cue"
        print(f"    {title:<32} {cue:<16} -> {rule.label}")
    print("\n  NOTE: verify the value semantics before training. Two-color arrays"
          "\n  (e.g. GSE21608) store log-ratios = perturbation RESPONSE, not"
          "\n  absolute memory state, and n is small. Use one-color/RNA-seq"
          "\n  series with absolute expression for genuine supervised training.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Ingest a GEO expression series to the model CSV schema")
    ap.add_argument("--gse", required=True, help="GEO series accession, e.g. GSE21608")
    ap.add_argument("--out", default=None, help="output CSV (default data/<gse>.csv)")
    ap.add_argument("--cache", default=None, help="download cache dir")
    args = ap.parse_args()
    out = Path(args.out) if args.out else DATA_DIR / f"{args.gse}.csv"
    cache = Path(args.cache) if args.cache else DATA_DIR / "geo_cache"
    ingest(args.gse, out, cache)


if __name__ == "__main__":
    main()
