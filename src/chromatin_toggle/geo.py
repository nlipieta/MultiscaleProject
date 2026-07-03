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


def _gse_suppl_url(gse: str, filename: str) -> str:
    stub = re.sub(r"\d{1,3}$", "nnn", gse)
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/suppl/{filename}"


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


@dataclass
class ScrnaDataset:
    """A GEO scRNA-seq series delivered as a genes x cells CSV + a per-cell
    label file. Maps the authors' cell annotation onto model programs.

    Two labelling modes: a simple `program_map` keyed by one `label_col`, or a
    `labeler(row)->program|None` for composite rules (e.g. cell-type x condition).
    `cue_of(row)->float` gives per-cell cue level (e.g. 0 for untreated baseline);
    otherwise the cue is applied uniformly at `level`."""
    counts_file: str          # suppl filename: genes (rows) x cells (cols) CSV
    labels_file: str          # suppl filename: barcode index + annotation columns
    label_col: str            # column for program_map mode ("" if using labeler)
    cue: str
    program_map: dict[str, str]   # celltypeLabel -> model program (or Quiescent)
    level: float = 1.0
    drop_unmapped: bool = True
    label_sep: str = ","          # labels file delimiter ("\t" for .txt)
    labeler: "object" = None      # optional callable(row)->program|None
    cue_of: "object" = None       # optional callable(row)->float cue level


def _gse120064_label(row):
    """GSE120064: cardiomyocytes only; baseline (0w) -> Quiescent, TAC -> Hypertrophy."""
    if str(row.get("CellType")) != "CM":
        return None
    return "Quiescent" if str(row.get("condition")) == "0w" else "Hypertrophy"


def _gse120064_cue(row):
    return 0.0 if str(row.get("condition")) == "0w" else 1.0  # stretch only post-TAC

# scRNA-seq datasets keyed by GEO accession.
SCRNA: dict[str, ScrnaDataset] = {
    # Ma et al. 2021, pancreatic acinar-to-ductal metaplasia. YFP+ lineage-traced
    # acinar cells, all under repeated caerulein injury. Label = authors' cell-
    # type call (whole-transcriptome clustering), so ADM vs acinar is not a
    # threshold on the KG input genes.
    "GSE172380": ScrnaDataset(
        counts_file="GSE172380_Feature_Barcode_rawCountMatrix_Filtered-YFP%2B_all-samples_QCed.csv.gz",
        labels_file="GSE172380_Cluster%2BCelltypeLabel_YFP%2B_all-samples_QCed.csv.gz",
        label_col="celltypeLabel",
        cue="Caerulein",
        program_map={
            "Acinar": "Quiescent",
            "Acinar.Prolif": "Quiescent",
            "MucinDuctal": "ADM",
            "Ductal-like": "ADM",
            "Acinar/MucinDuctal": "ADM",
            # Tuft, EEC, Tuft/EEC.Progenitor: metaplastic but not ADM -> dropped
        },
    ),
    # Ren et al. 2020, TAC pressure-overload mouse heart. Cardiomyocytes only;
    # baseline (0w) vs sustained overload (2-11w). Composite label + gated cue.
    "GSE120064": ScrnaDataset(
        counts_file="GSE120064_TAC_raw_umi_matrix.csv.gz",
        labels_file="GSE120064_TAC_clean_cell_info_summary.txt.gz",
        label_col="", cue="MechanicalStretch", program_map={},
        label_sep="\t", labeler=_gse120064_label, cue_of=_gse120064_cue,
    ),
}


@dataclass
class H5adDataset:
    """A dataset delivered as an AnnData .h5ad(.gz) with obs cell annotations."""
    url: str
    cell_type_col: str
    program_map: dict[str, str]
    cue: str | None
    level: float = 1.0
    layer: str | None = None


H5AD: dict[str, H5adDataset] = {
    # Rebboah et al. 2021, C2C12 myoblast->myotube (Split-seq). Differentiation is
    # serum-withdrawal-driven, NOT TGF-beta (which inhibits myogenesis), so no cue
    # node is set -- the intrinsic MyoD memory drives the fate.
    "GSE168776": H5adDataset(
        url="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5169nnn/GSM5169183/suppl/GSM5169183_sc_gene.h5ad.gz",
        cell_type_col="sample",
        program_map={"MT_nuclei": "MyogenicDiff",
                     "MB_cells": "Quiescent", "MB_nuclei": "Quiescent"},
        cue=None,
    ),
}


def _download_gunzip(url: str, cache: Path) -> Path:
    """Download url; if it's .gz, gunzip to the stripped name. Returns local path."""
    import gzip as _gz
    import shutil
    fname = url.split("/")[-1]
    gz = _download(url, cache / fname)
    if fname.endswith(".gz"):
        out = cache / fname[:-3]
        if not out.exists():
            with _gz.open(gz, "rb") as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return out
    return gz


def _norm_barcode(b: str) -> str:
    return re.sub(r"[.\-]", "-", str(b).strip().strip('"'))


def ingest_scrna(gse: str, out: Path, cache: Path,
                 include_program_marker_nodes: bool = True) -> None:
    """Build a per-cell training CSV from a genes x cells scRNA count matrix."""
    import pandas as pd

    ds = SCRNA.get(gse)
    if ds is None:
        raise SystemExit(f"No scRNA dataset registered for {gse}. Add to geo.SCRNA.")
    kg = load_kg()
    cache.mkdir(parents=True, exist_ok=True)

    counts = _download(_gse_suppl_url(gse, ds.counts_file),
                       cache / ds.counts_file.replace("%2B", "+"))
    labels = _download(_gse_suppl_url(gse, ds.labels_file),
                       cache / ds.labels_file.replace("%2B", "+"))

    # per-cell label + program (+ optional per-cell cue level)
    lab = pd.read_csv(labels, index_col=0, sep=ds.label_sep)
    lab.index = [_norm_barcode(b) for b in lab.index]
    if ds.labeler is not None:
        lab["__prog"] = lab.apply(ds.labeler, axis=1)
    else:
        lab["__prog"] = lab[ds.label_col].map(ds.program_map)
    lab["__cue"] = lab.apply(ds.cue_of, axis=1) if ds.cue_of is not None else ds.level
    if ds.drop_unmapped:
        lab = lab[lab["__prog"].notna()]
    cell_prog = lab["__prog"].to_dict()
    cell_cue = lab["__cue"].to_dict()

    # KG gene nodes -> mouse symbol (upper). Optionally drop program-marker genes
    # (Sox9 etc.) that co-define the label, to avoid an easy shortcut.
    marker_nodes = {"Sox9", "Autophagy", "mTORC1"} if not include_program_marker_nodes else set()
    want = {sym.upper(): node for node, sym in kg.gene_map.items()
            if node in kg.node_index and node not in marker_nodes}

    # stream the genes x cells matrix in row chunks: accumulate per-cell totals
    # and capture the target gene rows.
    node_counts, cell_ids, totals = {}, None, None
    for chunk in pd.read_csv(counts, index_col=0, chunksize=4000):
        if cell_ids is None:
            cell_ids = [_norm_barcode(c) for c in chunk.columns]
            totals = np.zeros(len(cell_ids), dtype=float)
        totals += chunk.to_numpy(dtype=float).sum(axis=0)
        up = {str(g).upper(): g for g in chunk.index}
        for sym, node in want.items():
            if sym in up:
                node_counts[node] = chunk.loc[up[sym]].to_numpy(dtype=float)

    # CP10K + log1p per captured gene, then min-max scale across cells
    node_cols = list(kg.node_ids)
    ncells = len(cell_ids)
    X = np.zeros((ncells, len(node_cols)), dtype=float)
    safe_tot = np.where(totals > 0, totals, 1.0)
    for node, cnt in node_counts.items():
        norm = np.log1p(cnt / safe_tot * 1e4)
        lo, hi = norm.min(), norm.max()
        X[:, node_cols.index(node)] = 0.0 if hi <= lo else (norm - lo) / (hi - lo)

    # write one row per labelled cell
    header = node_cols + ["label"]
    lines = [",".join(header)]
    kept, prog_count = 0, {}
    for i, bc in enumerate(cell_ids):
        prog = cell_prog.get(bc)
        if prog is None:
            continue
        row = {node_cols[j]: X[i, j] for j in range(len(node_cols))}
        if ds.cue in kg.node_index:
            row[ds.cue] = float(cell_cue.get(bc, ds.level))
        vals = [f"{row[c]}" if c != "label" else prog for c in header]
        lines.append(",".join(vals))
        kept += 1
        prog_count[prog] = prog_count.get(prog, 0) + 1
    out.write_text("\n".join(lines) + "\n")

    print(f"\nIngested {gse} scRNA ({ncells} cells) -> {out}")
    print(f"  mapped gene nodes: {', '.join(sorted(node_counts))}")
    print(f"  cue: {ds.cue}={'per-cell (gated)' if ds.cue_of else ds.level}")
    print(f"  wrote {kept} labelled cells: " +
          ", ".join(f"{p}={n}" for p, n in sorted(prog_count.items())))
    if include_program_marker_nodes:
        print("  NOTE: program-marker genes (Sox9, mTORC1/MTOR, Autophagy/ATG7) are"
              "\n  included as inputs and may correlate with the program label."
              "\n  Pass --no-marker-nodes for a harder, less-circular task.")


def ingest_h5ad_percell(h5ad: Path, out: Path, cell_type_col: str,
                        program_map: dict[str, str], cue: str | None,
                        level: float = 1.0, layer: str | None = None,
                        normalize: bool = True, drop_unmapped: bool = True,
                        max_cells: int | None = None, seed: int = 0) -> None:
    """Per-cell ingestion of an AnnData .h5ad (e.g. a CELLxGENE dataset).

    Labels come from obs[cell_type_col] via program_map; genes named in gene_map
    are read from X (or a named layer), CP10K+log1p normalized if `normalize`,
    then min-max scaled across cells. cue (if given) is applied uniformly.
    """
    import anndata as ad
    import numpy as np

    kg = load_kg()
    adata = ad.read_h5ad(h5ad)
    if cell_type_col not in adata.obs.columns:
        raise SystemExit(f"obs has no '{cell_type_col}'. Columns: {list(adata.obs.columns)}")

    prog = adata.obs[cell_type_col].astype(str).map(program_map)
    keep = prog.notna().to_numpy() if drop_unmapped else np.ones(adata.n_obs, bool)
    idx = np.where(keep)[0]
    if max_cells and idx.size > max_cells:
        idx = np.random.default_rng(seed).choice(idx, max_cells, replace=False)
    adata = adata[idx]
    prog = prog.iloc[idx]

    var_up = {str(v).upper(): i for i, v in enumerate(adata.var_names)}
    want = {node: var_up[sym.upper()] for node, sym in kg.gene_map.items()
            if node in kg.node_index and sym.upper() in var_up}

    M = adata.layers[layer] if layer else adata.X
    M = np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)
    # only CP10K+log1p if the matrix looks like raw counts; else it's already
    # normalized (min-max scaling below is still applied).
    sample = M[: min(M.shape[0], 500)]
    looks_raw = float(M.max()) > 30 and bool(np.allclose(sample, np.round(sample)))
    do_cp10k = normalize and looks_raw
    totals = M.sum(axis=1) if do_cp10k else None

    node_cols = list(kg.node_ids)
    X = np.zeros((adata.n_obs, len(node_cols)), dtype=float)
    for node, vi in want.items():
        col = M[:, vi].astype(float)
        if do_cp10k:
            col = np.log1p(col / np.where(totals > 0, totals, 1.0) * 1e4)
        lo, hi = col.min(), col.max()
        X[:, node_cols.index(node)] = 0.0 if hi <= lo else (col - lo) / (hi - lo)
    print(f"  normalization: {'CP10K+log1p (raw counts)' if do_cp10k else 'min-max only (pre-normalized)'}")

    header = node_cols + ["label"]
    lines = [",".join(header)]
    counts = {}
    labels = prog.to_numpy()
    for i in range(adata.n_obs):
        row = {node_cols[j]: X[i, j] for j in range(len(node_cols))}
        if cue and cue in kg.node_index:
            row[cue] = level
        lines.append(",".join(f"{row[c]}" if c != "label" else labels[i] for c in header))
        counts[labels[i]] = counts.get(labels[i], 0) + 1
    out.write_text("\n".join(lines) + "\n")
    print(f"\nIngested {h5ad.name} ({adata.n_obs} cells) -> {out}")
    print(f"  mapped {len(want)} gene nodes; cue={cue}={level if cue else '-'}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


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
    ap.add_argument("--gse", help="GEO series accession, e.g. GSE21608")
    ap.add_argument("--h5ad-dataset", help="registered h5ad dataset key (geo.H5AD), e.g. GSE168776")
    ap.add_argument("--scrna", action="store_true",
                    help="genes x cells scRNA path (dataset must be in geo.SCRNA)")
    ap.add_argument("--no-marker-nodes", action="store_true",
                    help="scRNA: drop program-marker genes (Sox9/mTORC1/ATG7) from inputs")
    ap.add_argument("--out", default=None, help="output CSV (default data/<gse>.csv)")
    ap.add_argument("--cache", default=None, help="download cache dir")
    args = ap.parse_args()
    cache = Path(args.cache) if args.cache else DATA_DIR / "geo_cache"
    if args.h5ad_dataset:
        ds = H5AD.get(args.h5ad_dataset)
        if ds is None:
            raise SystemExit(f"No h5ad dataset '{args.h5ad_dataset}' in geo.H5AD.")
        cache.mkdir(parents=True, exist_ok=True)
        h5ad = _download_gunzip(ds.url, cache)
        out = Path(args.out) if args.out else DATA_DIR / f"{args.h5ad_dataset}_h5ad.csv"
        ingest_h5ad_percell(h5ad, out, ds.cell_type_col, ds.program_map, ds.cue,
                            level=ds.level, layer=ds.layer)
        return
    if not args.gse:
        raise SystemExit("pass --gse <accession> or --h5ad-dataset <key>")
    if args.scrna:
        out = Path(args.out) if args.out else DATA_DIR / f"{args.gse}_scrna.csv"
        ingest_scrna(args.gse, out, cache,
                     include_program_marker_nodes=not args.no_marker_nodes)
    else:
        out = Path(args.out) if args.out else DATA_DIR / f"{args.gse}.csv"
        ingest(args.gse, out, cache)


if __name__ == "__main__":
    main()
