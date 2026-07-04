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
    label_sep: str = ","          # labels file delimiter ("\t" for .txt/.tsv)
    counts_sep: str = ","         # counts matrix delimiter ("\t" for .tsv)
    labeler: "object" = None      # optional callable(row)->program|None
    cue_of: "object" = None       # optional callable(row)->float cue level


def _gse120064_label(row):
    """GSE120064: cardiomyocytes only; baseline (0w) -> Quiescent, TAC -> Hypertrophy."""
    if str(row.get("CellType")) != "CM":
        return None
    return "Quiescent" if str(row.get("condition")) == "0w" else "Hypertrophy"


def _gse120064_cue(row):
    return 0.0 if str(row.get("condition")) == "0w" else 1.0  # stretch only post-TAC


def _gse113049_label(row):
    """GSE113049: alveolar epithelium after LPS lung injury. Injured AEC2
    substates -> Regeneration; naive alveolar cells -> Quiescent."""
    ct = str(row.get("cell_type"))
    if ct.startswith("Injured AEC2"):
        return "Regeneration"
    if ct in ("Naive AEC2", "Naive AEC1"):
        return "Quiescent"
    return None


def _gse113049_cue(row):
    return 1.0 if str(row.get("cell_type")).startswith("Injured") else 0.0


def _gse143437_label(row):
    """GSE143437: notexin muscle injury. Satellite/progenitor lineage only:
    uninjured (Day 0) = Quiescent satellite; post-injury = Regeneration."""
    if str(row.get("cell_annotation")) != "MuSCs and progenitors":
        return None
    return "Quiescent" if str(row.get("injury")) == "Day 0" else "Regeneration"

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
    # Riemondy/Zepp 2019, LPS lung injury -> alveolar regeneration. Injured AEC2
    # substates = Regeneration; naive alveolar cells = Quiescent. Same LPS cue as
    # trained immunity but a DIFFERENT context/program (alveolar epithelium vs
    # monocyte) -- a real instance of the thesis's context-dependent routing.
    "GSE113049": ScrnaDataset(
        counts_file="GSE113049_count_matrix.tsv.gz",
        labels_file="GSE113049_cell_metadata.tsv.gz",
        label_col="", cue="LPS", program_map={},
        label_sep="\t", counts_sep="\t",
        labeler=_gse113049_label, cue_of=_gse113049_cue,
    ),
    # De Micheli 2020, notexin skeletal-muscle regeneration (34k cells). MuSC/
    # progenitor lineage: uninjured = Quiescent satellite, post-injury =
    # Regeneration. A second regeneration tissue (muscle) alongside GSE113049
    # (lung). No matching KG cue node for notexin -> cue left absent.
    "GSE143437": ScrnaDataset(
        counts_file="GSE143437_DeMicheli_MuSCatlas_rawdata.txt.gz",
        labels_file="GSE143437_DeMicheli_MuSCatlas_metadata.txt.gz",
        label_col="", cue="none", program_map={},
        label_sep="\t", counts_sep="\t", labeler=_gse143437_label,
    ),
    # Zhou et al., caerulein acute pancreatitis (independent ADM replicate of
    # GSE172380). Within the caerulein sample: Ductal = ADM, Acinar = Quiescent.
    "GSE188819": ScrnaDataset(
        counts_file="GSE188819_CER_counts.txt.gz",
        labels_file="GSE188819_CER_metadata.txt.gz",
        label_col="annotated_clusters", cue="Caerulein",
        program_map={"Ductal": "ADM", "Acinar": "Quiescent"},
        label_sep="\t", counts_sep="\t",
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
    # Wellington 2024 iPSC embryoid bodies (CELLxGENE). Pluripotent compartment
    # -> Pluripotency; differentiated lineages -> Quiescent. Finally gives the
    # Pluripotency program real cells (was n=2 from bulk Mullen).
    "EB_pluripotency": H5adDataset(
        url="https://datasets.cellxgene.cziscience.com/734538f1-f640-4618-a78a-b180b9106156.h5ad",
        cell_type_col="cell_type",
        program_map={
            "pluripotent stem cell": "Pluripotency",
            "epithelial cell": "Quiescent", "endothelial cell": "Quiescent",
            "vein endothelial cell": "Quiescent", "cardiac endothelial cell": "Quiescent",
            "hematopoietic precursor cell": "Quiescent",
            "myeloid lineage restricted progenitor cell": "Quiescent",
            "progenitor cell": "Quiescent", "kidney interstitial fibroblast": "Quiescent",
        },
        cue=None,
    ),
    # Reck 2024 human kidney UUO fibrosis (CELLxGENE). Author Annotation.Lvl2:
    # Myofibroblast = Fibrosis, Fibroblast = Quiescent. Second fibrosis organ
    # (kidney) alongside lung IPF (GSE135893).
    "GSE254185": H5adDataset(
        url="https://datasets.cellxgene.cziscience.com/0fe5eee4-380d-4bd9-8735-ada5e03021d9.h5ad",
        cell_type_col="celltype_l2",
        program_map={"Myofibroblast": "Fibrosis", "Fibroblast": "Quiescent"},
        cue="MechanicalStiffness", level=1.0,
    ),
    # Fu/Wang human hypertrophic-cardiomyopathy cardiomyocytes (CELLxGENE).
    # disease HCM = Hypertrophy, normal = Quiescent. Human complement to the
    # mouse TAC set (GSE120064).
    "HCM_cardiac": H5adDataset(
        url="https://datasets.cellxgene.cziscience.com/47a98d37-ba8d-4146-b334-c8ed6385a9e5.h5ad",
        cell_type_col="disease",
        program_map={"hypertrophic cardiomyopathy": "Hypertrophy", "normal": "Quiescent"},
        cue="MechanicalStretch", level=1.0,
    ),
    # Reichart/Chaffin cardiomyopathy cardiac-fibroblast atlas (CELLxGENE, 143k
    # nuclei). Diseased cardiomyopathy fibroblasts = Fibrosis, normal = Quiescent.
    # Large, clean Fibrosis source (cardiac) for the scaling analysis.
    "DCM_fibroblast": H5adDataset(
        url="https://datasets.cellxgene.cziscience.com/9b7c7203-91cd-4e87-aff7-92ed572307dc.h5ad",
        cell_type_col="disease",
        program_map={
            "dilated cardiomyopathy": "Fibrosis",
            "arrhythmogenic right ventricular cardiomyopathy": "Fibrosis",
            "non-compaction cardiomyopathy": "Fibrosis",
            "normal": "Quiescent",
        },
        cue="MechanicalStiffness", level=1.0,
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
    for chunk in pd.read_csv(counts, index_col=0, chunksize=4000, sep=ds.counts_sep):
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

    # CELLxGENE h5ads index var by Ensembl ID; gene symbols live in feature_name.
    symbols = (adata.var["feature_name"] if "feature_name" in adata.var.columns
               else adata.var_names)
    var_up = {str(v).upper(): i for i, v in enumerate(symbols)}
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
        if cue and cue in kg.node_index:              # baseline (Quiescent) cells got no cue
            row[cue] = level if labels[i] != "Quiescent" else 0.0
        lines.append(",".join(f"{row[c]}" if c != "label" else labels[i] for c in header))
        counts[labels[i]] = counts.get(labels[i], 0) + 1
    out.write_text("\n".join(lines) + "\n")
    print(f"\nIngested {h5ad.name} ({adata.n_obs} cells) -> {out}")
    print(f"  mapped {len(want)} gene nodes; cue={cue}={level if cue else '-'}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


@dataclass
class MtxDataset:
    """A 10x-style dataset: sparse matrix.mtx (genes x cells) + genes.tsv +
    barcodes.tsv + a per-cell metadata CSV. Labelled via labeler(meta_row)."""
    mtx_file: str
    genes_file: str
    barcodes_file: str
    meta_file: str
    labeler: "object"          # callable(meta_row_dict) -> program | None
    cue: str
    cue_of: "object" = None    # callable(meta_row_dict) -> float
    meta_sep: str = ","


def _gse135893_label(row):
    ct, diag = str(row.get("celltype")), str(row.get("Diagnosis"))
    activated = {"HAS1 High Fibroblasts", "PLIN2+ Fibroblasts", "Myofibroblasts"}
    if ct in activated and diag == "IPF":
        return "Fibrosis"
    if ct == "Fibroblasts" and diag == "Control":
        return "Quiescent"
    return None


def _gse135893_cue(row):
    return 1.0 if str(row.get("Diagnosis")) == "IPF" else 0.0  # stiffness in fibrotic only


MTX: dict[str, MtxDataset] = {
    # Habermann et al. 2020, human IPF/ILD lung 10x scRNA-seq. Activated
    # fibroblast states (IPF) -> Fibrosis; homeostatic fibroblasts (control) ->
    # Quiescent. Only labelled fibroblast cells are kept.
    "GSE135893": MtxDataset(
        mtx_file="GSE135893_matrix.mtx.gz",
        genes_file="GSE135893_genes.tsv.gz",
        barcodes_file="GSE135893_barcodes.tsv.gz",
        meta_file="GSE135893_IPF_metadata.csv.gz",
        labeler=_gse135893_label, cue="MechanicalStiffness", cue_of=_gse135893_cue,
    ),
}


def ingest_mtx(gse: str, out: Path, cache: Path) -> None:
    """Stream a genes x cells .mtx (COO), keeping only KG-gene rows and labelled
    cell columns; CP10K+log1p normalize, min-max scale, write one row per cell."""
    import pandas as pd

    ds = MTX.get(gse)
    if ds is None:
        raise SystemExit(f"No MTX dataset registered for {gse}. Add to geo.MTX.")
    kg = load_kg()
    cache.mkdir(parents=True, exist_ok=True)
    mtx = _download(_gse_suppl_url(gse, ds.mtx_file), cache / ds.mtx_file)
    genes = _download(_gse_suppl_url(gse, ds.genes_file), cache / ds.genes_file)
    barcodes = _download(_gse_suppl_url(gse, ds.barcodes_file), cache / ds.barcodes_file)
    meta = _download(_gse_suppl_url(gse, ds.meta_file), cache / ds.meta_file)

    # gene symbols (1-based row idx) -> KG node
    want = {sym.upper(): node for node, sym in kg.gene_map.items() if node in kg.node_index}
    row_node = {}
    with gzip.open(genes, "rt") as fh:
        for i, line in enumerate(fh, start=1):
            sym = line.rstrip("\n").split("\t")[-1].strip().upper()
            if sym in want:
                row_node[i] = want[sym]

    # barcodes (1-based col idx) -> barcode string
    with gzip.open(barcodes, "rt") as fh:
        bc_list = [l.strip() for l in fh]
    bc_to_col = {b: i for i, b in enumerate(bc_list, start=1)}

    # metadata -> per-barcode (program, cue); then target columns
    md = pd.read_csv(meta, index_col=0, sep=ds.meta_sep)
    col_prog, col_cue = {}, {}
    for bc, r in md.iterrows():
        prog = ds.labeler(r)
        if prog is None:
            continue
        col = bc_to_col.get(str(bc).strip())
        if col is None:
            continue
        col_prog[col] = prog
        col_cue[col] = float(ds.cue_of(r)) if ds.cue_of else 1.0
    target_cols = set(col_prog)
    print(f"  target: {len(row_node)} gene rows, {len(target_cols)} labelled cells")

    # stream the COO matrix
    colsum = {c: 0.0 for c in target_cols}
    cell_gene = {c: {} for c in target_cols}
    with gzip.open(mtx, "rt") as fh:
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        # line now holds dims; iterate entries
        for line in fh:
            r_s, c_s, v_s = line.split()
            c = int(c_s)
            if c in target_cols:
                v = float(v_s)
                colsum[c] += v
                r = int(r_s)
                if r in row_node:
                    cell_gene[c][row_node[r]] = v

    node_cols = list(kg.node_ids)
    cols = sorted(target_cols)
    raw = {node: np.zeros(len(cols)) for node in set(row_node.values())}
    for k, c in enumerate(cols):
        tot = colsum[c] if colsum[c] > 0 else 1.0
        for node, v in cell_gene[c].items():
            raw[node][k] = np.log1p(v / tot * 1e4)
    scaled = {}
    for node, arr in raw.items():
        lo, hi = arr.min(), arr.max()
        scaled[node] = np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo)

    header = node_cols + ["label"]
    lines = [",".join(header)]
    counts = {}
    for k, c in enumerate(cols):
        row = {nc: 0.0 for nc in node_cols}
        for node in scaled:
            row[node] = scaled[node][k]
        if ds.cue in kg.node_index:
            row[ds.cue] = col_cue[c]
        prog = col_prog[c]
        lines.append(",".join(f"{row[nc]}" if nc != "label" else prog for nc in header))
        counts[prog] = counts.get(prog, 0) + 1
    out.write_text("\n".join(lines) + "\n")
    print(f"\nIngested {gse} MTX ({len(cols)} cells) -> {out}")
    print(f"  mapped {len(scaled)} gene nodes; cue={ds.cue}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


_EMTAB9702 = "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/702/E-MTAB-9702/Files"


def ingest_emtab9702(out: Path, cache: Path, min_counts: int = 500) -> None:
    """Assemble the Bakker 2022 trained-immunity SORT-seq plates (E-MTAB-9702).

    Each plate is one condition; each of its 384 wells is a cell. At the T2 LPS-
    restimulation timepoint, beta-glucan-primed cells -> InnateMemory, RPMI
    (unprimed) -> Quiescent; the cue (LPS) is applied to both, so the trained
    signal lives in the priming-shaped memory, not the cue (matches the theory:
    same cue, different memory -> different program).
    """
    import csv as _csv
    import urllib.request

    kg = load_kg()
    cache.mkdir(parents=True, exist_ok=True)
    meta = _download(f"{_EMTAB9702}/Metadata_TrainedImmunity.csv", cache / "emtab9702_meta.csv")

    # plate -> program (T2 only; BG=InnateMemory, RPMI=Quiescent), skip conflicts
    plate_prog, seen = {}, {}
    with open(meta) as fh:
        for r in _csv.DictReader(fh):
            if "T2" not in r["Timepoint"]:
                continue
            prog = {"BG": "InnateMemory", "RPMI": "Quiescent"}.get(r["Training stimulus"])
            if prog is None:
                continue
            pid = r["Plate_ID"]
            seen.setdefault(pid, set()).add(prog)
    plate_prog = {p: next(iter(v)) for p, v in seen.items() if len(v) == 1}

    # map plate_id -> ReadCounts filename from the directory listing
    listing = urllib.request.urlopen(f"{_EMTAB9702}/").read().decode("utf-8", "ignore")
    file_of = {}
    for fn in re.findall(r"RMC-SM-\d+_[A-Za-z0-9_]*\.ReadCounts\.tsv", listing):
        pid = re.match(r"(RMC-SM-\d+)", fn).group(1)
        file_of[pid] = fn

    want = {sym.upper(): node for node, sym in kg.gene_map.items() if node in kg.node_index}
    node_cols = list(kg.node_ids)
    cell_rows = []  # (program, {node: cp10k_log1p_value})

    for pid, prog in sorted(plate_prog.items()):
        fn = file_of.get(pid)
        if fn is None:
            continue
        tsv = _download(f"{_EMTAB9702}/{fn}", cache / fn)
        with open(tsv) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            nwell = len(header) - 1
            totals = np.zeros(nwell)
            gene_counts = {}  # node -> per-well array
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                vals = np.array(parts[1:], dtype=float)
                totals += vals
                sym = parts[0].split("__")[0].upper()
                if sym in want:
                    gene_counts[want[sym]] = gene_counts.get(want[sym], 0) + vals
        keep = np.where(totals >= min_counts)[0]
        for w in keep:
            tot = totals[w]
            row = {node: float(np.log1p(gene_counts[node][w] / tot * 1e4))
                   for node in gene_counts}
            cell_rows.append((prog, row))
        print(f"  {pid} {prog:<12} wells kept {len(keep)}/{nwell}")

    # min-max scale each node across all kept cells
    mapped = sorted({n for _, row in cell_rows for n in row})
    arrs = {n: np.array([row.get(n, 0.0) for _, row in cell_rows]) for n in mapped}
    for n, a in arrs.items():
        lo, hi = a.min(), a.max()
        arrs[n] = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)

    header = node_cols + ["label"]
    lines = [",".join(header)]
    counts = {}
    for i, (prog, _) in enumerate(cell_rows):
        row = {nc: 0.0 for nc in node_cols}
        for n in mapped:
            row[n] = arrs[n][i]
        if "LPS" in kg.node_index:
            row["LPS"] = 1.0
        lines.append(",".join(f"{row[nc]}" if nc != "label" else prog for nc in header))
        counts[prog] = counts.get(prog, 0) + 1
    out.write_text("\n".join(lines) + "\n")
    print(f"\nIngested E-MTAB-9702 ({len(cell_rows)} cells) -> {out}")
    print(f"  mapped {len(mapped)} gene nodes; cue=LPS (uniform, T2 restimulation)")
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
    ap.add_argument("--mtx", action="store_true",
                    help="10x MTX path (dataset must be in geo.MTX)")
    ap.add_argument("--emtab9702", action="store_true",
                    help="assemble the Bakker 2022 trained-immunity SORT-seq plates")
    ap.add_argument("--min-counts", type=int, default=500,
                    help="emtab9702: min total READ counts per well to call it a cell")
    ap.add_argument("--no-marker-nodes", action="store_true",
                    help="scRNA: drop program-marker genes (Sox9/mTORC1/ATG7) from inputs")
    ap.add_argument("--out", default=None, help="output CSV (default data/<gse>.csv)")
    ap.add_argument("--cache", default=None, help="download cache dir")
    args = ap.parse_args()
    cache = Path(args.cache) if args.cache else DATA_DIR / "geo_cache"
    if args.emtab9702:
        out = Path(args.out) if args.out else DATA_DIR / "emtab9702_trained_immunity.csv"
        ingest_emtab9702(out, cache, min_counts=args.min_counts)
        return
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
    if args.mtx:
        out = Path(args.out) if args.out else DATA_DIR / f"{args.gse}_mtx.csv"
        ingest_mtx(args.gse, out, cache)
        return
    if args.scrna:
        out = Path(args.out) if args.out else DATA_DIR / f"{args.gse}_scrna.csv"
        ingest_scrna(args.gse, out, cache,
                     include_program_marker_nodes=not args.no_marker_nodes)
    else:
        out = Path(args.out) if args.out else DATA_DIR / f"{args.gse}.csv"
        ingest(args.gse, out, cache)


if __name__ == "__main__":
    main()
