"""Ground the model's intrinsic-memory layer in REAL single-cell transcriptomes
from the CZ CELLxGENE Discover Census.

For each requested cell type we pull expression of the genes mapped to KG nodes
(see `gene_map` in kg.yaml), normalize (CP10K + log1p), average per cell type,
and min-max scale each gene across cell types to [0, 1]. The result is a table
of expression-grounded "memory" vectors -- one per cell type -- that replace the
hand-set 0/1 contexts in contexts.yaml.

`cellxgene-census` is an optional dependency:  uv sync --extra census

What this does and does not give you:
  * DOES: real, data-driven values for the lineage-TF / signaling memory nodes.
  * DOES NOT: an "applied cue" or a measured response-program label -- the Census
    catalogs cell states, not perturbation->phenotype pairs. Cues remain the
    applied perturbation; program labels for supervised training must come from
    an experiment (e.g. Perturb-seq). `--make-training` therefore still labels
    with the mechanistic oracle (honest bootstrap), now over REAL memory vectors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .kg import DATA_DIR, KnowledgeGraph, load_kg

# Our context names -> CELLxGENE cell_type ontology labels (human).
# Xenopus / planarian are not in the human Census, so they have no mapping.
DEFAULT_CELL_TYPES: dict[str, str] = {
    "macrophage": "macrophage",
    "ESC": "embryonic stem cell",
    "myoblast": "myoblast",
    "cardiomyocyte": "cardiac muscle cell",
    "epithelial": "epithelial cell",
    "acinar": "pancreatic acinar cell",
}


def _require():
    try:
        import cellxgene_census  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "cellxgene-census is not installed.\n"
            "Enable the optional data backend with:\n"
            "    uv sync --extra census"
        ) from exc
    import cellxgene_census

    return cellxgene_census


def fetch_mean_expression(
    genes: list[str],
    cell_type_labels: list[str],
    census_version: str = "stable",
    organism: str = "Homo sapiens",
    max_cells_per_type: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Return DataFrame indexed by cell_type label, columns = gene symbols,
    values = mean CP10K-log1p expression."""
    cxg = _require()
    import scipy.sparse as sp

    gene_filter = ", ".join(f"'{g}'" for g in genes)
    org_key = organism.lower().replace(" ", "_")
    rng = np.random.default_rng(seed)
    print(f"[census] opening Census ({census_version}); {len(genes)} genes x "
          f"{len(cell_type_labels)} cell types, <= {max_cells_per_type} cells/type ...")

    ct_filter = ", ".join(f"'{c}'" for c in cell_type_labels)
    with cxg.open_soma(census_version=census_version) as census:
        exp = census["census_data"][org_key]
        # 1) ONE obs scan for all cell types (reading soma_joinid + cell_type
        #    only), then subsample per type in-memory BEFORE touching expression
        #    (common types match millions of cells; a full expression pull is
        #    what made the naive version take ~30 min).
        obs_df = (
            exp.obs.read(
                value_filter=f"cell_type in [{ct_filter}] and is_primary_data == True",
                column_names=["soma_joinid", "cell_type"],
            )
            .concat()
            .to_pandas()
        )
        print(f"[census] matched {len(obs_df)} cells across "
              f"{obs_df['cell_type'].nunique()} types; subsampling ...")
        coords: list[int] = []
        for c in cell_type_labels:
            ids = obs_df.loc[obs_df["cell_type"] == c, "soma_joinid"].to_numpy()
            if ids.size == 0:
                print(f"[census] WARNING: no cells for '{c}'")
                continue
            if ids.size > max_cells_per_type:
                ids = rng.choice(ids, max_cells_per_type, replace=False)
            print(f"[census]   {c}: {ids.size} cells")
            coords.extend(int(i) for i in ids)

        # 2) fetch expression for ONLY the subsampled cells x mapped genes
        adata = cxg.get_anndata(
            census,
            organism=organism,
            obs_coords=np.sort(np.asarray(coords)),
            var_value_filter=f"feature_name in [{gene_filter}]",
            obs_column_names=["cell_type"],
            var_column_names=["feature_name"],
        )
    print(f"[census] fetched {adata.n_obs} cells x {adata.n_vars} genes")

    X = adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    totals = np.asarray(X.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    X = X.multiply(1e4 / totals[:, None]).tocsr()
    X.data = np.log1p(X.data)

    df = pd.DataFrame(X.toarray(), columns=np.asarray(adata.var["feature_name"]))
    df["cell_type"] = np.asarray(adata.obs["cell_type"])
    return df.groupby("cell_type").mean()


def build_real_contexts(
    kg: KnowledgeGraph,
    cell_type_map: dict[str, str] | None = None,
    out_csv: str | Path | None = None,
    **fetch_kw,
) -> pd.DataFrame:
    """Build expression-grounded memory vectors, one row per context name.
    Columns are KG node names (only gene-mapped nodes get values; rest 0)."""
    cell_type_map = cell_type_map or DEFAULT_CELL_TYPES
    node_by_gene = {g: n for n, g in kg.gene_map.items()}
    genes = list(node_by_gene.keys())
    labels = list(dict.fromkeys(cell_type_map.values()))

    if len(labels) < 3:
        print(f"[census] NOTE: only {len(labels)} cell types -> min-max scaling is "
              "degenerate (binary 0/1). Use >=3 types for graded memory values.")

    means = fetch_mean_expression(genes, labels, **fetch_kw)

    # min-max scale each gene across the cell-type panel -> [0, 1]
    lo, hi = means.min(axis=0), means.max(axis=0)
    scaled = (means - lo) / (hi - lo).replace(0, 1.0)

    rows = []
    for ctx_name, ct_label in cell_type_map.items():
        if ct_label not in scaled.index:
            continue
        row = {n: 0.0 for n in kg.node_ids}
        for gene in genes:
            if gene in scaled.columns:
                row[node_by_gene[gene]] = float(scaled.loc[ct_label, gene])
        row_out = {"context": ctx_name, **row}
        rows.append(row_out)

    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[census] wrote real contexts -> {out_csv}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build expression-grounded memory contexts from CELLxGENE Census"
    )
    ap.add_argument("--out", default=str(DATA_DIR / "cellxgene_contexts.csv"))
    ap.add_argument("--version", default="stable", help="Census version")
    ap.add_argument("--max-cells", type=int, default=2000, help="cells/type cap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kg = load_kg()
    df = build_real_contexts(
        kg, out_csv=args.out, census_version=args.version,
        max_cells_per_type=args.max_cells, seed=args.seed,
    )
    print("\nExpression-grounded memory (scaled 0-1), key lineage factors:")
    show = [c for c in ["context", "PU1", "Oct4", "MyoD", "Sox9", "Smad3"] if c in df.columns]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
