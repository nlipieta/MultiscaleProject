"""REAL-LABEL benchmark -- makes the numbers mean something.

Unlike train.py (whose labels come from the mechanistic oracle -> circular), this
trains and evaluates on REAL CELLxGENE annotations:

  features = per-cell expression of the KG-mapped genes (real)
  label    = a real curated obs annotation (default: cell_type; or disease, etc.)
  split    = held out BY DATASET (test datasets are unseen in training -> no
             donor/batch leakage, the honest way to estimate generalization)

It compares three models on the held-out test set:
  * majority-class  (floor)
  * logistic regression on the same features  (linear baseline to beat)
  * KGClassifier    (KG message passing over the features)

If the GNN only matches logistic regression, that's a real and honest finding:
the KG structure adds nothing over a linear model *for this task*. If it beats
it, the structure helps. Either way the number is non-circular.

Requires:  uv sync --extra census
"""
from __future__ import annotations

import argparse

import numpy as np

from .device import pick_device
from .kg import KnowledgeGraph, load_kg

# Default task: distinguish lineages that map to our intrinsic-memory nodes.
# These are well populated in the Census and should be separable by the KG's
# lineage genes (SPI1/macrophage, MYOD1/myoblast, ...), so it directly tests
# whether the KG-gene signature carries real, generalizable identity signal.
DEFAULT_CLASSES = [
    "macrophage",
    "myoblast",
    "cardiac muscle cell",
    "pancreatic acinar cell",
]


def _require():
    try:
        import cellxgene_census  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("cellxgene-census not installed. Run: uv sync --extra census") from exc
    import cellxgene_census

    return cellxgene_census


def fetch_labeled_cells(
    kg: KnowledgeGraph,
    obs_column: str,
    classes: list[str],
    census_version: str = "stable",
    organism: str = "Homo sapiens",
    max_cells_per_class: int = 1500,
    seed: int = 0,
):
    """Return (X [cells, n_nodes] real activations, y [cells] int labels,
    dataset_id [cells] str, classes list, feature_genes list)."""
    cxg = _require()
    import scipy.sparse as sp

    node_by_gene = {g: n for n, g in kg.gene_map.items()}
    genes = list(node_by_gene.keys())
    org_key = organism.lower().replace(" ", "_")
    rng = np.random.default_rng(seed)
    cls_filter = ", ".join(f"'{c}'" for c in classes)

    with cxg.open_soma(census_version=census_version) as census:
        exp = census["census_data"][org_key]
        obs_df = (
            exp.obs.read(
                value_filter=f"{obs_column} in [{cls_filter}] and is_primary_data == True",
                column_names=["soma_joinid", obs_column, "dataset_id"],
            )
            .concat()
            .to_pandas()
        )
        print(f"[bench] matched {len(obs_df)} cells; subsampling <= "
              f"{max_cells_per_class}/class ...")
        keep = []
        for c in classes:
            ids = obs_df.index[obs_df[obs_column] == c].to_numpy()
            if ids.size == 0:
                print(f"[bench] WARNING: no cells for '{c}'")
                continue
            if ids.size > max_cells_per_class:
                ids = rng.choice(ids, max_cells_per_class, replace=False)
            print(f"[bench]   {c}: {ids.size} cells")
            keep.append(ids)
        obs_df = obs_df.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)

        adata = cxg.get_anndata(
            census,
            organism=organism,
            obs_coords=obs_df["soma_joinid"].to_numpy(),
            var_value_filter="feature_name in [%s]" % ", ".join(f"'{g}'" for g in genes),
            obs_column_names=[obs_column, "dataset_id"],
            var_column_names=["feature_name"],
        )

    # CP10K + log1p normalize
    X = adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    totals = np.asarray(X.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    X = X.multiply(1e4 / totals[:, None]).tocsr()
    X.data = np.log1p(X.data)
    Xg = np.asarray(X.todense())
    fetched_genes = list(np.asarray(adata.var["feature_name"]))

    # place each gene's expression on its KG node column
    Xnodes = np.zeros((adata.n_obs, kg.num_nodes), dtype=np.float32)
    for j, gene in enumerate(fetched_genes):
        Xnodes[:, kg.node_index[node_by_gene[gene]]] = Xg[:, j]

    class_index = {c: i for i, c in enumerate(classes)}
    y = np.array([class_index[v] for v in np.asarray(adata.obs[obs_column])])
    dataset_id = np.asarray(adata.obs["dataset_id"]).astype(str)
    return Xnodes, y, dataset_id, classes, fetched_genes


def split_by_dataset(dataset_id, y, holdout_frac=0.3, seed=0):
    """Hold out whole datasets for test (no leakage). Falls back to a stratified
    cell split if there are too few datasets to split cleanly."""
    rng = np.random.default_rng(seed)
    datasets = np.unique(dataset_id)
    if datasets.size >= 3:
        rng.shuffle(datasets)
        n_test = max(1, int(round(datasets.size * holdout_frac)))
        test_ds = set(datasets[:n_test])
        test_mask = np.array([d in test_ds for d in dataset_id])
        # guard: test set must contain >1 class, else fall back
        if np.unique(y[test_mask]).size > 1 and np.unique(y[~test_mask]).size > 1:
            print(f"[bench] split BY DATASET: {datasets.size - n_test} train / "
                  f"{n_test} test datasets")
            return ~test_mask, test_mask
    print("[bench] too few datasets for clean split -> stratified cell split")
    idx = rng.permutation(len(y))
    n_test = int(len(y) * holdout_frac)
    test_mask = np.zeros(len(y), bool)
    test_mask[idx[:n_test]] = True
    return ~test_mask, test_mask


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-label CELLxGENE benchmark")
    ap.add_argument("--obs-column", default="cell_type")
    ap.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    ap.add_argument("--version", default="stable")
    ap.add_argument("--max-cells", type=int, default=1500, help="per class")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    from .model import KGClassifier

    torch.manual_seed(args.seed)
    kg = load_kg()
    X, y, dsid, classes, genes = fetch_labeled_cells(
        kg, args.obs_column, args.classes, census_version=args.version,
        max_cells_per_class=args.max_cells, seed=args.seed,
    )
    print(f"[bench] {X.shape[0]} cells, {len(genes)} genes on KG nodes, "
          f"{len(classes)} classes")

    tr, te = split_by_dataset(dsid, y, args.holdout_frac, args.seed)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    # scale features to [0,1] per gene, FIT ON TRAIN ONLY (no leakage)
    lo = Xtr.min(0, keepdims=True)
    hi = Xtr.max(0, keepdims=True)
    rng_ = np.where((hi - lo) == 0, 1.0, hi - lo)
    Xtr = np.clip((Xtr - lo) / rng_, 0, 1)
    Xte = np.clip((Xte - lo) / rng_, 0, 1)

    # --- baselines ---
    maj = np.bincount(ytr).argmax()
    acc_maj = accuracy_score(yte, np.full_like(yte, maj))

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr, ytr)
    p_lr = lr.predict(Xte)
    acc_lr, f1_lr = accuracy_score(yte, p_lr), f1_score(yte, p_lr, average="macro")

    # --- KG-GNN ---
    device = pick_device(args.device)
    model = KGClassifier(kg, len(classes), hidden=args.hidden, steps=args.steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr_t), ytr_t)
        loss.backward(); opt.step()
        if ep % 40 == 0:
            print(f"  [gnn] epoch {ep}  loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        p_gnn = model(torch.tensor(Xte, dtype=torch.float32, device=device)).argmax(-1).cpu().numpy()
    acc_gnn, f1_gnn = accuracy_score(yte, p_gnn), f1_score(yte, p_gnn, average="macro")

    print("\n================ HELD-OUT TEST (real labels) ================")
    print(f"task: predict '{args.obs_column}' among {classes}")
    print(f"test cells: {len(yte)}   (split held out by dataset)")
    print(f"{'model':<22}{'accuracy':>10}{'macro-F1':>10}")
    print("-" * 42)
    print(f"{'majority-class':<22}{acc_maj:>10.3f}{'-':>10}")
    print(f"{'logistic regression':<22}{acc_lr:>10.3f}{f1_lr:>10.3f}")
    print(f"{'KG-GNN':<22}{acc_gnn:>10.3f}{f1_gnn:>10.3f}")
    print("=============================================================")


if __name__ == "__main__":
    main()
