"""Colab runner for leakage-aware WLD state reconstruction on 10x PBMC multiome.

This runner performs the analysis that the single-snapshot dataset can support:
predicting held-out RNA state from ATAC accessibility under external regulatory
priors. It deliberately does not report temporal trajectory, AUPRC, fixed-point,
or attractor-stability results because this dataset has no donors, time points,
lineage tracing, perturbation endpoints, or predeclared state labels.

Expected companion file: wld_attractor_model_v2.py. Colab may append " (1)"
or another integer to the filename; the loader handles that automatically.
"""

from __future__ import annotations

import bisect
import glob
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import decoupler as dc
import matplotlib.pyplot as plt
import mygene
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Reproducibility and resource controls
# -----------------------------------------------------------------------------
SEED = 42
N_CELLS = 1500
N_GENES_REQUESTED = 400
N_PEAKS = 1000
N_TFS = 48
MAX_LINK_DISTANCE = 250_000
BATCH_SIZE = 128
MAX_EPOCHS = 160
PATIENCE = 18
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4

DATA_URL = (
    "https://cf.10xgenomics.com/samples/cell-arc/1.0.0/"
    "pbmc_granulocyte_sorted_10k/"
    "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
)
DATA_PATH = Path("pbmc10k_multiome.h5")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_module():
    candidates = sorted(
        glob.glob("wld_attractor_model_v2*.py"), key=os.path.getmtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            "Upload wld_attractor_model_v2.py into the Colab session before running."
        )
    path = candidates[0]
    spec = importlib.util.spec_from_file_location("wld_model_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print(f"Loaded model architecture: {path}")
    return module


def dense_float32(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Development-only cell split; not a donor-level OOD split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def choose_hv_genes(rna_raw, train_idx: np.ndarray, n_top: int) -> List[str]:
    train = rna_raw[train_idx].copy()
    sc.pp.filter_genes(train, min_cells=max(5, int(0.01 * train.n_obs)))
    sc.pp.normalize_total(train, target_sum=1e4)
    sc.pp.log1p(train)
    n_top = min(n_top, max(50, train.n_vars - 1))
    sc.pp.highly_variable_genes(train, n_top_genes=n_top, flavor="seurat")
    return train.var_names[train.var["highly_variable"]].tolist()


def normalize_rna(rna_raw, genes: Sequence[str]) -> np.ndarray:
    rna = rna_raw[:, list(genes)].copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    return dense_float32(rna.X)


def canonical_chrom(value) -> Optional[str]:
    value = str(value)
    value = value.removeprefix("chr")
    if value in {str(i) for i in range(1, 23)} | {"X", "Y"}:
        return "chr" + value
    return None


def gene_tss_from_mygene(genes: Sequence[str]) -> Dict[str, Tuple[str, int]]:
    """Resolve GRCh38 transcription start sites without using expression outcomes."""
    mg = mygene.MyGeneInfo()
    records = mg.querymany(
        list(genes),
        scopes="symbol",
        fields="symbol,genomic_pos_hg38",
        species="human",
        as_dataframe=False,
        verbose=False,
    )
    result: Dict[str, Tuple[str, int]] = {}
    requested = set(genes)
    for record in records:
        query = str(record.get("query", ""))
        symbol = str(record.get("symbol", query))
        key = query if query in requested else symbol
        positions = record.get("genomic_pos_hg38")
        if not positions:
            continue
        if isinstance(positions, dict):
            positions = [positions]
        chosen = None
        for pos in positions:
            chrom = canonical_chrom(pos.get("chr"))
            if chrom is None:
                continue
            strand = int(pos.get("strand", 1))
            start, end = int(pos["start"]), int(pos["end"])
            tss = start if strand >= 0 else end
            chosen = (chrom, tss)
            break
        if chosen is not None and key not in result:
            result[key] = chosen
    return result


PEAK_RE = re.compile(r"^(chr(?:[0-9]+|X|Y)):(\d+)-(\d+)$")


def map_peaks_to_nearest_gene(
    peak_names: Sequence[str],
    genes: Sequence[str],
    gene_tss: Dict[str, Tuple[str, int]],
    max_distance: int,
) -> List[Optional[Tuple[int, int]]]:
    by_chrom: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for gene_index, gene in enumerate(genes):
        if gene in gene_tss:
            chrom, tss = gene_tss[gene]
            by_chrom[chrom].append((tss, gene_index))
    for chrom in by_chrom:
        by_chrom[chrom].sort()

    mapped: List[Optional[Tuple[int, int]]] = []
    for peak in peak_names:
        match = PEAK_RE.match(str(peak))
        if match is None:
            mapped.append(None)
            continue
        chrom, start, end = match.groups()
        center = (int(start) + int(end)) // 2
        entries = by_chrom.get(chrom)
        if not entries:
            mapped.append(None)
            continue
        coordinates = [item[0] for item in entries]
        location = bisect.bisect_left(coordinates, center)
        neighbors = []
        if location < len(entries):
            neighbors.append(entries[location])
        if location > 0:
            neighbors.append(entries[location - 1])
        tss, gene_index = min(neighbors, key=lambda x: abs(x[0] - center))
        distance = abs(tss - center)
        mapped.append((gene_index, distance) if distance <= max_distance else None)
    return mapped


def select_accessible_linked_peaks(
    atac_raw,
    train_idx: np.ndarray,
    peak_mapping: Sequence[Optional[Tuple[int, int]]],
    n_peaks: int,
    n_genes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select variable accessible peaks on training cells, preserving gene coverage."""
    train_matrix = atac_raw[train_idx].X
    if sp.issparse(train_matrix):
        prevalence = np.asarray((train_matrix > 0).mean(axis=0)).ravel()
    else:
        prevalence = (np.asarray(train_matrix) > 0).mean(axis=0)
    score = prevalence * (1.0 - prevalence)

    candidates = [
        (peak_idx, mapping[0], mapping[1], score[peak_idx])
        for peak_idx, mapping in enumerate(peak_mapping)
        if mapping is not None and 0.002 <= prevalence[peak_idx] <= 0.80
    ]
    if not candidates:
        raise RuntimeError("No ATAC peaks could be linked to the selected genes.")

    # First keep the strongest accessible peak for every gene, then fill by
    # training-only accessibility variance.
    best_per_gene: Dict[int, Tuple[int, int, int, float]] = {}
    for item in candidates:
        peak_idx, gene_idx, distance, value = item
        current = best_per_gene.get(gene_idx)
        if current is None or value > current[3]:
            best_per_gene[gene_idx] = item
    selected = {item[0] for item in best_per_gene.values()}
    for item in sorted(candidates, key=lambda x: x[3], reverse=True):
        if len(selected) >= n_peaks:
            break
        selected.add(item[0])
    selected_idx = np.array(sorted(selected), dtype=int)
    if selected_idx.size > n_peaks:
        selected_idx = selected_idx[:n_peaks]

    peak_to_gene = np.zeros((selected_idx.size, n_genes), dtype=np.float32)
    for row, peak_idx in enumerate(selected_idx):
        gene_idx, distance = peak_mapping[peak_idx]  # type: ignore[index]
        peak_to_gene[row, gene_idx] = math.exp(-distance / 100_000.0)
    return selected_idx, peak_to_gene


def binary_atac(atac_raw, peak_idx: np.ndarray) -> np.ndarray:
    matrix = atac_raw[:, peak_idx].X
    if sp.issparse(matrix):
        matrix = matrix.copy()
        matrix.data = np.ones_like(matrix.data, dtype=np.float32)
    else:
        matrix = (np.asarray(matrix) > 0).astype(np.float32)
    return dense_float32(matrix)


def compile_collectri_priors(
    genes: Sequence[str], n_tfs: int
) -> Tuple[List[str], np.ndarray, np.ndarray, pd.DataFrame]:
    """Compile signed CollecTRI TF-target and TF-TF priors.

    CollecTRI interaction support is used here as a protein-DNA feasibility
    proxy. Manuscript-grade motif localization should replace motif_tf_gene
    with chromVAR, ChIP/CUT&RUN, SCENIC+, or equivalent occupancy evidence.
    """
    net = dc.op.collectri(organism="human")
    required = {"source", "target", "weight"}
    if not required.issubset(net.columns):
        raise RuntimeError(f"Unexpected CollecTRI columns: {net.columns.tolist()}")
    net = net.dropna(subset=["source", "target", "weight"]).copy()
    net["source"] = net["source"].astype(str)
    net["target"] = net["target"].astype(str)

    sources, targets = set(net["source"]), set(net["target"])
    circuit_capable = sources.intersection(targets)
    gene_edges = net[net["target"].isin(set(genes))]
    counts = gene_edges.groupby("source").size().sort_values(ascending=False)
    ranked = [tf for tf in counts.index if tf in circuit_capable]
    if len(ranked) < min(12, n_tfs):
        ranked.extend(tf for tf in counts.index if tf not in ranked)
    tfs = ranked[:n_tfs]
    if len(tfs) < 4:
        raise RuntimeError("Too few CollecTRI regulators overlap the selected genes.")

    tf_index = {tf: i for i, tf in enumerate(tfs)}
    gene_index = {gene: i for i, gene in enumerate(genes)}
    motif_tf_gene = np.zeros((len(tfs), len(genes)), dtype=np.float32)
    for row in gene_edges.itertuples(index=False):
        if row.source in tf_index and row.target in gene_index:
            motif_tf_gene[tf_index[row.source], gene_index[row.target]] = 1.0

    circuit = np.zeros((len(tfs), len(tfs)), dtype=np.float32)
    circuit_edges = net[net["source"].isin(tfs) & net["target"].isin(tfs)]
    for row in circuit_edges.itertuples(index=False):
        circuit[tf_index[row.source], tf_index[row.target]] = float(row.weight)
    return tfs, motif_tf_gene, circuit, circuit_edges


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def reconstruction_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    per_cell = [pearson_safe(true[i], pred[i]) for i in range(true.shape[0])]
    per_gene = [pearson_safe(true[:, j], pred[:, j]) for j in range(true.shape[1])]
    per_cell = [x for x in per_cell if np.isfinite(x)]
    per_gene = [x for x in per_gene if np.isfinite(x)]
    return {
        "global_pearson": pearson_safe(true, pred),
        "mean_per_cell_pearson": float(np.mean(per_cell)) if per_cell else float("nan"),
        "mean_per_gene_pearson": float(np.mean(per_gene)) if per_gene else float("nan"),
        "mse": float(mean_squared_error(true, pred)),
        "r2_flattened": float(r2_score(true.ravel(), pred.ravel())),
    }


@torch.no_grad()
def predict_state(model, atac: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    results = []
    for start in range(0, atac.shape[0], batch_size):
        batch = atac[start : start + batch_size].to(device)
        z0, binding_gate = model.encode(batch)
        pred = model.decode(z0, binding_gate)
        results.append(pred.cpu().numpy())
    return np.concatenate(results, axis=0)


def train_state_model(
    model,
    atac: torch.Tensor,
    rna: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
) -> Dict[str, List[float]]:
    train_dataset = TensorDataset(atac[train_idx], rna[train_idx])
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = {"train": [], "val": []}
    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        for batch_atac, batch_rna in loader:
            batch_atac, batch_rna = batch_atac.to(device), batch_rna.to(device)
            optimizer.zero_grad(set_to_none=True)
            z0, binding_gate = model.encode(batch_atac)
            pred = model.decode(z0, binding_gate)
            mse = F.mse_loss(pred, batch_rna)
            cosine = 1.0 - F.cosine_similarity(pred, batch_rna, dim=-1).mean()
            loss = mse + 0.20 * cosine
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_atac = atac[val_idx].to(device)
            val_rna = rna[val_idx].to(device)
            z0, binding_gate = model.encode(val_atac)
            val_pred = model.decode(z0, binding_gate)
            val_loss = float(F.mse_loss(val_pred, val_rna).cpu())
        train_loss = float(np.mean(batch_losses))
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val_mse={val_loss:.4f}")
        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best validation MSE={best_val:.4f}")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    return history


def plot_results(history: Dict[str, List[float]], metrics: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train"], label="training objective")
    axes[0].plot(history["val"], label="validation MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training history")
    axes[0].legend()

    names = list(metrics)
    values = [metrics[name]["global_pearson"] for name in names]
    axes[1].bar(names, values, color=["#78909c", "#90a4ae", "#147d84"][: len(names)])
    axes[1].set_ylim(-0.1, 1.0)
    axes[1].set_ylabel("Global Pearson r")
    axes[1].set_title("Held-out state reconstruction")
    for i, value in enumerate(values):
        axes[1].text(i, value + 0.025, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("wld_pbmc_state_results.png", dpi=180, bbox_inches="tight")
    plt.show()


def main() -> None:
    started = time.time()
    set_seed(SEED)
    model_module = load_model_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        "\nLIMITATION: This 10x dataset is a single biological sample and a single "
        "snapshot. The split below is a development-only random cell split, not "
        "donor-level OOD validation. Dynamics and attractor stability are N/A.\n"
    )

    if not DATA_PATH.exists():
        print("Downloading 10x PBMC multiome matrix...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    adata = sc.read_10x_h5(DATA_PATH, gex_only=False)
    adata.var_names_make_unique()
    rng = np.random.default_rng(SEED)
    if adata.n_obs > N_CELLS:
        chosen = np.sort(rng.choice(adata.n_obs, N_CELLS, replace=False))
        adata = adata[chosen].copy()
    rna_raw = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    atac_raw = adata[:, adata.var["feature_types"] == "Peaks"].copy()
    del adata
    print(f"Selected {rna_raw.n_obs} cells, {rna_raw.n_vars} genes, {atac_raw.n_vars} peaks")

    train_idx, val_idx, test_idx = split_indices(rna_raw.n_obs, SEED)
    hvg = choose_hv_genes(rna_raw, train_idx, N_GENES_REQUESTED)
    print(f"Resolving GRCh38 TSS coordinates for {len(hvg)} training-selected genes...")
    tss = gene_tss_from_mygene(hvg)
    genes = [gene for gene in hvg if gene in tss]
    if len(genes) < 100:
        raise RuntimeError(f"Only {len(genes)} genes had usable GRCh38 coordinates.")
    print(f"Found {len(genes)} training-selected genes with genomic coordinates")

    tfs, motif_tf_gene, circuit, circuit_edges = compile_collectri_priors(genes, N_TFS)
    supported = motif_tf_gene.sum(axis=0) > 0
    genes = [gene for gene, keep in zip(genes, supported) if keep]
    motif_tf_gene = motif_tf_gene[:, supported]
    if len(genes) < 100:
        raise RuntimeError(
            f"Only {len(genes)} genes had both genomic and CollecTRI support."
        )
    print(
        f"Compiled {len(tfs)} regulators, {int(motif_tf_gene.sum())} TF-gene edges, "
        f"and {int(np.count_nonzero(circuit))} signed TF-TF circuit edges"
    )
    print(
        "Binding note: CollecTRI is a curated interaction prior, not a sequence-level "
        "motif scan. Replace motif_tf_gene with occupancy-localized evidence for a "
        "manuscript-grade binding claim."
    )

    peak_mapping = map_peaks_to_nearest_gene(
        atac_raw.var_names.tolist(), genes, tss, MAX_LINK_DISTANCE
    )
    selected_peaks, peak_to_gene = select_accessible_linked_peaks(
        atac_raw,
        train_idx,
        peak_mapping,
        n_peaks=N_PEAKS,
        n_genes=len(genes),
    )
    atac_np = binary_atac(atac_raw, selected_peaks)
    rna_np = normalize_rna(rna_raw, genes)
    print(f"Model matrix: {atac_np.shape[0]} cells x {atac_np.shape[1]} linked peaks")

    priors = model_module.PriorMatrices(
        peak_to_gene=torch.from_numpy(peak_to_gene),
        motif_tf_gene=torch.from_numpy(motif_tf_gene),
        circuit_tf_tf=torch.from_numpy(circuit),
    )
    model = model_module.PriorConstrainedAttractorModel(
        priors=priors, cue_dim=0, hidden_dim=256
    ).to(device)
    # A snapshot does not identify dz/dt. Freeze the ODE rather than fitting an
    # arbitrary vector field to contemporaneous state-reconstruction targets.
    for parameter in model.vector_field.parameters():
        parameter.requires_grad_(False)
    atac_tensor = torch.from_numpy(atac_np)
    rna_tensor = torch.from_numpy(rna_np)
    history = train_state_model(
        model, atac_tensor, rna_tensor, train_idx, val_idx, device
    )

    true_test = rna_np[test_idx]
    model_pred = predict_state(model, atac_tensor[test_idx], device, BATCH_SIZE)
    mean_pred = np.repeat(rna_np[train_idx].mean(axis=0, keepdims=True), len(test_idx), axis=0)

    # Ridge receives the same accessibility-derived gene activity, not RNA or labels.
    gene_activity = atac_np @ peak_to_gene
    ridge = Ridge(alpha=10.0)
    ridge.fit(gene_activity[train_idx], rna_np[train_idx])
    ridge_pred = np.maximum(ridge.predict(gene_activity[test_idx]), 0.0)

    metrics = {
        "training_mean": reconstruction_metrics(true_test, mean_pred),
        "ridge_gene_activity": reconstruction_metrics(true_test, ridge_pred),
        "prior_constrained_model": reconstruction_metrics(true_test, model_pred),
    }
    report = {
        "scope": "single-snapshot cross-modal state reconstruction",
        "split": "random cells; development only; not donor-level OOD",
        "n_cells": int(rna_np.shape[0]),
        "n_genes": int(rna_np.shape[1]),
        "n_peaks": int(atac_np.shape[1]),
        "n_tfs": len(tfs),
        "metrics": metrics,
        "trajectory_metrics": "N/A - no observed transitions",
        "auprc": "N/A - no predeclared binary target",
        "fixed_point_stability": "N/A - temporal vector field not identified",
        "vector_field_training": "not trained; frozen for snapshot-only analysis",
        "binding_prior": "CollecTRI interaction proxy; not sequence-localized motif occupancy",
        "runtime_seconds": round(time.time() - started, 1),
    }
    print("\nHELD-OUT STATE RECONSTRUCTION RESULTS")
    print(json.dumps(report, indent=2))
    with open("wld_pbmc_results.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "genes": genes,
        "tfs": tfs,
        "selected_peak_names": atac_raw.var_names[selected_peaks].tolist(),
        "peak_to_gene": peak_to_gene,
        "motif_tf_gene": motif_tf_gene,
        "circuit_tf_tf": circuit,
        "limitations": report,
    }
    torch.save(checkpoint, "wld_pbmc_state_model.pt")
    plot_results(history, metrics)
    print(
        "\nSaved: wld_pbmc_results.json, wld_pbmc_state_model.pt, "
        "wld_pbmc_state_results.png"
    )


if __name__ == "__main__":
    main()
