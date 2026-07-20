"""Next experiments for the original WLD Colab notebook.

Usage in Colab
--------------
1. Run the original WLD cell first.
2. Upload this file.
3. Execute: %run -i wld_next_experiments.py

Colab may rename the upload. In that case use the actual filename, for example:
    %run -i "wld_next_experiments (1).py"

The default run performs the metric audit and a leakage-reduced rerun. Change
the RUN_* switches below to execute the more expensive modality, prior, and
seed audits one at a time.
"""

from __future__ import annotations

import copy
import math
import os
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# SELECT RUNS
# =============================================================================
def env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


RUN_CURRENT_METRIC_AUDIT = env_flag("WLD_RUN_CURRENT", True)
RUN_LEAKAGE_REDUCED_RERUN = env_flag("WLD_RUN_LEAKAGE", True)
RUN_MODALITY_AUDIT = env_flag("WLD_RUN_MODALITY", False)
RUN_PRIOR_AUDIT = env_flag("WLD_RUN_PRIOR", False)
RUN_SEED_AUDIT = env_flag("WLD_RUN_SEEDS", False)

AUDIT_EPOCHS = 251
SEED_EPOCHS = 201
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


REQUIRED_GLOBALS = [
    "GradedAttentionModel",
    "NUM_GENES",
    "NUM_PEAKS",
    "LATENT_DIM",
    "mask_motif",
    "mask_circuit",
    "X_init_rna",
    "X_init_atac",
    "X_init_atac_genes",
    "raw_target_rna",
    "tr_rna",
    "ts_rna",
    "tr_atac",
    "ts_atac",
    "tr_atac_g",
    "ts_atac_g",
    "tr_target",
    "ts_target",
    "ts_delta",
    "model",
]
missing = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the original WLD cell first. Missing notebook variables: "
        + ", ".join(missing)
    )

print(f"Audit device: {DEVICE}")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(x, y)[0])


def mean_valid(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def state_metrics(true_state, pred_state, initial_state) -> Dict[str, float]:
    true = np.asarray(true_state, dtype=float)
    pred = np.asarray(pred_state, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    true_delta = true - initial
    pred_delta = pred - initial

    per_cell = [safe_pearson(true[i], pred[i]) for i in range(true.shape[0])]
    per_gene = [safe_pearson(true[:, j], pred[:, j]) for j in range(true.shape[1])]
    delta_cosine = F.cosine_similarity(
        torch.tensor(pred_delta, dtype=torch.float32),
        torch.tensor(true_delta, dtype=torch.float32),
        dim=1,
    ).numpy()
    true_norm = np.linalg.norm(true_delta, axis=1)
    pred_norm = np.linalg.norm(pred_delta, axis=1)
    valid_norm = true_norm > 1e-8
    magnitude_ratio = pred_norm[valid_norm] / true_norm[valid_norm]

    return {
        "final_global_r": safe_pearson(true, pred),
        "final_mean_cell_r": mean_valid(per_cell),
        "final_mean_gene_r": mean_valid(per_gene),
        "final_mse": float(mean_squared_error(true, pred)),
        "final_r2": float(r2_score(true.ravel(), pred.ravel())),
        "delta_global_r": safe_pearson(true_delta, pred_delta),
        "delta_mse": float(mean_squared_error(true_delta, pred_delta)),
        "delta_mean_cosine": mean_valid(delta_cosine),
        "delta_positive_direction_fraction": float(np.mean(delta_cosine > 0)),
        "median_magnitude_ratio": float(np.median(magnitude_ratio))
        if magnitude_ratio.size
        else float("nan"),
    }


def print_table(results: Dict[str, Dict[str, float]], title: str) -> pd.DataFrame:
    frame = pd.DataFrame(results).T
    print("\n" + title)
    display(frame.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0))
    return frame


def audit_forward(
    fitted_model,
    baseline_rna: torch.Tensor,
    encoder_rna: torch.Tensor,
    encoder_atac: torch.Tensor,
    atac_gene_gate: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Separate baseline RNA from RNA supplied to the encoder.

    The original forward method uses the same RNA both as an encoder input and
    as the no-change residual. Separating them permits a fair modality audit.
    """
    z_activity = fitted_model.encoder(torch.cat([encoder_rna, encoder_atac], dim=1))
    attn_out, _ = fitted_model.attention(
        z_activity.unsqueeze(1), z_activity.unsqueeze(1), z_activity.unsqueeze(1)
    )
    z_context = attn_out.squeeze(1)
    dz_ode = torch.tanh(
        torch.matmul(
            z_context, fitted_model.W_circuit * fitted_model.circuit_mask
        )
    )
    dz_nn = fitted_model.nn_dynamics(z_activity)
    dz = dz_ode + dz_nn
    dynamic_motifs = torch.clamp(
        fitted_model.known_motifs
        + torch.sigmoid(fitted_model.de_novo_motifs) * 0.5,
        0.0,
        1.0,
    )
    physical_gate = dynamic_motifs.unsqueeze(0) * atac_gene_gate.unsqueeze(1)
    constrained_decoder = fitted_model.dec_weight.unsqueeze(0) * physical_gate
    pred_delta = torch.bmm(dz.unsqueeze(1), constrained_decoder).squeeze(1)
    return pred_delta, torch.relu(baseline_rna + pred_delta)


def fit_variant(
    train_ids: torch.Tensor,
    test_ids: torch.Tensor,
    train_target: torch.Tensor,
    test_target: torch.Tensor,
    *,
    seed: int = 42,
    epochs: int = AUDIT_EPOCHS,
    circuit_mask_override: Optional[torch.Tensor] = None,
    motif_mask_override: Optional[torch.Tensor] = None,
    use_encoder_rna: bool = True,
    use_encoder_atac: bool = True,
    use_atac_gene_gate: bool = True,
    shuffle_atac: bool = False,
    strict_known_motifs: bool = False,
) -> Dict[str, object]:
    set_all_seeds(seed)
    circuit = (
        mask_circuit.detach().clone()
        if circuit_mask_override is None
        else circuit_mask_override.detach().clone()
    )
    motif = (
        mask_motif.detach().clone()
        if motif_mask_override is None
        else motif_mask_override.detach().clone()
    )
    fitted = GradedAttentionModel(
        NUM_GENES, NUM_PEAKS, LATENT_DIM, motif, circuit
    ).to(DEVICE)

    if strict_known_motifs:
        with torch.no_grad():
            fitted.de_novo_motifs.fill_(-10.0)
        fitted.de_novo_motifs.requires_grad_(False)

    base_train = X_init_rna[train_ids].to(DEVICE)
    base_test = X_init_rna[test_ids].to(DEVICE)
    atac_train = X_init_atac[train_ids].to(DEVICE)
    atac_test = X_init_atac[test_ids].to(DEVICE)
    gate_train = X_init_atac_genes[train_ids].to(DEVICE)
    gate_test = X_init_atac_genes[test_ids].to(DEVICE)
    target_train = train_target.to(DEVICE)
    target_test = test_target.to(DEVICE)

    generator = torch.Generator().manual_seed(seed + 1000)
    if shuffle_atac:
        p_train = torch.randperm(len(train_ids), generator=generator)
        p_test = torch.randperm(len(test_ids), generator=generator)
        atac_train, gate_train = atac_train[p_train], gate_train[p_train]
        atac_test, gate_test = atac_test[p_test], gate_test[p_test]

    enc_rna_train = base_train if use_encoder_rna else torch.zeros_like(base_train)
    enc_rna_test = base_test if use_encoder_rna else torch.zeros_like(base_test)
    enc_atac_train = atac_train if use_encoder_atac else torch.zeros_like(atac_train)
    enc_atac_test = atac_test if use_encoder_atac else torch.zeros_like(atac_test)
    if not use_atac_gene_gate:
        gate_train = torch.ones_like(gate_train)
        gate_test = torch.ones_like(gate_test)

    train_delta = target_train - base_train
    optimizer = torch.optim.AdamW(
        [p for p in fitted.parameters() if p.requires_grad],
        lr=0.003,
        weight_decay=1e-4,
    )
    target_cosine = torch.ones(train_delta.shape[0], device=DEVICE)
    for _ in range(epochs):
        fitted.train()
        optimizer.zero_grad(set_to_none=True)
        pred_delta, _ = audit_forward(
            fitted, base_train, enc_rna_train, enc_atac_train, gate_train
        )
        mse = F.mse_loss(pred_delta, train_delta)
        cosine = F.cosine_embedding_loss(pred_delta, train_delta, target_cosine)
        penalty = pred_delta.new_zeros(())
        if fitted.de_novo_motifs.requires_grad:
            penalty = 0.01 * fitted.de_novo_motifs.abs().sum()
        loss = mse + cosine + penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fitted.parameters(), 5.0)
        optimizer.step()

    fitted.eval()
    with torch.no_grad():
        pred_delta, pred_state = audit_forward(
            fitted, base_test, enc_rna_test, enc_atac_test, gate_test
        )
    metrics = state_metrics(
        target_test.cpu().numpy(), pred_state.cpu().numpy(), base_test.cpu().numpy()
    )
    return {
        "model": fitted,
        "metrics": metrics,
        "pred_state": pred_state.cpu(),
        "pred_delta": pred_delta.cpu(),
    }


def training_only_pca_targets(
    train_ids: torch.Tensor, test_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_train = raw_target_rna[train_ids].cpu().numpy()
    raw_test = raw_target_rna[test_ids].cpu().numpy()
    components = min(50, raw_train.shape[0] - 1, raw_train.shape[1])
    if components < 2:
        raise RuntimeError("Too few training pairs for PCA target smoothing.")
    target_pca = PCA(n_components=components, random_state=42)
    target_pca.fit(raw_train)
    smooth_train = target_pca.inverse_transform(target_pca.transform(raw_train))
    smooth_test = target_pca.inverse_transform(target_pca.transform(raw_test))
    return (
        torch.tensor(np.maximum(smooth_train, 0.0), dtype=torch.float32),
        torch.tensor(np.maximum(smooth_test, 0.0), dtype=torch.float32),
    )


# =============================================================================
# RUN 1: CURRENT METRIC AUDIT
# =============================================================================
if RUN_CURRENT_METRIC_AUDIT:
    model.eval()
    with torch.no_grad():
        current_delta, current_state = model(ts_rna, ts_atac, ts_atac_g)
    current_true = ts_target.cpu().numpy()
    current_initial = ts_rna.cpu().numpy()
    current_results = {
        "identity_no_change": state_metrics(
            current_true, current_initial, current_initial
        ),
        "training_target_mean": state_metrics(
            current_true,
            np.repeat(
                tr_target.mean(0, keepdim=True).cpu().numpy(),
                len(ts_target),
                axis=0,
            ),
            current_initial,
        ),
        "current_WLD": state_metrics(
            current_true, current_state.cpu().numpy(), current_initial
        ),
    }
    current_audit = print_table(
        current_results,
        "RUN 1 - CURRENT METRIC AUDIT (final-state and delta metrics)",
    )
    if (
        current_results["identity_no_change"]["final_global_r"]
        >= current_results["current_WLD"]["final_global_r"]
    ):
        print(
            "CAUTION: no-change matches or beats WLD. The final-state Pearson "
            "does not demonstrate learned dynamics."
        )
    if current_results["current_WLD"]["delta_global_r"] < 0.20:
        print(
            "CAUTION: WLD has weak delta correlation even if final-state Pearson is high."
        )


# =============================================================================
# RUN 2: LEAKAGE-REDUCED RERUN
# =============================================================================
CLEAN_CONTEXT = globals().get("CLEAN_CONTEXT")
if RUN_LEAKAGE_REDUCED_RERUN:
    set_all_seeds(42)
    pair_order = torch.randperm(len(X_init_rna), generator=torch.Generator().manual_seed(42))
    clean_split = int(0.80 * len(pair_order))
    clean_train_ids = pair_order[:clean_split]
    clean_test_ids = pair_order[clean_split:]
    clean_train_target, clean_test_target = training_only_pca_targets(
        clean_train_ids, clean_test_ids
    )

    clean_full = fit_variant(
        clean_train_ids,
        clean_test_ids,
        clean_train_target,
        clean_test_target,
        seed=42,
        epochs=AUDIT_EPOCHS,
    )
    clean_initial = X_init_rna[clean_test_ids].cpu().numpy()
    clean_true = clean_test_target.cpu().numpy()

    clean_mean = np.repeat(
        clean_train_target.mean(0, keepdim=True).cpu().numpy(),
        len(clean_test_ids),
        axis=0,
    )
    clean_ridge = Ridge(alpha=1.0)
    clean_ridge.fit(
        X_init_atac[clean_train_ids].cpu().numpy(), clean_train_target.cpu().numpy()
    )
    clean_ridge_pred = np.maximum(
        clean_ridge.predict(X_init_atac[clean_test_ids].cpu().numpy()), 0.0
    )
    clean_results = {
        "identity_no_change": state_metrics(clean_true, clean_initial, clean_initial),
        "training_target_mean": state_metrics(clean_true, clean_mean, clean_initial),
        "ridge_ATAC_to_target": state_metrics(
            clean_true, clean_ridge_pred, clean_initial
        ),
        "WLD_cleaner_split": clean_full["metrics"],
    }
    clean_audit = print_table(
        clean_results,
        "RUN 2 - RANDOM PAIR SPLIT + TARGET PCA FIT ON TRAINING ONLY",
    )
    CLEAN_CONTEXT = {
        "train_ids": clean_train_ids,
        "test_ids": clean_test_ids,
        "train_target": clean_train_target,
        "test_target": clean_test_target,
        "full_result": clean_full,
    }
    print(
        "Remaining limitation: pseudotime, HVGs, pairing, and the early/late "
        "populations were still constructed before this split. This run removes "
        "the target-PCA leak and ordered split, but is not a fully OOD analysis."
    )


def require_clean_context() -> Dict[str, object]:
    if CLEAN_CONTEXT is None:
        raise RuntimeError(
            "RUN_LEAKAGE_REDUCED_RERUN must be True before modality/prior/seed audits."
        )
    return CLEAN_CONTEXT


# =============================================================================
# RUN 3: MODALITY / DIRECT-PROXY AUDIT
# =============================================================================
if RUN_MODALITY_AUDIT:
    ctx = require_clean_context()
    modality_results = {
        "full_RNA+ATAC": ctx["full_result"]["metrics"],
        "RNA_only_no_ATAC_gate": fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            use_encoder_rna=True,
            use_encoder_atac=False,
            use_atac_gene_gate=False,
        )["metrics"],
        "ATAC_only_delta_plus_identity": fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            use_encoder_rna=False,
            use_encoder_atac=True,
            use_atac_gene_gate=True,
        )["metrics"],
        "shuffled_ATAC": fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            shuffle_atac=True,
        )["metrics"],
    }
    modality_audit = print_table(
        modality_results,
        "RUN 3 - MODALITY AND DIRECT-PROXY AUDIT",
    )
    full_delta = modality_results["full_RNA+ATAC"]["delta_global_r"]
    shuffled_delta = modality_results["shuffled_ATAC"]["delta_global_r"]
    if abs(full_delta - shuffled_delta) < 0.03:
        print(
            "CAUTION: shuffling ATAC changes delta r by <0.03; the model is not "
            "showing meaningful cell-specific dependence on ATAC."
        )


# =============================================================================
# RUN 4: PRIOR ABLATION AUDIT
# =============================================================================
if RUN_PRIOR_AUDIT:
    ctx = require_clean_context()
    generator = torch.Generator().manual_seed(42)
    tf_perm = torch.randperm(LATENT_DIM, generator=generator)
    permuted_circuit = mask_circuit[tf_perm][:, tf_perm]
    shuffled_motif = mask_motif.clone()
    for row in range(LATENT_DIM):
        shuffled_motif[row] = shuffled_motif[
            row, torch.randperm(NUM_GENES, generator=generator)
        ].clone()

    prior_results = {
        "full_priors": ctx["full_result"]["metrics"],
        "no_circuit": fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            circuit_mask_override=torch.zeros_like(mask_circuit),
        )["metrics"],
        "permuted_circuit": fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            circuit_mask_override=permuted_circuit,
        )["metrics"],
        "no_motif_restriction": fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            motif_mask_override=torch.ones_like(mask_motif),
        )["metrics"],
        "shuffled_motif": fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            motif_mask_override=shuffled_motif,
        )["metrics"],
        "strict_known_motifs": fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            strict_known_motifs=True,
        )["metrics"],
    }
    prior_audit = print_table(prior_results, "RUN 4 - BIOLOGICAL PRIOR ABLATIONS")
    full = prior_results["full_priors"]["delta_global_r"]
    permuted = prior_results["permuted_circuit"]["delta_global_r"]
    if abs(full - permuted) < 0.03:
        print(
            "CAUTION: permuting the circuit changes delta r by <0.03; the validated "
            "circuit has not demonstrated a specific predictive contribution."
        )


# =============================================================================
# RUN 5: SEED STABILITY
# =============================================================================
if RUN_SEED_AUDIT:
    ctx = require_clean_context()
    seed_rows = {}
    for audit_seed in [1, 7, 21, 42, 99]:
        result = fit_variant(
            ctx["train_ids"],
            ctx["test_ids"],
            ctx["train_target"],
            ctx["test_target"],
            seed=audit_seed,
            epochs=SEED_EPOCHS,
        )
        seed_rows[f"seed_{audit_seed}"] = result["metrics"]
    seed_audit = print_table(seed_rows, "RUN 5 - RANDOM-SEED STABILITY")
    summary = seed_audit.agg(["mean", "std"])
    print("\nSeed summary")
    display(summary.style.format("{:.4f}"))
    if summary.loc["std", "delta_global_r"] > 0.05:
        print("CAUTION: delta r has >0.05 standard deviation across seeds.")


print(
    "\nAudit complete. Interpret final-state Pearson together with identity, "
    "delta correlation, delta cosine, modality shuffling, and prior shuffling."
)
