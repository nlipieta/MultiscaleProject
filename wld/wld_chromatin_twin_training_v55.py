"""Leakage-safe whole-target training for the WLD v5.5 chromatin twin.

The v5.3 response matrix remains sparse until a sampled population is moved
to the accelerator.  Model selection and uncertainty calibration use disjoint
whole perturbation targets.  Every fitted condition is content locked and
restart safe; the sealed test partition is never materialized or evaluated.

This module evaluates transient chromatin response.  It cannot identify an
ODE time scale, a fixed point, a basin, an attractor, or a continuously
synchronized biological/clinical digital twin.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import random
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import Tensor

from wld_chromatin_modules_v55 import (
    ComplexAccessibilityModuleAtlas,
    SparseFullChromatinBundle,
    load_complex_module_atlas,
    load_v53_sparse_full_bundle,
    sha256_file,
)
from wld_chromatin_twin_v55 import (
    BranchOverrides,
    ChromatinTwinPriors,
    WLDChromatinDigitalTwin,
    architecture_contract,
    degree_preserving_bipartite_shuffle,
    topology_digest,
)
from wld_foundation_model_v4 import WLDMultistudyFoundationModel
from wld_phase_b_priors import load_phase_b_priors, verify_phase_b_priors
from wld_twin_statistics_v55 import (
    calibrate_ensemble_intervals,
    evaluate_claims,
    paired_target_bootstrap,
    target_bootstrap_mean,
)


SCHEMA_VERSION = "wld-v5.5-whole-target-chromatin-twin-development"


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _artifact(path: Path, role: str, *, supplied_hash: Optional[str] = None) -> Dict[str, object]:
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"Missing or empty {role}: {path}")
    digest = supplied_hash or sha256_file(path)
    if len(digest) != 64:
        raise RuntimeError(f"Invalid SHA-256 for {role}")
    return {"role": role, "name": path.name, "bytes": path.stat().st_size, "sha256": digest}


def _verify_completed_condition_index(report: Mapping[str, object]) -> None:
    """Fail closed if a restored final report has missing or changed fits."""

    conditions = report.get("conditions", {})
    if not isinstance(conditions, Mapping) or not conditions:
        raise RuntimeError("Completed report has no fitted-condition index")
    config = report.get("config", {})
    expected_conditions = {"true_dual_routes"} | {
        f"degree_shuffle_{index + 1}"
        for index in range(int(config.get("shuffle_replicates", 0)))
    }
    expected_seeds = {str(int(seed)) for seed in config.get("seeds", ())}
    if set(conditions) != expected_conditions:
        raise RuntimeError("Completed report has a missing or unexpected fitted condition")
    for condition, seeds in conditions.items():
        if not isinstance(seeds, Mapping) or not seeds:
            raise RuntimeError(f"Completed condition has no seeds: {condition}")
        if set(seeds) != expected_seeds:
            raise RuntimeError(f"Completed condition has incomplete seeds: {condition}")
        for seed, record in seeds.items():
            if not isinstance(record, Mapping):
                raise RuntimeError(f"Malformed condition record: {condition}/{seed}")
            report_path = Path(str(record.get("report", "")))
            checkpoint_path = Path(str(record.get("checkpoint", "")))
            if not report_path.is_file() or not checkpoint_path.is_file():
                raise RuntimeError(f"Missing restored fit artifact: {condition}/{seed}")
            if sha256_file(report_path) != record.get("report_sha256"):
                raise RuntimeError(f"Changed condition report: {condition}/{seed}")
            checkpoint_hash = sha256_file(checkpoint_path)
            if checkpoint_hash != record.get("checkpoint_sha256"):
                raise RuntimeError(f"Changed condition checkpoint: {condition}/{seed}")
            condition_report = json.loads(report_path.read_text())
            if condition_report.get("checkpoint_sha256") != checkpoint_hash:
                raise RuntimeError(
                    f"Condition report/checkpoint disagreement: {condition}/{seed}"
                )
            blocks = report.get("validation_target_blocks")
            if not isinstance(blocks, Mapping):
                raise RuntimeError("Completed report lacks validation target blocks")
            _verify_condition_evaluation_blocks(
                condition_report, blocks, seed=int(seed)
            )


def _verify_training_only_module_manifest(
    manifest: Mapping[str, object],
) -> Dict[str, bool]:
    claims = manifest.get("claims", {})
    if not isinstance(claims, Mapping):
        raise RuntimeError("Complex-module manifest has malformed claims")
    flags = {
        "validation_values_used": manifest.get("validation_values_used"),
        "test_values_materialized": manifest.get("test_values_materialized"),
        "test_values_used": manifest.get("test_values_used"),
        "claim_validation_values_used": claims.get(
            "validation_values_used_for_module_construction"
        ),
        "claim_test_values_materialized": claims.get("test_values_materialized"),
        "claim_test_values_used": claims.get("test_values_used"),
    }
    if any(value is not False for value in flags.values()):
        raise RuntimeError(
            f"Complex-module manifest does not prove train-only construction: {flags}"
        )
    if tuple(manifest.get("construction_splits", ())) != ("train",):
        raise RuntimeError("Complex modules were not constructed exclusively from train")
    return {name: bool(value) for name, value in flags.items()}


def build_provenance_lock(
    prior_root: Path,
    foundation_checkpoint: Path,
    bundle_root: Path,
    route_root: Path,
    module_root: Path,
) -> Dict[str, object]:
    """Hash every source that can change the fitted model.

    The module NPZ and vocabulary are included explicitly rather than relying
    only on their manifest. The large response matrix is hashed directly and
    compared with any SHA-256 recorded by the v5.3 ingestion manifest.
    """

    prior_root, bundle_root = Path(prior_root), Path(bundle_root)
    route_root, module_root = Path(route_root), Path(module_root)
    foundation_checkpoint = Path(foundation_checkpoint)
    corpus_report_path = foundation_checkpoint.parent / "wld_corpus_pretraining_report.json"
    bundle_manifest_path = bundle_root / "wld_v53_ingestion_manifest.json"
    module_manifest_path = module_root / "complex_accessibility_module_manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    module_manifest = json.loads(module_manifest_path.read_text())
    module_flags = _verify_training_only_module_manifest(module_manifest)
    matrix = bundle_root / "atac_counts.GRCh38.2kb.npz"
    supplied_matrix_hash = (
        bundle_manifest.get("bundle", {}).get("matrix_sha256")
        or bundle_manifest.get("matrix_sha256")
        or bundle_manifest.get("artifacts", {}).get(matrix.name, {}).get("sha256")
    )
    current_matrix_hash = sha256_file(matrix)
    if supplied_matrix_hash and current_matrix_hash != supplied_matrix_hash:
        raise RuntimeError(
            "The current full-bin accessibility matrix does not match the "
            "SHA-256 recorded by the frozen v5.3 ingestion manifest"
        )
    source_root = Path(__file__).resolve().parent
    source_records = [
        _artifact(source_root / name, f"runtime source {name}")
        for name in (
            "wld_circuit_dynamics_v3.py",
            "wld_foundation_model_v4.py",
            "wld_foundation_data.py",
            "wld_phase_b_priors.py",
            "wld_chromatin_twin_v55.py",
            "wld_chromatin_modules_v55.py",
            "wld_twin_statistics_v55.py",
            "wld_chromatin_twin_training_v55.py",
        )
    ]
    records = source_records + [
        _artifact(prior_root / "prior_manifest.json", "foundation prior manifest"),
        _artifact(prior_root / "foundation_priors.npz", "foundation prior tensors"),
        _artifact(prior_root / "feature_vocab.json", "foundation vocabulary"),
        _artifact(foundation_checkpoint, "snapshot-pretrained foundation checkpoint"),
        _artifact(corpus_report_path, "expanded-corpus pretraining report"),
        _artifact(bundle_manifest_path, "v5.3 ingestion manifest"),
        _artifact(bundle_root / "whole_target_split.json", "whole-target split"),
        _artifact(matrix, "full-bin accessibility CSR", supplied_hash=current_matrix_hash),
        _artifact(bundle_root / "cells.tsv.gz", "cell metadata"),
        _artifact(bundle_root / "bins.GRCh38.2kb.tsv.gz", "response-bin vocabulary"),
        _artifact(route_root / "route_manifest.json", "TF-route manifest"),
        _artifact(route_root / "route_vocab.json", "TF-route vocabulary"),
        _artifact(route_root / "regulator_tf_routes.npz", "TF-route tensors"),
        _artifact(module_manifest_path, "complex-module manifest"),
        _artifact(module_root / "complex_accessibility_vocab.json", "complex-module vocabulary"),
        _artifact(module_root / "complex_accessibility_modules.npz", "complex-module tensors"),
    ]
    lock = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": records,
        "module_construction_targets": module_manifest.get("construction_targets", []),
        "module_leakage_flags": module_flags,
    }
    lock["digest"] = _canonical_digest(lock)
    return lock


@dataclass(frozen=True)
class TwinTrainingConfig:
    epochs: int = 28
    targets_per_epoch: int = 28
    batch_size: int = 48
    learning_rate: float = 2e-3
    representation_learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    integration_steps: int = 5
    horizon: float = 1.0
    projections: int = 32
    validation_cells_per_target: int = 128
    patience: int = 6
    seeds: Tuple[int, ...] = (42, 137, 911)
    shuffle_replicates: int = 2
    calibration_fraction: float = 0.5
    conformal_alpha: float = 0.2
    bootstrap_samples: int = 2000
    min_supported_training_targets: int = 5
    min_selection_targets: int = 2
    min_calibration_targets: int = 4
    min_audit_targets: int = 2

    def validate(self) -> None:
        integer_positive = (
            self.epochs,
            self.targets_per_epoch,
            self.batch_size,
            self.integration_steps,
            self.projections,
            self.validation_cells_per_target,
            self.patience,
            self.bootstrap_samples,
            self.min_supported_training_targets,
            self.min_selection_targets,
            self.min_calibration_targets,
            self.min_audit_targets,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or int(value) < 1
            for value in integer_positive
        ):
            raise ValueError("training counts must be positive integers")
        if self.bootstrap_samples < 100:
            raise ValueError("at least 100 target bootstrap samples are required")
        if (
            len(self.seeds) < 3
            or any(
                isinstance(seed, bool) or not isinstance(seed, numbers.Integral)
                or int(seed) < 0
                or int(seed) >= 2**32
                for seed in self.seeds
            )
            or len(set(map(int, self.seeds))) != len(self.seeds)
        ):
            raise ValueError("at least three unique integer seeds are required")
        if (
            isinstance(self.shuffle_replicates, bool)
            or not isinstance(self.shuffle_replicates, numbers.Integral)
            or self.shuffle_replicates < 1
        ):
            raise ValueError("at least one retrained topology shuffle is required")
        rates = (self.learning_rate, self.representation_learning_rate)
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in rates):
            raise ValueError("learning rates must be positive")
        if (
            not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0
            or not math.isfinite(float(self.horizon))
            or float(self.horizon) <= 0
        ):
            raise ValueError("invalid optimizer or integration configuration")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0,1)")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must be in (0,1)")
        conformal_minimum = int(math.ceil(1.0 / self.conformal_alpha - 1.0))
        if self.min_calibration_targets < conformal_minimum:
            raise ValueError(
                "min_calibration_targets is too small for the requested "
                "finite split-conformal alpha"
            )


def split_validation_targets(
    targets: Sequence[str],
    supported_targets: Iterable[str],
    *,
    seed: int,
    calibration_fraction: float,
    minimum_selection: int,
    minimum_calibration: int,
    minimum_audit: int,
) -> Dict[str, List[str]]:
    """Make deterministic whole-target selection/calibration/audit blocks.

    Only fixed route reachability and target names enter this split; no
    accessibility outcome is inspected.  Unsupported targets remain visible
    in the audit but cannot select checkpoints or calibrate predictions.
    """

    all_targets = sorted(set(map(str, targets)))
    supported_set = set(map(str, supported_targets))
    eligible = [target for target in all_targets if target in supported_set]
    needed = int(minimum_selection) + int(minimum_calibration) + int(minimum_audit)
    if len(eligible) < needed:
        raise RuntimeError(
            f"Only {len(eligible)} route-supported validation targets; {needed} are required"
        )
    ordered = sorted(
        eligible,
        key=lambda target: hashlib.sha256(f"{int(seed)}|{target}".encode()).hexdigest(),
    )
    minimum_heldout = int(minimum_calibration) + int(minimum_audit)
    heldout_total = max(
        minimum_heldout,
        int(round(len(ordered) * float(calibration_fraction))),
    )
    heldout_total = min(heldout_total, len(ordered) - int(minimum_selection))
    extra = heldout_total - minimum_heldout
    audit_count = int(minimum_audit) + (extra + 1) // 2
    calibration_count = int(minimum_calibration) + extra // 2
    calibration = sorted(ordered[:calibration_count])
    audit = sorted(ordered[calibration_count : calibration_count + audit_count])
    selection = sorted(ordered[calibration_count + audit_count :])
    if (
        set(selection) & set(calibration)
        or set(selection) & set(audit)
        or set(calibration) & set(audit)
    ):
        raise AssertionError("validation target blocks overlap")
    return {
        "selection": selection,
        "calibration": calibration,
        "audit": audit,
        "unsupported": sorted(set(all_targets) - set(eligible)),
    }


def _load_foundation(prior_root: Path, checkpoint: Path, device: torch.device) -> WLDMultistudyFoundationModel:
    priors = load_phase_b_priors(Path(prior_root), device)
    model = WLDMultistudyFoundationModel(priors, context_covariate_dim=0, context_dim=32).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    return model


def _device_priors(priors: ChromatinTwinPriors, device: torch.device) -> ChromatinTwinPriors:
    return ChromatinTwinPriors(
        **{field.name: torch.as_tensor(getattr(priors, field.name)).to(device) for field in fields(priors)}
    )


def load_twin_priors(
    prior_root: Path,
    route_root: Path,
    module_root: Path,
) -> Tuple[ChromatinTwinPriors, ComplexAccessibilityModuleAtlas, Dict[str, object]]:
    """Align the foundation motif anchors, TF routes and full-bin modules."""

    prior_root, route_root, module_root = map(Path, (prior_root, route_root, module_root))
    route_manifest = json.loads((route_root / "route_manifest.json").read_text())
    expected_route_hashes = route_manifest.get("artifact_sha256")
    route_artifacts = (
        "route_vocab.json",
        "regulator_tf_routes.npz",
        "regulator_tf_routes.tsv.gz",
    )
    if not isinstance(expected_route_hashes, Mapping) or set(expected_route_hashes) != set(
        route_artifacts
    ):
        raise RuntimeError("TF-route manifest lacks a complete artifact content lock")
    for name in route_artifacts:
        if sha256_file(route_root / name) != expected_route_hashes[name]:
            raise RuntimeError(f"TF-route artifact hash mismatch: {name}")
    route_vocab = json.loads((route_root / "route_vocab.json").read_text())
    feature_vocab = json.loads((prior_root / "feature_vocab.json").read_text())
    with np.load(route_root / "regulator_tf_routes.npz", allow_pickle=False) as values:
        route = np.asarray(values["regulator_tf_support"], dtype=np.float32)
    atlas = load_complex_module_atlas(module_root, verify_hashes=True)
    module_flags = _verify_training_only_module_manifest(atlas.provenance)
    regulators = tuple(str(value).upper() for value in route_vocab["regulators"])
    tfs = tuple(str(value).upper() for value in route_vocab["tfs"])
    if regulators != tuple(atlas.regulator_vocab):
        raise RuntimeError("TF routes and complex atlas disagree on regulator order")
    if tfs != tuple(str(value).upper() for value in feature_vocab["tfs"]):
        raise RuntimeError("TF routes and foundation priors disagree on TF order")
    if route.shape != (len(regulators), len(tfs)):
        raise RuntimeError("TF-route tensor and vocabulary disagree")

    foundation = load_phase_b_priors(prior_root, "cpu")
    anchor_motif = foundation.peak_tf_motif.transpose(0, 1).float()
    if anchor_motif.shape != (len(tfs), len(atlas.foundation_anchor_indices)):
        raise RuntimeError("foundation motif anchors and complex atlas disagree")
    foundation_peaks = tuple(
        map(str, feature_vocab.get("peaks", feature_vocab.get("atac", ())))
    )
    indexed_peaks = tuple(
        atlas.bins[int(index)] for index in atlas.foundation_anchor_indices
    )
    if indexed_peaks != foundation_peaks:
        raise RuntimeError(
            "foundation peak vocabulary does not exactly match the atlas anchor "
            "indices and ordering"
        )
    full_motif = torch.zeros((len(tfs), len(atlas.bins)), dtype=torch.float32)
    full_motif[:, torch.as_tensor(atlas.foundation_anchor_indices).long()] = anchor_motif.clamp_min(0)
    priors = ChromatinTwinPriors(
        regulator_tf_support=torch.as_tensor(route),
        tf_peak_motif=full_motif,
        regulator_complex_support=torch.as_tensor(atlas.regulator_complex_support.toarray()),
        complex_module_effect=torch.as_tensor(atlas.complex_module_effect.toarray()),
        module_peak_loading=torch.as_tensor(atlas.module_peak_loading.toarray()),
        foundation_peak_index=torch.as_tensor(atlas.foundation_anchor_indices).long(),
    )
    priors.validate()
    audit = {
        "regulators": list(regulators),
        "tfs": list(tfs),
        "complex_ids": list(atlas.complex_ids),
        "modules": list(atlas.module_vocab),
        "response_bins": len(atlas.bins),
        "foundation_anchor_bins": len(atlas.foundation_anchor_indices),
        "base_topology_sha256": topology_digest(priors),
        "module_construction_targets": list(atlas.construction_targets),
        "module_construction_splits": atlas.provenance.get("construction_splits", []),
        "module_leakage_flags": module_flags,
        "validation_values_used_for_modules": module_flags["validation_values_used"],
        "test_values_materialized": module_flags["test_values_materialized"],
    }
    return priors, atlas, audit


def _binary_dense(matrix: sparse.csr_matrix, rows: np.ndarray, device: torch.device) -> Tensor:
    selected = matrix[np.asarray(rows, dtype=np.int64)].astype(np.float32, copy=True).tocsr()
    selected.data.fill(1.0)
    selected.eliminate_zeros()
    return torch.as_tensor(selected.toarray(), dtype=torch.float32, device=device)


def _binary_mean(matrix: sparse.csr_matrix, rows: np.ndarray) -> np.ndarray:
    selected = matrix[np.asarray(rows, dtype=np.int64)].astype(np.float32, copy=True).tocsr()
    selected.data.fill(1.0)
    selected.eliminate_zeros()
    return np.asarray(selected.mean(axis=0), dtype=np.float32).ravel()


def _sample(rows: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if not len(rows):
        raise RuntimeError("cannot sample an empty population")
    return np.asarray(rng.choice(rows, size=int(count), replace=len(rows) < int(count)), dtype=np.int64)


def _projections(features: int, count: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    value = F.normalize(torch.randn(features, count, generator=generator), dim=0)
    return value.to(device)


def sliced_wasserstein(left: Tensor, right: Tensor, projections: Tensor) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("SWD populations must share a feature dimension")
    count = min(left.shape[0], right.shape[0])
    left_projected = torch.sort(left[:count] @ projections, dim=0).values
    right_projected = torch.sort(right[:count] @ projections, dim=0).values
    return (left_projected - right_projected).abs().mean()


def _intervention(target: str, regulator_index: Mapping[str, int], batch: int, device: torch.device) -> Tensor:
    value = torch.zeros((int(batch), len(regulator_index)), device=device)
    value[:, regulator_index[target]] = 1.0
    return value


def distribution_loss(
    prediction: Tensor,
    observed: Tensor,
    control: Tensor,
    projection: Tensor,
    observed_target_mean: Tensor,
    observed_control_mean: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    swd = sliced_wasserstein(prediction, observed, projection)
    mean_loss = F.mse_loss(prediction.mean(0), observed.mean(0))
    variance_loss = F.mse_loss(
        prediction.var(0, unbiased=False), observed.var(0, unbiased=False)
    )
    observed_response = observed_target_mean - observed_control_mean
    predicted_response = prediction.mean(0) - control.mean(0)
    response_rms = observed_response.square().mean().sqrt().clamp_min(2e-3)
    response_nrmse = (predicted_response - observed_response).square().mean().sqrt() / response_rms
    cosine_loss = 1.0 - F.cosine_similarity(
        predicted_response.unsqueeze(0), observed_response.unsqueeze(0), dim=1, eps=1e-8
    ).mean()
    total = swd + 2.0 * mean_loss + 0.25 * variance_loss + 0.05 * response_nrmse + 0.02 * cosine_loss
    return total, {
        "loss": float(total.detach()),
        "swd": float(swd.detach()),
        "mean_mse": float(mean_loss.detach()),
        "variance_mse": float(variance_loss.detach()),
        "response_nrmse": float(response_nrmse.detach()),
        "response_cosine_loss": float(cosine_loss.detach()),
    }


def _aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    numeric = (
        "model_swd",
        "persistence_swd",
        "gain_over_persistence",
        "response_cosine",
        "response_nrmse",
        "relative_swd",
        "selection_score",
        "observed_response_rms",
        "predicted_response_rms",
    )
    return {
        "targets": len(rows),
        **{
            name: (float(np.mean([float(row[name]) for row in rows])) if rows else None)
            for name in numeric
        },
    }


def evaluate_model(
    model: WLDChromatinDigitalTwin,
    bundle: SparseFullChromatinBundle,
    targets: Sequence[str],
    config: TwinTrainingConfig,
    *,
    seed: int,
    overrides: Optional[BranchOverrides] = None,
) -> Dict[str, object]:
    """Evaluate named whole targets with fixed cells/projections for pairing."""

    device = next(model.parameters()).device
    regulator_index = {value: index for index, value in enumerate(model_regulators(model, bundle))}
    projection = _projections(len(bundle.bins), config.projections, seed + 31, device)
    reachability = model.field.reachability()
    rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for number, target in enumerate(sorted(set(map(str, targets)))):
            if target not in regulator_index:
                raise RuntimeError(f"Unknown perturbation target {target}")
            screen_rows: List[Tuple[int, Dict[str, float]]] = []
            for screen_number, screen in enumerate(bundle.target_screens("validation", target)):
                target_rows = bundle.rows("validation", screen, target)
                control_rows = bundle.rows("validation", screen, "NTC")
                if not len(control_rows):
                    raise RuntimeError(f"No validation NTC for {target}/{screen}")
                n = min(config.validation_cells_per_target, len(target_rows), len(control_rows))
                rng = np.random.default_rng(seed + 1009 * (number + 1) + 7919 * screen_number)
                observed = _binary_dense(bundle.accessibility, _sample(target_rows, n, rng), device)
                control = _binary_dense(bundle.accessibility, _sample(control_rows, n, rng), device)
                prediction = model(
                    control,
                    _intervention(target, regulator_index, n, device),
                    horizon=config.horizon,
                    steps=config.integration_steps,
                    overrides=overrides,
                )["atac_t"]
                model_swd = float(sliced_wasserstein(prediction, observed, projection))
                persistence_swd = float(sliced_wasserstein(control, observed, projection))
                observed_response = observed.mean(0) - control.mean(0)
                predicted_response = prediction.mean(0) - control.mean(0)
                raw_response_rms = float(observed_response.square().mean().sqrt())
                response_scale = max(raw_response_rms, 2e-3)
                response_nrmse = float(
                    (predicted_response - observed_response).square().mean().sqrt()
                    / response_scale
                )
                cosine = float(F.cosine_similarity(
                    predicted_response.unsqueeze(0), observed_response.unsqueeze(0), dim=1, eps=1e-8
                )[0])
                relative = model_swd / max(persistence_swd, 1e-8)
                screen_rows.append((n, {
                    "model_swd": model_swd,
                    "persistence_swd": persistence_swd,
                    "gain_over_persistence": persistence_swd - model_swd,
                    "response_cosine": cosine,
                    "response_nrmse": response_nrmse,
                    "relative_swd": relative,
                    "selection_score": 0.70 * relative + 0.30 * response_nrmse,
                    "observed_response_rms": raw_response_rms,
                    "response_nrmse_denominator": response_scale,
                    "response_nrmse_floor_used": float(raw_response_rms < 2e-3),
                    "predicted_response_rms": float(predicted_response.square().mean().sqrt()),
                    "mean_absolute_predicted_change": float(predicted_response.abs().mean()),
                }))
            if not screen_rows:
                raise RuntimeError(f"No validation observations for target {target}")
            weights = np.asarray([count for count, _ in screen_rows], dtype=np.float64)
            weights /= weights.sum()
            keys = tuple(screen_rows[0][1])
            metrics = {
                key: float(sum(weight * values[key] for weight, (_count, values) in zip(weights, screen_rows)))
                for key in keys
            }
            index = regulator_index[target]
            tf_bins = int(torch.count_nonzero(reachability["tf"][index]))
            complex_bins = int(torch.count_nonzero(reachability["complex"][index]))
            rows.append({
                "target": target,
                "screens": bundle.target_screens("validation", target),
                "cells": int(sum(count for count, _ in screen_rows)),
                "tf_reachable_bins": tf_bins,
                "complex_reachable_bins": complex_bins,
                "route_supported": bool(tf_bins or complex_bins),
                **metrics,
            })
    supported = [row for row in rows if row["route_supported"]]
    return {
        "per_target": rows,
        "all_targets": _aggregate(rows),
        "route_supported_targets": _aggregate(supported),
        "unsupported_targets": sorted(row["target"] for row in rows if not row["route_supported"]),
        "evaluation_seed": int(seed),
        "test_values_materialized": False,
    }


def model_regulators(model: WLDChromatinDigitalTwin, bundle: SparseFullChromatinBundle) -> Tuple[str, ...]:
    regulators = tuple(bundle.provenance.get("regulator_vocab", ()))
    if regulators:
        if len(regulators) != model.field.num_regulators:
            raise RuntimeError("bundle regulator vocabulary and model disagree")
        return regulators
    # The v5.3 bundle does not store regulators because they are an external
    # whole-target vocabulary.  run_twin_development installs the immutable
    # aligned vocabulary here without placing it in encoder tensors.
    regulators = getattr(bundle, "regulator_vocab", ())
    if len(regulators) != model.field.num_regulators:
        raise RuntimeError("missing aligned regulator vocabulary")
    return tuple(regulators)


def _condition_priors(
    base: ChromatinTwinPriors,
    *,
    regulator_tf_support: Tensor,
    regulator_complex_support: Tensor,
) -> ChromatinTwinPriors:
    return ChromatinTwinPriors(
        regulator_tf_support=regulator_tf_support,
        tf_peak_motif=base.tf_peak_motif,
        regulator_complex_support=regulator_complex_support,
        complex_module_effect=base.complex_module_effect,
        module_peak_loading=base.module_peak_loading,
        foundation_peak_index=base.foundation_peak_index,
    )


def _end_to_end_regulator_reachability(priors: ChromatinTwinPriors) -> Tensor:
    """Return regulators that can reach at least one response bin."""

    tf_bins = (priors.regulator_tf_support > 0).float() @ (
        priors.tf_peak_motif > 0
    ).float()
    complex_modules = (priors.regulator_complex_support > 0).float() @ (
        priors.complex_module_effect != 0
    ).float()
    complex_bins = (complex_modules > 0).float() @ (
        priors.module_peak_loading != 0
    ).float()
    return ((tf_bins > 0) | (complex_bins > 0)).any(dim=1)


def _verify_condition_evaluation_blocks(
    report: Mapping[str, object],
    validation_blocks: Mapping[str, Sequence[str]],
    *,
    seed: int,
) -> None:
    expected_seed = int(seed) + 900_001
    for block in ("selection", "calibration", "audit"):
        evaluation = report.get(f"final_{block}")
        if not isinstance(evaluation, Mapping):
            raise RuntimeError(f"Condition report lacks final {block} evaluation")
        rows = evaluation.get("per_target", ())
        targets = [str(row.get("target")) for row in rows]
        if len(targets) != len(set(targets)) or set(targets) != set(
            map(str, validation_blocks[block])
        ):
            raise RuntimeError(f"Condition report has a changed {block} target roster")
        if int(evaluation.get("evaluation_seed", -1)) != expected_seed:
            raise RuntimeError(f"Condition report has a changed {block} evaluation seed")


def _support_topology_digest(priors: ChromatinTwinPriors) -> str:
    digest = hashlib.sha256()
    for name in ("regulator_tf_support", "regulator_complex_support"):
        value = (
            torch.as_tensor(getattr(priors, name))
            .detach()
            .cpu()
            .ne(0)
            .contiguous()
            .numpy()
        )
        digest.update(name.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _fit_condition(
    condition: str,
    seed: int,
    priors: ChromatinTwinPriors,
    prior_root: Path,
    checkpoint: Path,
    bundle: SparseFullChromatinBundle,
    validation_blocks: Mapping[str, Sequence[str]],
    output_root: Path,
    config: TwinTrainingConfig,
    provenance_digest: str,
    device: torch.device,
) -> Tuple[WLDChromatinDigitalTwin, Dict[str, object]]:
    root = Path(output_root) / condition / f"seed_{int(seed)}"
    root.mkdir(parents=True, exist_ok=True)
    report_path, model_path = root / "condition_report.json", root / "best_model.pt"
    config_value = asdict(config)
    prior_digest = topology_digest(priors)

    def new_model() -> WLDChromatinDigitalTwin:
        foundation = _load_foundation(prior_root, checkpoint, device)
        return WLDChromatinDigitalTwin(foundation, _device_priors(priors, device)).to(device)

    if report_path.is_file() and model_path.is_file():
        report = json.loads(report_path.read_text())
        checks = (
            report.get("provenance_digest") == provenance_digest,
            report.get("topology_sha256") == prior_digest,
            report.get("config") == json.loads(json.dumps(config_value)),
            int(report.get("seed", -1)) == int(seed),
            report.get("checkpoint_sha256") == sha256_file(model_path),
            report.get("claims", {}).get("test_targets_evaluated") is False,
            report.get("claims", {}).get("digital_twin_claim") is False,
            report.get("claims", {}).get("attractor_claim") is False,
        )
        if not all(checks):
            raise RuntimeError(f"Completed {condition}/seed {seed} does not match this locked run")
        _verify_condition_evaluation_blocks(
            report, validation_blocks, seed=int(seed)
        )
        model = new_model()
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"PASS: restored {condition} seed {seed}", flush=True)
        return model, report

    torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = new_model()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.field.parameters(), "lr": config.learning_rate},
            {"params": model.foundation.encoder.parameters(), "lr": config.representation_learning_rate},
            {"params": model.foundation.context_network.parameters(), "lr": config.representation_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    regulator_index = {value: index for index, value in enumerate(model_regulators(model, bundle))}
    reachable = model.field.reachability()["total"].any(dim=1).detach().cpu().numpy()
    training_targets = [
        target for target in bundle.split_targets("train")
        if target in regulator_index and bool(reachable[regulator_index[target]])
    ]
    if len(training_targets) < config.min_supported_training_targets:
        raise RuntimeError(f"Only {len(training_targets)} supported training targets for {condition}")
    selection = list(validation_blocks["selection"])
    projection = _projections(len(bundle.bins), config.projections, int(seed) + 17, device)
    state_path = root / "training_state.pt"
    start_epoch, best_score, best_state, waiting = 0, float("inf"), None, 0
    history: List[Dict[str, object]] = []
    if state_path.is_file():
        try:
            state = torch.load(state_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location=device)
        lock_checks = (
            state.get("provenance_digest") == provenance_digest,
            state.get("topology_sha256") == prior_digest,
            state.get("config") == config_value,
            state.get("condition") == condition,
            int(state.get("seed", -1)) == int(seed),
        )
        if not all(lock_checks):
            raise RuntimeError(f"Resume state for {condition}/seed {seed} changed inputs")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_score, best_state = float(state["best_score"]), state["best_state"]
        if best_state is not None:
            best_state = {
                name: value.detach().cpu() for name, value in best_state.items()
            }
        waiting, history = int(state["waiting"]), list(state["history"])
        print(f"   resumed {condition} seed {seed} at epoch {start_epoch}", flush=True)

    mean_cache: Dict[Tuple[str, str, str], Tensor] = {}
    for epoch in range(start_epoch, config.epochs):
        model.train()
        rng = np.random.default_rng(int(seed) + 100_003 * epoch)
        selected = rng.choice(
            training_targets,
            size=config.targets_per_epoch,
            replace=len(training_targets) < config.targets_per_epoch,
        )
        losses = []
        for target_value in selected:
            target = str(target_value)
            screens = bundle.target_screens("train", target)
            screen = screens[int(rng.integers(0, len(screens)))]
            target_rows = bundle.rows("train", screen, target)
            control_rows = bundle.rows("train", screen, "NTC")
            if not len(control_rows):
                raise RuntimeError(f"No train NTC for {target}/{screen}")
            observed = _binary_dense(bundle.accessibility, _sample(target_rows, config.batch_size, rng), device)
            control = _binary_dense(bundle.accessibility, _sample(control_rows, config.batch_size, rng), device)
            target_key, control_key = ("train", screen, target), ("train", screen, "NTC")
            if target_key not in mean_cache:
                mean_cache[target_key] = torch.as_tensor(_binary_mean(bundle.accessibility, target_rows), device=device)
            if control_key not in mean_cache:
                mean_cache[control_key] = torch.as_tensor(_binary_mean(bundle.accessibility, control_rows), device=device)
            prediction = model(
                control,
                _intervention(target, regulator_index, config.batch_size, device),
                horizon=config.horizon,
                steps=config.integration_steps,
            )["atac_t"]
            loss, _ = distribution_loss(
                prediction, observed, control, projection,
                mean_cache[target_key], mean_cache[control_key],
            )
            regularization = 1e-6 * sum(
                parameter.square().mean()
                for name, parameter in model.field.named_parameters()
                if name.startswith("raw_")
            )
            total = loss + regularization
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            parameters = (
                list(model.field.parameters())
                + list(model.foundation.encoder.parameters())
                + list(model.foundation.context_network.parameters())
            )
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation = evaluate_model(
            model, bundle, selection, config, seed=int(seed) + 500_001
        )
        score_value = validation["route_supported_targets"]["selection_score"]
        if score_value is None or not math.isfinite(float(score_value)):
            raise RuntimeError("No route-supported selection target")
        score = float(score_value)
        improved = score < best_score - 1e-6
        if improved:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1
        history.append({
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "selection": validation["route_supported_targets"],
            "improved": improved,
        })
        print(
            f"   {condition} seed {seed} epoch {epoch:03d} | "
            f"train {np.mean(losses):.6f} | selection {score:.6f}",
            flush=True,
        )
        temporary = state_path.with_suffix(".pt.tmp")
        torch.save({
            "schema_version": SCHEMA_VERSION,
            "condition": condition,
            "seed": int(seed),
            "epoch": epoch,
            "config": config_value,
            "provenance_digest": provenance_digest,
            "topology_sha256": prior_digest,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
            "best_state": best_state,
            "waiting": waiting,
            "history": history,
        }, temporary)
        os.replace(temporary, state_path)
        if waiting >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No validation-selected checkpoint for {condition}/seed {seed}")
    model.load_state_dict(best_state)
    final_selection = evaluate_model(
        model, bundle, validation_blocks["selection"], config, seed=int(seed) + 900_001
    )
    final_calibration = evaluate_model(
        model, bundle, validation_blocks["calibration"], config, seed=int(seed) + 900_001
    )
    final_audit = evaluate_model(
        model, bundle, validation_blocks["audit"], config, seed=int(seed) + 900_001
    )
    temporary_model = model_path.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), temporary_model)
    os.replace(temporary_model, model_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "condition": condition,
        "seed": int(seed),
        "config": config_value,
        "provenance_digest": provenance_digest,
        "topology_sha256": prior_digest,
        "best_selection_score": best_score,
        "epochs_attempted": len(history),
        "history": history,
        "final_selection": final_selection,
        "final_calibration": final_calibration,
        "final_audit": final_audit,
        "checkpoint_sha256": sha256_file(model_path),
        "claims": {
            "test_targets_evaluated": False,
            "external_subject_study_evaluated": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
        },
    }
    _verify_condition_evaluation_blocks(report, validation_blocks, seed=int(seed))
    atomic_json(report_path, report)
    return model, report


def _rows(report: Mapping[str, object], block: str) -> List[Mapping[str, object]]:
    return list(report[f"final_{block}"]["per_target"])


def _by_target(rows: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    result = {str(row["target"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate target metrics")
    return result


def _interval_dict(value, *, insufficient_targets: int = 0) -> Dict[str, object]:
    if value is None:
        return {
            "supported": False,
            "targets": int(insufficient_targets),
            "reason": "fewer than two route-reachable untouched audit targets",
        }
    return {"supported": True, **asdict(value)}


def _ensemble_scalar_rows(
    reports: Mapping[int, Mapping[str, object]], block: str
) -> List[Dict[str, object]]:
    seed_targets = {
        int(seed): {str(row["target"]) for row in _rows(report, block)}
        for seed, report in reports.items()
    }
    target_grids = list(seed_targets.values())
    if not target_grids or any(grid != target_grids[0] for grid in target_grids[1:]):
        raise RuntimeError(f"Ragged seed/target ensemble grid for {block}")
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for seed, report in reports.items():
        for row in _rows(report, block):
            grouped.setdefault(str(row["target"]), []).append({**row, "seed": int(seed)})
    output = []
    for target, rows in sorted(grouped.items()):
        observed_seeds = {int(row["seed"]) for row in rows}
        if observed_seeds != set(map(int, reports)) or len(rows) != len(reports):
            raise RuntimeError(f"Incomplete seed ensemble for target {target}")
        values = np.asarray([float(row["predicted_response_rms"]) for row in rows])
        observed = float(np.mean([float(row["observed_response_rms"]) for row in rows]))
        output.append({
            "target": target,
            "observed": observed,
            "ensemble_mean": float(values.mean()),
            "ensemble_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "seeds": len(values),
        })
    return output


def _paired_rows(
    true_reports: Mapping[int, Mapping[str, object]],
    frozen_reports: Mapping[
        int, Mapping[str, Mapping[str, Mapping[str, object]]]
    ],
    shuffle_reports: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    block: str,
) -> List[Dict[str, object]]:
    paired: List[Dict[str, object]] = []
    for seed, report in sorted(true_reports.items()):
        true = _by_target(_rows(report, block))
        frozen_all = _by_target(
            frozen_reports[seed]["all_routes_removed"][block]["per_target"]
        )
        frozen_tf = _by_target(
            frozen_reports[seed]["tf_removed"][block]["per_target"]
        )
        frozen_complex = _by_target(
            frozen_reports[seed]["complex_removed"][block]["per_target"]
        )
        shuffle_maps = [_by_target(_rows(value, block)) for value in shuffle_reports[seed]]
        expected_targets = set(true)
        compared = [frozen_all, frozen_tf, frozen_complex, *shuffle_maps]
        if any(set(mapping) != expected_targets for mapping in compared):
            raise RuntimeError(f"Ragged frozen/control target grid for seed {seed}")
        for target in sorted(expected_targets):
            if not bool(true[target]["route_supported"]):
                continue
            paired.append({
                "target": target,
                "seed": int(seed),
                "true_loss": float(true[target]["model_swd"]),
                "persistence_loss": float(true[target]["persistence_swd"]),
                "shuffle_loss": float(np.mean([mapping[target]["model_swd"] for mapping in shuffle_maps])),
                "frozen_all_loss": float(frozen_all[target]["model_swd"]),
                "frozen_tf_loss": float(frozen_tf[target]["model_swd"]),
                "frozen_complex_loss": float(frozen_complex[target]["model_swd"]),
                "response_nrmse": float(true[target]["response_nrmse"]),
                "response_cosine": float(true[target]["response_cosine"]),
                "tf_reachable_bins": int(true[target]["tf_reachable_bins"]),
                "complex_reachable_bins": int(true[target]["complex_reachable_bins"]),
            })
    return paired


def run_twin_development(
    prior_root: Path,
    foundation_checkpoint: Path,
    bundle_root: Path,
    route_root: Path,
    module_root: Path,
    output_root: Path,
    config: TwinTrainingConfig,
    *,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Run locked multi-seed development without opening the test partition."""

    config.validate()
    prior_root, foundation_checkpoint, bundle_root = map(Path, (prior_root, foundation_checkpoint, bundle_root))
    route_root, module_root, output_root = map(Path, (route_root, module_root, output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    verify_phase_b_priors(prior_root)
    corpus_report_path = foundation_checkpoint.parent / "wld_corpus_pretraining_report.json"
    if not corpus_report_path.is_file():
        raise FileNotFoundError(corpus_report_path)
    corpus_report = json.loads(corpus_report_path.read_text())
    if corpus_report.get("checkpoint_sha256") != sha256_file(foundation_checkpoint):
        raise RuntimeError(
            "Expanded-corpus report/checkpoint lineage does not match"
        )
    sealed_flags = {
        name: corpus_report.get(name)
        for name in (
            "sealed_test_downloaded",
            "sealed_test_evaluated",
            "model_assessed_on_sealed_test",
        )
    }
    if any(value is not False for value in sealed_flags.values()):
        raise RuntimeError(
            f"Expanded-corpus checkpoint crossed its sealed boundary: {sealed_flags}"
        )
    provenance = build_provenance_lock(
        prior_root, foundation_checkpoint, bundle_root, route_root, module_root
    )
    final_path = output_root / "wld_v55_chromatin_twin_report.json"
    config_json = json.loads(json.dumps(asdict(config)))
    if final_path.is_file():
        existing = json.loads(final_path.read_text())
        checks = (
            existing.get("provenance", {}).get("digest") == provenance["digest"],
            existing.get("config") == config_json,
            existing.get("claims", {}).get("test_targets_evaluated") is False,
            existing.get("claim_evaluation", {}).get("digital_twin_claim") is False,
            existing.get("claim_evaluation", {}).get("attractor_claim") is False,
        )
        if all(checks):
            _verify_completed_condition_index(existing)
            print("PASS: restored completed WLD v5.5 development", flush=True)
            return existing
        raise RuntimeError("Existing final report does not match the locked inputs/configuration")

    base_priors, atlas, prior_audit = load_twin_priors(prior_root, route_root, module_root)
    bundle = load_v53_sparse_full_bundle(bundle_root, prior_root=prior_root)
    if tuple(bundle.bins) != tuple(atlas.bins):
        raise RuntimeError("runtime response bins differ from the frozen module atlas")
    route_vocab = json.loads((route_root / "route_vocab.json").read_text())
    regulators = tuple(str(value).upper() for value in route_vocab["regulators"])
    split_contract = json.loads((bundle_root / "whole_target_split.json").read_text())
    frozen_rosters = {
        split: tuple(str(value).upper() for value in split_contract["targets"][split])
        for split in ("train", "validation", "test")
    }
    frozen_regulators = set().union(*map(set, frozen_rosters.values()))
    if len(regulators) != len(set(regulators)) or set(regulators) != frozen_regulators:
        raise RuntimeError(
            "TF-route regulator vocabulary does not exactly match the frozen "
            "whole-target roster"
        )
    if set(bundle.split_targets("train")) != set(frozen_rosters["train"]):
        raise RuntimeError("Materialized train targets do not match the frozen roster")
    if set(bundle.split_targets("validation")) != set(frozen_rosters["validation"]):
        raise RuntimeError("Materialized validation targets do not match the frozen roster")
    if set(atlas.construction_targets) != set(frozen_rosters["train"]):
        raise RuntimeError(
            "Complex modules were not constructed from the exact frozen train roster"
        )
    module_bundle = atlas.provenance.get("bundle", {})
    for key in (
        "v53_manifest_sha256",
        "whole_target_split_sha256",
        "whole_target_roster_sha256",
        "v53_matrix_sha256",
        "v53_cells_sha256",
        "v53_bins_sha256",
    ):
        if module_bundle.get(key) != bundle.provenance.get(key):
            raise RuntimeError(f"Complex modules and runtime bundle disagree on {key}")
    # Dataclass deliberately has no encoder-visible label field.  This dynamic
    # audit attribute is used only to map named interventions after encoding.
    setattr(bundle, "regulator_vocab", regulators)
    bundle.provenance["regulator_vocab"] = regulators
    if any(split.lower() == "test" for split in bundle.splits):
        raise RuntimeError("sealed test rows were materialized")

    total_reach = _end_to_end_regulator_reachability(base_priors).cpu().numpy()
    supported = [regulators[index] for index, value in enumerate(total_reach) if bool(value)]
    blocks = split_validation_targets(
        bundle.split_targets("validation"),
        supported,
        seed=int(config.seeds[0]) + 77_771,
        calibration_fraction=config.calibration_fraction,
        minimum_selection=config.min_selection_targets,
        minimum_calibration=config.min_calibration_targets,
        minimum_audit=config.min_audit_targets,
    )
    atomic_json(output_root / "validation_target_blocks.json", {
        "schema_version": SCHEMA_VERSION,
        "construction": "hash-ordered names and fixed topology only; no outcomes",
        **blocks,
        "disjoint": not bool(
            (set(blocks["selection"]) & set(blocks["calibration"]))
            | (set(blocks["selection"]) & set(blocks["audit"]))
            | (set(blocks["calibration"]) & set(blocks["audit"]))
        ),
        "test_targets_included": False,
    })

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    true_reports: Dict[int, Mapping[str, object]] = {}
    frozen_reports: Dict[
        int, Mapping[str, Mapping[str, Mapping[str, object]]]
    ] = {}
    shuffle_reports: Dict[int, List[Mapping[str, object]]] = {int(seed): [] for seed in config.seeds}
    condition_index: Dict[str, object] = {}

    shuffled_priors: List[ChromatinTwinPriors] = []
    seen_topologies = {_support_topology_digest(base_priors)}
    for replicate in range(config.shuffle_replicates):
        candidate = None
        for attempt in range(32):
            shuffle_seed = (
                int(config.seeds[0]) + 10_000 + replicate + 104_729 * attempt
            )
            try:
                proposed = _condition_priors(
                    base_priors,
                    regulator_tf_support=degree_preserving_bipartite_shuffle(
                        base_priors.regulator_tf_support, seed=shuffle_seed
                    ),
                    regulator_complex_support=degree_preserving_bipartite_shuffle(
                        base_priors.regulator_complex_support,
                        seed=shuffle_seed + 1_000_003,
                    ),
                )
            except RuntimeError:
                continue
            digest = _support_topology_digest(proposed)
            if digest not in seen_topologies:
                candidate = proposed
                seen_topologies.add(digest)
                break
        if candidate is None:
            raise RuntimeError(
                f"Could not create distinct degree-preserving control {replicate + 1}"
            )
        shuffled_priors.append(candidate)

    for seed_value in config.seeds:
        seed = int(seed_value)
        true_model, true_report = _fit_condition(
            "true_dual_routes", seed, base_priors, prior_root, foundation_checkpoint,
            bundle, blocks, output_root, config, provenance["digest"], resolved_device,
        )
        true_reports[seed] = true_report
        frozen_for_seed: Dict[str, Mapping[str, object]] = {}
        for label, overrides in (
            ("tf_removed", BranchOverrides(tf_scale=0.0)),
            ("complex_removed", BranchOverrides(complex_scale=0.0)),
            ("all_routes_removed", BranchOverrides(tf_scale=0.0, complex_scale=0.0)),
        ):
            frozen_for_seed[label] = {
                "selection": evaluate_model(
                    true_model, bundle, blocks["selection"], config,
                    seed=seed + 900_001, overrides=overrides,
                ),
                "calibration": evaluate_model(
                    true_model, bundle, blocks["calibration"], config,
                    seed=seed + 900_001, overrides=overrides,
                ),
                "audit": evaluate_model(
                    true_model, bundle, blocks["audit"], config,
                    seed=seed + 900_001, overrides=overrides,
                ),
            }
        frozen_reports[seed] = frozen_for_seed
        true_report_path = (
            output_root / "true_dual_routes" / f"seed_{seed}" / "condition_report.json"
        )
        condition_index.setdefault("true_dual_routes", {})[str(seed)] = {
            "report": str(true_report_path),
            "report_sha256": sha256_file(true_report_path),
            "checkpoint": str(output_root / "true_dual_routes" / f"seed_{seed}" / "best_model.pt"),
            "checkpoint_sha256": true_report["checkpoint_sha256"],
            "topology_sha256": topology_digest(base_priors),
            "support_topology_sha256": _support_topology_digest(base_priors),
        }
        del true_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for replicate, shuffled in enumerate(shuffled_priors):
            name = f"degree_shuffle_{replicate + 1}"
            _model, report = _fit_condition(
                name, seed, shuffled, prior_root, foundation_checkpoint, bundle,
                blocks, output_root, config, provenance["digest"], resolved_device,
            )
            shuffle_reports[seed].append(report)
            shuffle_report_path = output_root / name / f"seed_{seed}" / "condition_report.json"
            condition_index.setdefault(name, {})[str(seed)] = {
                "report": str(shuffle_report_path),
                "report_sha256": sha256_file(shuffle_report_path),
                "checkpoint": str(output_root / name / f"seed_{seed}" / "best_model.pt"),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "topology_sha256": topology_digest(shuffled),
                "support_topology_sha256": _support_topology_digest(shuffled),
            }
            del _model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The paired inferential analysis is confined to the untouched audit block.
    paired = _paired_rows(
        true_reports, frozen_reports, shuffle_reports, block="audit"
    )
    expected_paired_grid = {
        (str(target), int(seed))
        for target in blocks["audit"]
        for seed in config.seeds
    }
    observed_paired_grid = {
        (str(row["target"]), int(row["seed"])) for row in paired
    }
    if observed_paired_grid != expected_paired_grid or len(paired) != len(
        expected_paired_grid
    ):
        raise RuntimeError("Primary audit metrics have a ragged target/seed grid")
    persistence = paired_target_bootstrap(
        paired, true_key="true_loss", comparator_key="persistence_loss",
        samples=config.bootstrap_samples, random_seed=int(config.seeds[0]) + 301,
    )
    topology_shuffle = paired_target_bootstrap(
        paired, true_key="true_loss", comparator_key="shuffle_loss",
        samples=config.bootstrap_samples, random_seed=int(config.seeds[0]) + 302,
    )
    frozen_all = paired_target_bootstrap(
        paired, true_key="true_loss", comparator_key="frozen_all_loss",
        samples=config.bootstrap_samples, random_seed=int(config.seeds[0]) + 303,
    )
    tf_paired = [row for row in paired if int(row["tf_reachable_bins"]) > 0]
    complex_paired = [
        row for row in paired if int(row["complex_reachable_bins"]) > 0
    ]
    frozen_tf = (
        paired_target_bootstrap(
            tf_paired, true_key="true_loss", comparator_key="frozen_tf_loss",
            samples=config.bootstrap_samples, random_seed=int(config.seeds[0]) + 304,
        )
        if len({str(row["target"]) for row in tf_paired}) >= 2
        else None
    )
    frozen_complex = (
        paired_target_bootstrap(
            complex_paired, true_key="true_loss", comparator_key="frozen_complex_loss",
            samples=config.bootstrap_samples, random_seed=int(config.seeds[0]) + 305,
        )
        if len({str(row["target"]) for row in complex_paired}) >= 2
        else None
    )

    calibration_scalars = _ensemble_scalar_rows(true_reports, "calibration")
    audit_scalars = _ensemble_scalar_rows(true_reports, "audit")
    intervals = calibrate_ensemble_intervals(
        calibration_scalars, audit_scalars, alpha=config.conformal_alpha
    )
    observed_audit = {row["target"]: float(row["observed"]) for row in audit_scalars}
    covered = [
        float(row["lower"] <= observed_audit[row["target"]] <= row["upper"])
        for row in intervals
    ]
    normalized_widths = [
        float(row["interval_width"]) / max(observed_audit[row["target"]], 2e-3)
        for row in intervals
    ]
    calibrated_coverage = float(np.mean(covered)) if covered else 0.0
    normalized_width = float(np.mean(normalized_widths)) if normalized_widths else float("inf")
    response_nrmse = target_bootstrap_mean(
        paired,
        value_key="response_nrmse",
        samples=config.bootstrap_samples,
        random_seed=int(config.seeds[0]) + 306,
    )
    response_cosine = target_bootstrap_mean(
        paired,
        value_key="response_cosine",
        samples=config.bootstrap_samples,
        random_seed=int(config.seeds[0]) + 307,
    )
    claim_evaluation = evaluate_claims(
        persistence=persistence,
        topology_shuffle=topology_shuffle,
        frozen_removal=frozen_all,
        frozen_tf_removal=frozen_tf,
        frozen_complex_removal=frozen_complex,
        response_nrmse=response_nrmse,
        response_cosine=response_cosine,
        calibrated_coverage=calibrated_coverage,
        normalized_interval_width=normalized_width,
        external_subject_study_test=False,
        prospective_update_loop=False,
        longitudinal_return_data=False,
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "GSE161002 unpaired transient CRISPR-sciATAC development with complete response bins, "
            "whole held-out targets and a mechanistic chromatin digital-model prototype"
        ),
        "device": str(resolved_device),
        "config": config_json,
        "provenance": provenance,
        "prior_audit": prior_audit,
        "validation_target_blocks": blocks,
        "selection_calibration_audit_contract": {
            "selection_targets_choose_checkpoints": True,
            "calibration_targets_choose_checkpoints": False,
            "audit_targets_choose_checkpoints": False,
            "calibration_targets_fit_uncertainty_width": True,
            "audit_targets_measure_coverage_and_drive_primary_target_bootstrap": True,
            "target_blocks_disjoint": True,
            "test_targets_materialized": False,
        },
        "architecture": architecture_contract(
            WLDChromatinDigitalTwin(
                _load_foundation(prior_root, foundation_checkpoint, resolved_device),
                _device_priors(base_priors, resolved_device),
            )
        ),
        "conditions": condition_index,
        "paired_audit_target_seed_metrics": paired,
        "target_level_intervals": {
            "persistence_minus_true": _interval_dict(persistence),
            "shuffle_minus_true": _interval_dict(topology_shuffle),
            "frozen_all_routes_removed_minus_true": _interval_dict(frozen_all),
            "frozen_tf_removed_minus_true": _interval_dict(
                frozen_tf,
                insufficient_targets=len({str(row["target"]) for row in tf_paired}),
            ),
            "frozen_complex_removed_minus_true": _interval_dict(
                frozen_complex,
                insufficient_targets=len(
                    {str(row["target"]) for row in complex_paired}
                ),
            ),
            "response_nrmse": _interval_dict(response_nrmse),
            "response_cosine": _interval_dict(response_cosine),
        },
        "uncertainty": {
            "quantity": "target pseudobulk response RMS across complete response bins",
            "calibration_targets": calibration_scalars,
            "audit_target_intervals": intervals,
            "empirical_audit_coverage": calibrated_coverage,
            "mean_normalized_interval_width": normalized_width,
            "alpha": config.conformal_alpha,
            "development_interval_not_clinical_probability": True,
        },
        "claim_evaluation": claim_evaluation,
        "claims": {
            "unpaired_population_training": True,
            "complete_response_bin_state": True,
            "whole_target_model_selection": True,
            "whole_target_uncertainty_calibration": True,
            "target_identity_in_encoder": False,
            "test_targets_evaluated": False,
            "muscle_J_L_evaluated": False,
            "external_sealed_studies_evaluated": False,
            "ode_time_scale_identified": False,
            "fixed_point_claim": False,
            "basin_claim": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
        },
        "interpretation_rule": (
            "Transient mechanistic response requires target-bootstrap confidence bounds above zero "
            "against persistence, retrained degree-preserving topology shuffles and frozen route "
            "removal, plus prespecified response error/cosine thresholds. Regardless of this result, "
            "digital-twin and attractor claims remain false without prospective repeated-subject "
            "synchronization and longitudinal perturb-and-return observations."
        ),
    }
    atomic_json(final_path, report)
    return report


run_chromatin_twin_development = run_twin_development


__all__ = [
    "TwinTrainingConfig",
    "build_provenance_lock",
    "split_validation_targets",
    "load_twin_priors",
    "sliced_wasserstein",
    "distribution_loss",
    "evaluate_model",
    "run_twin_development",
    "run_chromatin_twin_development",
]
