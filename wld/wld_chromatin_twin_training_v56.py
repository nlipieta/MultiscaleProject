"""Development-only training for the WLD v5.6 null-aware chromatin model.

This module deliberately reuses every previously inspected v5.3 validation
target for model selection and descriptive comparison.  Consequently it does
not produce an untouched-audit confidence interval, a p-value, a test-set
metric, or a biological claim.  The frozen v5.3 test targets are never
materialized.

Relative to v5.5, optimization is response focused and the model may express
the exact persistence/null response through explicit mechanistic efficacy
gates.  Full-state sliced Wasserstein distance (SWD) is still reported against
the exact measured-control persistence comparator.  All negative controls use
the same true-topology-supported training roster and all reused development
targets, including targets made unreachable by a supplied control topology.
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
from torch import Tensor

from wld_chromatin_twin_v55 import BranchOverrides, ChromatinTwinPriors, topology_digest
from wld_chromatin_twin_v56 import (
    WLDNullAwareChromatinTwin,
    architecture_contract,
)
from wld_chromatin_twin_training_v55 import (
    _artifact,
    _base_supported_target_roster,
    _binary_dense,
    _binary_mean,
    _canonical_digest,
    _device_priors,
    _end_to_end_regulator_reachability,
    _intervention,
    _load_foundation,
    _projections,
    _sample,
    _support_topology_digest,
    atomic_json,
    build_provenance_lock as build_v55_provenance_lock,
    load_twin_priors,
    model_regulators,
    sliced_wasserstein,
)
from wld_chromatin_modules_v55 import SparseFullChromatinBundle, load_v53_sparse_full_bundle, sha256_file
from wld_phase_b_priors import verify_phase_b_priors


SCHEMA_VERSION = "wld-v5.6-null-aware-reused-development"


@dataclass(frozen=True)
class TwinTrainingConfig:
    """Locked v5.6 development configuration.

    A persistence prediction has response NRMSE=1, response cosine=0 and
    relative SWD=1 (apart from the documented weak-response floor), so the
    default response selection score is exactly 1 for persistence.
    """

    epochs: int = 36
    targets_per_epoch: int = 36
    batch_size: int = 48
    learning_rate: float = 2e-3
    representation_learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    integration_steps: int = 5
    horizon: float = 1.0
    projections: int = 32
    validation_cells_per_target: int = 128
    patience: int = 8
    seeds: Tuple[int, ...] = (42, 137, 911)
    control_replicates: int = 10
    control_seed: int = 56_042
    minimum_control_replicates: int = 10
    response_nrmse_weight: float = 0.65
    response_cosine_weight: float = 0.25
    full_state_relative_swd_weight: float = 0.10
    gate_regularization: float = 1e-3
    delta_regularization: float = 2e-2
    min_supported_training_targets: int = 5

    def validate(self) -> None:
        counts = (
            self.epochs,
            self.targets_per_epoch,
            self.batch_size,
            self.integration_steps,
            self.projections,
            self.validation_cells_per_target,
            self.patience,
            self.control_replicates,
            self.minimum_control_replicates,
            self.min_supported_training_targets,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or int(value) < 1
            for value in counts
        ):
            raise ValueError("training counts must be positive integers")
        if self.minimum_control_replicates < 10:
            raise ValueError("v5.6 requires at least ten supplied matched controls")
        if self.control_replicates < self.minimum_control_replicates:
            raise ValueError("control_replicates is below the locked minimum")
        if (
            isinstance(self.control_seed, bool)
            or not isinstance(self.control_seed, numbers.Integral)
            or not 0 <= int(self.control_seed) < 2**32
        ):
            raise ValueError("control_seed must be a non-negative integer")
        if (
            len(self.seeds) < 3
            or len(set(map(int, self.seeds))) != len(self.seeds)
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, numbers.Integral)
                or not 0 <= int(seed) < 2**32
                for seed in self.seeds
            )
        ):
            raise ValueError("at least three unique non-negative integer seeds are required")
        positive = (
            self.learning_rate,
            self.representation_learning_rate,
            self.horizon,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in positive):
            raise ValueError("learning rates and horizon must be finite and positive")
        nonnegative = (
            self.weight_decay,
            self.response_nrmse_weight,
            self.response_cosine_weight,
            self.full_state_relative_swd_weight,
            self.gate_regularization,
            self.delta_regularization,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in nonnegative):
            raise ValueError("loss weights and regularization must be finite and non-negative")
        selection_weights = (
            self.response_nrmse_weight,
            self.response_cosine_weight,
            self.full_state_relative_swd_weight,
        )
        if not math.isclose(sum(selection_weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("response-selection weights must sum to one")


def build_provenance_lock(
    prior_root: Path,
    foundation_checkpoint: Path,
    bundle_root: Path,
    route_root: Path,
    module_root: Path,
) -> Dict[str, object]:
    """Extend the v5.5 content lock with both v5.6 runtime sources."""

    lock = dict(
        build_v55_provenance_lock(
            prior_root,
            foundation_checkpoint,
            bundle_root,
            route_root,
            module_root,
        )
    )
    v55_digest = str(lock.pop("digest"))
    source_root = Path(__file__).resolve().parent
    records = list(lock.get("artifacts", ()))
    records.extend(
        (
            _artifact(source_root / "wld_chromatin_twin_v56.py", "v5.6 null-aware model source"),
            _artifact(source_root / "wld_v56_topology_controls.py", "v5.6 matched-control source"),
            _artifact(source_root / "wld_chromatin_twin_training_v56.py", "v5.6 trainer source"),
        )
    )
    lock.update(
        schema_version=SCHEMA_VERSION,
        artifacts=records,
        v55_lineage_digest=v55_digest,
    )
    lock["digest"] = _canonical_digest(lock)
    return lock


def _same_static_priors(left: ChromatinTwinPriors, right: ChromatinTwinPriors) -> bool:
    for name in (
        "tf_peak_motif",
        "complex_module_effect",
        "module_peak_loading",
        "foundation_peak_index",
    ):
        if not torch.equal(torch.as_tensor(getattr(left, name)), torch.as_tensor(getattr(right, name))):
            return False
    return True


def _degree_signature(value: Tensor) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    mask = torch.as_tensor(value).ne(0)
    return (
        tuple(map(int, torch.sort(mask.sum(1)).values.cpu().tolist())),
        tuple(map(int, mask.sum(0).cpu().tolist())),
    )


def _matched_control_digest(priors: ChromatinTwinPriors) -> str:
    """Reproduce the topology-control helper's nested tensor content digest."""

    digest = hashlib.sha256()
    for field in fields(priors):
        array = (
            torch.as_tensor(getattr(priors, field.name))
            .detach()
            .cpu()
            .contiguous()
            .numpy()
        )
        tensor_digest = hashlib.sha256()
        tensor_digest.update(str(array.dtype).encode("ascii"))
        tensor_digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        tensor_digest.update(array.tobytes())
        digest.update(field.name.encode("utf-8"))
        digest.update(tensor_digest.digest())
    return digest.hexdigest()


def validate_control_priors(
    true_priors: ChromatinTwinPriors,
    control_priors: Sequence[ChromatinTwinPriors],
    *,
    minimum: int,
) -> List[Dict[str, object]]:
    """Validate supplied controls and return their immutable topology lock.

    Controls must preserve both bipartite degree sequences and all downstream
    motif/module evidence.  Their per-regulator end-to-end reachable-bin count
    is disclosed because first-layer degree matching alone need not preserve
    that capacity exactly.
    """

    true_priors.validate()
    if len(control_priors) < int(minimum):
        raise ValueError(f"Expected at least {minimum} supplied control priors")
    true_digest = topology_digest(true_priors)
    true_tf_degree = _degree_signature(true_priors.regulator_tf_support)
    true_complex_degree = _degree_signature(true_priors.regulator_complex_support)
    true_reachability = _end_to_end_regulator_reachability(true_priors).cpu().numpy()
    seen = {true_digest}
    records: List[Dict[str, object]] = []
    for index, control in enumerate(control_priors, start=1):
        control.validate()
        digest = topology_digest(control)
        if digest in seen:
            raise ValueError(f"Control topology {index} is duplicated or equals the true topology")
        seen.add(digest)
        if not _same_static_priors(true_priors, control):
            raise ValueError(f"Control topology {index} changed downstream/static priors")
        if _degree_signature(control.regulator_tf_support) != true_tf_degree:
            raise ValueError(f"Control topology {index} changed TF-route degree sequences")
        if _degree_signature(control.regulator_complex_support) != true_complex_degree:
            raise ValueError(f"Control topology {index} changed complex-route degree sequences")
        control_reachability = _end_to_end_regulator_reachability(control).cpu().numpy()
        records.append(
            {
                "name": f"matched_control_{index:02d}",
                "topology_sha256": digest,
                "matched_control_sha256": _matched_control_digest(control),
                "support_topology_sha256": _support_topology_digest(control),
                "degree_sequences_matched": True,
                "static_downstream_priors_matched": True,
                "per_regulator_route_reachability_labels_matched": bool(
                    np.array_equal(true_reachability, control_reachability)
                ),
                "route_reachable_regulator_count_matched": bool(
                    np.count_nonzero(true_reachability)
                    == np.count_nonzero(control_reachability)
                ),
                "reachable_regulators": int(np.count_nonzero(control_reachability)),
            }
        )
    return records


def _validate_control_generation_audit(
    audit: Mapping[str, object],
    control_lock: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Fail closed unless the supplied controls satisfy the strong null."""

    if not isinstance(audit, Mapping):
        raise TypeError("matched control generation audit must be a mapping")
    records = audit.get("controls")
    if not isinstance(records, list) or len(records) != len(control_lock):
        raise RuntimeError("matched control audit has a changed replicate roster")
    if int(audit.get("replicates", -1)) != len(control_lock):
        raise RuntimeError("matched control audit replicate count disagrees")
    contract = audit.get("matching_contract")
    required_contract = (
        "joint_tf_and_complex_profile_permutation",
        "profiles_permuted_only_within_whole_target_split",
        "zero_fixed_regulator_labels",
        "support_column_degrees_exact",
        "support_row_degree_distributions_exact",
        "support_weight_multisets_exact",
        "end_to_end_footprint_distribution_exact",
        "end_to_end_complex_sign_distribution_exact",
        "end_to_end_signed_and_absolute_mass_distributions_exact",
        "downstream_evidence_tensors_unchanged",
    )
    if not isinstance(contract, Mapping) or any(
        contract.get(name) is not True for name in required_contract
    ):
        raise RuntimeError("matched control audit does not satisfy the strong null contract")
    if contract.get("test_outcomes_or_observations_read") is not False:
        raise RuntimeError("matched control construction accessed sealed outcomes")
    expected_digests = [str(record["matched_control_sha256"]) for record in control_lock]
    observed_digests = [str(record.get("topology_sha256", "")) for record in records]
    if observed_digests != expected_digests:
        raise RuntimeError("matched control audit topology hashes changed order/content")
    support_flags = (
        "column_degrees_exact",
        "row_degree_distribution_exact",
        "weight_multiset_exact",
        "total_mass_exact",
    )
    for index, record in enumerate(records, start=1):
        if (
            record.get("fixed_regulator_labels") != 0
            or record.get("split_boundaries_crossed") is not False
            or record.get("end_to_end_row_profile_permutation_exact") is not True
        ):
            raise RuntimeError(f"matched control {index} violates its permutation contract")
        strata = record.get("strata")
        if not isinstance(strata, Mapping) or set(strata) != {
            "train",
            "validation",
            "test",
        }:
            raise RuntimeError(f"matched control {index} lacks split-stratum audits")
        for label, stratum in strata.items():
            if (
                not isinstance(stratum, Mapping)
                or stratum.get("end_to_end_row_profile_permutation_exact") is not True
            ):
                raise RuntimeError(
                    f"matched control {index}/{label} changed end-to-end profiles"
                )
            for branch in ("tf_support", "complex_support"):
                support = stratum.get(branch)
                if not isinstance(support, Mapping) or any(
                    support.get(flag) is not True for flag in support_flags
                ):
                    raise RuntimeError(
                        f"matched control {index}/{label}/{branch} is not matched"
                    )
    return dict(audit)


def _aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    fields = (
        "model_swd",
        "persistence_swd",
        "gain_over_persistence",
        "relative_swd",
        "response_nrmse",
        "response_cosine",
        "response_cosine_loss",
        "response_selection_score",
        "observed_response_rms",
        "predicted_response_rms",
        "mean_absolute_predicted_change",
    )
    return {
        "targets": len(rows),
        **{
            field: (
                float(np.mean([float(row[field]) for row in rows])) if rows else None
            )
            for field in fields
        },
    }


def response_focused_loss(
    prediction: Tensor,
    observed: Tensor,
    control: Tensor,
    projection: Tensor,
    observed_target_mean: Tensor,
    observed_control_mean: Tensor,
    config: TwinTrainingConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    """Optimize perturbation response; retain full-state SWD as a small term."""

    model_swd = sliced_wasserstein(prediction, observed, projection)
    persistence_swd = sliced_wasserstein(control, observed, projection)
    relative_swd = model_swd / persistence_swd.clamp_min(1e-8)
    observed_response = observed_target_mean - observed_control_mean
    predicted_response = prediction.mean(0) - control.mean(0)
    response_rms = observed_response.square().mean().sqrt().clamp_min(2e-3)
    response_nrmse = (
        (predicted_response - observed_response).square().mean().sqrt() / response_rms
    )
    response_cosine = F.cosine_similarity(
        predicted_response.unsqueeze(0),
        observed_response.unsqueeze(0),
        dim=1,
        eps=1e-8,
    ).mean()
    cosine_loss = 1.0 - response_cosine
    total = (
        config.response_nrmse_weight * response_nrmse
        + config.response_cosine_weight * cosine_loss
        + config.full_state_relative_swd_weight * relative_swd
    )
    return total, {
        "response_loss": float(total.detach()),
        "model_swd": float(model_swd.detach()),
        "persistence_swd": float(persistence_swd.detach()),
        "relative_swd": float(relative_swd.detach()),
        "response_nrmse": float(response_nrmse.detach()),
        "response_cosine": float(response_cosine.detach()),
        "response_cosine_loss": float(cosine_loss.detach()),
    }


def _realized_regularization(
    model: WLDNullAwareChromatinTwin,
    output: Mapping[str, Tensor],
    control: Tensor,
    config: TwinTrainingConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    """Call the v5.6 realized regularizer with a narrow compatibility path."""

    method = getattr(model, "realized_regularization", None)
    if callable(method):
        value = method(
            output,
            control,
            gate_weight=config.gate_regularization,
            delta_weight=config.delta_regularization,
        )
        if isinstance(value, Tensor):
            return value, {"realized_regularization": float(value.detach())}
        if isinstance(value, Mapping):
            total = value.get("total", value.get("loss", value.get("regularization")))
            if not isinstance(total, Tensor):
                raise TypeError("realized_regularization mapping lacks a tensor total")
            metrics = {
                str(name): float(item.detach())
                for name, item in value.items()
                if isinstance(item, Tensor) and item.numel() == 1
            }
            metrics.setdefault("realized_regularization", float(total.detach()))
            return total, metrics
        raise TypeError("realized_regularization returned an unsupported value")

    # Compatibility for an implementation exposing only effective gates.
    gate_method = getattr(model.field, "effective_branch_gates", None)
    if not callable(gate_method):
        raise AttributeError("v5.6 model exposes neither realized regularization nor gates")
    gates = gate_method()
    values = list(gates.values()) if isinstance(gates, Mapping) else [gates]
    tensors = [value for value in values if isinstance(value, Tensor)]
    if not tensors:
        raise TypeError("effective_branch_gates returned no tensors")
    gate_penalty = torch.stack([value.square().mean() for value in tensors]).mean()
    delta_penalty = (output["atac_t"] - control).square().mean()
    total = (
        config.gate_regularization * gate_penalty
        + config.delta_regularization * delta_penalty
    )
    return total, {
        "realized_gate_penalty": float(gate_penalty.detach()),
        "realized_delta_penalty": float(delta_penalty.detach()),
        "realized_regularization": float(total.detach()),
    }


def _make_optimizer(
    model: WLDNullAwareChromatinTwin,
    config: TwinTrainingConfig,
) -> Tuple[torch.optim.Optimizer, List[Tensor], Dict[str, object]]:
    """Group parameters without decaying inverse-softplus/gate coordinates."""

    groups: Dict[Tuple[float, float], List[Tensor]] = {}
    names_by_group: Dict[Tuple[float, float], List[str]] = {}
    optimized: List[Tensor] = []
    frozen: List[str] = []
    zero_decay: List[str] = []
    for name, parameter in model.named_parameters():
        is_representation = name.startswith("foundation.encoder.") or name.startswith(
            "foundation.context_network."
        )
        is_mechanistic = name.startswith("field.")
        if not is_representation and not is_mechanistic:
            parameter.requires_grad_(False)
            frozen.append(name)
            continue
        parameter.requires_grad_(True)
        learning_rate = (
            config.representation_learning_rate if is_representation else config.learning_rate
        )
        lower = name.lower()
        no_decay = (
            ".raw_" in lower
            or "gate" in lower
            or "efficacy" in lower
            or lower.endswith(".bias")
        )
        decay = 0.0 if no_decay else config.weight_decay
        if no_decay:
            zero_decay.append(name)
        key = (float(learning_rate), float(decay))
        groups.setdefault(key, []).append(parameter)
        names_by_group.setdefault(key, []).append(name)
        optimized.append(parameter)
    if not optimized or len({id(parameter) for parameter in optimized}) != len(optimized):
        raise RuntimeError("optimizer parameter partition is empty or duplicated")
    raw_or_gate = {
        name
        for name, _parameter in model.named_parameters()
        if ".raw_" in name.lower() or "gate" in name.lower() or "efficacy" in name.lower()
    }
    if not raw_or_gate.issubset(set(zero_decay)):
        raise RuntimeError("an inverse-softplus/gate parameter received weight decay")
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters, "lr": lr, "weight_decay": decay}
            for (lr, decay), parameters in groups.items()
        ]
    )
    audit = {
        "groups": [
            {
                "learning_rate": lr,
                "weight_decay": decay,
                "parameters": names_by_group[(lr, decay)],
            }
            for lr, decay in sorted(groups)
        ],
        "zero_weight_decay_parameters": sorted(zero_decay),
        "frozen_parameters": sorted(frozen),
        "raw_parameter_l2": False,
        "realized_gate_regularization": True,
        "realized_delta_regularization": True,
    }
    return optimizer, optimized, audit


def compile_training_perturbed_mean_baseline(
    bundle: SparseFullChromatinBundle,
    training_targets: Sequence[str],
    output_root: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Compile a screen-matched generic perturbation shift from train only.

    This Systema-style baseline averages target-minus-NTC pseudobulk responses
    across the fixed true-topology training targets in each screen.  It is a
    generic perturbation response, not a target-specific prediction.  No
    validation/test value is read during construction.
    """

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    numeric_path = root / "training_perturbed_mean_response.npz"
    manifest_path = root / "training_perturbed_mean_manifest.json"
    targets = tuple(sorted(set(map(str, training_targets))))
    if not targets or not set(targets).issubset(set(bundle.split_targets("train"))):
        raise RuntimeError("perturbed-mean baseline received a non-training target")
    lineage = {
        "schema_version": "wld-v5.6-training-only-perturbed-mean-baseline",
        "construction_split": "train",
        "construction_targets": list(targets),
        "response_bins": len(bundle.bins),
        "v53_manifest_sha256": bundle.provenance.get("v53_manifest_sha256"),
        "whole_target_split_sha256": bundle.provenance.get(
            "whole_target_split_sha256"
        ),
        "whole_target_roster_sha256": bundle.provenance.get(
            "whole_target_roster_sha256"
        ),
        "v53_matrix_sha256": bundle.provenance.get("v53_matrix_sha256"),
        "v53_cells_sha256": bundle.provenance.get("v53_cells_sha256"),
        "v53_bins_sha256": bundle.provenance.get("v53_bins_sha256"),
        "definition": (
            "equal-target mean of binary pseudobulk target-minus-screen-matched-NTC "
            "responses, compiled independently per screen"
        ),
        "validation_values_used": False,
        "test_values_materialized": False,
        "test_values_used": False,
        "reference": {
            "name": "Systema",
            "doi": "10.1038/s41587-025-02777-8",
            "role": "motivation for a systematic perturbed-mean baseline",
        },
    }
    lineage_digest = _canonical_digest(lineage)
    if manifest_path.is_file() and numeric_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("lineage_digest") != lineage_digest
            or manifest.get("numeric_sha256") != sha256_file(numeric_path)
            or manifest.get("validation_values_used") is not False
            or manifest.get("test_values_materialized") is not False
            or manifest.get("test_values_used") is not False
        ):
            raise RuntimeError("restored perturbed-mean baseline changed lineage")
        with np.load(numeric_path, allow_pickle=False) as values:
            screens = tuple(map(str, values["screens"].tolist()))
            matrix = np.asarray(values["responses"], dtype=np.float32)
        if matrix.shape != (len(screens), len(bundle.bins)):
            raise RuntimeError("restored perturbed-mean baseline has invalid dimensions")
        return {screen: matrix[index] for index, screen in enumerate(screens)}, manifest

    screens = sorted(
        {
            screen
            for target in targets
            for screen in bundle.target_screens("train", target)
        }
    )
    if not screens:
        raise RuntimeError("training targets have no screens")
    responses: List[np.ndarray] = []
    screen_records: List[Dict[str, object]] = []
    for screen in screens:
        screen_targets = [
            target
            for target in targets
            if screen in bundle.target_screens("train", target)
        ]
        control_rows = bundle.rows("train", screen, "NTC")
        if not screen_targets or not len(control_rows):
            raise RuntimeError(f"cannot construct perturbed mean for screen {screen}")
        control_mean = _binary_mean(bundle.accessibility, control_rows)
        target_effects = []
        for target in screen_targets:
            target_rows = bundle.rows("train", screen, target)
            if not len(target_rows):
                raise RuntimeError(f"missing training rows for {target}/{screen}")
            target_effects.append(
                _binary_mean(bundle.accessibility, target_rows) - control_mean
            )
        response = np.mean(np.stack(target_effects), axis=0).astype(np.float32)
        responses.append(response)
        screen_records.append(
            {
                "screen": screen,
                "training_targets": screen_targets,
                "target_count": len(screen_targets),
                "control_cells": int(len(control_rows)),
                "mean_absolute_response": float(np.mean(np.abs(response))),
                "response_rms": float(np.sqrt(np.mean(response**2))),
            }
        )
    response_matrix = np.stack(responses).astype(np.float32)
    temporary = numeric_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            screens=np.asarray(screens),
            responses=response_matrix,
        )
    os.replace(temporary, numeric_path)
    manifest = {
        **lineage,
        "lineage_digest": lineage_digest,
        "screens": screen_records,
        "numeric_file": numeric_path.name,
        "numeric_sha256": sha256_file(numeric_path),
        "numeric_bytes": numeric_path.stat().st_size,
    }
    atomic_json(manifest_path, manifest)
    return {screen: response_matrix[index] for index, screen in enumerate(screens)}, manifest


def evaluate_model(
    model: WLDNullAwareChromatinTwin,
    bundle: SparseFullChromatinBundle,
    targets: Sequence[str],
    config: TwinTrainingConfig,
    *,
    seed: int,
    overrides: Optional[BranchOverrides] = None,
) -> Dict[str, object]:
    """Evaluate every named reused-development target with exact persistence."""

    device = next(model.parameters()).device
    regulators = model_regulators(model, bundle)
    regulator_index = {name: index for index, name in enumerate(regulators)}
    projection = _projections(len(bundle.bins), config.projections, seed + 31, device)
    reachability = model.field.reachability()
    rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for target_number, target in enumerate(sorted(set(map(str, targets)))):
            if target not in regulator_index:
                raise RuntimeError(f"Unknown development perturbation target {target}")
            screen_metrics: List[Tuple[int, Dict[str, float]]] = []
            for screen_number, screen in enumerate(bundle.target_screens("validation", target)):
                target_rows = bundle.rows("validation", screen, target)
                control_rows = bundle.rows("validation", screen, "NTC")
                if not len(control_rows):
                    raise RuntimeError(f"No validation NTC for {target}/{screen}")
                count = min(
                    config.validation_cells_per_target,
                    len(target_rows),
                    len(control_rows),
                )
                rng = np.random.default_rng(
                    seed + 1009 * (target_number + 1) + 7919 * screen_number
                )
                observed = _binary_dense(
                    bundle.accessibility, _sample(target_rows, count, rng), device
                )
                control = _binary_dense(
                    bundle.accessibility, _sample(control_rows, count, rng), device
                )
                output = model(
                    control,
                    _intervention(target, regulator_index, count, device),
                    horizon=config.horizon,
                    steps=config.integration_steps,
                    overrides=overrides,
                )
                prediction = output["atac_t"]
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
                response_cosine = float(
                    F.cosine_similarity(
                        predicted_response.unsqueeze(0),
                        observed_response.unsqueeze(0),
                        dim=1,
                        eps=1e-8,
                    )[0]
                )
                relative_swd = model_swd / max(persistence_swd, 1e-8)
                cosine_loss = 1.0 - response_cosine
                score = (
                    config.response_nrmse_weight * response_nrmse
                    + config.response_cosine_weight * cosine_loss
                    + config.full_state_relative_swd_weight * relative_swd
                )
                screen_metrics.append(
                    (
                        count,
                        {
                            "model_swd": model_swd,
                            "persistence_swd": persistence_swd,
                            "gain_over_persistence": persistence_swd - model_swd,
                            "relative_swd": relative_swd,
                            "response_nrmse": response_nrmse,
                            "response_cosine": response_cosine,
                            "response_cosine_loss": cosine_loss,
                            "response_selection_score": score,
                            "observed_response_rms": raw_response_rms,
                            "response_nrmse_denominator": response_scale,
                            "response_nrmse_floor_used": float(raw_response_rms < 2e-3),
                            "predicted_response_rms": float(
                                predicted_response.square().mean().sqrt()
                            ),
                            "mean_absolute_predicted_change": float(
                                predicted_response.abs().mean()
                            ),
                        },
                    )
                )
            if not screen_metrics:
                raise RuntimeError(f"No validation observations for {target}")
            weights = np.asarray([count for count, _ in screen_metrics], dtype=np.float64)
            weights /= weights.sum()
            keys = tuple(screen_metrics[0][1])
            metrics = {
                key: float(
                    sum(
                        weight * values[key]
                        for weight, (_count, values) in zip(weights, screen_metrics)
                    )
                )
                for key in keys
            }
            index = regulator_index[target]
            tf_bins = int(torch.count_nonzero(reachability["tf"][index]))
            complex_bins = int(torch.count_nonzero(reachability["complex"][index]))
            rows.append(
                {
                    "target": target,
                    "screens": bundle.target_screens("validation", target),
                    "cells": int(sum(count for count, _ in screen_metrics)),
                    "tf_reachable_bins": tf_bins,
                    "complex_reachable_bins": complex_bins,
                    "route_supported": bool(tf_bins or complex_bins),
                    **metrics,
                }
            )
    supported = [row for row in rows if bool(row["route_supported"])]
    return {
        "per_target": rows,
        "all_targets": _aggregate(rows),
        "route_supported_targets": _aggregate(supported),
        "unsupported_targets": sorted(
            str(row["target"]) for row in rows if not bool(row["route_supported"])
        ),
        "evaluation_seed": int(seed),
        "validation_targets_previously_used_in_v55": True,
        "untouched_audit_inference": False,
        "test_values_materialized": False,
    }


def evaluate_perturbed_mean_baseline(
    bundle: SparseFullChromatinBundle,
    targets: Sequence[str],
    responses: Mapping[str, np.ndarray],
    config: TwinTrainingConfig,
    *,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate the frozen train-only generic shift on reused validation."""

    projection = _projections(len(bundle.bins), config.projections, seed + 31, device)
    rows: List[Dict[str, object]] = []
    for target_number, target in enumerate(sorted(set(map(str, targets)))):
        screen_metrics: List[Tuple[int, Dict[str, float]]] = []
        for screen_number, screen in enumerate(bundle.target_screens("validation", target)):
            if screen not in responses:
                raise RuntimeError(f"no training-only perturbed mean for screen {screen}")
            target_rows = bundle.rows("validation", screen, target)
            control_rows = bundle.rows("validation", screen, "NTC")
            count = min(
                config.validation_cells_per_target,
                len(target_rows),
                len(control_rows),
            )
            if count < 1:
                raise RuntimeError(f"missing validation population for {target}/{screen}")
            rng = np.random.default_rng(
                seed + 1009 * (target_number + 1) + 7919 * screen_number
            )
            observed = _binary_dense(
                bundle.accessibility, _sample(target_rows, count, rng), device
            )
            control = _binary_dense(
                bundle.accessibility, _sample(control_rows, count, rng), device
            )
            shift = torch.as_tensor(responses[screen], dtype=control.dtype, device=device)
            prediction = (control + shift.unsqueeze(0)).clamp(0.0, 1.0)
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
            response_cosine = float(
                F.cosine_similarity(
                    predicted_response.unsqueeze(0),
                    observed_response.unsqueeze(0),
                    dim=1,
                    eps=1e-8,
                )[0]
            )
            relative_swd = model_swd / max(persistence_swd, 1e-8)
            cosine_loss = 1.0 - response_cosine
            score = (
                config.response_nrmse_weight * response_nrmse
                + config.response_cosine_weight * cosine_loss
                + config.full_state_relative_swd_weight * relative_swd
            )
            screen_metrics.append(
                (
                    count,
                    {
                        "model_swd": model_swd,
                        "persistence_swd": persistence_swd,
                        "gain_over_persistence": persistence_swd - model_swd,
                        "relative_swd": relative_swd,
                        "response_nrmse": response_nrmse,
                        "response_cosine": response_cosine,
                        "response_cosine_loss": cosine_loss,
                        "response_selection_score": score,
                        "observed_response_rms": raw_response_rms,
                        "response_nrmse_denominator": response_scale,
                        "response_nrmse_floor_used": float(raw_response_rms < 2e-3),
                        "predicted_response_rms": float(
                            predicted_response.square().mean().sqrt()
                        ),
                        "mean_absolute_predicted_change": float(
                            predicted_response.abs().mean()
                        ),
                    },
                )
            )
        weights = np.asarray([count for count, _ in screen_metrics], dtype=np.float64)
        weights /= weights.sum()
        keys = tuple(screen_metrics[0][1])
        metrics = {
            key: float(
                sum(
                    weight * values[key]
                    for weight, (_count, values) in zip(weights, screen_metrics)
                )
            )
            for key in keys
        }
        rows.append(
            {
                "target": target,
                "screens": bundle.target_screens("validation", target),
                "cells": int(sum(count for count, _ in screen_metrics)),
                **metrics,
            }
        )
    return {
        "per_target": rows,
        "all_targets": _aggregate(rows),
        "evaluation_seed": int(seed),
        "baseline": "screen-matched training-only perturbed mean",
        "validation_values_used_for_construction": False,
        "validation_targets_previously_used_in_v55": True,
        "untouched_audit_inference": False,
        "test_values_materialized": False,
    }


def _fit_condition(
    condition: str,
    seed: int,
    priors: ChromatinTwinPriors,
    prior_root: Path,
    foundation_checkpoint: Path,
    bundle: SparseFullChromatinBundle,
    training_targets: Sequence[str],
    development_targets: Sequence[str],
    output_root: Path,
    config: TwinTrainingConfig,
    provenance_digest: str,
    device: torch.device,
) -> Tuple[WLDNullAwareChromatinTwin, Dict[str, object]]:
    root = Path(output_root) / condition / f"seed_{int(seed)}"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "condition_report.json"
    model_path = root / "best_model.pt"
    state_path = root / "training_state.pt"
    config_json = json.loads(json.dumps(asdict(config)))
    training_targets = tuple(map(str, training_targets))
    development_targets = tuple(sorted(set(map(str, development_targets))))
    prior_digest = topology_digest(priors)

    def new_model() -> WLDNullAwareChromatinTwin:
        foundation = _load_foundation(prior_root, foundation_checkpoint, device)
        return WLDNullAwareChromatinTwin(
            foundation, _device_priors(priors, device)
        ).to(device)

    if report_path.is_file() and model_path.is_file():
        report = json.loads(report_path.read_text())
        checks = (
            report.get("schema_version") == SCHEMA_VERSION,
            report.get("condition") == condition,
            int(report.get("seed", -1)) == int(seed),
            report.get("config") == config_json,
            report.get("provenance_digest") == provenance_digest,
            report.get("topology_sha256") == prior_digest,
            report.get("fixed_true_topology_training_targets") == list(training_targets),
            report.get("reused_development_targets") == list(development_targets),
            report.get("checkpoint_sha256") == sha256_file(model_path),
            report.get("claims", {}).get("test_targets_evaluated") is False,
            report.get("claims", {}).get("untouched_audit_inference") is False,
        )
        if not all(checks):
            raise RuntimeError(f"Completed {condition}/seed {seed} changed locked inputs")
        model = new_model()
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"PASS: restored {condition} seed {seed}", flush=True)
        return model, report

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = new_model()
    optimizer, optimized_parameters, optimizer_audit = _make_optimizer(model, config)
    regulators = model_regulators(model, bundle)
    regulator_index = {name: index for index, name in enumerate(regulators)}
    if len(training_targets) < config.min_supported_training_targets:
        raise RuntimeError("Too few fixed true-topology-supported training targets")
    if any(target not in regulator_index for target in training_targets):
        raise RuntimeError("Fixed training target lacks a named intervention")
    projection = _projections(len(bundle.bins), config.projections, int(seed) + 17, device)
    start_epoch = 0
    best_score = float("inf")
    best_state = None
    waiting = 0
    history: List[Dict[str, object]] = []
    if state_path.is_file():
        try:
            state = torch.load(state_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location=device)
        checks = (
            state.get("schema_version") == SCHEMA_VERSION,
            state.get("condition") == condition,
            int(state.get("seed", -1)) == int(seed),
            state.get("config") == asdict(config),
            state.get("provenance_digest") == provenance_digest,
            state.get("topology_sha256") == prior_digest,
            state.get("fixed_true_topology_training_targets") == list(training_targets),
            state.get("reused_development_targets") == list(development_targets),
        )
        if not all(checks):
            raise RuntimeError(f"Resume state for {condition}/seed {seed} changed inputs")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_state = state["best_state"]
        if best_state is not None:
            best_state = {name: value.detach().cpu() for name, value in best_state.items()}
        waiting = int(state["waiting"])
        history = list(state["history"])
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
        epoch_rows: List[Dict[str, float]] = []
        for target_value in selected:
            target = str(target_value)
            screens = bundle.target_screens("train", target)
            screen = screens[int(rng.integers(0, len(screens)))]
            target_rows = bundle.rows("train", screen, target)
            control_rows = bundle.rows("train", screen, "NTC")
            if not len(control_rows):
                raise RuntimeError(f"No train NTC for {target}/{screen}")
            observed = _binary_dense(
                bundle.accessibility,
                _sample(target_rows, config.batch_size, rng),
                device,
            )
            control = _binary_dense(
                bundle.accessibility,
                _sample(control_rows, config.batch_size, rng),
                device,
            )
            target_key = ("train", screen, target)
            control_key = ("train", screen, "NTC")
            if target_key not in mean_cache:
                mean_cache[target_key] = torch.as_tensor(
                    _binary_mean(bundle.accessibility, target_rows), device=device
                )
            if control_key not in mean_cache:
                mean_cache[control_key] = torch.as_tensor(
                    _binary_mean(bundle.accessibility, control_rows), device=device
                )
            output = model(
                control,
                _intervention(target, regulator_index, config.batch_size, device),
                horizon=config.horizon,
                steps=config.integration_steps,
            )
            response_loss, response_metrics = response_focused_loss(
                output["atac_t"],
                observed,
                control,
                projection,
                mean_cache[target_key],
                mean_cache[control_key],
                config,
            )
            realized_penalty, realized_metrics = _realized_regularization(
                model, output, control, config
            )
            total = response_loss + realized_penalty
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(optimized_parameters, 5.0)
            optimizer.step()
            epoch_rows.append(
                {
                    **response_metrics,
                    **realized_metrics,
                    "total_loss": float(total.detach()),
                }
            )

        development = evaluate_model(
            model,
            bundle,
            development_targets,
            config,
            seed=int(seed) + 500_001,
        )
        score = float(development["all_targets"]["response_selection_score"])
        if not math.isfinite(score):
            raise RuntimeError("No finite all-target response selection score")
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
        training_summary = {
            key: float(np.mean([row[key] for row in epoch_rows]))
            for key in epoch_rows[0]
        }
        history.append(
            {
                "epoch": epoch,
                "training": training_summary,
                "reused_development_all_targets": development["all_targets"],
                "improved": improved,
            }
        )
        print(
            f"   {condition} seed {seed} epoch {epoch:03d} | "
            f"response {training_summary['response_loss']:.6f} | "
            f"realized-reg {training_summary['realized_regularization']:.6f} | "
            f"development {score:.6f}",
            flush=True,
        )
        temporary = state_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "condition": condition,
                "seed": int(seed),
                "epoch": epoch,
                "config": asdict(config),
                "provenance_digest": provenance_digest,
                "topology_sha256": prior_digest,
                "fixed_true_topology_training_targets": list(training_targets),
                "reused_development_targets": list(development_targets),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_score": best_score,
                "best_state": best_state,
                "waiting": waiting,
                "history": history,
            },
            temporary,
        )
        os.replace(temporary, state_path)
        if waiting >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No development-selected checkpoint for {condition}/seed {seed}")
    model.load_state_dict(best_state)
    final_development = evaluate_model(
        model,
        bundle,
        development_targets,
        config,
        seed=int(seed) + 900_001,
    )
    final_branch_gates = {
        name: float(value.detach().cpu())
        for name, value in model.field.effective_branch_gates().items()
    }
    temporary_model = model_path.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), temporary_model)
    os.replace(temporary_model, model_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "condition": condition,
        "seed": int(seed),
        "config": config_json,
        "provenance_digest": provenance_digest,
        "topology_sha256": prior_digest,
        "fixed_true_topology_training_targets": list(training_targets),
        "reused_development_targets": list(development_targets),
        "best_reused_development_score": best_score,
        "epochs_attempted": len(history),
        "history": history,
        "final_reused_development": final_development,
        "final_effective_branch_gates": final_branch_gates,
        "optimizer_audit": optimizer_audit,
        "checkpoint_sha256": sha256_file(model_path),
        "claims": {
            "development_only": True,
            "validation_targets_previously_used_in_v55": True,
            "untouched_audit_inference": False,
            "test_targets_evaluated": False,
            "external_subject_study_evaluated": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
        },
    }
    atomic_json(report_path, report)
    return model, report


def _target_map(report: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    rows = report["final_reused_development"]["per_target"]
    result = {str(row["target"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate reused-development target metrics")
    return result


def _descriptive_effect(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, object]:
    grouped: Dict[str, List[float]] = {}
    seed_effects: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["target"]), []).append(float(row[key]))
        seed_effects.setdefault(int(row["seed"]), []).append(float(row[key]))
    per_target = np.asarray(
        [np.mean(grouped[target]) for target in sorted(grouped)], dtype=np.float64
    )
    if not len(per_target):
        return {"targets": 0, "seeds": 0, "inference": False}
    return {
        "mean": float(per_target.mean()),
        "median": float(np.median(per_target)),
        "minimum": float(per_target.min()),
        "maximum": float(per_target.max()),
        "positive_target_fraction": float(np.mean(per_target > 0)),
        "positive_seed_fraction": float(
            np.mean([np.mean(values) > 0 for values in seed_effects.values()])
        ),
        "targets": len(per_target),
        "seeds": len(seed_effects),
        "inference": False,
        "inferential_statistics_computed": False,
        "note": "descriptive reuse of previously inspected v5.5 validation targets",
    }


def _verify_completed_report(
    report: Mapping[str, object],
    *,
    provenance_digest: str,
    config_json: Mapping[str, object],
    expected_conditions: Iterable[str],
    development_targets: Sequence[str],
) -> None:
    conditions = report.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(expected_conditions):
        raise RuntimeError("Completed v5.6 report has a changed condition index")
    checks = (
        report.get("schema_version") == SCHEMA_VERSION,
        report.get("provenance", {}).get("digest") == provenance_digest,
        report.get("config") == dict(config_json),
        report.get("reused_development_targets") == list(development_targets),
        report.get("claims", {}).get("test_targets_evaluated") is False,
        report.get("claims", {}).get("untouched_audit_inference") is False,
    )
    if not all(checks):
        raise RuntimeError("Completed v5.6 report does not match locked inputs/config")
    for condition, seeds in conditions.items():
        if set(seeds) != {str(int(seed)) for seed in config_json["seeds"]}:
            raise RuntimeError(f"Completed condition has missing seeds: {condition}")
        for seed, record in seeds.items():
            report_path = Path(str(record.get("report", "")))
            checkpoint_path = Path(str(record.get("checkpoint", "")))
            if not report_path.is_file() or not checkpoint_path.is_file():
                raise RuntimeError(f"Missing completed condition artifact: {condition}/{seed}")
            if sha256_file(report_path) != record.get("report_sha256"):
                raise RuntimeError(f"Changed completed condition report: {condition}/{seed}")
            if sha256_file(checkpoint_path) != record.get("checkpoint_sha256"):
                raise RuntimeError(f"Changed completed checkpoint: {condition}/{seed}")


def run_twin_development(
    prior_root: Path,
    foundation_checkpoint: Path,
    bundle_root: Path,
    route_root: Path,
    module_root: Path,
    output_root: Path,
    config: TwinTrainingConfig,
    *,
    control_priors: Sequence[ChromatinTwinPriors],
    control_generation_audit: Mapping[str, object],
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Run v5.6 descriptive development while keeping test targets sealed."""

    config.validate()
    prior_root = Path(prior_root)
    foundation_checkpoint = Path(foundation_checkpoint)
    bundle_root = Path(bundle_root)
    route_root = Path(route_root)
    module_root = Path(module_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    verify_phase_b_priors(prior_root)
    corpus_report_path = foundation_checkpoint.parent / "wld_corpus_pretraining_report.json"
    if not corpus_report_path.is_file():
        raise FileNotFoundError(corpus_report_path)
    corpus_report = json.loads(corpus_report_path.read_text())
    if corpus_report.get("checkpoint_sha256") != sha256_file(foundation_checkpoint):
        raise RuntimeError("expanded-corpus checkpoint/report lineage changed")
    sealed_flags = {
        name: corpus_report.get(name)
        for name in (
            "sealed_test_downloaded",
            "sealed_test_evaluated",
            "model_assessed_on_sealed_test",
        )
    }
    if any(value is not False for value in sealed_flags.values()):
        raise RuntimeError(f"foundation checkpoint crossed its sealed boundary: {sealed_flags}")
    base_provenance = build_provenance_lock(
        prior_root, foundation_checkpoint, bundle_root, route_root, module_root
    )
    true_priors, _atlas, prior_audit = load_twin_priors(
        prior_root, route_root, module_root
    )
    control_lock = validate_control_priors(
        true_priors, control_priors, minimum=config.minimum_control_replicates
    )
    if len(control_priors) != config.control_replicates:
        raise RuntimeError("supplied control count differs from locked config")
    validated_control_audit = _validate_control_generation_audit(
        control_generation_audit, control_lock
    )
    bundle = load_v53_sparse_full_bundle(bundle_root, prior_root=prior_root)
    if any(str(split).lower() == "test" for split in bundle.splits):
        raise RuntimeError("sealed test rows were materialized")
    route_vocab = json.loads((route_root / "route_vocab.json").read_text())
    regulators = tuple(str(value).upper() for value in route_vocab["regulators"])
    setattr(bundle, "regulator_vocab", regulators)
    bundle.provenance["regulator_vocab"] = regulators
    true_reachability = _end_to_end_regulator_reachability(true_priors).cpu().numpy()
    fixed_training_targets = _base_supported_target_roster(
        bundle.split_targets("train"), regulators, true_reachability
    )
    if len(fixed_training_targets) < config.min_supported_training_targets:
        raise RuntimeError("Too few true-topology-supported training targets")
    development_targets = tuple(bundle.split_targets("validation"))
    if not development_targets:
        raise RuntimeError("No existing validation targets for disclosed reuse")
    perturbed_mean_root = output_root / "training_perturbed_mean_baseline"
    perturbed_mean_responses, perturbed_mean_manifest = (
        compile_training_perturbed_mean_baseline(
            bundle, fixed_training_targets, perturbed_mean_root
        )
    )
    provenance = dict(base_provenance)
    provenance.pop("digest", None)
    provenance.update(
        true_topology_sha256=topology_digest(true_priors),
        supplied_control_priors=control_lock,
        matched_control_generation_audit=validated_control_audit,
        training_only_perturbed_mean_baseline={
            "manifest": _artifact(
                perturbed_mean_root / "training_perturbed_mean_manifest.json",
                "training-only perturbed-mean manifest",
            ),
            "numeric": _artifact(
                perturbed_mean_root / "training_perturbed_mean_response.npz",
                "training-only perturbed-mean response",
            ),
            "lineage_digest": perturbed_mean_manifest["lineage_digest"],
            "validation_values_used": False,
            "test_values_materialized": False,
        },
    )
    provenance["digest"] = _canonical_digest(provenance)

    conditions = [("true_null_aware_routes", true_priors)] + [
        (record["name"], prior)
        for record, prior in zip(control_lock, control_priors)
    ]
    config_json = json.loads(json.dumps(asdict(config)))
    final_path = output_root / "wld_v56_null_aware_development_report.json"
    if final_path.is_file():
        existing = json.loads(final_path.read_text())
        _verify_completed_report(
            existing,
            provenance_digest=provenance["digest"],
            config_json=config_json,
            expected_conditions=[name for name, _ in conditions],
            development_targets=development_targets,
        )
        print("PASS: restored completed WLD v5.6 development", flush=True)
        return existing

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    perturbed_mean_reports = {
        int(seed): evaluate_perturbed_mean_baseline(
            bundle,
            development_targets,
            perturbed_mean_responses,
            config,
            seed=int(seed) + 900_001,
            device=resolved_device,
        )
        for seed in config.seeds
    }
    reports: Dict[str, Dict[int, Mapping[str, object]]] = {
        name: {} for name, _ in conditions
    }
    condition_index: Dict[str, Dict[str, object]] = {
        name: {} for name, _ in conditions
    }
    frozen_reports: Dict[int, Dict[str, Mapping[str, object]]] = {}
    for condition, priors in conditions:
        for seed_value in config.seeds:
            seed = int(seed_value)
            model, condition_report = _fit_condition(
                condition,
                seed,
                priors,
                prior_root,
                foundation_checkpoint,
                bundle,
                fixed_training_targets,
                development_targets,
                output_root,
                config,
                provenance["digest"],
                resolved_device,
            )
            reports[condition][seed] = condition_report
            condition_path = output_root / condition / f"seed_{seed}" / "condition_report.json"
            checkpoint_path = output_root / condition / f"seed_{seed}" / "best_model.pt"
            condition_index[condition][str(seed)] = {
                "report": str(condition_path),
                "report_sha256": sha256_file(condition_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "topology_sha256": topology_digest(priors),
                "support_topology_sha256": _support_topology_digest(priors),
            }
            if condition == "true_null_aware_routes":
                frozen_reports[seed] = {
                    "tf_removed": evaluate_model(
                        model,
                        bundle,
                        development_targets,
                        config,
                        seed=seed + 900_001,
                        overrides=BranchOverrides(tf_scale=0.0),
                    ),
                    "complex_removed": evaluate_model(
                        model,
                        bundle,
                        development_targets,
                        config,
                        seed=seed + 900_001,
                        overrides=BranchOverrides(complex_scale=0.0),
                    ),
                    "all_routes_removed": evaluate_model(
                        model,
                        bundle,
                        development_targets,
                        config,
                        seed=seed + 900_001,
                        overrides=BranchOverrides(tf_scale=0.0, complex_scale=0.0),
                    ),
                }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    paired: List[Dict[str, object]] = []
    true_reports = reports["true_null_aware_routes"]
    control_names = [name for name, _prior in conditions if name != "true_null_aware_routes"]
    for seed in map(int, config.seeds):
        true = _target_map(true_reports[seed])
        controls = [_target_map(reports[name][seed]) for name in control_names]
        perturbed_mean = {
            str(row["target"]): row
            for row in perturbed_mean_reports[seed]["per_target"]
        }
        frozen_tf = {
            str(row["target"]): row
            for row in frozen_reports[seed]["tf_removed"]["per_target"]
        }
        frozen_complex = {
            str(row["target"]): row
            for row in frozen_reports[seed]["complex_removed"]["per_target"]
        }
        frozen_all = {
            str(row["target"]): row
            for row in frozen_reports[seed]["all_routes_removed"]["per_target"]
        }
        expected = set(development_targets)
        if any(
            set(mapping) != expected
            for mapping in [
                true,
                perturbed_mean,
                frozen_tf,
                frozen_complex,
                frozen_all,
                *controls,
            ]
        ):
            raise RuntimeError("Ragged reused-development target grid")
        for target in sorted(expected):
            row = true[target]
            true_loss = float(row["model_swd"])
            paired.append(
                {
                    "target": target,
                    "seed": seed,
                    "route_supported": bool(row["route_supported"]),
                    "true_swd": true_loss,
                    "persistence_swd": float(row["persistence_swd"]),
                    "persistence_minus_true": float(row["persistence_swd"]) - true_loss,
                    "perturbed_mean_swd": float(perturbed_mean[target]["model_swd"]),
                    "perturbed_mean_minus_true": float(
                        perturbed_mean[target]["model_swd"]
                    ) - true_loss,
                    "control_mean_swd": float(
                        np.mean([float(mapping[target]["model_swd"]) for mapping in controls])
                    ),
                    "control_mean_minus_true": float(
                        np.mean([float(mapping[target]["model_swd"]) for mapping in controls])
                    ) - true_loss,
                    "frozen_tf_minus_true": float(frozen_tf[target]["model_swd"]) - true_loss,
                    "frozen_complex_minus_true": float(frozen_complex[target]["model_swd"]) - true_loss,
                    "frozen_all_minus_true": float(frozen_all[target]["model_swd"]) - true_loss,
                    "response_nrmse": float(row["response_nrmse"]),
                    "response_cosine": float(row["response_cosine"]),
                }
            )

    descriptive = {
        key: _descriptive_effect(paired, key)
        for key in (
            "persistence_minus_true",
            "perturbed_mean_minus_true",
            "control_mean_minus_true",
            "frozen_tf_minus_true",
            "frozen_complex_minus_true",
            "frozen_all_minus_true",
        )
    }
    per_control = []
    for name in control_names:
        effects = []
        for seed in map(int, config.seeds):
            true = _target_map(true_reports[seed])
            control = _target_map(reports[name][seed])
            for target in development_targets:
                effects.append(
                    float(control[target]["model_swd"]) - float(true[target]["model_swd"])
                )
        per_control.append(
            {
                "condition": name,
                "mean_control_minus_true": float(np.mean(effects)),
                "median_control_minus_true": float(np.median(effects)),
                "target_seed_rows": len(effects),
            }
        )

    mean_response_nrmse = float(
        np.mean([float(row["response_nrmse"]) for row in paired])
    )
    mean_response_cosine = float(
        np.mean([float(row["response_cosine"]) for row in paired])
    )
    development_checks = {
        "true_beats_persistence_descriptively": bool(
            descriptive["persistence_minus_true"]["mean"] > 0.0
        ),
        "true_beats_training_perturbed_mean_descriptively": bool(
            descriptive["perturbed_mean_minus_true"]["mean"] > 0.0
        ),
        "true_beats_matched_controls_descriptively": bool(
            descriptive["control_mean_minus_true"]["mean"] > 0.0
        ),
        "fitted_routes_are_used_descriptively": bool(
            descriptive["frozen_all_minus_true"]["mean"] > 0.0
        ),
        "response_nrmse_below_persistence": bool(mean_response_nrmse < 1.0),
        "response_cosine_positive": bool(mean_response_cosine > 0.0),
        "mean_response_nrmse": mean_response_nrmse,
        "mean_response_cosine": mean_response_cosine,
        "inference": False,
        "validation_targets_reused": True,
    }
    development_checks["eligible_to_freeze_new_confirmation_plan"] = bool(
        all(
            value is True
            for name, value in development_checks.items()
            if name
            not in {
                "inference",
                "validation_targets_reused",
                "mean_response_nrmse",
                "mean_response_cosine",
            }
        )
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "GSE161002 null-aware transient chromatin response development; all "
            "previously inspected v5.5 validation targets are reused"
        ),
        "device": str(resolved_device),
        "config": config_json,
        "provenance": provenance,
        "prior_audit": prior_audit,
        "architecture": architecture_contract(
            WLDNullAwareChromatinTwin(
                _load_foundation(prior_root, foundation_checkpoint, resolved_device),
                _device_priors(true_priors, resolved_device),
            )
        ),
        "fixed_true_topology_training_targets": list(fixed_training_targets),
        "reused_development_targets": list(development_targets),
        "control_prior_lock": control_lock,
        "matched_control_generation_audit": validated_control_audit,
        "training_only_perturbed_mean_baseline": {
            "manifest": perturbed_mean_manifest,
            "evaluations_by_seed": perturbed_mean_reports,
            "validation_values_used_for_construction": False,
            "test_values_materialized": False,
            "reference_doi": "10.1038/s41587-025-02777-8",
        },
        "conditions": condition_index,
        "paired_reused_development_metrics": paired,
        "descriptive_effects": descriptive,
        "per_control_descriptive_effects": per_control,
        "frozen_true_model_evaluations": frozen_reports,
        "fitted_true_branch_gates_by_seed": {
            str(seed): true_reports[int(seed)]["final_effective_branch_gates"]
            for seed in config.seeds
        },
        "development_checks": development_checks,
        "claims": {
            "development_only": True,
            "validation_targets_previously_used_in_v55": True,
            "all_existing_validation_targets_evaluated": True,
            "untouched_audit_inference": False,
            "confidence_interval_claim": False,
            "p_value_claim": False,
            "perturbed_mean_baseline_training_only": True,
            "test_targets_materialized": False,
            "test_targets_evaluated": False,
            "external_subject_study_evaluated": False,
            "ode_time_scale_identified": False,
            "fixed_point_claim": False,
            "basin_claim": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
        },
        "interpretation_rule": (
            "These reused-development summaries may choose the next architecture but may not "
            "support confirmatory topology, digital-twin, or attractor language. The sealed "
            "v5.3 test targets remain unopened until a new analysis plan is frozen."
        ),
    }
    atomic_json(final_path, report)
    return report


NullAwareTrainingConfig = TwinTrainingConfig


def run_nullaware_development(
    prior_root: Path,
    foundation_checkpoint: Path,
    bundle_root: Path,
    route_root: Path,
    module_root: Path,
    output_root: Path,
    config: NullAwareTrainingConfig,
    *,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Build locked split-stratified controls, then run reused development.

    The matched-control builder consumes only the immutable target-name split.
    It never loads a test cell or outcome.  Joint TF/complex route profiles are
    permuted independently within train, validation and test name strata.
    """

    from wld_v56_topology_controls import build_matched_control_priors

    config.validate()
    prior_root = Path(prior_root)
    bundle_root = Path(bundle_root)
    route_root = Path(route_root)
    module_root = Path(module_root)
    true_priors, _atlas, _prior_audit = load_twin_priors(
        prior_root, route_root, module_root
    )
    route_vocab = json.loads((route_root / "route_vocab.json").read_text())
    regulators = tuple(str(value).upper() for value in route_vocab["regulators"])
    if len(regulators) != true_priors.num_regulators or len(set(regulators)) != len(regulators):
        raise RuntimeError("Route vocabulary and true priors disagree on regulators")
    split_payload = json.loads((bundle_root / "whole_target_split.json").read_text())
    raw_rosters = split_payload.get("targets")
    if not isinstance(raw_rosters, Mapping) or set(raw_rosters) != {
        "train",
        "validation",
        "test",
    }:
        raise RuntimeError("Frozen whole-target split has a changed schema")
    target_to_split: Dict[str, str] = {}
    for split in ("train", "validation", "test"):
        values = raw_rosters[split]
        if not isinstance(values, list):
            raise RuntimeError(f"Frozen {split} target roster is not a list")
        for value in values:
            target = str(value).upper()
            if not target or target in target_to_split:
                raise RuntimeError("Frozen whole-target rosters are empty or overlapping")
            target_to_split[target] = split
    if set(target_to_split) != set(regulators):
        raise RuntimeError("Frozen split names and regulator vocabulary do not match")
    aligned_split_labels = tuple(target_to_split[target] for target in regulators)
    controls, control_audit = build_matched_control_priors(
        true_priors,
        config.control_replicates,
        config.control_seed,
        strata=aligned_split_labels,
    )
    enriched_audit = {
        **dict(control_audit),
        "strata_source": "whole_target_split.json target names only",
        "aligned_regulator_vocab_sha256": hashlib.sha256(
            json.dumps(regulators, separators=(",", ":")).encode()
        ).hexdigest(),
        "test_outcomes_or_observations_read": False,
        "test_rows_materialized": False,
    }
    return run_twin_development(
        prior_root,
        foundation_checkpoint,
        bundle_root,
        route_root,
        module_root,
        output_root,
        config,
        control_priors=controls,
        control_generation_audit=enriched_audit,
        device=device,
    )


run_chromatin_twin_development = run_nullaware_development


__all__ = [
    "TwinTrainingConfig",
    "NullAwareTrainingConfig",
    "build_provenance_lock",
    "validate_control_priors",
    "response_focused_loss",
    "compile_training_perturbed_mean_baseline",
    "evaluate_model",
    "evaluate_perturbed_mean_baseline",
    "run_twin_development",
    "run_nullaware_development",
    "run_chromatin_twin_development",
]
