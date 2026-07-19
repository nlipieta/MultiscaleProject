"""Leakage-safe temporal training for the hard-constrained WLD v3 model.

The public PBMC snapshot is not used here.  This module consumes experiments
with declared biological groups, measured time intervals, time-zero ATAC and
cues, and future RNA (plus optional future ATAC).  Destructive single-cell
time courses default to distributional alignment; paired loss is allowed only
when explicit pair/lineage identifiers are supplied.

The test groups remain sealed during training and model selection.  They are
evaluated once after restoring the validation-selected checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from wld_circuit_dynamics_v3 import (
    CircuitDynamicsModel,
    MultiscaleCircuitPriors,
    temporal_circuit_objective,
    temporal_leakage_audit,
)


Tensor = torch.Tensor
SCHEMA_VERSION = 1
PRIOR_KEYS = (
    "peak_to_gene",
    "peak_tf_motif",
    "tf_gene_support",
    "circuit_tf_tf",
    "tf_gene_index",
    "signal_signal",
    "signal_tf",
    "cue_signal",
    "tf_peak_effect",
)
SAFE_CONDITION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class TransitionSpec:
    transition_id: str
    group_id: str
    horizon: float
    terminal: bool = False


@dataclass(frozen=True)
class TemporalTrainingConfig:
    epochs: int = 100
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    integration_steps: int = 12
    max_initial_cells: int = 128
    max_target_cells: int = 128
    projections: int = 32
    quantiles: int = 32
    validation_every: int = 5
    patience: int = 8
    rna_swd_weight: float = 1.0
    rna_mean_weight: float = 0.5
    rna_variance_weight: float = 0.1
    accessibility_swd_weight: float = 0.5
    terminal_velocity_weight: float = 0.1
    gain_weight: float = 1e-4
    fixed_point_iterations: int = 500
    fixed_point_learning_rate: float = 1e-2
    fixed_point_tolerance: float = 1e-6
    jacobian_max_dimension: int = 256
    basin_trials: int = 32
    basin_perturbation_scale: float = 0.05
    basin_horizon: float = 10.0
    basin_steps: int = 200
    basin_tolerance: float = 0.05
    seed: int = 42

    def validate(self) -> None:
        positive_ints = {
            "epochs": self.epochs,
            "integration_steps": self.integration_steps,
            "max_initial_cells": self.max_initial_cells,
            "max_target_cells": self.max_target_cells,
            "projections": self.projections,
            "quantiles": self.quantiles,
            "validation_every": self.validation_every,
            "patience": self.patience,
            "fixed_point_iterations": self.fixed_point_iterations,
            "jacobian_max_dimension": self.jacobian_max_dimension,
            "basin_trials": self.basin_trials,
            "basin_steps": self.basin_steps,
        }
        bad = [name for name, value in positive_ints.items() if value < 1]
        if bad:
            raise ValueError(f"Training configuration must be positive: {bad}")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        diagnostic_values = {
            "fixed_point_learning_rate": self.fixed_point_learning_rate,
            "fixed_point_tolerance": self.fixed_point_tolerance,
            "basin_perturbation_scale": self.basin_perturbation_scale,
            "basin_horizon": self.basin_horizon,
            "basin_tolerance": self.basin_tolerance,
        }
        if any(value <= 0 or not math.isfinite(value) for value in diagnostic_values.values()):
            raise ValueError("Attractor diagnostic values must be finite and positive.")
        weights = (
            self.rna_swd_weight,
            self.rna_mean_weight,
            self.rna_variance_weight,
            self.accessibility_swd_weight,
            self.terminal_velocity_weight,
            self.gain_weight,
        )
        if any(value < 0 or not math.isfinite(value) for value in weights):
            raise ValueError("Loss weights must be finite and non-negative.")


@dataclass
class TemporalCohort:
    priors: MultiscaleCircuitPriors
    alignment_mode: str
    initial_atac: Tensor
    initial_cues: Tensor
    initial_transition: np.ndarray
    target_rna: Tensor
    target_transition: np.ndarray
    transitions: Dict[str, TransitionSpec]
    split_groups: Dict[str, Tuple[str, ...]]
    priors_fit_groups: Tuple[str, ...]
    initial_feature_names: Tuple[str, ...]
    cue_names: Tuple[str, ...]
    cue_provenance: Tuple[Mapping[str, object], ...]
    control_priors: Dict[str, MultiscaleCircuitPriors]
    initial_cue_mask: Optional[Tensor] = None
    initial_rna: Optional[Tensor] = None
    target_atac: Optional[Tensor] = None
    target_derivative: Optional[Tensor] = None
    initial_pair_id: Optional[np.ndarray] = None
    target_pair_id: Optional[np.ndarray] = None

    def transition_ids(self, split: str) -> List[str]:
        if split not in self.split_groups:
            raise KeyError(f"Unknown split {split!r}.")
        groups = set(self.split_groups[split])
        return sorted(
            transition_id
            for transition_id, spec in self.transitions.items()
            if spec.group_id in groups
        )

    def initial_indices(self, transition_id: str) -> np.ndarray:
        return np.flatnonzero(self.initial_transition == transition_id)

    def target_indices(self, transition_id: str) -> np.ndarray:
        return np.flatnonzero(self.target_transition == transition_id)

    def validate(self) -> None:
        self.priors.validate()
        reserved_controls = {"true_circuit", "no_circuit"}.intersection(
            self.control_priors
        )
        if reserved_controls:
            raise ValueError(
                f"Custom control names are reserved: {sorted(reserved_controls)}"
            )
        unsafe_controls = [
            name for name in self.control_priors if SAFE_CONDITION.fullmatch(name) is None
        ]
        if unsafe_controls:
            raise ValueError(f"Unsafe custom control names: {sorted(unsafe_controls)}")
        reference_dimensions = (
            self.priors.num_peaks,
            self.priors.num_genes,
            self.priors.num_tfs,
            self.priors.num_signals,
            self.priors.num_cues,
        )
        for name, control in self.control_priors.items():
            control.validate()
            control_dimensions = (
                control.num_peaks,
                control.num_genes,
                control.num_tfs,
                control.num_signals,
                control.num_cues,
            )
            if control_dimensions != reference_dimensions:
                raise ValueError(
                    f"Custom control prior {name!r} changes model dimensions."
                )
        if self.alignment_mode not in {"distribution", "paired"}:
            raise ValueError("alignment_mode must be 'distribution' or 'paired'.")
        if self.initial_atac.ndim != 2 or self.initial_atac.shape[1] != self.priors.num_peaks:
            raise ValueError("initial_atac must have shape [initial_cells, peaks].")
        if self.initial_cues.shape != (
            self.initial_atac.shape[0],
            self.priors.num_cues,
        ):
            raise ValueError("initial_cues has incompatible dimensions.")
        if len(self.cue_names) != self.priors.num_cues:
            raise ValueError("cue_names must contain one unique name per cue.")
        if len(set(self.cue_names)) != len(self.cue_names):
            raise ValueError("cue_names must be unique.")
        if len(self.cue_provenance) != self.priors.num_cues:
            raise ValueError("cue_provenance must contain one record per cue.")
        for index, record in enumerate(self.cue_provenance):
            if str(record.get("name", "")) != self.cue_names[index]:
                raise ValueError("cue_provenance names must match cue_names in order.")
            if str(record.get("measurement_level", "")) not in {
                "cell",
                "sample",
                "subject",
                "experiment",
            }:
                raise ValueError(
                    "Every cue must declare cell, sample, subject, or experiment "
                    "measurement_level provenance."
                )
        if self.initial_cue_mask is not None:
            if self.initial_cue_mask.shape != self.initial_cues.shape:
                raise ValueError("initial_cue_mask must match initial_cues.")
            if not torch.isfinite(self.initial_cue_mask).all():
                raise ValueError("initial_cue_mask contains non-finite values.")
            if bool(
                ((self.initial_cue_mask != 0) & (self.initial_cue_mask != 1)).any()
            ):
                raise ValueError("initial_cue_mask must contain only zero or one.")
        if self.target_rna.ndim != 2 or self.target_rna.shape[1] != self.priors.num_genes:
            raise ValueError("target_rna must have shape [target_cells, genes].")
        if self.initial_transition.shape != (self.initial_atac.shape[0],):
            raise ValueError("initial_transition must have one ID per initial cell.")
        if self.target_transition.shape != (self.target_rna.shape[0],):
            raise ValueError("target_transition must have one ID per target cell.")
        if self.initial_rna is not None and self.initial_rna.shape != (
            self.initial_atac.shape[0],
            self.priors.num_genes,
        ):
            raise ValueError("initial_rna has incompatible dimensions.")
        if self.target_atac is not None and self.target_atac.shape != (
            self.target_rna.shape[0],
            self.priors.num_peaks,
        ):
            raise ValueError("target_atac has incompatible dimensions.")
        if self.target_derivative is not None and self.target_derivative.shape != (
            self.target_rna.shape[0],
            self.priors.num_signals
            + self.priors.num_tfs
            + self.priors.num_peaks
            + self.priors.num_genes,
        ):
            raise ValueError("target_derivative has incompatible dimensions.")
        if self.alignment_mode == "distribution" and self.target_derivative is not None:
            raise ValueError("target_derivative is supported only in paired mode.")

        tensors = {
            "initial_atac": self.initial_atac,
            "initial_cues": self.initial_cues,
            "target_rna": self.target_rna,
        }
        optional_tensors = {
            "initial_cue_mask": self.initial_cue_mask,
            "initial_rna": self.initial_rna,
            "target_atac": self.target_atac,
            "target_derivative": self.target_derivative,
        }
        tensors.update({key: value for key, value in optional_tensors.items() if value is not None})
        for name, value in tensors.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
        if bool(((self.initial_atac < 0) | (self.initial_atac > 1)).any()):
            raise ValueError("initial_atac must be normalized to [0, 1].")
        if self.target_atac is not None and bool(
            ((self.target_atac < 0) | (self.target_atac > 1)).any()
        ):
            raise ValueError("target_atac must be normalized to [0, 1].")
        if bool((self.initial_cues < 0).any()) or bool((self.target_rna < 0).any()):
            raise ValueError("Cues and RNA measurements must be non-negative.")
        if self.initial_rna is not None and bool((self.initial_rna < 0).any()):
            raise ValueError("initial_rna must be non-negative.")

        declared = set(self.transitions)
        observed_initial = set(map(str, np.unique(self.initial_transition)))
        observed_target = set(map(str, np.unique(self.target_transition)))
        if observed_initial != declared or observed_target != declared:
            raise ValueError(
                "Every declared transition must have initial and target cells, "
                "with no undeclared transition IDs."
            )
        for transition_id, spec in self.transitions.items():
            if spec.transition_id != transition_id:
                raise ValueError("Transition dictionary keys must match transition_id.")
            if not math.isfinite(spec.horizon) or spec.horizon <= 0:
                raise ValueError("Every transition horizon must be finite and positive.")

        required_splits = {"train", "validation", "test"}
        if set(self.split_groups) != required_splits:
            raise ValueError("split_groups must contain train, validation, and test.")
        split_sets = {name: set(values) for name, values in self.split_groups.items()}
        if any(not values for values in split_sets.values()):
            raise ValueError("Every split must contain at least one biological group.")
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = split_sets[left].intersection(split_sets[right])
            if overlap:
                raise ValueError(f"{left}/{right} biological groups overlap: {sorted(overlap)}")
        transition_groups = {spec.group_id for spec in self.transitions.values()}
        if transition_groups != set().union(*split_sets.values()):
            raise ValueError("Declared split groups must exactly cover transition groups.")
        if not self.priors_fit_groups or not set(self.priors_fit_groups).issubset(
            split_sets["train"]
        ):
            raise ValueError("priors_fit_groups must be a non-empty subset of train groups.")

        temporal_leakage_audit(
            self.split_groups["train"],
            self.split_groups["test"],
            self.initial_feature_names,
            initial_time=0.0,
            target_time=min(spec.horizon for spec in self.transitions.values()),
            uses_initial_rna=self.initial_rna is not None,
            initial_rna_time=0.0 if self.initial_rna is not None else None,
        )
        temporal_leakage_audit(
            self.split_groups["train"],
            self.split_groups["validation"],
            self.initial_feature_names,
            initial_time=0.0,
            target_time=min(spec.horizon for spec in self.transitions.values()),
            uses_initial_rna=self.initial_rna is not None,
            initial_rna_time=0.0 if self.initial_rna is not None else None,
        )

        if self.alignment_mode == "paired":
            if self.initial_pair_id is None or self.target_pair_id is None:
                raise ValueError("Paired mode requires initial_pair_id and target_pair_id.")
            if self.initial_pair_id.shape != (self.initial_atac.shape[0],):
                raise ValueError("initial_pair_id must have one value per initial cell.")
            if self.target_pair_id.shape != (self.target_rna.shape[0],):
                raise ValueError("target_pair_id must have one value per target cell.")
            for transition_id in sorted(self.transitions):
                initial_pairs = self.initial_pair_id[self.initial_indices(transition_id)]
                target_pairs = self.target_pair_id[self.target_indices(transition_id)]
                if len(set(map(str, initial_pairs))) != len(initial_pairs):
                    raise ValueError("Initial pair IDs must be unique within a transition.")
                if set(map(str, initial_pairs)) != set(map(str, target_pairs)):
                    raise ValueError("Paired initial and target IDs do not match.")
        elif self.initial_pair_id is not None or self.target_pair_id is not None:
            raise ValueError("Pair IDs must be omitted in distribution mode.")


def _tensor(array: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype).contiguous()


def load_priors(path: Path) -> MultiscaleCircuitPriors:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in PRIOR_KEYS if key not in archive]
        if missing:
            raise KeyError(f"Prior archive is missing: {missing}")
        values = {key: archive[key] for key in PRIOR_KEYS}
    priors = MultiscaleCircuitPriors(
        peak_to_gene=_tensor(values["peak_to_gene"]),
        peak_tf_motif=_tensor(values["peak_tf_motif"]),
        tf_gene_support=_tensor(values["tf_gene_support"]),
        circuit_tf_tf=_tensor(values["circuit_tf_tf"]),
        tf_gene_index=_tensor(values["tf_gene_index"], torch.long),
        signal_signal=_tensor(values["signal_signal"]),
        signal_tf=_tensor(values["signal_tf"]),
        cue_signal=_tensor(values["cue_signal"]),
        tf_peak_effect=_tensor(values["tf_peak_effect"]),
    )
    priors.validate()
    return priors


def save_priors(path: Path, priors: MultiscaleCircuitPriors) -> None:
    priors.validate()
    np.savez_compressed(
        path,
        **{
            key: getattr(priors, key).detach().cpu().numpy()
            for key in PRIOR_KEYS
        },
    )


def _optional_tensor(archive: Mapping[str, np.ndarray], key: str) -> Optional[Tensor]:
    return _tensor(archive[key]) if key in archive else None


def _optional_strings(
    archive: Mapping[str, np.ndarray], key: str
) -> Optional[np.ndarray]:
    if key not in archive:
        return None
    return np.asarray(archive[key]).astype(str)


def load_temporal_cohort(root: Path) -> TemporalCohort:
    root = Path(root)
    manifest_path = root / "manifest.json"
    observation_path = root / "observations.npz"
    prior_path = root / "priors.npz"
    for path in (manifest_path, observation_path, prior_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing temporal dataset file: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version {SCHEMA_VERSION}.")
    transitions = {}
    for item in manifest.get("transitions", []):
        spec = TransitionSpec(
            transition_id=str(item["transition_id"]),
            group_id=str(item["group_id"]),
            horizon=float(item["horizon"]),
            terminal=bool(item.get("terminal", False)),
        )
        if spec.transition_id in transitions:
            raise ValueError(f"Duplicate transition ID: {spec.transition_id}")
        transitions[spec.transition_id] = spec
    if not transitions:
        raise ValueError("Manifest must declare at least one transition.")

    control_priors = {}
    for name, relative_path in manifest.get("control_prior_archives", {}).items():
        control_name = str(name)
        candidate = (root / str(relative_path)).resolve()
        if root.resolve() not in candidate.parents:
            raise ValueError("Control prior paths must remain inside the dataset root.")
        control_priors[control_name] = load_priors(candidate)

    priors = load_priors(prior_path)
    cue_names = tuple(
        map(
            str,
            manifest.get(
                "cue_names",
                [f"cue_{index}" for index in range(priors.num_cues)],
            ),
        )
    )
    cue_provenance = tuple(
        dict(record)
        for record in manifest.get(
            "cue_provenance",
            [
                {
                    "name": name,
                    "measurement_level": "experiment",
                    "source": "legacy schema-v1 manifest",
                }
                for name in cue_names
            ],
        )
    )

    with np.load(observation_path, allow_pickle=False) as archive:
        required = {
            "initial_atac",
            "initial_cues",
            "initial_transition",
            "target_rna",
            "target_transition",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Observation archive is missing: {missing}")
        cohort = TemporalCohort(
            priors=priors,
            alignment_mode=str(manifest.get("alignment_mode", "distribution")),
            initial_atac=_tensor(archive["initial_atac"]),
            initial_cues=_tensor(archive["initial_cues"]),
            initial_transition=np.asarray(archive["initial_transition"]).astype(str),
            target_rna=_tensor(archive["target_rna"]),
            target_transition=np.asarray(archive["target_transition"]).astype(str),
            transitions=transitions,
            split_groups={
                name: tuple(map(str, manifest["split_groups"][name]))
                for name in ("train", "validation", "test")
            },
            priors_fit_groups=tuple(map(str, manifest["priors_fit_groups"])),
            initial_feature_names=tuple(map(str, manifest["initial_feature_names"])),
            cue_names=cue_names,
            cue_provenance=cue_provenance,
            control_priors=control_priors,
            initial_cue_mask=_optional_tensor(archive, "initial_cue_mask"),
            initial_rna=_optional_tensor(archive, "initial_rna"),
            target_atac=_optional_tensor(archive, "target_atac"),
            target_derivative=_optional_tensor(archive, "target_derivative"),
            initial_pair_id=_optional_strings(archive, "initial_pair_id"),
            target_pair_id=_optional_strings(archive, "target_pair_id"),
        )
    cohort.validate()
    return cohort


def _projection_matrix(
    dimension: int, projections: int, seed: int, reference: Tensor
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dimension, projections, generator=generator)
    matrix = matrix / torch.linalg.vector_norm(matrix, dim=0, keepdim=True).clamp_min(1e-8)
    return matrix.to(device=reference.device, dtype=reference.dtype)


def sliced_wasserstein(
    predicted: Tensor,
    observed: Tensor,
    projections: int,
    quantiles: int,
    seed: int,
) -> Tensor:
    if predicted.ndim != 2 or observed.ndim != 2:
        raise ValueError("Sliced Wasserstein inputs must be rank two.")
    if predicted.shape[1] != observed.shape[1]:
        raise ValueError("Sliced Wasserstein feature dimensions do not match.")
    projection = _projection_matrix(predicted.shape[1], projections, seed, predicted)
    predicted_projection = predicted @ projection
    observed_projection = observed @ projection
    q = torch.linspace(
        0.0,
        1.0,
        quantiles,
        dtype=predicted.dtype,
        device=predicted.device,
    )
    predicted_quantiles = torch.quantile(predicted_projection, q, dim=0)
    observed_quantiles = torch.quantile(observed_projection, q, dim=0)
    return F.mse_loss(predicted_quantiles, observed_quantiles)


def _edge_gain_regularization(model: CircuitDynamicsModel) -> Tensor:
    layers = (
        model.field.signal_recurrent,
        model.field.signal_to_tf,
        model.field.tf_circuit,
        model.field.tf_to_peak,
        model.field.tf_to_gene,
    )
    gains = [layer.effective_gain() for layer in layers if layer.num_edges]
    if not gains:
        return next(model.parameters()).new_zeros(())
    return torch.cat(gains).mean()


def distribution_objective(
    model: CircuitDynamicsModel,
    output: Dict[str, Tensor],
    target_rna: Tensor,
    target_atac: Optional[Tensor],
    terminal: bool,
    config: TemporalTrainingConfig,
) -> Dict[str, Tensor]:
    predicted_rna = torch.log1p(output["rna_t"].clamp_min(0.0))
    observed_rna = torch.log1p(target_rna.clamp_min(0.0))
    rna_swd = sliced_wasserstein(
        predicted_rna,
        observed_rna,
        config.projections,
        config.quantiles,
        config.seed,
    )
    rna_mean = F.mse_loss(predicted_rna.mean(0), observed_rna.mean(0))
    rna_variance = F.mse_loss(
        predicted_rna.var(0, unbiased=False),
        observed_rna.var(0, unbiased=False),
    )
    zero = rna_swd.new_zeros(())
    accessibility_swd = zero
    if target_atac is not None:
        accessibility_swd = sliced_wasserstein(
            output["accessibility_t"].clamp(0.0, 1.0),
            target_atac,
            config.projections,
            config.quantiles,
            config.seed + 1,
        )
    terminal_velocity = (
        output["terminal_velocity"].square().mean() if terminal else zero
    )
    edge_gain = _edge_gain_regularization(model)
    total = (
        config.rna_swd_weight * rna_swd
        + config.rna_mean_weight * rna_mean
        + config.rna_variance_weight * rna_variance
        + config.accessibility_swd_weight * accessibility_swd
        + config.terminal_velocity_weight * terminal_velocity
        + config.gain_weight * edge_gain
    )
    return {
        "total": total,
        "rna_swd": rna_swd,
        "rna_mean": rna_mean,
        "rna_variance": rna_variance,
        "accessibility_swd": accessibility_swd,
        "terminal_velocity": terminal_velocity,
        "edge_gain": edge_gain,
    }


def _stable_string_seed(value: str, base_seed: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + base_seed) % (2**31 - 1)


def _sample_indices(
    indices: np.ndarray, maximum: int, seed: int
) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def _paired_target_indices(
    cohort: TemporalCohort, transition_id: str, initial_indices: np.ndarray
) -> np.ndarray:
    if cohort.initial_pair_id is None or cohort.target_pair_id is None:
        raise ValueError("Paired target lookup requires pair IDs.")
    target_indices = cohort.target_indices(transition_id)
    lookup = {
        str(cohort.target_pair_id[index]): int(index) for index in target_indices
    }
    return np.asarray(
        [lookup[str(cohort.initial_pair_id[index])] for index in initial_indices],
        dtype=int,
    )


def _transition_loss(
    model: CircuitDynamicsModel,
    cohort: TemporalCohort,
    transition_id: str,
    config: TemporalTrainingConfig,
    device: torch.device,
    epoch: int,
    sample: bool,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    spec = cohort.transitions[transition_id]
    seed = _stable_string_seed(transition_id, config.seed + 1009 * epoch)
    initial_indices = cohort.initial_indices(transition_id)
    if sample:
        initial_indices = _sample_indices(
            initial_indices, config.max_initial_cells, seed
        )
    atac = cohort.initial_atac[initial_indices].to(device)
    cues = cohort.initial_cues[initial_indices].to(device)
    if cohort.initial_cue_mask is not None:
        cues = cues * cohort.initial_cue_mask[initial_indices].to(device)
    initial_rna = (
        cohort.initial_rna[initial_indices].to(device)
        if cohort.initial_rna is not None
        else None
    )
    output = model(
        atac,
        cues,
        horizon=spec.horizon,
        steps=config.integration_steps,
        initial_rna=initial_rna,
    )

    if cohort.alignment_mode == "paired":
        target_indices = _paired_target_indices(
            cohort, transition_id, initial_indices
        )
        target_rna = cohort.target_rna[target_indices].to(device)
        target_atac = (
            cohort.target_atac[target_indices].to(device)
            if cohort.target_atac is not None
            else None
        )
        target_derivative = (
            cohort.target_derivative[target_indices].to(device)
            if cohort.target_derivative is not None
            else None
        )
        losses = temporal_circuit_objective(
            output,
            target_rna,
            target_accessibility=target_atac,
            observed_derivative=target_derivative,
            terminal_mask=torch.full(
                (len(initial_indices),), spec.terminal, dtype=torch.bool, device=device
            ),
            model=model,
            rna_weight=config.rna_swd_weight,
            accessibility_weight=config.accessibility_swd_weight,
            derivative_weight=0.25,
            terminal_weight=config.terminal_velocity_weight,
            gain_weight=config.gain_weight,
        )
    else:
        target_indices = cohort.target_indices(transition_id)
        if sample:
            target_indices = _sample_indices(
                target_indices, config.max_target_cells, seed + 17
            )
        target_rna = cohort.target_rna[target_indices].to(device)
        target_atac = (
            cohort.target_atac[target_indices].to(device)
            if cohort.target_atac is not None
            else None
        )
        losses = distribution_objective(
            model,
            output,
            target_rna,
            target_atac,
            spec.terminal,
            config,
        )
    return losses["total"], losses, output


def _condition_priors(
    priors: MultiscaleCircuitPriors,
    condition: str,
    control_priors: Mapping[str, MultiscaleCircuitPriors],
) -> MultiscaleCircuitPriors:
    if condition == "true_circuit":
        return priors
    if condition == "no_circuit":
        control = replace(priors, circuit_tf_tf=torch.zeros_like(priors.circuit_tf_tf))
        control.validate()
        return control
    if condition in control_priors:
        return control_priors[condition]
    raise ValueError(
        f"Unknown condition {condition!r}. Available custom controls: "
        f"{sorted(control_priors)}"
    )


def _mean_split_loss(
    model: CircuitDynamicsModel,
    cohort: TemporalCohort,
    split: str,
    config: TemporalTrainingConfig,
    device: torch.device,
) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for transition_id in cohort.transition_ids(split):
            total, _, _ = _transition_loss(
                model,
                cohort,
                transition_id,
                config,
                device,
                epoch=0,
                sample=False,
            )
            values.append(float(total.detach().cpu()))
    if not values:
        raise ValueError(f"Split {split!r} contains no transitions.")
    return float(np.mean(values))


def train_temporal_model(
    cohort: TemporalCohort,
    config: TemporalTrainingConfig,
    condition: str,
    device: torch.device,
) -> Tuple[CircuitDynamicsModel, Dict[str, object]]:
    cohort.validate()
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = CircuitDynamicsModel(
        _condition_priors(cohort.priors, condition, cohort.control_priors)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale_checks = 0
    history: List[Dict[str, float]] = []

    train_transitions = cohort.transition_ids("train")
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        transition_losses = []
        for transition_id in train_transitions:
            total, _, _ = _transition_loss(
                model,
                cohort,
                transition_id,
                config,
                device,
                epoch=epoch,
                sample=True,
            )
            transition_losses.append(total)
        training_loss = torch.stack(transition_losses).mean()
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if epoch == 1 or epoch % config.validation_every == 0 or epoch == config.epochs:
            validation_loss = _mean_split_loss(
                model, cohort, "validation", config, device
            )
            record = {
                "epoch": float(epoch),
                "training_loss": float(training_loss.detach().cpu()),
                "validation_loss": validation_loss,
            }
            history.append(record)
            if validation_loss < best_validation - 1e-8:
                best_validation = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= config.patience:
                    break

    model.load_state_dict(best_state)
    return model, {
        "condition": condition,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "history": history,
        "test_groups_used_during_selection": False,
    }


def _pearson(left: Tensor, right: Tensor) -> Optional[float]:
    left = left.detach().float().cpu().flatten()
    right = right.detach().float().cpu().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-12:
        return None
    return float(torch.dot(left, right) / denominator)


def _transition_metrics(
    model: CircuitDynamicsModel,
    cohort: TemporalCohort,
    transition_id: str,
    config: TemporalTrainingConfig,
    device: torch.device,
) -> Dict[str, Optional[float]]:
    spec = cohort.transitions[transition_id]
    with torch.no_grad():
        _, _, output = _transition_loss(
            model,
            cohort,
            transition_id,
            config,
            device,
            epoch=0,
            sample=False,
        )
        target_indices = cohort.target_indices(transition_id)
        target_rna = cohort.target_rna[target_indices].to(device)
        predicted_log = torch.log1p(output["rna_t"].clamp_min(0.0))
        target_log = torch.log1p(target_rna)
        metrics: Dict[str, Optional[float]] = {
            "log_rna_swd": float(
                sliced_wasserstein(
                    predicted_log,
                    target_log,
                    config.projections,
                    config.quantiles,
                    config.seed,
                ).cpu()
            ),
            "log_rna_mean_mse": float(
                F.mse_loss(predicted_log.mean(0), target_log.mean(0)).cpu()
            ),
            "log_rna_mean_pearson": _pearson(
                predicted_log.mean(0), target_log.mean(0)
            ),
            "terminal_velocity_rms": float(
                output["terminal_velocity"].square().mean().sqrt().cpu()
            ),
        }
        if cohort.alignment_mode == "paired":
            paired_targets = _paired_target_indices(
                cohort, transition_id, cohort.initial_indices(transition_id)
            )
            paired_log = torch.log1p(cohort.target_rna[paired_targets].to(device))
            metrics["paired_log_rna_mse"] = float(
                F.mse_loss(predicted_log, paired_log).cpu()
            )
            metrics["paired_log_rna_pearson"] = _pearson(predicted_log, paired_log)
        if cohort.target_atac is not None:
            target_atac = cohort.target_atac[target_indices].to(device)
            metrics["atac_swd"] = float(
                sliced_wasserstein(
                    output["accessibility_t"].clamp(0.0, 1.0),
                    target_atac,
                    config.projections,
                    config.quantiles,
                    config.seed + 1,
                ).cpu()
            )
    return metrics


def evaluate_test_groups(
    model: CircuitDynamicsModel,
    cohort: TemporalCohort,
    config: TemporalTrainingConfig,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    by_transition = {}
    grouped: Dict[str, List[Mapping[str, Optional[float]]]] = {}
    for transition_id in cohort.transition_ids("test"):
        metrics = _transition_metrics(
            model, cohort, transition_id, config, device
        )
        by_transition[transition_id] = metrics
        group = cohort.transitions[transition_id].group_id
        grouped.setdefault(group, []).append(metrics)

    by_group: Dict[str, Dict[str, Optional[float]]] = {}
    for group, records in grouped.items():
        keys = sorted(set().union(*(record.keys() for record in records)))
        aggregate = {}
        for key in keys:
            values = [record[key] for record in records if record.get(key) is not None]
            aggregate[key] = float(np.mean(values)) if values else None
        by_group[group] = aggregate

    train_target_indices = np.concatenate(
        [cohort.target_indices(tid) for tid in cohort.transition_ids("train")]
    )
    training_mean = torch.log1p(cohort.target_rna[train_target_indices]).mean(0)
    baseline = {}
    for transition_id in cohort.transition_ids("test"):
        target_mean = torch.log1p(
            cohort.target_rna[cohort.target_indices(transition_id)]
        ).mean(0)
        target_log = torch.log1p(
            cohort.target_rna[cohort.target_indices(transition_id)]
        )
        constant_prediction = training_mean.unsqueeze(0).expand_as(target_log)
        baseline[transition_id] = {
            "log_rna_swd": float(
                sliced_wasserstein(
                    constant_prediction,
                    target_log,
                    config.projections,
                    config.quantiles,
                    config.seed,
                )
            ),
            "log_rna_mean_mse": float(F.mse_loss(training_mean, target_mean)),
            "log_rna_mean_pearson": _pearson(training_mean, target_mean),
        }
    return {
        "test_evaluated_once_after_model_selection": True,
        "by_transition": by_transition,
        "by_group": by_group,
        "training_mean_baseline": baseline,
        "attractor_diagnostics": _test_attractor_diagnostics(
            model, cohort, config, device
        ),
    }


def _test_attractor_diagnostics(
    model: CircuitDynamicsModel,
    cohort: TemporalCohort,
    config: TemporalTrainingConfig,
    device: torch.device,
) -> Dict[str, object]:
    """Audit prespecified terminal test transitions after model selection."""
    terminal_ids = [
        transition_id
        for transition_id in cohort.transition_ids("test")
        if cohort.transitions[transition_id].terminal
    ]
    if not terminal_ids:
        return {
            "status": "not_applicable",
            "reason": "No held-out test transition was prespecified as terminal.",
            "unconditional_attractor_claim": False,
        }

    diagnostics: Dict[str, object] = {}
    model.eval()
    for transition_id in terminal_ids:
        initial_indices = cohort.initial_indices(transition_id)
        with torch.no_grad():
            _, _, output = _transition_loss(
                model,
                cohort,
                transition_id,
                config,
                device,
                epoch=0,
                sample=False,
            )
            candidate = output["terminal_state"].mean(0)
            cue_values = cohort.initial_cues[initial_indices].to(device)
            if cohort.initial_cue_mask is None:
                cues = cue_values.mean(0)
            else:
                cue_mask = cohort.initial_cue_mask[initial_indices].to(device)
                cues = (cue_values * cue_mask).sum(0) / cue_mask.sum(0).clamp_min(1.0)

        fixed_state, residual = model.refine_fixed_point(
            candidate,
            cues,
            iterations=config.fixed_point_iterations,
            learning_rate=config.fixed_point_learning_rate,
            tolerance=config.fixed_point_tolerance,
        )
        record: Dict[str, object] = {
            "candidate_source": "mean simulated terminal state",
            "fixed_point_residual_rms": residual,
            "fixed_point_tolerance": config.fixed_point_tolerance,
            "state_dimension": model.state_dim,
        }
        if model.state_dim <= config.jacobian_max_dimension:
            eigenvalues = model.jacobian_eigenvalues(
                fixed_state,
                cues,
                max_dimension=config.jacobian_max_dimension,
            )
            record["jacobian"] = {
                "status": "evaluated",
                "max_real_eigenvalue": float(eigenvalues.real.max().cpu()),
                "unstable_eigenvalue_count": int((eigenvalues.real >= 0).sum().cpu()),
            }
        else:
            record["jacobian"] = {
                "status": "not_evaluated",
                "reason": (
                    f"state dimension {model.state_dim} exceeds configured full-"
                    f"Jacobian limit {config.jacobian_max_dimension}"
                ),
            }
        basin = model.basin_return_fraction(
            fixed_state,
            cues,
            trials=config.basin_trials,
            perturbation_scale=config.basin_perturbation_scale,
            horizon=config.basin_horizon,
            steps=config.basin_steps,
            tolerance=config.basin_tolerance,
            seed=_stable_string_seed(transition_id, config.seed),
        )
        distances = basin["normalized_distance"].detach().cpu()
        record["basin"] = {
            "fraction_returned": float(basin["fraction_returned"].cpu()),
            "trials": config.basin_trials,
            "perturbation_scale": config.basin_perturbation_scale,
            "return_tolerance": config.basin_tolerance,
            "mean_normalized_distance": float(distances.mean()),
            "max_normalized_distance": float(distances.max()),
        }
        diagnostics[transition_id] = record
    return {
        "status": "evaluated",
        "transitions": diagnostics,
        "unconditional_attractor_claim": False,
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_temporal_benchmark(
    data_root: Path,
    output_root: Path,
    config: TemporalTrainingConfig,
    conditions: Sequence[str] = ("true_circuit", "no_circuit"),
    device_name: Optional[str] = None,
) -> Dict[str, object]:
    cohort = load_temporal_cohort(data_root)
    config.validate()
    if len(set(conditions)) != len(conditions):
        raise ValueError("Benchmark conditions must be unique.")
    unsafe_conditions = [
        condition for condition in conditions if SAFE_CONDITION.fullmatch(condition) is None
    ]
    if unsafe_conditions:
        raise ValueError(f"Unsafe benchmark condition names: {unsafe_conditions}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device_name
        if device_name is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    results = {
        "schema_version": SCHEMA_VERSION,
        "alignment_mode": cohort.alignment_mode,
        "split_groups": {
            name: list(groups) for name, groups in cohort.split_groups.items()
        },
        "priors_fit_groups": list(cohort.priors_fit_groups),
        "cue_names": list(cohort.cue_names),
        "cue_provenance": [dict(record) for record in cohort.cue_provenance],
        "device": str(device),
        "config": asdict(config),
        "conditions": {},
        "claim_boundary": (
            "Test metrics assess held-out temporal prediction. An attractor claim "
            "still requires stable fixed points, basin return, and prospective "
            "held-out interventions in the biological system."
        ),
    }
    partial_path = output_root / "wld_temporal_results.partial.json"
    for condition in conditions:
        print(
            f"Training temporal condition {condition!r} on {device}...",
            flush=True,
        )
        model, training = train_temporal_model(
            cohort, config, condition, device
        )
        checkpoint_path = output_root / f"wld_temporal_{condition}.pt"
        torch.save(
            {
                "condition": condition,
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "config": asdict(config),
            },
            checkpoint_path,
        )
        results["conditions"][condition] = {
            "training": training,
            "test": evaluate_test_groups(model, cohort, config, device),
            "checkpoint": checkpoint_path.name,
        }
        _atomic_json(partial_path, results)
        print(
            f"Completed {condition!r}: best epoch {training['best_epoch']}, "
            f"validation loss {training['best_validation_loss']:.6g}",
            flush=True,
        )

    if "true_circuit" in results["conditions"]:
        true_groups = results["conditions"]["true_circuit"]["test"]["by_group"]
        comparison = {}
        for condition, condition_result in results["conditions"].items():
            if condition == "true_circuit":
                continue
            control_groups = condition_result["test"]["by_group"]
            comparison[condition] = {}
            for group in sorted(set(true_groups).intersection(control_groups)):
                true_value = true_groups[group].get("log_rna_swd")
                control_value = control_groups[group].get("log_rna_swd")
                comparison[condition][group] = (
                    float(control_value - true_value)
                    if true_value is not None and control_value is not None
                    else None
                )
        results["control_comparison"] = {
            "metric": "control log-RNA SWD minus true-circuit log-RNA SWD",
            "positive_favors_true_circuit": True,
            "by_condition_and_group": comparison,
            "unconditional_success_claim": False,
        }

    _atomic_json(output_root / "wld_temporal_results.json", results)
    partial_path.unlink(missing_ok=True)
    return results


def _parse_conditions(value: str) -> Tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    if not conditions:
        raise argparse.ArgumentTypeError("At least one condition is required.")
    if len(set(conditions)) != len(conditions):
        raise argparse.ArgumentTypeError("Condition names must be unique.")
    unsafe = [item for item in conditions if SAFE_CONDITION.fullmatch(item) is None]
    if unsafe:
        raise argparse.ArgumentTypeError(f"Unsafe condition names: {unsafe}")
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data", type=Path, required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--data", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--epochs", type=int, default=100)
    benchmark_parser.add_argument("--steps", type=int, default=12)
    benchmark_parser.add_argument("--patience", type=int, default=8)
    benchmark_parser.add_argument("--conditions", type=_parse_conditions, default=("true_circuit", "no_circuit"))
    benchmark_parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.command == "validate":
        cohort = load_temporal_cohort(args.data)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "alignment_mode": cohort.alignment_mode,
                    "transitions": len(cohort.transitions),
                    "split_groups": {
                        key: list(value) for key, value in cohort.split_groups.items()
                    },
                },
                indent=2,
            )
        )
        return

    config = TemporalTrainingConfig(
        epochs=args.epochs,
        integration_steps=args.steps,
        patience=args.patience,
    )
    result = run_temporal_benchmark(
        args.data,
        args.output,
        config,
        conditions=args.conditions,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
