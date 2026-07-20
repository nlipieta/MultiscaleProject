"""Temporal fine-tuning for the pretrained WLD v4 foundation model.

The real-data entry point consumes the GSE240061 training-atlas projection,
the frozen E/G/N -> I subject split, and a completed expanded-corpus WLD v4
checkpoint.  Pre and 3.5-hour cells are destructive population observations;
they are sampled independently and never paired by expression, embedding,
pseudotime, optimal transport, or cell label.

Measured time-zero RNA initializes the RNA component of the ODE state but is
not supplied to the encoder or context network.  The encoder receives ATAC and
the declared exercise cue.  All representation and mechanistic-field
parameters remain trainable, with separate learning rates.  The transient
3.5-hour endpoint is not penalized toward zero velocity and is not called an
attractor.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import Tensor

from wld_foundation_data import (
    atomic_json,
    read_barcodes,
    read_bundle_features,
    sha256_file,
    verify_bundle,
)
from wld_foundation_model_v4 import (
    FoundationPriors,
    WLDMultistudyFoundationModel,
    architecture_contract,
    no_circuit_priors,
    supported_sign_shuffle_priors,
)
from wld_phase_b_priors import load_phase_b_priors, verify_phase_b_priors


SCHEMA_VERSION = "1.0"
EXPECTED_SPLIT = {
    "train": ("E", "G", "N"),
    "validation": ("I",),
    "test": ("J", "L"),
}
EXPECTED_CONDITION = {
    "E": "exercise",
    "G": "exercise",
    "I": "exercise",
    "J": "exercise",
    "L": "control",
    "N": "control",
}


@dataclass(frozen=True)
class TemporalFinetuningConfig:
    epochs: int = 40
    batch_size: int = 64
    batches_per_transition: int = 2
    integration_steps: int = 6
    horizon_hours: float = 3.5
    representation_learning_rate: float = 2e-4
    field_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    projections: int = 32
    validation_cells: int = 256
    patience: int = 8
    rna_swd_weight: float = 1.0
    rna_mean_weight: float = 0.5
    rna_variance_weight: float = 0.1
    atac_swd_weight: float = 0.5
    adaptation_weight: float = 1e-6
    seed: int = 42

    def validate(self) -> None:
        positive_ints = (
            self.epochs,
            self.batch_size,
            self.batches_per_transition,
            self.integration_steps,
            self.projections,
            self.validation_cells,
            self.patience,
        )
        if any(value < 1 for value in positive_ints):
            raise ValueError("Temporal integer configuration values must be positive")
        positive = (
            self.horizon_hours,
            self.representation_learning_rate,
            self.field_learning_rate,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Horizon and learning rates must be finite and positive")
        nonnegative = (
            self.weight_decay,
            self.rna_swd_weight,
            self.rna_mean_weight,
            self.rna_variance_weight,
            self.atac_swd_weight,
            self.adaptation_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("Loss weights and weight decay must be finite and non-negative")


@dataclass
class SubjectTransition:
    subject: str
    condition: str
    split: str
    initial_rna: sparse.csr_matrix
    initial_atac: sparse.csr_matrix
    target_rna: sparse.csr_matrix
    target_atac: sparse.csr_matrix

    def validate(self, genes: int, peaks: int) -> None:
        if self.split not in {"train", "validation"}:
            raise ValueError("Only development transitions may be materialized")
        if self.condition not in {"exercise", "control"}:
            raise ValueError("Unknown experimental condition")
        if self.initial_rna.shape[1] != genes or self.target_rna.shape[1] != genes:
            raise ValueError("RNA feature dimensions changed")
        if self.initial_atac.shape[1] != peaks or self.target_atac.shape[1] != peaks:
            raise ValueError("ATAC feature dimensions changed")
        if self.initial_rna.shape[0] != self.initial_atac.shape[0]:
            raise ValueError("Time-zero RNA/ATAC must be exact same-cell measurements")
        if self.target_rna.shape[0] != self.target_atac.shape[0]:
            raise ValueError("Future RNA/ATAC must be exact same-cell measurements")
        if min(self.initial_rna.shape[0], self.target_rna.shape[0]) < 20:
            raise ValueError("Each subject/time population requires at least 20 cells")


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_payload(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_state_dict(path: Path, device: torch.device) -> Mapping[str, Tensor]:
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, Mapping):
        raise ValueError("Pretrained checkpoint is not a state dictionary")
    return value


def _feature_indices(path: Path, selected: Sequence[str]) -> np.ndarray:
    names = [value["feature_name"] for value in read_bundle_features(path)]
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise ValueError(f"Prior features are absent from the harmonized bundle: {missing[:5]}")
    return np.asarray([lookup[name] for name in selected], dtype=np.int64)


def _read_metadata(path: Path) -> Dict[str, Mapping[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cell_id", "subject", "condition", "timepoint"}
        if not required.issubset(reader.fieldnames or ()): 
            raise ValueError(f"Metadata is missing {sorted(required)}")
        result: Dict[str, Mapping[str, str]] = {}
        for row in reader:
            cell = str(row["cell_id"])
            if not cell or cell in result:
                raise ValueError("Metadata cell IDs must be nonempty and unique")
            result[cell] = {key: str(value) for key, value in row.items()}
    return result


def _assert_frozen_split(path: Path) -> dict:
    value = _json(path)
    actual = {
        name: tuple(sorted(map(str, value.get(name, ()))))
        for name in ("train", "validation", "test")
    }
    expected = {name: tuple(sorted(values)) for name, values in EXPECTED_SPLIT.items()}
    if actual != expected:
        raise RuntimeError(f"Subject split changed: expected {expected}, found {actual}")
    return value


def load_gse240061_transitions(
    bundle_root: Path,
    export_root: Path,
    feature_vocab: Mapping[str, Sequence[str]],
) -> Tuple[List[SubjectTransition], dict]:
    manifest = verify_bundle(bundle_root)
    if manifest.get("study_id") != "GSE240061":
        raise RuntimeError("Temporal bundle is not GSE240061")
    if manifest.get("species") != "Homo sapiens" or manifest.get("genome_build") != "GRCh38":
        raise RuntimeError("Temporal bundle must be human GRCh38")
    if manifest.get("pairing", {}).get("pairing") != "exact":
        raise RuntimeError("GSE240061 RNA and ATAC modalities must be exact same-cell measurements")
    _assert_frozen_split(export_root / "split.json")
    metadata = _read_metadata(export_root / "metadata.tsv")

    rna_root = bundle_root / "modalities" / "rna"
    atac_root = bundle_root / "modalities" / "atac"
    rna_barcodes = read_barcodes(rna_root / "barcodes.tsv.gz")
    atac_barcodes = read_barcodes(atac_root / "barcodes.tsv.gz")
    if rna_barcodes != atac_barcodes:
        raise RuntimeError("Harmonized RNA/ATAC barcode order changed")
    if set(rna_barcodes) != set(metadata):
        raise RuntimeError("Harmonized observations and frozen metadata do not match")

    rna_columns = _feature_indices(
        rna_root / "features.tsv.gz", feature_vocab["genes"]
    )
    atac_columns = _feature_indices(
        atac_root / "features.tsv.gz", feature_vocab["peaks"]
    )
    rna = sparse.load_npz(rna_root / "counts.csr.npz").tocsr()[:, rna_columns]
    atac = sparse.load_npz(atac_root / "counts.csr.npz").tocsr()[:, atac_columns]

    by_subject_time: Dict[Tuple[str, str], List[int]] = {}
    observed_conditions: Dict[str, str] = {}
    for index, barcode in enumerate(rna_barcodes):
        row = metadata[barcode]
        subject = row["subject"]
        condition = row["condition"].lower()
        timepoint = row["timepoint"].lower()
        if subject not in EXPECTED_CONDITION:
            raise RuntimeError(f"Unexpected GSE240061 subject: {subject}")
        if condition != EXPECTED_CONDITION[subject]:
            raise RuntimeError(f"Condition changed for subject {subject}")
        if timepoint not in {"pre", "post_3.5h"}:
            raise RuntimeError(f"Unexpected timepoint: {timepoint}")
        observed_conditions[subject] = condition
        by_subject_time.setdefault((subject, timepoint), []).append(index)

    split_lookup = {
        subject: split
        for split, subjects in EXPECTED_SPLIT.items()
        for subject in subjects
    }
    transitions: List[SubjectTransition] = []
    excluded = []
    for subject in sorted(EXPECTED_CONDITION):
        split = split_lookup[subject]
        if split == "test":
            excluded.append(subject)
            continue
        pre = np.asarray(by_subject_time[(subject, "pre")], dtype=np.int64)
        post = np.asarray(by_subject_time[(subject, "post_3.5h")], dtype=np.int64)
        transition = SubjectTransition(
            subject=subject,
            condition=observed_conditions[subject],
            split=split,
            initial_rna=rna[pre].tocsr(),
            initial_atac=atac[pre].tocsr(),
            target_rna=rna[post].tocsr(),
            target_atac=atac[post].tocsr(),
        )
        transition.validate(len(feature_vocab["genes"]), len(feature_vocab["peaks"]))
        transitions.append(transition)
    if {value.subject for value in transitions if value.split == "train"} != {"E", "G", "N"}:
        raise RuntimeError("Temporal training subjects changed")
    if {value.subject for value in transitions if value.split == "validation"} != {"I"}:
        raise RuntimeError("Temporal validation subject changed")
    return transitions, {
        "training_subjects": ["E", "G", "N"],
        "validation_subjects": ["I"],
        "excluded_subjects": excluded,
        "excluded_subject_status": (
            "not evaluated here; upstream representation validation used the full "
            "GSE240061 bundle, so J/L are not claimed as fully sealed tests"
        ),
        "fabricated_pairs": False,
        "time_zero_rna_encoder_input": False,
        "time_zero_rna_ode_initial_state": True,
        "cell_labels_or_subject_ids_encoder_input": False,
    }


def _library_normalize(matrix: sparse.csr_matrix, scale: float = 1e4) -> np.ndarray:
    dense = matrix.toarray().astype(np.float32, copy=False)
    totals = dense.sum(axis=1, keepdims=True)
    return dense * (float(scale) / np.maximum(totals, 1.0))


def _sample_rows(cells: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(cells, size=batch, replace=cells < batch).astype(np.int64)


def _tensor_batch(
    transition: SubjectTransition,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    cue_index: int,
    num_cues: int,
) -> dict:
    source = _sample_rows(transition.initial_rna.shape[0], batch_size, rng)
    target = _sample_rows(transition.target_rna.shape[0], batch_size, rng)
    initial_rna = torch.as_tensor(
        _library_normalize(transition.initial_rna[source]), device=device
    )
    initial_atac = torch.as_tensor(
        (transition.initial_atac[source].toarray() > 0).astype(np.float32),
        device=device,
    )
    target_rna = torch.as_tensor(
        _library_normalize(transition.target_rna[target]), device=device
    )
    target_atac = torch.as_tensor(
        (transition.target_atac[target].toarray() > 0).astype(np.float32),
        device=device,
    )
    cues = torch.zeros((batch_size, num_cues), dtype=torch.float32, device=device)
    if transition.condition == "exercise":
        cues[:, cue_index] = 1.0
    return {
        "initial_rna": initial_rna,
        "initial_atac": initial_atac,
        "target_rna": target_rna,
        "target_atac": target_atac,
        "cues": cues,
    }


def sliced_wasserstein(left: Tensor, right: Tensor, projections: int) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("Sliced-Wasserstein populations must have equal rank-two shape")
    directions = torch.randn(
        (left.shape[1], projections), dtype=left.dtype, device=left.device
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    left_projection = torch.sort(left @ directions, dim=0).values
    right_projection = torch.sort(right @ directions, dim=0).values
    return F.smooth_l1_loss(left_projection, right_projection)


def population_objective(output: Mapping[str, Tensor], batch: Mapping[str, Tensor], config: TemporalFinetuningConfig) -> Dict[str, Tensor]:
    predicted_rna = torch.log1p(output["rna_t"].clamp_min(0.0))
    target_rna = torch.log1p(batch["target_rna"].clamp_min(0.0))
    rna_swd = sliced_wasserstein(predicted_rna, target_rna, config.projections)
    rna_mean = F.smooth_l1_loss(predicted_rna.mean(0), target_rna.mean(0))
    rna_variance = F.smooth_l1_loss(
        predicted_rna.var(0, unbiased=False), target_rna.var(0, unbiased=False)
    )
    atac_swd = sliced_wasserstein(
        output["accessibility_t"], batch["target_atac"], config.projections
    )
    total = (
        config.rna_swd_weight * rna_swd
        + config.rna_mean_weight * rna_mean
        + config.rna_variance_weight * rna_variance
        + config.atac_swd_weight * atac_swd
    )
    return {
        "loss": total,
        "rna_swd": rna_swd,
        "rna_mean": rna_mean,
        "rna_variance": rna_variance,
        "atac_swd": atac_swd,
        "terminal_velocity_l1": output["terminal_velocity"].abs().mean(),
    }


def _parameter_reference(model: WLDMultistudyFoundationModel) -> Dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _adaptation_penalty(
    model: WLDMultistudyFoundationModel,
    reference: Mapping[str, Tensor],
) -> Tensor:
    penalties = [
        (parameter - reference[name]).square().mean()
        for name, parameter in model.named_parameters()
        if parameter.numel() and name in reference
    ]
    if not penalties:
        return next(model.parameters()).new_zeros(())
    return torch.stack(penalties).mean()


def _copy_compatible_parameters(
    model: WLDMultistudyFoundationModel,
    source: Mapping[str, Tensor],
) -> None:
    copied = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = source.get(name)
            if value is not None and tuple(value.shape) == tuple(parameter.shape):
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                copied += 1
    if copied < 10:
        raise RuntimeError("Too few pretrained parameters were compatible")


def _make_control_priors(priors: FoundationPriors, condition: str, seed: int) -> FoundationPriors:
    if condition == "true_circuit":
        return priors
    if condition == "no_circuit":
        return no_circuit_priors(priors)
    if condition == "sign_shuffled_circuit":
        original = priors.circuit_tf_tf
        for offset in range(100):
            candidate = supported_sign_shuffle_priors(priors, seed + offset)
            if not torch.equal(candidate.circuit_tf_tf, original):
                return candidate
        raise RuntimeError("Could not produce a changed supported-sign control")
    raise ValueError(f"Unknown temporal condition: {condition}")


def _pearson(left: Tensor, right: Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) <= 0:
        return float("nan")
    return float((left @ right / denominator).detach().cpu())


class TemporalConditionTrainer:
    def __init__(
        self,
        model: WLDMultistudyFoundationModel,
        config: TemporalFinetuningConfig,
        cue_index: int,
        condition_name: str,
        device: torch.device,
    ) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.cue_index = int(cue_index)
        self.condition_name = condition_name
        self.device = device
        representation_parameters = list(model.encoder.parameters()) + list(
            model.context_network.parameters()
        )
        field_parameters = list(model.field.parameters())
        if not representation_parameters or not field_parameters:
            raise RuntimeError("Both representation and field parameters must be trainable")
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Temporal fine-tuning must not freeze model parameters")
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": representation_parameters,
                    "lr": config.representation_learning_rate,
                },
                {"params": field_parameters, "lr": config.field_learning_rate},
            ],
            weight_decay=config.weight_decay,
        )
        self.reference = _parameter_reference(model)

    def _run_batch(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        output = self.model(
            cues=batch["cues"],
            horizon=self.config.horizon_hours,
            steps=self.config.integration_steps,
            atac=batch["initial_atac"],
            rna_encoder_input=None,
            protein=None,
            metabolic=None,
            initial_rna=batch["initial_rna"],
        )
        metrics = population_objective(output, batch, self.config)
        metrics["adaptation"] = _adaptation_penalty(self.model, self.reference)
        metrics["loss"] = metrics["loss"] + self.config.adaptation_weight * metrics["adaptation"]
        metrics["prediction_rna"] = output["rna_t"]
        metrics["prediction_atac"] = output["accessibility_t"]
        metrics["rna_decay_variance"] = output["rna_decay"].var(0, unbiased=False).mean()
        metrics["circuit_gain_variance"] = (
            output["circuit_gain_scale"].var(0, unbiased=False).mean()
            if output["circuit_gain_scale"].numel()
            else output["rna_t"].new_zeros(())
        )
        return metrics

    def _train_epoch(self, transitions: Sequence[SubjectTransition], epoch: int) -> Dict[str, float]:
        self.model.train()
        condition_offset = sum(
            (index + 1) * byte
            for index, byte in enumerate(self.condition_name.encode("utf-8"))
        ) % 997
        rng = np.random.default_rng(
            self.config.seed + 1009 * epoch + condition_offset
        )
        rows = []
        order = list(transitions)
        random.Random(self.config.seed + epoch).shuffle(order)
        for transition in order:
            for _ in range(self.config.batches_per_transition):
                batch = _tensor_batch(
                    transition,
                    self.config.batch_size,
                    rng,
                    self.device,
                    self.cue_index,
                    self.model.priors.num_cues,
                )
                self.optimizer.zero_grad(set_to_none=True)
                metrics = self._run_batch(batch)
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                rows.append({key: float(value.detach().cpu()) for key, value in metrics.items() if value.ndim == 0})
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    @torch.no_grad()
    def evaluate(self, transitions: Sequence[SubjectTransition], seed: int) -> Dict[str, object]:
        self.model.eval()
        by_subject = {}
        for index, transition in enumerate(transitions):
            rng = np.random.default_rng(seed + index)
            batch = _tensor_batch(
                transition,
                self.config.validation_cells,
                rng,
                self.device,
                self.cue_index,
                self.model.priors.num_cues,
            )
            torch.manual_seed(seed + index)
            metrics = self._run_batch(batch)
            predicted_log_mean = torch.log1p(metrics["prediction_rna"]).mean(0)
            target_log_mean = torch.log1p(batch["target_rna"]).mean(0)
            persistence_rna = sliced_wasserstein(
                torch.log1p(batch["initial_rna"]),
                torch.log1p(batch["target_rna"]),
                self.config.projections,
            )
            persistence_atac = sliced_wasserstein(
                batch["initial_atac"], batch["target_atac"], self.config.projections
            )
            by_subject[transition.subject] = {
                key: float(value.detach().cpu())
                for key, value in metrics.items()
                if isinstance(value, Tensor) and value.ndim == 0
            }
            by_subject[transition.subject].update(
                {
                    "rna_log_mean_pearson": _pearson(predicted_log_mean, target_log_mean),
                    "persistence_rna_swd": float(persistence_rna.detach().cpu()),
                    "persistence_atac_swd": float(persistence_atac.detach().cpu()),
                }
            )
        scalar_keys = [
            key
            for key, value in next(iter(by_subject.values())).items()
            if isinstance(value, (float, int)) and math.isfinite(float(value))
        ]
        aggregate = {
            key: float(np.mean([row[key] for row in by_subject.values()]))
            for key in scalar_keys
        }
        return {"aggregate": aggregate, "by_subject": by_subject}

    def fit(
        self,
        training: Sequence[SubjectTransition],
        validation: Sequence[SubjectTransition],
        state_path: Path,
        input_signature: str,
    ) -> dict:
        initial_state_sha = _state_sha256(self.model.state_dict())
        start_epoch = 0
        best_loss = float("inf")
        best_epoch = -1
        best_state = None
        history: List[dict] = []
        stale = 0
        if state_path.is_file():
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            if state.get("input_signature") != input_signature or state.get("config") != asdict(self.config):
                raise RuntimeError("Temporal resume inputs or configuration changed")
            self.model.load_state_dict(state["model_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            initial_state_sha = state["initial_state_sha256"]
            start_epoch = int(state["epoch_completed"]) + 1
            best_loss = float(state["best_loss"])
            best_epoch = int(state["best_epoch"])
            best_state = state["best_state"]
            history = list(state["history"])
            stale = int(state["stale"])
            print(f"RESUME {self.condition_name}: epoch {start_epoch}", flush=True)

        # An early-stopped condition is complete.  Without this guard, every
        # Colab rerun would add one unnecessary epoch before stopping again.
        if stale >= self.config.patience:
            start_epoch = self.config.epochs

        for epoch in range(start_epoch, self.config.epochs):
            torch.manual_seed(self.config.seed + epoch)
            train_metrics = self._train_epoch(training, epoch)
            validation_metrics = self.evaluate(validation, self.config.seed + 20000)
            validation_loss = float(validation_metrics["aggregate"]["loss"])
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_loss,
                "validation_rna_swd": validation_metrics["aggregate"]["rna_swd"],
                "validation_atac_swd": validation_metrics["aggregate"]["atac_swd"],
                "validation_rna_log_mean_pearson": validation_metrics["aggregate"]["rna_log_mean_pearson"],
            }
            history.append(row)
            print(
                f"{self.condition_name} epoch {epoch:03d} | train {row['train_loss']:.5f} | "
                f"validation {validation_loss:.5f} | RNA SWD {row['validation_rna_swd']:.5f}",
                flush=True,
            )
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            temporary = state_path.with_suffix(state_path.suffix + ".tmp")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "condition": self.condition_name,
                    "input_signature": input_signature,
                    "config": asdict(self.config),
                    "initial_state_sha256": initial_state_sha,
                    "epoch_completed": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "history": history,
                    "stale": stale,
                },
                temporary,
            )
            os.replace(temporary, state_path)
            if stale >= self.config.patience:
                print(f"EARLY STOP {self.condition_name}: validation patience exhausted", flush=True)
                break
        if best_state is None:
            raise RuntimeError("No validation-selected temporal checkpoint")
        self.model.load_state_dict(best_state)
        final_validation = self.evaluate(validation, self.config.seed + 20000)
        final_state_sha = _state_sha256(self.model.state_dict())
        if final_state_sha == initial_state_sha:
            raise RuntimeError("Temporal fine-tuning did not update the model")
        return {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
            "validation": final_validation,
            "initial_state_sha256": initial_state_sha,
            "final_state_sha256": final_state_sha,
            "model_updated": True,
            "all_parameters_require_grad": all(parameter.requires_grad for parameter in self.model.parameters()),
            "test_subjects_evaluated": False,
        }


def _atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, temporary)
    os.replace(temporary, path)


def run_real_temporal_finetuning(
    phase_a_root: Path,
    phase_b_root: Path,
    corpus_root: Path,
    export_root: Path,
    output_root: Path,
    config: TemporalFinetuningConfig,
    *,
    device: Optional[str] = None,
) -> dict:
    config.validate()
    priors_root = phase_b_root / "priors" / "homo_sapiens_grch38"
    bundle_root = (
        phase_a_root
        / "harmonized"
        / "homo_sapiens_grch38"
        / "GSE240061_muscle_exercise"
    )
    pretrained_model = corpus_root / "wld_corpus_pretrained_model.pt"
    corpus_report_path = corpus_root / "wld_corpus_pretraining_report.json"
    required = (
        phase_a_root / "phase_a_ingestion_report.json",
        priors_root / "prior_manifest.json",
        priors_root / "feature_vocab.json",
        bundle_root / "bundle_manifest.json",
        export_root / "metadata.tsv",
        export_root / "split.json",
        pretrained_model,
        corpus_report_path,
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    verify_phase_b_priors(priors_root)
    corpus_report = _json(corpus_report_path)
    if corpus_report.get("checkpoint_sha256") != sha256_file(pretrained_model):
        raise RuntimeError("Expanded-corpus checkpoint hash mismatch")
    if corpus_report.get("sealed_test_evaluated") is not False:
        raise RuntimeError("Upstream corpus checkpoint crossed its sealed-study boundary")

    input_hashes = {
        "phase_a_report_sha256": sha256_file(phase_a_root / "phase_a_ingestion_report.json"),
        "prior_manifest_sha256": sha256_file(priors_root / "prior_manifest.json"),
        "feature_vocab_sha256": sha256_file(priors_root / "feature_vocab.json"),
        "temporal_bundle_manifest_sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        "metadata_sha256": sha256_file(export_root / "metadata.tsv"),
        "split_sha256": sha256_file(export_root / "split.json"),
        "pretrained_checkpoint_sha256": sha256_file(pretrained_model),
        "corpus_report_sha256": sha256_file(corpus_report_path),
    }
    signature = _sha_payload({"inputs": input_hashes, "config": asdict(config)})
    report_path = output_root / "wld_v4_temporal_finetuning_report.json"
    true_model_path = output_root / "wld_v4_temporal_true_circuit.pt"
    if report_path.is_file() and true_model_path.is_file():
        existing = _json(report_path)
        if existing.get("input_sha256") != input_hashes or existing.get("config") != asdict(config):
            raise RuntimeError("Completed temporal run inputs/configuration changed")
        if existing.get("true_circuit_checkpoint_sha256") != sha256_file(true_model_path):
            raise RuntimeError("Completed true-circuit checkpoint hash mismatch")
        if existing.get("test_subjects_evaluated") is not False or existing.get("attractor_claim") is not False:
            raise RuntimeError("Completed temporal report crossed its claim boundary")
        print("PASS: completed WLD v4 temporal fine-tuning restored; skipping retraining", flush=True)
        return existing

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_vocab = _json(priors_root / "feature_vocab.json")
    if "exercise" not in feature_vocab.get("cues", []):
        raise RuntimeError(
            f"The pretrained prior has no measured exercise cue: {feature_vocab.get('cues')}"
        )
    cue_index = list(feature_vocab["cues"]).index("exercise")
    transitions, data_contract = load_gse240061_transitions(
        bundle_root, export_root, feature_vocab
    )
    training = [value for value in transitions if value.split == "train"]
    validation = [value for value in transitions if value.split == "validation"]
    priors = load_phase_b_priors(priors_root, resolved_device)
    pretrained_state = _load_state_dict(pretrained_model, resolved_device)

    output_root.mkdir(parents=True, exist_ok=True)
    conditions = ("true_circuit", "no_circuit", "sign_shuffled_circuit")
    results = {}
    checkpoint_hashes = {}
    for condition_index, condition in enumerate(conditions):
        print("\n" + "=" * 76, flush=True)
        print(f"TEMPORAL CONDITION: {condition}", flush=True)
        print("=" * 76, flush=True)
        condition_priors = _make_control_priors(
            priors, condition, config.seed + 1000 * condition_index
        )
        model = WLDMultistudyFoundationModel(
            condition_priors, context_covariate_dim=0, context_dim=32
        ).to(resolved_device)
        if condition == "true_circuit":
            model.load_state_dict(pretrained_state)
        else:
            _copy_compatible_parameters(model, pretrained_state)
        field_before = _state_sha256(model.field.state_dict())
        representation_before = _state_sha256(
            {
                **{f"encoder.{key}": value for key, value in model.encoder.state_dict().items()},
                **{f"context.{key}": value for key, value in model.context_network.state_dict().items()},
            }
        )
        trainer = TemporalConditionTrainer(
            model, config, cue_index, condition, resolved_device
        )
        result = trainer.fit(
            training,
            validation,
            output_root / f"{condition}_training_state.pt",
            signature + ":" + condition,
        )
        field_after = _state_sha256(model.field.state_dict())
        representation_after = _state_sha256(
            {
                **{f"encoder.{key}": value for key, value in model.encoder.state_dict().items()},
                **{f"context.{key}": value for key, value in model.context_network.state_dict().items()},
            }
        )
        if field_after == field_before or representation_after == representation_before:
            raise RuntimeError("Temporal stage must update both representation and circuit field")
        result["field_updated"] = True
        result["representation_updated"] = True
        result["condition"] = condition
        checkpoint = output_root / f"wld_v4_temporal_{condition}.pt"
        _atomic_torch_save(checkpoint, model.state_dict())
        checkpoint_hashes[condition] = sha256_file(checkpoint)
        results[condition] = result
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    true_rna = results["true_circuit"]["validation"]["aggregate"]["rna_swd"]
    no_rna = results["no_circuit"]["validation"]["aggregate"]["rna_swd"]
    shuffle_rna = results["sign_shuffled_circuit"]["validation"]["aggregate"]["rna_swd"]
    reliance = {
        "rna_swd_advantage_over_no_circuit": float(no_rna - true_rna),
        "rna_swd_advantage_over_sign_shuffle": float(shuffle_rna - true_rna),
        "true_circuit_lower_than_both_controls": bool(true_rna < no_rna and true_rna < shuffle_rna),
        "forced_claim": False,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "GSE240061 transient pre-to-3.5-hour temporal development; not an attractor test",
        "device": str(resolved_device),
        "input_sha256": input_hashes,
        "config": asdict(config),
        "architecture": architecture_contract(
            WLDMultistudyFoundationModel(priors, context_dim=32)
        ),
        "data_contract": data_contract,
        "conditions": results,
        "checkpoint_sha256": checkpoint_hashes,
        "true_circuit_checkpoint_sha256": checkpoint_hashes["true_circuit"],
        "circuit_reliance": reliance,
        "training_contract": {
            "pre_and_post_cells_independently_sampled": True,
            "fabricated_cell_pairs": False,
            "time_zero_rna_used_only_as_ode_initial_state": True,
            "time_zero_rna_enters_encoder": False,
            "cell_identity_state_label_pseudotime_or_embedding_enters_encoder": False,
            "all_context_and_field_parameters_trainable": True,
            "transient_endpoint_zero_velocity_penalty": False,
            "validation_selects_checkpoint": True,
        },
        "upstream_exposure_note": (
            "The expanded-corpus checkpoint selected its representation using the full "
            "GSE240061 validation bundle. J/L are therefore excluded here but are not "
            "claimed as fully sealed tests. An external temporal study is required for "
            "clean held-out assessment."
        ),
        "test_subjects_evaluated": False,
        "external_sealed_test_evaluated": False,
        "attractor_diagnostics_run": False,
        "attractor_claim": False,
    }
    atomic_json(report_path, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-transition", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    report = run_real_temporal_finetuning(
        args.phase_a_root,
        args.phase_b_root,
        args.corpus_root,
        args.export_root,
        args.output,
        TemporalFinetuningConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_transition=args.batches_per_transition,
            integration_steps=args.steps,
            patience=args.patience,
            seed=args.seed,
        ),
        device=args.device,
    )
    print("\n" + "=" * 76)
    print("COMPLETE: WLD V4 TEMPORAL FINE-TUNING")
    print("=" * 76)
    for condition, record in report["conditions"].items():
        metrics = record["validation"]["aggregate"]
        print(
            f"{condition}: epoch={record['best_epoch']} loss={record['best_validation_loss']:.6f} "
            f"RNA_SWD={metrics['rna_swd']:.6f} Pearson={metrics['rna_log_mean_pearson']:.4f}"
        )
    print(f"Circuit reliance: {report['circuit_reliance']}")
    print("J/L evaluated: False")
    print("Attractor claim: False")
    print(f"Report: {args.output / 'wld_v4_temporal_finetuning_report.json'}")


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_SPLIT",
    "SubjectTransition",
    "TemporalConditionTrainer",
    "TemporalFinetuningConfig",
    "load_gse240061_transitions",
    "population_objective",
    "run_real_temporal_finetuning",
    "sliced_wasserstein",
]
