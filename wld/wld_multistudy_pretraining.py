"""Multi-study training and leakage contract for WLD v4.

This module deliberately separates metadata used to *partition* data from
measurements used to *predict* biology.  Study, donor and cell identifiers are
never tensors accepted by the model.  Longitudinal observations need not be
cell-paired: predicted and observed future populations are compared with
distributional losses within a biological group.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from wld_foundation_model_v4 import (
    FoundationPriors,
    WLDMultistudyFoundationModel,
    architecture_contract,
    audit_foundation_inputs,
    no_circuit_priors,
    supported_sign_shuffle_priors,
)


ALLOWED_MODALITIES = {"atac", "rna", "protein", "metabolic", "cue"}


@dataclass(frozen=True)
class StudySpec:
    study_id: str
    species: str
    genome_build: str
    tissue: str
    modalities: Tuple[str, ...]
    donor_ids: Tuple[str, ...]
    source_url: str = ""
    longitudinal: bool = False
    perturbation: bool = False

    def validate(self) -> None:
        if not self.study_id or not self.species or not self.tissue:
            raise ValueError("study_id, species and tissue are required")
        if not self.genome_build:
            raise ValueError("genome_build is required")
        modalities = {value.lower() for value in self.modalities}
        if not modalities or not modalities.issubset(ALLOWED_MODALITIES):
            raise ValueError(f"Unsupported modalities: {sorted(modalities)}")
        if len(set(self.donor_ids)) != len(self.donor_ids) or not self.donor_ids:
            raise ValueError("donor_ids must be non-empty and unique within a study")


@dataclass(frozen=True)
class ObservationGroup:
    """One biological population; cells within source/target are unpaired."""

    group_id: str
    study_id: str
    donor_id: str
    source_time: float
    target_time: float
    condition: str = ""

    def validate(self) -> None:
        if not self.group_id or not self.study_id or not self.donor_id:
            raise ValueError("group, study and donor identifiers are required")
        if not math.isfinite(self.source_time) or not math.isfinite(self.target_time):
            raise ValueError("times must be finite")
        if self.target_time <= self.source_time:
            raise ValueError("Every transition must point forward in time")


@dataclass(frozen=True)
class SealedSplit:
    train_groups: Tuple[str, ...]
    validation_groups: Tuple[str, ...]
    test_groups: Tuple[str, ...]
    validation_studies: Tuple[str, ...]
    test_studies: Tuple[str, ...]

    def validate(self) -> None:
        sets = [set(self.train_groups), set(self.validation_groups), set(self.test_groups)]
        if any(sets[i].intersection(sets[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Group leakage across train/validation/test")
        if not all(sets):
            raise ValueError("All three partitions must contain at least one group")


def make_sealed_split(
    groups: Sequence[ObservationGroup],
    *,
    validation_studies: Sequence[str],
    test_studies: Sequence[str],
) -> SealedSplit:
    """Split whole studies before feature selection or prior compilation."""

    for value in groups:
        value.validate()
    validation = set(validation_studies)
    test = set(test_studies)
    if validation.intersection(test):
        raise ValueError("A study cannot be both validation and test")
    known = {value.study_id for value in groups}
    if not validation.issubset(known) or not test.issubset(known):
        raise ValueError("Requested held-out studies are absent")
    train_groups, validation_groups, test_groups = [], [], []
    for value in groups:
        target = (
            test_groups
            if value.study_id in test
            else validation_groups
            if value.study_id in validation
            else train_groups
        )
        target.append(value.group_id)
    split = SealedSplit(
        tuple(train_groups),
        tuple(validation_groups),
        tuple(test_groups),
        tuple(sorted(validation)),
        tuple(sorted(test)),
    )
    split.validate()
    return split


def verify_donor_separation(
    groups: Sequence[ObservationGroup], split: SealedSplit
) -> Dict[str, List[str]]:
    by_id = {value.group_id: value for value in groups}
    donor_sets: Dict[str, set[Tuple[str, str]]] = {}
    for name, identifiers in (
        ("train", split.train_groups),
        ("validation", split.validation_groups),
        ("test", split.test_groups),
    ):
        donor_sets[name] = {
            (by_id[value].study_id, by_id[value].donor_id) for value in identifiers
        }
    names = list(donor_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = donor_sets[left].intersection(donor_sets[right])
            if overlap:
                raise ValueError(f"Donor leakage between {left} and {right}: {overlap}")
    return {
        key: [f"{study}:{donor}" for study, donor in sorted(value)]
        for key, value in donor_sets.items()
    }


@dataclass
class FoundationBatch:
    """Tensor batch with metadata kept out of the model call."""

    group_id: str
    study_id: str
    donor_id: str
    cues: Tensor
    horizon: float
    source_atac: Optional[Tensor] = None
    source_rna: Optional[Tensor] = None
    source_protein: Optional[Tensor] = None
    source_metabolic: Optional[Tensor] = None
    context_covariates: Optional[Tensor] = None
    target_rna: Optional[Tensor] = None
    target_atac: Optional[Tensor] = None
    modality_masks: Optional[Mapping[str, Tensor]] = None

    def model_inputs(self, *, steps: int, use_source_rna: bool) -> Dict[str, object]:
        return {
            "cues": self.cues,
            "horizon": float(self.horizon),
            "steps": int(steps),
            "atac": self.source_atac,
            "rna_encoder_input": self.source_rna if use_source_rna else None,
            "protein": self.source_protein,
            "metabolic": self.source_metabolic,
            "context_covariates": self.context_covariates,
            "modality_masks": self.modality_masks,
            "initial_rna": self.source_rna if use_source_rna else None,
        }


def _moments(value: Tensor) -> Tuple[Tensor, Tensor]:
    transformed = torch.log1p(value.clamp_min(0.0))
    return transformed.mean(0), transformed.var(0, unbiased=False)


def population_moment_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Unpaired population loss; no source cell is matched to a future cell."""

    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape[1] != target.shape[1]:
        raise ValueError("Prediction and target must share a feature dimension")
    pred_mean, pred_var = _moments(prediction)
    target_mean, target_var = _moments(target)
    return F.smooth_l1_loss(pred_mean, target_mean) + 0.25 * F.smooth_l1_loss(
        pred_var, target_var
    )


def multiscale_distribution_loss(
    output: Mapping[str, Tensor], batch: FoundationBatch
) -> Tuple[Tensor, Dict[str, float]]:
    pieces: List[Tensor] = []
    metrics: Dict[str, float] = {}
    if batch.target_rna is not None:
        value = population_moment_loss(output["rna_t"], batch.target_rna)
        pieces.append(value)
        metrics["rna_population"] = float(value.detach())
    if batch.target_atac is not None:
        value = population_moment_loss(output["accessibility_t"], batch.target_atac)
        pieces.append(value)
        metrics["atac_population"] = float(value.detach())
    if not pieces:
        raise ValueError("At least one measured future modality is required")
    total = torch.stack(pieces).mean()
    return total, metrics


def adaptation_regularizer(model: WLDMultistudyFoundationModel) -> Tensor:
    """Shrink context adaptations toward shared biology without freezing them."""

    terms = []
    for name, parameter in model.named_parameters():
        if "context_gain.weight" in name or ".adapter.weight" in name:
            terms.append(parameter.square().mean())
    if not terms:
        return next(model.parameters()).new_zeros(())
    return torch.stack(terms).mean()


@dataclass
class PretrainingConfig:
    epochs: int = 10
    steps: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    adaptation_penalty: float = 1e-4
    use_source_rna: bool = False
    seed: int = 42


class MultistudyPretrainer:
    """Validation-selected training with an untouched study-level test split."""

    def __init__(
        self,
        model: WLDMultistudyFoundationModel,
        config: PretrainingConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _run_batch(self, batch: FoundationBatch, *, training: bool) -> Tuple[Tensor, Dict[str, float]]:
        output = self.model(**batch.model_inputs(steps=self.config.steps, use_source_rna=self.config.use_source_rna))
        biological, metrics = multiscale_distribution_loss(output, batch)
        regularizer = adaptation_regularizer(self.model)
        total = biological + self.config.adaptation_penalty * regularizer
        metrics.update(
            loss=float(total.detach()),
            adaptation_regularizer=float(regularizer.detach()),
            context_variation=float(output["context"].var(0, unbiased=False).mean().detach()),
            rna_decay_variation=float(output["rna_decay"].var(0, unbiased=False).mean().detach()),
        )
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
        return total, metrics

    def fit(
        self,
        train: Sequence[FoundationBatch],
        validation: Sequence[FoundationBatch],
    ) -> Dict[str, object]:
        if not train or not validation:
            raise ValueError("Training and validation batches are required")
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        best_loss = float("inf")
        best_state = None
        history: List[Dict[str, float]] = []
        for epoch in range(self.config.epochs):
            self.model.train()
            order = list(train)
            random.shuffle(order)
            train_losses = [float(self._run_batch(batch, training=True)[0].detach()) for batch in order]
            self.model.eval()
            with torch.no_grad():
                validation_losses = [float(self._run_batch(batch, training=False)[0]) for batch in validation]
            row = {
                "epoch": float(epoch),
                "train_loss": sum(train_losses) / len(train_losses),
                "validation_loss": sum(validation_losses) / len(validation_losses),
            }
            history.append(row)
            if row["validation_loss"] < best_loss:
                best_loss = row["validation_loss"]
                best_state = copy.deepcopy(self.model.state_dict())
        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected")
        self.model.load_state_dict(best_state)
        return {
            "best_validation_loss": best_loss,
            "history": history,
            "test_evaluated": False,
            "architecture": architecture_contract(self.model),
        }

    def evaluate_sealed_test(self, test: Sequence[FoundationBatch]) -> Dict[str, float]:
        """Intentional one-way door: call only after development is frozen."""

        if not test:
            raise ValueError("A sealed test set is required")
        self.model.eval()
        with torch.no_grad():
            losses = [float(self._run_batch(batch, training=False)[0]) for batch in test]
        return {"sealed_test_loss": sum(losses) / len(losses)}


def make_scientific_controls(
    priors: FoundationPriors, seed: int
) -> Dict[str, FoundationPriors]:
    return {
        "supported_circuit": priors,
        "no_tf_circuit": no_circuit_priors(priors),
        "supported_sign_shuffle": supported_sign_shuffle_priors(priors, seed),
    }


def validate_catalog(
    studies: Sequence[StudySpec], feature_names: Sequence[str]
) -> Dict[str, object]:
    if len({value.study_id for value in studies}) != len(studies):
        raise ValueError("study_id must be unique")
    for value in studies:
        value.validate()
    leakage = audit_foundation_inputs(feature_names)
    return {
        "studies": len(studies),
        "tissues": sorted({value.tissue for value in studies}),
        "modalities": sorted({item for value in studies for item in value.modalities}),
        "longitudinal_studies": sum(value.longitudinal for value in studies),
        "perturbation_studies": sum(value.perturbation for value in studies),
        "leakage": leakage,
    }


def save_development_checkpoint(
    root: Path,
    model: WLDMultistudyFoundationModel,
    report: Mapping[str, object],
    split: SealedSplit,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root / "wld_v4_development.pt")
    payload = dict(report)
    payload["sealed_split"] = {
        "train_groups": list(split.train_groups),
        "validation_groups": list(split.validation_groups),
        "test_groups_sha256_only": True,
        "test_groups_count": len(split.test_groups),
        "validation_studies": list(split.validation_studies),
        "test_studies": list(split.test_studies),
    }
    (root / "wld_v4_development.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


__all__ = [
    "FoundationBatch",
    "MultistudyPretrainer",
    "ObservationGroup",
    "PretrainingConfig",
    "SealedSplit",
    "StudySpec",
    "adaptation_regularizer",
    "make_scientific_controls",
    "make_sealed_split",
    "multiscale_distribution_loss",
    "population_moment_loss",
    "save_development_checkpoint",
    "validate_catalog",
    "verify_donor_separation",
]
