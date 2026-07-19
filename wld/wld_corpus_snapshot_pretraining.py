"""Pairing-aware continuation pretraining for the expanded WLD corpus.

This stage continues the validated human Phase B representation checkpoint on
the original Phase A cohorts plus new human GRCh38 expansion cohorts.  Exact
RNA/ATAC pairs receive cell-level contrastive losses.  Cohorts without proven
cell pairing receive distributional losses with independently sampled cells.

Cross-sectional snapshots do not identify a temporal vector field.  The
context-conditioned circuit field remains present and trainable for a later
longitudinal/perturbational stage, but snapshot optimization is restricted to
the structured encoder and context network.  Human priors are never applied to
mouse cohorts, and sealed studies are never loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import Tensor

from wld_foundation_data import (
    atomic_json,
    project_bundle_to_atlas,
    read_barcodes,
    read_bundle_features,
    sha256_file,
    verify_bundle,
)
from wld_foundation_model_v4 import (
    WLDMultistudyFoundationModel,
    architecture_contract,
)
from wld_phase_b_priors import load_phase_b_priors, verify_phase_b_priors
from wld_phase_b_snapshot_pretraining import symmetric_info_nce, variance_floor


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class BundleRecord:
    root: Path
    cohort_id: str
    study_id: str
    tissue: str
    split: str
    pairing: str
    source: str


@dataclass
class CorpusSnapshotCohort:
    cohort_id: str
    study_id: str
    tissue: str
    split: str
    pairing: str
    rna_barcodes: Tuple[str, ...]
    atac_barcodes: Tuple[str, ...]
    rna: sparse.csr_matrix
    atac: sparse.csr_matrix
    protein_barcodes: Tuple[str, ...] = ()
    protein: Optional[sparse.csr_matrix] = None

    @property
    def rna_cells(self) -> int:
        return int(self.rna.shape[0])

    @property
    def atac_cells(self) -> int:
        return int(self.atac.shape[0])

    def validate(self) -> None:
        if self.pairing not in {"exact", "unpaired_population"}:
            raise ValueError(f"Unsupported pairing state: {self.pairing}")
        if self.rna_cells < 2 or self.atac_cells < 2:
            raise ValueError("Each measured modality needs at least two cells")
        if len(self.rna_barcodes) != self.rna_cells or len(set(self.rna_barcodes)) != self.rna_cells:
            raise ValueError("RNA observation IDs must be unique and match the matrix")
        if len(self.atac_barcodes) != self.atac_cells or len(set(self.atac_barcodes)) != self.atac_cells:
            raise ValueError("ATAC observation IDs must be unique and match the matrix")
        if self.pairing == "exact" and (
            self.rna_barcodes != self.atac_barcodes or self.rna_cells != self.atac_cells
        ):
            raise ValueError("Exact pairing requires identical RNA/ATAC observation order")
        if self.protein is not None:
            if self.protein.shape[0] != len(self.protein_barcodes):
                raise ValueError("Protein observation IDs do not match the matrix")
            if len(set(self.protein_barcodes)) != len(self.protein_barcodes):
                raise ValueError("Protein observation IDs must be unique")


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _feature_indices(path: Path, selected: Sequence[str]) -> List[int]:
    names = [value["feature_name"] for value in read_bundle_features(path)]
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise ValueError(
            f"Prior features are absent from the training-atlas projection: {missing[:5]}"
        )
    return [lookup[name] for name in selected]


def _subsample_indices(cells: int, maximum: int, seed: int) -> np.ndarray:
    if cells <= maximum:
        return np.arange(cells, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(cells, size=maximum, replace=False)).astype(np.int64)


def load_corpus_cohort(
    record: BundleRecord,
    feature_vocab: Mapping[str, Sequence[str]],
    *,
    max_cells: int,
    seed: int,
) -> CorpusSnapshotCohort:
    manifest = verify_bundle(record.root)
    if manifest["cohort_id"] != record.cohort_id or manifest["study_id"] != record.study_id:
        raise RuntimeError("Bundle record identity changed")
    if manifest.get("species") != "Homo sapiens" or manifest.get("genome_build") != "GRCh38":
        raise ValueError("Human GRCh38 priors cannot be used with another species/build")
    if not {"rna", "atac"}.issubset(manifest["modalities"]):
        raise ValueError(f"RNA/ATAC are required: {record.cohort_id}")

    rna_root = record.root / "modalities" / "rna"
    atac_root = record.root / "modalities" / "atac"
    rna_barcodes = read_barcodes(rna_root / "barcodes.tsv.gz")
    atac_barcodes = read_barcodes(atac_root / "barcodes.tsv.gz")
    actual_pairing = "exact" if rna_barcodes == atac_barcodes else "unpaired_population"
    if actual_pairing != record.pairing:
        raise RuntimeError(
            f"Pairing manifest changed for {record.cohort_id}: {record.pairing} -> {actual_pairing}"
        )

    rna_columns = _feature_indices(
        rna_root / "features.tsv.gz", feature_vocab["genes"]
    )
    atac_columns = _feature_indices(
        atac_root / "features.tsv.gz", feature_vocab["peaks"]
    )
    rna = sparse.load_npz(rna_root / "counts.csr.npz").tocsr()[:, rna_columns]
    atac = sparse.load_npz(atac_root / "counts.csr.npz").tocsr()[:, atac_columns]

    if actual_pairing == "exact":
        selected = _subsample_indices(len(rna_barcodes), max_cells, seed)
        rna_rows = atac_rows = selected
    else:
        rna_rows = _subsample_indices(len(rna_barcodes), max_cells, seed)
        atac_rows = _subsample_indices(len(atac_barcodes), max_cells, seed + 7919)
    rna = rna[rna_rows].tocsr()
    atac = atac[atac_rows].tocsr()
    selected_rna_barcodes = tuple(rna_barcodes[index] for index in rna_rows)
    selected_atac_barcodes = tuple(atac_barcodes[index] for index in atac_rows)

    protein = None
    selected_protein_barcodes: Tuple[str, ...] = ()
    if "protein" in manifest["modalities"] and feature_vocab.get("proteins"):
        protein_root = record.root / "modalities" / "protein"
        protein_barcodes = read_barcodes(protein_root / "barcodes.tsv.gz")
        if protein_barcodes == rna_barcodes:
            protein_columns = _feature_indices(
                protein_root / "features.tsv.gz", feature_vocab["proteins"]
            )
            protein = sparse.load_npz(protein_root / "counts.csr.npz").tocsr()[
                rna_rows
            ][:, protein_columns]
            selected_protein_barcodes = selected_rna_barcodes

    result = CorpusSnapshotCohort(
        cohort_id=record.cohort_id,
        study_id=record.study_id,
        tissue=record.tissue,
        split=record.split,
        pairing=actual_pairing,
        rna_barcodes=selected_rna_barcodes,
        atac_barcodes=selected_atac_barcodes,
        rna=rna,
        atac=atac,
        protein_barcodes=selected_protein_barcodes,
        protein=protein,
    )
    result.validate()
    return result


def _library_normalize(matrix: sparse.csr_matrix, scale: float) -> np.ndarray:
    dense = matrix.toarray().astype(np.float32, copy=False)
    totals = dense.sum(axis=1, keepdims=True)
    return dense * (float(scale) / np.maximum(totals, 1.0))


def _rna_tensor(matrix: sparse.csr_matrix, rows: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(_library_normalize(matrix[rows], 1e4), device=device)


def _atac_tensor(matrix: sparse.csr_matrix, rows: np.ndarray, device: torch.device) -> Tensor:
    values = (matrix[rows].toarray() > 0).astype(np.float32)
    return torch.as_tensor(values, device=device)


def _protein_tensor(matrix: sparse.csr_matrix, rows: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(_library_normalize(matrix[rows], 1e3), device=device)


def feature_moment_loss(left: Tensor, right: Tensor, *, log1p: bool = False) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Distribution views must share a feature dimension")
    if log1p:
        left = torch.log1p(left.clamp_min(0.0))
        right = torch.log1p(right.clamp_min(0.0))
    return F.smooth_l1_loss(left.mean(0), right.mean(0)) + 0.25 * F.smooth_l1_loss(
        left.var(0, unbiased=False), right.var(0, unbiased=False)
    )


def covariance_alignment_loss(left: Tensor, right: Tensor) -> Tensor:
    if left.shape[1] != right.shape[1]:
        raise ValueError("Covariance views must share a feature dimension")
    left_centered = left - left.mean(0, keepdim=True)
    right_centered = right - right.mean(0, keepdim=True)
    left_cov = left_centered.transpose(0, 1) @ left_centered / max(1, left.shape[0] - 1)
    right_cov = right_centered.transpose(0, 1) @ right_centered / max(1, right.shape[0] - 1)
    return F.smooth_l1_loss(left_cov, right_cov)


def _encode(
    model: WLDMultistudyFoundationModel,
    *,
    atac: Optional[Tensor] = None,
    rna: Optional[Tensor] = None,
    protein: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    reference = atac if atac is not None else rna if rna is not None else protein
    if reference is None:
        raise ValueError("At least one measured modality is required")
    cues = reference.new_zeros((reference.shape[0], model.priors.num_cues))
    encoded = model.encoder(cues=cues, atac=atac, rna=rna, protein=protein)
    context = model.context_network(encoded["biological_context"])
    return context, encoded["tf"]


def _zero_metrics(reference: Tensor) -> Dict[str, Tensor]:
    zero = reference.new_zeros(())
    return {
        "paired_contrastive": zero,
        "unpaired_distribution": zero,
        "tf_alignment": zero,
        "variance_floor": zero,
        "protein_contrastive": zero,
    }


@dataclass
class CorpusPretrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    batches_per_cohort: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.15
    tf_weight: float = 0.25
    variance_weight: float = 0.10
    covariance_weight: float = 0.10
    protein_weight: float = 0.25
    max_cells_per_cohort: int = 4096
    seed: int = 42


class PairingAwareCorpusPretrainer:
    def __init__(
        self,
        model: WLDMultistudyFoundationModel,
        config: CorpusPretrainingConfig,
    ) -> None:
        self.model = model
        self.config = config
        if config.epochs < 1:
            raise ValueError("epochs must be at least one")
        if config.batch_size < 2:
            raise ValueError("batch_size must be at least two for contrastive training")
        if config.batches_per_cohort < 1 or config.max_cells_per_cohort < 2:
            raise ValueError("batches_per_cohort >= 1 and max_cells_per_cohort >= 2 are required")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        # Field parameters remain requires_grad=True for the later temporal
        # stage, but cross-sectional losses optimize only identifiable
        # representation parameters.
        self.parameters = list(model.encoder.parameters()) + list(model.context_network.parameters())
        optimized_ids = {id(parameter) for parameter in self.parameters}
        field_ids = {id(parameter) for parameter in model.field.parameters()}
        if optimized_ids.intersection(field_ids):
            raise RuntimeError("Snapshot optimizer unexpectedly contains circuit-field parameters")
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _paired_loss(
        self, cohort: CorpusSnapshotCohort, rng: np.random.Generator
    ) -> Tuple[Tensor, Dict[str, float]]:
        size = min(self.config.batch_size, cohort.rna_cells)
        rows = rng.choice(cohort.rna_cells, size=size, replace=False)
        device = next(self.model.parameters()).device
        rna = _rna_tensor(cohort.rna, rows, device)
        atac = _atac_tensor(cohort.atac, rows, device)
        atac_context, atac_tf = _encode(self.model, atac=atac)
        rna_context, rna_tf = _encode(self.model, rna=rna)
        metrics = _zero_metrics(atac_context)
        metrics["paired_contrastive"] = symmetric_info_nce(
            atac_context, rna_context, self.config.temperature
        )
        metrics["tf_alignment"] = 1.0 - F.cosine_similarity(
            torch.log1p(atac_tf), torch.log1p(rna_tf), dim=1
        ).mean()
        metrics["variance_floor"] = variance_floor(atac_context, rna_context)
        if cohort.protein is not None:
            protein = _protein_tensor(cohort.protein, rows, device)
            protein_context, _ = _encode(self.model, protein=protein)
            metrics["protein_contrastive"] = 0.5 * (
                symmetric_info_nce(protein_context, atac_context, self.config.temperature)
                + symmetric_info_nce(protein_context, rna_context, self.config.temperature)
            )
        total = (
            metrics["paired_contrastive"]
            + self.config.tf_weight * metrics["tf_alignment"]
            + self.config.variance_weight * metrics["variance_floor"]
            + self.config.protein_weight * metrics["protein_contrastive"]
        )
        return total, {key: float(value.detach()) for key, value in metrics.items()}

    def _unpaired_loss(
        self, cohort: CorpusSnapshotCohort, rng: np.random.Generator
    ) -> Tuple[Tensor, Dict[str, float]]:
        rna_size = min(self.config.batch_size, cohort.rna_cells)
        atac_size = min(self.config.batch_size, cohort.atac_cells)
        rna_rows = rng.choice(cohort.rna_cells, size=rna_size, replace=False)
        atac_rows = rng.choice(cohort.atac_cells, size=atac_size, replace=False)
        device = next(self.model.parameters()).device
        rna = _rna_tensor(cohort.rna, rna_rows, device)
        atac = _atac_tensor(cohort.atac, atac_rows, device)
        atac_context, atac_tf = _encode(self.model, atac=atac)
        rna_context, rna_tf = _encode(self.model, rna=rna)
        metrics = _zero_metrics(atac_context)
        metrics["unpaired_distribution"] = feature_moment_loss(
            atac_context, rna_context
        ) + self.config.covariance_weight * covariance_alignment_loss(
            atac_context, rna_context
        )
        metrics["tf_alignment"] = feature_moment_loss(
            atac_tf, rna_tf, log1p=True
        )
        metrics["variance_floor"] = variance_floor(atac_context, rna_context)
        total = (
            metrics["unpaired_distribution"]
            + self.config.tf_weight * metrics["tf_alignment"]
            + self.config.variance_weight * metrics["variance_floor"]
        )
        return total, {key: float(value.detach()) for key, value in metrics.items()}

    def _step(
        self,
        cohort: CorpusSnapshotCohort,
        rng: np.random.Generator,
        *,
        training: bool,
    ) -> Dict[str, float]:
        total, metrics = (
            self._paired_loss(cohort, rng)
            if cohort.pairing == "exact"
            else self._unpaired_loss(cohort, rng)
        )
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 5.0)
            self.optimizer.step()
        metrics["loss"] = float(total.detach())
        metrics["exact_batch"] = float(cohort.pairing == "exact")
        metrics["unpaired_batch"] = float(cohort.pairing == "unpaired_population")
        return metrics

    def _epoch(
        self,
        cohorts: Sequence[CorpusSnapshotCohort],
        *,
        training: bool,
        epoch: int,
    ) -> Dict[str, float]:
        if not cohorts:
            raise ValueError("Cohorts are required")
        rows: List[Dict[str, float]] = []
        ordered = list(cohorts)
        if training:
            random.Random(self.config.seed + epoch).shuffle(ordered)
        for cohort_index, cohort in enumerate(ordered):
            repeats = self.config.batches_per_cohort if training else max(
                1, self.config.batches_per_cohort // 2
            )
            rng = np.random.default_rng(
                self.config.seed
                + 1009 * epoch
                + 9176 * cohort_index
                + (0 if training else 10_000_000)
            )
            for _ in range(repeats):
                rows.append(self._step(cohort, rng, training=training))
        return {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }

    def fit(
        self,
        training: Sequence[CorpusSnapshotCohort],
        validation: Sequence[CorpusSnapshotCohort],
        *,
        state_path: Path,
        input_signature: str,
    ) -> Dict[str, object]:
        if not training or not validation:
            raise ValueError("Whole-study training and validation cohorts are required")
        train_studies = {value.study_id for value in training}
        validation_studies = {value.study_id for value in validation}
        if train_studies.intersection(validation_studies):
            raise ValueError("Study leakage between training and validation")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        initial_field_sha = state_dict_sha256(self.model.field.state_dict())
        initial_representation_sha = state_dict_sha256(
            representation_state_dict(self.model)
        )
        best_loss = float("inf")
        best_state = None
        history: List[dict] = []
        start_epoch = 0
        if state_path.is_file():
            try:
                state = torch.load(
                    state_path,
                    map_location=next(self.model.parameters()).device,
                    weights_only=False,
                )
            except TypeError:
                state = torch.load(state_path, map_location=next(self.model.parameters()).device)
            if state.get("config") != vars(self.config) or state.get("input_signature") != input_signature:
                raise RuntimeError("Resume checkpoint inputs/configuration changed")
            self.model.load_state_dict(state["model_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            if state_dict_sha256(self.model.field.state_dict()) != state["initial_field_sha256"]:
                raise RuntimeError("Circuit field changed inside snapshot-only resume state")
            initial_field_sha = state["initial_field_sha256"]
            initial_representation_sha = state["initial_representation_sha256"]
            best_loss = float(state["best_loss"])
            best_state = state["best_state"]
            history = list(state["history"])
            start_epoch = int(state["epoch_completed"]) + 1
            print(f"RESUME: expanded-corpus pretraining from epoch {start_epoch}", flush=True)

        for epoch in range(start_epoch, self.config.epochs):
            self.model.train()
            train_metrics = self._epoch(training, training=True, epoch=epoch)
            self.model.eval()
            with torch.no_grad():
                validation_metrics = self._epoch(validation, training=False, epoch=epoch)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "train_paired_contrastive": train_metrics["paired_contrastive"],
                "train_unpaired_distribution": train_metrics["unpaired_distribution"],
                "validation_paired_contrastive": validation_metrics["paired_contrastive"],
                "validation_unpaired_distribution": validation_metrics["unpaired_distribution"],
            }
            history.append(row)
            print(
                f"Epoch {epoch:03d} | train {row['train_loss']:.5f} | "
                f"validation {row['validation_loss']:.5f}",
                flush=True,
            )
            if row["validation_loss"] < best_loss:
                best_loss = row["validation_loss"]
                best_state = copy.deepcopy(self.model.state_dict())
            temporary = state_path.with_suffix(state_path.suffix + ".tmp")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "epoch_completed": epoch,
                    "config": vars(self.config),
                    "input_signature": input_signature,
                    "initial_field_sha256": initial_field_sha,
                    "initial_representation_sha256": initial_representation_sha,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "history": history,
                },
                temporary,
            )
            os.replace(temporary, state_path)

        if best_state is None:
            raise RuntimeError("No validation-selected corpus checkpoint")
        self.model.load_state_dict(best_state)
        final_field_sha = state_dict_sha256(self.model.field.state_dict())
        final_representation_sha = state_dict_sha256(
            representation_state_dict(self.model)
        )
        if final_field_sha != initial_field_sha:
            raise RuntimeError("Snapshot loss modified temporal circuit-field parameters")
        if final_representation_sha == initial_representation_sha:
            raise RuntimeError("Snapshot training did not update the representation modules")
        optimized_ids = {id(parameter) for parameter in self.parameters}
        field_ids = {id(parameter) for parameter in self.model.field.parameters()}
        return {
            "best_validation_loss": best_loss,
            "history": history,
            "training_cohorts": [value.cohort_id for value in training],
            "validation_cohorts": [value.cohort_id for value in validation],
            "training_studies": sorted(train_studies),
            "validation_studies": sorted(validation_studies),
            "training_pairing_modes": {
                "exact": sum(value.pairing == "exact" for value in training),
                "unpaired_population": sum(
                    value.pairing == "unpaired_population" for value in training
                ),
            },
            "field_state_initial_sha256": initial_field_sha,
            "field_state_final_sha256": final_field_sha,
            "representation_state_initial_sha256": initial_representation_sha,
            "representation_state_final_sha256": final_representation_sha,
            "representation_updated": final_representation_sha != initial_representation_sha,
            "snapshot_optimizer_excludes_field": not bool(
                optimized_ids.intersection(field_ids)
            ),
            "field_parameters_require_grad": all(
                parameter.requires_grad for parameter in self.model.field.parameters()
            ),
            "sealed_test_evaluated": False,
        }


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def representation_state_dict(
    model: WLDMultistudyFoundationModel,
) -> Dict[str, Tensor]:
    """Return only the modules identifiable from cross-sectional snapshots."""

    state: Dict[str, Tensor] = {}
    state.update({f"encoder.{key}": value for key, value in model.encoder.state_dict().items()})
    state.update(
        {
            f"context_network.{key}": value
            for key, value in model.context_network.state_dict().items()
        }
    )
    return state


def _registry_splits(path: Path) -> Tuple[Dict[str, str], set[str]]:
    registry = _json(path)
    studies = registry.get("studies", {})
    if isinstance(studies, list):
        splits = {str(row["accession"]): str(row.get("split", "train")) for row in studies}
    else:
        splits = {str(key): str(value.get("split", "")) for key, value in studies.items()}
    sealed = set(registry.get("sealed_exclusions", []))
    sealed.update(key for key, split in splits.items() if split == "sealed_test")
    return splits, sealed


def prepare_human_records(
    phase_a_root: Path,
    expansion_root: Path,
    phase_a_sources: Path,
    expansion_sources: Path,
    output_root: Path,
) -> Tuple[List[BundleRecord], List[dict]]:
    phase_a_report = _json(phase_a_root / "phase_a_ingestion_report.json")
    expansion_report = _json(expansion_root / "wld_corpus_expansion_report.json")
    if phase_a_report.get("sealed_test_downloaded") is not False:
        raise RuntimeError("Phase A indicates a sealed study was downloaded")
    if expansion_report.get("sealed_test_downloaded") is not False:
        raise RuntimeError("Expansion indicates a sealed study was downloaded")
    phase_splits, phase_sealed = _registry_splits(phase_a_sources)
    expansion_splits, expansion_sealed = _registry_splits(expansion_sources)
    sealed = phase_sealed | expansion_sealed
    atlas_root = phase_a_root / "training_atlas" / "homo_sapiens_grch38"
    if not (atlas_root / "atlas_manifest.json").is_file():
        raise FileNotFoundError("Missing Phase A human training atlas")

    records: List[BundleRecord] = []
    phase_harmonized = phase_a_root / "harmonized" / "homo_sapiens_grch38"
    for manifest_path in sorted(phase_harmonized.glob("*/bundle_manifest.json")):
        manifest = verify_bundle(manifest_path.parent)
        study_id = str(manifest["study_id"])
        if study_id in sealed:
            raise RuntimeError(f"Sealed Phase A bundle entered development: {study_id}")
        split = phase_splits.get(study_id, "")
        if split not in {"train", "validation"}:
            continue
        records.append(
            BundleRecord(
                root=manifest_path.parent,
                cohort_id=str(manifest["cohort_id"]),
                study_id=study_id,
                tissue=str(manifest.get("tissue", "")),
                split=split,
                pairing=str(manifest["pairing"]["pairing"]),
                source="phase_a",
            )
        )

    expansion_registry = _json(expansion_sources)
    expansion_by_cohort = {
        str(row["cohort_id"]): row for row in expansion_registry.get("cohorts", [])
    }
    staged_species: List[dict] = []
    for manifest_path in sorted((expansion_root / "bundles").glob("*/bundle_manifest.json")):
        source_manifest = verify_bundle(manifest_path.parent)
        cohort_id = str(source_manifest["cohort_id"])
        study_id = str(source_manifest["study_id"])
        if study_id in sealed:
            raise RuntimeError(f"Sealed expansion bundle entered development: {study_id}")
        if source_manifest.get("species") != "Homo sapiens" or source_manifest.get("genome_build") != "GRCh38":
            staged_species.append(
                {
                    "cohort_id": cohort_id,
                    "species": source_manifest.get("species"),
                    "genome_build": source_manifest.get("genome_build"),
                    "reason": "requires species/build-specific circuit and contact priors",
                }
            )
            continue
        declared = expansion_by_cohort.get(cohort_id, {})
        split = str(declared.get("split", expansion_splits.get(study_id, "train")))
        if split != "train":
            raise RuntimeError("Expansion cohorts must not alter the frozen validation split")
        projected_root = output_root / "projected_expansion" / cohort_id
        if (projected_root / "bundle_manifest.json").is_file():
            projected_manifest = verify_bundle(projected_root)
        else:
            projected_manifest = project_bundle_to_atlas(
                manifest_path.parent, atlas_root, projected_root
            )
        records.append(
            BundleRecord(
                root=projected_root,
                cohort_id=cohort_id,
                study_id=study_id,
                tissue=str(source_manifest.get("tissue", declared.get("tissue", ""))),
                split="train",
                pairing=str(projected_manifest["pairing"]["pairing"]),
                source="corpus_expansion",
            )
        )

    cohort_ids = [value.cohort_id for value in records]
    if len(set(cohort_ids)) != len(cohort_ids):
        raise RuntimeError("Duplicate cohort IDs in combined corpus")
    if not any(value.split == "train" for value in records) or not any(
        value.split == "validation" for value in records
    ):
        raise RuntimeError("Combined corpus requires train and validation cohorts")
    train_studies = {value.study_id for value in records if value.split == "train"}
    validation_studies = {value.study_id for value in records if value.split == "validation"}
    if train_studies.intersection(validation_studies):
        raise RuntimeError("Whole-study validation leakage in combined corpus")
    return records, staged_species


def _input_signature(records: Sequence[BundleRecord]) -> str:
    payload = [
        {
            "cohort_id": value.cohort_id,
            "study_id": value.study_id,
            "split": value.split,
            "pairing": value.pairing,
            "manifest_sha256": sha256_file(value.root / "bundle_manifest.json"),
        }
        for value in sorted(records, key=lambda row: row.cohort_id)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_state_dict(path: Path, device: torch.device) -> Mapping[str, Tensor]:
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, Mapping):
        raise ValueError("Phase B checkpoint is not a state dictionary")
    return value


def run_real_corpus_pretraining(
    phase_a_root: Path,
    phase_b_root: Path,
    expansion_root: Path,
    phase_a_sources: Path,
    expansion_sources: Path,
    output_root: Path,
    config: CorpusPretrainingConfig,
    *,
    device: Optional[str] = None,
) -> dict:
    phase_a_report_path = phase_a_root / "phase_a_ingestion_report.json"
    expansion_report_path = expansion_root / "wld_corpus_expansion_report.json"
    phase_b_report_path = phase_b_root / "snapshot_pretraining" / "wld_phase_b_pretraining.json"
    phase_b_checkpoint = phase_b_root / "snapshot_pretraining" / "wld_phase_b_snapshot_model.pt"
    priors_root = phase_b_root / "priors" / "homo_sapiens_grch38"
    for path in (
        phase_a_report_path,
        expansion_report_path,
        phase_b_report_path,
        phase_b_checkpoint,
        priors_root / "prior_manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    phase_b_report = _json(phase_b_report_path)
    if phase_b_report.get("sealed_test_downloaded") is not False or phase_b_report.get("sealed_test_evaluated") is not False:
        raise RuntimeError("Phase B checkpoint crossed the sealed-test boundary")
    if phase_b_report.get("checkpoint_sha256") != sha256_file(phase_b_checkpoint):
        raise RuntimeError("Phase B checkpoint hash mismatch")
    verify_phase_b_priors(priors_root)

    final_report_path = output_root / "wld_corpus_pretraining_report.json"
    final_model_path = output_root / "wld_corpus_pretrained_model.pt"
    expected_inputs = {
        "phase_a_report_sha256": sha256_file(phase_a_report_path),
        "phase_b_checkpoint_sha256": sha256_file(phase_b_checkpoint),
        "expansion_report_sha256": sha256_file(expansion_report_path),
        "phase_a_sources_sha256": sha256_file(phase_a_sources),
        "expansion_sources_sha256": sha256_file(expansion_sources),
    }
    if final_report_path.is_file() and final_model_path.is_file():
        existing = _json(final_report_path)
        if existing.get("input_sha256") != expected_inputs:
            raise RuntimeError("Completed corpus-pretraining inputs changed")
        if existing.get("config") != vars(config):
            raise RuntimeError("Completed corpus-pretraining configuration changed")
        if existing.get("checkpoint_sha256") != sha256_file(final_model_path):
            raise RuntimeError("Completed corpus-pretraining checkpoint hash mismatch")
        if existing.get("sealed_test_evaluated") is not False or existing.get("attractor_claim") is not False:
            raise RuntimeError("Completed corpus report crossed its claim boundary")
        print("PASS: completed expanded-corpus pretraining restored; skipping retraining")
        return existing

    records, staged_species = prepare_human_records(
        phase_a_root,
        expansion_root,
        phase_a_sources,
        expansion_sources,
        output_root,
    )
    signature = _input_signature(records)
    feature_vocab = _json(priors_root / "feature_vocab.json")
    training_records = [value for value in records if value.split == "train"]
    validation_records = [value for value in records if value.split == "validation"]
    training = [
        load_corpus_cohort(
            record,
            feature_vocab,
            max_cells=config.max_cells_per_cohort,
            seed=config.seed + index,
        )
        for index, record in enumerate(training_records)
    ]
    validation = [
        load_corpus_cohort(
            record,
            feature_vocab,
            max_cells=config.max_cells_per_cohort,
            seed=config.seed + 10000 + index,
        )
        for index, record in enumerate(validation_records)
    ]

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    priors = load_phase_b_priors(priors_root, resolved_device)
    model = WLDMultistudyFoundationModel(
        priors, context_covariate_dim=0, context_dim=32
    ).to(resolved_device)
    model.load_state_dict(_load_state_dict(phase_b_checkpoint, resolved_device))
    trainer = PairingAwareCorpusPretrainer(model, config)
    output_root.mkdir(parents=True, exist_ok=True)
    development = trainer.fit(
        training,
        validation,
        state_path=output_root / "wld_corpus_training_state.pt",
        input_signature=signature,
    )
    temporary_model = final_model_path.with_suffix(final_model_path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary_model)
    os.replace(temporary_model, final_model_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "pairing-aware expanded-corpus representation pretraining; not temporal dynamics",
        "device": str(resolved_device),
        "input_sha256": expected_inputs,
        "combined_bundle_signature": signature,
        "config": vars(config),
        "checkpoint_sha256": sha256_file(final_model_path),
        "architecture": architecture_contract(model),
        "development": development,
        "training_contract": {
            "exact_pairs_use_cell_contrastive_loss": True,
            "unpaired_populations_use_distributional_loss": True,
            "expression_or_label_pairing_used": False,
            "study_donor_barcode_or_cell_label_enter_encoder": False,
            "context_learned_from_measured_modalities": True,
            "phase_a_validation_study_unchanged": True,
            "snapshot_ode_kinetics_updated": False,
            "field_parameters_available_for_temporal_finetuning": True,
        },
        "staged_nonhuman_cohorts": staged_species,
        "limitations": [
            "Mouse cohorts require mouse-specific motif, contact, signed-circuit and signaling priors.",
            "Cross-sectional snapshots pretrain state representation but cannot identify temporal rates.",
            "Attractor claims require longitudinal or perturbational fine-tuning followed by sealed evaluation.",
        ],
        "sealed_test_downloaded": False,
        "sealed_test_evaluated": False,
        "model_assessed_on_sealed_test": False,
        "ode_kinetics_fitted_from_snapshots": False,
        "attractor_claim": False,
    }
    atomic_json(final_report_path, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--expansion-root", type=Path, required=True)
    parser.add_argument("--phase-a-sources", type=Path, required=True)
    parser.add_argument("--expansion-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches-per-cohort", type=int, default=8)
    parser.add_argument("--max-cells-per-cohort", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    report = run_real_corpus_pretraining(
        args.phase_a_root,
        args.phase_b_root,
        args.expansion_root,
        args.phase_a_sources,
        args.expansion_sources,
        args.output,
        CorpusPretrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_cohort=args.batches_per_cohort,
            max_cells_per_cohort=args.max_cells_per_cohort,
            seed=args.seed,
        ),
        device=args.device,
    )
    print("\n" + "=" * 76)
    print("COMPLETE: WLD EXPANDED-CORPUS REPRESENTATION PRETRAINING")
    print("=" * 76)
    print(f"Best validation loss: {report['development']['best_validation_loss']:.6f}")
    print(f"Training studies: {report['development']['training_studies']}")
    print(f"Validation studies: {report['development']['validation_studies']}")
    print(f"Pairing modes: {report['development']['training_pairing_modes']}")
    print("Snapshot ODE kinetics updated: False")
    print("Sealed test evaluated: False")
    print("Attractor claim: False")
    print(f"Report: {args.output / 'wld_corpus_pretraining_report.json'}")


if __name__ == "__main__":
    main()


__all__ = [
    "BundleRecord",
    "CorpusPretrainingConfig",
    "CorpusSnapshotCohort",
    "PairingAwareCorpusPretrainer",
    "covariance_alignment_loss",
    "feature_moment_loss",
    "load_corpus_cohort",
    "prepare_human_records",
    "representation_state_dict",
    "run_real_corpus_pretraining",
    "state_dict_sha256",
]
