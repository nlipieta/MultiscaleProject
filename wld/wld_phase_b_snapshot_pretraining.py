"""Real multi-study snapshot pretraining for the WLD v4 representation.

Snapshot cohorts identify cross-modal cell state but do not identify a temporal
vector field.  This stage therefore trains the structured encoder and context
network with exact-barcode multimodal contrastive learning.  It deliberately
does not fit ODE kinetics from fabricated cell pairs or pretend snapshots are
time series.  The context-conditioned ODE remains available for a later
longitudinal/perturbational stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import Tensor

from wld_foundation_data import read_barcodes, read_bundle_features, sha256_file, verify_bundle
from wld_foundation_model_v4 import WLDMultistudyFoundationModel, architecture_contract
from wld_phase_b_priors import load_phase_b_priors, verify_phase_b_priors


@dataclass
class SnapshotCohort:
    cohort_id: str
    study_id: str
    barcodes: Tuple[str, ...]
    rna: sparse.csr_matrix
    atac: sparse.csr_matrix
    protein: Optional[sparse.csr_matrix]

    @property
    def cells(self) -> int:
        return int(self.rna.shape[0])

    def validate(self) -> None:
        if self.cells < 2 or self.atac.shape[0] != self.cells:
            raise ValueError("Snapshot RNA/ATAC cell dimensions disagree")
        if len(self.barcodes) != self.cells or len(set(self.barcodes)) != self.cells:
            raise ValueError("Snapshot barcodes must be unique and match the matrices")
        if self.protein is not None and self.protein.shape[0] != self.cells:
            raise ValueError("Protein cells do not match the exact paired snapshot")


def _indices(feature_path: Path, selected: Sequence[str]) -> List[int]:
    names = [value["feature_name"] for value in read_bundle_features(feature_path)]
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise ValueError(f"Selected prior features are absent from the harmonized bundle: {missing[:5]}")
    return [lookup[name] for name in selected]


def load_snapshot_cohort(
    bundle_root: Path,
    feature_vocab: Mapping[str, Sequence[str]],
    *,
    max_cells: int,
    seed: int,
) -> SnapshotCohort:
    manifest = verify_bundle(bundle_root)
    if not {"rna", "atac"}.issubset(manifest["modalities"]):
        raise ValueError(f"Snapshot cohort lacks RNA/ATAC: {manifest['cohort_id']}")
    rna_root = bundle_root / "modalities" / "rna"
    atac_root = bundle_root / "modalities" / "atac"
    rna_barcodes = read_barcodes(rna_root / "barcodes.tsv.gz")
    atac_barcodes = read_barcodes(atac_root / "barcodes.tsv.gz")
    if rna_barcodes != atac_barcodes:
        raise ValueError(
            f"Refusing fabricated RNA/ATAC cell pairs for {manifest['cohort_id']}"
        )
    rna_columns = _indices(rna_root / "features.tsv.gz", feature_vocab["genes"])
    atac_columns = _indices(atac_root / "features.tsv.gz", feature_vocab["peaks"])
    rna = sparse.load_npz(rna_root / "counts.csr.npz").tocsr()[:, rna_columns]
    atac = sparse.load_npz(atac_root / "counts.csr.npz").tocsr()[:, atac_columns]
    protein = None
    if "protein" in manifest["modalities"] and feature_vocab.get("proteins"):
        protein_root = bundle_root / "modalities" / "protein"
        if read_barcodes(protein_root / "barcodes.tsv.gz") == rna_barcodes:
            protein_columns = _indices(
                protein_root / "features.tsv.gz", feature_vocab["proteins"]
            )
            protein = sparse.load_npz(protein_root / "counts.csr.npz").tocsr()[:, protein_columns]
    if len(rna_barcodes) > max_cells:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(rna_barcodes), size=max_cells, replace=False))
        rna, atac = rna[rows], atac[rows]
        if protein is not None:
            protein = protein[rows]
        rna_barcodes = [rna_barcodes[index] for index in rows]
    result = SnapshotCohort(
        cohort_id=str(manifest["cohort_id"]),
        study_id=str(manifest["study_id"]),
        barcodes=tuple(rna_barcodes),
        rna=rna,
        atac=atac,
        protein=protein,
    )
    result.validate()
    return result


def _library_normalize(matrix: sparse.csr_matrix, scale: float) -> np.ndarray:
    dense = matrix.toarray().astype(np.float32, copy=False)
    totals = dense.sum(axis=1, keepdims=True)
    return dense * (float(scale) / np.maximum(totals, 1.0))


def tensor_batch(
    cohort: SnapshotCohort,
    rows: np.ndarray,
    device: torch.device,
) -> Dict[str, Optional[Tensor]]:
    rna = torch.as_tensor(_library_normalize(cohort.rna[rows], 1e4), device=device)
    atac = torch.as_tensor((cohort.atac[rows].toarray() > 0).astype(np.float32), device=device)
    protein = None
    if cohort.protein is not None:
        protein = torch.as_tensor(
            _library_normalize(cohort.protein[rows], 1e3), device=device
        )
    return {"rna": rna, "atac": atac, "protein": protein}


def symmetric_info_nce(left: Tensor, right: Tensor, temperature: float) -> Tensor:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
        raise ValueError("InfoNCE views must have the same [cells, features] shape")
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    logits = left @ right.transpose(0, 1) / float(temperature)
    target = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.transpose(0, 1), target))


def variance_floor(*values: Tensor, floor: float = 0.20) -> Tensor:
    terms = [F.relu(floor - value.std(dim=0, unbiased=False)).mean() for value in values]
    return torch.stack(terms).mean()


@dataclass
class SnapshotPretrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    batches_per_cohort: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    temperature: float = 0.15
    tf_alignment_weight: float = 0.25
    variance_weight: float = 0.10
    max_cells_per_cohort: int = 4096
    seed: int = 42


class SnapshotFoundationPretrainer:
    def __init__(
        self,
        model: WLDMultistudyFoundationModel,
        config: SnapshotPretrainingConfig,
    ) -> None:
        self.model = model
        self.config = config
        # Snapshot data do not identify kinetic rates.  We train only the
        # multimodal representation here rather than manufacturing dynamics.
        parameters = list(model.encoder.parameters()) + list(model.context_network.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def _loss(
        self,
        cohort: SnapshotCohort,
        rows: np.ndarray,
        *,
        training: bool,
    ) -> Tuple[Tensor, Dict[str, float]]:
        device = next(self.model.parameters()).device
        values = tensor_batch(cohort, rows, device)
        batch = len(rows)
        cues = torch.zeros((batch, self.model.priors.num_cues), device=device)
        atac_view = self.model.encoder(cues=cues, atac=values["atac"])
        rna_view = self.model.encoder(cues=cues, rna=values["rna"])
        atac_context = self.model.context_network(atac_view["biological_context"])
        rna_context = self.model.context_network(rna_view["biological_context"])
        contrastive = symmetric_info_nce(atac_context, rna_context, self.config.temperature)
        tf_alignment = 1.0 - F.cosine_similarity(
            torch.log1p(atac_view["tf"]), torch.log1p(rna_view["tf"]), dim=1
        ).mean()
        variance = variance_floor(atac_context, rna_context)
        total = (
            contrastive
            + self.config.tf_alignment_weight * tf_alignment
            + self.config.variance_weight * variance
        )

        protein_contrastive = total.new_zeros(())
        if values["protein"] is not None and int(torch.count_nonzero(self.model.priors.protein_signal)):
            protein_view = self.model.encoder(cues=cues, protein=values["protein"])
            protein_context = self.model.context_network(protein_view["biological_context"])
            protein_contrastive = 0.5 * (
                symmetric_info_nce(protein_context, atac_context, self.config.temperature)
                + symmetric_info_nce(protein_context, rna_context, self.config.temperature)
            )
            total = total + 0.25 * protein_contrastive

        if training:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters()) + list(self.model.context_network.parameters()), 5.0
            )
            self.optimizer.step()
        metrics = {
            "loss": float(total.detach()),
            "contrastive": float(contrastive.detach()),
            "tf_alignment": float(tf_alignment.detach()),
            "variance_floor": float(variance.detach()),
            "protein_contrastive": float(protein_contrastive.detach()),
        }
        return total, metrics

    def _cohort_epoch(
        self,
        cohorts: Sequence[SnapshotCohort],
        *,
        training: bool,
        epoch: int,
    ) -> Dict[str, float]:
        rows = []
        for cohort_index, cohort in enumerate(cohorts):
            rng = np.random.default_rng(
                self.config.seed + 1009 * epoch + 9176 * cohort_index + (0 if training else 10_000_000)
            )
            repeats = self.config.batches_per_cohort if training else max(1, self.config.batches_per_cohort // 2)
            for _ in range(repeats):
                size = min(self.config.batch_size, cohort.cells)
                selected = rng.choice(cohort.cells, size=size, replace=False)
                rows.append(self._loss(cohort, selected, training=training)[1])
        return {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }

    def fit(
        self,
        training: Sequence[SnapshotCohort],
        validation: Sequence[SnapshotCohort],
        *,
        state_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        if not training or not validation:
            raise ValueError("Training and validation studies are both required")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        best_loss = float("inf")
        best_state = None
        history = []
        start_epoch = 0
        if state_path is not None and state_path.is_file():
            try:
                state = torch.load(state_path, map_location=next(self.model.parameters()).device, weights_only=False)
            except TypeError:
                state = torch.load(state_path, map_location=next(self.model.parameters()).device)
            if state.get("config") != vars(self.config):
                raise RuntimeError("Resume checkpoint configuration does not match this run")
            self.model.load_state_dict(state["model_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            best_loss = float(state["best_loss"])
            best_state = state["best_state"]
            history = list(state["history"])
            start_epoch = int(state["epoch_completed"]) + 1
            print(f"RESUME: snapshot pretraining from epoch {start_epoch}", flush=True)
        for epoch in range(start_epoch, self.config.epochs):
            self.model.train()
            train_metrics = self._cohort_epoch(training, training=True, epoch=epoch)
            self.model.eval()
            with torch.no_grad():
                validation_metrics = self._cohort_epoch(validation, training=False, epoch=epoch)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "train_contrastive": train_metrics["contrastive"],
                "validation_contrastive": validation_metrics["contrastive"],
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
            if state_path is not None:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = state_path.with_suffix(state_path.suffix + ".tmp")
                torch.save(
                    {
                        "epoch_completed": epoch,
                        "config": vars(self.config),
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
            raise RuntimeError("No validation-selected snapshot checkpoint")
        self.model.load_state_dict(best_state)
        return {
            "best_validation_loss": best_loss,
            "history": history,
            "training_cohorts": [value.cohort_id for value in training],
            "validation_cohorts": [value.cohort_id for value in validation],
            "sealed_test_evaluated": False,
        }


def run_real_snapshot_pretraining(
    phase_a_root: Path,
    priors_root: Path,
    source_registry: Path,
    output_root: Path,
    config: SnapshotPretrainingConfig,
    *,
    device: Optional[str] = None,
) -> Dict[str, object]:
    final_report_path = output_root / "wld_phase_b_pretraining.json"
    final_model_path = output_root / "wld_phase_b_snapshot_model.pt"
    if final_report_path.is_file() and final_model_path.is_file():
        existing = json.loads(final_report_path.read_text())
        if (
            existing.get("sealed_test_downloaded") is False
            and existing.get("sealed_test_evaluated") is False
            and existing.get("attractor_claim") is False
        ):
            if existing.get("prior_manifest_sha256") != sha256_file(priors_root / "prior_manifest.json"):
                raise RuntimeError("Phase B priors changed after the completed checkpoint")
            if existing.get("phase_a_report_sha256") != sha256_file(phase_a_root / "phase_a_ingestion_report.json"):
                raise RuntimeError("Phase A report changed after the completed checkpoint")
            if existing.get("checkpoint_sha256") != sha256_file(final_model_path):
                raise RuntimeError("Completed Phase B checkpoint hash mismatch")
            print("PASS: completed Phase B pretraining restored; skipping retraining")
            return existing
    prior_manifest = verify_phase_b_priors(priors_root)
    registry = json.loads(source_registry.read_text())
    studies = registry["studies"]
    feature_vocab = json.loads((priors_root / "feature_vocab.json").read_text())
    harmonized = phase_a_root / "harmonized" / "homo_sapiens_grch38"
    train_roots, validation_roots = [], []
    for path in sorted(harmonized.glob("*/bundle_manifest.json")):
        manifest = json.loads(path.read_text())
        split = studies.get(manifest["study_id"], {}).get("split")
        if split == "sealed_test":
            raise RuntimeError("A sealed test bundle appeared in the development root")
        if split == "train":
            train_roots.append(path.parent)
        elif split == "validation":
            validation_roots.append(path.parent)
    if not train_roots or not validation_roots:
        raise RuntimeError("Whole-study train and validation cohorts are required")
    training = [
        load_snapshot_cohort(
            root, feature_vocab, max_cells=config.max_cells_per_cohort,
            seed=config.seed + index,
        )
        for index, root in enumerate(train_roots)
    ]
    validation = [
        load_snapshot_cohort(
            root, feature_vocab, max_cells=config.max_cells_per_cohort,
            seed=config.seed + 1000 + index,
        )
        for index, root in enumerate(validation_roots)
    ]
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    priors = load_phase_b_priors(priors_root, resolved_device)
    model = WLDMultistudyFoundationModel(
        priors, context_covariate_dim=0, context_dim=32
    ).to(resolved_device)
    trainer = SnapshotFoundationPretrainer(model, config)
    output_root.mkdir(parents=True, exist_ok=True)
    development = trainer.fit(
        training,
        validation,
        state_path=output_root / "wld_phase_b_training_state.pt",
    )
    temporary_model = output_root / "wld_phase_b_snapshot_model.pt.tmp"
    torch.save(model.state_dict(), temporary_model)
    os.replace(temporary_model, final_model_path)
    report = {
        "schema_version": "1.0",
        "scope": "multi-study exact-barcode snapshot representation pretraining; not temporal dynamics",
        "device": str(resolved_device),
        "prior_manifest_sha256": sha256_file(priors_root / "prior_manifest.json"),
        "phase_a_report_sha256": sha256_file(phase_a_root / "phase_a_ingestion_report.json"),
        "checkpoint_sha256": sha256_file(final_model_path),
        "architecture": architecture_contract(model),
        "development": development,
        "training_contract": {
            "study_ids_enter_encoder": False,
            "donor_ids_enter_encoder": False,
            "cell_labels_enter_encoder": False,
            "exact_barcode_pairing_required": True,
            "missing_modalities_masked": True,
            "condition_labels_guessed": False,
            "snapshot_ode_kinetics_updated": False,
            "reason_ode_not_updated": "snapshots do not identify temporal kinetics",
            "context_conditioned_variation_modules_present_for_temporal_stage": True,
        },
        "limitations": [
            "Current Phase A training diversity is not yet a large pan-tissue atlas.",
            "Mouse cohorts are not silently trained with human GRCh38 regulatory priors.",
            "Attractor or temporal claims require longitudinal/perturbational fine-tuning and sealed evaluation.",
        ],
        "sealed_test_downloaded": False,
        "sealed_test_evaluated": False,
        "attractor_claim": False,
    }
    temporary_report = final_report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_report, final_report_path)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches-per-cohort", type=int, default=8)
    parser.add_argument("--max-cells-per-cohort", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    config = SnapshotPretrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        batches_per_cohort=args.batches_per_cohort,
        max_cells_per_cohort=args.max_cells_per_cohort,
        seed=args.seed,
    )
    report = run_real_snapshot_pretraining(
        args.phase_a_root, args.priors, args.sources, args.output,
        config, device=args.device,
    )
    print("\nCOMPLETE: Phase B snapshot representation pretraining")
    print(json.dumps(report["development"], indent=2))
    print("No ODE dynamics or attractor claim was fitted from snapshots.")


if __name__ == "__main__":
    main()


__all__ = [
    "SnapshotCohort", "SnapshotFoundationPretrainer", "SnapshotPretrainingConfig",
    "load_snapshot_cohort", "run_real_snapshot_pretraining", "symmetric_info_nce",
]
