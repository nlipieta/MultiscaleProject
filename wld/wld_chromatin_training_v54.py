"""Real whole-target development for WLD v5.4 chromatin responses.

The GSE161002 CRISPR-sciATAC observations are population snapshots, not paired
trajectories.  Training therefore compares unpaired predicted and observed
accessibility distributions.  Test targets are never materialized by this
module.  Validation targets are whole unseen perturbations.
"""

from __future__ import annotations

import bz2
import copy
import csv
import gzip
import hashlib
import json
import lzma
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import Tensor

from wld_chromatin_response_v54 import (
    ChromatinRoutePriors,
    WLDChromatinResponseModel,
    architecture_contract,
    degree_preserving_bipartite_shuffle,
)
from wld_foundation_data import sha256_file
from wld_foundation_model_v4 import WLDMultistudyFoundationModel
from wld_phase_b_priors import load_phase_b_priors


SAFE_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _open_text(path: Path):
    suffixes = set(path.suffixes)
    if ".gz" in suffixes:
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if ".xz" in suffixes:
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    if ".bz2" in suffixes:
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _symbol(row: Mapping[str, str], side: str) -> str:
    candidates = (
        f"{side}_genesymbol",
        f"{side}_gene_symbol",
        f"{side}_symbol",
        side,
    )
    for field in candidates:
        value = str(row.get(field, "")).strip().upper()
        if value and SAFE_SYMBOL.fullmatch(value):
            return value
    return ""


def _evidence(row: Mapping[str, str]) -> float:
    try:
        effort = float(row.get("curation_effort", "") or 0.0)
    except (TypeError, ValueError):
        effort = 0.0
    references = {
        value
        for value in re.split(r"[;,]", str(row.get("references", "")))
        if value.strip()
    }
    sources = {
        value
        for value in re.split(r"[;,]", str(row.get("sources", "")))
        if value.strip()
    }
    return max(1.0, effort, math.sqrt(max(len(references), 1)), math.sqrt(max(len(sources), 1)))


def compile_regulator_tf_routes(
    interaction_path: Path,
    regulators: Sequence[str],
    tfs: Sequence[str],
    output_root: Path,
    *,
    max_tfs_per_regulator: int = 16,
) -> Dict[str, object]:
    """Compile direct/two-hop protein-association routes to modeled TFs.

    The interaction source supplies topology and evidence confidence only.  It
    does not supply an opening/closing sign.  That sign is learned later from
    training-target CRISPR-sciATAC responses on the fixed route mask.
    """

    interaction_path = Path(interaction_path)
    if not interaction_path.is_file():
        raise FileNotFoundError(interaction_path)
    output_root.mkdir(parents=True, exist_ok=True)
    regulator_vocab = [str(value).upper() for value in regulators]
    tf_vocab = [str(value).upper() for value in tfs]
    regulator_set, tf_set = set(regulator_vocab), set(tf_vocab)
    needed = regulator_set | tf_set
    adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
    raw_rows = retained_rows = 0

    with _open_text(interaction_path) as handle:
        sample = handle.readline()
        delimiter = "\t" if "\t" in sample else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Interaction table has no header: {interaction_path}")
        for row in reader:
            raw_rows += 1
            source, target = _symbol(row, "source"), _symbol(row, "target")
            if not source or not target or source == target:
                continue
            # For a two-hop route, every required edge touches either a named
            # regulator or a modeled TF.  This bounds memory without losing a
            # regulator--intermediate--TF path.
            if source not in needed and target not in needed:
                continue
            score = _evidence(row)
            adjacency[source][target] = max(adjacency[source].get(target, 0.0), score)
            adjacency[target][source] = max(adjacency[target].get(source, 0.0), score)
            retained_rows += 1

    positive = [value for neighbors in adjacency.values() for value in neighbors.values()]
    maximum = max(positive, default=1.0)
    regulator_index = {value: index for index, value in enumerate(regulator_vocab)}
    tf_index = {value: index for index, value in enumerate(tf_vocab)}
    route = np.zeros((len(regulator_vocab), len(tf_vocab)), dtype=np.float32)
    route_records: List[Dict[str, object]] = []

    # Index TF neighbors by intermediate for efficient two-hop intersection.
    tf_by_intermediate: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for tf in tf_vocab:
        for intermediate, score in adjacency.get(tf, {}).items():
            tf_by_intermediate[intermediate].append((tf, score))

    for regulator in regulator_vocab:
        candidates: Dict[str, Tuple[float, str, str]] = {}
        if regulator in tf_index:
            candidates[regulator] = (1.0, "self", regulator)
        for tf in tf_vocab:
            score = adjacency.get(regulator, {}).get(tf)
            if score is not None:
                confidence = min(1.0, score / maximum)
                candidates[tf] = max(
                    candidates.get(tf, (0.0, "", "")),
                    (confidence, "direct", ""),
                )
        for intermediate, first_score in adjacency.get(regulator, {}).items():
            for tf, second_score in tf_by_intermediate.get(intermediate, ()):
                confidence = 0.5 * math.sqrt(
                    min(1.0, first_score / maximum)
                    * min(1.0, second_score / maximum)
                )
                if confidence > candidates.get(tf, (0.0, "", ""))[0]:
                    candidates[tf] = (confidence, "two_hop", intermediate)
        selected = sorted(
            candidates.items(), key=lambda item: (-item[1][0], item[0])
        )[: int(max_tfs_per_regulator)]
        for tf, (confidence, path_type, intermediate) in selected:
            route[regulator_index[regulator], tf_index[tf]] = float(confidence)
            route_records.append(
                {
                    "regulator": regulator,
                    "tf": tf,
                    "confidence": float(confidence),
                    "path_type": path_type,
                    "intermediate": intermediate,
                }
            )

    covered = [
        regulator_vocab[index]
        for index in range(len(regulator_vocab))
        if np.any(route[index] > 0)
    ]
    if len(covered) < max(10, int(0.25 * len(regulator_vocab))):
        raise RuntimeError(
            f"Only {len(covered)}/{len(regulator_vocab)} perturbation targets have "
            "a supported protein-association route to the modeled TF vocabulary"
        )

    np.savez_compressed(
        output_root / "regulator_tf_routes.npz",
        regulator_tf_support=route,
    )
    with gzip.open(output_root / "regulator_tf_routes.tsv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["regulator", "tf", "confidence", "path_type", "intermediate"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(route_records)
    atomic_json(
        output_root / "route_vocab.json",
        {"regulators": regulator_vocab, "tfs": tf_vocab},
    )
    report = {
        "schema_version": "wld-v5.4-regulator-tf-routes",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interaction_source": str(interaction_path),
        "interaction_source_sha256": sha256_file(interaction_path),
        "source_semantics": (
            "curated protein interaction/causal association topology; direction and "
            "opening/closing sign are not inferred from physical association"
        ),
        "raw_rows": raw_rows,
        "rows_touching_named_nodes": retained_rows,
        "regulators": len(regulator_vocab),
        "tfs": len(tf_vocab),
        "route_edges": int(np.count_nonzero(route)),
        "covered_regulators": covered,
        "covered_regulator_count": len(covered),
        "unsupported_regulators": sorted(set(regulator_vocab) - set(covered)),
        "max_tfs_per_regulator": int(max_tfs_per_regulator),
        "candidate_membership_used_as_edge_evidence": False,
        "opening_closing_sign_fixed_from_association": False,
    }
    atomic_json(output_root / "route_manifest.json", report)
    return report


@dataclass
class ChromatinBundle:
    accessibility: np.ndarray
    targets: Tuple[str, ...]
    screens: Tuple[str, ...]
    splits: Tuple[str, ...]
    regulator_vocab: Tuple[str, ...]
    foundation_peaks: Tuple[str, ...]
    row_groups: Dict[Tuple[str, str, str], np.ndarray]

    def rows(self, split: str, screen: str, target: str) -> np.ndarray:
        return self.row_groups.get((split, screen, target), np.zeros(0, dtype=np.int64))

    def target_screen(self, split: str, target: str) -> str:
        values = {
            screen
            for (candidate_split, screen, candidate_target), rows in self.row_groups.items()
            if candidate_split == split and candidate_target == target and len(rows)
        }
        if len(values) != 1:
            raise RuntimeError(f"Expected one screen for {split}/{target}, found {sorted(values)}")
        return next(iter(values))

    def split_targets(self, split: str) -> List[str]:
        return sorted(
            {
                target
                for (candidate_split, _screen, target), rows in self.row_groups.items()
                if candidate_split == split and target != "NTC" and len(rows)
            }
        )


def load_chromatin_bundle(
    bundle_root: Path,
    prior_root: Path,
    regulator_vocab: Sequence[str],
) -> ChromatinBundle:
    bundle_root, prior_root = Path(bundle_root), Path(prior_root)
    manifest = json.loads((bundle_root / "wld_v53_ingestion_manifest.json").read_text())
    if manifest.get("claims", {}).get("test_evaluated") is not False:
        raise RuntimeError("v5.3 manifest no longer records a sealed target test")
    feature_vocab = json.loads((prior_root / "feature_vocab.json").read_text())
    foundation_peaks = tuple(map(str, feature_vocab["peaks"]))
    with gzip.open(bundle_root / "bins.GRCh38.2kb.tsv.gz", "rt") as handle:
        bins = [line.strip() for line in handle if line.strip()]
    bin_index = {value: index for index, value in enumerate(bins)}
    missing = [value for value in foundation_peaks if value not in bin_index]
    if missing:
        raise RuntimeError(f"v5.3 bundle lacks {len(missing)} frozen foundation peaks")
    columns = [bin_index[value] for value in foundation_peaks]

    metadata = []
    with gzip.open(bundle_root / "cells.tsv.gz", "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            # Test response values are not materialized in memory.  Their rows
            # remain in the durable sealed bundle for a later frozen decision.
            if row["split"] != "test":
                metadata.append(row)
    source_matrix = sparse.load_npz(
        bundle_root / "atac_counts.GRCh38.2kb.npz"
    ).tocsr()
    source_rows = np.asarray([int(row["row"]) for row in metadata], dtype=np.int64)
    accessibility = (
        source_matrix[source_rows][:, columns] > 0
    ).astype(np.float32).toarray()
    del source_matrix

    targets = tuple(str(row["target"]).upper() for row in metadata)
    screens = tuple(str(row["screen"]) for row in metadata)
    splits = tuple(str(row["split"]) for row in metadata)
    groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for index, (split, screen, target) in enumerate(zip(splits, screens, targets)):
        groups[(split, screen, target)].append(index)
    row_groups = {
        key: np.asarray(values, dtype=np.int64) for key, values in groups.items()
    }
    regulators = tuple(str(value).upper() for value in regulator_vocab)
    observed = {target for target in targets if target != "NTC"}
    unknown = observed - set(regulators)
    if unknown:
        raise RuntimeError(f"Observed perturbations absent from route vocabulary: {sorted(unknown)}")
    for split in ("train", "validation"):
        if not any(key[0] == split and key[2] == "NTC" for key in row_groups):
            raise RuntimeError(f"No {split} control cells")
    return ChromatinBundle(
        accessibility=accessibility,
        targets=targets,
        screens=screens,
        splits=splits,
        regulator_vocab=regulators,
        foundation_peaks=foundation_peaks,
        row_groups=row_groups,
    )


def _load_foundation(
    prior_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> WLDMultistudyFoundationModel:
    priors = load_phase_b_priors(prior_root, device)
    foundation = WLDMultistudyFoundationModel(
        priors, context_covariate_dim=0, context_dim=32
    ).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    foundation.load_state_dict(state, strict=True)
    return foundation


def _projections(features: int, count: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(features, count, generator=generator)
    value = F.normalize(value, dim=0)
    return value.to(device)


def sliced_wasserstein(left: Tensor, right: Tensor, projections: Tensor) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("SWD inputs must share a feature dimension")
    n = min(left.shape[0], right.shape[0])
    left_projection = torch.sort(left[:n] @ projections, dim=0).values
    right_projection = torch.sort(right[:n] @ projections, dim=0).values
    return (left_projection - right_projection).abs().mean()


def distribution_loss(
    prediction: Tensor,
    target: Tensor,
    projections: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    swd = sliced_wasserstein(prediction, target, projections)
    mean = F.mse_loss(prediction.mean(0), target.mean(0))
    variance = F.mse_loss(
        prediction.var(0, unbiased=False), target.var(0, unbiased=False)
    )
    total = swd + 2.0 * mean + 0.25 * variance
    return total, {
        "loss": float(total.detach()),
        "swd": float(swd.detach()),
        "mean_mse": float(mean.detach()),
        "variance_mse": float(variance.detach()),
    }


@dataclass
class ChromatinTrainingConfig:
    epochs: int = 32
    targets_per_epoch: int = 32
    batch_size: int = 64
    learning_rate: float = 2e-3
    representation_learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    integration_steps: int = 5
    horizon: float = 1.0
    projections: int = 32
    validation_cells_per_target: int = 128
    patience: int = 7
    shuffle_replicates: int = 2
    seed: int = 42


def _sample_rows(rows: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if not len(rows):
        raise RuntimeError("Cannot sample an empty cell group")
    return rng.choice(rows, size=int(count), replace=len(rows) < int(count))


def _intervention(
    target: str,
    regulator_index: Mapping[str, int],
    batch: int,
    device: torch.device,
) -> Tensor:
    value = torch.zeros((batch, len(regulator_index)), device=device)
    value[:, regulator_index[target]] = 1.0
    return value


def evaluate_model(
    model: WLDChromatinResponseModel,
    bundle: ChromatinBundle,
    split: str,
    support: Tensor,
    config: ChromatinTrainingConfig,
    *,
    seed: int,
    support_override: Optional[Tensor] = None,
) -> Dict[str, object]:
    device = next(model.parameters()).device
    regulator_index = {value: index for index, value in enumerate(bundle.regulator_vocab)}
    projection = _projections(
        len(bundle.foundation_peaks), config.projections, seed + 31, device
    )
    metrics = []
    model.eval()
    with torch.no_grad():
        for target_number, target in enumerate(bundle.split_targets(split)):
            screen = bundle.target_screen(split, target)
            target_rows = bundle.rows(split, screen, target)
            control_rows = bundle.rows(split, screen, "NTC")
            n = min(
                config.validation_cells_per_target,
                max(len(target_rows), 1),
                max(len(control_rows), 1),
            )
            rng = np.random.default_rng(seed + 1009 * (target_number + 1))
            target_sample = _sample_rows(target_rows, n, rng)
            control_sample = _sample_rows(control_rows, n, rng)
            observed = torch.as_tensor(bundle.accessibility[target_sample], device=device)
            control = torch.as_tensor(bundle.accessibility[control_sample], device=device)
            intervention = _intervention(target, regulator_index, n, device)
            prediction = model(
                control,
                intervention,
                horizon=config.horizon,
                steps=config.integration_steps,
                support_override=support_override,
            )["atac_t"]
            model_swd = float(sliced_wasserstein(prediction, observed, projection))
            persistence_swd = float(sliced_wasserstein(control, observed, projection))
            observed_response = observed.mean(0) - control.mean(0)
            predicted_response = prediction.mean(0) - control.mean(0)
            cosine = float(
                F.cosine_similarity(
                    predicted_response.unsqueeze(0),
                    observed_response.unsqueeze(0),
                    dim=1,
                    eps=1e-8,
                )[0]
            )
            route_edges = int(torch.count_nonzero(support[regulator_index[target]]))
            metrics.append(
                {
                    "target": target,
                    "screen": screen,
                    "cells": n,
                    "route_edges": route_edges,
                    "route_supported": bool(route_edges),
                    "model_swd": model_swd,
                    "persistence_swd": persistence_swd,
                    "gain_over_persistence": persistence_swd - model_swd,
                    "response_cosine": cosine,
                    "mean_absolute_predicted_change": float(predicted_response.abs().mean()),
                }
            )

    def aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
        if not rows:
            return {
                "targets": 0,
                "model_swd": float("nan"),
                "persistence_swd": float("nan"),
                "gain_over_persistence": float("nan"),
                "response_cosine": float("nan"),
            }
        return {
            "targets": len(rows),
            "model_swd": float(np.mean([row["model_swd"] for row in rows])),
            "persistence_swd": float(np.mean([row["persistence_swd"] for row in rows])),
            "gain_over_persistence": float(np.mean([row["gain_over_persistence"] for row in rows])),
            "response_cosine": float(np.mean([row["response_cosine"] for row in rows])),
        }

    supported = [row for row in metrics if row["route_supported"]]
    return {
        "split": split,
        "all_targets": aggregate(metrics),
        "route_supported_targets": aggregate(supported),
        "unsupported_targets": sorted(row["target"] for row in metrics if not row["route_supported"]),
        "per_target": metrics,
    }


def _fit_condition(
    name: str,
    prior_root: Path,
    checkpoint: Path,
    bundle: ChromatinBundle,
    route_support: Tensor,
    motif_support: Tensor,
    output_root: Path,
    config: ChromatinTrainingConfig,
    device: torch.device,
) -> Tuple[WLDChromatinResponseModel, Dict[str, object]]:
    condition_root = output_root / name
    condition_root.mkdir(parents=True, exist_ok=True)
    final_report = condition_root / "condition_report.json"
    final_model = condition_root / "best_model.pt"
    route_digest = hashlib.sha256(route_support.detach().cpu().numpy().tobytes()).hexdigest()
    if final_report.is_file() and final_model.is_file():
        report = json.loads(final_report.read_text())
        if report.get("route_sha256") != route_digest:
            raise RuntimeError(f"Completed {name} route topology changed")
        foundation = _load_foundation(prior_root, checkpoint, device)
        model = WLDChromatinResponseModel(
            foundation,
            ChromatinRoutePriors(route_support.to(device), motif_support.to(device)),
        ).to(device)
        try:
            state = torch.load(final_model, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(final_model, map_location=device)
        model.load_state_dict(state)
        print(f"PASS: restored completed {name}", flush=True)
        return model, report

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    foundation = _load_foundation(prior_root, checkpoint, device)
    model = WLDChromatinResponseModel(
        foundation,
        ChromatinRoutePriors(route_support.to(device), motif_support.to(device)),
    ).to(device)
    # Snapshot pretraining initializes biological state.  This perturbation
    # stage fine-tunes the ATAC encoder/context slowly; no varying quantity is
    # declared globally fixed.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.field.parameters(), "lr": config.learning_rate},
            {
                "params": model.foundation.encoder.parameters(),
                "lr": config.representation_learning_rate,
            },
            {
                "params": model.foundation.context_network.parameters(),
                "lr": config.representation_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    regulator_index = {value: index for index, value in enumerate(bundle.regulator_vocab)}
    covered_train = [
        target
        for target in bundle.split_targets("train")
        if int(torch.count_nonzero(route_support[regulator_index[target]]))
    ]
    if len(covered_train) < 5:
        raise RuntimeError(f"Only {len(covered_train)} route-supported training targets")
    projection = _projections(
        len(bundle.foundation_peaks), config.projections, config.seed + 17, device
    )
    state_path = condition_root / "training_state.pt"
    start_epoch = 0
    best_score = float("inf")
    best_state = None
    history: List[Dict[str, object]] = []
    waiting = 0
    if state_path.is_file():
        try:
            state = torch.load(state_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location=device)
        if state.get("route_sha256") != route_digest or state.get("config") != asdict(config):
            raise RuntimeError(f"Resume state for {name} does not match this run")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_state = state["best_state"]
        history = list(state["history"])
        waiting = int(state["waiting"])
        print(f"   resumed {name} at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        rng = np.random.default_rng(config.seed + 100_003 * epoch)
        selected = rng.choice(
            covered_train,
            size=config.targets_per_epoch,
            replace=len(covered_train) < config.targets_per_epoch,
        )
        losses = []
        for target in selected:
            target = str(target)
            screen = bundle.target_screen("train", target)
            target_rows = bundle.rows("train", screen, target)
            control_rows = bundle.rows("train", screen, "NTC")
            observed_rows = _sample_rows(target_rows, config.batch_size, rng)
            control_sample = _sample_rows(control_rows, config.batch_size, rng)
            observed = torch.as_tensor(bundle.accessibility[observed_rows], device=device)
            control = torch.as_tensor(bundle.accessibility[control_sample], device=device)
            intervention = _intervention(
                target, regulator_index, config.batch_size, device
            )
            prediction = model(
                control,
                intervention,
                horizon=config.horizon,
                steps=config.integration_steps,
            )["atac_t"]
            loss, _ = distribution_loss(prediction, observed, projection)
            # Small regularization keeps the response field finite without
            # forcing cell-varying gains or rates to a shared constant.
            regularization = 1e-5 * (
                model.field.raw_tf_gain.square().mean()
                + model.field.raw_motif_gain.square().mean()
            )
            total = loss + regularization
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.field.parameters())
                + list(model.foundation.encoder.parameters())
                + list(model.foundation.context_network.parameters()),
                5.0,
            )
            optimizer.step()
            losses.append(float(loss.detach()))

        validation = evaluate_model(
            model,
            bundle,
            "validation",
            route_support,
            config,
            # Fixed validation subsample/projections across epochs; checkpoint
            # selection must not improve merely because a different group of
            # validation cells was drawn.
            seed=config.seed + 500_001,
        )
        score = validation["route_supported_targets"]["model_swd"]
        if not math.isfinite(score):
            raise RuntimeError("No route-supported validation target for checkpoint selection")
        improved = score < best_score - 1e-6
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation": validation["route_supported_targets"],
            "improved": improved,
        }
        history.append(row)
        print(
            f"   {name} epoch {epoch:03d} | train {row['training_loss']:.6f} | "
            f"val SWD {score:.6f} | persistence "
            f"{validation['route_supported_targets']['persistence_swd']:.6f}",
            flush=True,
        )
        temporary_state = state_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "epoch": epoch,
                "route_sha256": route_digest,
                "config": asdict(config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_score": best_score,
                "best_state": best_state,
                "history": history,
                "waiting": waiting,
            },
            temporary_state,
        )
        os.replace(temporary_state, state_path)
        if waiting >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No validation-selected checkpoint for {name}")
    model.load_state_dict(best_state)
    final_validation = evaluate_model(
        model,
        bundle,
        "validation",
        route_support,
        config,
        seed=config.seed + 900_001,
    )
    temporary_model = final_model.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), temporary_model)
    os.replace(temporary_model, final_model)
    report = {
        "condition": name,
        "route_sha256": route_digest,
        "best_validation_swd_during_training": best_score,
        "epochs_attempted": len(history),
        "history": history,
        "final_validation": final_validation,
        "checkpoint_sha256": sha256_file(final_model),
    }
    atomic_json(final_report, report)
    return model, report


def run_chromatin_response_development(
    prior_root: Path,
    foundation_checkpoint: Path,
    bundle_root: Path,
    route_root: Path,
    output_root: Path,
    config: ChromatinTrainingConfig,
    *,
    device: Optional[str] = None,
) -> Dict[str, object]:
    prior_root = Path(prior_root)
    foundation_checkpoint = Path(foundation_checkpoint)
    route_root = Path(route_root)
    output_root = Path(output_root)
    final_report = output_root / "wld_v54_chromatin_response_report.json"
    if final_report.is_file():
        existing = json.loads(final_report.read_text())
        if (
            existing.get("claims", {}).get("test_targets_evaluated") is False
            and existing.get("claims", {}).get("attractor_claim") is False
        ):
            print("PASS: completed WLD v5.4 development restored", flush=True)
            return existing

    vocab = json.loads((route_root / "route_vocab.json").read_text())
    arrays = np.load(route_root / "regulator_tf_routes.npz", allow_pickle=False)
    route = torch.as_tensor(arrays["regulator_tf_support"], dtype=torch.float32)
    bundle = load_chromatin_bundle(bundle_root, prior_root, vocab["regulators"])
    foundation_priors = load_phase_b_priors(prior_root, "cpu")
    motif = foundation_priors.peak_tf_motif.transpose(0, 1).float()
    if motif.shape != (route.shape[1], len(bundle.foundation_peaks)):
        raise RuntimeError("Route, motif and v5.3 foundation views disagree")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root.mkdir(parents=True, exist_ok=True)
    conditions: Dict[str, Dict[str, object]] = {}
    true_model, true_report = _fit_condition(
        "true_routes",
        prior_root,
        foundation_checkpoint,
        bundle,
        route,
        motif,
        output_root,
        config,
        resolved_device,
    )
    conditions["true_routes"] = true_report

    shuffled_supports = []
    for replicate in range(config.shuffle_replicates):
        shuffled = degree_preserving_bipartite_shuffle(
            route, seed=config.seed + 10_000 + replicate
        )
        shuffled_supports.append(shuffled)
        _, control_report = _fit_condition(
            f"degree_shuffle_{replicate + 1}",
            prior_root,
            foundation_checkpoint,
            bundle,
            shuffled,
            motif,
            output_root,
            config,
            resolved_device,
        )
        conditions[f"degree_shuffle_{replicate + 1}"] = control_report

    frozen_zero = evaluate_model(
        true_model,
        bundle,
        "validation",
        route,
        config,
        seed=config.seed + 700_001,
        support_override=torch.zeros_like(route, device=resolved_device),
    )
    frozen_shuffles = [
        evaluate_model(
            true_model,
            bundle,
            "validation",
            route,
            config,
            seed=config.seed + 710_001 + replicate,
            support_override=shuffled.to(resolved_device),
        )
        for replicate, shuffled in enumerate(shuffled_supports)
    ]
    true_metrics = true_report["final_validation"]["route_supported_targets"]
    shuffled_metrics = [
        conditions[f"degree_shuffle_{index + 1}"]["final_validation"]["route_supported_targets"]
        for index in range(config.shuffle_replicates)
    ]
    shuffled_swd = [value["model_swd"] for value in shuffled_metrics if math.isfinite(value["model_swd"])]
    specificity = {
        "true_model_swd": true_metrics["model_swd"],
        "persistence_swd": true_metrics["persistence_swd"],
        "true_gain_over_persistence": true_metrics["gain_over_persistence"],
        "mean_retrained_degree_shuffle_swd": float(np.mean(shuffled_swd)) if shuffled_swd else float("nan"),
        "true_advantage_over_retrained_shuffles": (
            float(np.mean(shuffled_swd)) - true_metrics["model_swd"]
            if shuffled_swd else float("nan")
        ),
        "frozen_zero_swd": frozen_zero["route_supported_targets"]["model_swd"],
        "frozen_zero_effect": (
            frozen_zero["route_supported_targets"]["model_swd"] - true_metrics["model_swd"]
        ),
        "frozen_shuffle_swds": [
            value["route_supported_targets"]["model_swd"] for value in frozen_shuffles
        ],
    }
    report = {
        "schema_version": "wld-v5.4-chromatin-response-development",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "GSE161002 unpaired CRISPR-sciATAC transient response development on "
            "whole held-out validation targets"
        ),
        "device": str(resolved_device),
        "config": asdict(config),
        "architecture": architecture_contract(true_model),
        "route_manifest": json.loads((route_root / "route_manifest.json").read_text()),
        "conditions": conditions,
        "frozen_ablations": {
            "zero_routes": frozen_zero,
            "degree_shuffles": frozen_shuffles,
            "no_retraining": True,
        },
        "specificity": specificity,
        "claims": {
            "unpaired_population_training": True,
            "whole_validation_targets": True,
            "target_identity_in_encoder": False,
            "test_targets_evaluated": False,
            "muscle_J_L_evaluated": False,
            "external_sealed_studies_evaluated": False,
            "ode_kinetics_identified": False,
            "attractor_claim": False,
            "forced_circuit_specificity_claim": False,
        },
        "interpretation_rule": (
            "Claim route-specific transient response only if true routes beat persistence, "
            "retrained degree-preserving shuffles, and frozen route removal. Even a positive "
            "result does not establish a fixed point, basin, or attractor."
        ),
    }
    atomic_json(final_report, report)
    return report
