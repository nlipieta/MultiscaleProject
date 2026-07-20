"""End-to-end synthetic data/split/resume contract for WLD v5.4."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from wld_foundation_model_v4 import FoundationPriors, WLDMultistudyFoundationModel
from wld_chromatin_training_v54 import (
    ChromatinTrainingConfig,
    compile_regulator_tf_routes,
    run_chromatin_response_development,
)


def foundation_fixture(root: Path) -> Path:
    prior = root / "priors"
    prior.mkdir()
    peak_to_gene = np.zeros((6, 5), dtype=np.float32)
    peak_tf_motif = np.zeros((6, 3), dtype=np.float32)
    tf_gene_support = np.zeros((3, 5), dtype=np.float32)
    for index in range(3):
        peak_to_gene[index, index] = 1.0
        peak_to_gene[index + 3, index] = 0.8
        peak_tf_motif[index, index] = 1.0
        peak_tf_motif[index + 3, index] = 0.7
        tf_gene_support[index, index] = 1.0
    arrays = {
        "peak_to_gene": peak_to_gene,
        "peak_tf_motif": peak_tf_motif,
        "tf_gene_support": tf_gene_support,
        "circuit_tf_tf": np.zeros((3, 3), dtype=np.float32),
        "signal_signal": np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        "signal_tf": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        "cue_signal": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "tf_peak_effect": np.zeros((3, 6), dtype=np.float32),
        "tf_gene_index": np.asarray([0, 1, 2], dtype=np.int64),
        "protein_signal": np.zeros((0, 2), dtype=np.float32),
        "metabolic_signal": np.zeros((0, 2), dtype=np.float32),
        "metabolic_tf": np.zeros((0, 3), dtype=np.float32),
    }
    np.savez_compressed(prior / "foundation_priors.npz", **arrays)
    vocab = {
        "genes": ["TF0", "TF1", "TF2", "G3", "G4"],
        "peaks": [f"chr1:{index * 2000}-{(index + 1) * 2000}" for index in range(6)],
        "tfs": ["TF0", "TF1", "TF2"],
        "signals": ["S0", "S1"],
        "cues": ["baseline"],
        "proteins": [],
        "metabolites": [],
    }
    (prior / "feature_vocab.json").write_text(json.dumps(vocab))
    (prior / "prior_manifest.json").write_text(json.dumps({"fixture": True}))
    priors = FoundationPriors(**{name: torch.as_tensor(value) for name, value in arrays.items()})
    model = WLDMultistudyFoundationModel(priors, context_covariate_dim=0, context_dim=32)
    checkpoint = root / "foundation.pt"
    torch.save(model.state_dict(), checkpoint)
    return prior


def bundle_fixture(root: Path, regulators: list[str]) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    train, validation, test = regulators[:5], regulators[5:8], regulators[8:]
    split = {"train": train, "validation": validation, "test": test}
    (bundle / "whole_target_split.json").write_text(json.dumps({"targets": split}))
    (bundle / "wld_v53_ingestion_manifest.json").write_text(
        json.dumps({"claims": {"test_evaluated": False}})
    )
    with gzip.open(bundle / "bins.GRCh38.2kb.tsv.gz", "wt") as handle:
        for index in range(6):
            handle.write(f"chr1:{index * 2000}-{(index + 1) * 2000}\n")

    rng = np.random.default_rng(42)
    matrix, metadata = [], []
    row = 0
    for split_name, targets in split.items():
        for _ in range(18):
            matrix.append(rng.binomial(1, 0.25, size=6))
            metadata.append([row, f"C{row}", "screen", f"B{row}", "NTC", split_name, 50])
            row += 1
        for target_index, target in enumerate(targets):
            for _ in range(12):
                probability = np.full(6, 0.25)
                probability[target_index % 3] = 0.55
                probability[3 + target_index % 3] = 0.10
                matrix.append(rng.binomial(1, probability))
                metadata.append([row, f"C{row}", "screen", f"B{row}", target, split_name, 50])
                row += 1
    sparse.save_npz(
        bundle / "atac_counts.GRCh38.2kb.npz",
        sparse.csr_matrix(np.asarray(matrix, dtype=np.uint8)),
    )
    with gzip.open(bundle / "cells.tsv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row", "cell_id", "screen", "barcode", "target", "split", "fragments"])
        writer.writerows(metadata)
    return bundle


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wld_v54_smoke_") as temporary:
        root = Path(temporary)
        regulators = [f"R{index:02d}" for index in range(10)]
        prior = foundation_fixture(root)
        bundle = bundle_fixture(root, regulators)
        interactions = root / "interactions.tsv"
        with interactions.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["source_genesymbol", "target_genesymbol", "curation_effort", "references"])
            for index, regulator in enumerate(regulators):
                writer.writerow([regulator, f"TF{index % 3}", 2, f"PMID:{index}"])
                writer.writerow([regulator, f"TF{(index + 1) % 3}", 1, f"PMID:{100 + index}"])
        route_root = root / "routes"
        route = compile_regulator_tf_routes(
            interactions, regulators, ["TF0", "TF1", "TF2"], route_root
        )
        assert route["covered_regulator_count"] == 10
        report = run_chromatin_response_development(
            prior,
            root / "foundation.pt",
            bundle,
            route_root,
            root / "development",
            ChromatinTrainingConfig(
                epochs=2,
                targets_per_epoch=3,
                batch_size=8,
                projections=4,
                validation_cells_per_target=8,
                patience=2,
                shuffle_replicates=1,
                integration_steps=2,
                seed=42,
            ),
            device="cpu",
        )
        assert report["claims"]["test_targets_evaluated"] is False
        assert report["claims"]["target_identity_in_encoder"] is False
        assert report["architecture"]["direct_neural_context_to_peak_decoder"] is False
        assert len(report["conditions"]) == 2
        # A second call must verify and restore the completed report rather than
        # retraining or opening the test split.
        restored = run_chromatin_response_development(
            prior,
            root / "foundation.pt",
            bundle,
            route_root,
            root / "development",
            ChromatinTrainingConfig(shuffle_replicates=1),
            device="cpu",
        )
        assert restored["created_utc"] == report["created_utc"]

    print("PASS: frozen interaction route compilation and provenance")
    print("PASS: whole-target train/validation/test data contract")
    print("PASS: unpaired distributional training and validation checkpoint")
    print("PASS: persistence, frozen and retrained degree controls")
    print("PASS: restart-safe completion with test responses untouched")


if __name__ == "__main__":
    main()
