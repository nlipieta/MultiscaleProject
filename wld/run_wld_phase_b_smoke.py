"""Synthetic contract test for WLD Phase B priors and snapshot pretraining."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from wld_foundation_data import ModalityBlock, atomic_json, save_bundle, write_lines_gz
from wld_foundation_model_v4 import WLDMultistudyFoundationModel, architecture_contract
from wld_phase_b_priors import compile_phase_b_priors, load_phase_b_priors, verify_phase_b_priors
from wld_phase_b_snapshot_pretraining import (
    SnapshotFoundationPretrainer,
    SnapshotPretrainingConfig,
    load_snapshot_cohort,
)


def write_tsv(path: Path, fields, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def make_atlas(root: Path) -> None:
    root.mkdir(parents=True)
    genes = ["TF1", "TF2", "G1", "G2"]
    peaks = ["chr1:0-2000", "chr1:2000-4000", "chr1:4000-6000", "chr1:6000-8000"]
    write_lines_gz(root / "shared_genes.tsv.gz", genes)
    write_lines_gz(root / "shared_peak_bins.tsv.gz", peaks)
    write_lines_gz(root / "shared_proteins.tsv.gz", ["S1"])
    write_lines_gz(root / "shared_metabolites.tsv.gz", [])
    atomic_json(
        root / "atlas_manifest.json",
        {
            "schema_version": "1.0",
            "training_cohorts_only": True,
            "species": "Homo sapiens",
            "genome_build": "GRCh38",
            "genes": 4,
            "peak_bins": 4,
            "proteins": 1,
            "metabolites": 0,
            "peak_bin_size": 2000,
            "source_bundle_manifest_sha256": {"train": "fixture"},
        },
    )


def make_evidence(root: Path) -> None:
    root.mkdir(parents=True)
    write_tsv(
        root / "peak_gene_links.tsv",
        ("peak_id", "gene", "score", "evidence_source"),
        [
            ("chr1:100-300", "TF2", 5, "contact"),
            ("chr1:2100-2400", "TF1", 5, "contact"),
            ("chr1:4100-4500", "G1", 4, "contact"),
            ("chr1:6100-6500", "G2", 4, "contact"),
        ],
    )
    write_tsv(
        root / "motif_hits.tsv",
        ("peak_id", "tf", "score", "evidence_source"),
        [
            ("chr1:100-300", "TF1", 8, "motif"),
            ("chr1:2100-2400", "TF2", 8, "motif"),
            ("chr1:4100-4500", "TF1", 7, "motif"),
            ("chr1:6100-6500", "TF2", 7, "motif"),
        ],
    )
    write_tsv(
        root / "tf_gene_edges.tsv",
        ("source", "target", "sign", "score", "sources", "references"),
        [
            ("TF1", "TF2", 1, 5, "fixture", "r1"),
            ("TF2", "TF1", -1, 5, "fixture", "r2"),
            ("TF1", "G1", 1, 4, "fixture", "r3"),
            ("TF2", "G2", -1, 4, "fixture", "r4"),
        ],
    )
    write_tsv(
        root / "signaling_edges.tsv",
        ("source", "target", "source_type", "target_type", "sign", "score", "sources", "references"),
        [
            ("exercise", "S1", "cue", "signal", 1, 2, "fixture", "s1"),
            ("S1", "S2", "signal", "signal", 1, 2, "fixture", "s2"),
            ("S1", "TF1", "signal", "tf", 1, 3, "fixture", "s3"),
            ("S2", "TF2", "signal", "tf", -1, 3, "fixture", "s4"),
        ],
    )


def feature_rows(names, modality):
    return [
        {"feature_id": name, "feature_name": name, "modality": modality, "source_index": str(index)}
        for index, name in enumerate(names)
    ]


def make_bundle(root: Path, cohort_id: str, study_id: str, *, protein: bool, mismatched: bool = False) -> None:
    rng = np.random.default_rng(11 if study_id == "train_study" else 17)
    cells = 12
    barcodes = [f"cell{i:03d}" for i in range(cells)]
    atac_barcodes = list(reversed(barcodes)) if mismatched else barcodes
    blocks = {
        "rna": ModalityBlock(
            sparse.csr_matrix(rng.poisson(2.0, size=(cells, 4)).astype(np.float32)),
            barcodes,
            feature_rows(["TF1", "TF2", "G1", "G2"], "rna"),
        ),
        "atac": ModalityBlock(
            sparse.csr_matrix(rng.binomial(1, 0.35, size=(cells, 4)).astype(np.float32)),
            atac_barcodes,
            feature_rows(["chr1:0-2000", "chr1:2000-4000", "chr1:4000-6000", "chr1:6000-8000"], "atac"),
        ),
    }
    if protein:
        blocks["protein"] = ModalityBlock(
            sparse.csr_matrix(rng.poisson(3.0, size=(cells, 1)).astype(np.float32)),
            barcodes,
            feature_rows(["S1"], "protein"),
        )
    save_bundle(
        root,
        {
            "cohort_id": cohort_id,
            "study_id": study_id,
            "species": "Homo sapiens",
            "genome_build": "GRCh38",
            "adapter": "synthetic_harmonized",
            "donor_scope": "fixture",
        },
        blocks,
        {"fixture": {"sha256": "fixture"}},
    )


def main() -> None:
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)
        atlas, evidence, prior_root = root / "atlas", root / "evidence", root / "priors"
        make_atlas(atlas)
        make_evidence(evidence)
        report = compile_phase_b_priors(
            atlas, evidence, prior_root,
            max_genes=4, max_peaks=4, max_tfs=2, max_signals=2,
            min_tfs=2, min_localized_edges=4,
        )
        priors = load_phase_b_priors(prior_root)
        assert priors.num_genes == 4 and priors.num_peaks == 4 and priors.num_tfs == 2
        assert int(torch.count_nonzero(priors.circuit_tf_tf)) == 2
        assert int(torch.count_nonzero(priors.tf_peak_effect)) == 0
        verify_phase_b_priors(prior_root)
        print("PASS: motif x contact x signed-regulation prior intersection")

        feature_vocab = json.loads((prior_root / "feature_vocab.json").read_text())
        train_root, validation_root = root / "train", root / "validation"
        make_bundle(train_root, "train_cohort", "train_study", protein=True)
        make_bundle(validation_root, "validation_cohort", "validation_study", protein=False)
        train = load_snapshot_cohort(train_root, feature_vocab, max_cells=12, seed=1)
        validation = load_snapshot_cohort(validation_root, feature_vocab, max_cells=12, seed=2)
        assert train.protein is not None and validation.protein is None
        print("PASS: exact paired modalities loaded; absent protein remains missing")

        bad_root = root / "bad_unpaired"
        make_bundle(bad_root, "bad", "bad_study", protein=False, mismatched=True)
        try:
            load_snapshot_cohort(bad_root, feature_vocab, max_cells=12, seed=3)
        except ValueError as error:
            assert "fabricated" in str(error)
        else:
            raise AssertionError("Unpaired RNA/ATAC cells were silently matched")
        print("PASS: fabricated cell pairing rejected")

        model = WLDMultistudyFoundationModel(priors, context_dim=8)
        before = model.context_network[1].weight.detach().clone()
        trainer = SnapshotFoundationPretrainer(
            model,
            SnapshotPretrainingConfig(
                epochs=2, batch_size=6, batches_per_cohort=2,
                learning_rate=1e-3, max_cells_per_cohort=12, seed=7,
            ),
        )
        resume_state = root / "snapshot_training_state.pt"
        development = trainer.fit([train], [validation], state_path=resume_state)
        assert not torch.equal(before, model.context_network[1].weight.detach())
        assert development["sealed_test_evaluated"] is False
        resumed_model = WLDMultistudyFoundationModel(priors, context_dim=8)
        resumed = SnapshotFoundationPretrainer(resumed_model, trainer.config).fit(
            [train], [validation], state_path=resume_state
        )
        assert resumed["history"] == development["history"]
        contract = architecture_contract(model)
        assert contract["context_conditioned_production_and_decay"] is True
        print("PASS: validation-selected restart-safe snapshot representation pretraining")
        print("PASS: variable kinetic architecture retained without fitting kinetics from snapshots")

        validation_report = {
            "scope": "synthetic Phase B software contract only",
            "real_biological_pretraining_complete": False,
            "checks": [
                "evidence_intersection", "hard_topology", "exact_pairing",
                "missing_modality_mask", "snapshot_pretraining", "no_snapshot_kinetic_claim",
            ],
            "prior_dimensions": report["dimensions"],
            "sealed_test_evaluated": False,
            "attractor_claim": False,
        }
        Path(__file__).with_name("wld_phase_b_validation.json").write_text(
            json.dumps(validation_report, indent=2) + "\n"
        )
        print("PASS: wrote wld_phase_b_validation.json")


if __name__ == "__main__":
    main()
