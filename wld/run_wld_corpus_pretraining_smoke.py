"""Synthetic contract tests for WLD expanded-corpus pretraining."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from run_wld_phase_b_smoke import make_atlas, make_bundle, make_evidence
from wld_foundation_data import atomic_json, sha256_file
from wld_foundation_model_v4 import WLDMultistudyFoundationModel
from wld_phase_b_priors import compile_phase_b_priors, load_phase_b_priors
from wld_corpus_snapshot_pretraining import (
    CorpusPretrainingConfig,
    run_real_corpus_pretraining,
)


def rewrite_species(bundle_root: Path, species: str, genome_build: str, tissue: str) -> None:
    path = bundle_root / "bundle_manifest.json"
    value = json.loads(path.read_text())
    value["species"] = species
    value["genome_build"] = genome_build
    value["tissue"] = tissue
    atomic_json(path, value)


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)
        phase_a = root / "phase_a"
        phase_b = root / "phase_b"
        expansion = root / "expansion"
        output = root / "corpus_pretraining"
        atlas = phase_a / "training_atlas" / "homo_sapiens_grch38"
        evidence = root / "evidence"
        priors_root = phase_b / "priors" / "homo_sapiens_grch38"
        make_atlas(atlas)
        make_evidence(evidence)
        compile_phase_b_priors(
            atlas,
            evidence,
            priors_root,
            max_genes=4,
            max_peaks=4,
            max_tfs=2,
            max_signals=2,
            min_tfs=2,
            min_localized_edges=4,
        )

        phase_harmonized = phase_a / "harmonized" / "homo_sapiens_grch38"
        make_bundle(
            phase_harmonized / "phase_train",
            "phase_train",
            "phase_train_study",
            protein=True,
        )
        make_bundle(
            phase_harmonized / "phase_validation",
            "phase_validation",
            "phase_validation_study",
            protein=False,
        )
        atomic_json(
            phase_a / "phase_a_ingestion_report.json",
            {
                "schema_version": "fixture",
                "sealed_test_downloaded": False,
            },
        )
        phase_sources = root / "phase_a_sources.json"
        atomic_json(
            phase_sources,
            {
                "studies": {
                    "phase_train_study": {"split": "train"},
                    "phase_validation_study": {"split": "validation"},
                    "sealed_phase_study": {"split": "sealed_test"},
                }
            },
        )

        expansion_bundles = expansion / "bundles"
        make_bundle(
            expansion_bundles / "expansion_exact",
            "expansion_exact",
            "expansion_exact_study",
            protein=False,
        )
        make_bundle(
            expansion_bundles / "expansion_unpaired",
            "expansion_unpaired",
            "expansion_unpaired_study",
            protein=False,
            mismatched=True,
        )
        make_bundle(
            expansion_bundles / "mouse_staged",
            "mouse_staged",
            "mouse_study",
            protein=False,
            mismatched=True,
        )
        rewrite_species(
            expansion_bundles / "mouse_staged",
            "Mus musculus",
            "mm10",
            "brain",
        )
        atomic_json(
            expansion / "wld_corpus_expansion_report.json",
            {
                "schema_version": "fixture",
                "sealed_test_downloaded": False,
                "all_verified_expansion_bundles": [
                    "expansion_exact",
                    "expansion_unpaired",
                    "mouse_staged",
                ],
            },
        )
        expansion_sources = root / "expansion_sources.json"
        atomic_json(
            expansion_sources,
            {
                "sealed_exclusions": ["sealed_expansion_study"],
                "cohorts": [
                    {
                        "cohort_id": "expansion_exact",
                        "study_id": "expansion_exact_study",
                        "split": "train",
                        "tissue": "bone marrow",
                    },
                    {
                        "cohort_id": "expansion_unpaired",
                        "study_id": "expansion_unpaired_study",
                        "split": "train",
                        "tissue": "blood",
                    },
                    {
                        "cohort_id": "mouse_staged",
                        "study_id": "mouse_study",
                        "split": "train",
                        "tissue": "brain",
                    },
                ],
            },
        )

        priors = load_phase_b_priors(priors_root)
        phase_b_model = WLDMultistudyFoundationModel(priors, context_dim=32)
        phase_b_checkpoint = (
            phase_b / "snapshot_pretraining" / "wld_phase_b_snapshot_model.pt"
        )
        phase_b_checkpoint.parent.mkdir(parents=True)
        torch.save(phase_b_model.state_dict(), phase_b_checkpoint)
        atomic_json(
            phase_b / "snapshot_pretraining" / "wld_phase_b_pretraining.json",
            {
                "checkpoint_sha256": sha256_file(phase_b_checkpoint),
                "sealed_test_downloaded": False,
                "sealed_test_evaluated": False,
                "attractor_claim": False,
            },
        )

        config = CorpusPretrainingConfig(
            epochs=2,
            batch_size=6,
            batches_per_cohort=2,
            learning_rate=1e-3,
            max_cells_per_cohort=12,
            seed=23,
        )
        report = run_real_corpus_pretraining(
            phase_a,
            phase_b,
            expansion,
            phase_sources,
            expansion_sources,
            output,
            config,
            device="cpu",
        )
        development = report["development"]
        assert set(development["training_studies"]) == {
            "phase_train_study",
            "expansion_exact_study",
            "expansion_unpaired_study",
        }
        assert development["validation_studies"] == ["phase_validation_study"]
        assert development["training_pairing_modes"] == {
            "exact": 2,
            "unpaired_population": 1,
        }
        assert development["field_state_initial_sha256"] == development["field_state_final_sha256"]
        assert development["representation_state_initial_sha256"] != development["representation_state_final_sha256"]
        assert development["representation_updated"] is True
        assert development["snapshot_optimizer_excludes_field"] is True
        assert development["field_parameters_require_grad"] is True
        assert report["training_contract"]["expression_or_label_pairing_used"] is False
        assert report["sealed_test_evaluated"] is False
        assert report["attractor_claim"] is False
        assert [value["cohort_id"] for value in report["staged_nonhuman_cohorts"]] == [
            "mouse_staged"
        ]
        print("PASS: Phase A and expansion human cohorts share the training-derived atlas")
        print("PASS: exact cohorts use paired contrastive batches")
        print("PASS: unpaired cohorts use distributional batches without fabricated pairs")
        print("PASS: whole-study validation remains disjoint from training")
        print("PASS: mouse data remain staged without human circuit priors")
        print("PASS: context representation trains while snapshot kinetics remain unchanged")

        restored = run_real_corpus_pretraining(
            phase_a,
            phase_b,
            expansion,
            phase_sources,
            expansion_sources,
            output,
            config,
            device="cpu",
        )
        assert restored == report
        print("PASS: completed checkpoint is restart-safe and input-hash locked")

        validation_report = {
            "scope": "synthetic expanded-corpus pretraining contract only",
            "real_biological_pretraining_complete": False,
            "checks": [
                "phase_a_plus_expansion_projection",
                "exact_pair_contrastive_loss",
                "unpaired_distributional_loss",
                "whole_study_validation",
                "species_prior_isolation",
                "snapshot_kinetic_identifiability_guard",
                "restart_and_input_hash_lock",
            ],
            "sealed_test_evaluated": False,
            "attractor_claim": False,
        }
        Path(__file__).with_name("wld_corpus_pretraining_validation.json").write_text(
            json.dumps(validation_report, indent=2) + "\n"
        )
        print("PASS: wrote wld_corpus_pretraining_validation.json")


if __name__ == "__main__":
    main()
