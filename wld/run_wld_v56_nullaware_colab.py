"""Restart-safe real-data entry point for WLD v5.6 null-aware development.

This stage reuses the already inspected v5.5 validation targets for architecture
development.  It never materializes the v5.3 test partition and therefore
produces descriptive comparisons only, not confirmatory confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import platform
import runpy
from pathlib import Path

import numpy as np
import scipy
import torch

from wld_chromatin_twin_training_v55 import load_twin_priors
from wld_chromatin_twin_training_v56 import (
    NullAwareTrainingConfig,
    run_twin_development,
)
from wld_chromatin_modules_v55 import sha256_file
from wld_v56_topology_controls import build_matched_control_priors


def require(path: Path, label: str) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def _split_strata(regulators: tuple[str, ...], split_path: Path) -> tuple[str, ...]:
    contract = json.loads(Path(split_path).read_text())
    rosters = contract.get("targets")
    if not isinstance(rosters, dict) or set(rosters) != {"train", "validation", "test"}:
        raise RuntimeError("whole-target split no longer has train/validation/test rosters")
    assignment = {}
    for split, targets in rosters.items():
        for target in targets:
            normalized = str(target).upper()
            if normalized in assignment:
                raise RuntimeError(f"Target occurs in multiple split rosters: {normalized}")
            assignment[normalized] = split
    if set(regulators) != set(assignment):
        raise RuntimeError("route vocabulary and whole-target split disagree")
    return tuple(assignment[regulator] for regulator in regulators)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v53-bundle", type=Path, required=True)
    parser.add_argument("--v55-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--targets-per-epoch", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--control-replicates", type=int, default=10)
    parser.add_argument("--seeds", default="42,137,911")
    parser.add_argument("--device")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())

    print("WLD V5.6 NULL-AWARE CHROMATIN DEVELOPMENT", flush=True)
    print(f"Python {platform.python_version()}", flush=True)
    print(f"NumPy {np.__version__} | SciPy {scipy.__version__}", flush=True)
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        "Previously inspected validation targets are reused for development. "
        "The sealed test remains unopened.\nDigital-twin claim: False. "
        "Attractor claim: False.\n",
        flush=True,
    )

    prior_root = args.phase_b_root / "priors" / "homo_sapiens_grch38"
    foundation_checkpoint = args.corpus_root / "wld_corpus_pretrained_model.pt"
    route_root = args.v55_root / "tf_routes"
    module_root = args.v55_root / "complex_modules"
    v55_report_path = (
        args.v55_root / "development" / "wld_v55_chromatin_twin_report.json"
    )
    split_path = args.v53_bundle / "whole_target_split.json"
    for path, label in (
        (prior_root / "prior_manifest.json", "Phase B prior manifest"),
        (prior_root / "foundation_priors.npz", "Phase B numerical priors"),
        (prior_root / "feature_vocab.json", "Phase B vocabulary"),
        (foundation_checkpoint, "expanded-corpus foundation checkpoint"),
        (
            args.corpus_root / "wld_corpus_pretraining_report.json",
            "expanded-corpus report",
        ),
        (args.v53_bundle / "wld_v53_ingestion_manifest.json", "v5.3 manifest"),
        (split_path, "v5.3 whole-target split"),
        (args.v53_bundle / "atac_counts.GRCh38.2kb.npz", "v5.3 sparse ATAC"),
        (args.v53_bundle / "cells.tsv.gz", "v5.3 cell metadata"),
        (args.v53_bundle / "bins.GRCh38.2kb.tsv.gz", "v5.3 response bins"),
        (route_root / "route_manifest.json", "completed v5.5 TF routes"),
        (
            module_root / "complex_accessibility_module_manifest.json",
            "completed v5.5 complex modules",
        ),
        (v55_report_path, "completed v5.5 result"),
    ):
        require(path, label)

    v55_report = json.loads(v55_report_path.read_text())
    if (
        v55_report.get("claims", {}).get("test_targets_evaluated") is not False
        or v55_report.get("claim_evaluation", {}).get("digital_twin_claim") is not False
        or v55_report.get("claim_evaluation", {}).get("attractor_claim") is not False
    ):
        raise RuntimeError("The v5.5 lineage crossed a sealed scientific boundary")
    if v55_report.get("claim_evaluation", {}).get("useful_perturbation_prediction") is not False:
        raise RuntimeError(
            "This null-aware repair is locked to the observed v5.5 predictive failure"
        )
    print(
        "PASS: v5.5 failure lineage locked "
        f"({sha256_file(v55_report_path)[:12]}…)",
        flush=True,
    )

    config = NullAwareTrainingConfig(
        epochs=args.epochs,
        targets_per_epoch=args.targets_per_epoch,
        batch_size=args.batch_size,
        patience=args.patience,
        seeds=seeds,
        control_replicates=args.control_replicates,
        minimum_control_replicates=args.control_replicates,
    )
    config.validate()

    print("\n1. Running the null-aware architecture/training contract...", flush=True)
    runpy.run_path(
        str(Path(__file__).with_name("run_wld_v56_nullaware_smoke.py")),
        run_name="__main__",
    )

    print("\n2. Building split-stratified matched topology controls...", flush=True)
    true_priors, _atlas, _prior_audit = load_twin_priors(
        prior_root, route_root, module_root
    )
    route_vocab = json.loads((route_root / "route_vocab.json").read_text())
    regulators = tuple(str(value).upper() for value in route_vocab["regulators"])
    strata = _split_strata(regulators, split_path)
    controls, control_audit = build_matched_control_priors(
        true_priors,
        config.control_replicates,
        config.control_seed,
        strata=strata,
    )
    if control_audit.get("replicates") != args.control_replicates:
        raise RuntimeError("Matched-control builder returned the wrong replicate count")
    print(
        f"PASS: {len(controls)} controls preserve route-profile distributions "
        "inside train/validation/test strata",
        flush=True,
    )

    print("\n3. Starting restart-safe v5.6 development...", flush=True)
    print(
        "   Conditions: true null-aware routes plus "
        f"{len(controls)} matched controls across {len(seeds)} seeds.",
        flush=True,
    )
    print(
        "   Rerun the identical launcher after interruption; completed fits are retained.",
        flush=True,
    )
    development_root = args.output_root / "development"
    report = run_twin_development(
        prior_root,
        foundation_checkpoint,
        args.v53_bundle,
        route_root,
        module_root,
        development_root,
        config,
        control_priors=controls,
        control_generation_audit=control_audit,
        device=args.device,
    )

    if (
        report.get("claims", {}).get("test_targets_evaluated") is not False
        or report.get("claims", {}).get("untouched_audit_inference") is not False
        or report.get("claims", {}).get("digital_twin_claim") is not False
        or report.get("claims", {}).get("attractor_claim") is not False
    ):
        raise RuntimeError("v5.6 report crossed its development-only claim boundary")

    effects = report["descriptive_effects"]
    print("\n" + "=" * 78, flush=True)
    print("COMPLETE: WLD V5.6 NULL-AWARE DEVELOPMENT", flush=True)
    print("=" * 78, flush=True)
    for label, key in (
        ("Persistence minus true", "persistence_minus_true"),
        ("Training perturbed mean minus true", "perturbed_mean_minus_true"),
        ("Matched-control mean minus true", "control_mean_minus_true"),
        ("Frozen TF minus true", "frozen_tf_minus_true"),
        ("Frozen complex minus true", "frozen_complex_minus_true"),
    ):
        value = effects[key]
        print(
            f"{label:36s} {value['mean']:+.8f} "
            f"(positive targets {value['positive_target_fraction']:.1%})",
            flush=True,
        )
    checks = report["development_checks"]
    print(
        f"Mean response NRMSE:                {checks['mean_response_nrmse']:.6f}",
        flush=True,
    )
    print(
        f"Mean response cosine:               {checks['mean_response_cosine']:+.6f}",
        flush=True,
    )
    print(
        "Eligible to freeze new confirmation plan: "
        + str(checks["eligible_to_freeze_new_confirmation_plan"]),
        flush=True,
    )
    print("Inference from reused validation: False", flush=True)
    print("Test targets evaluated:          False", flush=True)
    print("Digital-twin claim:              False", flush=True)
    print("Attractor claim:                 False", flush=True)
    print(
        "Report: "
        + str(development_root / "wld_v56_null_aware_development_report.json"),
        flush=True,
    )


if __name__ == "__main__":
    main()
