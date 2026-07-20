"""Restart-safe WLD v5.4 CRISPR-sciATAC development entry point."""

from __future__ import annotations

import argparse
import json
import platform
import runpy
from pathlib import Path

import numpy as np
import scipy
import torch

from wld_chromatin_training_v54 import (
    ChromatinTrainingConfig,
    compile_regulator_tf_routes,
    run_chromatin_response_development,
)


def require(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def interaction_source(root: Path) -> Path:
    preferred = [
        root / "omnipath_core_human.tsv",
        *sorted(root.glob("omnipath_core_human.tsv.*")),
        *sorted(root.glob("omnipath_webservice_interactions*.tsv.xz")),
    ]
    for path in preferred:
        if path.is_file() and path.stat().st_size:
            return path
    raise FileNotFoundError(
        "No frozen OmniPath core interaction table was found under " + str(root)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v53-bundle", type=Path, required=True)
    parser.add_argument("--prior-sources", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--targets-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--shuffle-replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()

    print("WLD V5.4.1 RESPONSE-CALIBRATED CHROMATIN DEVELOPMENT", flush=True)
    print(f"Python {platform.python_version()}", flush=True)
    print(f"NumPy {np.__version__} | SciPy {scipy.__version__}", flush=True)
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("J/L, external studies and the 16 v5.3 test targets remain sealed.\n", flush=True)

    prior_root = args.phase_b_root / "priors" / "homo_sapiens_grch38"
    checkpoint = args.corpus_root / "wld_corpus_pretrained_model.pt"
    require(prior_root / "prior_manifest.json", "Phase B prior manifest")
    require(prior_root / "foundation_priors.npz", "Phase B foundation priors")
    require(prior_root / "feature_vocab.json", "Phase B feature vocabulary")
    require(checkpoint, "expanded-corpus foundation checkpoint")
    require(args.corpus_root / "wld_corpus_pretraining_report.json", "corpus report")
    require(args.v53_bundle / "wld_v53_ingestion_manifest.json", "v5.3 manifest")
    require(args.v53_bundle / "whole_target_split.json", "v5.3 whole-target split")
    require(args.v53_bundle / "atac_counts.GRCh38.2kb.npz", "v5.3 ATAC matrix")
    require(args.v53_bundle / "cells.tsv.gz", "v5.3 cell metadata")

    print("1. Running the numerical/mechanistic architecture contract...", flush=True)
    runpy.run_path(
        str(Path(__file__).with_name("run_wld_v54_chromatin_smoke.py")),
        run_name="__main__",
    )
    runpy.run_path(
        str(Path(__file__).with_name("run_wld_v54_training_smoke.py")),
        run_name="__main__",
    )

    print("\n2. Compiling regulator-to-TF routes from frozen interactions...", flush=True)
    route_root = args.output_root / "routes"
    split = json.loads((args.v53_bundle / "whole_target_split.json").read_text())["targets"]
    regulators = sorted(
        set(split["train"]) | set(split["validation"]) | set(split["test"])
    )
    feature_vocab = json.loads((prior_root / "feature_vocab.json").read_text())
    source = interaction_source(args.prior_sources)
    route_manifest = route_root / "route_manifest.json"
    if route_manifest.is_file():
        route_report = json.loads(route_manifest.read_text())
        if route_report.get("interaction_source") != str(source):
            raise RuntimeError("Completed route scaffold used a different interaction source")
        print("PASS: restored compiled regulator-to-TF routes", flush=True)
    else:
        route_report = compile_regulator_tf_routes(
            source,
            regulators,
            feature_vocab["tfs"],
            route_root,
        )
    covered = set(route_report["covered_regulators"])
    coverage = {
        name: len(covered.intersection(values)) for name, values in split.items()
    }
    print(
        f"PASS: {route_report['route_edges']} supported routes; "
        f"target coverage {coverage}",
        flush=True,
    )
    if coverage["train"] < 5 or coverage["validation"] < 3:
        raise RuntimeError(
            "Frozen interaction topology does not cover enough whole targets for a "
            f"fair route-specific experiment: {coverage}. Do not add dense guide edges."
        )

    print("\n3. Training true routes and degree-preserving controls...", flush=True)
    print("   Control cells are sampled independently; no cell pairs are fabricated.", flush=True)
    print("   Validation consists of entire unseen perturbation targets.", flush=True)
    print("   Rerunning resumes each condition from its last completed epoch.\n", flush=True)
    report = run_chromatin_response_development(
        prior_root,
        checkpoint,
        args.v53_bundle,
        route_root,
        args.output_root / "development",
        ChromatinTrainingConfig(
            epochs=args.epochs,
            targets_per_epoch=args.targets_per_epoch,
            batch_size=args.batch_size,
            patience=args.patience,
            shuffle_replicates=args.shuffle_replicates,
            seed=args.seed,
        ),
        device=args.device,
    )

    metrics = report["specificity"]
    print("\n" + "=" * 78, flush=True)
    print("VERIFIED COMPLETE: WLD V5.4.1 CHROMATIN RESPONSE DEVELOPMENT", flush=True)
    print("=" * 78, flush=True)
    print(f"True-route validation SWD:              {metrics['true_model_swd']:.6f}", flush=True)
    print(f"Persistence validation SWD:             {metrics['persistence_swd']:.6f}", flush=True)
    print(f"True gain over persistence:              {metrics['true_gain_over_persistence']:+.6f}", flush=True)
    print(f"True response NRMSE:                     {metrics['true_response_nrmse']:.6f}", flush=True)
    print(f"True response cosine:                    {metrics['true_response_cosine']:.6f}", flush=True)
    print(
        "True advantage over retrained shuffles: "
        f"{metrics['true_advantage_over_retrained_shuffles']:+.6f}",
        flush=True,
    )
    print(f"Frozen-zero route effect:                {metrics['frozen_zero_effect']:+.6f}", flush=True)
    print("\nTarget/guide identity in encoder: False", flush=True)
    print("Test targets evaluated:             False", flush=True)
    print("Muscle J/L evaluated:               False", flush=True)
    print("ODE kinetics identified:            False", flush=True)
    print("Attractor claim:                    False", flush=True)
    print(
        "Report: " + str(args.output_root / "development" / "wld_v54_chromatin_response_report.json"),
        flush=True,
    )


if __name__ == "__main__":
    main()
