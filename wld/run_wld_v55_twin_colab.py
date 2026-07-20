"""Restart-safe real-data entry point for the WLD v5.5 chromatin twin prototype."""

from __future__ import annotations

import argparse
import json
import platform
import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import torch

from wld_chromatin_modules_v55 import (
    ComplexModuleConfig,
    compile_training_complex_modules,
    load_complex_module_atlas,
    load_v53_sparse_full_bundle,
    parse_corum_complexes,
    sha256_file,
)
from wld_chromatin_training_v54 import compile_regulator_tf_routes
from wld_chromatin_twin_training_v55 import TwinTrainingConfig, run_twin_development


def require(path: Path, label: str) -> None:
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def interaction_source(root: Path) -> Path:
    candidates = [
        root / "omnipath_core_human.tsv",
        *sorted(root.glob("omnipath_core_human.tsv.*")),
        *sorted(root.glob("omnipath_webservice_interactions*.tsv.xz")),
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size:
            return path
    raise FileNotFoundError(f"No frozen OmniPath interaction source under {root}")


def route_artifact_hashes(root: Path) -> dict:
    names = (
        "route_vocab.json",
        "regulator_tf_routes.npz",
        "regulator_tf_routes.tsv.gz",
    )
    return {name: sha256_file(root / name) for name in names}


def write_route_lock(root: Path, report: dict) -> dict:
    report = dict(report)
    report["artifact_sha256"] = route_artifact_hashes(root)
    destination = root / "route_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v53-bundle", type=Path, required=True)
    parser.add_argument("--prior-sources", type=Path, required=True)
    parser.add_argument("--corum-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--targets-per-epoch", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--shuffle-replicates", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--seeds", default="42,137,911")
    parser.add_argument("--device")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())

    print("WLD V5.5 MECHANISTIC CHROMATIN DIGITAL-MODEL DEVELOPMENT", flush=True)
    print(f"Python {platform.python_version()}", flush=True)
    print(f"NumPy {np.__version__} | SciPy {scipy.__version__}", flush=True)
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        "Digital-twin claim: False. Attractor claim: False. "
        "Sealed test targets and muscle J/L remain unopened.\n",
        flush=True,
    )

    prior_root = args.phase_b_root / "priors" / "homo_sapiens_grch38"
    checkpoint = args.corpus_root / "wld_corpus_pretrained_model.pt"
    for path, label in (
        (prior_root / "prior_manifest.json", "Phase B prior manifest"),
        (prior_root / "foundation_priors.npz", "foundation priors"),
        (prior_root / "feature_vocab.json", "foundation vocabulary"),
        (checkpoint, "expanded-corpus foundation checkpoint"),
        (args.corpus_root / "wld_corpus_pretraining_report.json", "expanded-corpus report"),
        (args.v53_bundle / "wld_v53_ingestion_manifest.json", "v5.3 manifest"),
        (args.v53_bundle / "whole_target_split.json", "whole-target split"),
        (args.v53_bundle / "atac_counts.GRCh38.2kb.npz", "full-bin sparse ATAC"),
        (args.v53_bundle / "cells.tsv.gz", "v5.3 cell metadata"),
        (args.v53_bundle / "bins.GRCh38.2kb.tsv.gz", "v5.3 response bins"),
        (args.corum_file, "frozen CORUM complex catalog"),
    ):
        require(path, label)

    print("1. Running dual-route architecture/statistics contract...", flush=True)
    runpy.run_path(
        str(Path(__file__).with_name("run_wld_v55_twin_smoke.py")),
        run_name="__main__",
    )

    split = json.loads((args.v53_bundle / "whole_target_split.json").read_text())["targets"]
    regulators = sorted(set(split["train"]) | set(split["validation"]) | set(split["test"]))
    feature_vocab = json.loads((prior_root / "feature_vocab.json").read_text())

    print("\n2. Compiling/restoring evidence-backed TF routes...", flush=True)
    route_root = args.output_root / "tf_routes"
    route_manifest = route_root / "route_manifest.json"
    source = interaction_source(args.prior_sources)
    if route_manifest.is_file():
        route_report = json.loads(route_manifest.read_text())
        if route_report.get("interaction_source") != str(source):
            raise RuntimeError("restored TF routes used a different frozen source")
        if route_report.get("interaction_source_sha256") != sha256_file(source):
            raise RuntimeError("restored TF routes used different interaction content")
        if route_report.get("artifact_sha256") != route_artifact_hashes(route_root):
            raise RuntimeError("restored TF-route artifacts failed their content lock")
        print("PASS: restored TF routes", flush=True)
    else:
        route_report = compile_regulator_tf_routes(
            source, regulators, feature_vocab["tfs"], route_root
        )
        route_report = write_route_lock(route_root, route_report)
    print(
        f"PASS: {route_report['route_edges']} regulator-to-TF evidence routes",
        flush=True,
    )

    print("\n3. Compiling/restoring training-only complex accessibility modules...", flush=True)
    module_root = args.output_root / "complex_modules"
    module_manifest = module_root / "complex_accessibility_module_manifest.json"
    if module_manifest.is_file():
        atlas = load_complex_module_atlas(module_root)
        manifest = json.loads(module_manifest.read_text())
        expected = asdict(
            ComplexModuleConfig(bootstrap_replicates=args.bootstrap_replicates)
        )
        if manifest.get("config") != expected:
            raise RuntimeError(
                "restored module atlas used different compiler thresholds; "
                "use a new output directory rather than mixing runs"
            )
        if manifest.get("curated_complexes", {}).get("source_sha256") != sha256_file(
            args.corum_file
        ):
            raise RuntimeError("restored complex modules used different CORUM content")
        print("PASS: restored hash-verified complex module atlas", flush=True)
    else:
        bundle = load_v53_sparse_full_bundle(
            args.v53_bundle,
            prior_root=prior_root,
            materialized_splits=("train",),
        )
        catalog = parse_corum_complexes(args.corum_file)
        atlas = compile_training_complex_modules(
            bundle,
            catalog,
            regulators,
            config=ComplexModuleConfig(
                bootstrap_replicates=args.bootstrap_replicates
            ),
            output_root=module_root,
        )
        manifest = json.loads(module_manifest.read_text())
        del bundle
    print(
        f"PASS: {len(atlas.complex_ids)} complexes, {len(atlas.module_vocab)} modules, "
        f"{atlas.module_peak_loading.nnz} stable module-bin effects",
        flush=True,
    )
    print(
        "   Validation values used for module construction: False\n"
        "   Test values materialized: False",
        flush=True,
    )

    print("\n4. Starting multi-seed whole-target development...", flush=True)
    print("   The run trains true dual routes and matched degree-shuffled controls.", flush=True)
    print("   Every condition is restart-safe; rerun this command after interruption.", flush=True)
    config = TwinTrainingConfig(
        epochs=args.epochs,
        targets_per_epoch=args.targets_per_epoch,
        batch_size=args.batch_size,
        patience=args.patience,
        seeds=seeds,
        shuffle_replicates=args.shuffle_replicates,
    )
    report = run_twin_development(
        prior_root,
        checkpoint,
        args.v53_bundle,
        route_root,
        module_root,
        args.output_root / "development",
        config,
        device=args.device,
    )

    intervals = report["target_level_intervals"]
    claims = report["claim_evaluation"]
    print("\n" + "=" * 78, flush=True)
    print("COMPLETE: WLD V5.5 WHOLE-TARGET DEVELOPMENT", flush=True)
    print("=" * 78, flush=True)
    for label, key in (
        ("Persistence minus true", "persistence_minus_true"),
        ("Topology shuffle minus true", "shuffle_minus_true"),
        ("Frozen removal minus true", "frozen_all_routes_removed_minus_true"),
    ):
        value = intervals[key]
        print(
            f"{label:32s} {value['effect']:+.6f} "
            f"(95% target CI {value['lower']:+.6f}, {value['upper']:+.6f})",
            flush=True,
        )
    print(f"Fitted path reliance:             {claims['fitted_path_reliance']}", flush=True)
    print(f"Topology specificity:             {claims['topology_specificity']}", flush=True)
    print(f"Useful perturbation prediction:   {claims['useful_perturbation_prediction']}", flush=True)
    print("Digital-twin claim:               False", flush=True)
    print("Attractor claim:                  False", flush=True)
    print("Test targets evaluated:           False", flush=True)
    print(
        "Report: "
        + str(args.output_root / "development" / "wld_v55_chromatin_twin_report.json"),
        flush=True,
    )


if __name__ == "__main__":
    main()
