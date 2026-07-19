"""Colab entrypoint for real WLD Phase B prior compilation and pretraining."""

from __future__ import annotations

import argparse
import json
import platform
import runpy
from pathlib import Path

import h5py
import numpy as np
import scipy
import torch

from wld_phase_b_priors import compile_phase_b_priors, verify_phase_b_priors
from wld_phase_b_snapshot_pretraining import (
    SnapshotPretrainingConfig,
    run_real_snapshot_pretraining,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches-per-cohort", type=int, default=8)
    parser.add_argument("--max-cells-per-cohort", type=int, default=4096)
    parser.add_argument("--max-genes", type=int, default=2048)
    parser.add_argument("--max-peaks", type=int, default=4096)
    parser.add_argument("--max-tfs", type=int, default=128)
    parser.add_argument("--max-signals", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    print("WLD PHASE B: BIOLOGICAL PRIORS + MULTI-STUDY SNAPSHOT PRETRAINING")
    print(f"Python {platform.python_version()}")
    print(f"NumPy {np.__version__} | SciPy {scipy.__version__} | h5py {h5py.__version__}")
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    runpy.run_path(
        str(Path(__file__).with_name("run_wld_phase_b_smoke.py")),
        run_name="__main__",
    )
    if args.smoke_only:
        print("COMPLETE: Phase B smoke-only contract")
        return

    phase_a_report_path = args.phase_a_root / "phase_a_ingestion_report.json"
    if not phase_a_report_path.is_file():
        raise FileNotFoundError(f"Missing Phase A report: {phase_a_report_path}")
    phase_a = json.loads(phase_a_report_path.read_text())
    protein_shape = (
        phase_a.get("harmonized_cohorts", {})
        .get("GSE158013_GSM5123951_TEA", {})
        .get("modalities", {})
        .get("protein", {})
        .get("shape")
    )
    if protein_shape != [7966, 47]:
        raise RuntimeError(
            f"Phase A GSE158013 protein repair is not verified: {protein_shape}"
        )
    if phase_a.get("sealed_test_downloaded") is not False:
        raise RuntimeError("Phase A indicates that a sealed study was opened")

    prior_root = args.output_root / "priors" / "homo_sapiens_grch38"
    if (prior_root / "prior_manifest.json").is_file():
        prior_report = verify_phase_b_priors(
            prior_root,
            args.phase_a_root / "training_atlas" / "homo_sapiens_grch38",
            args.evidence_root,
        )
        print("PASS: verified existing Phase B priors")
    else:
        print("\nCompiling motif x contact x signed-circuit foundation priors...", flush=True)
        prior_report = compile_phase_b_priors(
            args.phase_a_root / "training_atlas" / "homo_sapiens_grch38",
            args.evidence_root,
            prior_root,
            max_genes=args.max_genes,
            max_peaks=args.max_peaks,
            max_tfs=args.max_tfs,
            max_signals=args.max_signals,
        )
    print(json.dumps(prior_report["dimensions"], indent=2))

    print("\nStarting whole-study snapshot representation pretraining...", flush=True)
    report = run_real_snapshot_pretraining(
        args.phase_a_root,
        prior_root,
        args.sources,
        args.output_root / "snapshot_pretraining",
        SnapshotPretrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_cohort=args.batches_per_cohort,
            max_cells_per_cohort=args.max_cells_per_cohort,
            seed=args.seed,
        ),
    )
    print("\n" + "=" * 72)
    print("COMPLETE: WLD PHASE B DEVELOPMENT PRETRAINING")
    print("=" * 72)
    print(f"Best validation loss: {report['development']['best_validation_loss']:.6f}")
    print(f"Training cohorts: {report['development']['training_cohorts']}")
    print(f"Validation cohorts: {report['development']['validation_cohorts']}")
    print("ODE kinetics fitted from snapshots: False")
    print("Sealed test evaluated: False")
    print("Attractor claim: False")
    print(f"Report: {args.output_root / 'snapshot_pretraining' / 'wld_phase_b_pretraining.json'}")


if __name__ == "__main__":
    main()
