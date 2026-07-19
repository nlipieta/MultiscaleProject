"""Colab entry point for restart-safe WLD v4 temporal fine-tuning."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

from wld_v4_temporal_finetuning import (
    TemporalFinetuningConfig,
    run_real_temporal_finetuning,
)


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-transition", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()

    print("WLD V4 PRETRAINED TEMPORAL FINE-TUNING", flush=True)
    print(f"Python {platform.python_version()}", flush=True)
    print(f"NumPy {np.__version__} | SciPy {scipy.__version__}", flush=True)
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    require(args.phase_a_root / "phase_a_ingestion_report.json", "Phase A report")
    require(
        args.phase_b_root / "priors" / "homo_sapiens_grch38" / "prior_manifest.json",
        "Phase B human prior manifest",
    )
    require(args.corpus_root / "wld_corpus_pretrained_model.pt", "corpus checkpoint")
    require(args.corpus_root / "wld_corpus_pretraining_report.json", "corpus report")
    require(args.export_root / "metadata.tsv", "GSE240061 metadata")
    require(args.export_root / "split.json", "GSE240061 frozen subject split")
    print("PASS: durable upstream artifacts found", flush=True)

    smoke = Path(__file__).with_name("run_wld_v4_temporal_smoke.py")
    completed = subprocess.run([sys.executable, "-u", str(smoke)], check=False)
    if completed.returncode:
        raise RuntimeError(f"Temporal software contract failed with exit code {completed.returncode}")
    print("PASS: temporal numerical and leakage contract", flush=True)

    report = run_real_temporal_finetuning(
        args.phase_a_root,
        args.phase_b_root,
        args.corpus_root,
        args.export_root,
        args.output,
        TemporalFinetuningConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_transition=args.batches_per_transition,
            integration_steps=args.steps,
            patience=args.patience,
            seed=args.seed,
        ),
        device=args.device,
    )
    print("\n" + "=" * 76, flush=True)
    print("VERIFIED COMPLETE: WLD V4 TEMPORAL FINE-TUNING", flush=True)
    print("=" * 76, flush=True)
    for condition, record in report["conditions"].items():
        metrics = record["validation"]["aggregate"]
        print(
            f"{condition:24s} | best epoch {record['best_epoch']:3d} | "
            f"RNA SWD {metrics['rna_swd']:.6f} | "
            f"RNA mean Pearson {metrics['rna_log_mean_pearson']:.4f}",
            flush=True,
        )
    print("\nCircuit reliance:", json.dumps(report["circuit_reliance"], indent=2), flush=True)
    print("\nJ/L temporal evaluation: False", flush=True)
    print("External sealed test evaluation: False", flush=True)
    print("Attractor claim: False", flush=True)
    print(f"Durable output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
