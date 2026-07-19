"""Colab entrypoint for restart-safe WLD expanded-corpus pretraining."""

from __future__ import annotations

import argparse
import json
import platform
import runpy
from pathlib import Path

import numpy as np
import scipy
import torch

from wld_corpus_snapshot_pretraining import (
    CorpusPretrainingConfig,
    run_real_corpus_pretraining,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--expansion-root", type=Path, required=True)
    parser.add_argument("--phase-a-sources", type=Path, required=True)
    parser.add_argument("--expansion-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches-per-cohort", type=int, default=8)
    parser.add_argument("--max-cells-per-cohort", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    print("WLD PAIRING-AWARE EXPANDED-CORPUS PRETRAINING")
    print(f"Python {platform.python_version()}")
    print(f"NumPy {np.__version__} | SciPy {scipy.__version__}")
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    runpy.run_path(
        str(Path(__file__).with_name("run_wld_corpus_pretraining_smoke.py")),
        run_name="__main__",
    )
    print("PASS: expanded-corpus numerical and leakage contract", flush=True)
    if args.smoke_only:
        print("COMPLETE: smoke-only expanded-corpus contract")
        return

    report = run_real_corpus_pretraining(
        args.phase_a_root,
        args.phase_b_root,
        args.expansion_root,
        args.phase_a_sources,
        args.expansion_sources,
        args.output,
        CorpusPretrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_cohort=args.batches_per_cohort,
            max_cells_per_cohort=args.max_cells_per_cohort,
            seed=args.seed,
        ),
        device=args.device,
    )
    print("\n" + "=" * 76)
    print("VERIFIED COMPLETE: WLD EXPANDED-CORPUS PRETRAINING")
    print("=" * 76)
    print(json.dumps(report["development"], indent=2))
    print("ODE kinetics fitted from snapshots: False")
    print("Sealed test evaluated: False")
    print("Attractor claim: False")
    print(f"Durable output: {args.output}")


if __name__ == "__main__":
    main()
