"""Colab entry point for WLD v4 architecture/contract validation."""

from __future__ import annotations

import json
import platform
import runpy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "wld_circuit_dynamics_v3.py",
    "wld_foundation_model_v4.py",
    "wld_multistudy_pretraining.py",
    "run_wld_v4_foundation_smoke.py",
    "wld_v4_study_registry.json",
]


def main() -> None:
    print("WLD V4 MULTI-STUDY FOUNDATION CONTRACT")
    print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing WLD v4 files: {missing}")
    registry = json.loads((ROOT / "wld_v4_study_registry.json").read_text())
    print(f"Candidate studies: {len(registry['studies'])}")
    runpy.run_path(str(ROOT / "run_wld_v4_foundation_smoke.py"), run_name="__main__")
    print("\nCOMPLETE: WLD v4 software contract passed")
    print("Biological multi-study pretraining is not yet complete.")


if __name__ == "__main__":
    main()
