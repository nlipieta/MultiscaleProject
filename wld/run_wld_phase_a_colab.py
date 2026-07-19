"""Colab wrapper for dependency and Phase A ingestion validation."""

from __future__ import annotations

import argparse
import platform
import runpy
import subprocess
import sys
from pathlib import Path
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("wld_phase_a_sources.json"))
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--muscle-export", type=Path)
    args = parser.parse_args()
    print("WLD V4 PHASE A REAL-DATA INGESTION")
    print(f"Python: {platform.python_version()}")
    for module in ("numpy", "scipy", "h5py"):
        imported = __import__(module)
        print(f"{module}: {imported.__version__}")
    runpy.run_path(str(Path(__file__).with_name("run_wld_phase_a_data_smoke.py")), run_name="__main__")
    if args.smoke_only:
        print("COMPLETE: smoke-only data contract")
        return
    existing_report = args.root / "phase_a_ingestion_report.json"
    if existing_report.is_file():
        existing = json.loads(existing_report.read_text())
        protein = (
            existing.get("harmonized_cohorts", {})
            .get("GSE158013_GSM5123951_TEA", {})
            .get("modalities", {})
            .get("protein", {})
        )
        shape = protein.get("shape", [])
        if len(shape) == 2 and (shape[0] < 1000 or shape[1] >= 5000):
            print(f"Detected legacy transposed ADT shape {shape}; applying repair first.", flush=True)
            repair = subprocess.run(
                [
                    sys.executable, "-u",
                    str(Path(__file__).with_name("repair_wld_phase_a_adt.py")),
                    "--root", str(args.root),
                    "--sources", str(args.sources),
                ]
            )
            if repair.returncode:
                raise RuntimeError(f"ADT repair failed with exit code {repair.returncode}")
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("run_wld_phase_a_ingestion.py")),
        "--root", str(args.root), "--sources", str(args.sources),
    ]
    if args.include:
        command.extend(["--include", *args.include])
    if args.muscle_export:
        command.extend(["--muscle-export", str(args.muscle_export)])
    result = subprocess.run(command)
    if result.returncode:
        raise RuntimeError(f"Phase A ingestion failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
