"""Colab entrypoint for restart-safe WLD training-corpus expansion."""

from __future__ import annotations

import argparse
import platform
import runpy
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).with_name("wld_corpus_expansion_sources.json"),
    )
    parser.add_argument("--tiers", nargs="+", default=["core", "extended"])
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    print("WLD RESTART-SAFE TRAINING-CORPUS EXPANSION")
    print(f"Python: {platform.python_version()}")
    for module in ("numpy", "scipy", "h5py"):
        imported = __import__(module)
        print(f"{module}: {imported.__version__}")

    runpy.run_path(
        str(Path(__file__).with_name("run_wld_corpus_expansion_smoke.py")),
        run_name="__main__",
    )
    print("PASS: corpus-expansion numerical and leakage contract", flush=True)
    if args.smoke_only:
        print("COMPLETE: smoke-only corpus-expansion contract")
        return

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("run_wld_corpus_expansion.py")),
        "--root",
        str(args.root),
        "--sources",
        str(args.sources),
    ]
    if args.include:
        command.extend(["--include", *args.include])
    else:
        command.extend(["--tiers", *args.tiers])
    result = subprocess.run(command)
    if result.returncode:
        raise RuntimeError(
            f"Corpus expansion failed with exit code {result.returncode}; rerun this same cell to resume"
        )


if __name__ == "__main__":
    main()
