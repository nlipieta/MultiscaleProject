"""Build and verify the restart-safe WLD v6.1 axolotl measurement corpus.

This runner materializes only development sources declared in the sibling
measurement registry.  GSE315993 remains a sealed external spatial study, and
this stage neither trains a model nor makes digital-twin or attractor claims.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Mapping

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "WLD_V61_OUTPUT_ROOT",
        "/content/drive/MyDrive/WLD_Backup/wld_v61_axolotl_corpus",
    )
)
REGISTRY_NAME = "wld_v61_axolotl_measurement_sources.json"
RUN_REPORT_NAME = "wld_v61_axolotl_corpus_run.json"
SEALED_ACCESSION = "GSE315993"

REQUIRED_FALSE_CLAIMS = (
    "gse315993_measurement_values_materialized",
    "model_trained",
    "digital_twin_claim",
    "attractor_claim",
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} did not return a structured mapping")
    return value


def _validate_claim_boundary(verification: Mapping[str, object]) -> Mapping[str, object]:
    claims = _mapping(verification.get("claims"), "verify_corpus claims")
    violated = [name for name in REQUIRED_FALSE_CLAIMS if claims.get(name) is not False]
    if violated:
        raise RuntimeError(
            "WLD v6.1 crossed its development-only claim boundary: "
            + ", ".join(violated)
        )

    # These additional switches are enforced whenever the verifier reports them.
    # Their absence is not substituted for the four mandatory claims above.
    for name in (
        "sealed_study_evaluated",
        "sealed_values_fetched",
        "test_measurement_values_materialized",
        "biological_prediction_claim",
        "model_checkpoint_written",
    ):
        if name in claims and claims[name] is not False:
            raise RuntimeError(f"WLD v6.1 unexpectedly asserted {name}")
    return claims


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable ({type(exc).__name__})"
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Durable corpus root (default: WLD_V61_OUTPUT_ROOT or Google Drive)",
    )
    parser.add_argument(
        "--no-timecourse",
        action="store_true",
        help="Skip the optional GSE121737 time-course matrix tier",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Skip the unsealed GSE243225 spatial-reference tier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and plan acquisition without materializing assay values",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing corpus without rebuilding it",
    )
    args = parser.parse_args()
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")

    # Import only after the pinned launcher has installed and selected its
    # isolated numerical environment.
    from wld_axolotl_corpus_v61 import atomic_json, build_from_registry, verify_corpus

    sibling = Path(__file__).resolve().parent
    registry_path = sibling / REGISTRY_NAME
    if not registry_path.is_file() or registry_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty WLD v6.1 registry: {registry_path}")

    output_root = args.output_root.expanduser().resolve()
    if str(output_root).startswith("/content/drive/"):
        if not Path("/content/drive/MyDrive").is_dir() or not os.path.ismount("/content/drive"):
            raise RuntimeError(
                "Google Drive is not mounted; refusing to create an ephemeral /content/drive lookalike"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(output_root).free
    try:
        ram_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        ram_bytes = 0
    print(f"Durable free space: {free_bytes / 1024**3:.1f} GB", flush=True)
    if ram_bytes:
        print(f"Physical RAM: {ram_bytes / 1024**3:.1f} GB", flush=True)
    if not args.dry_run and not args.verify_only:
        if free_bytes < 3 * 1024**3:
            raise RuntimeError("WLD v6.1 requires at least 3 GB free durable space")
        if ram_bytes and ram_bytes < 8 * 1024**3:
            raise RuntimeError("WLD v6.1 requires at least 8 GB physical RAM")

    print("WLD V6.1 RESTART-SAFE AXOLOTL MEASUREMENT CORPUS", flush=True)
    print(f"Python: {platform.python_version()} ({sys.executable})", flush=True)
    print(f"Platform: {platform.platform()}", flush=True)
    print(
        "Packages: "
        f"NumPy {_package_version('numpy')} | "
        f"SciPy {_package_version('scipy')}",
        flush=True,
    )
    print(f"Registry: {registry_path}", flush=True)
    print(f"Durable output: {output_root}", flush=True)
    print(f"Include time course: {not args.no_timecourse}", flush=True)
    print(f"Include spatial reference: {not args.no_spatial}", flush=True)
    print(f"Dry run: {args.dry_run}", flush=True)
    print(f"Verify only: {args.verify_only}", flush=True)
    print(
        f"{SEALED_ACCESSION} remains sealed; no model training, digital-twin claim, "
        "or attractor claim is permitted.\n",
        flush=True,
    )

    build_report: Mapping[str, object] | None = None
    if args.verify_only:
        print("1. Build skipped (--verify-only).", flush=True)
    else:
        print("1. Building/resuming the registered development corpus...", flush=True)
        build_report = _mapping(
            build_from_registry(
                registry_path=registry_path,
                output_root=output_root,
                include_timecourse=not args.no_timecourse,
                include_spatial=not args.no_spatial,
                dry_run=args.dry_run,
            ),
            "build_from_registry",
        )
        print("PASS: development-corpus build/resume stage", flush=True)

    print("\n2. Verifying corpus integrity, splits, provenance, and seals...", flush=True)
    if args.dry_run:
        verification = {
            "verified": True,
            "dry_run": True,
            "claims": dict(_mapping(build_report, "dry-run build report")["claims"]),
        }
    else:
        verification = _mapping(verify_corpus(output_root), "verify_corpus")
    claims = _validate_claim_boundary(verification)
    print("PASS: corpus verification and sealed-study boundary", flush=True)

    run_report = {
        "schema_version": "wld-v6.1-axolotl-corpus-run-v1",
        "registry": str(registry_path),
        "output_root": str(output_root),
        "options": {
            "include_timecourse": not args.no_timecourse,
            "include_spatial": not args.no_spatial,
            "dry_run": bool(args.dry_run),
            "verify_only": bool(args.verify_only),
        },
        "build": dict(build_report) if build_report is not None else None,
        "verification": dict(verification),
        "claims": dict(claims),
    }
    run_report_path = output_root / (
        "wld_v61_axolotl_corpus_dry_run.json" if args.dry_run else RUN_REPORT_NAME
    )
    atomic_json(run_report_path, run_report)

    print("\n" + "=" * 78, flush=True)
    print("VERIFIED COMPLETE: WLD V6.1 AXOLOTL MEASUREMENT CORPUS", flush=True)
    print("=" * 78, flush=True)
    print(f"Development build executed:       {not args.verify_only}", flush=True)
    print(f"Dry run:                         {bool(args.dry_run)}", flush=True)
    print(
        "Development assay values allowed: "
        f"{not args.dry_run and not args.verify_only}",
        flush=True,
    )
    print(f"{SEALED_ACCESSION} values materialized:  False", flush=True)
    print("Biological model trained:         False", flush=True)
    print("Digital-twin claim:               False", flush=True)
    print("Attractor claim:                  False", flush=True)
    print(f"Run report: {run_report_path}", flush=True)


if __name__ == "__main__":
    main()
