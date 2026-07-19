"""Build and train the real-data WLD temporal benchmark in Google Colab.

This runner begins from the already exported GSE240061 matrices and the
already compiled GSE240061/GSE126100 biological scaffold.  It pins the model,
cohort builder, and temporal trainer to one reviewed repository commit, checks
the exact successful prior manifest, constructs a leakage-safe unpaired
pre-to-post cohort, and compares the hard circuit against a no-circuit control.

The 3.5-hour endpoint is a transient response.  This run tests held-out
temporal prediction; it does not label that endpoint as an attractor.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


ROOT = Path("/content/wld_real_data")
EXPORT = ROOT / "gse240061_export"
PRIORS = ROOT / "gse240061_priors"
COHORT = ROOT / "gse240061_temporal_cohort"
RESULTS = ROOT / "gse240061_temporal_results_seed42"
CODE = ROOT / "temporal_code"

DEPENDENCY_COMMIT = "d62ce237192d01cfdd3e8042c58d692a734f707f"
EXPECTED_COMPILER_COMMIT = "16a2656857e0e5003d9ea31b382b65cf03efec31"
REPOSITORY = "nlipieta/MultiscaleProject"
CODE_HASHES = {
    "build_wld_muscle_exercise_dataset.py": (
        "197a6b4d4776f3778c9687b4c3384e053c83da63bcccd0933df95750bb3f1b6a"
    ),
    "wld_circuit_dynamics_v3.py": (
        "2ffcd9d0a60551dd06db2646c60747ba0680e47150fd5f91bf42b7d8eadfe068"
    ),
    "wld_temporal_training.py": (
        "6ea338e0a5989a935f6e781e14e471999b1a526993b54eb2e14022e38991e277"
    ),
}
EXPECTED_SPLIT = {
    "train": {"E", "G", "N"},
    "validation": {"I"},
    "test": {"J", "L"},
}
EXPECTED_PRIOR_COUNTS = {
    "peak_gene_links": 28449,
    "motif_hits": 12702,
    "tf_gene_edges": 41280,
    "signaling_edges": 2552,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            f"Missing or empty {label} files:\n  " + "\n  ".join(missing)
        )


def assert_split(value: Mapping[str, Sequence[object]], label: str) -> None:
    if set(value) != set(EXPECTED_SPLIT):
        raise RuntimeError(f"{label} has unexpected split names: {sorted(value)}")
    actual = {name: {str(item) for item in value[name]} for name in EXPECTED_SPLIT}
    if actual != EXPECTED_SPLIT:
        raise RuntimeError(
            f"{label} split differs from the frozen subject contract: {actual}"
        )


def download_pinned(name: str) -> Path:
    destination = CODE / name
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/"
        f"{DEPENDENCY_COMMIT}/wld/{name}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "WLD-Colab/1.0"})
    error = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            destination.write_bytes(payload)
            error = None
            break
        except Exception as exc:  # Preserve the final network exception for diagnosis.
            error = exc
            print(f"   download attempt {attempt}/5 failed for {name}: {exc}", flush=True)
    if error is not None:
        raise RuntimeError(f"Could not download pinned {name}: {error}")
    observed = sha256(destination)
    expected = CODE_HASHES[name]
    if observed != expected:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {name}: expected {expected}, observed {observed}"
        )
    print(f"   verified: {name} ({observed[:12]}...)", flush=True)
    return destination


def run_logged(command: Sequence[object], log_path: Path) -> None:
    rendered = [str(item) for item in command]
    print("\nRunning:", " ".join(rendered), flush=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "42",
            "MPLBACKEND": "Agg",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            rendered,
            cwd=CODE,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        partial = RESULTS / "wld_temporal_results.partial.json"
        partial_note = f" Partial results: {partial}" if partial.exists() else ""
        raise RuntimeError(
            f"Command failed with exit code {return_code}. Full log: {log_path}."
            f"{partial_note}"
        )


def validate_prior_manifest() -> dict:
    path = PRIORS / "prior_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert_split(manifest["group_split"], "Prior manifest")

    compiler = manifest.get("compiler", {})
    if compiler.get("git_commit") != EXPECTED_COMPILER_COMMIT:
        raise RuntimeError(
            "The prior scaffold is not the exact successful pinned build. "
            f"Expected compiler {EXPECTED_COMPILER_COMMIT}; found {compiler.get('git_commit')}."
        )
    selection = manifest.get("selection", {})
    expected_selection = {
        "training_initial_cells": 12105,
        "validation_and_test_cells_used_for_ranking": 0,
        "contact_linked_peaks_before_ranking": 47575,
        "selected_candidate_peaks": 5000,
    }
    for key, expected in expected_selection.items():
        observed = selection.get(key)
        if observed != expected:
            raise RuntimeError(
                f"Prior selection audit failed for {key}: expected {expected}, found {observed}"
            )
    if manifest.get("output_counts") != EXPECTED_PRIOR_COUNTS:
        raise RuntimeError(
            "Prior edge counts differ from the verified scaffold: "
            f"{manifest.get('output_counts')}"
        )
    leakage = manifest.get("leakage_contract", {})
    if set(leakage.get("prior_fit_groups", [])) != EXPECTED_SPLIT["train"]:
        raise RuntimeError("Prior fit groups are not restricted to E/G/N.")
    if leakage.get("cohort_values_used_for_prior_selection") != [
        "training-subject pre ATAC counts"
    ]:
        raise RuntimeError("Unexpected cohort values were used for prior selection.")
    print("PASS: exact biological scaffold and leakage contract", flush=True)
    print(f"   prior manifest SHA-256: {sha256(path)}", flush=True)
    print(f"   edge counts: {manifest['output_counts']}", flush=True)
    return manifest


def validate_cohort() -> tuple[dict, dict]:
    manifest_path = COHORT / "manifest.json"
    report_path = COHORT / "build_report.json"
    require_files(
        [
            COHORT / "observations.npz",
            COHORT / "priors.npz",
            manifest_path,
            COHORT / "feature_registry.json",
            report_path,
        ],
        "cohort output",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert_split(manifest["split_groups"], "Temporal cohort")
    if set(manifest.get("priors_fit_groups", [])) != EXPECTED_SPLIT["train"]:
        raise RuntimeError("Cohort priors were not fit only on E/G/N.")
    controls = manifest.get("leakage_controls", {})
    required_controls = {
        "split_before_feature_selection": True,
        "cell_type_input": False,
        "pseudotime_input": False,
        "target_state_input": False,
        "fabricated_cell_pairs": False,
        "initial_rna_included": False,
    }
    if any(controls.get(key) != value for key, value in required_controls.items()):
        raise RuntimeError(f"Cohort leakage controls failed: {controls}")
    if manifest.get("initial_feature_names") != [
        "ATAC_peaks",
        "measured_cue:exercise",
    ]:
        raise RuntimeError(
            "Initial inputs must be ATAC plus the measured exercise cue only; "
            f"found {manifest.get('initial_feature_names')}"
        )
    if manifest.get("cue_names") != ["exercise"]:
        raise RuntimeError(f"Unexpected cue registry: {manifest.get('cue_names')}")
    if manifest.get("alignment_mode") != "distribution":
        raise RuntimeError("Destructive single-cell endpoints must use distribution alignment.")
    expected_limits = {"genes": 400, "peaks": 1000, "tfs": 64}
    for key, maximum in expected_limits.items():
        observed = int(report.get(key, 0))
        if observed < 2 or observed > maximum:
            raise RuntimeError(f"Invalid selected {key} count: {observed}")
    if int(report.get("initial_cells", 0)) < 100 or int(report.get("target_cells", 0)) < 100:
        raise RuntimeError("Too few cells survived temporal cohort construction.")
    print("PASS: cohort leakage and feature-input audit", flush=True)
    print(
        "   dimensions: "
        f"{report['initial_cells']} initial cells, {report['target_cells']} target cells, "
        f"{report['peaks']} peaks, {report['genes']} genes, {report['tfs']} TFs, "
        f"{report['signals']} signals",
        flush=True,
    )
    return manifest, report


def finite_metric(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def print_results(results: Mapping[str, object]) -> None:
    print("\n" + "=" * 72, flush=True)
    print("HELD-OUT TEMPORAL BENCHMARK SUMMARY", flush=True)
    print("=" * 72, flush=True)
    conditions = results.get("conditions", {})
    for condition, record in conditions.items():
        training = record["training"]
        print(
            f"\n{condition}: best epoch {training['best_epoch']} | "
            f"validation loss {finite_metric(training['best_validation_loss'])}",
            flush=True,
        )
        for group, metrics in record["test"]["by_group"].items():
            print(
                f"  test subject {group}: "
                f"RNA SWD={finite_metric(metrics.get('log_rna_swd'))}, "
                f"RNA mean MSE={finite_metric(metrics.get('log_rna_mean_mse'))}, "
                f"RNA mean Pearson={finite_metric(metrics.get('log_rna_mean_pearson'))}, "
                f"ATAC SWD={finite_metric(metrics.get('atac_swd'))}",
                flush=True,
            )
        diagnostics = record["test"].get("attractor_diagnostics", {})
        print(
            "  attractor diagnostics: "
            f"{diagnostics.get('status', 'N/A')} — "
            f"{diagnostics.get('reason', 'endpoint was not declared terminal')}",
            flush=True,
        )

    comparison = results.get("control_comparison", {})
    print("\nCircuit advantage (positive means true circuit has lower RNA SWD):", flush=True)
    for condition, by_group in comparison.get("by_condition_and_group", {}).items():
        for group, value in by_group.items():
            print(f"  versus {condition}, subject {group}: {finite_metric(value)}", flush=True)
    print(
        "\nINTERPRETATION BOUNDARY: this is subject-held-out 3.5-hour transient "
        "prediction. It does not by itself demonstrate a stable attractor.",
        flush=True,
    )


def main() -> None:
    print("WLD REAL-DATA TEMPORAL BUILD + TRAIN", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    try:
        import numpy
        import scipy
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "This runner expects a standard Colab runtime with NumPy, SciPy, and PyTorch."
        ) from exc
    print(
        f"NumPy: {numpy.__version__} | SciPy: {scipy.__version__} | "
        f"PyTorch: {torch.__version__}",
        flush=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: CUDA is unavailable; training will be much slower.", flush=True)

    export_files = [
        EXPORT / "rna.mtx.gz",
        EXPORT / "atac.mtx.gz",
        EXPORT / "genes.tsv",
        EXPORT / "peaks.tsv",
        EXPORT / "barcodes.tsv",
        EXPORT / "metadata.tsv",
        EXPORT / "split.json",
    ]
    prior_files = [
        PRIORS / "peak_gene_links.tsv",
        PRIORS / "motif_hits.tsv",
        PRIORS / "tf_gene_edges.tsv",
        PRIORS / "signaling_edges.tsv",
        PRIORS / "prior_manifest.json",
    ]
    require_files(export_files, "GSE240061 export")
    require_files(prior_files, "biological prior")
    print("PASS: exported matrices and prior tables are present", flush=True)
    prior_manifest = validate_prior_manifest()

    CODE.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading pinned model code from {DEPENDENCY_COMMIT}...", flush=True)
    downloaded = {name: download_pinned(name) for name in CODE_HASHES}
    for path in downloaded.values():
        subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)
    print("PASS: pinned code hashes and syntax", flush=True)

    build_command = [
        sys.executable,
        "-u",
        downloaded["build_wld_muscle_exercise_dataset.py"],
        "--rna-mtx",
        EXPORT / "rna.mtx.gz",
        "--atac-mtx",
        EXPORT / "atac.mtx.gz",
        "--genes",
        EXPORT / "genes.tsv",
        "--peaks",
        EXPORT / "peaks.tsv",
        "--barcodes",
        EXPORT / "barcodes.tsv",
        "--metadata",
        EXPORT / "metadata.tsv",
        "--peak-gene-links",
        PRIORS / "peak_gene_links.tsv",
        "--motif-hits",
        PRIORS / "motif_hits.tsv",
        "--tf-gene-edges",
        PRIORS / "tf_gene_edges.tsv",
        "--signaling-edges",
        PRIORS / "signaling_edges.tsv",
        "--split-json",
        EXPORT / "split.json",
        "--output",
        COHORT,
        "--max-genes",
        "400",
        "--max-peaks",
        "1000",
        "--max-tfs",
        "64",
        "--seed",
        "42",
        "--overwrite",
    ]
    print(
        "\nBuilding ATAC + measured-exercise-cue cohort. "
        "No initial RNA, cell labels, pseudotime, or fabricated cell pairs are supplied.",
        flush=True,
    )
    run_logged(build_command, ROOT / "wld_temporal_cohort_build.log")
    cohort_manifest, cohort_report = validate_cohort()

    validate_command = [
        sys.executable,
        "-u",
        downloaded["wld_temporal_training.py"],
        "validate",
        "--data",
        COHORT,
    ]
    run_logged(validate_command, ROOT / "wld_temporal_cohort_validation.log")

    benchmark_command = [
        sys.executable,
        "-u",
        downloaded["wld_temporal_training.py"],
        "benchmark",
        "--data",
        COHORT,
        "--output",
        RESULTS,
        "--epochs",
        "100",
        "--steps",
        "12",
        "--patience",
        "8",
        "--conditions",
        "true_circuit,no_circuit",
        "--device",
        device,
    ]
    print(
        "\nTraining the hard biological circuit and the no-circuit control. "
        "Validation subject I selects checkpoints; test subjects J/L remain sealed "
        "until model selection is complete.",
        flush=True,
    )
    run_logged(benchmark_command, ROOT / "wld_temporal_benchmark.log")

    results_path = RESULTS / "wld_temporal_results.json"
    require_files(
        [
            results_path,
            RESULTS / "wld_temporal_true_circuit.pt",
            RESULTS / "wld_temporal_no_circuit.pt",
        ],
        "temporal benchmark",
    )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert_split(results["split_groups"], "Temporal benchmark")
    if set(results.get("conditions", {})) != {"true_circuit", "no_circuit"}:
        raise RuntimeError("Both true-circuit and no-circuit results are required.")

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "dependency_commit": DEPENDENCY_COMMIT,
        "code_sha256": CODE_HASHES,
        "prior_compiler_commit": prior_manifest["compiler"]["git_commit"],
        "prior_manifest_sha256": sha256(PRIORS / "prior_manifest.json"),
        "cohort_manifest_sha256": sha256(COHORT / "manifest.json"),
        "cohort_build_report": cohort_report,
        "device": device,
        "benchmark_config": {
            "seed": 42,
            "epochs": 100,
            "integration_steps": 12,
            "patience": 8,
            "conditions": ["true_circuit", "no_circuit"],
        },
        "claim_boundary": cohort_manifest["dataset"]["scope"],
        "logs": {
            "build": str(ROOT / "wld_temporal_cohort_build.log"),
            "validation": str(ROOT / "wld_temporal_cohort_validation.log"),
            "benchmark": str(ROOT / "wld_temporal_benchmark.log"),
        },
    }
    (RESULTS / "temporal_run_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print_results(results)
    print("\nCOMPLETE", flush=True)
    print(f"Results: {results_path}", flush=True)
    print(f"Provenance: {RESULTS / 'temporal_run_manifest.json'}", flush=True)
    print(f"Benchmark log: {ROOT / 'wld_temporal_benchmark.log'}", flush=True)


if __name__ == "__main__":
    main()
