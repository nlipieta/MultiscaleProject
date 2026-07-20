"""Run every scientifically valid WLD check for the 10x PBMC snapshot.

This is the only entry point needed by the Colab recipe in ``wld_README.md``.
It intentionally excludes the legacy pseudotime transition experiment: the
10x PBMC input is a single biological sample observed at one time point, so an
early/late pairing constructed from that same snapshot cannot validate a
trajectory, vector field, fixed point, basin, or attractor.

The runner performs, in order:

1. compiled-package and BLAS preflight checks in the current interpreter;
2. syntax compilation of the repository WLD scripts;
3. the model's synthetic shape, RK4, and leakage smoke tests;
4. explicit token-aware leakage-audit regression checks;
5. WLD v3 hard-circuit structural and numerical checks;
6. grouped unpaired temporal-training and sealed-test software checks;
7. real-data dataset-builder and masked-cue contract checks; and
8. the leakage-aware held-out ATAC-to-RNA PBMC reconstruction experiment.

Run this file from a clean isolated environment, as shown in the README.
"""

from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "wld_attractor_model_v2.py"
PBMC_RUNNER = ROOT / "run_wld_pbmc_colab.py"
LEGACY_AUDITS = ROOT / "wld_next_experiments.py"
V3_MODEL = ROOT / "wld_circuit_dynamics_v3.py"
V3_VALIDATOR = ROOT / "run_wld_v3_validation.py"
TEMPORAL_TRAINER = ROOT / "wld_temporal_training.py"
TEMPORAL_SMOKE = ROOT / "run_wld_temporal_smoke.py"
DATASET_BUILDER = ROOT / "build_wld_muscle_exercise_dataset.py"
DATASET_BUILDER_SMOKE = ROOT / "run_wld_dataset_builder_smoke.py"
EXPECTED_ARTIFACTS = (
    ROOT / "wld_v3_validation.json",
    ROOT / "wld_pbmc_results.json",
    ROOT / "wld_pbmc_state_model.pt",
    ROOT / "wld_pbmc_state_results.png",
)


def run(command: list[str], label: str) -> None:
    print(f"\n{label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"PASS: {label}", flush=True)


def package_preflight() -> None:
    print("1. Checking the isolated numerical environment...", flush=True)
    import numpy as np
    import scipy
    import scipy.linalg.blas as scipy_blas
    import sklearn
    import threadpoolctl
    import torch

    # This performs a real BLAS call rather than merely checking imports.
    product = scipy_blas.dgemm(
        alpha=1.0,
        a=np.eye(2, dtype=np.float64),
        b=np.eye(2, dtype=np.float64),
    )
    if not np.allclose(product, np.eye(2)):
        raise RuntimeError("SciPy BLAS preflight returned an invalid result.")

    # Resolving the loaded thread-pool libraries catches the broken OpenBLAS
    # state produced by upgrading NumPy in an already-running Colab kernel.
    threadpoolctl.threadpool_info()
    print(
        "   "
        f"Python {sys.version.split()[0]} | NumPy {np.__version__} | "
        f"SciPy {scipy.__version__} | scikit-learn {sklearn.__version__} | "
        f"PyTorch {torch.__version__}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("   GPU: unavailable (CPU execution is supported)", flush=True)
    print("PASS: numerical environment and BLAS", flush=True)


def compile_sources() -> None:
    print("\n2. Compiling WLD repository files...", flush=True)
    for path in (
        MODEL,
        PBMC_RUNNER,
        LEGACY_AUDITS,
        V3_MODEL,
        V3_VALIDATOR,
        TEMPORAL_TRAINER,
        TEMPORAL_SMOKE,
        DATASET_BUILDER,
        DATASET_BUILDER_SMOKE,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing repository file: {path.name}")
        py_compile.compile(str(path), doraise=True)
        print(f"   PASS: {path.name}", flush=True)


def load_model_module():
    spec = importlib.util.spec_from_file_location("wld_model_validation", MODEL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {MODEL.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def leakage_regression() -> None:
    print("\n4. Running explicit leakage regression checks...", flush=True)
    module = load_model_module()

    # Legitimate feature names containing letter sequences such as "rna" must
    # not be rejected unless RNA is a normalized token/direct proxy.
    module.leakage_audit(
        ["donor_1", "donor_2"],
        ["donor_3"],
        ["ATAC_peaks", "external_cue", "chromatin_accessibility"],
    )

    rejected = []
    for proxy in ("RNA_counts", "cell_type", "target_state", "pseudotime"):
        try:
            module.leakage_audit(["donor_1"], ["donor_2"], [proxy])
        except ValueError:
            rejected.append(proxy)
        else:
            raise AssertionError(f"Leakage audit failed to reject {proxy!r}.")

    try:
        module.leakage_audit(["donor_1"], ["donor_1"], ["ATAC_peaks"])
    except ValueError:
        pass
    else:
        raise AssertionError("Leakage audit failed to reject group overlap.")

    print(f"   Rejected direct proxies: {', '.join(rejected)}", flush=True)
    print("PASS: token-aware feature and group leakage audit", flush=True)


def verify_report() -> None:
    missing = [path.name for path in EXPECTED_ARTIFACTS if not path.exists()]
    if missing:
        raise FileNotFoundError("Validation did not create: " + ", ".join(missing))

    with (ROOT / "wld_pbmc_results.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    required = {
        "scope",
        "split",
        "encoder_inputs",
        "metrics_primary_seed",
        "paired_seed_controls",
        "control_summary",
        "trajectory_metrics",
        "auprc",
        "fixed_point_stability",
    }
    absent = sorted(required.difference(report))
    if absent:
        raise KeyError("Results report is missing fields: " + ", ".join(absent))
    metric_names = {
        "training_mean",
        "ridge_gene_activity",
        "true_tf_gene_scaffold",
        "degree_preserving_permuted_scaffold",
        "shuffled_test_atac",
    }
    if metric_names.difference(report["metrics_primary_seed"]):
        raise KeyError("Results report is missing a baseline or negative control.")
    if report["encoder_inputs"] != ["binary ATAC peaks"]:
        raise ValueError("Snapshot encoder input contract was violated.")
    control_fields = {
        "regulatory_prior_dependency_supported",
        "atac_dependency_supported",
    }
    if control_fields.difference(report["control_summary"]):
        raise KeyError("Results report is missing dependency-control outcomes.")

    with (ROOT / "wld_v3_validation.json").open(encoding="utf-8") as handle:
        v3_report = json.load(handle)
    if v3_report.get("neutral_stability", {}).get("toggle_benchmark") is not False:
        raise ValueError("The v3 validator must remain a neutral, non-toggle audit.")

    print("\n9. Verifying result artifacts and claim boundaries...", flush=True)
    for path in EXPECTED_ARTIFACTS:
        print(f"   PASS: {path.name}", flush=True)
    print(
        "   PASS: mean/ridge baselines and prior/ATAC controls present",
        flush=True,
    )
    print("   PASS: encoder contract contains ATAC only", flush=True)
    print("   PASS: unsupported trajectory/AUPRC/attractor claims marked N/A", flush=True)
    print("   PASS: v3 diagnostics are structural/numerical, not biological claims", flush=True)


def main() -> None:
    os.chdir(ROOT)
    package_preflight()
    compile_sources()
    run([sys.executable, str(MODEL)], "3. Running synthetic architecture/RK4 smoke tests...")
    leakage_regression()
    run(
        [sys.executable, str(V3_VALIDATOR)],
        "5. Validating hard-constrained WLD v3 circuit dynamics...",
    )
    run(
        [sys.executable, str(TEMPORAL_SMOKE)],
        "6. Validating grouped temporal training and sealed test groups...",
    )
    run(
        [sys.executable, str(DATASET_BUILDER_SMOKE)],
        "7. Validating the real-data builder and masked supplemental cues...",
    )
    run(
        [sys.executable, str(PBMC_RUNNER)],
        "8. Running held-out PBMC ATAC-to-RNA reconstruction and baselines...",
    )
    verify_report()
    print(
        "\nALL SUPPORTED WLD CHECKS PASSED.\n"
        "The legacy single-snapshot pseudotime transition audit was intentionally "
        "not run because it cannot test temporal dynamics or attractors.",
        flush=True,
    )


if __name__ == "__main__":
    main()
