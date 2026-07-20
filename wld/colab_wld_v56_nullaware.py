"""Pinned, restart-safe Colab launcher for WLD v5.6 null-aware development."""

from google.colab import drive

drive.mount("/content/drive")

import datetime as dt
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "nlipieta/MultiscaleProject"
SOURCE_REF = "c54be0e01e3309a88b43516be8f2c44370104ed6"
BACKUP = Path("/content/drive/MyDrive/WLD_Backup")
CODE = Path("/content/wld_v56_code")
PACKAGES = Path("/content/wld_v56_packages_py312_np1264")
PHASE_B = BACKUP / "wld_phase_b"
CORPUS = BACKUP / "wld_corpus_pretraining"
V53_BUNDLE = BACKUP / "wld_v53_crispr_sciatac_ingestion" / "bundle"
V55_ROOT = BACKUP / "wld_v55_chromatin_twin_r3"
OUTPUT = BACKUP / "wld_v56_nullaware_r2"
LAUNCHER_MANIFEST = OUTPUT / "launcher_provenance.json"
LOG = OUTPUT / "wld_v56_complete.log"
REPORT = OUTPUT / "development" / "wld_v56_null_aware_development_report.json"

FILES = {
    "wld_circuit_dynamics_v3.py": "2ffcd9d0a60551dd06db2646c60747ba0680e47150fd5f91bf42b7d8eadfe068",
    "wld_foundation_model_v4.py": "0999e2f5de11883dfb05e18d2cddc272b4501aba16d43cef1600afaa27cf7071",
    "wld_foundation_data.py": "446d52ea61f882ba6aaeeec275077213a3d94dc4b705d5d8427081111134720a",
    "wld_phase_b_priors.py": "d3b216b1d11c7ec3f767126787abc5df9abc2cc22f6b6c12d1bc2bc6566d58ce",
    "wld_chromatin_response_v54.py": "7743c61e415dbe3fc9bb941448d99c07a1d6ec3469235c585964ef58a7340579",
    "wld_chromatin_training_v54.py": "24b0a314d38730045745fc6361f1912581e7730b50c88d97acf1ac5c615cdfa4",
    "wld_chromatin_twin_v55.py": "c9ec8a16c1355dd59f02fcf8492dff3bf4ed089575b4413d2d4568dbb4c16e4e",
    "wld_chromatin_modules_v55.py": "eecda349e2bba4a03071fb9018cff87ff6c480c3a68e6b621b0c242512f3f91a",
    "wld_twin_statistics_v55.py": "b0bc34f52d77bbe396b8f0111907321415daadf3b2dab293c8902069a360f25e",
    "wld_chromatin_twin_training_v55.py": "a3315be4afe6a325474ac6842af0524e48169761324b25a401e5355d3fb7e18a",
    "wld_chromatin_twin_v56.py": "8c44e7b8b355cb9dceabf49c70861adac6bdc697b000531b7c44586766dc3be5",
    "wld_chromatin_twin_training_v56.py": "42646a587df132199be2b4741ef673ba2b2606af6353cb04307bf7874aafc0e2",
    "wld_v56_topology_controls.py": "b95338a5d70bfc0362cc3073c16acdcb3565ea71e74bd0a1ca6c3623d5490a37",
    "run_wld_v56_nullaware_smoke.py": "9f838b3cc919a02637fbda1e1735b2510c5f04fa9cb8878d98855d2bd7159f4b",
    "run_wld_v56_nullaware_colab.py": "e82a98e4db6c02340127af6d584558f7d922d4476e50d1653b5ea687f47db344",
    "wld_v56_nullaware_contract.md": "6f69de6b7dd76077cb5a5ddaa7028f8d19cd706fb734242471fff5210890ebeb",
}

RUN_CONFIG = {
    "epochs": 36,
    "targets_per_epoch": 36,
    "batch_size": 48,
    "patience": 8,
    "control_replicates": 10,
    "seeds": [42, 137, 911],
    "device": "cuda",
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fetch_bytes(url, *, attempts=6, timeout=180):
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "WLD-v5.6-Colab/1.0",
                    "Cache-Control": "no-cache",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                delay = min(2**attempt, 20)
                print(f"   retry {attempt + 1}/{attempts - 1} after {error}")
                time.sleep(delay)
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}") from last_error


def download_verified_source(name, expected_hash):
    destination = CODE / name
    if destination.is_file() and sha256_file(destination) == expected_hash:
        print(f"   PASS cached: {name}")
        return
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_REF}/wld/"
        f"{urllib.parse.quote(name)}"
    )
    data = fetch_bytes(url)
    observed = sha256_bytes(data)
    if observed != expected_hash:
        raise RuntimeError(
            f"Pinned source hash mismatch for {name}: {observed} != {expected_hash}"
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    print(f"   PASS downloaded: {name}")


def environment_probe(child_env):
    command = [
        sys.executable,
        "-c",
        (
            "import h5py,numpy,scipy,torch; "
            "assert numpy.__version__=='1.26.4'; "
            "assert scipy.__version__=='1.16.3'; "
            "assert h5py.__version__=='3.16.0'; "
            "assert torch.cuda.is_available(), "
            "'Select Runtime > Change runtime type > T4 GPU'; "
            "print('NumPy',numpy.__version__,'| SciPy',scipy.__version__,"
            "'| h5py',h5py.__version__,'| PyTorch',torch.__version__); "
            "print('GPU:',torch.cuda.get_device_name(0))"
        ),
    ]
    return subprocess.run(command, env=child_env, text=True, capture_output=True)


CODE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

print("WLD V5.6 PINNED NULL-AWARE DEVELOPMENT LAUNCHER")
print(f"Repository source commit: {SOURCE_REF}")
print("Previously inspected validation targets are reused for development.")
print("Sealed test targets, muscle J/L, and external test studies remain closed.\n")

print("1. Downloading and SHA-verifying the exact v5.6 implementation...")
for filename, expected in FILES.items():
    download_verified_source(filename, expected)

print("\n2. Checking the isolated numerical environment...")
child_env = os.environ.copy()
child_env["PYTHONPATH"] = str(CODE) + os.pathsep + str(PACKAGES)
child_env["PYTHONNOUSERSITE"] = "1"
child_env["PYTHONUNBUFFERED"] = "1"
child_env["MPLBACKEND"] = "Agg"
child_env["OMP_NUM_THREADS"] = "2"
child_env["MKL_NUM_THREADS"] = "2"

probe = environment_probe(child_env)
if probe.returncode:
    # PACKAGES is an ephemeral directory owned only by this launcher.
    if PACKAGES.exists():
        shutil.rmtree(PACKAGES)
    PACKAGES.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "--target",
            str(PACKAGES),
            "--no-cache-dir",
            "numpy==1.26.4",
            "scipy==1.16.3",
            "h5py==3.16.0",
        ],
        check=True,
    )
    probe = environment_probe(child_env)
if probe.returncode:
    print(probe.stdout)
    print(probe.stderr)
    raise RuntimeError(
        "Could not create the isolated WLD environment. Confirm this is a fresh "
        "Colab Python 3.12 runtime with a T4 GPU, then rerun this cell."
    )
print(probe.stdout.strip())
print("   PASS: compatible isolated packages and CUDA")

print("\n3. Compiling the pinned implementation...")
for filename in FILES:
    if filename.endswith(".py"):
        py_compile.compile(str(CODE / filename), doraise=True)
print("   PASS: Python compilation")

required = {
    "Phase B prior manifest": PHASE_B / "priors" / "homo_sapiens_grch38" / "prior_manifest.json",
    "Phase B numeric priors": PHASE_B / "priors" / "homo_sapiens_grch38" / "foundation_priors.npz",
    "Phase B feature vocabulary": PHASE_B / "priors" / "homo_sapiens_grch38" / "feature_vocab.json",
    "expanded-corpus checkpoint": CORPUS / "wld_corpus_pretrained_model.pt",
    "expanded-corpus report": CORPUS / "wld_corpus_pretraining_report.json",
    "v5.3 ingestion manifest": V53_BUNDLE / "wld_v53_ingestion_manifest.json",
    "v5.3 whole-target split": V53_BUNDLE / "whole_target_split.json",
    "v5.3 full response matrix": V53_BUNDLE / "atac_counts.GRCh38.2kb.npz",
    "v5.3 cell metadata": V53_BUNDLE / "cells.tsv.gz",
    "v5.3 response bins": V53_BUNDLE / "bins.GRCh38.2kb.tsv.gz",
    "v5.5 TF route manifest": V55_ROOT / "tf_routes" / "route_manifest.json",
    "v5.5 TF route vocabulary": V55_ROOT / "tf_routes" / "route_vocab.json",
    "v5.5 TF route tensors": V55_ROOT / "tf_routes" / "regulator_tf_routes.npz",
    "v5.5 TF route table": V55_ROOT / "tf_routes" / "regulator_tf_routes.tsv.gz",
    "v5.5 complex-module manifest": V55_ROOT / "complex_modules" / "complex_accessibility_module_manifest.json",
    "v5.5 complex-module vocabulary": V55_ROOT / "complex_modules" / "complex_accessibility_vocab.json",
    "v5.5 complex-module tensors": V55_ROOT / "complex_modules" / "complex_accessibility_modules.npz",
    "v5.5 complex-module table": V55_ROOT / "complex_modules" / "complex_accessibility_modules.tsv",
    "v5.5 complex-module construction targets": V55_ROOT / "complex_modules" / "complex_module_construction_targets.tsv",
    "v5.5 completed report": V55_ROOT / "development" / "wld_v55_chromatin_twin_report.json",
}
missing = [
    f"{label}: {path}"
    for label, path in required.items()
    if not path.is_file() or path.stat().st_size == 0
]
if missing:
    raise FileNotFoundError(
        "Missing durable upstream WLD artifacts. Restore the Drive backup first:\n"
        + "\n".join(missing)
    )
print("   PASS: Phase B, corpus, v5.3, and completed v5.5 artifacts found")

launcher_lock = {
    "schema_version": "wld-v5.6-colab-launcher-lock",
    "repository": REPOSITORY,
    "source_ref": SOURCE_REF,
    "source_sha256": FILES,
    "v55_root": str(V55_ROOT),
    "run_config": RUN_CONFIG,
    "claims": {
        "previously_inspected_validation_reused": True,
        "untouched_audit_inference": False,
        "test_targets_evaluated": False,
        "muscle_j_l_evaluated": False,
        "external_test_studies_evaluated": False,
        "digital_twin_claim": False,
        "attractor_claim": False,
    },
}
if LAUNCHER_MANIFEST.is_file():
    existing_lock = json.loads(LAUNCHER_MANIFEST.read_text())
    if existing_lock != launcher_lock:
        raise RuntimeError(
            "The existing v5.6 output directory belongs to a different locked "
            "launcher. Preserve it and use a new output directory."
        )
else:
    atomic_json(LAUNCHER_MANIFEST, launcher_lock)
print("   PASS: immutable launcher provenance lock")

command = [
    sys.executable,
    "-u",
    str(CODE / "run_wld_v56_nullaware_colab.py"),
    "--phase-b-root",
    str(PHASE_B),
    "--corpus-root",
    str(CORPUS),
    "--v53-bundle",
    str(V53_BUNDLE),
    "--v55-root",
    str(V55_ROOT),
    "--output-root",
    str(OUTPUT),
    "--epochs",
    str(RUN_CONFIG["epochs"]),
    "--targets-per-epoch",
    str(RUN_CONFIG["targets_per_epoch"]),
    "--batch-size",
    str(RUN_CONFIG["batch_size"]),
    "--patience",
    str(RUN_CONFIG["patience"]),
    "--control-replicates",
    str(RUN_CONFIG["control_replicates"]),
    "--seeds",
    ",".join(map(str, RUN_CONFIG["seeds"])),
    "--device",
    RUN_CONFIG["device"],
]

print("\n4. Starting restart-safe v5.6 development...")
print("   Fits: 3 seeds x (true null-aware routes + 10 matched controls) = 33")
print("   This is substantially larger than v5.5 and can take several hours.")
print("   Rerun this exact cell after a disconnect; completed fits are retained.")
print("   This is disclosed development reuse, not a confirmatory test.\n")

with LOG.open("a", encoding="utf-8") as log_handle:
    boundary = (
        "\n"
        + "=" * 78
        + "\n"
        + f"LAUNCH {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
        + "=" * 78
        + "\n"
    )
    log_handle.write(boundary)
    log_handle.flush()
    process = subprocess.Popen(
        command,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
        log_handle.flush()
    return_code = process.wait()

if return_code:
    print("\n" + "=" * 78)
    print("FINAL 200 CHILD-PROCESS LOG LINES")
    print("=" * 78)
    print("\n".join(LOG.read_text(errors="replace").splitlines()[-200:]))
    raise RuntimeError(
        f"WLD v5.6 exited with code {return_code}. "
        f"The complete log is at {LOG}. Rerun this same cell to resume."
    )

if not REPORT.is_file() or REPORT.stat().st_size == 0:
    raise RuntimeError(f"v5.6 exited without its final report: {REPORT}")
result = json.loads(REPORT.read_text())
claims = result.get("claims", {})
if any(
    claims.get(name) is not False
    for name in (
        "untouched_audit_inference",
        "confidence_interval_claim",
        "p_value_claim",
        "test_targets_materialized",
        "test_targets_evaluated",
        "external_subject_study_evaluated",
        "ode_time_scale_identified",
        "fixed_point_claim",
        "basin_claim",
        "digital_twin_claim",
        "attractor_claim",
    )
):
    raise RuntimeError("The completed v5.6 report crossed a scientific boundary")
if any(
    claims.get(name) is not True
    for name in (
        "development_only",
        "validation_targets_previously_used_in_v55",
        "all_existing_validation_targets_evaluated",
        "perturbed_mean_baseline_training_only",
    )
):
    raise RuntimeError("The completed v5.6 report omitted required development disclosures")

checks = result.get("development_checks", {})
effects = result.get("descriptive_effects", {})
print("\n" + "=" * 78)
print("VERIFIED COMPLETE: WLD V5.6 NULL-AWARE DEVELOPMENT")
print("=" * 78)
print(f"Report: {REPORT}")
print(f"Log:    {LOG}")
for label, key in (
    ("Persistence minus true", "persistence_minus_true"),
    ("Perturbed mean minus true", "perturbed_mean_minus_true"),
    ("Matched controls minus true", "control_mean_minus_true"),
    ("Frozen all routes minus true", "frozen_all_minus_true"),
):
    value = effects.get(key, {})
    print(f"{label:33s} {value.get('mean', float('nan')):+.8f}")
print(f"Mean response NRMSE:             {checks.get('mean_response_nrmse')}")
print(f"Mean response cosine:            {checks.get('mean_response_cosine')}")
print(
    "Eligible to freeze a new confirmation plan: "
    f"{checks.get('eligible_to_freeze_new_confirmation_plan')}"
)
print("Inference from reused validation: False")
print("Test targets evaluated:           False")
print("Digital-twin claim:               False")
print("Attractor claim:                  False")
