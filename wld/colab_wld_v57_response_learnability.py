"""Pinned, restart-safe Colab launcher for the WLD v5.7 diagnostic ladder."""

from google.colab import drive

drive.mount("/content/drive")

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
# Immutable commit containing the verified v5.7 implementation.
SOURCE_REF = "7384d6837a533ac8c71bf90075831de2939cf672"
BACKUP = Path("/content/drive/MyDrive/WLD_Backup")
CODE = Path("/content/wld_v57_response_learnability_code")
PACKAGES = Path("/content/wld_v57_packages_py312_np1264")
PHASE_B = BACKUP / "wld_phase_b"
V53_BUNDLE = BACKUP / "wld_v53_crispr_sciatac_ingestion" / "bundle"
V55_ROOT = BACKUP / "wld_v55_chromatin_twin_r3"
V56_AUDIT = (
    BACKUP
    / "wld_v56_nullaware_r2"
    / "practical_effect_audit"
    / "wld_v56_practical_effect_audit.json"
)
OUTPUT = BACKUP / "wld_v57_response_learnability"
LOG = OUTPUT / "wld_v57_complete.log"
REPORT = OUTPUT / "wld_v57_response_learnability_report.json"

# SHA-256 values from the immutable source commit above.
FILES = {
    "wld_chromatin_modules_v55.py": (
        "604de65d8174975dbe85e4523924732ff600a25b3c863fee2b888b6a1948b501"
    ),
    "wld_response_learnability_v57.py": (
        "0129ce665568f28c61577eb03f439c6dcd4b8a363ad955741aedfbcef7417e1c"
    ),
    "run_wld_v57_learnability_smoke.py": (
        "6c8c12773cbf3f4ec4280c9b8814c6c0c51b02862edd979b613a13760c576974"
    ),
    "run_wld_v57_learnability_colab.py": (
        "05af7bd2b1ae558e0594369338a873ca908c153151d6fddd1509dbed9c45c684"
    ),
    "wld_v57_learnability_contract.md": (
        "b494dabf38613d50734ccd2594e02570e552733dbe433302a6195e0b0964ef38"
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, expected: str) -> None:
    destination = CODE / name
    if destination.is_file() and sha256_file(destination) == expected:
        print(f"   PASS cached: {name}")
        return
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_REF}/wld/"
        f"{urllib.parse.quote(name)}"
    )
    payload = None
    last_error = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "WLD-v5.7-response-learnability/1.0",
                    "Cache-Control": "no-cache",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            break
        except Exception as error:
            last_error = error
            if attempt < 5:
                time.sleep(min(2**attempt, 16))
    if payload is None:
        raise RuntimeError(f"Could not download pinned source: {name}") from last_error
    observed = sha256_bytes(payload)
    if observed != expected:
        raise RuntimeError(f"Source hash mismatch for {name}: {observed} != {expected}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    print(f"   PASS downloaded: {name}")


def probe(environment):
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import numpy,scipy; "
                "assert numpy.__version__=='1.26.4'; "
                "assert scipy.__version__=='1.16.3'; "
                "from scipy import sparse; assert sparse.csr_matrix([[1]]).nnz==1; "
                "print('NumPy',numpy.__version__,'| SciPy',scipy.__version__)"
            ),
        ],
        env=environment,
        text=True,
        capture_output=True,
    )


CODE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

print("WLD V5.7 RESPONSE-LEARNABILITY DIAGNOSTIC")
print("This is development-only and does not train a fresh WLD model.")
print("The v5.3 test targets, muscle J/L, and external test studies stay sealed.\n")

if SOURCE_REF.startswith("__") or any(value.startswith("__") for value in FILES.values()):
    raise RuntimeError("This launcher was not pinned to its immutable source commit")

print("1. Downloading and SHA-verifying the exact v5.7 implementation...")
for filename, digest in FILES.items():
    fetch(filename, digest)

print("\n2. Preparing an isolated numerical environment...")
environment = os.environ.copy()
environment["PYTHONPATH"] = str(CODE) + os.pathsep + str(PACKAGES)
environment["PYTHONNOUSERSITE"] = "1"
environment["PYTHONUNBUFFERED"] = "1"
environment["OMP_NUM_THREADS"] = "2"
environment["MKL_NUM_THREADS"] = "2"

checked = probe(environment)
if checked.returncode:
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
        ],
        check=True,
    )
    checked = probe(environment)
if checked.returncode:
    print(checked.stdout)
    print(checked.stderr)
    raise RuntimeError("Could not create the isolated NumPy/SciPy environment")
print(checked.stdout.strip())
print("   PASS: compatible isolated packages")

print("\n3. Compiling the pinned implementation...")
for filename in FILES:
    if filename.endswith(".py"):
        py_compile.compile(str(CODE / filename), doraise=True)
print("   PASS: Python compilation")

required = {
    "Phase B human prior manifest": PHASE_B / "priors" / "homo_sapiens_grch38" / "prior_manifest.json",
    "Phase B feature vocabulary": PHASE_B / "priors" / "homo_sapiens_grch38" / "feature_vocab.json",
    "v5.3 ingestion manifest": V53_BUNDLE / "wld_v53_ingestion_manifest.json",
    "v5.3 whole-target split": V53_BUNDLE / "whole_target_split.json",
    "v5.3 sparse response matrix": V53_BUNDLE / "atac_counts.GRCh38.2kb.npz",
    "v5.3 cell metadata": V53_BUNDLE / "cells.tsv.gz",
    "v5.3 response bins": V53_BUNDLE / "bins.GRCh38.2kb.tsv.gz",
    "v5.5 TF routes": V55_ROOT / "tf_routes" / "regulator_tf_routes.npz",
    "v5.5 TF-route vocabulary": V55_ROOT / "tf_routes" / "route_vocab.json",
    "v5.5 TF-route manifest": V55_ROOT / "tf_routes" / "route_manifest.json",
    "v5.5 TF-route table": V55_ROOT / "tf_routes" / "regulator_tf_routes.tsv.gz",
    "v5.5 complex modules": V55_ROOT / "complex_modules" / "complex_accessibility_modules.npz",
    "v5.5 module manifest": V55_ROOT / "complex_modules" / "complex_accessibility_module_manifest.json",
    "v5.5 module vocabulary": V55_ROOT / "complex_modules" / "complex_accessibility_vocab.json",
    "v5.6 failed practical audit": V56_AUDIT,
}
missing = [
    f"{label}: {path}"
    for label, path in required.items()
    if not path.is_file() or path.stat().st_size == 0
]
if missing:
    raise FileNotFoundError(
        "Missing durable upstream WLD artifacts. Restore WLD_Backup first:\n"
        + "\n".join(missing)
    )
print("   PASS: durable upstream artifacts found")

print("\n4. Running the restart-safe diagnostic ladder...")
print("   This can take time because it repeatedly streams pseudobulk responses.")
command = [
    sys.executable,
    "-u",
    str(CODE / "run_wld_v57_learnability_colab.py"),
    "--phase-b-root",
    str(PHASE_B),
    "--v53-bundle",
    str(V53_BUNDLE),
    "--route-root",
    str(V55_ROOT / "tf_routes"),
    "--module-root",
    str(V55_ROOT / "complex_modules"),
    "--v56-audit",
    str(V56_AUDIT),
    "--output-root",
    str(OUTPUT),
]
with LOG.open("a", buffering=1) as log_handle:
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
    return_code = process.wait()
if return_code:
    raise RuntimeError(
        f"WLD v5.7 exited with code {return_code}. Complete log: {LOG}. "
        "Rerun this same cell to resume."
    )

if not REPORT.is_file() or REPORT.stat().st_size == 0:
    raise FileNotFoundError(f"Completed runner did not write {REPORT}")
report = json.loads(REPORT.read_text())
claims = report.get("claims", {})
if claims.get("test_values_materialized") is not False or claims.get(
    "test_targets_evaluated"
) is not False:
    raise RuntimeError("Completed report crossed the sealed-test boundary")
if claims.get("fresh_wld_training") is not False:
    raise RuntimeError("Completed report incorrectly claims fresh WLD training")

print("\nVERIFIED COMPLETE")
print("Primary diagnosis:", report["diagnosis"]["primary_failure_class"])
print("Next action:", report["diagnosis"]["next_action"])
print("Fresh WLD training: False")
print("Sealed test opened: False")
print(f"Report: {REPORT}")
print(f"Log:    {LOG}")
