# WLD v5.4.1 — response-calibrated, restart-safe development launcher
from google.colab import drive

drive.mount("/content/drive")

import hashlib
import os
import py_compile
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPOSITORY = "nlipieta/MultiscaleProject"
REF = "f10ac0584dde0728a2cbf990bd469d8b68564990"
BACKUP = Path("/content/drive/MyDrive/WLD_Backup")
CODE = Path("/content/wld_v541_code")
PACKAGES = Path("/content/wld_v54_packages")
PHASE_B = BACKUP / "wld_phase_b"
CORPUS = BACKUP / "wld_corpus_pretraining"
V53_BUNDLE = (
    BACKUP / "wld_v53_crispr_sciatac_ingestion" / "bundle"
)
PRIOR_SOURCES = BACKUP / "wld_real_data" / "prior_sources"
OUTPUT = BACKUP / "wld_v541_response_calibrated"
LOG = OUTPUT / "wld_v541_complete.log"

FILES = {
    "wld_circuit_dynamics_v3.py": "2ffcd9d0a60551dd06db2646c60747ba0680e47150fd5f91bf42b7d8eadfe068",
    "wld_foundation_model_v4.py": "0999e2f5de11883dfb05e18d2cddc272b4501aba16d43cef1600afaa27cf7071",
    "wld_foundation_data.py": "446d52ea61f882ba6aaeeec275077213a3d94dc4b705d5d8427081111134720a",
    "wld_phase_b_priors.py": "d3b216b1d11c7ec3f767126787abc5df9abc2cc22f6b6c12d1bc2bc6566d58ce",
    "wld_chromatin_response_v54.py": "7743c61e415dbe3fc9bb941448d99c07a1d6ec3469235c585964ef58a7340579",
    "wld_chromatin_training_v54.py": "24b0a314d38730045745fc6361f1912581e7730b50c88d97acf1ac5c615cdfa4",
    "run_wld_v54_chromatin_smoke.py": "dbddb49471ae6452576367430e7f923f6d12435f058c66285c078835ef187bdd",
    "run_wld_v54_training_smoke.py": "73a059dab3f2e99c737a63daab7c685d925f62d6601c2366d34085b1ed40aa9e",
    "run_wld_v54_chromatin_colab.py": "4ea9e16911143c82dc84f0a5ba0deeb86785b4108b5ea73b931220579350e817",
    "wld_v54_chromatin_response_contract.md": "0252406ffbb8caf8317b372ed036aed211816533613fb026532c5f162dd95b79",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download_verified(name, expected):
    destination = CODE / name
    if destination.is_file() and digest(destination.read_bytes()) == expected:
        print(f"  PASS cached: {name}")
        return
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REF}/wld/{name}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "WLD-v5.4-Colab/1.0",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    observed = digest(data)
    if observed != expected:
        raise RuntimeError(
            f"Pinned hash mismatch for {name}: {observed} != {expected}"
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    print(f"  PASS downloaded: {name}")


CODE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
print("WLD V5.4.1 PINNED RESPONSE-CALIBRATED LAUNCHER")
print(f"Repository ref: {REF}")

print("\n1. Downloading and verifying the pinned implementation...")
for filename, expected_hash in FILES.items():
    download_verified(filename, expected_hash)

print("\n2. Checking the isolated numerical environment...")
probe = [
    sys.executable,
    "-c",
    (
        "import numpy, scipy, h5py, torch; "
        "assert numpy.__version__ == '1.26.4'; "
        "print('NumPy', numpy.__version__, '| SciPy', scipy.__version__, "
        "'| h5py', h5py.__version__, '| PyTorch', torch.__version__)"
    ),
]
child_env = os.environ.copy()
child_env["PYTHONPATH"] = str(PACKAGES)
child_env["PYTHONNOUSERSITE"] = "1"
child_env["PYTHONUNBUFFERED"] = "1"
child_env["MPLBACKEND"] = "Agg"
check = subprocess.run(probe, env=child_env, text=True, capture_output=True)
if check.returncode:
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
    check = subprocess.run(probe, env=child_env, text=True, capture_output=True)
if check.returncode:
    print(check.stdout)
    print(check.stderr)
    raise RuntimeError("Could not create the isolated WLD numerical environment")
print(check.stdout.strip())
print("PASS: isolated environment")

print("\n3. Compiling the pinned source...")
for filename in FILES:
    if filename.endswith(".py"):
        py_compile.compile(str(CODE / filename), doraise=True)
print("PASS: Python compilation")

required = {
    "Phase B prior manifest": PHASE_B / "priors" / "homo_sapiens_grch38" / "prior_manifest.json",
    "foundation checkpoint": CORPUS / "wld_corpus_pretrained_model.pt",
    "corpus report": CORPUS / "wld_corpus_pretraining_report.json",
    "v5.3 manifest": V53_BUNDLE / "wld_v53_ingestion_manifest.json",
    "v5.3 ATAC matrix": V53_BUNDLE / "atac_counts.GRCh38.2kb.npz",
    "v5.3 cell metadata": V53_BUNDLE / "cells.tsv.gz",
}
missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
if missing:
    raise FileNotFoundError(
        "Missing durable upstream WLD artifacts:\n" + "\n".join(missing)
    )
if not any(
    path.is_file()
    for path in [
        PRIOR_SOURCES / "omnipath_core_human.tsv",
        *PRIOR_SOURCES.glob("omnipath_core_human.tsv.*"),
        *PRIOR_SOURCES.glob("omnipath_webservice_interactions*.tsv.xz"),
    ]
):
    raise FileNotFoundError(
        f"No frozen OmniPath core interaction table under {PRIOR_SOURCES}"
    )
print("PASS: durable Phase B, corpus, v5.3 and interaction artifacts")

print("\n4. Starting restart-safe v5.4.1 development...")
print("   This trains only on the 73 training targets and selects on 16 validation targets.")
print("   The 16 test targets, muscle J/L and external studies remain sealed.")
print("   Rerun this same cell after a disconnect; completed epochs are retained.\n")

command = [
    sys.executable,
    "-u",
    str(CODE / "run_wld_v54_chromatin_colab.py"),
    "--phase-b-root",
    str(PHASE_B),
    "--corpus-root",
    str(CORPUS),
    "--v53-bundle",
    str(V53_BUNDLE),
    "--prior-sources",
    str(PRIOR_SOURCES),
    "--output-root",
    str(OUTPUT),
    "--epochs",
    "32",
    "--targets-per-epoch",
    "32",
    "--batch-size",
    "64",
    "--patience",
    "7",
    "--shuffle-replicates",
    "2",
    "--seed",
    "42",
]

with LOG.open("a", encoding="utf-8") as log_handle:
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
    print("FINAL 160 CHILD-PROCESS LOG LINES")
    print("=" * 78)
    tail = LOG.read_text(errors="replace").splitlines()[-160:]
    print("\n".join(tail))
    raise RuntimeError(
        f"WLD v5.4.1 exited with code {return_code}. "
        f"The complete log is at {LOG}; rerun this same cell to resume."
    )

report = OUTPUT / "development" / "wld_v54_chromatin_response_report.json"
if not report.is_file():
    raise RuntimeError(f"v5.4.1 exited without its final report: {report}")
print("\nVERIFIED COMPLETE: WLD V5.4.1")
print(f"Report: {report}")
print(f"Log:    {LOG}")
print("Test targets evaluated: False")
print("Muscle J/L evaluated:   False")
print("Attractor claim:        False")
