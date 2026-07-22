"""Pinned Colab launcher for the WLD v6.0 software and metadata contract."""

from google.colab import drive

drive.mount("/content/drive")

import hashlib
import json
import os
import py_compile
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "nlipieta/MultiscaleProject"
SOURCE_REF = "90a4812e77f95e9d53685d1877e50ae77e336ca1"
BACKUP = Path("/content/drive/MyDrive/WLD_Backup")
CODE = Path("/content/wld_v60_virtual_tissue_code")
PACKAGES = Path("/content/wld_v60_packages_py312_np1264")
OUTPUT = BACKUP / "wld_v60_virtual_tissue_contract"
LOG = OUTPUT / "wld_v60_complete.log"
REPORT = OUTPUT / "wld_v60_virtual_tissue_validation.json"

# SHA-256 values from the immutable SOURCE_REF.
FILES = {
    "wld_regulatory_twin_v60.py": "3b9c9bcfc189d5738d155782cd1662cf1b2bfea1f314f33470d9b0e0bac56cb3",
    "wld_axolotl_data_v60.py": "76e06c0e1b9c5d25b8e2bf5a8bf7973585b5e34a667df4b0b16655f8c34c7124",
    "wld_v60_axolotl_sources.json": "641c82879f126c61575f5f5ca771a10743cd3ec1fcb22e1aa2a47dd1e2e853ca",
    "run_wld_v60_virtual_tissue_smoke.py": "e1cce768570c9e9a61ab8a7da46aaa2fdbdd559a45795622fc3a1a6d3dab4431",
    "run_wld_v60_virtual_tissue_colab.py": "4d1aac5df0e9e57f54fa7e6a4d03bbf10cb5bd10ff8ff6f8eeafdf3dbad42f05",
    "wld_v60_virtual_tissue_contract.md": "7b3a3dd43a7f44926642d19560f1ed4d38974516c8b73975974852a9cdace8e9",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
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
    error = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "WLD-v6.0-virtual-tissue/1.0",
                    "Cache-Control": "no-cache",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            break
        except Exception as caught:
            error = caught
            if attempt < 5:
                time.sleep(min(2**attempt, 16))
    if payload is None:
        raise RuntimeError(f"Could not download pinned source: {name}") from error
    observed = sha256_bytes(payload)
    if observed != expected:
        raise RuntimeError(f"Source hash mismatch for {name}: {observed} != {expected}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    print(f"   PASS downloaded: {name}")


def probe(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import numpy; "
                "assert numpy.__version__=='1.26.4'; "
                "x=numpy.asarray([1.0]); assert bool(numpy.isfinite(x).all()); "
                "print('NumPy',numpy.__version__)"
            ),
        ],
        env=environment,
        text=True,
        capture_output=True,
    )


CODE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

print("WLD V6.0 ATLAS-CONDITIONED VIRTUAL-TISSUE TEST")
print("This run validates software architecture and public accession metadata only.")
print("No assay matrix is downloaded and no biological model is trained.")
print("GSE315993 remains sealed.\n")

if SOURCE_REF.startswith("__") or any(value.startswith("__") for value in FILES.values()):
    raise RuntimeError("This launcher has not been pinned to an immutable source commit")

print("1. Downloading and SHA-verifying the pinned v6.0 implementation...")
for filename, digest in FILES.items():
    fetch(filename, digest)

print("\n2. Preparing the isolated numerical environment...")
environment = os.environ.copy()
environment["PYTHONPATH"] = str(CODE) + os.pathsep + str(PACKAGES)
environment["PYTHONNOUSERSITE"] = "1"
environment["PYTHONUNBUFFERED"] = "1"
environment["OMP_NUM_THREADS"] = "2"
environment["MKL_NUM_THREADS"] = "2"

checked = probe(environment)
if checked.returncode:
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
        ],
        check=True,
    )
    checked = probe(environment)
if checked.returncode:
    print(checked.stdout)
    print(checked.stderr)
    raise RuntimeError(
        "Could not create a compatible NumPy environment. "
        "Use a standard Colab Python 3 runtime and rerun this same cell."
    )
print(checked.stdout.strip())
print("   PASS: compatible isolated environment")

print("\n3. Compiling the pinned implementation...")
for filename in FILES:
    if filename.endswith(".py"):
        py_compile.compile(str(CODE / filename), doraise=True)
print("   PASS: Python compilation")

print("\n4. Running the synthetic contract and live metadata-only audit...")
print("   Matrix and supplementary-value URLs are prohibited by the source auditor.")
command = [
    sys.executable,
    "-u",
    str(CODE / "run_wld_v60_virtual_tissue_colab.py"),
    "--registry",
    str(CODE / "wld_v60_axolotl_sources.json"),
    "--output-root",
    str(OUTPUT),
    "--live",
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
        f"WLD v6.0 exited with code {return_code}. Complete log: {LOG}. "
        "Rerun this same cell to resume."
    )

if not REPORT.is_file() or REPORT.stat().st_size == 0:
    raise FileNotFoundError(f"Completed runner did not write {REPORT}")
report = json.loads(REPORT.read_text())
claims = report.get("claims", {})
required_false = (
    "large_count_matrices_downloaded",
    "sealed_external_measurement_urls_downloaded",
    "gse315993_measurement_values_materialized",
    "test_measurement_values_materialized",
    "assay_values_downloaded",
    "sealed_values_fetched",
    "fresh_model_training",
    "model_trained",
    "digital_twin_claim",
    "attractor_claim",
    "biological_prediction_claim",
    "sealed_study_evaluated",
    "model_checkpoint_written",
)
bad = [name for name in required_false if claims.get(name) is not False]
if claims.get("metadata_only") is not True or bad:
    raise RuntimeError("Completed report crossed its claim boundary: " + ", ".join(bad))

print("\nVERIFIED COMPLETE")
print("Synthetic graph contract: True")
print("Metadata-only live audit requested: True")
print("Assay matrices downloaded: False")
print("GSE315993 values fetched: False")
print("Biological model trained: False")
print("Digital-twin claim: False")
print("Attractor claim: False")
print(f"Report: {REPORT}")
print(f"Log:    {LOG}")
