"""Pinned one-cell launcher for the WLD v6.1 axolotl measurement corpus.

Run this file inside Google Colab. It mounts Drive, installs an isolated
NumPy/SciPy runtime, verifies every source file against an immutable Git commit,
runs the adversarial synthetic contract, and then builds or resumes the real
development corpus. GSE315993 remains sealed and no model is trained.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


SOURCE_REF = "62913095ffd42d4abae53e29bf1fdcee07733056"
REPOSITORY = "nlipieta/MultiscaleProject"
SOURCE_HASHES = {
    "wld_axolotl_corpus_v61.py": "7b5174aa089fd3294645fc2062d3117ff31e932edea829c18ba87a0374c06cc0",
    "wld_v61_axolotl_measurement_sources.json": "bf134a6187e98af4c707044f354dbdea885bf3eeb7ecd3a5a3cefe668c010668",
    "run_wld_v61_axolotl_corpus_smoke.py": "3aa024bc624b9cd263800a7e2b750ce0b5a11e57cfdb48d362fd7583ba2d1400",
    "run_wld_v61_axolotl_corpus_colab.py": "783ab892d4524431d0bef37f5c44240fc390fccc13620740d699bfd9870785dc",
    "wld_v61_axolotl_corpus_contract.md": "67650389f5086edaf10f5a3a79648926d76edf56ffa8e15ebf19836f7d1d919f",
}
CODE_ROOT = Path("/content/wld_v61_code")
PACKAGE_ROOT = Path("/content/wld_v61_packages")
OUTPUT_ROOT = Path("/content/drive/MyDrive/WLD_Backup/wld_v61_axolotl_corpus")
LOG_PATH = OUTPUT_ROOT / "wld_v61_complete.log"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pinned(name: str, expected: str) -> None:
    destination = CODE_ROOT / name
    if destination.is_file() and sha256_file(destination) == expected:
        print(f"   PASS cached: {name}", flush=True)
        return
    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_REF}/wld/{name}"
    last_error = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "WLD-v6.1-pinned-colab-launcher/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            observed = sha256_file(temporary)
            if observed != expected:
                raise RuntimeError(
                    f"SHA-256 mismatch for {name}: expected {expected}, observed {observed}"
                )
            os.replace(temporary, destination)
            print(f"   PASS downloaded: {name}", flush=True)
            return
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt == 5:
                break
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"Unable to fetch pinned source {name}") from last_error


def isolated_environment_ok() -> bool:
    if not PACKAGE_ROOT.is_dir():
        return False
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import numpy, scipy; "
                "assert numpy.__version__ == '1.26.4'; "
                "assert scipy.__version__ == '1.16.3'; "
                "from scipy import sparse; assert sparse.csr_matrix((1,1)).shape == (1,1)"
            ),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return probe.returncode == 0


def install_isolated_environment() -> None:
    if isolated_environment_ok():
        print("   PASS: compatible isolated packages already present", flush=True)
        return
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    PACKAGE_ROOT.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--target",
            str(PACKAGE_ROOT),
            "numpy==1.26.4",
            "scipy==1.16.3",
        ],
        check=True,
    )
    if not isolated_environment_ok():
        raise RuntimeError("The isolated NumPy/SciPy environment failed validation")
    print("   PASS: isolated NumPy 1.26.4 / SciPy 1.16.3", flush=True)


def run_logged(command: list[str], environment: dict[str, str]) -> None:
    print("\nRunning: " + " ".join(command), flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 78 + "\n")
        log.write("Running: " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=CODE_ROOT,
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
        raise RuntimeError(
            f"Command exited with code {return_code}. Complete log: {LOG_PATH}"
        )


print("WLD V6.1 PINNED AXOLOTL MEASUREMENT-CORPUS LAUNCHER", flush=True)
print(f"Repository commit: {SOURCE_REF}", flush=True)
print("GSE315993 remains sealed. This stage does not train a model.\n", flush=True)

try:
    from google.colab import drive
except ImportError as error:
    raise RuntimeError("This launcher must run in Google Colab") from error

drive.mount("/content/drive")
if not Path("/content/drive/MyDrive").is_dir() or not os.path.ismount("/content/drive"):
    raise RuntimeError("Google Drive did not mount at /content/drive")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CODE_ROOT.mkdir(parents=True, exist_ok=True)
free_drive = shutil.disk_usage(OUTPUT_ROOT).free
free_local = shutil.disk_usage("/content").free
print(f"Drive free: {free_drive / 1024**3:.1f} GB", flush=True)
print(f"Local free: {free_local / 1024**3:.1f} GB", flush=True)
if free_drive < 3 * 1024**3 or free_local < 3 * 1024**3:
    raise RuntimeError("At least 3 GB free space is required on both Drive and /content")

print("1. Downloading and SHA-verifying immutable v6.1 sources...", flush=True)
for source_name, source_hash in SOURCE_HASHES.items():
    download_pinned(source_name, source_hash)

print("\n2. Preparing a kernel-independent numerical environment...", flush=True)
install_isolated_environment()

run_env = os.environ.copy()
run_env["PYTHONPATH"] = os.pathsep.join((str(PACKAGE_ROOT), str(CODE_ROOT)))
run_env["PYTHONNOUSERSITE"] = "1"
run_env["PYTHONDONTWRITEBYTECODE"] = "1"
run_env["PYTHONUNBUFFERED"] = "1"
run_env["WLD_V61_OUTPUT_ROOT"] = str(OUTPUT_ROOT)

print("\n3. Running adversarial architecture, leakage, seal, restart and integrity tests...", flush=True)
run_logged([sys.executable, str(CODE_ROOT / "run_wld_v61_axolotl_corpus_smoke.py")], run_env)

print("\n4. Building/resuming the real unsealed axolotl development corpus...", flush=True)
runner_command = [
    sys.executable,
    str(CODE_ROOT / "run_wld_v61_axolotl_corpus_colab.py"),
    "--output-root",
    str(OUTPUT_ROOT),
]
if os.environ.get("WLD_V61_LAUNCHER_DRY_RUN") == "1":
    runner_command.append("--dry-run")
run_logged(runner_command, run_env)

print("\n" + "=" * 78, flush=True)
print("VERIFIED COMPLETE: WLD V6.1 REAL AXOLOTL MEASUREMENT CORPUS", flush=True)
print("=" * 78, flush=True)
print(f"Durable corpus: {OUTPUT_ROOT}", flush=True)
print(f"Complete log:   {LOG_PATH}", flush=True)
print("GSE315993 values materialized: False", flush=True)
print("Biological model trained:      False", flush=True)
print("Digital-twin claim:            False", flush=True)
print("Attractor claim:               False", flush=True)
