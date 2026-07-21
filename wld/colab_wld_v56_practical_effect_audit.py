"""Pinned Colab cell for auditing a completed WLD v5.6 report without retraining."""

from google.colab import drive

drive.mount("/content/drive")

import hashlib
import py_compile
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "nlipieta/MultiscaleProject"
SOURCE_REF = "4e481dd52bcdd2ef817cc03d87aea03e081d5ee6"
CODE = Path("/content/wld_v56_practical_audit_code")
SOURCE_REPORT = Path(
    "/content/drive/MyDrive/WLD_Backup/wld_v56_nullaware_r2/development/"
    "wld_v56_null_aware_development_report.json"
)
OUTPUT = Path(
    "/content/drive/MyDrive/WLD_Backup/wld_v56_nullaware_r2/"
    "practical_effect_audit"
)
FILES = {
    "wld_v56_practical_effect_audit.py": (
        "01f20e2cd21d7257c7b5bf910d7acf2bf881e9093e1ea73a3f7871d472d6dad4"
    ),
    "run_wld_v56_practical_effect_smoke.py": (
        "dd939ae0fff2c3f7bc72045fbcf09009b04f8347d78e7489f3cf51258d99b80f"
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch(name: str, expected: str) -> None:
    destination = CODE / name
    if destination.is_file() and sha256_bytes(destination.read_bytes()) == expected:
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
                headers={"User-Agent": "WLD-v5.6-practical-audit/1.0"},
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


CODE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

print("WLD V5.6 NO-RETRAINING PRACTICAL-EFFECT AUDIT")
print("This cell reads the completed development JSON only.")
print("No checkpoint, raw cell, test target, or external study is opened.\n")

if not SOURCE_REPORT.is_file() or SOURCE_REPORT.stat().st_size == 0:
    raise FileNotFoundError(
        f"Missing completed v5.6 report: {SOURCE_REPORT}\n"
        "Restore WLD_Backup from Google Drive or finish the v5.6 cell first."
    )

print("1. Downloading and verifying the pinned audit implementation...")
for filename, digest in FILES.items():
    fetch(filename, digest)

print("\n2. Compiling and testing the numerical-dust regression contract...")
for filename in FILES:
    py_compile.compile(str(CODE / filename), doraise=True)
environment = dict(**__import__("os").environ)
environment["PYTHONPATH"] = str(CODE)
environment["PYTHONNOUSERSITE"] = "1"
subprocess.run(
    [sys.executable, str(CODE / "run_wld_v56_practical_effect_smoke.py")],
    check=True,
    env=environment,
)

print("\n3. Auditing the completed Drive report...")
subprocess.run(
    [
        sys.executable,
        str(CODE / "wld_v56_practical_effect_audit.py"),
        "--report",
        str(SOURCE_REPORT),
        "--output-root",
        str(OUTPUT),
    ],
    check=True,
    env=environment,
)

print("\nVERIFIED COMPLETE")
print(f"Audit: {OUTPUT / 'wld_v56_practical_effect_audit.json'}")
print(f"Table: {OUTPUT / 'wld_v56_target_effects.tsv'}")
print("The source report is unchanged and the sealed test remains closed.")
