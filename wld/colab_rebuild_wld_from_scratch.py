"""Restart-safe full rebuild of the real-data WLD validation pipeline.

Run through the small bootstrap cell documented in the repository.  Google
Drive is used as an artifact cache; every completed stage is copied there and
is restored automatically after a Colab runtime reset.  Test subjects J/L stay
sealed throughout this development and circuit-reliance audit.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path("/content/wld_real_data")
DRIVE = Path("/content/drive/MyDrive/WLD_Backup/wld_real_data")
CODE = ROOT / "rebuild_code"
SOURCE_REF = os.environ.get("WLD_SOURCE_REF", "agent/add-wld-attractor-model")
REPOSITORY = "nlipieta/MultiscaleProject"

EXPORT = ROOT / "gse240061_export"
PRIORS = ROOT / "gse240061_priors"
COHORT = ROOT / "gse240061_temporal_cohort"
DEVELOPMENT = ROOT / "gse240061_temporal_development_v2_seed42"
AUDIT = ROOT / "gse240061_validation_reliance_seed42"

RAW_OBJECT = "GSE240061_integrated11723.rds.gz"
PCHIC = "GSE126100_interactions.csv.gz"
RAW_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE240nnn/GSE240061/suppl/"
    + RAW_OBJECT
)
PCHIC_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126100/suppl/"
    + PCHIC
)

EXPORT_REQUIREMENTS = {
    "rna.mtx.gz": 100_000_000,
    "atac.mtx.gz": 500_000_000,
    "genes.tsv": 100_000,
    "peaks.tsv": 1_000_000,
    "barcodes.tsv": 100_000,
    "metadata.tsv": 500_000,
    "subject_design.tsv": 100,
    "split.json": 20,
    "export_manifest.json": 100,
}
PRIOR_REQUIREMENTS = {
    "peak_gene_links.tsv": 1,
    "motif_hits.tsv": 1,
    "tf_gene_edges.tsv": 1,
    "signaling_edges.tsv": 1,
    "prior_manifest.json": 100,
}

PRIOR_SOURCE_DOWNLOADS = {
    "gencode.v44.primary_assembly.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/"
        "gencode.v44.primary_assembly.annotation.gtf.gz",
        30_000_000,
    ),
    "GRCh38.primary_assembly.genome.fa.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/"
        "GRCh38.primary_assembly.genome.fa.gz",
        700_000_000,
    ),
    "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt": (
        "https://jaspar.elixir.no/download/data/2024/CORE/"
        "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
        400_000,
    ),
    "CollecTRI_regulons_v2.0.csv": (
        "https://zenodo.org/records/8192729/files/CollecTRI_regulons.csv?download=1",
        4_000_000,
    ),
    "omnipath_webservice_interactions__20230728-20250813.tsv.xz": (
        "https://archive.omnipathdb.org/"
        "omnipath_webservice_interactions__20230728-20250813.tsv.xz",
        20_000_000,
    ),
}


def run(command, *, env=None) -> None:
    rendered = [str(item) for item in command]
    print("Running:", " ".join(rendered), flush=True)
    subprocess.run(rendered, check=True, env=env)


def files_complete(directory: Path, requirements: dict[str, int]) -> bool:
    return all(
        (directory / name).is_file()
        and (directory / name).stat().st_size >= minimum
        for name, minimum in requirements.items()
    )


def valid_development() -> bool:
    report = DEVELOPMENT / "wld_temporal_development.json"
    if not report.is_file() or report.stat().st_size < 100:
        return False
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("test_groups_evaluated") is not False:
            return False
        if set(payload.get("conditions", {})) != {
            "true_circuit",
            "no_circuit",
            "sign_shuffled_circuit",
        }:
            return False
        return all(
            (DEVELOPMENT / str(record["checkpoint"])).is_file()
            for record in payload["conditions"].values()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def valid_audit() -> bool:
    report = AUDIT / "wld_validation_reliance.json"
    if not report.is_file() or report.stat().st_size < 100:
        return False
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        return payload.get("test_groups_evaluated") is False
    except (OSError, ValueError, TypeError):
        return False


def copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    temporary = destination.with_name(destination.name + ".part")
    copied = 0
    next_report = 256 * 1024 * 1024
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        while True:
            block = input_handle.read(16 * 1024 * 1024)
            if not block:
                break
            output_handle.write(block)
            copied += len(block)
            if copied >= next_report:
                print(
                    f"   copied {source.name}: {copied / 1e9:.2f}/"
                    f"{source.stat().st_size / 1e9:.2f} GB",
                    flush=True,
                )
                next_report += 256 * 1024 * 1024
    shutil.copystat(source, temporary)
    temporary.replace(destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.name.endswith((".part", ".tmp")):
            copy_file_atomic(path, destination / path.relative_to(source))


def restore_from_drive() -> None:
    print("\n1. Restoring any completed or resumable artifacts from Google Drive...", flush=True)
    for name in (
        "gse240061_export",
        "gse240061_priors",
        "prior_sources",
        "gse240061_temporal_cohort",
        "gse240061_temporal_development_v2_seed42",
        "gse240061_validation_reliance_seed42",
    ):
        copy_tree(DRIVE / name, ROOT / name)
    for name in (RAW_OBJECT, PCHIC):
        source = DRIVE / name
        if source.is_file():
            copy_file_atomic(source, ROOT / name)
    print("PASS: Drive restore scan complete", flush=True)


def backup_tree(name: str) -> None:
    print(f"Backing up {name} to Google Drive...", flush=True)
    copy_tree(ROOT / name, DRIVE / name)
    print(f"PASS: Drive backup complete for {name}", flush=True)


def download_resumable(url: str, destination: Path, minimum: int) -> Path:
    if destination.is_file() and destination.stat().st_size >= minimum:
        print(f"Already present: {destination} ({destination.stat().st_size / 1e9:.2f} GB)")
        return destination
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-fL",
        "--retry",
        "12",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--user-agent",
        "WLD-rebuild/1.0",
        "--continue-at",
        "-",
        "--output",
        partial,
        url,
    ]
    result = subprocess.run(command)
    if result.returncode:
        partial.unlink(missing_ok=True)
        restart = command.copy()
        index = restart.index("--continue-at")
        del restart[index : index + 2]
        run(restart)
    if not partial.is_file() or partial.stat().st_size < minimum:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {partial}")
    partial.replace(destination)
    return destination


def download_repo_runner(name: str) -> Path:
    CODE.mkdir(parents=True, exist_ok=True)
    destination = CODE / name
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_REF}/wld/{name}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "WLD-rebuild/1.0"})
    error = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if len(payload) < 1_000:
                raise RuntimeError(f"Downloaded runner is too small: {name}")
            destination.write_bytes(payload)
            error = None
            break
        except Exception as exc:
            error = exc
            print(f"   GitHub attempt {attempt}/5 failed for {name}: {exc}", flush=True)
    if error is not None:
        raise RuntimeError(f"Could not download {name}: {error}")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"   downloaded: {name} ({digest[:12]}...)", flush=True)
    return destination


def prepare_raw_source_cache() -> None:
    if files_complete(EXPORT, EXPORT_REQUIREMENTS):
        return
    print("\n2. Caching the 3.5 GB GEO object in Drive (resumable across restarts)...")
    drive_raw = download_resumable(RAW_URL, DRIVE / RAW_OBJECT, 3_000_000_000)
    copy_file_atomic(drive_raw, ROOT / RAW_OBJECT)
    drive_pchic = download_resumable(PCHIC_URL, DRIVE / PCHIC, 1_000)
    copy_file_atomic(drive_pchic, ROOT / PCHIC)
    print("PASS: raw source cache is durable", flush=True)


def run_export() -> None:
    if files_complete(EXPORT, EXPORT_REQUIREMENTS):
        print("\n3. PASS: GSE240061 export restored; skipping export", flush=True)
        return
    print("\n3. Rebuilding the raw RNA/ATAC export...", flush=True)
    runner = download_repo_runner("colab_export_wld_real_data.py")
    run([sys.executable, "-u", runner])
    if not files_complete(EXPORT, EXPORT_REQUIREMENTS):
        raise RuntimeError("Export runner ended without a complete validated export.")
    backup_tree("gse240061_export")
    copy_file_atomic(ROOT / PCHIC, DRIVE / PCHIC)


def prefetch_prior_sources() -> None:
    if files_complete(PRIORS, PRIOR_REQUIREMENTS):
        return
    print("\n4. Caching frozen prior sources in Drive...", flush=True)
    local_sources = ROOT / "prior_sources"
    drive_sources = DRIVE / "prior_sources"
    for name, (url, minimum) in PRIOR_SOURCE_DOWNLOADS.items():
        drive_file = download_resumable(url, drive_sources / name, minimum)
        copy_file_atomic(drive_file, local_sources / name)
    print("PASS: large frozen source downloads are durable", flush=True)


def run_priors() -> None:
    if files_complete(PRIORS, PRIOR_REQUIREMENTS):
        print("\n5. PASS: compiled biological priors restored; skipping compilation")
        return
    print("\n5. Compiling contact x motif x signed-regulation x signaling priors...")
    runner = download_repo_runner("colab_repair_omnipath_and_resume.py")
    run([sys.executable, "-u", runner])
    if not files_complete(PRIORS, PRIOR_REQUIREMENTS):
        raise RuntimeError("Prior compiler ended without complete prior tables.")
    backup_tree("gse240061_priors")
    backup_tree("prior_sources")


def run_development() -> None:
    if valid_development() and (COHORT / "manifest.json").is_file():
        print("\n6. PASS: validation-only development restored; skipping training")
        return
    print("\n6. Building the cohort and training matched circuit controls...")
    runner = download_repo_runner("colab_build_train_wld_real.py")
    run([sys.executable, "-u", runner])
    if not valid_development():
        raise RuntimeError("Temporal training ended without a valid sealed-test report.")
    backup_tree("gse240061_temporal_cohort")
    backup_tree("gse240061_temporal_development_v2_seed42")


def run_audit() -> None:
    if valid_audit():
        print("\n7. PASS: convergence/reliance audit restored; skipping audit")
        return
    print("\n7. Running validation-only convergence and frozen circuit-reliance audit...")
    print("J/L remain sealed.", flush=True)
    runner = download_repo_runner("run_wld_validation_reliance.py")
    run([sys.executable, "-u", runner])
    if not valid_audit():
        raise RuntimeError("Audit ended without a valid sealed-test report.")
    backup_tree("gse240061_validation_reliance_seed42")


def main() -> None:
    print("WLD RESTART-SAFE FULL REBUILD", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    print(f"Repository ref: {SOURCE_REF}", flush=True)
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive is not mounted. Run drive.mount('/content/drive') first."
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    DRIVE.mkdir(parents=True, exist_ok=True)

    restore_from_drive()
    prepare_raw_source_cache()
    run_export()
    prefetch_prior_sources()
    run_priors()
    run_development()
    run_audit()

    print("\n" + "=" * 72)
    print("COMPLETE: WLD validation pipeline rebuilt and backed up to Drive")
    print("J/L remain sealed; no held-out test metrics were computed.")
    print(f"Durable backup: {DRIVE}")
    print(f"Audit report: {AUDIT / 'wld_validation_reliance.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
