# Run this cell only after stopping the currently retrying OmniPath request.
# It reuses the existing GSE240061 export and large GRCh38/JASPAR downloads.

import csv
import hashlib
import json
import lzma
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path("/content/wld_real_data")
SOURCES = ROOT / "prior_sources"
EXPORT = ROOT / "gse240061_export"
SOURCES.mkdir(parents=True, exist_ok=True)

required_export = [
    "rna.mtx.gz",
    "atac.mtx.gz",
    "genes.tsv",
    "peaks.tsv",
    "barcodes.tsv",
    "metadata.tsv",
    "split.json",
]
missing = [name for name in required_export if not (EXPORT / name).exists()]
if missing:
    raise FileNotFoundError(
        f"Missing prior export files: {missing}. This must run in the same "
        "Colab runtime as the successful GSE240061 export."
    )


def run(command, *, check=True):
    command = list(map(str, command))
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, check=check)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_exact(url, path, min_bytes):
    path = pathlib.Path(path)
    if path.exists() and path.stat().st_size >= min_bytes:
        print(f"Already present: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    temporary = path.with_name(path.name + ".part")
    command = [
        "curl",
        "-fL",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "30",
        "--user-agent",
        "WLD-static-prior/1.0",
        "--continue-at",
        "-",
        "--output",
        temporary,
        url,
    ]
    result = run(command, check=False)
    if result.returncode:
        print("Resume was rejected; restarting only this small snapshot.", flush=True)
        temporary.unlink(missing_ok=True)
        restart = command.copy()
        resume_index = restart.index("--continue-at")
        del restart[resume_index : resume_index + 2]
        run(restart)
    if not temporary.exists() or temporary.stat().st_size < min_bytes:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {temporary}")
    temporary.replace(path)
    return path


def delimiter_from_header(header):
    return "\t" if header.count("\t") >= header.count(",") else ","


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def pick_column(columns, candidates):
    normalized = {column.strip().lower().lstrip("\ufeff"): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise RuntimeError(
        f"None of the required columns {candidates} were found. Columns: {columns}"
    )


print("1. Downloading immutable network snapshots (no live OmniPath API)...", flush=True)

COLLECTRI_URL = (
    "https://zenodo.org/records/8192729/files/"
    "CollecTRI_regulons.csv?download=1"
)
COLLECTRI_MD5 = "cee4a3943c059e6dd8796ced6dec44f6"
COLLECTRI_RAW = download_exact(
    COLLECTRI_URL,
    SOURCES / "CollecTRI_regulons_v2.0.csv",
    4_000_000,
)
actual_md5 = hashlib.md5(COLLECTRI_RAW.read_bytes()).hexdigest()
if actual_md5 != COLLECTRI_MD5:
    raise RuntimeError(
        f"CollecTRI checksum mismatch: expected {COLLECTRI_MD5}, got {actual_md5}"
    )
print("PASS: frozen CollecTRI v2.0 checksum", flush=True)

OMNIPATH_ARCHIVE_URL = (
    "https://archive.omnipathdb.org/"
    "omnipath_webservice_interactions__20230728-20250813.tsv.xz"
)
OMNIPATH_ARCHIVE = download_exact(
    OMNIPATH_ARCHIVE_URL,
    SOURCES / "omnipath_webservice_interactions__20230728-20250813.tsv.xz",
    20_000_000,
)
print(f"OmniPath archive SHA-256: {sha256(OMNIPATH_ARCHIVE)}", flush=True)


print("\n2. Materializing compiler-compatible signed CollecTRI...", flush=True)
COLLECTRI_OUT = SOURCES / "omnipath_collectri_human.tsv"
COLLECTRI_TMP = COLLECTRI_OUT.with_suffix(".tsv.tmp")

with COLLECTRI_RAW.open("r", encoding="utf-8-sig", newline="") as source:
    sample = source.read(16_384)
    source.seek(0)
    reader = csv.DictReader(source, delimiter=delimiter_from_header(sample.splitlines()[0]))
    columns = list(reader.fieldnames or ())
    source_column = pick_column(
        columns, ["source", "source_genesymbol", "tf", "regulator"]
    )
    target_column = pick_column(
        columns, ["target", "target_genesymbol", "gene"]
    )
    sign_column = pick_column(
        columns, ["mor", "weight", "sign", "mode_of_regulation"]
    )
    output_columns = [
        "source_genesymbol",
        "target_genesymbol",
        "is_directed",
        "is_stimulation",
        "is_inhibition",
        "consensus_direction",
        "consensus_stimulation",
        "consensus_inhibition",
        "sources",
        "references",
        "curation_effort",
    ]
    count = 0
    with COLLECTRI_TMP.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=output_columns, delimiter="\t")
        writer.writeheader()
        for row in reader:
            source_gene = (row.get(source_column) or "").strip()
            target_gene = (row.get(target_column) or "").strip()
            try:
                mode = float((row.get(sign_column) or "0").strip())
            except ValueError:
                continue
            if not source_gene or not target_gene or mode == 0:
                continue
            stimulation = int(mode > 0)
            inhibition = int(mode < 0)
            writer.writerow(
                {
                    "source_genesymbol": source_gene,
                    "target_genesymbol": target_gene,
                    "is_directed": 1,
                    "is_stimulation": stimulation,
                    "is_inhibition": inhibition,
                    "consensus_direction": 1,
                    "consensus_stimulation": stimulation,
                    "consensus_inhibition": inhibition,
                    "sources": "CollecTRI_regulons_v2.0",
                    "references": "10.1093/nar/gkad841",
                    "curation_effort": 1,
                }
            )
            count += 1

if count < 40_000:
    COLLECTRI_TMP.unlink(missing_ok=True)
    raise RuntimeError(f"Only {count} signed CollecTRI edges were converted.")
COLLECTRI_TMP.replace(COLLECTRI_OUT)
collectri_count = count
print(f"PASS: {count:,} signed CollecTRI TF-gene edges", flush=True)


print("\n3. Extracting the curated OmniPath signaling dataset...", flush=True)
OMNIPATH_OUT = SOURCES / "omnipath_core_human.tsv"
OMNIPATH_TMP = OMNIPATH_OUT.with_suffix(".tsv.tmp")
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

with lzma.open(OMNIPATH_ARCHIVE, "rt", encoding="utf-8-sig", newline="") as source:
    header = source.readline()
    if not header:
        raise RuntimeError("The OmniPath archive is empty.")
    delimiter = delimiter_from_header(header)
    source.seek(0)
    reader = csv.DictReader(source, delimiter=delimiter)
    columns = list(reader.fieldnames or ())
    normalized = {
        column.strip().lower().lstrip("\ufeff"): column for column in columns
    }
    source_symbol = normalized.get("source_genesymbol")
    target_symbol = normalized.get("target_genesymbol")
    if not source_symbol or not target_symbol:
        raise RuntimeError(
            "Archived OmniPath table lacks gene symbols. "
            f"Columns: {columns}"
        )
    membership_column = normalized.get("omnipath")
    datasets_column = normalized.get("datasets") or normalized.get("dataset")
    if not membership_column and not datasets_column:
        raise RuntimeError(
            "Archived OmniPath table lacks dataset membership columns. "
            f"Columns: {columns}"
        )

    count = 0
    with OMNIPATH_TMP.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in reader:
            if membership_column:
                keep = truthy(row.get(membership_column, ""))
            else:
                memberships = {
                    item.strip().lower()
                    for item in (row.get(datasets_column, "") or "")
                    .replace(",", ";")
                    .split(";")
                }
                keep = "omnipath" in memberships
            if keep:
                writer.writerow(row)
                count += 1

if count < 20_000:
    OMNIPATH_TMP.unlink(missing_ok=True)
    raise RuntimeError(f"Only {count} OmniPath signaling rows were extracted.")
OMNIPATH_TMP.replace(OMNIPATH_OUT)
omnipath_count = count
print(f"PASS: {count:,} curated OmniPath signaling interactions", flush=True)


source_manifest = {
    "collectri": {
        "release": "CollecTRI regulons v2.0 (Zenodo 8192729)",
        "url": COLLECTRI_URL,
        "md5": COLLECTRI_MD5,
        "converted_sha256": sha256(COLLECTRI_OUT),
        "signed_edges": collectri_count,
    },
    "omnipath": {
        "release": "OmniPath web-service archive 2023-07-28 through 2025-08-13",
        "url": OMNIPATH_ARCHIVE_URL,
        "archive_sha256": sha256(OMNIPATH_ARCHIVE),
        "extracted_sha256": sha256(OMNIPATH_OUT),
        "interaction_rows": omnipath_count,
    },
}
(SOURCES / "wld_network_source_manifest.json").write_text(
    json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
)


print("\n4. Resuming the existing biological-prior compiler...", flush=True)
RUNNER_COMMIT = "7259bca3248bf84ab147bb69a48a5bc23c9c6987"
RUNNER_SHA256 = "7ff102501a8b4389b7c50d09aacc654b8f5e76a5bf686500905f6d22d1ca2140"
RUNNER_URL = (
    "https://raw.githubusercontent.com/nlipieta/MultiscaleProject/"
    f"{RUNNER_COMMIT}/wld/colab_compile_wld_real_priors.py"
)
RUNNER = ROOT / "colab_compile_wld_real_priors.py"
run(
    [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-all-errors",
        "--output",
        RUNNER,
        RUNNER_URL,
    ]
)
actual_runner_hash = sha256(RUNNER)
if actual_runner_hash != RUNNER_SHA256:
    raise RuntimeError(
        f"Colab runner hash mismatch: expected {RUNNER_SHA256}, "
        f"got {actual_runner_hash}"
    )
print("PASS: corrected Colab runner hash verified", flush=True)

environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
result = subprocess.run([sys.executable, "-u", RUNNER], env=environment)
if result.returncode:
    raise RuntimeError(
        f"WLD prior compilation failed with exit code {result.returncode}. "
        f"Full compiler log: {ROOT / 'wld_prior_compilation.log'}"
    )

prior_manifest_path = ROOT / "gse240061_priors" / "prior_manifest.json"
prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
prior_manifest["network_source_releases"] = source_manifest
prior_manifest_path.write_text(
    json.dumps(prior_manifest, indent=2) + "\n", encoding="utf-8"
)

print("\nCOMPLETE: static-source repair and prior compilation passed.", flush=True)
print("No GSE240061 re-export or large reference re-download was performed.", flush=True)
