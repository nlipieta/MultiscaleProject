"""Export the real GSE240061 multiome object for the WLD temporal pipeline.

This runner is intentionally restartable.  It downloads the nested-gzip GEO
Seurat object, extracts only raw RNA/ATAC counts and experimental design fields,
validates the published six-subject design, and writes feature x cell Matrix
Market files.  Integrated assays, cell labels, clusters, pseudotime, and other
direct state proxies are never exported.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/wld_real_data")
EXPORT = ROOT / "gse240061_export"
STAGING = ROOT / "gse240061_export.partial"
OBJECT_GZ = ROOT / "GSE240061_integrated11723.rds.gz"
INNER_GZ = ROOT / "GSE240061_integrated11723.inner.rds.gz"
OBJECT_RDS = ROOT / "GSE240061_integrated11723.rds"
PCHIC = ROOT / "GSE126100_interactions.csv.gz"

GSE240061_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE240nnn/GSE240061/suppl/"
    "GSE240061_integrated11723.rds.gz"
)
GSE126100_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126100/suppl/"
    "GSE126100_interactions.csv.gz"
)

REQUIRED_EXPORT = {
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


def run(command, *, check=True, stdout=None, env=None):
    rendered = [str(item) for item in command]
    print("Running:", " ".join(rendered), flush=True)
    return subprocess.run(
        rendered,
        check=check,
        stdout=stdout,
        env=env,
    )


def complete_export(path: Path) -> bool:
    return all(
        (path / name).is_file() and (path / name).stat().st_size >= minimum
        for name, minimum in REQUIRED_EXPORT.items()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url: str, destination: Path, minimum: int) -> Path:
    if destination.is_file() and destination.stat().st_size >= minimum:
        print(
            f"Already present: {destination.name} "
            f"({destination.stat().st_size / 1e9:.3f} GB)",
            flush=True,
        )
        return destination
    partial = destination.with_name(destination.name + ".part")
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
        "WLD-GEO-export/1.0",
        "--continue-at",
        "-",
        "--output",
        partial,
        url,
    ]
    result = run(command, check=False)
    if result.returncode:
        print("The server rejected resume; restarting this download once.", flush=True)
        partial.unlink(missing_ok=True)
        restart = command.copy()
        resume_index = restart.index("--continue-at")
        del restart[resume_index : resume_index + 2]
        run(restart)
    if not partial.is_file() or partial.stat().st_size < minimum:
        raise RuntimeError(f"Downloaded file is missing or too small: {partial}")
    partial.replace(destination)
    return destination


def gzip_magic(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(2)


def inner_magic(path: Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read(2)


def install_r_dependencies() -> None:
    print("\n1. Installing binary R export dependencies...", flush=True)
    run(["apt-get", "update", "-qq"])
    run(
        [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "pigz",
            "r-base-core",
            "r-cran-matrix",
            "r-cran-jsonlite",
            "r-cran-seuratobject",
        ]
    )
    run(["Rscript", "--version"])


def expand_nested_object() -> None:
    if gzip_magic(OBJECT_GZ) != b"\x1f\x8b":
        raise RuntimeError(f"Outer GEO file is not gzip: {OBJECT_GZ}")
    if inner_magic(OBJECT_GZ) != b"\x1f\x8b":
        raise RuntimeError(
            "The downloaded GSE240061 file no longer has the expected nested-gzip format."
        )
    print("PASS: expected double-gzip GEO payload", flush=True)

    if not INNER_GZ.is_file() or INNER_GZ.stat().st_size < 2_000_000_000:
        temporary = INNER_GZ.with_name(INNER_GZ.name + ".part")
        with temporary.open("wb") as output:
            run(["pigz", "-dc", OBJECT_GZ], stdout=output)
        temporary.replace(INNER_GZ)
    if gzip_magic(INNER_GZ) != b"\x1f\x8b":
        raise RuntimeError("First decompression layer did not yield gzip data.")

    if not OBJECT_RDS.is_file() or OBJECT_RDS.stat().st_size < 5_000_000_000:
        temporary = OBJECT_RDS.with_name(OBJECT_RDS.name + ".part")
        with temporary.open("wb") as output:
            run(["pigz", "-dc", INNER_GZ], stdout=output)
        temporary.replace(OBJECT_RDS)
    with OBJECT_RDS.open("rb") as handle:
        if handle.read(2) != b"X\n":
            raise RuntimeError("Second decompression layer is not an R XDR serialization.")
    print(
        f"PASS: expanded R serialization ({OBJECT_RDS.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )


R_EXPORT_SOURCE = r'''
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(SeuratObject))

args <- commandArgs(trailingOnly = TRUE)
input_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Loading nested-compressed GSE240061 object...\n")
obj <- readRDS(input_path)

direct_slot <- function(value, name) {
  attrs <- attributes(value)
  if (!is.null(attrs) && name %in% names(attrs)) return(attrs[[name]])
  if (isS4(value) && name %in% methods::slotNames(value)) {
    return(methods::slot(value, name))
  }
  stop(sprintf("Object of class %s has no directly stored slot %s",
               paste(class(value), collapse = "/"), name))
}

assays <- direct_slot(obj, "assays")
if (!all(c("RNA", "ATAC") %in% names(assays))) {
  stop(sprintf("RNA/ATAC assays missing; found: %s", paste(names(assays), collapse = ", ")))
}
rna <- direct_slot(assays[["RNA"]], "counts")
atac <- direct_slot(assays[["ATAC"]], "counts")
meta <- direct_slot(obj, "meta.data")

if (!inherits(rna, "sparseMatrix") || !inherits(atac, "sparseMatrix")) {
  stop("Raw RNA and ATAC counts must be sparse matrices")
}
if (nrow(rna) != 36601L || nrow(atac) != 144663L || ncol(rna) != 37154L) {
  stop(sprintf("Unexpected dimensions: RNA %d x %d; ATAC %d x %d",
               nrow(rna), ncol(rna), nrow(atac), ncol(atac)))
}
if (!identical(colnames(rna), colnames(atac))) stop("RNA/ATAC barcodes differ")
barcodes <- colnames(rna)
if (is.null(rownames(meta)) || !all(barcodes %in% rownames(meta))) {
  stop("Object metadata does not align to matrix barcodes")
}
meta <- meta[barcodes, , drop = FALSE]

required_metadata <- c("Sample", "Time", "Group", "Sex")
missing_metadata <- setdiff(required_metadata, colnames(meta))
if (length(missing_metadata)) {
  stop(sprintf("Missing experimental metadata: %s", paste(missing_metadata, collapse = ", ")))
}

sample_name <- as.character(meta$Sample)
subject <- sub("_.*$", "", sample_name)
time_raw <- tolower(trimws(as.character(meta$Time)))
group_raw <- tolower(trimws(as.character(meta$Group)))
sex <- as.character(meta$Sex)
timepoint <- ifelse(time_raw == "pre", "pre",
                    ifelse(time_raw == "post", "post_3.5h", NA_character_))
condition <- ifelse(group_raw == "exercise", "exercise",
                    ifelse(group_raw == "control", "control", NA_character_))

if (anyNA(timepoint) || anyNA(condition)) stop("Unrecognized Time or Group values")
if (!setequal(unique(subject), c("E", "G", "I", "J", "L", "N"))) {
  stop(sprintf("Unexpected subjects: %s", paste(sort(unique(subject)), collapse = ", ")))
}

export_meta <- data.frame(
  cell_id = barcodes,
  subject = subject,
  condition = condition,
  timepoint = timepoint,
  sex = sex,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

actual_counts <- xtabs(~ subject + timepoint, export_meta)
expected_counts <- matrix(
  c(3759L, 2860L, 3107L, 1994L, 451L, 951L,
    5680L, 1911L, 4662L, 6911L, 354L, 4514L),
  nrow = 6L,
  ncol = 2L,
  dimnames = list(
    subject = c("E", "G", "I", "J", "L", "N"),
    timepoint = c("post_3.5h", "pre")
  )
)
if (!identical(dimnames(actual_counts), dimnames(expected_counts)) ||
    !identical(as.integer(actual_counts), as.integer(expected_counts))) {
  print(actual_counts)
  stop("Subject/time counts differ from the frozen GSE240061 design")
}

design <- unique(export_meta[, c("subject", "condition", "sex")])
design <- design[order(design$subject), ]
if (nrow(design) != 6L) stop("Subject condition/sex design is inconsistent")
expected_condition <- c(E="exercise", G="exercise", I="exercise", J="exercise",
                        L="control", N="control")
if (!all(design$condition == unname(expected_condition[design$subject]))) {
  stop("Subject condition labels differ from the frozen design")
}

genes <- rownames(rna)
peaks <- rownames(atac)
if (is.null(genes) || is.null(peaks) || anyDuplicated(genes) || anyDuplicated(peaks)) {
  stop("Gene and peak identifiers must be present and unique")
}

cat(sprintf("Loaded %d cells\n", length(barcodes)))
cat(sprintf("RNA raw counts: %d features x %d cells\n", nrow(rna), ncol(rna)))
cat(sprintf("ATAC raw counts: %d peaks x %d cells\n", nrow(atac), ncol(atac)))
cat("\nSubject/time/condition table:\n")
print(xtabs(~ subject + timepoint + condition, export_meta))
cat("\nSubject-level experimental design:\n")
print(design)

writeLines(genes, file.path(output_dir, "genes.tsv"), useBytes = TRUE)
writeLines(peaks, file.path(output_dir, "peaks.tsv"), useBytes = TRUE)
writeLines(barcodes, file.path(output_dir, "barcodes.tsv"), useBytes = TRUE)
write.table(export_meta, file.path(output_dir, "metadata.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE, col.names = TRUE)
write.table(design, file.path(output_dir, "subject_design.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE, col.names = TRUE)

threads <- max(1L, min(8L, parallel::detectCores(logical = TRUE)))
write_and_compress <- function(matrix, stem) {
  raw_path <- file.path(output_dir, paste0(stem, ".mtx"))
  cat(sprintf("Writing %s Matrix Market file...\n", toupper(stem)))
  Matrix::writeMM(matrix, raw_path)
  status <- system2("pigz", c("-f", "-p", as.character(threads), raw_path))
  if (status != 0L) stop(sprintf("pigz failed for %s", raw_path))
}
write_and_compress(rna, "rna")
rm(rna)
gc()
write_and_compress(atac, "atac")

cat("R EXPORT COMPLETE\n")
'''


def export_object() -> None:
    if complete_export(EXPORT):
        print("PASS: complete GSE240061 export already exists", flush=True)
        return
    if EXPORT.exists():
        raise RuntimeError(
            f"An incomplete export exists at {EXPORT}. Move it aside, then rerun."
        )
    if STAGING.exists():
        print("Removing an incomplete staging directory from the interrupted export.")
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    r_script = ROOT / "export_gse240061.R"
    r_script.write_text(R_EXPORT_SOURCE, encoding="utf-8")
    environment = os.environ.copy()
    environment["R_MAX_VSIZE"] = "45Gb"
    run(["Rscript", "--vanilla", r_script, OBJECT_RDS, STAGING], env=environment)

    split = {
        "train": ["E", "G", "N"],
        "validation": ["I"],
        "test": ["J", "L"],
    }
    (STAGING / "split.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset": "GSE240061",
        "contact_scaffold": "GSE126100",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_orientation": "features_x_cells",
        "dimensions": {
            "cells": 37154,
            "rna_features": 36601,
            "atac_peaks": 144663,
        },
        "experimental_fields": ["subject", "condition", "timepoint", "sex"],
        "excluded_state_proxies": [
            "integrated_assays",
            "cell_identity_annotations",
            "cluster_labels",
            "pseudotime",
        ],
        "split": split,
        "source": {
            "url": GSE240061_URL,
            "nested_gzip": True,
            "download_sha256": sha256(OBJECT_GZ),
        },
    }
    (STAGING / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not complete_export(STAGING):
        sizes = {
            name: (STAGING / name).stat().st_size if (STAGING / name).exists() else 0
            for name in REQUIRED_EXPORT
        }
        raise RuntimeError(f"Export validation failed; observed sizes: {sizes}")
    STAGING.replace(EXPORT)
    print("PASS: real GSE240061 RNA/ATAC export completed", flush=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    print("WLD REAL-DATA EXPORT", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    if complete_export(EXPORT):
        print("PASS: complete export already exists; nothing to rebuild", flush=True)
    else:
        install_r_dependencies()
        print("\n2. Downloading/resuming frozen GEO inputs...", flush=True)
        download_resumable(GSE240061_URL, OBJECT_GZ, 3_000_000_000)
        download_resumable(GSE126100_URL, PCHIC, 1_000)
        print("\n3. Expanding the nested-gzip Seurat object...", flush=True)
        expand_nested_object()
        print("\n4. Exporting raw RNA/ATAC counts and experimental metadata...", flush=True)
        export_object()

    print("\nExported files:", flush=True)
    for name in REQUIRED_EXPORT:
        path = EXPORT / name
        print(f"  {name:24s} {path.stat().st_size / 1e6:10.2f} MB", flush=True)
    print("PASS: integrated assays and identity labels excluded from model inputs")
    print("PASS: E/G/N train, I validation, and J/L sealed-test split frozen")
    print(f"Export directory: {EXPORT}")


if __name__ == "__main__":
    main()
