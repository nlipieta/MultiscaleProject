"""Auditable sparse-data layer for WLD v4 Phase A pretraining.

The data layer preserves raw counts, modality-specific barcodes and features.
It never manufactures cell pairing.  Study/donor/label metadata are stored for
partitioning and auditing, not returned as encoder tensors.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy import sparse
from scipy.io import mmread


CORE_MODALITIES = {"rna", "atac", "protein", "metabolic"}
FORBIDDEN_ENCODER_METADATA = {
    "barcode", "cell_type", "celltype", "cluster", "donor_id", "label",
    "leiden", "louvain", "pseudotime", "study_id", "subject_id", "umap",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def open_text(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open("rt", newline="")


def read_rows(path: Path) -> List[List[str]]:
    with open_text(path) as handle:
        return [line.rstrip("\n\r").split("\t") for line in handle if line.strip()]


def read_barcodes(path: Path) -> List[str]:
    return [row[0] for row in read_rows(path)]


def _decode(values: np.ndarray) -> List[str]:
    return [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]


def _canonical_modality(value: str, feature_name: str) -> str:
    token = value.strip().lower()
    if token in {"gene expression", "rna", "gex"}:
        return "rna"
    if token in {"peaks", "atac", "chromatin accessibility"}:
        return "atac"
    if token in {"antibody capture", "adt", "protein"}:
        return "protein"
    if re.match(r"^(chr|[0-9xy]+[:_-])", feature_name, flags=re.I):
        return "atac"
    return "rna"


def parse_features(path: Path, default_modality: str = "") -> List[Dict[str, str]]:
    features = []
    for index, row in enumerate(read_rows(path)):
        if len(row) >= 3 and row[0].lower().startswith("chr") and row[1].isdigit() and row[2].isdigit():
            identifier = f"{row[0]}:{row[1]}-{row[2]}"
            name = identifier
            supplied_type = "Peaks"
        else:
            identifier = row[0]
            name = row[1] if len(row) > 1 and row[1] else row[0]
            supplied_type = row[2] if len(row) > 2 else default_modality
        modality = default_modality or _canonical_modality(supplied_type, name)
        features.append({
            "feature_id": identifier,
            "feature_name": name,
            "modality": modality,
            "source_index": str(index),
        })
    return features


def write_lines_gz(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, newline="") as handle:
                for value in values:
                    handle.write(str(value) + "\n")


def write_features_gz(path: Path, features: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["feature_id", "feature_name", "modality", "source_index"])
                for value in features:
                    writer.writerow([value["feature_id"], value["feature_name"], value["modality"], value["source_index"]])


def read_bundle_features(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@dataclass
class ModalityBlock:
    matrix: sparse.csr_matrix
    barcodes: List[str]
    features: List[Dict[str, str]]

    def validate(self) -> None:
        if self.matrix.shape != (len(self.barcodes), len(self.features)):
            raise ValueError("Matrix, barcode and feature dimensions disagree")
        if len(set(self.barcodes)) != len(self.barcodes):
            raise ValueError("Duplicate barcodes within a modality")
        if self.matrix.data.size and (not np.isfinite(self.matrix.data).all() or (self.matrix.data < 0).any()):
            raise ValueError("Raw abundance/count matrices must be finite and non-negative")


def split_modalities(matrix: sparse.spmatrix, barcodes: List[str], features: List[Dict[str, str]]) -> Dict[str, ModalityBlock]:
    matrix = matrix.tocsr()
    result = {}
    for modality in sorted({value["modality"] for value in features}):
        if modality not in CORE_MODALITIES:
            continue
        indices = [index for index, value in enumerate(features) if value["modality"] == modality]
        selected = [features[index] for index in indices]
        for new_index, value in enumerate(selected):
            value = dict(value)
            value["source_index"] = str(new_index)
            selected[new_index] = value
        block = ModalityBlock(matrix[:, indices].tocsr(), list(barcodes), selected)
        block.validate()
        result[modality] = block
    if not result:
        raise ValueError("No supported modalities found")
    return result


def read_10x_h5(path: Path) -> Dict[str, ModalityBlock]:
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        feature_group = group["features"]
        ids = _decode(feature_group["id"][:])
        names = _decode(feature_group["name"][:])
        type_key = "feature_type" if "feature_type" in feature_group else "feature_types"
        types = _decode(feature_group[type_key][:])
        barcodes = _decode(group["barcodes"][:])
        shape = tuple(int(value) for value in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        ).transpose().tocsr()
    features = [
        {"feature_id": identifier, "feature_name": name, "modality": _canonical_modality(kind, name), "source_index": str(index)}
        for index, (identifier, name, kind) in enumerate(zip(ids, names, types))
    ]
    return split_modalities(matrix, barcodes, features)


def read_10x_mtx(matrix_path: Path, barcode_path: Path, feature_path: Path) -> Dict[str, ModalityBlock]:
    with gzip.open(matrix_path, "rb") if matrix_path.suffix == ".gz" else matrix_path.open("rb") as handle:
        matrix = mmread(handle).tocsr().transpose().tocsr()
    barcodes = read_barcodes(barcode_path)
    features = parse_features(feature_path)
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError(f"10x dimensions disagree: matrix {matrix.shape}, barcodes {len(barcodes)}, features {len(features)}")
    return split_modalities(matrix, barcodes, features)


def read_single_mtx(matrix_path: Path, barcode_path: Path, feature_path: Path, modality: str) -> ModalityBlock:
    with gzip.open(matrix_path, "rb") if matrix_path.suffix == ".gz" else matrix_path.open("rb") as handle:
        matrix = mmread(handle).tocsr().transpose().tocsr()
    block = ModalityBlock(matrix, read_barcodes(barcode_path), parse_features(feature_path, modality))
    block.validate()
    return block


def read_adt_csv(path: Path, expected_barcodes: Sequence[str]) -> ModalityBlock:
    expected_lookup = {_barcode_key(value): value for value in expected_barcodes}
    if len(expected_lookup) != len(expected_barcodes):
        raise ValueError("Expected 10x barcodes are not unique after normalization")
    with open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("ADT CSV is empty") from error
        if len(header) < 2:
            raise ValueError("ADT CSV is malformed")

        # ADT count tables are commonly cells x antibodies (tens of columns),
        # but some tools export antibodies x cells.  Barcode overlap alone is
        # insufficient because sample prefixes are often prepended.  Infer the
        # orientation from the panel width, then require actual normalized
        # barcode matches before accepting the table.
        if len(header) - 1 <= 5000:
            names = header[1:]
            matched: Dict[str, np.ndarray] = {}
            for row in reader:
                if not row:
                    continue
                key = _barcode_key(row[0])
                if key not in expected_lookup:
                    continue
                if len(row) != len(header):
                    raise ValueError("ADT row width does not match its header")
                if key in matched:
                    raise ValueError(f"Duplicate ADT barcode after normalization: {key}")
                matched[key] = np.asarray([float(value or 0) for value in row[1:]], dtype=np.float32)
            keys = [_barcode_key(value) for value in expected_barcodes if _barcode_key(value) in matched]
            barcodes = [expected_lookup[key] for key in keys]
            values = np.stack([matched[key] for key in keys]) if keys else np.empty((0, len(names)), dtype=np.float32)
        else:
            source_barcodes = header[1:]
            selected_columns = []
            selected_keys = []
            for column, barcode in enumerate(source_barcodes, start=1):
                key = _barcode_key(barcode)
                if key in expected_lookup:
                    selected_columns.append(column)
                    selected_keys.append(key)
            names = []
            feature_rows = []
            for row in reader:
                if not row:
                    continue
                names.append(row[0])
                feature_rows.append([float(row[column] or 0) for column in selected_columns])
            barcodes = [expected_lookup[key] for key in selected_keys]
            values = np.asarray(feature_rows, dtype=np.float32).T
    if not barcodes:
        raise ValueError("No ADT barcodes matched the filtered RNA/ATAC cells")
    if len(names) > 5000:
        raise ValueError(
            f"Implausible ADT feature count ({len(names)}); orientation is unresolved"
        )
    features = [
        {"feature_id": name, "feature_name": name, "modality": "protein", "source_index": str(index)}
        for index, name in enumerate(names)
    ]
    block = ModalityBlock(sparse.csr_matrix(values), barcodes, features)
    block.validate()
    return block


def _barcode_key(value: str) -> str:
    value = str(value).strip().strip('"').strip("'")
    matches = re.findall(r"[ACGTN]{12,}(?:-\d+)?", value.upper())
    if matches:
        return re.sub(r"-\d+$", "", matches[-1])
    return value


def pairing_report(blocks: Mapping[str, ModalityBlock]) -> Dict[str, object]:
    modalities = sorted(blocks)
    pairs = {}
    exact_all = True
    for index, left in enumerate(modalities):
        for right in modalities[index + 1:]:
            a, b = blocks[left].barcodes, blocks[right].barcodes
            intersection = len(set(a).intersection(b))
            exact = a == b
            exact_all = exact_all and exact
            pairs[f"{left}:{right}"] = {
                "exact_same_order": exact,
                "intersection": intersection,
                "left_cells": len(a),
                "right_cells": len(b),
            }
    return {"pairing": "exact" if exact_all else "unpaired_population", "pairs": pairs, "fabricated_pairs": False}


def save_bundle(root: Path, cohort: Mapping[str, object], blocks: Mapping[str, ModalityBlock], source_lock: Mapping[str, object]) -> Dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_blocks = {}
    for modality, block in blocks.items():
        block.validate()
        modality_root = root / "modalities" / modality
        modality_root.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(modality_root / "counts.csr.npz", block.matrix, compressed=True)
        write_lines_gz(modality_root / "barcodes.tsv.gz", block.barcodes)
        write_features_gz(modality_root / "features.tsv.gz", block.features)
        manifest_blocks[modality] = {
            "shape": list(block.matrix.shape),
            "nnz": int(block.matrix.nnz),
            "counts_sha256": sha256_file(modality_root / "counts.csr.npz"),
            "barcodes_sha256": sha256_file(modality_root / "barcodes.tsv.gz"),
            "features_sha256": sha256_file(modality_root / "features.tsv.gz"),
            "normalized": False,
        }
    manifest = {
        "schema_version": "1.0",
        "cohort_id": cohort["cohort_id"],
        "study_id": cohort["study_id"],
        "species": cohort["species"],
        "genome_build": cohort["genome_build"],
        "adapter": cohort["adapter"],
        "donor_scope": cohort.get("donor_scope", ""),
        "modalities": manifest_blocks,
        "pairing": pairing_report(blocks),
        "source_lock": source_lock,
        "encoder_contract": {
            "counts_only": True,
            "metadata_identifiers_excluded": sorted(FORBIDDEN_ENCODER_METADATA),
            "labels_are_metadata_only": True,
        },
    }
    atomic_json(root / "bundle_manifest.json", manifest)
    return manifest


def verify_bundle(root: Path) -> Dict[str, object]:
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for modality, expected in manifest["modalities"].items():
        modality_root = root / "modalities" / modality
        paths = {
            "counts_sha256": modality_root / "counts.csr.npz",
            "barcodes_sha256": modality_root / "barcodes.tsv.gz",
            "features_sha256": modality_root / "features.tsv.gz",
        }
        for field, path in paths.items():
            if not path.is_file() or sha256_file(path) != expected[field]:
                raise RuntimeError(f"Bundle integrity failure: {path}")
        matrix = sparse.load_npz(paths["counts_sha256"])
        if list(matrix.shape) != expected["shape"] or int(matrix.nnz) != expected["nnz"]:
            raise RuntimeError(f"Bundle shape/nnz failure: {modality}")
    return manifest


def download_locked(files: Sequence[Mapping[str, str]], root: Path) -> Tuple[Dict[str, Path], Dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "source.lock.json"
    previous = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
    resolved, locked = {}, {}
    for spec in files:
        role, name, url = spec["role"], spec["name"], spec["url"]
        destination = root / name
        partial = root / (name + ".part")
        if not destination.is_file():
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={"User-Agent": "WLD-Phase-A/1.0"})
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(request, timeout=180) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                with partial.open("ab" if append else "wb") as output:
                    shutil.copyfileobj(response, output, 8 * 1024 * 1024)
            os.replace(partial, destination)
        if destination.suffix == ".gz":
            with gzip.open(destination, "rb") as handle:
                if not handle.read(1):
                    raise RuntimeError(f"Empty gzip source: {destination}")
        digest = sha256_file(destination)
        if role in previous:
            expected = previous[role]
            if expected.get("url") != url or expected.get("name") != name:
                raise RuntimeError(f"Immutable source identity changed for role {role}")
            if expected.get("sha256") != digest or int(expected.get("bytes", -1)) != destination.stat().st_size:
                raise RuntimeError(f"Immutable source checksum changed for {destination}")
        resolved[role] = destination
        locked[role] = {"name": name, "url": url, "bytes": destination.stat().st_size, "sha256": digest}
    if set(previous).difference(locked):
        raise RuntimeError("The source manifest removed previously locked roles")
    atomic_json(lock_path, locked)
    return resolved, locked


def audit_encoder_metadata(names: Sequence[str]) -> None:
    bad = []
    for name in names:
        token = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        compact = token.replace("_", "")
        if token in FORBIDDEN_ENCODER_METADATA or compact in {value.replace("_", "") for value in FORBIDDEN_ENCODER_METADATA}:
            bad.append(name)
    if bad:
        raise ValueError(f"Metadata identifiers/proxies cannot be encoder features: {bad}")


def build_training_atlas(bundle_roots: Sequence[Path], output: Path, *, max_genes: int = 20000, max_peak_bins: int = 200000, peak_bin_size: int = 2000) -> Dict[str, object]:
    gene_studies: Counter[str] = Counter()
    peak_studies: Counter[str] = Counter()
    protein_studies: Counter[str] = Counter()
    metabolic_studies: Counter[str] = Counter()
    source_hashes = {}
    species = set()
    genome_builds = set()
    for root in bundle_roots:
        manifest_path = root / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        species.add(manifest["species"])
        genome_builds.add(manifest["genome_build"])
        source_hashes[str(manifest["cohort_id"])] = sha256_file(manifest_path)
        if "rna" in manifest["modalities"]:
            genes = {value["feature_name"] for value in read_bundle_features(root / "modalities" / "rna" / "features.tsv.gz")}
            gene_studies.update(genes)
        if "atac" in manifest["modalities"]:
            bins = set()
            for value in read_bundle_features(root / "modalities" / "atac" / "features.tsv.gz"):
                parsed = parse_peak(value["feature_name"])
                if parsed is not None:
                    chrom, start, end = parsed
                    center = (start + end) // 2
                    begin = (center // peak_bin_size) * peak_bin_size
                    bins.add(f"{chrom}:{begin}-{begin + peak_bin_size}")
            peak_studies.update(bins)
        if "protein" in manifest["modalities"]:
            proteins = {value["feature_name"] for value in read_bundle_features(root / "modalities" / "protein" / "features.tsv.gz")}
            protein_studies.update(proteins)
        if "metabolic" in manifest["modalities"]:
            metabolites = {value["feature_name"] for value in read_bundle_features(root / "modalities" / "metabolic" / "features.tsv.gz")}
            metabolic_studies.update(metabolites)
    if len(species) != 1:
        raise ValueError(f"Species cannot share a feature atlas without an audited ortholog contract: {sorted(species)}")
    if len(genome_builds) != 1:
        raise ValueError(f"Genome builds cannot share a peak atlas before explicit liftover: {sorted(genome_builds)}")
    genes = sorted(gene_studies, key=lambda value: (-gene_studies[value], value))[:max_genes]
    peaks = sorted(peak_studies, key=lambda value: (-peak_studies[value], value))[:max_peak_bins]
    proteins = sorted(protein_studies, key=lambda value: (-protein_studies[value], value))
    metabolites = sorted(metabolic_studies, key=lambda value: (-metabolic_studies[value], value))
    output.mkdir(parents=True, exist_ok=True)
    write_lines_gz(output / "shared_genes.tsv.gz", genes)
    write_lines_gz(output / "shared_peak_bins.tsv.gz", peaks)
    write_lines_gz(output / "shared_proteins.tsv.gz", proteins)
    write_lines_gz(output / "shared_metabolites.tsv.gz", metabolites)
    manifest = {
        "schema_version": "1.0",
        "training_cohorts_only": True,
        "species": next(iter(species)),
        "genome_build": next(iter(genome_builds)),
        "source_bundle_manifest_sha256": source_hashes,
        "genes": len(genes),
        "peak_bins": len(peaks),
        "proteins": len(proteins),
        "metabolites": len(metabolites),
        "peak_bin_size": peak_bin_size,
        "selection": "study prevalence then lexical tie-break; no expression labels or held-out studies",
    }
    atomic_json(output / "atlas_manifest.json", manifest)
    return manifest


def _read_vocab(path: Path) -> List[str]:
    return [row[0] for row in read_rows(path)]


def _projection_matrix(source_names: Sequence[str], vocabulary: Sequence[str]) -> sparse.csr_matrix:
    target = {name: index for index, name in enumerate(vocabulary)}
    rows, columns = [], []
    for source_index, name in enumerate(source_names):
        if name in target:
            rows.append(source_index)
            columns.append(target[name])
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, columns)), shape=(len(source_names), len(vocabulary)))


def project_bundle_to_atlas(bundle_root: Path, atlas_root: Path, output_root: Path) -> Dict[str, object]:
    """Project source-specific columns into a training-derived shared atlas."""

    source_manifest = verify_bundle(bundle_root)
    atlas_manifest = json.loads((atlas_root / "atlas_manifest.json").read_text())
    if source_manifest["species"] != atlas_manifest["species"]:
        raise ValueError("Bundle species does not match the atlas")
    if source_manifest["genome_build"] != atlas_manifest["genome_build"]:
        raise ValueError("Explicit liftover is required before atlas projection")
    projected: Dict[str, ModalityBlock] = {}
    for modality in source_manifest["modalities"]:
        block = project_modality_to_atlas(bundle_root, atlas_root, modality)
        if block is not None:
            projected[modality] = block
    if not projected:
        raise RuntimeError("No bundle modalities overlap the shared atlas")
    cohort = {
        "cohort_id": source_manifest["cohort_id"],
        "study_id": source_manifest["study_id"],
        "species": source_manifest["species"],
        "genome_build": source_manifest["genome_build"],
        "adapter": "training_atlas_projection",
        "donor_scope": source_manifest.get("donor_scope", ""),
    }
    lock = {
        "source_bundle_manifest": {
            "path": str(bundle_root / "bundle_manifest.json"),
            "sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        },
        "training_atlas_manifest": {
            "path": str(atlas_root / "atlas_manifest.json"),
            "sha256": sha256_file(atlas_root / "atlas_manifest.json"),
        },
    }
    return save_bundle(output_root, cohort, projected, lock)


def project_modality_to_atlas(
    bundle_root: Path, atlas_root: Path, modality: str
) -> Optional[ModalityBlock]:
    source_manifest = verify_bundle(bundle_root)
    atlas_manifest = json.loads((atlas_root / "atlas_manifest.json").read_text())
    if source_manifest["species"] != atlas_manifest["species"]:
        raise ValueError("Bundle species does not match the atlas")
    if source_manifest["genome_build"] != atlas_manifest["genome_build"]:
        raise ValueError("Explicit liftover is required before atlas projection")
    vocabularies = {
        "rna": _read_vocab(atlas_root / "shared_genes.tsv.gz"),
        "atac": _read_vocab(atlas_root / "shared_peak_bins.tsv.gz"),
        "protein": _read_vocab(atlas_root / "shared_proteins.tsv.gz"),
        "metabolic": _read_vocab(atlas_root / "shared_metabolites.tsv.gz"),
    }
    if modality not in source_manifest["modalities"]:
        return None
    vocabulary = vocabularies.get(modality, [])
    if not vocabulary:
        return None
    peak_bin_size = int(atlas_manifest["peak_bin_size"])
    source_root = bundle_root / "modalities" / modality
    matrix = sparse.load_npz(source_root / "counts.csr.npz").tocsr()
    source_features = read_bundle_features(source_root / "features.tsv.gz")
    if modality == "atac":
        names = []
        for value in source_features:
            parsed = parse_peak(value["feature_name"])
            if parsed is None:
                names.append("")
                continue
            chrom, start, end = parsed
            center = (start + end) // 2
            begin = (center // peak_bin_size) * peak_bin_size
            names.append(f"{chrom}:{begin}-{begin + peak_bin_size}")
    else:
        names = [value["feature_name"] for value in source_features]
    mapping = _projection_matrix(names, vocabulary)
    values = (matrix @ mapping).tocsr()
    features = [
        {"feature_id": name, "feature_name": name, "modality": modality, "source_index": str(index)}
        for index, name in enumerate(vocabulary)
    ]
    block = ModalityBlock(values, read_barcodes(source_root / "barcodes.tsv.gz"), features)
    block.validate()
    return block


def parse_peak(value: str) -> Optional[Tuple[str, int, int]]:
    match = re.match(r"^([^:_-]+(?:\.[0-9]+)?)[_:-](\d+)[_-](\d+)$", value)
    if not match:
        return None
    chrom = match.group(1)
    if not chrom.lower().startswith("chr"):
        chrom = "chr" + chrom
    return chrom, int(match.group(2)), int(match.group(3))


__all__ = [
    "ModalityBlock", "audit_encoder_metadata", "atomic_json", "build_training_atlas",
    "download_locked", "pairing_report", "parse_features", "parse_peak", "read_10x_h5",
    "read_10x_mtx", "read_adt_csv", "read_single_mtx", "save_bundle", "sha256_file",
    "project_bundle_to_atlas", "project_modality_to_atlas", "verify_bundle",
]
