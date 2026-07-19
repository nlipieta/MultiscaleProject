"""Raw-count SHARE-seq adapters for WLD corpus expansion.

The adapters preserve submitted RNA/ATAC observations and deposited barcode
relations.  They never pair cells by expression similarity, cell label, or an
embedding.  Experimental metadata remains in the locked raw source and is
described by a context manifest; it is not appended to encoder tensors.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.io import mmread

from wld_foundation_data import (
    ModalityBlock,
    atomic_json,
    parse_features,
    read_barcodes,
)


def _open_text(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open("rt", newline="")


def _clean(value: object) -> str:
    return str(value).strip().strip('"').strip("'")


def _delimiter(line: str) -> Optional[str]:
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return None


def _split(line: str, delimiter: Optional[str]) -> List[str]:
    values = line.rstrip("\n\r").split(delimiter) if delimiter else line.split()
    return [_clean(value) for value in values]


def read_metadata_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read submitted metadata without assigning any column to the encoder.

    Some GEO tables were written with row names enabled.  In those files the
    header has one fewer field than every observation because the row-name
    column is deliberately unnamed.  Preserve that field under an internal
    name instead of dropping it or shifting the biological columns.
    """

    with _open_text(path) as handle:
        first = handle.readline()
        while first and (not first.strip() or first.lstrip().startswith("#")):
            first = handle.readline()
        if not first:
            raise ValueError(f"Empty metadata table: {path}")
        delimiter = _delimiter(first)

        if delimiter:
            reader = csv.reader(chain((first,), handle), delimiter=delimiter)
            parsed = [
                [_clean(value) for value in row]
                for row in reader
                if row and any(_clean(value) for value in row)
            ]
        else:
            parsed = [_split(first, None)]
            parsed.extend(_split(line, None) for line in handle if line.strip())

    if len(parsed) < 2:
        raise ValueError(f"Metadata has no observations: {path}")

    header = parsed[0]
    rows = parsed[1:]
    observed_widths = Counter(len(row) for row in rows)
    modal_width, modal_count = observed_widths.most_common(1)[0]

    if len(header) + 1 == modal_width:
        # R write.table(row.names=TRUE) and several submitted GEO tables use
        # an unnamed first column.  This name is intentionally ineligible as
        # a biological pairing key (see _name_priority).
        header = ["__row_id__", *header]
    elif len(header) != modal_width:
        raise ValueError(
            f"Metadata header width differs from observations in {path}: "
            f"header={len(header)}, modal_observation={modal_width} "
            f"({modal_count}/{len(rows)} rows)"
        )

    bad_widths = Counter(len(row) for row in rows if len(row) != len(header))
    if bad_widths:
        raise ValueError(
            f"Metadata observation widths are inconsistent in {path}: "
            f"expected {len(header)}, observed {dict(sorted(bad_widths.items()))}"
        )

    # A blank first heading is the other common representation of submitted
    # row names.  It may be used only as deposited alignment evidence; metadata
    # fields are never appended to encoder tensors.
    header = [
        value or ("__row_id__" if index == 0 else f"__unnamed_{index}__")
        for index, value in enumerate(header)
    ]
    if not rows:
        raise ValueError(f"Metadata has no observations: {path}")
    return header, rows


def _column(header: Sequence[str], rows: Sequence[Sequence[str]], index: int) -> List[str]:
    return [_clean(row[index]) for row in rows]


def _candidate_columns(
    header: Sequence[str], rows: Sequence[Sequence[str]], expected_cells: int
) -> Dict[str, List[str]]:
    if len(rows) != expected_cells:
        raise ValueError(
            f"Metadata has {len(rows)} rows but the count matrix has {expected_cells} cells"
        )
    candidates = {}
    for index, name in enumerate(header):
        deposited_row_id = name == "__row_id__"
        if not deposited_row_id and _name_priority(name) <= 0:
            continue
        values = _column(header, rows, index)
        nonempty = [value for value in values if value]
        if len(nonempty) != expected_cells:
            continue
        uniqueness = len(set(nonempty)) / max(1, len(nonempty))
        numeric = []
        for value in nonempty:
            try:
                numeric.append(int(value))
            except ValueError:
                numeric = []
                break
        sequential = bool(numeric) and (
            numeric == list(range(len(numeric)))
            or numeric == list(range(1, len(numeric) + 1))
        )
        # ModalityBlock requires unique observation identifiers.  Never repair
        # duplicates with labels, embeddings, expression, or row order.
        if uniqueness == 1.0 and not sequential:
            candidates[name] = values
    return candidates


def _name_priority(name: str) -> int:
    if name.startswith("__"):
        return 0
    token = re.sub(r"[^a-z0-9]+", "_", name.lower())
    score = 0
    for word, points in (("barcode", 12), ("rna", 5), ("atac", 5), ("cell", 4), ("id", 1)):
        if word in token:
            score += points
    for word in ("cluster", "type", "label", "umap", "pseudotime"):
        if word in token:
            score -= 20
    return score


def _pairing_priority(name: str) -> int:
    # Deposited row names are legitimate alignment evidence, but named barcode
    # or cell-ID fields win when both are available.
    return 1 if name == "__row_id__" else _name_priority(name)


def _unpaired_observation_ids(prefix: str, cells: int) -> List[str]:
    """Create disjoint bookkeeping IDs without asserting biological pairing."""

    return [f"__{prefix}_unpaired_{index:09d}" for index in range(cells)]


def _best_shared_metadata_key(
    left_header: Sequence[str],
    left_rows: Sequence[Sequence[str]],
    right_header: Sequence[str],
    right_rows: Sequence[Sequence[str]],
    left_cells: int,
    right_cells: int,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    left = _candidate_columns(left_header, left_rows, left_cells)
    right = _candidate_columns(right_header, right_rows, right_cells)
    if not left or not right:
        evidence = {
            "method": "no_shared_deposited_identifier",
            "left_candidate_fields": sorted(left),
            "right_candidate_fields": sorted(right),
            "intersection": 0,
            "overlap_fraction": 0.0,
            "expression_or_label_matching_used": False,
            "synthetic_cell_pairing_used": False,
        }
        return (
            _unpaired_observation_ids("rna", left_cells),
            _unpaired_observation_ids("atac", right_cells),
            evidence,
        )

    best = None
    for left_name, left_values in left.items():
        left_set = set(left_values)
        for right_name, right_values in right.items():
            right_set = set(right_values)
            overlap = len(left_set.intersection(right_set))
            fraction = overlap / max(1, min(len(left_set), len(right_set)))
            score = (
                fraction,
                overlap,
                _pairing_priority(left_name) + _pairing_priority(right_name),
            )
            if best is None or score > best[0]:
                best = (score, left_name, left_values, right_name, right_values)
    assert best is not None
    score, left_name, left_values, right_name, right_values = best
    evidence = {
        "method": "deposited_metadata_identifier_intersection",
        "left_column": left_name,
        "right_column": right_name,
        "intersection": int(score[1]),
        "overlap_fraction": float(score[0]),
        "expression_or_label_matching_used": False,
        "synthetic_cell_pairing_used": False,
    }
    if score[0] < 0.80:
        evidence["method"] = "insufficient_shared_deposited_identifier_overlap"
        return (
            _unpaired_observation_ids("rna", left_cells),
            _unpaired_observation_ids("atac", right_cells),
            evidence,
        )
    return left_values, right_values, evidence


def _read_feature_axis(path: Path, modality: str) -> List[Dict[str, str]]:
    values = parse_features(path, modality)
    if values and values[0]["feature_name"].strip().lower() in {
        "gene", "genes", "gene_name", "gene_names", "peak", "peaks"
    }:
        values = values[1:]
        for index, value in enumerate(values):
            value["source_index"] = str(index)
    return values


def _read_matrix_market(path: Path) -> sparse.csr_matrix:
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as handle:
        matrix = mmread(handle)
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    return matrix.tocsr()


def _orient(
    matrix: sparse.spmatrix,
    *,
    features: Sequence[Mapping[str, str]],
    cells: int,
    label: str,
) -> sparse.csr_matrix:
    expected = (cells, len(features))
    if matrix.shape == expected:
        result = matrix.tocsr()
    elif matrix.shape == expected[::-1]:
        result = matrix.transpose().tocsr()
    else:
        raise ValueError(
            f"{label} dimensions {matrix.shape} do not match {cells} cells x {len(features)} features"
        )
    if result.data.size and ((result.data < 0).any() or not np.isfinite(result.data).all()):
        raise ValueError(f"{label} must contain finite non-negative raw counts")
    return result


def read_dense_feature_table(path: Path, modality: str) -> ModalityBlock:
    """Stream a deposited feature-by-cell count table into sparse storage."""

    with _open_text(path) as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"Empty count table: {path}")
        if header_line.lstrip().startswith("%%MatrixMarket"):
            raise ValueError(
                f"{path} is Matrix Market but has no separate submitted feature/barcode axes"
            )
        delimiter = _delimiter(header_line)
        header = _split(header_line, delimiter)
        if len(header) < 2:
            raise ValueError(f"Count-table header is malformed: {path}")
        first_token = header[0].lower()
        if first_token in {"", "gene", "genes", "feature", "features", "id", "name"}:
            barcodes = header[1:]
        else:
            # The first field is the feature-index heading even when it has a
            # dataset-specific name such as ``gene_name`` or ``row.names``.
            barcodes = header[1:]
        if len(set(barcodes)) != len(barcodes):
            raise ValueError(f"Duplicate cell columns in {path}")

        features: List[Dict[str, str]] = []
        chunks: List[sparse.csr_matrix] = []
        dense_rows: List[np.ndarray] = []
        chunk_size = 64

        def flush() -> None:
            if dense_rows:
                chunks.append(sparse.csr_matrix(np.vstack(dense_rows), dtype=np.float32))
                dense_rows.clear()

        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            if delimiter:
                feature_name, separator, remainder = line.rstrip("\n\r").partition(delimiter)
                if not separator:
                    raise ValueError(f"Malformed count row {line_number} in {path}")
                values = np.fromstring(remainder, sep=delimiter, dtype=np.float32)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Malformed count row {line_number} in {path}")
                feature_name, remainder = parts
                values = np.fromstring(remainder, sep=" ", dtype=np.float32)
            if values.size != len(barcodes):
                raise ValueError(
                    f"Count row {line_number} has {values.size} cells; expected {len(barcodes)}"
                )
            if (values < 0).any() or not np.isfinite(values).all():
                raise ValueError(f"Invalid raw count at row {line_number} in {path}")
            index = len(features)
            name = _clean(feature_name)
            features.append({
                "feature_id": name,
                "feature_name": name,
                "modality": modality,
                "source_index": str(index),
            })
            dense_rows.append(values)
            if len(dense_rows) >= chunk_size:
                flush()
        flush()
    if not features:
        raise ValueError(f"Count table has no features: {path}")
    feature_by_cell = sparse.vstack(chunks, format="csr")
    block = ModalityBlock(feature_by_cell.transpose().tocsr(), barcodes, features)
    block.validate()
    return block


def _align_exact(
    rna: ModalityBlock,
    atac: ModalityBlock,
    evidence: Mapping[str, object],
    *,
    minimum_fraction: float = 0.80,
) -> Tuple[ModalityBlock, ModalityBlock, Dict[str, object]]:
    atac_lookup = {barcode: index for index, barcode in enumerate(atac.barcodes)}
    common = [barcode for barcode in rna.barcodes if barcode in atac_lookup]
    fraction = len(common) / max(1, min(len(rna.barcodes), len(atac.barcodes)))
    updated = dict(evidence)
    updated["aligned_intersection"] = len(common)
    updated["aligned_fraction"] = float(fraction)
    if fraction < minimum_fraction:
        updated["result"] = "unpaired_population"
        return rna, atac, updated
    rna_lookup = {barcode: index for index, barcode in enumerate(rna.barcodes)}
    rna_rows = [rna_lookup[value] for value in common]
    atac_rows = [atac_lookup[value] for value in common]
    rna_aligned = ModalityBlock(rna.matrix[rna_rows].tocsr(), list(common), list(rna.features))
    atac_aligned = ModalityBlock(atac.matrix[atac_rows].tocsr(), list(common), list(atac.features))
    rna_aligned.validate()
    atac_aligned.validate()
    updated["result"] = "exact_after_deposited_identifier_alignment"
    updated["dropped_rna_cells"] = len(rna.barcodes) - len(common)
    updated["dropped_atac_cells"] = len(atac.barcodes) - len(common)
    return rna_aligned, atac_aligned, updated


def ingest_shareseq_metadata_pair(
    files: Mapping[str, Path]
) -> Tuple[Dict[str, ModalityBlock], Dict[str, object], Dict[str, object]]:
    rna_features = _read_feature_axis(files["rna_features"], "rna")
    atac_features = _read_feature_axis(files["atac_features"], "atac")
    rna_header, rna_rows = read_metadata_table(files["rna_metadata"])
    atac_header, atac_rows = read_metadata_table(files["atac_metadata"])
    rna_matrix_raw = _read_matrix_market(files["rna_matrix"])
    atac_matrix_raw = _read_matrix_market(files["atac_matrix"])
    rna_matrix = _orient(
        rna_matrix_raw, features=rna_features, cells=len(rna_rows), label="RNA"
    )
    atac_matrix = _orient(
        atac_matrix_raw, features=atac_features, cells=len(atac_rows), label="ATAC"
    )
    rna_barcodes, atac_barcodes, evidence = _best_shared_metadata_key(
        rna_header, rna_rows, atac_header, atac_rows, rna_matrix.shape[0], atac_matrix.shape[0]
    )
    rna = ModalityBlock(rna_matrix, rna_barcodes, rna_features)
    atac = ModalityBlock(atac_matrix, atac_barcodes, atac_features)
    rna.validate()
    atac.validate()
    rna, atac, evidence = _align_exact(rna, atac, evidence)
    context = {
        "metadata_roles": ["rna_metadata", "atac_metadata"],
        "rna_metadata_fields": rna_header,
        "atac_metadata_fields": atac_header,
        "values_retained_in_locked_raw_sources": True,
        "metadata_appended_to_encoder": False,
    }
    return {"rna": rna, "atac": atac}, evidence, context


def _best_crosswalk(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    atac_barcodes: Sequence[str],
    rna_barcodes: Sequence[str],
) -> Tuple[Optional[Dict[str, str]], Dict[str, object]]:
    atac_set, rna_set = set(atac_barcodes), set(rna_barcodes)
    candidates = {}
    for index, name in enumerate(header):
        if _name_priority(name) <= 0:
            continue
        values = _column(header, rows, index)
        if len(set(values)) / max(1, len(values)) >= 0.80:
            candidates[name] = values
    best = None
    for atac_name, atac_values in candidates.items():
        atac_overlap = len(set(atac_values).intersection(atac_set))
        for rna_name, rna_values in candidates.items():
            rna_overlap = len(set(rna_values).intersection(rna_set))
            paired = sum(
                1 for a, r in zip(atac_values, rna_values)
                if a in atac_set and r in rna_set
            )
            score = (paired, atac_overlap + rna_overlap, _name_priority(atac_name) + _name_priority(rna_name))
            if best is None or score > best[0]:
                best = (score, atac_name, atac_values, rna_name, rna_values)
    if best is None or best[0][0] == 0:
        return None, {"method": "no_usable_deposited_crosswalk", "expression_or_label_matching_used": False}
    _, atac_name, atac_values, rna_name, rna_values = best
    mapping = {
        atac_value: rna_value
        for atac_value, rna_value in zip(atac_values, rna_values)
        if atac_value in atac_set and rna_value in rna_set
    }
    return mapping, {
        "method": "deposited_barcode_crosswalk",
        "atac_column": atac_name,
        "rna_column": rna_name,
        "mapped_barcodes": len(mapping),
        "expression_or_label_matching_used": False,
    }


def ingest_shareseq_legacy_pair(
    files: Mapping[str, Path]
) -> Tuple[Dict[str, ModalityBlock], Dict[str, object], Dict[str, object]]:
    rna = read_dense_feature_table(files["rna_table"], "rna")
    atac_features = _read_feature_axis(files["atac_features"], "atac")
    atac_barcodes = read_barcodes(files["atac_barcodes"])
    atac_matrix = _orient(
        _read_matrix_market(files["atac_matrix"]),
        features=atac_features,
        cells=len(atac_barcodes),
        label="ATAC",
    )
    atac = ModalityBlock(atac_matrix, atac_barcodes, atac_features)
    atac.validate()
    evidence: Dict[str, object] = {
        "method": "submitted_matrix_barcodes",
        "expression_or_label_matching_used": False,
    }
    context: Dict[str, object] = {
        "metadata_roles": [],
        "values_retained_in_locked_raw_sources": True,
        "metadata_appended_to_encoder": False,
    }
    if set(rna.barcodes) != set(atac.barcodes) and "pairing_metadata" in files:
        header, rows = read_metadata_table(files["pairing_metadata"])
        crosswalk, evidence = _best_crosswalk(header, rows, atac.barcodes, rna.barcodes)
        context["metadata_roles"] = ["pairing_metadata"]
        context["pairing_metadata_fields"] = header
        if crosswalk:
            mapped = [crosswalk.get(value, value) for value in atac.barcodes]
            if len(set(mapped)) == len(mapped):
                atac = ModalityBlock(atac.matrix, mapped, atac.features)
                atac.validate()
    rna, atac, evidence = _align_exact(rna, atac, evidence)
    return {"rna": rna, "atac": atac}, evidence, context


def write_context_manifest(
    bundle_root: Path,
    cohort: Mapping[str, object],
    discovered: Mapping[str, object],
) -> Dict[str, object]:
    payload = {
        "schema_version": "1.0",
        "cohort_id": cohort["cohort_id"],
        "study_id": cohort["study_id"],
        "tissue": cohort.get("tissue", ""),
        "declared_contract": cohort.get("context_contract", {}),
        "discovered_metadata": dict(discovered),
        "policy": {
            "observation_level_context_is_not_frozen": True,
            "context_values_are_retained_outside_encoder": True,
            "context_may_enter_future_conditioning_only_after_fold_local_encoding": True,
            "identity_and_state_proxy_fields_remain_forbidden": True,
        },
    }
    atomic_json(bundle_root / "context_manifest.json", payload)
    return payload


__all__ = [
    "ingest_shareseq_legacy_pair",
    "ingest_shareseq_metadata_pair",
    "read_dense_feature_table",
    "read_metadata_table",
    "write_context_manifest",
]
