"""Build a leakage-safe WLD cohort from temporal single-cell multiome data.

The reference use case is the human skeletal-muscle exercise multiome in
GSE240061, with a tissue-matched promoter-capture Hi-C scaffold from GSE126100.
The builder deliberately consumes canonical evidence tables rather than
silently guessing columns in changing repository files.

Input matrices use Matrix Market orientation ``features x cells`` and share a
single barcode file.  Feature selection is fitted only on training subjects.
The output follows ``wld_temporal_data_contract.md`` and can be passed directly
to ``wld_temporal_training.py validate`` or ``benchmark``.

This script does not infer cell pairs, pseudotime, cell identity, or attractor
labels.  Hi-C, motif, TF-regulatory, and signaling evidence become hard sparse
topology; measured ATAC and time-zero cues remain dynamic observations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.io import mmread


SCHEMA_VERSION = 1
CANONICAL_TABLES = {
    "peak_gene": ("peak_id", "gene", "score"),
    "motif": ("peak_id", "tf", "score"),
    "tf_gene": ("source", "target", "sign", "score"),
    "signaling": ("source", "target", "source_type", "target_type", "sign", "score"),
    "tf_peak": ("tf", "peak_id", "sign", "score"),
}


@dataclass(frozen=True)
class Evidence:
    peak_gene: Tuple[Mapping[str, str], ...]
    motif: Tuple[Mapping[str, str], ...]
    tf_gene: Tuple[Mapping[str, str], ...]
    signaling: Tuple[Mapping[str, str], ...]
    tf_peak: Tuple[Mapping[str, str], ...]


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def read_table(path: Path, required: Sequence[str]) -> Tuple[Mapping[str, str], ...]:
    with _open_text(path) as handle:
        sample = handle.read(8192)
        handle.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = tuple(reader.fieldnames or ())
        missing = sorted(set(required).difference(fields))
        if missing:
            raise ValueError(f"{path} is missing canonical columns {missing}; found {fields}")
        return tuple({key: (value or "").strip() for key, value in row.items()} for row in reader)


def read_features(path: Path, column: int) -> List[str]:
    values = []
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n\r").split("\t")
            if column >= len(fields):
                raise ValueError(f"{path}:{line_number} has no zero-based column {column}")
            values.append(fields[column].strip())
    if not values or any(not value for value in values):
        raise ValueError(f"{path} contains empty or no feature names")
    return values


def read_barcodes(path: Path) -> List[str]:
    with _open_text(path) as handle:
        values = [line.strip().split("\t")[0] for line in handle if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("Barcodes must be unique.")
    return values


def read_matrix(path: Path, features: int, cells: int) -> sparse.csr_matrix:
    matrix = mmread(path)
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    if matrix.shape != (features, cells):
        raise ValueError(
            f"{path} has shape {matrix.shape}; expected features x cells {(features, cells)}"
        )
    if matrix.nnz and (not np.isfinite(matrix.data).all() or (matrix.data < 0).any()):
        raise ValueError(f"{path} must contain finite non-negative counts")
    return matrix.transpose().tocsr()


def read_metadata(path: Path, cell_column: str) -> Dict[str, Mapping[str, str]]:
    rows = read_table(path, (cell_column,))
    result = {}
    for row in rows:
        cell = row[cell_column]
        if not cell:
            raise ValueError("Metadata contains an empty cell identifier.")
        if cell in result:
            raise ValueError(f"Duplicate metadata cell identifier: {cell}")
        result[cell] = row
    return result


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    if not normalized:
        raise ValueError(f"Cannot construct an identifier from {value!r}")
    return normalized


def _parse_sign(value: str) -> float:
    normalized = value.strip().lower()
    positive = {"1", "+1", "+", "positive", "activation", "activating", "stimulation"}
    negative = {"-1", "-", "negative", "repression", "repressive", "inhibition"}
    if normalized in positive:
        return 1.0
    if normalized in negative:
        return -1.0
    try:
        number = float(normalized)
    except ValueError as error:
        raise ValueError(f"Unrecognized edge sign: {value!r}") from error
    if number == 0 or not math.isfinite(number):
        raise ValueError(f"Edge sign must be finite and non-zero: {value!r}")
    return float(np.sign(number))


def _positive_score(value: str) -> float:
    score = float(value)
    if not math.isfinite(score) or score < 0:
        raise ValueError(f"Evidence score must be finite and non-negative: {value!r}")
    return score


def _normalize_nonnegative(matrix: np.ndarray) -> np.ndarray:
    maximum = float(matrix.max(initial=0.0))
    if maximum > 0:
        matrix /= maximum
    return matrix


def _library_normalize(
    matrix: sparse.csr_matrix, scale: float = 1e4
) -> sparse.csr_matrix:
    result = matrix.astype(np.float32, copy=True)
    totals = np.asarray(result.sum(axis=1)).ravel()
    factors = np.divide(scale, totals, out=np.zeros_like(totals), where=totals > 0)
    result = sparse.diags(factors) @ result
    return result.tocsr()


def _log_normalize(matrix: sparse.csr_matrix, scale: float = 1e4) -> sparse.csr_matrix:
    result = _library_normalize(matrix, scale)
    result.data = np.log1p(result.data)
    return result.tocsr()


def _sparse_variance(matrix: sparse.csr_matrix) -> np.ndarray:
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_square = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    return np.maximum(mean_square - mean * mean, 0.0)


def _split_groups(
    group_condition: Mapping[str, str], seed: int
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    by_condition: MutableMapping[str, List[str]] = defaultdict(list)
    for group, condition in group_condition.items():
        by_condition[condition].append(group)
    split = {"train": [], "validation": [], "test": []}
    for condition in sorted(by_condition):
        groups = sorted(by_condition[condition])
        rng.shuffle(groups)
        if len(groups) >= 3:
            split["validation"].append(groups[-2])
            split["test"].append(groups[-1])
            split["train"].extend(groups[:-2])
        elif len(groups) == 2:
            split["train"].append(groups[0])
            split["test"].append(groups[1])
        else:
            split["train"].append(groups[0])
    if not split["validation"]:
        candidates = [group for group in split["train"] if len(split["train"]) > 1]
        if not candidates:
            raise ValueError("At least three biological groups are required for train/validation/test.")
        moved = sorted(candidates)[-1]
        split["train"].remove(moved)
        split["validation"].append(moved)
    if not split["test"]:
        candidates = [group for group in split["train"] if len(split["train"]) > 1]
        if not candidates:
            raise ValueError("At least three biological groups are required for train/validation/test.")
        moved = sorted(candidates)[-1]
        split["train"].remove(moved)
        split["test"].append(moved)
    return {name: sorted(values) for name, values in split.items()}


def _read_split(path: Path, groups: Iterable[str]) -> Dict[str, List[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    split = {name: list(map(str, value[name])) for name in ("train", "validation", "test")}
    flattened = [group for values in split.values() for group in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Split file assigns at least one group more than once.")
    if set(flattened) != set(groups):
        raise ValueError("Split file must assign every eligible group exactly once.")
    if any(not values for values in split.values()):
        raise ValueError("Every split must contain at least one group.")
    return split


def load_evidence(args: argparse.Namespace) -> Evidence:
    return Evidence(
        peak_gene=read_table(args.peak_gene_links, CANONICAL_TABLES["peak_gene"]),
        motif=read_table(args.motif_hits, CANONICAL_TABLES["motif"]),
        tf_gene=read_table(args.tf_gene_edges, CANONICAL_TABLES["tf_gene"]),
        signaling=read_table(args.signaling_edges, CANONICAL_TABLES["signaling"]),
        tf_peak=(
            read_table(args.tf_peak_effects, CANONICAL_TABLES["tf_peak"])
            if args.tf_peak_effects is not None
            else tuple()
        ),
    )


def _compile_signaling(
    rows: Sequence[Mapping[str, str]],
    candidate_cues: Sequence[str],
    selected_tfs: Sequence[str],
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    signed: MutableMapping[Tuple[str, str, str, str], float] = defaultdict(float)
    for row in rows:
        source_type = row["source_type"].lower()
        target_type = row["target_type"].lower()
        if (source_type, target_type) not in {
            ("cue", "signal"),
            ("signal", "signal"),
            ("signal", "tf"),
        }:
            raise ValueError(
                "Signaling edges must be cue->signal, signal->signal, or signal->tf."
            )
        key = (row["source"], row["target"], source_type, target_type)
        signed[key] += _parse_sign(row["sign"]) * _positive_score(row["score"])

    cue_set = set(candidate_cues)
    tf_set = set(selected_tfs)
    forward: MutableMapping[str, set[str]] = defaultdict(set)
    reverse: MutableMapping[str, set[str]] = defaultdict(set)
    cue_edges = []
    signal_edges = []
    tf_edges = []
    for (source, target, source_type, target_type), value in signed.items():
        if value == 0:
            continue
        if source_type == "cue":
            if source not in cue_set:
                continue
            cue_edges.append((source, target, value))
            forward[f"cue:{source}"].add(f"signal:{target}")
            reverse[f"signal:{target}"].add(f"cue:{source}")
        elif target_type == "tf":
            if target not in tf_set:
                continue
            tf_edges.append((source, target, value))
            forward[f"signal:{source}"].add(f"tf:{target}")
            reverse[f"tf:{target}"].add(f"signal:{source}")
        else:
            signal_edges.append((source, target, value))
            forward[f"signal:{source}"].add(f"signal:{target}")
            reverse[f"signal:{target}"].add(f"signal:{source}")

    reachable = set()
    queue = deque(f"cue:{name}" for name in candidate_cues)
    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        queue.extend(forward[node])
    productive = set()
    queue = deque(f"tf:{name}" for name in selected_tfs)
    while queue:
        node = queue.popleft()
        if node in productive:
            continue
        productive.add(node)
        queue.extend(reverse[node])
    retained = reachable.intersection(productive)
    cues = [name for name in candidate_cues if f"cue:{name}" in retained]
    excluded = [name for name in candidate_cues if name not in cues]
    signals = sorted(node.split(":", 1)[1] for node in retained if node.startswith("signal:"))
    if not cues or not signals:
        raise ValueError("No complete measured-cue -> signaling -> selected-TF path was found.")
    cue_index = {value: index for index, value in enumerate(cues)}
    signal_index = {value: index for index, value in enumerate(signals)}
    tf_index = {value: index for index, value in enumerate(selected_tfs)}
    cue_signal = np.zeros((len(cues), len(signals)), dtype=np.float32)
    signal_signal = np.zeros((len(signals), len(signals)), dtype=np.float32)
    signal_tf = np.zeros((len(signals), len(selected_tfs)), dtype=np.float32)
    for source, target, value in cue_edges:
        if source in cue_index and target in signal_index:
            cue_signal[cue_index[source], signal_index[target]] = value
    for source, target, value in signal_edges:
        if source in signal_index and target in signal_index:
            signal_signal[signal_index[source], signal_index[target]] = value
    for source, target, value in tf_edges:
        if source in signal_index and target in tf_index:
            signal_tf[signal_index[source], tf_index[target]] = value
    return cues, signals, cue_signal, signal_signal, signal_tf, excluded


def compile_priors(
    evidence: Evidence,
    genes_all: Sequence[str],
    peaks_all: Sequence[str],
    gene_variance: np.ndarray,
    peak_variance: np.ndarray,
    candidate_cues: Sequence[str],
    max_genes: int,
    max_peaks: int,
    max_tfs: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    gene_lookup = {name: index for index, name in enumerate(genes_all)}
    peak_lookup = {name: index for index, name in enumerate(peaks_all)}
    linked: MutableMapping[Tuple[str, str], float] = defaultdict(float)
    for row in evidence.peak_gene:
        if row["peak_id"] in peak_lookup and row["gene"] in gene_lookup:
            linked[(row["peak_id"], row["gene"])] = max(
                linked[(row["peak_id"], row["gene"])], _positive_score(row["score"])
            )
    motifs: MutableMapping[Tuple[str, str], float] = defaultdict(float)
    for row in evidence.motif:
        if row["peak_id"] in peak_lookup and row["tf"] in gene_lookup:
            motifs[(row["peak_id"], row["tf"])] = max(
                motifs[(row["peak_id"], row["tf"])], _positive_score(row["score"])
            )
    signed_tf_gene: MutableMapping[Tuple[str, str], float] = defaultdict(float)
    for row in evidence.tf_gene:
        if row["source"] in gene_lookup and row["target"] in gene_lookup:
            signed_tf_gene[(row["source"], row["target"])] += (
                _parse_sign(row["sign"]) * _positive_score(row["score"])
            )
    if not linked or not motifs or not signed_tf_gene:
        raise ValueError("The canonical evidence tables have no overlap with matrix features.")

    peaks_with_links = {peak for peak, _ in linked}
    regulatory_sources = {source for source, _ in signed_tf_gene}
    tf_local_score: MutableMapping[str, int] = defaultdict(int)
    for peak, tf in motifs:
        if peak in peaks_with_links and tf in regulatory_sources:
            tf_local_score[tf] += 1
    ranked_tfs = sorted(tf_local_score, key=lambda tf: (-tf_local_score[tf], tf))
    selected_tfs = ranked_tfs[:max_tfs]
    if len(selected_tfs) < 2:
        raise ValueError("Fewer than two TFs have motif, contact, expression, and regulatory support.")

    link_genes = {gene for _, gene in linked}
    ranked_genes = sorted(
        link_genes,
        key=lambda gene: (-float(gene_variance[gene_lookup[gene]]), gene),
    )
    selected_gene_set = set(selected_tfs)
    for gene in ranked_genes:
        if len(selected_gene_set) >= max_genes:
            break
        selected_gene_set.add(gene)
    selected_genes = sorted(
        selected_gene_set,
        key=lambda gene: (-float(gene_variance[gene_lookup[gene]]), gene),
    )
    gene_index = {value: index for index, value in enumerate(selected_genes)}
    tf_index = {value: index for index, value in enumerate(selected_tfs)}

    peaks_with_selected_motif = {
        peak for peak, tf in motifs if tf in tf_index
    }
    eligible_peaks = {
        peak
        for peak, gene in linked
        if gene in gene_index and peak in peaks_with_selected_motif
    }
    selected_peaks = sorted(
        eligible_peaks,
        key=lambda peak: (-float(peak_variance[peak_lookup[peak]]), peak),
    )[:max_peaks]
    if len(selected_peaks) < 2:
        raise ValueError("Fewer than two peaks satisfy contact and localized motif evidence.")
    selected_peaks.sort()
    peak_index = {value: index for index, value in enumerate(selected_peaks)}

    peak_to_gene = np.zeros((len(selected_peaks), len(selected_genes)), dtype=np.float32)
    for (peak, gene), score in linked.items():
        if peak in peak_index and gene in gene_index:
            peak_to_gene[peak_index[peak], gene_index[gene]] = score
    peak_tf_motif = np.zeros((len(selected_peaks), len(selected_tfs)), dtype=np.float32)
    for (peak, tf), score in motifs.items():
        if peak in peak_index and tf in tf_index:
            peak_tf_motif[peak_index[peak], tf_index[tf]] = score
    _normalize_nonnegative(peak_to_gene)
    _normalize_nonnegative(peak_tf_motif)

    localized = (peak_tf_motif.T @ peak_to_gene) > 0
    tf_gene_support = np.zeros((len(selected_tfs), len(selected_genes)), dtype=np.float32)
    for (source, target), value in signed_tf_gene.items():
        if source in tf_index and target in gene_index:
            source_index = tf_index[source]
            target_index = gene_index[target]
            if localized[source_index, target_index] and value != 0:
                tf_gene_support[source_index, target_index] = value
    if not np.count_nonzero(tf_gene_support):
        raise ValueError("No TF-gene edge survived the required motif x contact intersection.")
    maximum_tf_score = float(np.abs(tf_gene_support).max(initial=0.0))
    if maximum_tf_score > 0:
        tf_gene_support /= maximum_tf_score

    circuit_tf_tf = np.zeros((len(selected_tfs), len(selected_tfs)), dtype=np.float32)
    for source, source_index in tf_index.items():
        for target, target_index in tf_index.items():
            support = tf_gene_support[source_index, gene_index[target]]
            if support != 0:
                circuit_tf_tf[source_index, target_index] = support
    if not np.count_nonzero(circuit_tf_tf):
        raise ValueError("No TF-to-TF circuit edge survived localized support checks.")
    tf_gene_index = np.asarray([gene_index[tf] for tf in selected_tfs], dtype=np.int64)

    tf_peak_effect = np.zeros((len(selected_tfs), len(selected_peaks)), dtype=np.float32)
    for row in evidence.tf_peak:
        tf, peak = row["tf"], row["peak_id"]
        if tf in tf_index and peak in peak_index and peak_tf_motif[peak_index[peak], tf_index[tf]] > 0:
            tf_peak_effect[tf_index[tf], peak_index[peak]] += (
                _parse_sign(row["sign"]) * _positive_score(row["score"])
            )
    maximum_peak_effect = float(np.abs(tf_peak_effect).max(initial=0.0))
    if maximum_peak_effect > 0:
        tf_peak_effect /= maximum_peak_effect

    cues, signals, cue_signal, signal_signal, signal_tf, excluded_cues = _compile_signaling(
        evidence.signaling, candidate_cues, selected_tfs
    )
    if candidate_cues[0] not in cues:
        raise ValueError(
            f"The primary experimental cue {candidate_cues[0]!r} has no complete "
            "cue-to-signal-to-TF path."
        )
    for matrix in (cue_signal, signal_signal, signal_tf):
        maximum = float(np.abs(matrix).max(initial=0.0))
        if maximum > 0:
            matrix /= maximum

    priors = {
        "peak_to_gene": peak_to_gene,
        "peak_tf_motif": peak_tf_motif,
        "tf_gene_support": tf_gene_support,
        "circuit_tf_tf": circuit_tf_tf,
        "tf_gene_index": tf_gene_index,
        "signal_signal": signal_signal,
        "signal_tf": signal_tf,
        "cue_signal": cue_signal,
        "tf_peak_effect": tf_peak_effect,
    }
    registry: Dict[str, object] = {
        "genes": selected_genes,
        "peaks": selected_peaks,
        "tfs": selected_tfs,
        "signals": signals,
        "cues": cues,
        "excluded_cues_without_complete_signaling_path": excluded_cues,
        "edge_counts": {
            "peak_to_gene": int(np.count_nonzero(peak_to_gene)),
            "peak_tf_motif": int(np.count_nonzero(peak_tf_motif)),
            "tf_gene_support": int(np.count_nonzero(tf_gene_support)),
            "circuit_tf_tf": int(np.count_nonzero(circuit_tf_tf)),
            "signal_signal": int(np.count_nonzero(signal_signal)),
            "signal_tf": int(np.count_nonzero(signal_tf)),
            "cue_signal": int(np.count_nonzero(cue_signal)),
            "tf_peak_effect": int(np.count_nonzero(tf_peak_effect)),
        },
    }
    return priors, registry


def _metabolic_values(
    path: Optional[Path],
    group_column: str,
    columns: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    if path is None:
        return {}
    rows = read_table(path, (group_column, *columns))
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for row in rows:
        group = row[group_column]
        if group in result:
            raise ValueError(f"Duplicate metabolic-covariate group: {group}")
        values = {}
        for column in columns:
            raw = row[column].strip()
            if raw == "" or raw.lower() in {"na", "nan", "null"}:
                values[column] = None
            else:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite metabolic value for {group}/{column}")
                values[column] = value
        result[group] = values
    return result


def _normalize_group_cues(
    groups: Sequence[str],
    train_groups: Sequence[str],
    conditions: Mapping[str, float],
    condition_cue_name: str,
    metabolic: Mapping[str, Mapping[str, Optional[float]]],
    metabolic_columns: Sequence[str],
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray], List[Mapping[str, object]]]:
    names = [condition_cue_name] + [f"metabolic:{column}" for column in metabolic_columns]
    values = {group: np.zeros(len(names), dtype=np.float32) for group in groups}
    masks = {group: np.zeros(len(names), dtype=np.float32) for group in groups}
    for group in groups:
        values[group][0] = float(conditions[group])
        masks[group][0] = 1.0
    provenance: List[Mapping[str, object]] = [
        {
            "name": condition_cue_name,
            "measurement_level": "subject",
            "source": "experimental assignment in multiome metadata",
            "initial_time_only": True,
            "normalization": "binary prespecified condition",
        }
    ]
    for offset, column in enumerate(metabolic_columns, start=1):
        observed_train = [
            metabolic[group][column]
            for group in train_groups
            if group in metabolic and metabolic[group].get(column) is not None
        ]
        if len(observed_train) < 2:
            raise ValueError(
                f"Metabolic cue {column!r} needs at least two observed training groups."
            )
        lower, upper = np.quantile(np.asarray(observed_train, dtype=float), [0.05, 0.95])
        if not upper > lower:
            upper = lower + 1.0
        for group in groups:
            raw = metabolic.get(group, {}).get(column)
            if raw is None:
                continue
            values[group][offset] = float(np.clip((raw - lower) / (upper - lower), 0.0, 1.0))
            masks[group][offset] = 1.0
        provenance.append(
            {
                "name": f"metabolic:{column}",
                "measurement_level": "subject",
                "source": str(column),
                "initial_time_only": True,
                "normalization": "training-subject 5th/95th percentile to [0,1]; held-out clipped",
                "missing_values": "masked to zero; never imputed",
            }
        )
    return names, values, masks, provenance


def build(args: argparse.Namespace) -> Dict[str, object]:
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    genes_all = read_features(args.genes, args.gene_column)
    peaks_all = read_features(args.peaks, args.peak_column)
    if len(genes_all) != len(set(genes_all)):
        raise ValueError("Gene feature names must be unique before dataset construction.")
    if len(peaks_all) != len(set(peaks_all)):
        raise ValueError("Peak identifiers must be unique before dataset construction.")
    barcodes = read_barcodes(args.barcodes)
    metadata = read_metadata(args.metadata, args.cell_column)
    missing_metadata = [barcode for barcode in barcodes if barcode not in metadata]
    if missing_metadata:
        raise ValueError(f"{len(missing_metadata)} matrix barcodes lack metadata; first={missing_metadata[0]}")
    rna = read_matrix(args.rna_mtx, len(genes_all), len(barcodes))
    atac = read_matrix(args.atac_mtx, len(peaks_all), len(barcodes))

    treated_values = {value.strip().lower() for value in args.treated_values.split(",") if value.strip()}
    by_group_time: MutableMapping[Tuple[str, str], List[int]] = defaultdict(list)
    group_condition_label: Dict[str, str] = {}
    group_condition_value: Dict[str, float] = {}
    for index, barcode in enumerate(barcodes):
        row = metadata[barcode]
        group = row.get(args.group_column, "").strip()
        time = row.get(args.time_column, "").strip()
        condition = row.get(args.condition_column, "").strip()
        if not group or not time or not condition:
            continue
        previous = group_condition_label.setdefault(group, condition)
        if previous != condition:
            raise ValueError(f"Biological group {group} has inconsistent condition labels.")
        group_condition_value[group] = float(condition.lower() in treated_values)
        if time in {args.initial_time, args.target_time}:
            by_group_time[(group, time)].append(index)
    groups = sorted(
        group
        for group in group_condition_label
        if len(by_group_time[(group, args.initial_time)]) >= args.min_cells_per_time
        and len(by_group_time[(group, args.target_time)]) >= args.min_cells_per_time
    )
    if len(groups) < 3:
        raise ValueError("Fewer than three biological groups have enough initial and target cells.")
    split = (
        _read_split(args.split_json, groups)
        if args.split_json is not None
        else _split_groups({group: group_condition_label[group] for group in groups}, args.seed)
    )

    train_initial = np.concatenate(
        [np.asarray(by_group_time[(group, args.initial_time)], dtype=int) for group in split["train"]]
    )
    train_target = np.concatenate(
        [np.asarray(by_group_time[(group, args.target_time)], dtype=int) for group in split["train"]]
    )
    # Select genes in standard log1p(CP10K) space, but store CP10K targets.
    # The temporal objective applies log1p exactly once.  Storing the already
    # logged matrix here previously caused an accidental second log1p in the
    # trainer and compressed the biological dynamic range.
    rna_cp10k = _library_normalize(rna)
    rna_logged = rna_cp10k.copy()
    rna_logged.data = np.log1p(rna_logged.data)
    gene_variance = _sparse_variance(rna_logged[train_target])
    train_binary_atac = atac[train_initial].astype(bool).astype(np.float32)
    peak_prevalence = np.asarray(train_binary_atac.mean(axis=0)).ravel()
    peak_variance = peak_prevalence * (1.0 - peak_prevalence)

    metabolic_columns = [value for value in args.metabolic_columns.split(",") if value]
    metabolic = _metabolic_values(
        args.metabolic_covariates,
        args.metabolic_group_column,
        metabolic_columns,
    )
    candidate_cues, group_cues, group_masks, cue_provenance = _normalize_group_cues(
        groups,
        split["train"],
        group_condition_value,
        args.condition_cue_name,
        metabolic,
        metabolic_columns,
    )

    evidence = load_evidence(args)
    priors, registry = compile_priors(
        evidence,
        genes_all,
        peaks_all,
        gene_variance,
        peak_variance,
        candidate_cues,
        args.max_genes,
        args.max_peaks,
        args.max_tfs,
    )
    cue_indices = [candidate_cues.index(name) for name in registry["cues"]]
    cue_provenance_by_name = {record["name"]: record for record in cue_provenance}
    cue_provenance = [cue_provenance_by_name[name] for name in registry["cues"]]
    gene_indices = np.asarray([{name: index for index, name in enumerate(genes_all)}[name] for name in registry["genes"]])
    peak_indices = np.asarray([{name: index for index, name in enumerate(peaks_all)}[name] for name in registry["peaks"]])

    initial_atac = []
    initial_cues = []
    initial_cue_mask = []
    initial_transition: List[str] = []
    target_rna = []
    target_atac = []
    target_transition: List[str] = []
    initial_rna = []
    transitions = []
    for group in groups:
        transition_id = f"{_safe_id(group)}_{_safe_id(args.initial_time)}_to_{_safe_id(args.target_time)}"
        initial_index = np.asarray(by_group_time[(group, args.initial_time)], dtype=int)
        target_index = np.asarray(by_group_time[(group, args.target_time)], dtype=int)
        initial_atac.append(atac[initial_index][:, peak_indices].astype(bool).astype(np.float32).toarray())
        repeated_cues = np.repeat(group_cues[group][None, cue_indices], len(initial_index), axis=0)
        repeated_masks = np.repeat(group_masks[group][None, cue_indices], len(initial_index), axis=0)
        initial_cues.append(repeated_cues)
        initial_cue_mask.append(repeated_masks)
        initial_transition.extend([transition_id] * len(initial_index))
        target_rna.append(rna_cp10k[target_index][:, gene_indices].toarray().astype(np.float32))
        target_atac.append(atac[target_index][:, peak_indices].astype(bool).astype(np.float32).toarray())
        target_transition.extend([transition_id] * len(target_index))
        if args.include_initial_rna:
            initial_rna.append(rna_cp10k[initial_index][:, gene_indices].toarray().astype(np.float32))
        transitions.append(
            {
                "transition_id": transition_id,
                "group_id": group,
                "horizon": float(args.horizon),
                "terminal": False,
                "condition": group_condition_label[group],
                "initial_time": args.initial_time,
                "target_time": args.target_time,
                "initial_cells": len(initial_index),
                "target_cells": len(target_index),
            }
        )

    observation_values: Dict[str, np.ndarray] = {
        "initial_atac": np.concatenate(initial_atac).astype(np.float32),
        "initial_cues": np.concatenate(initial_cues).astype(np.float32),
        "initial_cue_mask": np.concatenate(initial_cue_mask).astype(np.float32),
        "initial_transition": np.asarray(initial_transition, dtype="U128"),
        "target_rna": np.concatenate(target_rna).astype(np.float32),
        "target_atac": np.concatenate(target_atac).astype(np.float32),
        "target_transition": np.asarray(target_transition, dtype="U128"),
    }
    if args.include_initial_rna:
        observation_values["initial_rna"] = np.concatenate(initial_rna).astype(np.float32)
    np.savez_compressed(output / "observations.npz", **observation_values)
    np.savez_compressed(output / "priors.npz", **priors)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "alignment_mode": "distribution",
        "rna_representation": "cp10k_library_size_10000",
        "initial_feature_names": ["ATAC_peaks"]
        + [f"measured_cue:{name}" for name in registry["cues"]]
        + (["time_zero_RNA"] if args.include_initial_rna else []),
        "cue_names": registry["cues"],
        "cue_provenance": cue_provenance,
        "priors_fit_groups": split["train"],
        "split_groups": split,
        "transitions": transitions,
        "dataset": {
            "primary_multiome": "GSE240061",
            "tissue_contact_scaffold": "GSE126100",
            "genome_build": args.genome_build,
            "scope": "transient pre/post perturbation prediction; endpoints are not declared attractors",
        },
        "feature_selection": {
            "fit_groups": split["train"],
            "rna": (
                "feature ranking uses log1p(CP10K) on training-group target cells; "
                "stored RNA targets are CP10K so the trainer applies log1p once"
            ),
            "atac": "binary accessibility variance on training-group initial cells only",
            "regulatory_intersection": "open peak x localized TF motif x peak-gene contact x signed TF-gene support",
        },
        "leakage_controls": {
            "split_before_feature_selection": True,
            "cell_type_input": False,
            "pseudotime_input": False,
            "target_state_input": False,
            "fabricated_cell_pairs": False,
            "initial_rna_included": bool(args.include_initial_rna),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "built",
        "groups": len(groups),
        "split_groups": split,
        "initial_cells": int(observation_values["initial_atac"].shape[0]),
        "target_cells": int(observation_values["target_rna"].shape[0]),
        "genes": len(registry["genes"]),
        "peaks": len(registry["peaks"]),
        "tfs": len(registry["tfs"]),
        "signals": len(registry["signals"]),
        "cues": registry["cues"],
        "excluded_cues_without_complete_signaling_path": registry[
            "excluded_cues_without_complete_signaling_path"
        ],
        "edge_counts": registry["edge_counts"],
        "claim_boundary": "This build supports grouped transient prediction, not an attractor claim.",
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--rna-mtx", type=Path, required=True)
    result.add_argument("--atac-mtx", type=Path, required=True)
    result.add_argument("--genes", type=Path, required=True)
    result.add_argument("--peaks", type=Path, required=True)
    result.add_argument("--barcodes", type=Path, required=True)
    result.add_argument("--metadata", type=Path, required=True)
    result.add_argument("--peak-gene-links", type=Path, required=True)
    result.add_argument("--motif-hits", type=Path, required=True)
    result.add_argument("--tf-gene-edges", type=Path, required=True)
    result.add_argument("--signaling-edges", type=Path, required=True)
    result.add_argument("--tf-peak-effects", type=Path)
    result.add_argument("--metabolic-covariates", type=Path)
    result.add_argument("--metabolic-columns", default="")
    result.add_argument("--metabolic-group-column", default="subject")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--cell-column", default="cell_id")
    result.add_argument("--group-column", default="subject")
    result.add_argument("--time-column", default="timepoint")
    result.add_argument("--condition-column", default="condition")
    result.add_argument("--initial-time", default="pre")
    result.add_argument("--target-time", default="post_3.5h")
    result.add_argument("--treated-values", default="exercise")
    result.add_argument("--condition-cue-name", default="exercise")
    result.add_argument("--horizon", type=float, default=3.5)
    result.add_argument("--genome-build", default="GRCh38")
    result.add_argument("--gene-column", type=int, default=0)
    result.add_argument("--peak-column", type=int, default=0)
    result.add_argument("--max-genes", type=int, default=400)
    result.add_argument("--max-peaks", type=int, default=1000)
    result.add_argument("--max-tfs", type=int, default=64)
    result.add_argument("--min-cells-per-time", type=int, default=20)
    result.add_argument("--split-json", type=Path)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--include-initial-rna", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.horizon <= 0 or not math.isfinite(args.horizon):
        raise ValueError("--horizon must be finite and positive")
    for name in ("max_genes", "max_peaks", "max_tfs", "min_cells_per_time"):
        if getattr(args, name) < 2:
            raise ValueError(f"--{name.replace('_', '-')} must be at least two")
    if args.max_genes < args.max_tfs:
        raise ValueError("--max-genes must be at least --max-tfs so every TF gene is retained.")
    if bool(args.metabolic_covariates) != bool(args.metabolic_columns):
        raise ValueError("Supply both --metabolic-covariates and --metabolic-columns, or neither.")
    report = build(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
