"""Compile a versioned, leakage-safe biological scaffold for WLD muscle data.

This compiler turns four external evidence layers into the canonical tables
consumed by ``build_wld_muscle_exercise_dataset.py``:

* GSE126100 promoter-capture Hi-C: GRCh38 peak-to-promoter topology;
* JASPAR 2024 CORE vertebrate motifs: TF binding feasibility in open peaks;
* CollecTRI: signed TF-to-gene and TF-to-TF regulatory support;
* OmniPath core: signed cue-to-signal-to-TF paths.

The only cohort-derived operation is ranking contact-linked peaks from ATAC
counts in *training subjects at the initial time*.  Validation/test counts,
RNA, cell labels, integrated embeddings, and future observations are never
read for prior selection.  The experimental ``exercise`` indicator enters by
three explicit, reviewable hypothesis bridges (AMPK, CaMKII, and p38); these
are not misrepresented as measured protein activities.

Inputs are intentionally downloaded outside this script so the caller can
resume large transfers and retain exact source files.  Every input is hashed
in ``prior_manifest.json``.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


GENCODE_RELEASE = "44"
JASPAR_RELEASE = "2024"
COORDINATE_SYSTEM = "GRCh38/hg38, BED-style 0-based half-open intervals"
EXERCISE_ENTRY_SIGNALS = ("PRKAA1", "CAMK2D", "MAPK14")
SAFE_GENE = re.compile(r"^[A-Za-z0-9.-]+$")
PEAK_PATTERN = re.compile(r"^(chr(?:[0-9]+|X|Y|M|MT))[:-](\d+)[-:](\d+)$", re.I)


@dataclass(frozen=True)
class Peak:
    peak_id: str
    chrom: str
    start: int
    end: int
    index: int


@dataclass(frozen=True)
class MotifBlock:
    motif_id: str
    tf: str
    lines: Tuple[str, ...]


def open_text(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", newline="")
    )


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def delimiter_for(sample: str) -> str:
    return "\t" if sample.count("\t") >= sample.count(",") else ","


def read_dict_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open_text(path) as handle:
        sample = handle.read(16384)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter_for(sample))
        fields = list(reader.fieldnames or ())
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return fields, rows


def first_column(path: Path) -> List[str]:
    values: List[str] = []
    with open_text(path) as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line:
                values.append(line.split("\t", 1)[0])
    return values


def normalize_chrom(value: str) -> str:
    value = value.strip()
    if not value.lower().startswith("chr"):
        value = "chr" + value
    if value == "chrMT":
        value = "chrM"
    return value


def parse_peak(line: str, index: int) -> Peak:
    fields = line.rstrip("\r\n").split("\t")
    peak_id = fields[0].strip()
    if len(fields) >= 4 and fields[1].lower().startswith("chr"):
        chrom, start, end = fields[1], fields[2], fields[3]
    elif len(fields) >= 3 and fields[0].lower().startswith("chr"):
        chrom, start, end = fields[0], fields[1], fields[2]
        peak_id = f"{chrom}-{start}-{end}"
    else:
        match = PEAK_PATTERN.fullmatch(peak_id)
        if match is None:
            raise ValueError(
                f"Cannot parse peak {peak_id!r}; expected chrN-start-end, "
                "chrN:start-end, or explicit chrom/start/end columns."
            )
        chrom, start, end = match.groups()
    start_i, end_i = int(start), int(end)
    if start_i < 0 or end_i <= start_i:
        raise ValueError(f"Invalid peak interval: {peak_id}")
    return Peak(peak_id, normalize_chrom(chrom), start_i, end_i, index)


def read_peaks(path: Path) -> List[Peak]:
    peaks: List[Peak] = []
    with open_text(path) as handle:
        for index, line in enumerate(handle):
            if line.strip():
                peaks.append(parse_peak(line, index))
    if not peaks:
        raise ValueError("No ATAC peaks were found.")
    identifiers = [peak.peak_id for peak in peaks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("ATAC peak identifiers must be unique.")
    return peaks


def parse_gtf_attributes(value: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        key, _, raw = item.partition(" ")
        result[key] = raw.strip().strip('"')
    return result


def read_protein_coding_tss(
    path: Path, allowed_genes: Iterable[str]
) -> Dict[str, Tuple[Tuple[int, str], ...]]:
    allowed = set(allowed_genes)
    by_chrom: MutableMapping[str, List[Tuple[int, str]]] = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attrs = parse_gtf_attributes(fields[8])
            gene = attrs.get("gene_name", "")
            gene_type = attrs.get("gene_type", attrs.get("gene_biotype", ""))
            if gene not in allowed or gene_type != "protein_coding":
                continue
            start, end = int(fields[3]), int(fields[4])
            tss = start - 1 if fields[6] == "+" else end - 1
            by_chrom[normalize_chrom(fields[0])].append((tss, gene))
    result = {
        chrom: tuple(sorted(set(records))) for chrom, records in by_chrom.items()
    }
    if sum(map(len, result.values())) < 10000:
        raise ValueError(
            "Too few GENCODE protein-coding TSS records matched the RNA genes; "
            "check gene symbols and genome build."
        )
    return result


def tss_in_interval(
    tss_by_chrom: Mapping[str, Sequence[Tuple[int, str]]],
    positions_by_chrom: Mapping[str, Sequence[int]],
    chrom: str,
    start: int,
    end: int,
) -> Tuple[str, ...]:
    records = tss_by_chrom.get(chrom, ())
    positions = positions_by_chrom.get(chrom, ())
    left = bisect.bisect_left(positions, start)
    right = bisect.bisect_left(positions, end)
    return tuple(sorted({gene for _, gene in records[left:right]}))


class PeakIndex:
    def __init__(self, peaks: Sequence[Peak]):
        self.records: Dict[str, Tuple[Peak, ...]] = {}
        self.starts: Dict[str, Tuple[int, ...]] = {}
        self.max_width: Dict[str, int] = {}
        grouped: MutableMapping[str, List[Peak]] = defaultdict(list)
        for peak in peaks:
            grouped[peak.chrom].append(peak)
        for chrom, records in grouped.items():
            ordered = tuple(sorted(records, key=lambda peak: (peak.start, peak.end)))
            self.records[chrom] = ordered
            self.starts[chrom] = tuple(peak.start for peak in ordered)
            self.max_width[chrom] = max(peak.end - peak.start for peak in ordered)

    def overlap(self, chrom: str, start: int, end: int) -> Tuple[Peak, ...]:
        records = self.records.get(chrom, ())
        starts = self.starts.get(chrom, ())
        if not records:
            return ()
        left = bisect.bisect_left(starts, max(0, start - self.max_width[chrom]))
        right = bisect.bisect_left(starts, end)
        return tuple(peak for peak in records[left:right] if peak.end > start)


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result > 0 else 0.0


def compile_contacts(
    pchic_path: Path,
    tss_by_chrom: Mapping[str, Sequence[Tuple[int, str]]],
    peak_index: PeakIndex,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, object]]:
    fields, rows = read_dict_rows(pchic_path)
    required = {
        "chr_first", "start_first", "end_first",
        "chr_second", "start_second", "end_second",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"GSE126100 table is missing columns: {missing}")
    score_columns = [name for name in fields if name not in required]
    if not score_columns:
        raise ValueError("GSE126100 table has no interaction-count columns.")

    links: Dict[Tuple[str, str], float] = {}
    tss_positions = {
        chrom: tuple(position for position, _ in records)
        for chrom, records in tss_by_chrom.items()
    }
    promoter_annotated = 0
    directional_assignments = 0
    for row in rows:
        first = (
            normalize_chrom(row["chr_first"]),
            int(row["start_first"]),
            int(row["end_first"]),
        )
        second = (
            normalize_chrom(row["chr_second"]),
            int(row["start_second"]),
            int(row["end_second"]),
        )
        if first[0] != second[0]:
            continue
        genes_first = tss_in_interval(tss_by_chrom, tss_positions, *first)
        genes_second = tss_in_interval(tss_by_chrom, tss_positions, *second)
        if not genes_first and not genes_second:
            continue
        promoter_annotated += 1
        counts = [positive_float(row.get(column, "")) for column in score_columns]
        score = math.log1p(sum(counts) / max(len(counts), 1))
        if score <= 0:
            score = 1.0
        assignments = []
        if genes_first:
            assignments.append((second, genes_first))
        if genes_second:
            assignments.append((first, genes_second))
        for distal, genes in assignments:
            distal_peaks = peak_index.overlap(*distal)
            if not distal_peaks:
                continue
            directional_assignments += 1
            for peak in distal_peaks:
                for gene in genes:
                    key = (peak.peak_id, gene)
                    links[key] = max(links.get(key, 0.0), score)

    promoter_fraction = promoter_annotated / max(len(rows), 1)
    unique_peaks = {peak for peak, _ in links}
    unique_genes = {gene for _, gene in links}
    if promoter_fraction < 0.20:
        raise RuntimeError(
            f"Only {promoter_fraction:.1%} of GSE126100 interactions contained a "
            "GENCODE v44 protein-coding TSS. This is incompatible with the "
            "frozen GRCh38 assumption; refusing to guess or silently liftover."
        )
    if len(unique_peaks) < 100 or len(unique_genes) < 100:
        raise RuntimeError(
            "Too few GSE126100 contacts overlap the GSE240061 peaks/genes; "
            "check coordinate and peak formats."
        )
    report = {
        "interactions_total": len(rows),
        "interactions_with_protein_coding_tss": promoter_annotated,
        "promoter_annotation_fraction": promoter_fraction,
        "directional_contact_assignments_with_atac_overlap": directional_assignments,
        "peak_gene_links": len(links),
        "linked_peaks": len(unique_peaks),
        "linked_genes": len(unique_genes),
        "contact_score": "log1p(mean pCHi-C count across six external muscle libraries)",
    }
    return links, report


def read_split(path: Path) -> Dict[str, Tuple[str, ...]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    split = {
        name: tuple(map(str, value[name]))
        for name in ("train", "validation", "test")
    }
    flat = [group for groups in split.values() for group in groups]
    if len(flat) != len(set(flat)) or any(not groups for groups in split.values()):
        raise ValueError("split.json must contain three nonempty, disjoint group lists.")
    return split


def training_initial_columns(
    barcodes_path: Path,
    metadata_path: Path,
    split: Mapping[str, Sequence[str]],
    initial_time: str,
) -> Tuple[set[int], Dict[str, object]]:
    barcodes = first_column(barcodes_path)
    fields, rows = read_dict_rows(metadata_path)
    cell_column = "cell_id" if "cell_id" in fields else fields[0]
    required = {cell_column, "subject", "timepoint"}
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"metadata.tsv is missing columns: {missing}")
    metadata = {row[cell_column]: row for row in rows}
    train_groups = set(split["train"])
    selected = {
        index
        for index, barcode in enumerate(barcodes)
        if barcode in metadata
        and metadata[barcode]["subject"] in train_groups
        and metadata[barcode]["timepoint"] == initial_time
    }
    if len(selected) < 100:
        raise ValueError("Fewer than 100 training-subject initial-time ATAC cells found.")
    return selected, {
        "training_groups": sorted(train_groups),
        "initial_time": initial_time,
        "training_initial_cells": len(selected),
        "validation_and_test_cells_used_for_ranking": 0,
    }


def stream_contact_peak_variance(
    matrix_path: Path,
    candidate_rows: Mapping[int, str],
    selected_columns: set[int],
    expected_rows: int,
    expected_columns: int,
) -> Dict[str, float]:
    row_to_local = {row: index for index, row in enumerate(sorted(candidate_rows))}
    sums = [0.0] * len(row_to_local)
    sum_squares = [0.0] * len(row_to_local)
    with open_text(matrix_path) as handle:
        header = handle.readline().strip()
        if not header.startswith("%%MatrixMarket matrix coordinate"):
            raise ValueError("ATAC matrix must be Matrix Market coordinate format.")
        for line in handle:
            if line.startswith("%"):
                continue
            n_rows, n_cols, _ = map(int, line.split())
            break
        else:
            raise ValueError("ATAC Matrix Market dimension line is missing.")
        if (n_rows, n_cols) != (expected_rows, expected_columns):
            raise ValueError(
                f"ATAC shape {(n_rows, n_cols)} does not match "
                f"features x cells {(expected_rows, expected_columns)}."
            )
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) < 3:
                continue
            row = int(fields[0]) - 1
            column = int(fields[1]) - 1
            local = row_to_local.get(row)
            if local is None or column not in selected_columns:
                continue
            value = float(fields[2])
            sums[local] += value
            sum_squares[local] += value * value
            if line_number % 20_000_000 == 0:
                print(f"   streamed {line_number:,} ATAC non-zero entries", flush=True)
    n = float(len(selected_columns))
    variances: Dict[str, float] = {}
    for row, local in row_to_local.items():
        mean = sums[local] / n
        variance = max(sum_squares[local] / n - mean * mean, 0.0)
        variances[candidate_rows[row]] = variance
    return variances


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def genesymbol(row: Mapping[str, str], side: str) -> str:
    return row.get(f"{side}_genesymbol", row.get(side, "")).strip()


def signed_omnipath_row(row: Mapping[str, str]) -> Optional[int]:
    directed = row.get("consensus_direction", row.get("is_directed", "1"))
    if directed and not parse_bool(directed):
        return None
    stimulation = parse_bool(
        row.get("consensus_stimulation", row.get("is_stimulation", "0"))
    )
    inhibition = parse_bool(
        row.get("consensus_inhibition", row.get("is_inhibition", "0"))
    )
    if stimulation == inhibition:
        return None
    return 1 if stimulation else -1


def evidence_score(row: Mapping[str, str]) -> float:
    effort = positive_float(row.get("curation_effort", ""))
    references = [item for item in row.get("references", "").split(";") if item]
    return max(1.0, effort, float(len(references)))


def compile_collectri(
    path: Path, genes: Iterable[str]
) -> Tuple[Dict[Tuple[str, str], Tuple[int, float, str, str]], Dict[str, object]]:
    gene_set = set(genes)
    _, rows = read_dict_rows(path)
    aggregated: MutableMapping[Tuple[str, str], List[Tuple[int, float, str, str]]] = defaultdict(list)
    for row in rows:
        source, target = genesymbol(row, "source"), genesymbol(row, "target")
        sign = signed_omnipath_row(row)
        if (
            sign is None
            or source not in gene_set
            or target not in gene_set
            or SAFE_GENE.fullmatch(source) is None
        ):
            continue
        aggregated[(source, target)].append(
            (
                sign,
                evidence_score(row),
                row.get("sources", "CollecTRI"),
                row.get("references", ""),
            )
        )
    edges: Dict[Tuple[str, str], Tuple[int, float, str, str]] = {}
    ambiguous = 0
    for key, records in aggregated.items():
        signed_score = sum(sign * score for sign, score, _, _ in records)
        if signed_score == 0:
            ambiguous += 1
            continue
        sign = 1 if signed_score > 0 else -1
        sources = sorted({item for _, _, value, _ in records for item in value.split(";") if item})
        references = sorted({item for _, _, _, value in records for item in value.split(";") if item})
        edges[key] = (sign, abs(signed_score), ";".join(sources), ";".join(references))
    sources = {source for source, _ in edges}
    if len(edges) < 1000 or len(sources) < 50:
        raise RuntimeError("Too few signed single-TF CollecTRI edges matched RNA genes.")
    return edges, {
        "raw_records": len(rows),
        "signed_single_tf_edges": len(edges),
        "tf_sources": len(sources),
        "ambiguous_aggregate_edges_removed": ambiguous,
        "complex_sources_split": False,
    }


def parse_jaspar_meme(
    path: Path, candidate_tfs: Iterable[str]
) -> Tuple[List[str], List[MotifBlock]]:
    candidates = {name.upper(): name for name in candidate_tfs}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header: List[str] = []
    blocks: List[List[str]] = []
    current: Optional[List[str]] = None
    for line in lines:
        if line.startswith("MOTIF "):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is None:
            header.append(line)
        else:
            current.append(line)
    if current is not None:
        blocks.append(current)
    selected: List[MotifBlock] = []
    for block in blocks:
        fields = block[0].strip().split(maxsplit=2)
        motif_id = fields[1]
        alt = fields[2].strip() if len(fields) > 2 else ""
        if any(token in alt for token in ("::", "/", "(", ")")):
            continue
        tf = candidates.get(alt.upper())
        if tf is not None:
            selected.append(MotifBlock(motif_id, tf, tuple(block)))
    if len(selected) < 50:
        raise RuntimeError(
            "Fewer than 50 single-TF JASPAR motifs overlap signed CollecTRI TFs."
        )
    return header, selected


def write_selected_meme(
    path: Path, header: Sequence[str], motifs: Sequence[MotifBlock]
) -> Dict[str, str]:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(header)
        for motif in motifs:
            handle.writelines(motif.lines)
    return {motif.motif_id: motif.tf for motif in motifs}


def run_checked(command: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    print("Running:", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def scan_motifs(
    peaks: Sequence[Peak],
    genome_fasta: Path,
    meme_path: Path,
    motif_to_tf: Mapping[str, str],
    output: Path,
    p_threshold: float,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, object]]:
    bedtools = shutil.which("bedtools")
    fimo = shutil.which("fimo")
    if bedtools is None or fimo is None:
        raise RuntimeError("bedtools and FIMO must be installed and present on PATH.")
    bed_path = output / "candidate_peaks.bed"
    fasta_path = output / "candidate_peaks.fa"
    fimo_path = output / "fimo_hits.tsv"
    sequence_to_peak: Dict[str, str] = {}
    with bed_path.open("w", encoding="utf-8") as handle:
        for index, peak in enumerate(peaks):
            sequence = f"peak_{index:06d}"
            sequence_to_peak[sequence] = peak.peak_id
            handle.write(f"{peak.chrom}\t{peak.start}\t{peak.end}\t{sequence}\n")
    run_checked(
        [bedtools, "getfasta", "-fi", genome_fasta, "-bed", bed_path, "-nameOnly", "-fo", fasta_path]
    )
    with fimo_path.open("w", encoding="utf-8") as handle:
        run_checked(
            [fimo, "--text", "--verbosity", "1", "--thresh", str(p_threshold), meme_path, fasta_path],
            stdout=handle,
        )
    hits: Dict[Tuple[str, str], float] = {}
    with fimo_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(
            (line for line in handle if line.strip() and not line.startswith("#")),
            delimiter="\t",
        )
        for row in reader:
            motif_id = row.get("motif_id", "")
            sequence = row.get("sequence_name", "").split("::", 1)[0]
            tf = motif_to_tf.get(motif_id)
            peak = sequence_to_peak.get(sequence)
            if tf is None or peak is None:
                continue
            pvalue = max(float(row["p-value"]), 1e-300)
            score = -math.log10(pvalue)
            key = (peak, tf)
            hits[key] = max(hits.get(key, 0.0), score)
    if len(hits) < 100:
        raise RuntimeError(
            "Fewer than 100 localized motif hits passed FIMO; check FASTA build and threshold."
        )
    return hits, {
        "candidate_peaks_scanned": len(peaks),
        "single_tf_motifs_scanned": len(motif_to_tf),
        "localized_peak_tf_pairs": len(hits),
        "fimo_p_threshold": p_threshold,
        "score": "-log10(best FIMO p-value per peak/TF)",
    }


def compile_signaling(
    path: Path,
    target_tfs: Iterable[str],
    max_depth: int,
) -> Tuple[List[Tuple[str, str, str, str, int, float, str, str]], Dict[str, object]]:
    targets = set(target_tfs)
    _, rows = read_dict_rows(path)
    aggregate: MutableMapping[Tuple[str, str], List[Tuple[int, float, str, str]]] = defaultdict(list)
    for row in rows:
        source, target = genesymbol(row, "source"), genesymbol(row, "target")
        sign = signed_omnipath_row(row)
        if (
            sign is None
            or SAFE_GENE.fullmatch(source) is None
            or SAFE_GENE.fullmatch(target) is None
        ):
            continue
        aggregate[(source, target)].append(
            (sign, evidence_score(row), row.get("sources", "OmniPath"), row.get("references", ""))
        )
    edges: Dict[Tuple[str, str], Tuple[int, float, str, str]] = {}
    forward: MutableMapping[str, set[str]] = defaultdict(set)
    reverse: MutableMapping[str, set[str]] = defaultdict(set)
    for key, records in aggregate.items():
        signed_score = sum(sign * score for sign, score, _, _ in records)
        if signed_score == 0:
            continue
        source, target = key
        sign = 1 if signed_score > 0 else -1
        sources = sorted({item for _, _, value, _ in records for item in value.split(";") if item})
        references = sorted({item for _, _, _, value in records for item in value.split(";") if item})
        edges[key] = (sign, abs(signed_score), ";".join(sources), ";".join(references))
        forward[source].add(target)
        reverse[target].add(source)

    distance: Dict[str, int] = {}
    queue = deque((seed, 0) for seed in EXERCISE_ENTRY_SIGNALS)
    while queue:
        node, depth = queue.popleft()
        if node in distance and distance[node] <= depth:
            continue
        distance[node] = depth
        if depth < max_depth:
            queue.extend((target, depth + 1) for target in forward[node])
    reachable_targets = targets.intersection(distance)
    if not reachable_targets:
        raise RuntimeError(
            "No selected CollecTRI/JASPAR TF is reachable from the prespecified "
            "exercise entry signals in signed OmniPath."
        )
    reverse_distance: Dict[str, int] = {}
    queue = deque((target, 0) for target in reachable_targets)
    while queue:
        node, depth = queue.popleft()
        if node in reverse_distance and reverse_distance[node] <= depth:
            continue
        reverse_distance[node] = depth
        if depth < max_depth:
            queue.extend((source, depth + 1) for source in reverse[node])

    retained: List[Tuple[str, str, str, str, int, float, str, str]] = []
    for (source, target), (sign, score, sources, references) in sorted(edges.items()):
        if source not in distance or target not in reverse_distance:
            continue
        if distance[source] + 1 + reverse_distance[target] > max_depth:
            continue
        target_type = "tf" if target in reachable_targets else "signal"
        retained.append(
            (source, target, "signal", target_type, sign, score, sources, references)
        )
    for seed in EXERCISE_ENTRY_SIGNALS:
        if seed in distance and seed in reverse_distance:
            retained.append(
                (
                    "exercise", seed, "cue", "signal", 1, 1.0,
                    "prespecified_exercise_entry_hypothesis", "",
                )
            )
    if not any(row[2] == "cue" for row in retained):
        raise RuntimeError("No productive exercise cue bridge survived path filtering.")
    return retained, {
        "raw_records": len(rows),
        "signed_directed_aggregated_edges": len(edges),
        "maximum_path_depth": max_depth,
        "exercise_entry_signals": list(EXERCISE_ENTRY_SIGNALS),
        "reachable_candidate_tfs": sorted(reachable_targets),
        "retained_edges_including_cue_bridges": len(retained),
        "cue_bridge_measurement_status": (
            "exercise is measured experimental assignment; downstream kinase activity "
            "is a prespecified mechanistic hypothesis, not a measured protein input"
        ),
    }


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Sequence[object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def build(args: argparse.Namespace) -> Dict[str, object]:
    export = args.export.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    paths = {
        "genes": export / "genes.tsv",
        "peaks": export / "peaks.tsv",
        "barcodes": export / "barcodes.tsv",
        "metadata": export / "metadata.tsv",
        "split": export / "split.json",
        "atac": export / "atac.mtx.gz",
        "pchic": args.pchic.resolve(),
        "gencode_gtf": args.gencode_gtf.resolve(),
        "collectri": args.collectri.resolve(),
        "omnipath": args.omnipath.resolve(),
        "jaspar_meme": args.jaspar_meme.resolve(),
        "genome_fasta": args.genome_fasta.resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    print("1. Reading exported features and frozen subject split...", flush=True)
    genes = first_column(paths["genes"])
    peaks = read_peaks(paths["peaks"])
    barcodes = first_column(paths["barcodes"])
    split = read_split(paths["split"])
    training_columns, selection_report = training_initial_columns(
        paths["barcodes"], paths["metadata"], split, args.initial_time
    )

    print("2. Annotating GRCh38 promoter-capture Hi-C contacts...", flush=True)
    tss = read_protein_coding_tss(paths["gencode_gtf"], genes)
    contact_links, contact_report = compile_contacts(
        paths["pchic"], tss, PeakIndex(peaks)
    )

    print("3. Ranking only contact-linked peaks on training/pre ATAC...", flush=True)
    peak_by_id = {peak.peak_id: peak for peak in peaks}
    candidate_rows = {
        peak_by_id[peak_id].index: peak_id
        for peak_id, _ in contact_links
        if peak_id in peak_by_id
    }
    variance = stream_contact_peak_variance(
        paths["atac"], candidate_rows, training_columns, len(peaks), len(barcodes)
    )
    selected_ids = set(
        sorted(
            variance,
            key=lambda peak_id: (-variance[peak_id], peak_id),
        )[: args.max_candidate_peaks]
    )
    contact_links = {
        key: score for key, score in contact_links.items() if key[0] in selected_ids
    }
    selected_peaks = sorted(
        (peak_by_id[peak_id] for peak_id in selected_ids), key=lambda peak: peak.index
    )
    selection_report.update(
        {
            "contact_linked_peaks_before_ranking": len(variance),
            "selected_candidate_peaks": len(selected_peaks),
            "ranking_statistic": "raw-count variance across training-subject pre cells only",
        }
    )

    print("4. Compiling signed CollecTRI edges and JASPAR motif subset...", flush=True)
    collectri, collectri_report = compile_collectri(paths["collectri"], genes)
    linked_genes = {gene for _, gene in contact_links}
    candidate_tfs = {
        source for source, target in collectri if target in linked_genes
    }
    meme_header, motifs = parse_jaspar_meme(paths["jaspar_meme"], candidate_tfs)
    selected_meme = output / "selected_jaspar2024_single_tf.meme"
    motif_to_tf = write_selected_meme(selected_meme, meme_header, motifs)

    print("5. Scanning localized motifs in selected contact-linked peaks...", flush=True)
    motif_hits, motif_report = scan_motifs(
        selected_peaks,
        paths["genome_fasta"],
        selected_meme,
        motif_to_tf,
        output,
        args.fimo_p_threshold,
    )
    genes_by_peak: MutableMapping[str, set[str]] = defaultdict(set)
    for peak, gene in contact_links:
        genes_by_peak[peak].add(gene)
    raw_motif_pairs = len(motif_hits)
    motif_hits = {
        (peak, tf): score
        for (peak, tf), score in motif_hits.items()
        if any((tf, gene) in collectri for gene in genes_by_peak[peak])
    }
    motif_report["raw_localized_peak_tf_pairs"] = raw_motif_pairs
    motif_report["productive_motif_contact_collectri_pairs"] = len(motif_hits)
    if len(motif_hits) < 100:
        raise RuntimeError(
            "Fewer than 100 motif hits survived the required motif x contact x "
            "signed-CollecTRI intersection."
        )
    tf_local_score: MutableMapping[str, int] = defaultdict(int)
    for _, tf in motif_hits:
        tf_local_score[tf] += 1
    expected_tfs = sorted(
        tf_local_score, key=lambda tf: (-tf_local_score[tf], tf)
    )[: args.expected_max_tfs]
    expected_tf_set = set(expected_tfs)
    localized_targets = {
        (tf, gene)
        for peak, tf in motif_hits
        for gene in genes_by_peak[peak]
    }
    expected_circuit = {
        (source, target)
        for source, target in collectri
        if source in expected_tf_set
        and target in expected_tf_set
        and (source, target) in localized_targets
    }
    if not expected_circuit:
        raise RuntimeError(
            "The expected top TF set has no localized signed TF-to-TF edge; "
            "the downstream circuit builder would have no circuit topology."
        )
    motif_report["expected_top_tfs"] = expected_tfs
    motif_report["expected_localized_tf_tf_edges"] = len(expected_circuit)

    print("6. Restricting signed OmniPath to productive exercise-to-TF paths...", flush=True)
    signaling, signaling_report = compile_signaling(
        paths["omnipath"], expected_tfs, args.max_signal_depth
    )

    print("7. Writing canonical evidence tables and provenance...", flush=True)
    write_tsv(
        output / "peak_gene_links.tsv",
        ("peak_id", "gene", "score", "evidence_source"),
        (
            (peak, gene, f"{score:.8g}", "GSE126100_promoter_capture_HiC")
            for (peak, gene), score in sorted(contact_links.items())
        ),
    )
    write_tsv(
        output / "motif_hits.tsv",
        ("peak_id", "tf", "score", "evidence_source"),
        (
            (peak, tf, f"{score:.8g}", "JASPAR2024_CORE_FIMO")
            for (peak, tf), score in sorted(motif_hits.items())
        ),
    )
    write_tsv(
        output / "tf_gene_edges.tsv",
        ("source", "target", "sign", "score", "sources", "references"),
        (
            (source, target, sign, f"{score:.8g}", sources, references)
            for (source, target), (sign, score, sources, references)
            in sorted(collectri.items())
        ),
    )
    write_tsv(
        output / "signaling_edges.tsv",
        (
            "source", "target", "source_type", "target_type",
            "sign", "score", "sources", "references",
        ),
        signaling,
    )

    source_hashes = {
        name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": COORDINATE_SYSTEM,
        "frozen_releases": {
            "GENCODE": GENCODE_RELEASE,
            "JASPAR": JASPAR_RELEASE,
            "GSE240061": "GEO processed Seurat object exported as raw RNA/ATAC counts",
            "GSE126100": "GEO processed promoter-capture Hi-C interactions",
            "CollecTRI_and_OmniPath": "exact retrieved response frozen by SHA-256 below",
        },
        "source_files": source_hashes,
        "group_split": {key: list(value) for key, value in split.items()},
        "selection": selection_report,
        "contacts": contact_report,
        "collectri": collectri_report,
        "motifs": motif_report,
        "signaling": signaling_report,
        "output_counts": {
            "peak_gene_links": len(contact_links),
            "motif_hits": len(motif_hits),
            "tf_gene_edges": len(collectri),
            "signaling_edges": len(signaling),
        },
        "leakage_contract": {
            "cohort_values_used_for_prior_selection": ["training-subject pre ATAC counts"],
            "cohort_values_not_used": [
                "RNA counts", "post/future ATAC", "validation/test ATAC",
                "cell type", "cluster", "integrated embeddings", "pseudotime",
            ],
            "prior_fit_groups": list(split["train"]),
        },
        "claim_boundary": (
            "This is a mechanistic topology scaffold for a transient 3.5-hour "
            "exercise response. It does not establish a terminal attractor or "
            "measured protein activity."
        ),
    }
    (output / "prior_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--pchic", type=Path, required=True)
    parser.add_argument("--gencode-gtf", type=Path, required=True)
    parser.add_argument("--collectri", type=Path, required=True)
    parser.add_argument("--omnipath", type=Path, required=True)
    parser.add_argument("--jaspar-meme", type=Path, required=True)
    parser.add_argument("--genome-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-time", default="pre")
    parser.add_argument("--max-candidate-peaks", type=int, default=5000)
    parser.add_argument("--expected-max-tfs", type=int, default=64)
    parser.add_argument("--max-signal-depth", type=int, default=4)
    parser.add_argument("--fimo-p-threshold", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_candidate_peaks < 100:
        parser.error("--max-candidate-peaks must be at least 100")
    if args.expected_max_tfs < 2:
        parser.error("--expected-max-tfs must be at least 2")
    if args.max_signal_depth < 1:
        parser.error("--max-signal-depth must be positive")
    if not 0 < args.fimo_p_threshold < 1:
        parser.error("--fimo-p-threshold must be between 0 and 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = build(args)
    print("\nPASS: WLD muscle biological priors compiled.")
    print(json.dumps(manifest["output_counts"], indent=2))
    print("Manifest:", args.output / "prior_manifest.json")


if __name__ == "__main__":
    main(sys.argv[1:])
