"""Compile Phase A atlases and frozen evidence into WLD v4 foundation priors.

This module does not learn an interaction graph from cell labels.  It intersects
four independently auditable evidence layers:

* the training-only Phase A feature atlas;
* peak-to-gene contact/promoter evidence;
* localized TF motif evidence;
* signed TF-gene and signaling evidence.

The result is a resource-bounded subnetwork that can be loaded as
``FoundationPriors``.  Edge existence and evidence sign are fixed.  Edge gains,
production, decay and chromatin time scales remain context-conditioned model
parameters and therefore may vary between cells, subjects and tissues.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from wld_foundation_data import atomic_json, parse_peak, sha256_file
from wld_foundation_model_v4 import FoundationPriors


REQUIRED_EVIDENCE = (
    "peak_gene_links.tsv",
    "motif_hits.tsv",
    "tf_gene_edges.tsv",
    "signaling_edges.tsv",
)


def open_text(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open("rt", newline="")


def read_vocab(path: Path) -> List[str]:
    with open_text(path) as handle:
        return [line.rstrip("\r\n").split("\t", 1)[0] for line in handle if line.strip()]


def read_dict_rows(path: Path) -> List[Dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Evidence table has no header: {path}")
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def positive(value: object, default: float = 1.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) and result > 0 else default


def signed_score(row: Mapping[str, str]) -> float:
    sign = float(row.get("sign", "0") or 0)
    if sign == 0 or not math.isfinite(sign):
        return 0.0
    return (1.0 if sign > 0 else -1.0) * positive(row.get("score", 1.0))


def peak_bin(value: str, bin_size: int) -> Optional[str]:
    parsed = parse_peak(value)
    if parsed is None:
        return None
    chrom, start, end = parsed
    center = (start + end) // 2
    begin = (center // bin_size) * bin_size
    return f"{chrom}:{begin}-{begin + bin_size}"


def normalized_score(value: float, maximum: float) -> float:
    return float(math.log1p(max(value, 0.0)) / max(math.log1p(maximum), 1e-8))


def _canonical_token(value: str) -> str:
    token = re.sub(r"[^A-Z0-9]+", "", value.upper())
    for prefix in ("ANTIHUMAN", "ANTIMOUSE", "HUMAN", "MOUSE", "ANTI"):
        if token.startswith(prefix):
            token = token[len(prefix):]
    return token


def _load_atlas(atlas_root: Path) -> Tuple[dict, Dict[str, List[str]]]:
    manifest_path = atlas_root / "atlas_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Phase A atlas manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("training_cohorts_only") is not True:
        raise ValueError("Phase B requires a training-only feature atlas")
    vocab = {
        "genes": read_vocab(atlas_root / "shared_genes.tsv.gz"),
        "peaks": read_vocab(atlas_root / "shared_peak_bins.tsv.gz"),
        "proteins": read_vocab(atlas_root / "shared_proteins.tsv.gz"),
        "metabolites": read_vocab(atlas_root / "shared_metabolites.tsv.gz"),
    }
    return manifest, vocab


def _max_values(values: Iterable[float]) -> float:
    return max((abs(value) for value in values), default=1.0)


def compile_phase_b_priors(
    atlas_root: Path,
    evidence_root: Path,
    output_root: Path,
    *,
    max_genes: int = 2048,
    max_peaks: int = 4096,
    max_tfs: int = 128,
    max_signals: int = 64,
    min_tfs: int = 4,
    min_localized_edges: int = 8,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Compile a bounded, evidence-intersection subnetwork.

    Selection depends only on the training atlas and frozen evidence tables.
    Validation/test counts, labels and embeddings are never read.
    """

    atlas_root = atlas_root.resolve()
    evidence_root = evidence_root.resolve()
    output_root = output_root.resolve()
    missing = [str(evidence_root / name) for name in REQUIRED_EVIDENCE if not (evidence_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical evidence tables: {missing}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        manifest_path = output_root / "prior_manifest.json"
        if manifest_path.is_file():
            return verify_phase_b_priors(output_root, atlas_root, evidence_root)
        raise FileExistsError(f"Output exists without a complete manifest: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    atlas, vocab = _load_atlas(atlas_root)
    if atlas.get("species") != "Homo sapiens" or atlas.get("genome_build") != "GRCh38":
        raise ValueError(
            "This evidence package is human GRCh38. Mouse or other builds require "
            "their own signed circuit, genome and contact compilation."
        )
    gene_universe = set(vocab["genes"])
    peak_universe = set(vocab["peaks"])
    bin_size = int(atlas["peak_bin_size"])

    contacts: Dict[Tuple[str, str], float] = {}
    for row in read_dict_rows(evidence_root / "peak_gene_links.tsv"):
        peak = peak_bin(row.get("peak_id", ""), bin_size)
        gene = row.get("gene", "")
        if peak in peak_universe and gene in gene_universe:
            key = (peak, gene)
            contacts[key] = max(contacts.get(key, 0.0), positive(row.get("score", 1.0)))

    motifs: Dict[Tuple[str, str], float] = {}
    for row in read_dict_rows(evidence_root / "motif_hits.tsv"):
        peak = peak_bin(row.get("peak_id", ""), bin_size)
        tf = row.get("tf", "")
        if peak in peak_universe and tf in gene_universe:
            key = (peak, tf)
            motifs[key] = max(motifs.get(key, 0.0), positive(row.get("score", 1.0)))

    tf_gene_all: Dict[Tuple[str, str], float] = {}
    for row in read_dict_rows(evidence_root / "tf_gene_edges.tsv"):
        source, target = row.get("source", ""), row.get("target", "")
        value = signed_score(row)
        if value and source in gene_universe and target in gene_universe:
            key = (source, target)
            previous = tf_gene_all.get(key, 0.0)
            if abs(value) > abs(previous):
                tf_gene_all[key] = value

    motifs_by_peak: MutableMapping[str, List[Tuple[str, float]]] = defaultdict(list)
    for (peak, tf), score in motifs.items():
        motifs_by_peak[peak].append((tf, score))
    triples: List[Tuple[str, str, str, float]] = []
    contact_max = _max_values(contacts.values())
    motif_max = _max_values(motifs.values())
    regulatory_max = _max_values(tf_gene_all.values())
    for (peak, gene), contact_score in contacts.items():
        for tf, motif_score in motifs_by_peak.get(peak, ()):
            regulatory = tf_gene_all.get((tf, gene), 0.0)
            if regulatory:
                score = (
                    normalized_score(contact_score, contact_max)
                    * normalized_score(motif_score, motif_max)
                    * normalized_score(abs(regulatory), regulatory_max)
                )
                triples.append((peak, tf, gene, score))
    if len(triples) < min_localized_edges:
        raise RuntimeError(
            f"Only {len(triples)} motif x contact x signed-regulation triples "
            "overlap the training atlas. Recompile motifs/contacts on the Phase A atlas."
        )

    tf_score: MutableMapping[str, float] = defaultdict(float)
    gene_score: MutableMapping[str, float] = defaultdict(float)
    peak_score: MutableMapping[str, float] = defaultdict(float)
    for peak, tf, gene, score in triples:
        tf_score[tf] += score
        gene_score[gene] += score
        peak_score[peak] += score
    selected_tfs = sorted(tf_score, key=lambda name: (-tf_score[name], name))[:max_tfs]
    if len(selected_tfs) < min_tfs:
        raise RuntimeError(f"Only {len(selected_tfs)} localized TFs survived; need {min_tfs}")
    tf_set = set(selected_tfs)
    selected_peaks = sorted(
        (name for name in peak_score if any(tf in tf_set for tf, _ in motifs_by_peak[name])),
        key=lambda name: (-peak_score[name], name),
    )[:max_peaks]
    peak_set = set(selected_peaks)
    targets = sorted(gene_score, key=lambda name: (-gene_score[name], name))
    selected_genes = list(selected_tfs)
    for gene in targets:
        if gene not in tf_set and len(selected_genes) < max_genes:
            selected_genes.append(gene)
    selected_genes = sorted(set(selected_genes), key=lambda name: (name not in tf_set, -gene_score.get(name, 0.0), name))
    if len(selected_genes) > max_genes:
        selected_genes = selected_genes[:max_genes]
    gene_set = set(selected_genes)
    triples = [row for row in triples if row[0] in peak_set and row[1] in tf_set and row[2] in gene_set]
    if len(triples) < min_localized_edges:
        raise RuntimeError("Resource bounding removed too many localized regulatory triples")

    gene_index = {name: index for index, name in enumerate(selected_genes)}
    peak_index = {name: index for index, name in enumerate(selected_peaks)}
    tf_index = {name: index for index, name in enumerate(selected_tfs)}
    peak_to_gene = np.zeros((len(selected_peaks), len(selected_genes)), dtype=np.float32)
    peak_tf_motif = np.zeros((len(selected_peaks), len(selected_tfs)), dtype=np.float32)
    tf_gene_support = np.zeros((len(selected_tfs), len(selected_genes)), dtype=np.float32)
    for peak, tf, gene, _ in triples:
        pi, ti, gi = peak_index[peak], tf_index[tf], gene_index[gene]
        peak_to_gene[pi, gi] = max(
            peak_to_gene[pi, gi], normalized_score(contacts[(peak, gene)], contact_max)
        )
        peak_tf_motif[pi, ti] = max(
            peak_tf_motif[pi, ti], normalized_score(motifs[(peak, tf)], motif_max)
        )
        tf_gene_support[ti, gi] = (
            math.copysign(normalized_score(abs(tf_gene_all[(tf, gene)]), regulatory_max), tf_gene_all[(tf, gene)])
        )

    circuit_tf_tf = np.zeros((len(selected_tfs), len(selected_tfs)), dtype=np.float32)
    for source in selected_tfs:
        for target in selected_tfs:
            value = tf_gene_support[tf_index[source], gene_index[target]]
            if value:
                circuit_tf_tf[tf_index[source], tf_index[target]] = value

    signaling_rows = read_dict_rows(evidence_root / "signaling_edges.tsv")
    reverse: MutableMapping[str, set[str]] = defaultdict(set)
    direct_tf_edges: List[Tuple[str, str, float]] = []
    signed_rows: List[Tuple[str, str, str, str, float]] = []
    for row in signaling_rows:
        source, target = row.get("source", ""), row.get("target", "")
        source_type, target_type = row.get("source_type", ""), row.get("target_type", "")
        value = signed_score(row)
        if not source or not target or not value:
            continue
        signed_rows.append((source, target, source_type, target_type, value))
        if target_type == "tf" and target in tf_set:
            direct_tf_edges.append((source, target, value))
            reverse[target].add(source)
        elif source_type == "signal" and target_type == "signal":
            reverse[target].add(source)
    productive = set(selected_tfs)
    queue = deque(selected_tfs)
    while queue:
        node = queue.popleft()
        for parent in reverse.get(node, ()):
            if parent not in productive:
                productive.add(parent)
                queue.append(parent)
    signal_score: MutableMapping[str, int] = defaultdict(int)
    for source, target, source_type, target_type, _ in signed_rows:
        if source_type == "signal" and source in productive:
            signal_score[source] += 1
        if target_type == "signal" and target in productive:
            signal_score[target] += 1
    selected_signals = sorted(signal_score, key=lambda name: (-signal_score[name], name))[:max_signals]
    if not selected_signals:
        raise RuntimeError("No signed signaling node reaches a selected localized TF")
    signal_index = {name: index for index, name in enumerate(selected_signals)}
    cues = sorted({source for source, _, source_type, _, _ in signed_rows if source_type == "cue"})
    if not cues:
        cues = ["baseline"]
    cue_index = {name: index for index, name in enumerate(cues)}
    signal_signal = np.zeros((len(selected_signals), len(selected_signals)), dtype=np.float32)
    signal_tf = np.zeros((len(selected_signals), len(selected_tfs)), dtype=np.float32)
    cue_signal = np.zeros((len(cues), len(selected_signals)), dtype=np.float32)
    signal_max = _max_values(value for *_, value in signed_rows)
    for source, target, source_type, target_type, value in signed_rows:
        score = math.copysign(normalized_score(abs(value), signal_max), value)
        if source_type == "signal" and target_type == "signal" and source in signal_index and target in signal_index:
            signal_signal[signal_index[source], signal_index[target]] = score
        elif source_type == "signal" and target_type == "tf" and source in signal_index and target in tf_index:
            signal_tf[signal_index[source], tf_index[target]] = score
        elif source_type == "cue" and target_type == "signal" and source in cue_index and target in signal_index:
            cue_signal[cue_index[source], signal_index[target]] = score

    proteins = list(vocab["proteins"])
    protein_signal = np.zeros((len(proteins), len(selected_signals)), dtype=np.float32)
    protein_tokens = {_canonical_token(name): index for index, name in enumerate(proteins)}
    mapped_proteins = []
    for signal, si in signal_index.items():
        token = _canonical_token(signal)
        if token in protein_tokens:
            protein_signal[protein_tokens[token], si] = 1.0
            mapped_proteins.append((proteins[protein_tokens[token]], signal))
    metabolites = list(vocab["metabolites"])
    metabolic_signal = np.zeros((len(metabolites), len(selected_signals)), dtype=np.float32)
    metabolic_tf = np.zeros((len(metabolites), len(selected_tfs)), dtype=np.float32)
    tf_peak_effect = np.zeros((len(selected_tfs), len(selected_peaks)), dtype=np.float32)
    tf_gene_index = np.asarray([gene_index[name] for name in selected_tfs], dtype=np.int64)

    arrays = {
        "peak_to_gene": peak_to_gene,
        "peak_tf_motif": peak_tf_motif,
        "tf_gene_support": tf_gene_support,
        "circuit_tf_tf": circuit_tf_tf,
        "signal_signal": signal_signal,
        "signal_tf": signal_tf,
        "cue_signal": cue_signal,
        "tf_peak_effect": tf_peak_effect,
        "tf_gene_index": tf_gene_index,
        "protein_signal": protein_signal,
        "metabolic_signal": metabolic_signal,
        "metabolic_tf": metabolic_tf,
    }
    np.savez_compressed(output_root / "foundation_priors.npz", **arrays)
    features = {
        "genes": selected_genes,
        "peaks": selected_peaks,
        "tfs": selected_tfs,
        "signals": selected_signals,
        "cues": cues,
        "proteins": proteins,
        "metabolites": metabolites,
    }
    atomic_json(output_root / "feature_vocab.json", features)

    source_paths = {
        "atlas_manifest": atlas_root / "atlas_manifest.json",
        **{name: evidence_root / name for name in REQUIRED_EVIDENCE},
    }
    manifest = {
        "schema_version": "2.0",
        "scope": "Phase B evidence-intersection prior; no dynamics or attractor claim",
        "species": atlas["species"],
        "genome_build": atlas["genome_build"],
        "training_atlas_only": True,
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "artifact_sha256": {
            "foundation_priors.npz": sha256_file(output_root / "foundation_priors.npz"),
            "feature_vocab.json": sha256_file(output_root / "feature_vocab.json"),
        },
        "dimensions": {name: len(values) for name, values in features.items()},
        "evidence": {
            "atlas_contact_links": len(contacts),
            "atlas_motif_hits": len(motifs),
            "signed_tf_gene_edges": len(tf_gene_all),
            "productive_localized_triples": len(triples),
            "tf_gene_edges_retained": int(np.count_nonzero(tf_gene_support)),
            "tf_circuit_edges_retained": int(np.count_nonzero(circuit_tf_tf)),
            "signal_edges_retained": int(np.count_nonzero(signal_signal) + np.count_nonzero(signal_tf) + np.count_nonzero(cue_signal)),
            "protein_signal_mappings": mapped_proteins,
            "tf_peak_effect_edges": 0,
        },
        "selection": {
            "uses": ["training-only Phase A atlas", "frozen external evidence"],
            "does_not_use": ["cell labels", "integrated embeddings", "validation counts", "sealed-test data", "future RNA"],
            "resource_limits": {"genes": max_genes, "peaks": max_peaks, "tfs": max_tfs, "signals": max_signals},
        },
        "variation_contract": {
            "cell_specific_accessibility": True,
            "context_conditioned_supported_edge_gains": True,
            "context_conditioned_production_decay_and_chromatin_timescale": True,
            "study_or_tissue_ids_as_encoder_inputs": False,
            "unsupported_edges_trainable": False,
            "missing_protein_or_metabolic_measurements_imputed": False,
        },
        "contact_scope_warning": (
            "Peak-gene evidence retains the scope stated by the source prior manifest. "
            "It is topology/confidence evidence, not a claim that contact strength is fixed across tissues."
        ),
        "chromatin_effect_boundary": (
            "TF-to-peak signed effects remain empty because motif occupancy alone does not establish "
            "opening versus closing. They require perturbational chromatin evidence."
        ),
        "sealed_test_downloaded": False,
        "model_trained": False,
    }
    atomic_json(output_root / "prior_manifest.json", manifest)
    verify_phase_b_priors(output_root, atlas_root, evidence_root)
    return manifest


def load_phase_b_priors(root: Path, device: str | torch.device = "cpu") -> FoundationPriors:
    root = Path(root)
    arrays = np.load(root / "foundation_priors.npz", allow_pickle=False)
    values = {}
    for name in (
        "peak_to_gene", "peak_tf_motif", "tf_gene_support", "circuit_tf_tf",
        "signal_signal", "signal_tf", "cue_signal", "tf_peak_effect",
        "protein_signal", "metabolic_signal", "metabolic_tf",
    ):
        values[name] = torch.as_tensor(arrays[name], dtype=torch.float32, device=device)
    values["tf_gene_index"] = torch.as_tensor(arrays["tf_gene_index"], dtype=torch.long, device=device)
    priors = FoundationPriors(**values)
    priors.validate()
    return priors


def verify_phase_b_priors(
    root: Path,
    atlas_root: Optional[Path] = None,
    evidence_root: Optional[Path] = None,
) -> Dict[str, object]:
    root = Path(root)
    for name in ("foundation_priors.npz", "feature_vocab.json", "prior_manifest.json"):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing Phase B artifact: {path}")
    manifest = json.loads((root / "prior_manifest.json").read_text())
    for name, expected_sha in manifest.get("artifact_sha256", {}).items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Phase B artifact hash mismatch: {path}")
    if atlas_root is not None:
        atlas_path = Path(atlas_root) / "atlas_manifest.json"
        if sha256_file(atlas_path) != manifest["source_sha256"]["atlas_manifest"]:
            raise RuntimeError("Phase A atlas changed after Phase B prior compilation")
    if evidence_root is not None:
        for name in REQUIRED_EVIDENCE:
            if sha256_file(Path(evidence_root) / name) != manifest["source_sha256"][name]:
                raise RuntimeError(f"Evidence source changed after compilation: {name}")
    vocab = json.loads((root / "feature_vocab.json").read_text())
    priors = load_phase_b_priors(root)
    expected = {
        "genes": priors.num_genes,
        "peaks": priors.num_peaks,
        "tfs": priors.num_tfs,
        "signals": priors.num_signals,
        "cues": priors.num_cues,
        "proteins": priors.num_proteins,
        "metabolites": priors.num_metabolites,
    }
    if manifest.get("dimensions") != expected:
        raise RuntimeError(f"Prior manifest dimensions disagree: {manifest.get('dimensions')} != {expected}")
    if any(len(vocab[name]) != size for name, size in expected.items()):
        raise RuntimeError("Feature vocabulary dimensions disagree with prior tensors")
    if manifest.get("sealed_test_downloaded") is not False or manifest.get("model_trained") is not False:
        raise RuntimeError("Phase B prior manifest crossed its claim boundary")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-genes", type=int, default=2048)
    parser.add_argument("--max-peaks", type=int, default=4096)
    parser.add_argument("--max-tfs", type=int, default=128)
    parser.add_argument("--max-signals", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = compile_phase_b_priors(
        args.atlas, args.evidence, args.output,
        max_genes=args.max_genes, max_peaks=args.max_peaks,
        max_tfs=args.max_tfs, max_signals=args.max_signals,
        overwrite=args.overwrite,
    )
    print("PASS: Phase B foundation priors compiled")
    print(json.dumps(report["dimensions"], indent=2))
    print(f"Manifest: {args.output / 'prior_manifest.json'}")


if __name__ == "__main__":
    main()


__all__ = [
    "compile_phase_b_priors", "load_phase_b_priors", "peak_bin",
    "verify_phase_b_priors",
]
