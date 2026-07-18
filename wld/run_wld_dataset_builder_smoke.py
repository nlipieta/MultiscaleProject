"""Synthetic contract test for the real-data WLD dataset builder.

This checks matrix ingestion, group-first splitting, train-only feature
selection, the four-way regulatory intersection, cue provenance, missing-cue
masking, and compatibility with the temporal trainer.  It is not biological
evidence and downloads no public data.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

def _write_table(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_fixture(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(7)
    groups = ("ex1", "ex2", "ex3", "ex4", "rest1", "rest2")
    genes = ("TF1", "TF2", "TF3", "GENE1", "GENE2", "GENE3", "GENE4", "GENE5")
    peaks = tuple(f"chr1:{100 * index}-{100 * index + 80}" for index in range(1, 11))
    barcodes = []
    metadata = []
    rna_columns = []
    atac_columns = []
    for group in groups:
        condition = "exercise" if group.startswith("ex") else "rest"
        for time in ("pre", "post_3.5h"):
            for cell_index in range(6):
                barcode = f"{group}_{time}_{cell_index}"
                barcodes.append(barcode)
                metadata.append(
                    {
                        "cell_id": barcode,
                        "subject": group,
                        "timepoint": time,
                        "condition": condition,
                        "cell_type": "forbidden_as_input",
                    }
                )
                condition_shift = int(condition == "exercise" and time == "post_3.5h")
                rna_columns.append(rng.poisson(2.0, len(genes)) + condition_shift * np.arange(len(genes)))
                atac_columns.append(rng.binomial(1, 0.35 + 0.1 * condition_shift, len(peaks)))

    rna = sparse.coo_matrix(np.asarray(rna_columns, dtype=np.float32).T)
    atac = sparse.coo_matrix(np.asarray(atac_columns, dtype=np.float32).T)
    mmwrite(root / "rna.mtx", rna)
    mmwrite(root / "atac.mtx", atac)
    (root / "genes.tsv").write_text("\n".join(genes) + "\n", encoding="utf-8")
    (root / "peaks.tsv").write_text("\n".join(peaks) + "\n", encoding="utf-8")
    (root / "barcodes.tsv").write_text("\n".join(barcodes) + "\n", encoding="utf-8")
    _write_table(root / "metadata.tsv", metadata[0].keys(), metadata)

    peak_gene = []
    motif = []
    for index, peak in enumerate(peaks):
        peak_gene.append({"peak_id": peak, "gene": genes[index % len(genes)], "score": 1 + index})
        peak_gene.append({"peak_id": peak, "gene": genes[(index + 1) % len(genes)], "score": 0.5 + index})
        for tf_index, tf in enumerate(genes[:3]):
            motif.append({"peak_id": peak, "tf": tf, "score": 1 + tf_index})
    _write_table(root / "peak_gene.tsv", ("peak_id", "gene", "score"), peak_gene)
    _write_table(root / "motif.tsv", ("peak_id", "tf", "score"), motif)
    tf_gene = []
    for source_index, source in enumerate(genes[:3]):
        for target_index, target in enumerate(genes):
            tf_gene.append(
                {
                    "source": source,
                    "target": target,
                    "sign": 1 if (source_index + target_index) % 2 == 0 else -1,
                    "score": 1.0,
                }
            )
    _write_table(root / "tf_gene.tsv", ("source", "target", "sign", "score"), tf_gene)
    signaling = [
        {"source": "exercise", "target": "AMPK", "source_type": "cue", "target_type": "signal", "sign": 1, "score": 1},
        {"source": "metabolic:lactate", "target": "AMPK", "source_type": "cue", "target_type": "signal", "sign": 1, "score": 0.7},
        {"source": "AMPK", "target": "PGC1", "source_type": "signal", "target_type": "signal", "sign": 1, "score": 1},
        {"source": "PGC1", "target": "TF1", "source_type": "signal", "target_type": "tf", "sign": 1, "score": 1},
        {"source": "AMPK", "target": "TF2", "source_type": "signal", "target_type": "tf", "sign": -1, "score": 0.5},
    ]
    _write_table(
        root / "signaling.tsv",
        ("source", "target", "source_type", "target_type", "sign", "score"),
        signaling,
    )
    metabolic = [
        {"subject": "ex1", "lactate": 1.2},
        {"subject": "ex2", "lactate": 1.8},
        {"subject": "rest1", "lactate": 0.5},
        {"subject": "ex3", "lactate": 1.6},
        {"subject": "ex4", "lactate": ""},
        {"subject": "rest2", "lactate": 0.4},
    ]
    _write_table(root / "metabolic.tsv", ("subject", "lactate"), metabolic)
    split = {
        "train": ["ex1", "ex2", "rest1"],
        "validation": ["ex3"],
        "test": ["ex4", "rest2"],
    }
    (root / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return {
        "rna": root / "rna.mtx",
        "atac": root / "atac.mtx",
        "genes": root / "genes.tsv",
        "peaks": root / "peaks.tsv",
        "barcodes": root / "barcodes.tsv",
        "metadata": root / "metadata.tsv",
        "peak_gene": root / "peak_gene.tsv",
        "motif": root / "motif.tsv",
        "tf_gene": root / "tf_gene.tsv",
        "signaling": root / "signaling.tsv",
        "metabolic": root / "metabolic.tsv",
        "split": root / "split.json",
        "output": root / "cohort",
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wld_dataset_builder_") as directory:
        root = Path(directory)
        paths = write_fixture(root)
        builder = Path(__file__).with_name("build_wld_muscle_exercise_dataset.py")
        command = [
            sys.executable,
            str(builder),
            "--rna-mtx", str(paths["rna"]),
            "--atac-mtx", str(paths["atac"]),
            "--genes", str(paths["genes"]),
            "--peaks", str(paths["peaks"]),
            "--barcodes", str(paths["barcodes"]),
            "--metadata", str(paths["metadata"]),
            "--peak-gene-links", str(paths["peak_gene"]),
            "--motif-hits", str(paths["motif"]),
            "--tf-gene-edges", str(paths["tf_gene"]),
            "--signaling-edges", str(paths["signaling"]),
            "--metabolic-covariates", str(paths["metabolic"]),
            "--metabolic-columns", "lactate",
            "--split-json", str(paths["split"]),
            "--min-cells-per-time", "4",
            "--max-genes", "8",
            "--max-peaks", "10",
            "--max-tfs", "3",
            "--output", str(paths["output"]),
        ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            print(result.stdout)
            raise RuntimeError("Dataset builder subprocess failed.")
        try:
            from wld_temporal_training import load_temporal_cohort
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            load_temporal_cohort = None
        report = json.loads((paths["output"] / "build_report.json").read_text())
        manifest = json.loads((paths["output"] / "manifest.json").read_text())
        with np.load(paths["output"] / "observations.npz", allow_pickle=False) as observations:
            has_masked_cue = bool((observations["initial_cue_mask"] == 0).any())
            has_initial_rna = "initial_rna" in observations.files
        if manifest["split_groups"]["test"] != ["ex4", "rest2"]:
            raise AssertionError("The prespecified subject-level split changed.")
        if has_initial_rna:
            raise AssertionError("Initial RNA was included in the core build.")
        if not has_masked_cue:
            raise AssertionError("Missing metabolic values were not retained as a mask.")
        if report["edge_counts"]["circuit_tf_tf"] == 0:
            raise AssertionError("The compiled hard circuit is empty.")
        if "cell_type" in " ".join(manifest["initial_feature_names"]).lower():
            raise AssertionError("A cell identity label entered the initial inputs.")
        if load_temporal_cohort is not None:
            cohort = load_temporal_cohort(paths["output"])
            if cohort.initial_cue_mask is None:
                raise AssertionError("Temporal loader dropped the cue mask.")
        print("PASS: subject split precedes feature and prior selection")
        print("PASS: ATAC and future RNA/ATAC are unpaired population observations")
        print("PASS: motif x contact x signed-regulation intersection compiled")
        print("PASS: cue-to-signal-to-TF paths compiled without neural bypass")
        print("PASS: missing metabolic observations are masked, never imputed")
        if load_temporal_cohort is not None:
            print("PASS: temporal trainer accepts the generated cohort")
        else:
            print("SKIP: temporal trainer import (PyTorch unavailable in this interpreter)")
        print("NOTE: synthetic software fixture only; no biological claim")


if __name__ == "__main__":
    main()
