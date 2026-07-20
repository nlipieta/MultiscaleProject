"""Dependency-light synthetic validation of the Phase A data layer."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from scipy.io import mmwrite

from wld_foundation_data import (
    audit_encoder_metadata,
    build_training_atlas,
    pairing_report,
    project_bundle_to_atlas,
    read_10x_h5,
    read_10x_mtx,
    read_adt_csv,
    save_bundle,
)


def write_gzip_mtx(path: Path, matrix: sparse.spmatrix) -> None:
    plain = path.with_suffix("")
    mmwrite(plain, matrix)
    with plain.open("rb") as source, gzip.open(path, "wb") as target:
        target.write(source.read())
    plain.unlink()


def make_h5(path: Path) -> None:
    matrix = sparse.csc_matrix(
        np.asarray(
            [
                [1, 0, 2, 0],
                [0, 3, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 1],
            ], dtype=np.int32,
        )
    )
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        group.create_dataset("shape", data=matrix.shape)
        group.create_dataset("barcodes", data=np.asarray([b"c1", b"c2", b"c3", b"c4"]))
        features = group.create_group("features")
        features.create_dataset("id", data=np.asarray([b"g1", b"g2", b"p1", b"p2"]))
        features.create_dataset("name", data=np.asarray([b"G1", b"G2", b"chr1:0-100", b"chr2:200-300"]))
        features.create_dataset("feature_type", data=np.asarray([b"Gene Expression", b"Gene Expression", b"Peaks", b"Peaks"]))


def main() -> None:
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)
        h5_path = root / "fixture.h5"
        make_h5(h5_path)
        blocks = read_10x_h5(h5_path)
        assert set(blocks) == {"rna", "atac"}
        assert blocks["rna"].matrix.shape == (4, 2)
        assert pairing_report(blocks)["pairing"] == "exact"
        print("PASS: 10x H5 raw RNA/ATAC ingestion")

        matrix_path = root / "matrix.mtx.gz"
        barcode_path = root / "barcodes.tsv.gz"
        feature_path = root / "features.tsv.gz"
        write_gzip_mtx(matrix_path, sparse.coo_matrix(np.asarray([[1, 0, 2], [0, 1, 1], [1, 1, 0]])))
        with gzip.open(barcode_path, "wt") as handle:
            handle.write("m1\nm2\nm3\n")
        with gzip.open(feature_path, "wt") as handle:
            handle.write("g1\tG1\tGene Expression\n")
            handle.write("g2\tG2\tGene Expression\n")
            handle.write("p1\tchr1:100-200\tPeaks\n")
        mtx_blocks = read_10x_mtx(matrix_path, barcode_path, feature_path)
        assert set(mtx_blocks) == {"rna", "atac"}
        print("PASS: combined 10x Matrix Market modality split")

        adt_path = root / "adt.csv.gz"
        with gzip.open(adt_path, "wt", newline="") as handle:
            handle.write("barcode,CD3,CD4\n")
            handle.write("sample_AAACCCCCGGGGTTTT-1,4,2\n")
            handle.write("sample_TTTTGGGGCCCCAAAA-1,1,7\n")
            handle.write("unfiltered_AAAAAAAAAAAAAAAA-1,9,9\n")
        adt = read_adt_csv(
            adt_path,
            ["AAACCCCCGGGGTTTT-1", "TTTTGGGGCCCCAAAA-1"],
        )
        assert adt.matrix.shape == (2, 2)
        assert adt.barcodes == ["AAACCCCCGGGGTTTT-1", "TTTTGGGGCCCCAAAA-1"]
        print("PASS: prefixed ADT barcodes filtered and oriented as cells x proteins")

        cohort_a = {"cohort_id": "train_a", "study_id": "study_a", "species": "Homo sapiens", "genome_build": "GRCh38", "adapter": "fixture", "donor_scope": "a"}
        cohort_b = {"cohort_id": "train_b", "study_id": "study_b", "species": "Homo sapiens", "genome_build": "GRCh38", "adapter": "fixture", "donor_scope": "b"}
        bundle_a = root / "bundle_a"
        bundle_b = root / "bundle_b"
        save_bundle(bundle_a, cohort_a, blocks, {"fixture": {"sha256": "a"}})
        save_bundle(bundle_b, cohort_b, mtx_blocks, {"fixture": {"sha256": "b"}})
        atlas = build_training_atlas([bundle_a, bundle_b], root / "atlas", max_genes=10, max_peak_bins=10)
        assert atlas["genes"] == 2 and atlas["peak_bins"] >= 1
        print("PASS: training-only gene and genomic-bin atlas")
        projected = project_bundle_to_atlas(bundle_a, root / "atlas", root / "projected_a")
        assert projected["modalities"]["rna"]["shape"][1] == atlas["genes"]
        assert projected["modalities"]["atac"]["shape"][1] == atlas["peak_bins"]
        print("PASS: cohort projection into shared training-derived feature space")

        mouse = dict(cohort_b, cohort_id="mouse", study_id="mouse_study", species="Mus musculus", genome_build="mm10")
        mouse_root = root / "mouse_bundle"
        save_bundle(mouse_root, mouse, mtx_blocks, {"fixture": {"sha256": "mouse"}})
        try:
            build_training_atlas([bundle_a, mouse_root], root / "invalid_mixed_atlas")
        except ValueError:
            pass
        else:
            raise AssertionError("Human and mouse features were merged without an ortholog contract")
        print("PASS: species/genome atlases cannot be silently merged")

        try:
            audit_encoder_metadata(["age", "metabolite", "study_id", "cell_type"])
        except ValueError:
            pass
        else:
            raise AssertionError("Identity/label metadata reached the encoder")
        print("PASS: metadata identifiers and state labels rejected from encoder")

        report = {
            "scope": "synthetic Phase A data-contract validation only",
            "real_data_downloaded": False,
            "biological_model_trained": False,
            "sealed_test_evaluated": False,
            "checks": ["tenx_h5", "tenx_mtx", "adt_orientation", "bundle", "training_atlas", "atlas_projection", "species_isolation", "metadata_leakage"],
        }
        Path(__file__).with_name("wld_phase_a_data_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PASS: wrote wld_phase_a_data_validation.json")


if __name__ == "__main__":
    main()
