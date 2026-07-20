"""Synthetic contract tests for WLD corpus expansion adapters."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

from wld_corpus_expansion import (
    ingest_shareseq_legacy_pair,
    ingest_shareseq_metadata_pair,
    read_metadata_table,
    write_context_manifest,
)
from wld_foundation_data import build_training_atlas, save_bundle, verify_bundle


def write_mtx(path: Path, matrix: sparse.spmatrix) -> None:
    plain = path.parent / (path.name + ".fixture.mtx")
    mmwrite(plain, matrix)
    with plain.open("rb") as source, gzip.open(path, "wb") as target:
        target.write(source.read())
    plain.unlink()


def write_gz(path: Path, value: str) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write(value)


def main() -> None:
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)

        # Modern SHARE-seq fixture: modality metadata supplies a common,
        # deposited observation identifier in different row orders.
        modern = root / "modern"
        modern.mkdir()
        write_mtx(modern / "rna.mtx.gz", sparse.coo_matrix(np.array([[1, 0, 2], [0, 3, 1]])))
        write_mtx(modern / "atac.mtx.gz", sparse.coo_matrix(np.array([[5, 0, 2], [0, 4, 1]])))
        write_gz(modern / "genes.tsv.gz", "G1\nG2\n")
        write_gz(modern / "peaks.bed.gz", "chr1\t100\t200\nchr2\t300\t400\n")
        write_gz(
            modern / "rna_meta.tsv.gz",
            "donor\tcell_type\n"
            "cell_a\td1\tT\n"
            "cell_b\td1\tB\n"
            "cell_c\td2\tT\n",
        )
        write_gz(
            modern / "atac_meta.tsv.gz",
            "\tdonor\tcluster\n"
            "cell_c\td2\t2\n"
            "cell_a\td1\t0\n"
            "cell_b\td1\t1\n",
        )
        modern_files = {
            "rna_matrix": modern / "rna.mtx.gz",
            "rna_features": modern / "genes.tsv.gz",
            "rna_metadata": modern / "rna_meta.tsv.gz",
            "atac_matrix": modern / "atac.mtx.gz",
            "atac_features": modern / "peaks.bed.gz",
            "atac_metadata": modern / "atac_meta.tsv.gz",
        }
        repaired_header, repaired_rows = read_metadata_table(modern / "rna_meta.tsv.gz")
        assert repaired_header[0] == "__row_id__"
        assert repaired_header[1:] == ["donor", "cell_type"]
        assert repaired_rows[0] == ["cell_a", "d1", "T"]
        print("PASS: unnamed submitted row-ID columns are preserved without shifting metadata")

        blocks, evidence, context = ingest_shareseq_metadata_pair(modern_files)
        assert blocks["rna"].barcodes == blocks["atac"].barcodes == ["cell_a", "cell_b", "cell_c"]
        assert evidence["left_column"] == evidence["right_column"] == "__row_id__"
        assert evidence["expression_or_label_matching_used"] is False
        assert evidence["result"] == "exact_after_deposited_identifier_alignment"
        assert "cell_type" in context["rna_metadata_fields"]
        print("PASS: deposited SHARE-seq row names align RNA/ATAC without entering the encoder")

        # If neither modality contains a unique deposited observation ID, the
        # adapter must retain two unpaired populations rather than fabricating
        # cell pairs from expression, labels, embeddings, or row position.
        write_gz(
            modern / "rna_meta.tsv.gz",
            "donor\tcell_type\n"
            "d1\tT\n"
            "d1\tB\n"
            "d2\tT\n",
        )
        write_gz(
            modern / "atac_meta.tsv.gz",
            "donor\tcluster\n"
            "d2\t2\n"
            "d1\t0\n"
            "d1\t1\n",
        )
        unpaired_blocks, unpaired_evidence, _ = ingest_shareseq_metadata_pair(modern_files)
        assert unpaired_evidence["result"] == "unpaired_population"
        assert unpaired_evidence["expression_or_label_matching_used"] is False
        assert unpaired_evidence["synthetic_cell_pairing_used"] is False
        assert set(unpaired_blocks["rna"].barcodes).isdisjoint(unpaired_blocks["atac"].barcodes)
        print("PASS: missing identifiers produce unpaired populations, never fabricated cell pairs")

        # Legacy fixture: the ATAC metadata explicitly translates ATAC to RNA
        # barcodes.  The cell-type column is present but cannot drive pairing.
        legacy = root / "legacy"
        legacy.mkdir()
        write_gz(
            legacy / "rna.txt.gz",
            "gene\tR1\tR2\tR3\nG1\t1\t0\t2\nG2\t0\t3\t1\n",
        )
        write_mtx(legacy / "atac.txt.gz", sparse.coo_matrix(np.array([[5, 0, 2], [0, 4, 1]])))
        write_gz(legacy / "barcodes.txt.gz", "A1\nA2\nA3\n")
        write_gz(legacy / "peaks.bed.gz", "chr1\t100\t200\nchr2\t300\t400\n")
        write_gz(
            legacy / "celltype.txt.gz",
            "atac.bc\trna.bc\tcelltype\nA1\tR1\tT\nA2\tR2\tB\nA3\tR3\tT\n",
        )
        legacy_files = {
            "rna_table": legacy / "rna.txt.gz",
            "atac_matrix": legacy / "atac.txt.gz",
            "atac_barcodes": legacy / "barcodes.txt.gz",
            "atac_features": legacy / "peaks.bed.gz",
            "pairing_metadata": legacy / "celltype.txt.gz",
        }
        legacy_blocks, legacy_evidence, legacy_context = ingest_shareseq_legacy_pair(legacy_files)
        assert legacy_blocks["rna"].barcodes == legacy_blocks["atac"].barcodes == ["R1", "R2", "R3"]
        assert legacy_evidence["method"] == "deposited_barcode_crosswalk"
        assert legacy_evidence["expression_or_label_matching_used"] is False
        assert legacy_context["metadata_appended_to_encoder"] is False
        print("PASS: deposited legacy barcode translation is used; cell labels remain metadata")

        cohort_human = {
            "cohort_id": "human_fixture",
            "study_id": "human_study",
            "split": "train",
            "species": "Homo sapiens",
            "genome_build": "GRCh38",
            "tissue": "bone marrow",
            "adapter": "shareseq_metadata_pair",
            "donor_scope": "two donors",
            "pairing_evidence": evidence,
            "context_contract": {
                "retain_outside_encoder": ["donor", "age"],
                "never_encoder": ["cell_type", "barcode", "pseudotime"],
                "fold_local_context_encoding_required": True,
            },
        }
        human_root = root / "human_bundle"
        save_bundle(human_root, cohort_human, blocks, {"fixture": {"sha256": "human"}})
        write_context_manifest(human_root, cohort_human, context)
        manifest = verify_bundle(human_root)
        assert manifest["context_contract"]["fold_local_context_encoding_required"] is True
        assert manifest["pairing"]["pairing"] == "exact"
        print("PASS: variable donor/tissue context is retained outside the encoder")

        cohort_mouse = dict(
            cohort_human,
            cohort_id="mouse_fixture",
            study_id="mouse_study",
            species="Mus musculus",
            genome_build="mm10",
            tissue="brain",
            adapter="shareseq_legacy_pair",
            pairing_evidence=legacy_evidence,
        )
        mouse_root = root / "mouse_bundle"
        save_bundle(mouse_root, cohort_mouse, legacy_blocks, {"fixture": {"sha256": "mouse"}})
        write_context_manifest(mouse_root, cohort_mouse, legacy_context)
        human_atlas = build_training_atlas([human_root], root / "human_atlas")
        mouse_atlas = build_training_atlas([mouse_root], root / "mouse_atlas")
        assert human_atlas["species"] == "Homo sapiens"
        assert mouse_atlas["species"] == "Mus musculus"
        try:
            build_training_atlas([human_root, mouse_root], root / "invalid_mixed_atlas")
        except ValueError:
            pass
        else:
            raise AssertionError("Human and mouse cohorts were silently merged")
        print("PASS: species and genome-build feature atlases remain separate")

        registry = json.loads(Path(__file__).with_name("wld_corpus_expansion_sources.json").read_text())
        registered = {value["study_id"] for value in registry["cohorts"]}
        assert not registered.intersection(registry["sealed_exclusions"])
        assert all(value["split"] == "train" for value in registry["cohorts"])
        assert "GSE217215" in registry["staged_not_ingested"]
        print("PASS: expansion registry excludes sealed tests and marks unresolved layouts honestly")

        report = {
            "scope": "synthetic corpus-expansion contract only",
            "real_data_downloaded": False,
            "model_trained": False,
            "sealed_test_evaluated": False,
            "checks": [
                "metadata_identifier_pairing",
                "deposited_unnamed_row_id_pairing",
                "unpaired_population_fallback",
                "legacy_barcode_crosswalk",
                "context_outside_encoder",
                "species_build_isolation",
                "sealed_registry_exclusion",
            ],
        }
        Path(__file__).with_name("wld_corpus_expansion_validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("PASS: wrote wld_corpus_expansion_validation.json")


if __name__ == "__main__":
    main()
