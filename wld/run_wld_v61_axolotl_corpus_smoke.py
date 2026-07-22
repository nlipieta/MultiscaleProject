"""Synthetic adversarial contract for the WLD v6.1 corpus builder.

No network request or biological measurement is used.  Tiny fixtures exercise
the same download locks, parsers, group split, feature fitting, provenance,
pairing, restart and sealed-accession gates used by the real Colab build.
"""

from __future__ import annotations

import copy
import csv
import gzip
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

from wld_axolotl_corpus_v61 import (
    CorpusIntegrityError,
    SCHEMA_VERSION,
    SealedSourceRefused,
    _SafeRedirectHandler,
    _download_https,
    assert_unsealed,
    audit_encoder_feature_names,
    build_from_registry,
    parse_dense_sc_barcode_maps,
    sha256_file,
    validate_context_record,
    validate_initial_provenance,
    validate_registry,
    verify_corpus,
)


def raises(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"{function.__name__} did not raise {error_type.__name__}")


class FakeResponse:
    def __init__(self, headers: dict[str, str], body: bytes = b"fixture") -> None:
        self.headers = headers
        self.body = body
        self.offset = 0
        self.read_count = 0
        self.status = 200

    def geturl(self):
        return "https://fixture.test/public.tsv.gz"

    def getcode(self):
        return self.status

    def read(self, size=-1):
        self.read_count += 1
        if self.offset >= len(self.body):
            return b""
        end = len(self.body) if size < 0 else min(len(self.body), self.offset + size)
        value = self.body[self.offset:end]
        self.offset = end
        return value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeOpener:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls = 0

    def open(self, _request, timeout=120):
        self.calls += 1
        return self.response


def exercise_network_seals() -> None:
    original = urllib.request.build_opener
    try:
        calls = []
        urllib.request.build_opener = lambda *_args, **_kwargs: calls.append(True)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "never" / "sealed.gz"
            raises(
                SealedSourceRefused,
                _download_https,
                "https://fixture.test/GSE315993.tsv.gz",
                destination,
                allowed_hosts=["fixture.test"],
            )
            assert not calls and not destination.parent.exists()

        handler = _SafeRedirectHandler(["fixture.test"])
        raises(
            SealedSourceRefused,
            handler.redirect_request,
            urllib.request.Request("https://fixture.test/public"),
            None, 302, "Found", {},
            "https://fixture.test/GSE%253315993/data",
        )
        assert handler.chain == []

        dangerous_headers = [
            {"Content-Disposition": "attachment; filename=GSE315993.tsv"},
            {"Content-Disposition": "attachment; name=GSE315993"},
            {"Content-Disposition": "attachment; filename*0*=UTF-8''GSE31; filename*1*=5993.tsv"},
            {"Content-Location": "/GSE%253315993/data.bin"},
        ]
        for headers in dangerous_headers:
            response = FakeResponse(headers)
            opener = FakeOpener(response)
            urllib.request.build_opener = lambda *_args, _opener=opener, **_kwargs: _opener
            with tempfile.TemporaryDirectory() as directory:
                raises(
                    SealedSourceRefused,
                    _download_https,
                    "https://fixture.test/public.tsv.gz",
                    Path(directory) / "public.tsv.gz",
                    allowed_hosts=["fixture.test"],
                )
            assert opener.calls == 1 and response.read_count == 0

        response = FakeResponse({"Content-Length": "99"}, b"short")
        urllib.request.build_opener = lambda *_args, **_kwargs: FakeOpener(response)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "length.bin"
            raises(
                CorpusIntegrityError, _download_https,
                "https://fixture.test/public.bin", destination,
                allowed_hosts=["fixture.test"],
            )
            assert not destination.exists()

        response = FakeResponse({"Content-Length": "7"}, b"notgzip")
        urllib.request.build_opener = lambda *_args, **_kwargs: FakeOpener(response)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "invalid.gz"
            raises(
                CorpusIntegrityError, _download_https,
                "https://fixture.test/public.tsv.gz", destination,
                allowed_hosts=["fixture.test"],
            )
            assert not destination.exists()

        response = FakeResponse(
            {"Content-Length": "4", "Content-Range": "bytes 0-3/7"}, b"more"
        )
        response.status = 206
        urllib.request.build_opener = lambda *_args, **_kwargs: FakeOpener(response)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resume.bin"
            destination.with_suffix(".bin.part").write_bytes(b"abc")
            raises(
                CorpusIntegrityError, _download_https,
                "https://fixture.test/public.bin", destination,
                allowed_hosts=["fixture.test"],
            )
            assert not destination.exists()
    finally:
        urllib.request.build_opener = original


def write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(text.encode())


def write_mtx_gz(path: Path, matrix: sparse.spmatrix) -> None:
    temporary = path.with_suffix("")
    mmwrite(temporary, matrix)
    with temporary.open("rb") as source, path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as target:
            shutil.copyfileobj(source, target)
    temporary.unlink()


def registry() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "sealed_exclusions": ["GSE315993"],
        "allowed_hosts": ["fixture.test"],
        "studies": [
            {
                "accession": "DEVTRAIN1",
                "role": "synthetic_train",
                "default_download": True,
                "adapter": "bulk_gene_table",
                "source_assembly": "AXO_FIXTURE_V1",
                "measurement_level": "synthetic_bulk_sample",
                "partition": "train",
                "sample_map": [
                    {"sample": "train_A", "group_id": "DEVTRAIN1:animal_A"},
                    {"sample": "train_B", "group_id": "DEVTRAIN1:animal_B"},
                ],
                "artifacts": [{
                    "artifact_id": "train_counts", "role": "raw_gene_counts",
                    "url": "https://fixture.test/train.tsv.gz",
                    "filename": "train.tsv.gz", "content_kind": "fixture", "checksum": None,
                }],
            },
            {
                "accession": "DEVVALID1",
                "role": "synthetic_validation",
                "default_download": True,
                "adapter": "bulk_gene_table",
                "source_assembly": "AXO_FIXTURE_V1",
                "measurement_level": "synthetic_bulk_sample",
                "partition": "validation",
                "sample_map": [
                    {"sample": "valid_A", "group_id": "DEVVALID1:animal_A"},
                    {"sample": "valid_B", "group_id": "DEVVALID1:animal_B"},
                ],
                "artifacts": [{
                    "artifact_id": "valid_counts", "role": "deposited_gene_counts",
                    "url": "https://fixture.test/valid.tsv.gz",
                    "filename": "valid.tsv.gz", "content_kind": "fixture", "checksum": None,
                }],
            },
            {
                "accession": "DEVSPACE1",
                "role": "synthetic_spatial_reference",
                "default_download": True,
                "adapter": "visium_bundle",
                "source_assembly": "AXO_FIXTURE_V2",
                "measurement_level": "synthetic_spatial_spot",
                "partition": "reference",
                "specimen_id": "section_1",
                "physical_time_hours": 120.0,
                "artifacts": [
                    {"artifact_id": "space_matrix", "role": "spot_gene_counts", "url": "https://fixture.test/matrix.mtx.gz", "filename": "matrix.mtx.gz", "content_kind": "fixture", "checksum": None},
                    {"artifact_id": "space_barcodes", "role": "spot_barcodes", "url": "https://fixture.test/barcodes.tsv.gz", "filename": "barcodes.tsv.gz", "content_kind": "fixture", "checksum": None},
                    {"artifact_id": "space_features", "role": "gene_features", "url": "https://fixture.test/features.tsv.gz", "filename": "features.tsv.gz", "content_kind": "fixture", "checksum": None},
                    {"artifact_id": "space_coordinates", "role": "spot_coordinates", "url": "https://fixture.test/positions.csv.gz", "filename": "positions.csv.gz", "content_kind": "fixture", "checksum": None},
                ],
            },
            {
                "accession": "DEVCHROM1",
                "role": "synthetic_staged_chromatin",
                "default_download": False,
                "adapter": "acquisition_plan_only",
                "source_assembly": "AXO_FIXTURE_V2",
                "measurement_level": "synthetic_bulk_atac",
                "artifacts": [{
                    "artifact_id": "chromatin_plan", "role": "atac",
                    "url": "https://fixture.test/chromatin.bw",
                    "filename": "chromatin.bw", "content_kind": "fixture", "checksum": None,
                }],
            },
        ],
    }


def make_sources(root: Path, holdout_scale: float = 1.0) -> dict[str, Path]:
    sources = root / "sources"
    sources.mkdir(parents=True)
    write_gzip(
        sources / "train.tsv.gz",
        "Geneid Chr Start End Strand Length train_A train_B\n"
        "TRAIN_VAR chr1 1 2 + 1 1 7\n"
        "STABLE chr1 3 4 + 1 2 2\n"
        "HOLDOUT_SPIKE chr1 5 6 + 1 0 0\n",
    )
    write_gzip(
        sources / "valid.tsv.gz",
        "gene\tvalid_A\tvalid_B\n"
        f"TRAIN_VAR\t2\t3\nSTABLE\t2\t2\nHOLDOUT_SPIKE\t{999999 * holdout_scale:g}\t0\n",
    )
    # Matrix Market is feature x spot, with coordinates deliberately shuffled.
    write_mtx_gz(
        sources / "matrix.mtx.gz",
        sparse.csr_matrix(np.asarray([[1, 0], [0, 4], [2, 3]], dtype=np.int64)),
    )
    write_gzip(sources / "barcodes.tsv.gz", "spot_A\nspot_B\n")
    write_gzip(sources / "features.tsv.gz", "f1\tDUPLICATE_SYMBOL\tGene Expression\nf2\tDUPLICATE_SYMBOL\tGene Expression\nf3\tGENE3\tGene Expression\n")
    write_gzip(
        sources / "positions.csv.gz",
        "spot_B,1,1,0,20,30\nspot_A,1,0,0,10,15\nunused_array_spot,0,2,2,40,50\n",
    )
    return {path.name: path for path in sources.iterdir()}


class FixtureFetcher:
    def __init__(self, sources: dict[str, Path]) -> None:
        self.sources = sources
        self.calls = 0

    def __call__(self, url, destination, allowed_hosts, expected_sha256):
        self.calls += 1
        source = self.sources[Path(url).name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = sha256_file(destination)
        if expected_sha256 is not None:
            assert digest == expected_sha256
        return {
            "requested_url": url,
            "effective_url": url,
            "redirect_chain": [],
            "sha256": digest,
            "byte_size": destination.stat().st_size,
            "checksum_provenance": "locally_computed_not_publisher_supplied",
        }


class InterruptAfterOne(FixtureFetcher):
    def __call__(self, url, destination, allowed_hosts, expected_sha256):
        if self.calls == 1:
            self.calls += 1
            raise RuntimeError("synthetic disconnect after first artifact")
        return super().__call__(url, destination, allowed_hosts, expected_sha256)


def exercise_crash_resume() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sources = root / "sources"
        sources.mkdir()
        write_gzip(sources / "counts.tsv.gz", "gene\tsample_A\nGENE1\t3\n")
        write_gzip(sources / "sidecar.tsv.gz", "key\tvalue\nfixture\t1\n")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sealed_exclusions": ["GSE315993"],
            "allowed_hosts": ["fixture.test"],
            "studies": [{
                "accession": "RESTART1",
                "role": "synthetic_train",
                "default_download": True,
                "adapter": "bulk_gene_table",
                "source_assembly": "AXO_FIXTURE_V1",
                "measurement_level": "synthetic_bulk_sample",
                "partition": "train",
                "sample_map": [{"sample": "sample_A", "group_id": "RESTART1:A"}],
                "artifacts": [
                    {"artifact_id": "counts", "role": "raw_gene_counts", "url": "https://fixture.test/counts.tsv.gz", "filename": "counts.tsv.gz", "content_kind": "fixture", "checksum": None},
                    {"artifact_id": "sidecar", "role": "metadata", "url": "https://fixture.test/sidecar.tsv.gz", "filename": "sidecar.tsv.gz", "content_kind": "fixture", "checksum": None},
                ],
            }],
        }
        registry_path = root / "registry.json"
        registry_path.write_text(json.dumps(payload))
        output = root / "corpus"
        interrupted = InterruptAfterOne({path.name: path for path in sources.iterdir()})
        raises(
            RuntimeError, build_from_registry, registry_path, output,
            fetcher=interrupted,
        )
        first = output / "raw" / "RESTART1" / "counts.tsv.gz"
        assert first.is_file()
        assert first.with_suffix(first.suffix + ".download.json").is_file()
        resumed = FixtureFetcher({path.name: path for path in sources.iterdir()})
        build_from_registry(registry_path, output, fetcher=resumed)
        assert resumed.calls == 1
        assert verify_corpus(output)["verified"] is True


def run_build(root: Path, holdout_scale: float = 1.0):
    sources = make_sources(root, holdout_scale)
    registry_path = root / "registry.json"
    registry_path.write_text(json.dumps(registry(), indent=2))
    fetcher = FixtureFetcher(sources)
    output = root / "corpus"
    report = build_from_registry(registry_path, output, fetcher=fetcher)
    return report, output, fetcher


def exercise_derived_tamper_detection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report, output, _ = run_build(Path(directory))
        manifest = report["bundle_manifests"][0]
        targets = [
            output / Path(manifest["files"][key])
            for key in ("features", "observations", "pairing", "provenance")
        ] + [
            output / Path(report["access_ledger"]["path"]),
            output / Path(report["harmonization_manifest"]["path"]),
            output / "registry" / "source_registry.lock.json",
        ]
        for path in targets:
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            raises(CorpusIntegrityError, verify_corpus, output)
            path.write_bytes(original)
        assert verify_corpus(output)["verified"] is True


def main() -> None:
    # Seal is recursive, token-normalized and checked before any fetcher call.
    for value in (
        "GSE315993", "gse-315993", "GSE%253315993/counts.mtx.gz",
        "%47%53%45%33%31%35%39%39%33", "ＧＳＥ３１５９９３",
        "R1NFMzE1OTkz", ["GSE31", "5993"], {"a": "GSE31", "b": "5993"},
        {"url": "https://fixture.test/?filename=GSE315993%2Etsv"},
        ["reference", {"source_lineage": "GSE_315993_ATAC"}],
    ):
        raises(SealedSourceRefused, assert_unsealed, value, purpose="synthetic test")

    bad_registry = registry()
    bad_registry["studies"][0]["artifacts"][0]["url"] = (
        "https://fixture.test/GSE%253315993_counts.tsv.gz"
    )
    raises(SealedSourceRefused, validate_registry, bad_registry)
    exercise_network_seals()
    print("PASS: sealed accession refused before URL/body/file materialization")

    for unsafe in ("../escape.tsv", "/tmp/escape.tsv", "safe/../escape.tsv", r"..\escape.tsv", "%2e%2e%2fescape.tsv"):
        for location in ("accession", "artifact_id", "filename", "subcohort", "matrix_scope"):
            unsafe_registry = registry()
            if location == "accession":
                unsafe_registry["studies"][0]["accession"] = unsafe
            else:
                unsafe_registry["studies"][0]["artifacts"][0][location] = unsafe
            raises(ValueError, validate_registry, unsafe_registry)
    print("PASS: registry path components cannot escape the corpus root")

    # Token-aware leakage checks reject direct proxies but allow a safe cue.
    audit_encoder_feature_names(["ATAC_peaks", "external_cue"], ["external_cue"])
    for proxy in (
        "cell_type", "cellType", "cluster", "pseudotime", "donor_id",
        "animal-ID", "sample_id", "barcode", "target_RNA", "future_ATAC", "UMAP_1",
        "RNA_counts", "rna_expression", "cell_state", "state",
        "integrated_embedding", "cell_embedding", "response_label", "outcome",
    ):
        raises(ValueError, audit_encoder_feature_names, [proxy])
    raises(ValueError, audit_encoder_feature_names, ["cell_type"], ["cell_type"])
    print("PASS: identity, state, future-outcome and grouping proxies stay outside encoder")

    # Known zero is not missing; inference needs lineage and uncertainty.
    validate_context_record({
        "provenance_state": "observed", "known": True, "value": 0.0,
        "uncertainty": 0.0, "evidence_weight": 1.0, "source_accessions": ["DEVTRAIN1"],
    })
    validate_context_record({
        "provenance_state": "unknown", "known": False, "value": None,
        "uncertainty": 0.0, "evidence_weight": 0.0, "source_accessions": [],
    })
    validate_context_record({
        "provenance_state": "model_inferred", "known": True, "value": 0.4,
        "uncertainty": 0.2, "evidence_weight": 0.7,
        "method_lineage": "training_only_spatial_transfer", "source_accessions": ["DEVSPACE1"],
    })
    raises(ValueError, validate_context_record, {
        "provenance_state": "model_inferred", "known": True, "value": 0.4,
        "uncertainty": 0.0, "evidence_weight": 1.0, "source_accessions": ["DEVSPACE1"],
    })
    validate_initial_provenance([{
        "feature_name": "ATAC_peaks", "measurement_time": 0.0,
        "method_lineage": "observed_at_origin", "source_accessions": ["DEVTRAIN1"],
    }], 0.0)
    raises(ValueError, validate_initial_provenance, [{
        "feature_name": "ATAC_peaks", "measurement_time": 24.0,
        "method_lineage": "future_RNA_regression", "source_accessions": ["DEVTRAIN1"],
    }], 0.0)
    print("PASS: observed zero, unknown, inferred uncertainty and temporal provenance")

    # GSE121737-like deposited sample prefixes define groups before cells.
    with tempfile.TemporaryDirectory() as dense_directory:
        dense_root = Path(dense_directory)
        barcode_path = dense_root / "barcodes.tsv.gz"
        matrix_path = dense_root / "counts.tsv.gz"
        write_gzip(
            barcode_path,
            "S1_bcA\tAAAA\nS1_bcB\tAAAB\nS2_bcA\tBBBB\n"
            "S3_bcA\tCCCC\nS4_bcA\tDDDD\n",
        )
        write_gzip(
            matrix_path,
            "gene\tS1_bcA\tS1_bcB\tS2_bcA\tS3_bcA\tS4_bcA\n"
            "TRAIN_VAR\t1\t2\t0\t3\t4\n"
            "STABLE\t1\t1\t1\t1\t1\n",
        )
        dense_study = {
            "accession": "DENSEDEV1",
            "adapter": "dense_sc_barcode_maps",
            "source_assembly": "AXO_DENSE_V1",
            "measurement_level": "single_cell",
            "split_policy": {"mode": "group_first", "validation_fraction": 0.5, "seed": 61},
            "sample_map": [
                {"experimental_unit_id": "DENSEDEV1:S1", "matrix_scope": "scope", "barcode_prefix": "S1", "stage": "early", "time_dpa": 3},
                {"experimental_unit_id": "DENSEDEV1:S2", "matrix_scope": "scope", "barcode_prefix": "S2", "stage": "early", "time_dpa": 3},
                {"experimental_unit_id": "DENSEDEV1:S3", "matrix_scope": "scope", "barcode_prefix": "S3", "stage": "late", "time_dpa": 23},
                {"experimental_unit_id": "DENSEDEV1:S4", "matrix_scope": "scope", "barcode_prefix": "S4", "stage": "late", "time_dpa": 23},
            ],
            "artifacts": [
                {"artifact_id": "dense_barcodes", "role": "cell_barcode_map", "matrix_scope": "scope"},
                {"artifact_id": "dense_counts", "role": "single_cell_gene_counts", "matrix_scope": "scope"},
            ],
        }
        parsed = parse_dense_sc_barcode_maps(
            {"dense_barcodes": barcode_path, "dense_counts": matrix_path}, dense_study
        )
        assert len(parsed) == 1
        _, dense_block = parsed[0]
        assert dense_block.matrix.shape == (5, 2)
        group_partitions = {}
        for observation in dense_block.observations:
            group_partitions.setdefault(observation["group_id"], observation["partition"])
            assert group_partitions[observation["group_id"]] == observation["partition"]
        assert set(group_partitions.values()) == {"train", "validation"}
        assert dense_block.pairing["mode"] == "unpaired_population"
    print("PASS: deposited cell-prefix parsing and stage-stratified specimen split")

    exercise_crash_resume()
    print("PASS: per-artifact receipts resume an interrupted multi-file study")

    with tempfile.TemporaryDirectory() as directory_a, tempfile.TemporaryDirectory() as directory_b:
        report_a, output_a, fetcher_a = run_build(Path(directory_a), 1.0)
        verification = verify_corpus(output_a)
        assert verification["verified"] and verification["bundles_verified"] == 3
        assert fetcher_a.calls == 6
        assert report_a["claims"]["development_assay_values_downloaded"] is True
        assert report_a["claims"]["gse315993_measurement_values_materialized"] is False
        assert report_a["claims"]["model_trained"] is False
        assert report_a["claims"]["digital_twin_claim"] is False
        assert report_a["claims"]["attractor_claim"] is False

        # Exact same-spot pairing is barcode-based; one specimen remains one group.
        spatial_manifest = next(
            value for value in report_a["bundle_manifests"] if value["study_accession"] == "DEVSPACE1"
        )
        pairing = json.loads((output_a / Path(spatial_manifest["files"]["pairing"])).read_text())
        assert pairing["mode"] == "same_spot_exact"
        assert pairing["verification_status"] == "verified_from_materialized_schema"
        assert pairing["unused_coordinate_rows"] == 1
        with gzip.open(
            output_a / Path(spatial_manifest["files"]["features"]),
            "rt", encoding="utf-8", newline="",
        ) as handle:
            feature_rows = list(csv.DictReader(handle, delimiter="\t"))
        assert [row["feature_id"] for row in feature_rows] == ["f1", "f2", "f3"]
        assert feature_rows[0]["feature_name"] == feature_rows[1]["feature_name"]
        provenance = json.loads(
            (output_a / Path(spatial_manifest["files"]["provenance"])).read_text()
        )
        assert provenance["raw_unnormalized"] is True
        assert provenance["study_accession"] == "DEVSPACE1"
        assert provenance["source_artifacts"]
        split = report_a["split_manifest"]
        assert split["partitions"]["reference"] == ["DEVSPACE1:section_1"]
        train = set(split["partitions"]["train"])
        validation = set(split["partitions"]["validation"])
        reference = set(split["partitions"]["reference"])
        assert train.isdisjoint(validation | reference)
        assert validation.isdisjoint(reference)
        print("PASS: exact spatial barcode pairing and biological-group-first partitions")

        # Changing validation-only values cannot alter training-selected features.
        report_b, output_b, _ = run_build(Path(directory_b), 10_000.0)
        assert (
            report_a["feature_registry"]["content_sha256"]
            == report_b["feature_registry"]["content_sha256"]
        )
        registry_a = json.loads(
            (output_a / report_a["feature_registry"]["path"]).read_text()
        )
        train_features = {
            value["feature_id"]
            for values in registry_a["registries"].values()
            for value in values
        }
        assert "TRAIN_VAR" in train_features
        assert "HOLDOUT_SPIKE" not in train_features
        print("PASS: feature selection is invariant to validation-only value changes")

        # A complete rerun uses locks and performs zero fetches.
        rerun_fetcher = FixtureFetcher(make_sources(Path(directory_a) / "rerun_sources"))
        registry_path = Path(directory_a) / "registry.json"
        build_from_registry(registry_path, output_a, fetcher=rerun_fetcher)
        assert rerun_fetcher.calls == 0
        verify_corpus(output_a)
        print("PASS: restart reuses checksum-valid immutable sources")

        # Corruption is detected, never silently overwritten by a resumed fetch.
        raw_file = output_a / "raw" / "DEVTRAIN1" / "train.tsv.gz"
        with raw_file.open("ab") as handle:
            handle.write(b"corruption")
        raises(CorpusIntegrityError, verify_corpus, output_a)
        raises(
            CorpusIntegrityError,
            build_from_registry,
            registry_path,
            output_a,
            fetcher=rerun_fetcher,
        )
        print("PASS: source corruption and source-lock disagreement fail closed")

    print("PASS: sparse unnormalized corpus, provenance, splits and non-claims")
    exercise_derived_tamper_detection()
    print("PASS: derived tables, provenance and global lineage hashes detect tampering")
    print("COMPLETE: WLD v6.1 axolotl-corpus synthetic contract passed")


if __name__ == "__main__":
    main()
