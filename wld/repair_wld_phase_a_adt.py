"""Repair the GSE158013 ADT orientation without rebuilding other cohorts.

The v1 Phase A adapter used barcode overlap to infer CSV orientation.  The
GSE158013 ADT table prefixes filtered 10x barcodes, making overlap zero and
causing a 47 x 720,873 transposition.  This repair streams the source table,
normalizes barcode prefixes, retains only filtered H5 cells, and surgically
rebuilds the protein block and human protein atlas.  Replaced artifacts are
moved to a quarantine directory and are never deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

from wld_foundation_data import (
    atomic_json,
    build_training_atlas,
    pairing_report,
    project_modality_to_atlas,
    read_10x_h5,
    read_adt_csv,
    read_barcodes,
    save_bundle,
    sha256_file,
    verify_bundle,
)


COHORT_ID = "GSE158013_GSM5123951_TEA"
ATLAS_SLUG = "homo_sapiens_grch38"
REPAIR_ID = "gse158013_adt_orientation_v2"


def cohort_spec(registry: dict) -> dict:
    return next(value for value in registry["cohorts"] if value["cohort_id"] == COHORT_ID)


def role_path(registry: dict, root: Path, role: str) -> Path:
    spec = cohort_spec(registry)
    file_spec = next(value for value in spec["files"] if value["role"] == role)
    return root / "raw" / COHORT_ID / file_spec["name"]


def block_barcodes(bundle_root: Path, manifest: dict) -> dict:
    return {
        modality: SimpleNamespace(
            barcodes=read_barcodes(bundle_root / "modalities" / modality / "barcodes.tsv.gz")
        )
        for modality in manifest["modalities"]
    }


def install_protein_block(
    root: Path,
    bundle_root: Path,
    block,
    quarantine_root: Path,
    label: str,
) -> dict:
    manifest = json.loads((bundle_root / "bundle_manifest.json").read_text())
    temporary = root / "_repair_tmp" / label
    if temporary.exists():
        shutil.rmtree(temporary)
    cohort = {
        "cohort_id": manifest["cohort_id"],
        "study_id": manifest["study_id"],
        "species": manifest["species"],
        "genome_build": manifest["genome_build"],
        "adapter": manifest["adapter"],
        "donor_scope": manifest.get("donor_scope", ""),
    }
    temporary_manifest = save_bundle(
        temporary, cohort, {"protein": block}, manifest["source_lock"]
    )
    target = bundle_root / "modalities" / "protein"
    backup = quarantine_root / label / "protein"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not backup.exists():
        shutil.move(str(target), str(backup))
    elif target.exists():
        shutil.rmtree(target)
    shutil.move(str(temporary / "modalities" / "protein"), str(target))
    manifest_backup = quarantine_root / label / "bundle_manifest.json"
    if not manifest_backup.exists():
        shutil.copy2(bundle_root / "bundle_manifest.json", manifest_backup)
    manifest["modalities"]["protein"] = temporary_manifest["modalities"]["protein"]
    manifest["adapter_repair"] = {
        "repair_id": REPAIR_ID,
        "reason": "streamed cells x antibodies orientation with normalized filtered 10x barcodes",
    }
    manifest["pairing"] = pairing_report(block_barcodes(bundle_root, manifest))
    atomic_json(bundle_root / "bundle_manifest.json", manifest)
    verify_bundle(bundle_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.sources.read_text())
    root = args.root
    bundle_root = root / "bundles" / COHORT_ID
    harmonized_root = root / "harmonized" / ATLAS_SLUG / COHORT_ID
    atlas_root = root / "training_atlas" / ATLAS_SLUG
    report_path = root / "phase_a_ingestion_report.json"
    for path in (bundle_root / "bundle_manifest.json", harmonized_root / "bundle_manifest.json", atlas_root / "atlas_manifest.json", report_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase A artifact is missing: {path}")

    current = verify_bundle(bundle_root)
    harmonized_current = verify_bundle(harmonized_root)
    report_current = json.loads(report_path.read_text())
    protein_shape = current["modalities"]["protein"]["shape"]
    harmonized_shape = harmonized_current["modalities"]["protein"]["shape"]
    atlas = json.loads((atlas_root / "atlas_manifest.json").read_text())
    fully_repaired = (
        protein_shape[0] > 1000
        and protein_shape[1] < 5000
        and harmonized_shape[0] > 1000
        and harmonized_shape[1] < 5000
        and 0 < int(atlas["proteins"]) < 5000
        and REPAIR_ID in report_current.get("repairs", [])
        and report_current.get("model_trained") is False
        and report_current.get("sealed_test_downloaded") is False
    )
    if fully_repaired:
        print("PASS: GSE158013 ADT orientation was already repaired")
        return

    print(f"Detected invalid protein shape: {protein_shape}", flush=True)
    h5_path = role_path(registry, root, "tenx_h5")
    adt_path = role_path(registry, root, "adt_csv")
    if not h5_path.is_file() or not adt_path.is_file():
        raise FileNotFoundError("The locked GSE158013 H5/ADT sources are missing")
    h5_blocks = read_10x_h5(h5_path)
    expected = h5_blocks["rna"].barcodes
    protein = read_adt_csv(adt_path, expected)
    if protein.matrix.shape[0] < int(0.5 * len(expected)):
        raise RuntimeError(
            f"Only {protein.matrix.shape[0]} of {len(expected)} filtered cells matched ADT"
        )
    if protein.matrix.shape[1] >= 5000:
        raise RuntimeError(f"Implausible corrected protein feature count: {protein.matrix.shape}")
    print(f"Corrected raw protein shape: {protein.matrix.shape}", flush=True)

    quarantine = root / "quarantine" / REPAIR_ID
    install_protein_block(root, bundle_root, protein, quarantine, "raw_bundle")
    print("PASS: corrected raw protein bundle; original quarantined", flush=True)

    # Rebuild the human atlas from every completed *training* human bundle.
    studies = registry["studies"]
    training_roots = []
    for path in sorted((root / "bundles").glob("*/bundle_manifest.json")):
        manifest = verify_bundle(path.parent)
        if (
            studies.get(manifest["study_id"], {}).get("split") == "train"
            and manifest["species"] == "Homo sapiens"
            and manifest["genome_build"] == "GRCh38"
        ):
            training_roots.append(path.parent)
    temporary_atlas = root / "_repair_tmp" / "human_atlas"
    if temporary_atlas.exists():
        shutil.rmtree(temporary_atlas)
    new_atlas = build_training_atlas(
        training_roots, temporary_atlas, max_genes=20000, max_peak_bins=200000
    )
    if not 0 < int(new_atlas["proteins"]) < 5000:
        raise RuntimeError(f"Corrected human protein atlas is implausible: {new_atlas['proteins']}")
    atlas_backup = quarantine / "training_atlas"
    atlas_backup.parent.mkdir(parents=True, exist_ok=True)
    if atlas_root.exists() and not atlas_backup.exists():
        shutil.move(str(atlas_root), str(atlas_backup))
    elif atlas_root.exists():
        shutil.rmtree(atlas_root)
    atlas_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temporary_atlas), str(atlas_root))
    print(f"PASS: rebuilt human atlas with {new_atlas['proteins']} proteins", flush=True)

    corrected_projected = project_modality_to_atlas(bundle_root, atlas_root, "protein")
    if corrected_projected is None:
        raise RuntimeError("Corrected protein block did not project into the human atlas")
    harmonized_manifest = install_protein_block(
        root, harmonized_root, corrected_projected, quarantine, "harmonized_bundle"
    )

    # Every harmonized human bundle now points to the repaired atlas manifest.
    atlas_sha = sha256_file(atlas_root / "atlas_manifest.json")
    for path in sorted((root / "harmonized" / ATLAS_SLUG).glob("*/bundle_manifest.json")):
        manifest = json.loads(path.read_text())
        backup = quarantine / "harmonized_manifests" / manifest["cohort_id"] / "bundle_manifest.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        manifest["source_lock"]["training_atlas_manifest"]["sha256"] = atlas_sha
        if manifest["cohort_id"] == COHORT_ID:
            manifest["source_lock"]["source_bundle_manifest"]["sha256"] = sha256_file(bundle_root / "bundle_manifest.json")
            manifest["modalities"]["protein"] = harmonized_manifest["modalities"]["protein"]
            manifest["adapter_repair"] = harmonized_manifest["adapter_repair"]
        atomic_json(path, manifest)
        verify_bundle(path.parent)
    print(f"PASS: harmonized protein shape {corrected_projected.matrix.shape}", flush=True)

    report = json.loads(report_path.read_text())
    report["training_atlases"][ATLAS_SLUG] = new_atlas
    report["harmonized_cohorts"][COHORT_ID]["modalities"]["protein"] = harmonized_manifest["modalities"]["protein"]
    report["repairs"] = report.get("repairs", [])
    if REPAIR_ID not in report["repairs"]:
        report["repairs"].append(REPAIR_ID)
    report["model_trained"] = False
    report["sealed_test_downloaded"] = False
    atomic_json(report_path, report)
    print("\nCOMPLETE: GSE158013 ADT orientation repaired", flush=True)
    print(f"Quarantine backup: {quarantine}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
