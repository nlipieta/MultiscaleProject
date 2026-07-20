"""Restart-safe real-cohort ingestion for WLD v4 Phase A."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Mapping

from wld_foundation_data import (
    atomic_json,
    build_training_atlas,
    download_locked,
    read_10x_h5,
    read_10x_mtx,
    read_adt_csv,
    read_single_mtx,
    project_bundle_to_atlas,
    save_bundle,
    sha256_file,
    verify_bundle,
)


def ingest_downloaded_cohort(
    cohort: Mapping[str, object], raw_root: Path, bundle_root: Path
) -> Dict[str, object]:
    existing = bundle_root / "bundle_manifest.json"
    if existing.is_file():
        return verify_bundle(bundle_root)
    files, source_lock = download_locked(cohort["files"], raw_root)
    adapter = cohort["adapter"]
    if adapter == "tenx_h5_with_adt_csv":
        blocks = read_10x_h5(files["tenx_h5"])
        reference = blocks["rna"].barcodes if "rna" in blocks else blocks["atac"].barcodes
        blocks["protein"] = read_adt_csv(files["adt_csv"], reference)
    elif adapter == "tenx_mtx":
        blocks = read_10x_mtx(files["matrix"], files["barcodes"], files["features"])
    elif adapter == "paired_mtx":
        blocks = {
            "rna": read_single_mtx(files["rna_matrix"], files["rna_barcodes"], files["rna_features"], "rna"),
            "atac": read_single_mtx(files["atac_matrix"], files["atac_barcodes"], files["atac_features"], "atac"),
        }
    else:
        raise ValueError(f"Unsupported adapter {adapter}")
    return save_bundle(bundle_root, cohort, blocks, source_lock)


def ingest_muscle_export(export_root: Path, bundle_root: Path) -> Dict[str, object]:
    required = {
        "rna_matrix": export_root / "rna.mtx.gz",
        "rna_features": export_root / "genes.tsv",
        "rna_barcodes": export_root / "barcodes.tsv",
        "atac_matrix": export_root / "atac.mtx.gz",
        "atac_features": export_root / "peaks.tsv",
        "atac_barcodes": export_root / "barcodes.tsv",
    }
    missing = [str(value) for value in required.values() if not value.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete GSE240061 export: {missing}")
    cohort = {
        "cohort_id": "GSE240061_muscle_exercise",
        "study_id": "GSE240061",
        "species": "Homo sapiens",
        "genome_build": "GRCh38",
        "adapter": "existing_wld_export",
        "donor_scope": "six subjects; donor metadata retained outside encoder",
    }
    lock = {
        role: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for role, path in required.items()
    }
    blocks = {
        "rna": read_single_mtx(required["rna_matrix"], required["rna_barcodes"], required["rna_features"], "rna"),
        "atac": read_single_mtx(required["atac_matrix"], required["atac_barcodes"], required["atac_features"], "atac"),
    }
    return save_bundle(bundle_root, cohort, blocks, lock)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("wld_phase_a_sources.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--muscle-export", type=Path)
    parser.add_argument("--max-genes", type=int, default=20000)
    parser.add_argument("--max-peak-bins", type=int, default=200000)
    args = parser.parse_args()

    registry = json.loads(args.sources.read_text())
    studies = registry["studies"]
    selected = set(args.include)
    cohorts = [
        value for value in registry["cohorts"]
        if not selected or value["cohort_id"] in selected
    ]
    if not cohorts and not args.muscle_export:
        raise ValueError("No cohorts selected")
    args.root.mkdir(parents=True, exist_ok=True)
    reports = {}
    for cohort in cohorts:
        split = studies[cohort["study_id"]]["split"]
        if split == "sealed_test":
            raise RuntimeError(f"Refusing to ingest sealed test study {cohort['study_id']}")
        print(f"\nINGEST {cohort['cohort_id']} [{split}]", flush=True)
        manifest = ingest_downloaded_cohort(
            cohort,
            args.root / "raw" / cohort["cohort_id"],
            args.root / "bundles" / cohort["cohort_id"],
        )
        reports[cohort["cohort_id"]] = manifest
        print(f"PASS: {', '.join(sorted(manifest['modalities']))}", flush=True)

    if args.muscle_export:
        print("\nINGEST GSE240061_muscle_exercise [validation]", flush=True)
        manifest = ingest_muscle_export(
            args.muscle_export,
            args.root / "bundles" / "GSE240061_muscle_exercise",
        )
        reports["GSE240061_muscle_exercise"] = manifest
        print(f"PASS: {', '.join(sorted(manifest['modalities']))}", flush=True)

    # Always include every completed training bundle, including bundles restored
    # from an earlier Colab session.  Progressive runs therefore enlarge the
    # atlas instead of accidentally replacing it with the latest cohort.
    training_roots = []
    for path in sorted((args.root / "bundles").glob("*/bundle_manifest.json")):
        manifest = verify_bundle(path.parent)
        if studies.get(manifest["study_id"], {}).get("split") == "train":
            training_roots.append(path.parent)
    if not training_roots:
        raise RuntimeError("No training bundles available for atlas construction")
    grouped = {}
    for root in training_roots:
        manifest = json.loads((root / "bundle_manifest.json").read_text())
        key = (manifest["species"], manifest["genome_build"])
        grouped.setdefault(key, []).append(root)
    atlases = {}
    for (species, genome_build), roots in sorted(grouped.items()):
        slug = re.sub(r"[^a-z0-9]+", "_", f"{species}_{genome_build}".lower()).strip("_")
        atlases[slug] = build_training_atlas(
            roots,
            args.root / "training_atlas" / slug,
            max_genes=args.max_genes,
            max_peak_bins=args.max_peak_bins,
        )
    harmonized = {}
    for path in sorted((args.root / "bundles").glob("*/bundle_manifest.json")):
        source_manifest = verify_bundle(path.parent)
        if studies.get(source_manifest["study_id"], {}).get("split") == "sealed_test":
            raise RuntimeError("A sealed test bundle appeared in the development data root")
        slug = re.sub(
            r"[^a-z0-9]+", "_",
            f"{source_manifest['species']}_{source_manifest['genome_build']}".lower(),
        ).strip("_")
        if slug not in atlases:
            continue
        output_root = args.root / "harmonized" / slug / source_manifest["cohort_id"]
        if (output_root / "bundle_manifest.json").is_file():
            projected_manifest = verify_bundle(output_root)
        else:
            projected_manifest = project_bundle_to_atlas(
                path.parent, args.root / "training_atlas" / slug, output_root
            )
        harmonized[source_manifest["cohort_id"]] = {
            "atlas": slug,
            "modalities": projected_manifest["modalities"],
        }
    report = {
        "schema_version": "1.0",
        "scope": "real raw-count ingestion and training-only feature atlas; no biological model assessment",
        "cohorts_completed": sorted(reports),
        "training_atlases": atlases,
        "harmonized_cohorts": harmonized,
        "sealed_test_studies": sorted(key for key, value in studies.items() if value["split"] == "sealed_test"),
        "sealed_test_downloaded": False,
        "model_trained": False,
    }
    atomic_json(args.root / "phase_a_ingestion_report.json", report)
    print("\nCOMPLETE: Phase A ingestion and training-only atlas", flush=True)
    print(f"Report: {args.root / 'phase_a_ingestion_report.json'}", flush=True)
    print("No sealed test study was downloaded or evaluated.", flush=True)


if __name__ == "__main__":
    main()
