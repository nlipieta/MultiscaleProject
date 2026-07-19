"""Restart-safe corpus expansion for WLD foundation pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Mapping

from wld_corpus_expansion import (
    ingest_shareseq_legacy_pair,
    ingest_shareseq_metadata_pair,
    write_context_manifest,
)
from wld_foundation_data import (
    atomic_json,
    build_training_atlas,
    download_locked,
    project_bundle_to_atlas,
    save_bundle,
    sha256_file,
    verify_bundle,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _selection_digest(bundle_roots: list[Path]) -> str:
    digest = hashlib.sha256()
    for root in sorted(bundle_roots):
        manifest = root / "bundle_manifest.json"
        digest.update(str(root.name).encode())
        digest.update(sha256_file(manifest).encode())
    return digest.hexdigest()[:16]


def _validate_registry(registry: Mapping[str, object]) -> None:
    sealed = set(registry["sealed_exclusions"])
    seen = set()
    for cohort in registry["cohorts"]:
        cohort_id = cohort["cohort_id"]
        if cohort_id in seen:
            raise ValueError(f"Duplicate cohort id: {cohort_id}")
        seen.add(cohort_id)
        if cohort["study_id"] in sealed or cohort.get("split") == "sealed_test":
            raise RuntimeError(f"Sealed study appeared in expansion registry: {cohort['study_id']}")
        if cohort.get("split") != "train":
            raise ValueError(f"Expansion cohorts must be training-only: {cohort_id}")
        if not cohort.get("context_contract", {}).get("fold_local_context_encoding_required"):
            raise ValueError(f"Missing fold-local context contract: {cohort_id}")


def ingest_cohort(
    cohort: Mapping[str, object], raw_root: Path, bundle_root: Path
) -> Dict[str, object]:
    manifest_path = bundle_root / "bundle_manifest.json"
    context_path = bundle_root / "context_manifest.json"
    if manifest_path.is_file() and context_path.is_file():
        return verify_bundle(bundle_root)
    files, source_lock = download_locked(cohort["files"], raw_root)
    adapter = cohort["adapter"]
    if adapter == "shareseq_metadata_pair":
        blocks, pairing_evidence, context = ingest_shareseq_metadata_pair(files)
    elif adapter == "shareseq_legacy_pair":
        blocks, pairing_evidence, context = ingest_shareseq_legacy_pair(files)
    else:
        raise ValueError(f"Unsupported expansion adapter: {adapter}")
    resolved = dict(cohort)
    resolved["pairing_evidence"] = pairing_evidence
    manifest = save_bundle(bundle_root, resolved, blocks, source_lock)
    write_context_manifest(bundle_root, resolved, context)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).with_name("wld_corpus_expansion_sources.json"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tiers", nargs="+", default=["core", "extended"])
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--max-genes", type=int, default=20000)
    parser.add_argument("--max-peak-bins", type=int, default=200000)
    args = parser.parse_args()

    registry = json.loads(args.sources.read_text())
    _validate_registry(registry)
    selected_ids = set(args.include)
    selected_tiers = set(args.tiers)
    selected = [
        cohort for cohort in registry["cohorts"]
        if (not selected_ids and cohort["tier"] in selected_tiers)
        or (selected_ids and cohort["cohort_id"] in selected_ids)
    ]
    if not selected:
        raise ValueError("No corpus-expansion cohorts selected")

    args.root.mkdir(parents=True, exist_ok=True)
    completed = {}
    for cohort in selected:
        print(
            f"\nINGEST {cohort['cohort_id']} | {cohort['species']} | "
            f"{cohort['genome_build']} | {cohort['tissue']}",
            flush=True,
        )
        manifest = ingest_cohort(
            cohort,
            args.root / "raw" / cohort["cohort_id"],
            args.root / "bundles" / cohort["cohort_id"],
        )
        completed[cohort["cohort_id"]] = manifest
        print(
            f"PASS: modalities={sorted(manifest['modalities'])}; "
            f"pairing={manifest['pairing']['pairing']}",
            flush=True,
        )

    # Include every already-completed expansion bundle.  Atlas snapshots are
    # immutable and content-addressed, so adding cohorts never silently changes
    # the feature space beneath an earlier checkpoint.
    all_bundles = []
    for path in sorted((args.root / "bundles").glob("*/bundle_manifest.json")):
        manifest = verify_bundle(path.parent)
        if manifest.get("split") not in {"", "train"}:
            raise RuntimeError(f"Non-training bundle in expansion root: {path.parent}")
        if manifest["study_id"] in set(registry["sealed_exclusions"]):
            raise RuntimeError(f"Sealed bundle in expansion root: {path.parent}")
        if not (path.parent / "context_manifest.json").is_file():
            raise FileNotFoundError(f"Missing context manifest: {path.parent}")
        all_bundles.append(path.parent)

    grouped: Dict[tuple[str, str], list[Path]] = {}
    for root in all_bundles:
        manifest = json.loads((root / "bundle_manifest.json").read_text())
        grouped.setdefault((manifest["species"], manifest["genome_build"]), []).append(root)

    atlas_snapshots = {}
    harmonized = {}
    for (species, genome_build), roots in sorted(grouped.items()):
        group_slug = _slug(f"{species}_{genome_build}")
        snapshot = _selection_digest(roots)
        atlas_root = args.root / "atlas_snapshots" / group_slug / snapshot
        if (atlas_root / "atlas_manifest.json").is_file():
            atlas_manifest = json.loads((atlas_root / "atlas_manifest.json").read_text())
        else:
            atlas_manifest = build_training_atlas(
                roots,
                atlas_root,
                max_genes=args.max_genes,
                max_peak_bins=args.max_peak_bins,
            )
        atlas_snapshots[group_slug] = {
            "snapshot": snapshot,
            "path": str(atlas_root),
            "manifest": atlas_manifest,
            "cohorts": sorted(root.name for root in roots),
        }
        for root in roots:
            output = args.root / "harmonized_snapshots" / group_slug / snapshot / root.name
            if (output / "bundle_manifest.json").is_file():
                projected = verify_bundle(output)
            else:
                projected = project_bundle_to_atlas(root, atlas_root, output)
            harmonized[root.name] = {
                "atlas_group": group_slug,
                "atlas_snapshot": snapshot,
                "path": str(output),
                "modalities": projected["modalities"],
                "pairing": projected["pairing"],
            }

    report = {
        "schema_version": "1.0",
        "scope": "training-corpus expansion and immutable feature-atlas snapshots; no model assessment",
        "cohorts_completed": sorted(completed),
        "all_verified_expansion_bundles": sorted(root.name for root in all_bundles),
        "studies": {
            cohort["study_id"]: {
                "species": cohort["species"],
                "genome_build": cohort["genome_build"],
                "tissue": cohort["tissue"],
            }
            for cohort in registry["cohorts"]
            if cohort["cohort_id"] in {root.name for root in all_bundles}
        },
        "atlas_snapshots": atlas_snapshots,
        "harmonized_cohorts": harmonized,
        "staged_not_ingested": registry.get("staged_not_ingested", {}),
        "context_policy": {
            "observation_level_context_retained": True,
            "variable_biological_parameters_frozen_globally": False,
            "identity_or_state_proxy_encoder_inputs": False,
        },
        "sealed_test_studies": registry["sealed_exclusions"],
        "sealed_test_downloaded": False,
        "model_trained": False,
        "ode_kinetics_fitted": False,
        "attractor_claim": False,
    }
    atomic_json(args.root / "wld_corpus_expansion_report.json", report)
    print("\n" + "=" * 76)
    print("COMPLETE: WLD TRAINING-CORPUS EXPANSION")
    print("=" * 76)
    print(f"Verified bundles: {report['all_verified_expansion_bundles']}")
    print(f"Atlas groups: {sorted(atlas_snapshots)}")
    print("Context retained outside encoder; no cell identity/state proxy was added.")
    print("GSE183273 and GSE214546 remain sealed and were not downloaded.")
    print("No model metric or attractor claim was computed.")
    print(f"Report: {args.root / 'wld_corpus_expansion_report.json'}")


if __name__ == "__main__":
    main()
