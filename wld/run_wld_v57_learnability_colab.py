"""Colab entry point for the WLD v5.7 response-learnability ladder.

This stage diagnoses whether GSE161002 contains a reproducible and
whole-target-generalizable chromatin response before another WLD architecture
is trained.  It may fit the prespecified diagnostic baselines implemented by
``wld_response_learnability_v57.py``; it never fits or updates a WLD model and
never materializes the sealed v5.3 test partition.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Mapping


EXPECTED_AUDIT_SCHEMA = "wld_v56_practical_effect_audit_v1"
EXPECTED_V57_SCHEMA = "wld-v5.7-response-learnability-development"
EXPECTED_REPORT = "wld_v57_response_learnability_report.json"


def require_file(path: Path, label: str) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def read_object(path: Path, label: str) -> Mapping[str, object]:
    require_file(path, label)
    value = json.loads(Path(path).read_text())
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def validate_v56_lineage(audit: Mapping[str, object]) -> None:
    if audit.get("schema_version") != EXPECTED_AUDIT_SCHEMA:
        raise RuntimeError("The v5.6 practical audit has an unsupported schema")
    data_contract = audit.get("data_contract")
    decision = audit.get("decision")
    claims = audit.get("claims")
    source = audit.get("source")
    if not all(
        isinstance(value, Mapping)
        for value in (data_contract, decision, claims, source)
    ):
        raise RuntimeError("The v5.6 practical audit is missing its contract mappings")
    assert isinstance(data_contract, Mapping)
    assert isinstance(decision, Mapping)
    assert isinstance(claims, Mapping)
    assert isinstance(source, Mapping)
    if any(
        data_contract.get(name) is not False
        for name in (
            "test_targets_materialized",
            "test_targets_evaluated",
            "external_subject_study_evaluated",
            "digital_twin_claim",
            "attractor_claim",
            "training_performed",
        )
    ):
        raise RuntimeError("The v5.6 lineage crossed a sealed or claim boundary")
    if data_contract.get("development_only") is not True:
        raise RuntimeError("The v5.6 lineage is not explicitly development-only")
    if any(
        claims.get(name) is not False
        for name in (
            "inference",
            "confidence_interval_claim",
            "p_value_claim",
            "digital_twin_claim",
            "attractor_claim",
            "confirmation_claim",
        )
    ):
        raise RuntimeError("The v5.6 audit contains a prohibited scientific claim")
    if source.get("source_report_immutable") is not True:
        raise RuntimeError("The completed v5.6 source report is not locked immutable")
    if decision.get("open_sealed_test") is not False:
        raise RuntimeError("The v5.6 audit no longer keeps the sealed test closed")
    if decision.get("eligible_to_freeze_new_confirmation_plan") is not False:
        raise RuntimeError(
            "v5.7 learnability diagnosis is locked to the failed v5.6 practical gate"
        )


def validate_upstream_data_contracts(
    v53_manifest: Mapping[str, object],
    split_contract: Mapping[str, object],
    module_manifest: Mapping[str, object],
) -> None:
    if v53_manifest.get("schema_version") != "wld-v5.3-crispr-sciatac-ingestion":
        raise RuntimeError("The v5.3 response bundle has an unsupported schema")
    v53_claims = v53_manifest.get("claims")
    leakage = v53_manifest.get("leakage_contract")
    if not isinstance(v53_claims, Mapping) or not isinstance(leakage, Mapping):
        raise RuntimeError("The v5.3 response bundle has no leakage/claim contract")
    if any(
        v53_claims.get(name) is not False
        for name in (
            "model_trained",
            "test_evaluated",
            "muscle_J_L_evaluated",
            "attractor_claim",
        )
    ) or any(
        leakage.get(name) is not False
        for name in (
            "guide_identity_in_encoder",
            "target_identity_in_encoder",
            "cell_type_label_in_encoder",
            "test_values_used_for_feature_selection",
        )
    ):
        raise RuntimeError("The v5.3 response bundle crossed a leakage/claim boundary")
    if leakage.get("split_before_feature_selection") is not True or leakage.get(
        "whole_target_split"
    ) is not True:
        raise RuntimeError("The v5.3 response bundle was not split before feature selection")

    rosters = split_contract.get("targets")
    if (
        not isinstance(rosters, Mapping)
        or set(rosters) != {"train", "validation", "test"}
        or split_contract.get("test_evaluated") is not False
    ):
        raise RuntimeError("The v5.3 whole-target split is invalid or unsealed")
    normalized = {}
    for split in ("train", "validation", "test"):
        values = rosters[split]
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"The v5.3 {split} target roster is invalid")
        targets = {str(value).strip().upper() for value in values}
        if not all(targets) or len(targets) != len(values) or "NTC" in targets:
            raise RuntimeError(f"The v5.3 {split} target roster contains invalid targets")
        normalized[split] = targets
    if any(
        normalized[left] & normalized[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise RuntimeError("The v5.3 whole-target rosters overlap")

    module_claims = module_manifest.get("claims")
    if not isinstance(module_claims, Mapping) or any(
        module_claims.get(name) is not False
        for name in (
            "validation_values_used_for_module_construction",
            "test_values_materialized",
            "test_values_used",
            "model_trained",
            "attractor_claim",
        )
    ):
        raise RuntimeError("The v5.5 module atlas crossed a leakage/claim boundary")


def validate_v57_report(report: Mapping[str, object]) -> Mapping[str, object]:
    if report.get("schema_version") != EXPECTED_V57_SCHEMA:
        raise RuntimeError("The v5.7 report has an unsupported schema")
    claims = report.get("claims")
    if not isinstance(claims, Mapping):
        raise RuntimeError("The v5.7 report has no claims mapping")
    required_true = ("development_only", "historical_wld_results_only")
    required_false = (
        "fresh_wld_training",
        "test_values_materialized",
        "test_targets_evaluated",
        "confirmatory_inference",
        "digital_twin_claim",
        "attractor_claim",
    )
    bad_true = [name for name in required_true if claims.get(name) is not True]
    bad_false = [name for name in required_false if claims.get(name) is not False]
    if bad_true or bad_false:
        raise RuntimeError(
            "The v5.7 report crossed its diagnostic-only claim boundary: "
            + ", ".join(bad_true + bad_false)
        )
    decision = report.get("decision")
    provenance = report.get("provenance")
    if not isinstance(decision, Mapping) or decision.get("open_sealed_test") is not False:
        raise RuntimeError("The v5.7 report no longer keeps the sealed test closed")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("materialized_splits") != ["train", "validation"]
        or provenance.get("test_values_materialized") is not False
    ):
        raise RuntimeError("The v5.7 report has unsafe or incomplete data provenance")
    boundary = provenance.get("sealed_boundary")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(name) is not False
        for name in (
            "test_metadata_fragments_field_used",
            "test_csr_data_or_indices_materialized",
            "test_csr_row_pointer_values_materialized",
        )
    ):
        raise RuntimeError("The v5.7 report lacks strict sparse-payload sealing evidence")
    diagnosis = report.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        raise RuntimeError("The v5.7 report has no structured diagnosis")
    primary = diagnosis.get("primary_failure_class")
    next_action = diagnosis.get("next_action")
    if not isinstance(primary, str) or not primary.strip():
        raise RuntimeError("The v5.7 diagnosis has no primary failure class")
    if not isinstance(next_action, str) or not next_action.strip():
        raise RuntimeError("The v5.7 diagnosis has no next action")
    return diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--v53-bundle", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--module-root", type=Path, required=True)
    parser.add_argument("--v56-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    print("WLD V5.7 RESPONSE-LEARNABILITY LADDER", flush=True)
    print(f"Python {platform.python_version()}", flush=True)
    print(
        "No fresh WLD training. Previously inspected validation targets remain "
        "development data; the v5.3 test partition remains sealed.\n",
        flush=True,
    )

    prior_root = args.phase_b_root / "priors" / "homo_sapiens_grch38"
    prerequisites = (
        (prior_root / "prior_manifest.json", "Phase B human prior manifest"),
        (prior_root / "foundation_priors.npz", "Phase B numerical priors"),
        (prior_root / "feature_vocab.json", "Phase B feature vocabulary"),
        (args.v53_bundle / "wld_v53_ingestion_manifest.json", "v5.3 manifest"),
        (args.v53_bundle / "whole_target_split.json", "v5.3 whole-target split"),
        (args.v53_bundle / "atac_counts.GRCh38.2kb.npz", "v5.3 sparse ATAC"),
        (args.v53_bundle / "cells.tsv.gz", "v5.3 cell metadata"),
        (args.v53_bundle / "bins.GRCh38.2kb.tsv.gz", "v5.3 response bins"),
        (args.route_root / "route_manifest.json", "v5.5 TF-route manifest"),
        (args.route_root / "route_vocab.json", "v5.5 TF-route vocabulary"),
        (args.route_root / "regulator_tf_routes.npz", "v5.5 TF-route tensors"),
        (args.route_root / "regulator_tf_routes.tsv.gz", "v5.5 TF-route table"),
        (
            args.module_root / "complex_accessibility_module_manifest.json",
            "v5.5 complex-module manifest",
        ),
        (
            args.module_root / "complex_accessibility_vocab.json",
            "v5.5 complex-module vocabulary",
        ),
        (
            args.module_root / "complex_accessibility_modules.npz",
            "v5.5 complex-module tensors",
        ),
        (args.v56_audit, "v5.6 practical-effect audit"),
    )
    for path, label in prerequisites:
        require_file(path, label)
    validate_upstream_data_contracts(
        read_object(
            args.v53_bundle / "wld_v53_ingestion_manifest.json", "v5.3 manifest"
        ),
        read_object(args.v53_bundle / "whole_target_split.json", "v5.3 split"),
        read_object(
            args.module_root / "complex_accessibility_module_manifest.json",
            "v5.5 complex-module manifest",
        ),
    )
    validate_v56_lineage(read_object(args.v56_audit, "v5.6 practical-effect audit"))
    print("PASS: durable priors, response bundle, routes, modules and failure lineage", flush=True)

    sibling = Path(__file__).resolve().parent
    smoke = sibling / "run_wld_v57_learnability_smoke.py"
    core = sibling / "wld_response_learnability_v57.py"
    require_file(smoke, "v5.7 learnability smoke suite")
    require_file(core, "v5.7 learnability implementation")

    print("\n1. Running the synthetic learnability and leakage contract...", flush=True)
    subprocess.run([sys.executable, str(smoke)], check=True)

    print("\n2. Running the real response-learnability ladder...", flush=True)
    print(
        "   Diagnostic baselines may be fitted on training targets only; no WLD "
        "checkpoint is initialized or updated.",
        flush=True,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(core),
            "--v53-bundle",
            str(args.v53_bundle),
            "--route-root",
            str(args.route_root),
            "--module-root",
            str(args.module_root),
            "--v56-audit",
            str(args.v56_audit),
            "--output-root",
            str(args.output_root),
        ],
        check=True,
    )

    report_path = args.output_root / EXPECTED_REPORT
    report = read_object(report_path, "completed v5.7 learnability report")
    diagnosis = validate_v57_report(report)

    print("\n" + "=" * 78, flush=True)
    print("VERIFIED COMPLETE: WLD V5.7 RESPONSE-LEARNABILITY LADDER", flush=True)
    print("=" * 78, flush=True)
    print(f"Primary diagnosis: {diagnosis['primary_failure_class']}", flush=True)
    flags = diagnosis.get("flags", [])
    if isinstance(flags, list) and flags:
        print("Diagnostic flags: " + ", ".join(map(str, flags)), flush=True)
    print(f"Next action: {diagnosis['next_action']}", flush=True)
    print("Fresh WLD training:             False", flush=True)
    print("Sealed test values materialized: False", flush=True)
    print("Confirmatory inference:         False", flush=True)
    print("Digital-twin claim:             False", flush=True)
    print("Attractor claim:                False", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
