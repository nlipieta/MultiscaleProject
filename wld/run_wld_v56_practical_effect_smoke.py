"""Synthetic regression tests for the WLD v5.6 practical-effect audit."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from wld_v56_practical_effect_audit import audit_payload, run_audit


TARGETS = ("ARID1A", "CHD4", "SMARCA4", "TET2")
SEEDS = (42, 137, 911)


def fixture(*, meaningful: bool) -> dict:
    if meaningful:
        true = 0.090
        persistence = 0.100
        control = 0.095
        frozen_all = 0.096
        nrmse = 0.84
        cosine = 0.24
    else:
        true = 0.100000000000
        persistence = true + 1e-10
        control = true + 1e-10
        frozen_all = true + 1e-10
        nrmse = 0.9999999615053335
        cosine = 0.011501414725595774
    perturbed_mean = true + (0.010 if meaningful else 0.00023467)
    rows = []
    for target in TARGETS:
        for seed in SEEDS:
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "route_supported": True,
                    "true_swd": true,
                    "persistence_swd": persistence,
                    "persistence_minus_true": persistence - true,
                    "perturbed_mean_swd": perturbed_mean,
                    "perturbed_mean_minus_true": perturbed_mean - true,
                    "control_mean_swd": control,
                    "control_mean_minus_true": control - true,
                    "frozen_tf_minus_true": (frozen_all - true) / 2.0,
                    "frozen_complex_minus_true": (frozen_all - true) / 2.0,
                    "frozen_all_minus_true": frozen_all - true,
                    "response_nrmse": nrmse,
                    "response_cosine": cosine,
                }
            )
    return {
        "schema_version": "wld-v5.6-null-aware-reused-development",
        "config": {
            "seeds": list(SEEDS),
            "control_replicates": 10,
            "minimum_control_replicates": 10,
        },
        "reused_development_targets": list(TARGETS),
        "paired_reused_development_metrics": rows,
        "training_only_perturbed_mean_baseline": {
            "validation_values_used_for_construction": False,
            "test_values_materialized": False,
        },
        "conditions": {
            "true_null_aware_routes": {str(seed): {} for seed in SEEDS},
            **{
                f"matched_control_{index:02d}": {str(seed): {} for seed in SEEDS}
                for index in range(1, 11)
            },
        },
        "per_control_descriptive_effects": [
            {
                "condition": f"matched_control_{index:02d}",
                "mean_control_minus_true": 0.005 if meaningful else 1e-10,
                "median_control_minus_true": 0.005 if meaningful else 1e-10,
                "target_seed_rows": len(TARGETS) * len(SEEDS),
            }
            for index in range(1, 11)
        ],
        "fitted_true_branch_gates_by_seed": {
            str(seed): {"tf": 0.0015, "complex": 0.0012} for seed in SEEDS
        },
        "frozen_true_model_evaluations": {
            str(seed): {
                "all_routes_removed": {
                    "per_target": [
                        {
                            "target": target,
                            "cells": 96,
                            "observed_response_rms": 0.01,
                            "response_nrmse_floor_used": 0.0,
                            "persistence_swd": (
                                0.100 if meaningful else 0.1000000001
                            ),
                        }
                        for target in TARGETS
                    ]
                }
            }
            for seed in SEEDS
        },
        "development_checks": {
            "eligible_to_freeze_new_confirmation_plan": True,
        },
        "claims": {
            "development_only": True,
            "validation_targets_previously_used_in_v55": True,
            "all_existing_validation_targets_evaluated": True,
            "untouched_audit_inference": False,
            "confidence_interval_claim": False,
            "p_value_claim": False,
            "perturbed_mean_baseline_training_only": True,
            "test_targets_materialized": False,
            "test_targets_evaluated": False,
            "external_subject_study_evaluated": False,
            "ode_time_scale_identified": False,
            "fixed_point_claim": False,
            "basin_claim": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
        },
    }


def main() -> None:
    dust = audit_payload(
        fixture(meaningful=False),
        source_report_sha256="0" * 64,
        source_report_path="synthetic-dust.json",
    )
    assert dust["source"]["source_legacy_eligibility"] is True
    assert dust["numerical_checks"]["all_pass"] is False
    assert dust["practical_checks"]["all_pass"] is False
    assert dust["decision"]["legacy_flag_superseded"] is True
    assert dust["decision"]["eligible_to_freeze_new_confirmation_plan"] is False
    assert dust["decision"]["open_sealed_test"] is False

    meaningful = audit_payload(
        fixture(meaningful=True),
        source_report_sha256="1" * 64,
        source_report_path="synthetic-meaningful.json",
    )
    assert meaningful["numerical_checks"]["all_pass"] is True
    # Four targets are intentionally below the predeclared eight-target route
    # coverage requirement, so an otherwise meaningful small fixture fails closed.
    assert meaningful["practical_checks"]["all_pass"] is False
    assert meaningful["decision"]["eligible_to_freeze_new_confirmation_plan"] is False

    expanded = fixture(meaningful=True)
    for suffix in ("A", "B", "C", "D"):
        source = copy.deepcopy(
            [
                row
                for row in expanded["paired_reused_development_metrics"]
                if row["target"] == "ARID1A"
            ]
        )
        target = f"ARID1A_{suffix}"
        for row in source:
            row["target"] = target
        expanded["paired_reused_development_metrics"].extend(source)
        expanded["reused_development_targets"].append(target)
        for seed in SEEDS:
            expanded["frozen_true_model_evaluations"][str(seed)][
                "all_routes_removed"
            ]["per_target"].append(
                {
                    "target": target,
                    "cells": 96,
                    "observed_response_rms": 0.01,
                    "response_nrmse_floor_used": 0.0,
                    "persistence_swd": 0.100,
                }
            )
    for row in expanded["per_control_descriptive_effects"]:
        row["target_seed_rows"] = (
            len(expanded["reused_development_targets"]) * len(SEEDS)
        )
    eligible = audit_payload(
        expanded,
        source_report_sha256="5" * 64,
        source_report_path="synthetic-eligible.json",
    )
    assert eligible["practical_checks"]["all_pass"] is True
    assert eligible["decision"]["eligible_to_freeze_new_confirmation_plan"] is True
    assert eligible["decision"]["open_sealed_test"] is False

    unsafe = fixture(meaningful=True)
    unsafe["claims"]["test_targets_evaluated"] = True
    try:
        audit_payload(
            unsafe,
            source_report_sha256="2" * 64,
            source_report_path="synthetic-unsafe.json",
        )
    except ValueError as error:
        assert "sealed-development contract" in str(error)
    else:
        raise AssertionError("Unsafe source report was not rejected")

    nonfinite = fixture(meaningful=True)
    nonfinite["paired_reused_development_metrics"][0]["response_nrmse"] = float("nan")
    try:
        audit_payload(
            nonfinite,
            source_report_sha256="3" * 64,
            source_report_path="synthetic-nonfinite.json",
        )
    except ValueError as error:
        assert "not finite" in str(error)
    else:
        raise AssertionError("Non-finite metric was not rejected")

    ragged = fixture(meaningful=True)
    ragged["paired_reused_development_metrics"].pop()
    try:
        audit_payload(
            ragged,
            source_report_sha256="4" * 64,
            source_report_path="synthetic-ragged.json",
        )
    except ValueError as error:
        assert "ragged" in str(error)
    else:
        raise AssertionError("Ragged target/seed grid was not rejected")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "completed.json"
        output = root / "audit"
        source.write_text(json.dumps(fixture(meaningful=False)))
        first = run_audit(source, output)
        second = run_audit(source, output)
        assert first == second
        table_header = (output / "wld_v56_target_effects.tsv").read_text().splitlines()[0]
        assert "\t" in table_header and "," not in table_header

    print("PASS: numerical dust cannot trigger continuation eligibility")
    print("PASS: explicit practical effects can pass development gates")
    print("PASS: sealed-test, finite-metric, and complete-grid contracts")
    print("PASS: immutable restart-safe sidecar and tab-delimited target table")


if __name__ == "__main__":
    main()
