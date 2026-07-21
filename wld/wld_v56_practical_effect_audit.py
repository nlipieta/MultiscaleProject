"""Post-hoc practical-effect audit for a completed WLD v5.6 report.

The v5.6 development runner intentionally reused previously inspected targets.
Its original continuation rule, however, treated every positive floating-point
value as evidence.  This module leaves the completed report immutable and
writes a provenance-linked sidecar using explicit numerical and practical
effect thresholds.

No checkpoint, count matrix, held-out target, or external study is opened.
The resulting decision is a development decision, not statistical inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = "wld_v56_practical_effect_audit_v1"
NUMERICAL_ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_EFFECT_FLOOR = 1e-8

# These are prospective development-viability gates, not universal biological
# constants and not confirmatory significance thresholds.
DEFAULT_THRESHOLDS = {
    "minimum_persistence_mean_relative_gain": 0.02,
    "minimum_perturbed_mean_relative_gain": 0.01,
    "minimum_matched_control_mean_relative_gain": 0.01,
    "minimum_frozen_all_mean_relative_gain": 0.01,
    "minimum_persistence_positive_target_fraction": 0.75,
    "minimum_perturbed_mean_positive_target_fraction": 0.75,
    "minimum_matched_control_positive_target_fraction": 0.75,
    "minimum_frozen_all_positive_target_fraction": 0.60,
    "minimum_detectable_target_fraction": 0.50,
    "minimum_detectable_target_seed_fraction": 0.50,
    "minimum_route_supported_targets": 8,
    "maximum_mean_response_nrmse": 0.90,
    "maximum_median_response_nrmse": 0.95,
    "minimum_mean_response_cosine": 0.20,
    "minimum_median_response_cosine": 0.10,
}

PRACTICAL_RELATIVE_THRESHOLDS = {
    "persistence": "minimum_persistence_mean_relative_gain",
    "training_perturbed_mean": "minimum_perturbed_mean_relative_gain",
    "matched_control_mean": "minimum_matched_control_mean_relative_gain",
    "frozen_all_routes": "minimum_frozen_all_mean_relative_gain",
}

EFFECT_COLUMNS = {
    "persistence": ("persistence_minus_true", "persistence_swd"),
    "training_perturbed_mean": (
        "perturbed_mean_minus_true",
        "perturbed_mean_swd",
    ),
    "matched_control_mean": ("control_mean_minus_true", "control_mean_swd"),
    "frozen_tf": ("frozen_tf_minus_true", None),
    "frozen_complex": ("frozen_complex_minus_true", None),
    "frozen_all_routes": ("frozen_all_minus_true", None),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {number!r}")
    return number


def _summary(values: Sequence[float]) -> Dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
        "rows": len(values),
    }


def _validate_claims(report: Mapping[str, object]) -> Dict[str, object]:
    claims = report.get("claims")
    if not isinstance(claims, Mapping):
        raise ValueError("Source report has no claims mapping")
    required_false = (
        "untouched_audit_inference",
        "confidence_interval_claim",
        "p_value_claim",
        "test_targets_materialized",
        "test_targets_evaluated",
        "external_subject_study_evaluated",
        "ode_time_scale_identified",
        "fixed_point_claim",
        "basin_claim",
        "digital_twin_claim",
        "attractor_claim",
    )
    unsafe = [name for name in required_false if claims.get(name) is not False]
    if unsafe:
        raise ValueError(
            "Source report does not preserve the sealed-development contract: "
            + ", ".join(unsafe)
        )
    if claims.get("development_only") is not True:
        raise ValueError("Source report is not explicitly development-only")
    if claims.get("validation_targets_previously_used_in_v55") is not True:
        raise ValueError("Source report does not declare reused validation targets")
    if claims.get("all_existing_validation_targets_evaluated") is not True:
        raise ValueError("Source report does not declare the complete reused target set")
    if claims.get("perturbed_mean_baseline_training_only") is not True:
        raise ValueError("Perturbed-mean baseline is not declared training-only")
    baseline = report.get("training_only_perturbed_mean_baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("Source report has no training-only perturbed-mean baseline")
    if baseline.get("validation_values_used_for_construction") is not False:
        raise ValueError("Perturbed-mean baseline used validation values")
    if baseline.get("test_values_materialized") is not False:
        raise ValueError("Perturbed-mean baseline materialized sealed test values")
    return {
        "development_only": True,
        "validation_targets_reused": True,
        "inference": False,
        "confidence_interval_claim": False,
        "p_value_claim": False,
        "test_targets_materialized": False,
        "test_targets_evaluated": False,
        "external_subject_study_evaluated": False,
        "fixed_point_claim": False,
        "basin_claim": False,
        "digital_twin_claim": False,
        "attractor_claim": False,
    }


def _validate_grid(
    report: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> tuple[List[str], List[int]]:
    if not rows:
        raise ValueError("Source report has no paired reused-development rows")
    observed_pairs = set()
    targets = set()
    seeds = set()
    for index, row in enumerate(rows):
        target = str(row.get("target", "")).strip().upper()
        if not target:
            raise ValueError(f"Row {index} has an empty target")
        if "seed" not in row or type(row["seed"]) is not int:
            raise ValueError(f"Row {index} has an invalid seed")
        seed = row["seed"]
        pair = (target, seed)
        if pair in observed_pairs:
            raise ValueError(f"Duplicate target/seed row: {target}/{seed}")
        observed_pairs.add(pair)
        targets.add(target)
        seeds.add(seed)

    expected_pairs = {(target, seed) for target in targets for seed in seeds}
    if observed_pairs != expected_pairs:
        raise ValueError("Paired reused-development target/seed grid is ragged")

    raw_declared_targets = report.get("reused_development_targets")
    if not isinstance(raw_declared_targets, (list, tuple)) or not raw_declared_targets:
        raise ValueError("Source report has no declared reused-development targets")
    declared_targets = {
        str(value).strip().upper() for value in raw_declared_targets
    }
    if "" in declared_targets or len(declared_targets) != len(raw_declared_targets):
        raise ValueError("Declared reused-development targets are empty or duplicated")
    if declared_targets != targets:
        raise ValueError("Declared and observed reused-development targets disagree")
    config = report.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Source report has no config mapping")
    raw_declared_seeds = config.get("seeds")
    if not isinstance(raw_declared_seeds, (list, tuple)) or not raw_declared_seeds:
        raise ValueError("Source report has no declared development seeds")
    if any(type(value) is not int for value in raw_declared_seeds):
        raise ValueError("Declared development seeds are not integers")
    declared_seeds = set(raw_declared_seeds)
    if len(declared_seeds) != len(raw_declared_seeds) or declared_seeds != seeds:
        raise ValueError("Declared and observed seed rosters disagree")
    return sorted(targets), sorted(seeds)


def _relative_gain(effect: float, comparator: float) -> float:
    return effect / max(abs(comparator), RELATIVE_EFFECT_FLOOR)


def _condition_count(report: Mapping[str, object]) -> int:
    conditions = report.get("conditions", {})
    if not isinstance(conditions, Mapping):
        return 0
    return sum(str(name).startswith("matched_control_") for name in conditions)


def _detectability(
    report: Mapping[str, object],
    targets: Sequence[str],
    seeds: Sequence[int],
) -> tuple[Dict[str, object], set[tuple[str, int]]]:
    frozen = report.get("frozen_true_model_evaluations")
    if not isinstance(frozen, Mapping):
        return {
            "available": False,
            "reason": "frozen_true_model_evaluations is absent",
            "detectable_target_fraction": 0.0,
        }, set()

    expected = {(target, seed) for target in targets for seed in seeds}
    seen: set[tuple[str, int]] = set()
    detectable: set[tuple[str, int]] = set()
    response_rms: List[float] = []
    cells: List[float] = []
    persistence_swd: List[float] = []
    floor_rows = 0
    for seed in seeds:
        seed_report = frozen.get(str(seed), frozen.get(seed))
        if not isinstance(seed_report, Mapping):
            raise ValueError(f"Missing frozen evaluation for seed {seed}")
        all_removed = seed_report.get("all_routes_removed")
        if not isinstance(all_removed, Mapping):
            raise ValueError(f"Missing all-routes-removed evaluation for seed {seed}")
        rows = all_removed.get("per_target")
        if not isinstance(rows, list):
            raise ValueError(f"Invalid frozen target table for seed {seed}")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"Invalid frozen row {index} for seed {seed}")
            target = str(row.get("target", "")).strip().upper()
            pair = (target, seed)
            if pair not in expected or pair in seen:
                raise ValueError(f"Unexpected or duplicate frozen row {target}/{seed}")
            seen.add(pair)
            floor_used = _finite(
                row.get("response_nrmse_floor_used"),
                f"frozen row {target}/{seed} response_nrmse_floor_used",
            )
            if floor_used < 0.0 or floor_used > 1.0:
                raise ValueError("response_nrmse_floor_used is outside [0, 1]")
            rms = _finite(
                row.get("observed_response_rms"),
                f"frozen row {target}/{seed} observed_response_rms",
            )
            cell_count = _finite(row.get("cells"), f"frozen row {target}/{seed} cells")
            persistence_value = _finite(
                row.get("persistence_swd"),
                f"frozen row {target}/{seed} persistence_swd",
            )
            if rms < 0.0 or cell_count <= 0.0 or persistence_value < 0.0:
                raise ValueError("Frozen detectability diagnostics have invalid ranges")
            if floor_used == 0.0 and rms < 2e-3:
                raise ValueError("Response floor flag contradicts observed response RMS")
            if floor_used == 1.0 and rms >= 2e-3:
                raise ValueError("Response floor flag contradicts observed response RMS")
            floor_rows += int(floor_used > 0.0)
            if floor_used <= 0.0:
                detectable.add(pair)
            response_rms.append(rms)
            cells.append(cell_count)
            persistence_swd.append(persistence_value)
    if seen != expected:
        raise ValueError("Frozen all-routes-removed target/seed grid is ragged")
    minimum_seed_count = math.ceil(len(seeds) / 2)
    detectable_targets = {
        target
        for target in targets
        if sum((target, seed) in detectable for seed in seeds) >= minimum_seed_count
    }
    return {
        "available": True,
        "response_floor": 2e-3,
        "floor_exposed_target_seed_rows": floor_rows,
        "floor_exposed_target_seed_fraction": floor_rows / len(expected),
        "detectable_target_seed_rows": len(detectable),
        "detectable_target_seed_fraction": len(detectable) / len(expected),
        "detectable_targets": len(detectable_targets),
        "detectable_target_fraction": len(detectable_targets) / len(targets),
        "minimum_above_floor_seeds_per_detectable_target": minimum_seed_count,
        "observed_response_rms": _summary(response_rms),
        "cells": _summary(cells),
        "persistence_swd": _summary(persistence_swd),
        "note": (
            "Detectability is defined only by whether the observed response RMS "
            "exceeded the trainer's predeclared 0.002 NRMSE denominator floor."
        ),
    }, detectable


def audit_payload(
    report: Mapping[str, object],
    *,
    source_report_sha256: str,
    source_report_path: str,
    thresholds: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    """Audit a decoded v5.6 report without reading any other artifact."""

    threshold_values = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        threshold_values.update({key: float(value) for key, value in thresholds.items()})
    for name, value in threshold_values.items():
        _finite(value, f"threshold {name}")

    if report.get("schema_version") != "wld-v5.6-null-aware-reused-development":
        raise ValueError(
            "Unsupported source schema: " + str(report.get("schema_version"))
        )

    claims = _validate_claims(report)
    raw_rows = report.get("paired_reused_development_metrics")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ValueError("Source report has an invalid paired metric table")
    rows: List[Mapping[str, object]] = list(raw_rows)
    targets, seeds = _validate_grid(report, rows)
    if len(seeds) < 3:
        raise ValueError("At least three unique development seeds are required")
    conditions = report.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("Source report has no condition index mapping")
    condition_names = {str(name) for name in conditions}
    expected_condition_names = {
        "true_null_aware_routes",
        *{name for name in condition_names if name.startswith("matched_control_")},
    }
    if condition_names != expected_condition_names:
        raise ValueError("Source report has an unexpected condition name")
    expected_seed_keys = {str(seed) for seed in seeds}
    for name, seed_index in conditions.items():
        if not isinstance(seed_index, Mapping) or set(seed_index) != expected_seed_keys:
            raise ValueError(f"Condition {name} has an invalid seed index")

    numeric_rows: List[MutableMapping[str, object]] = []
    by_target: Dict[str, List[MutableMapping[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        target = str(row["target"]).strip().upper()
        seed = int(row["seed"])
        true_swd = _finite(row.get("true_swd"), f"row {index} true_swd")
        route_supported = row.get("route_supported")
        if type(route_supported) is not bool:
            raise ValueError(f"row {index} route_supported is not Boolean")
        response_nrmse_value = _finite(
            row.get("response_nrmse"), f"row {index} response_nrmse"
        )
        response_cosine_value = _finite(
            row.get("response_cosine"), f"row {index} response_cosine"
        )
        if true_swd < 0.0 or response_nrmse_value < 0.0:
            raise ValueError(f"row {index} has a negative SWD or NRMSE")
        if response_cosine_value < -1.0 or response_cosine_value > 1.0:
            raise ValueError(f"row {index} response cosine is outside [-1, 1]")
        converted: MutableMapping[str, object] = {
            "target": target,
            "seed": seed,
            "route_supported": route_supported,
            "true_swd": true_swd,
            "response_nrmse": response_nrmse_value,
            "response_cosine": response_cosine_value,
        }
        for name, (effect_column, comparator_column) in EFFECT_COLUMNS.items():
            effect = _finite(row.get(effect_column), f"row {index} {effect_column}")
            if comparator_column is None:
                comparator = true_swd + effect
            else:
                comparator = _finite(
                    row.get(comparator_column), f"row {index} {comparator_column}"
                )
                expected_effect = comparator - true_swd
                if not math.isclose(
                    effect, expected_effect, rel_tol=1e-7, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"row {index} {effect_column} does not equal "
                        f"{comparator_column} - true_swd"
                    )
            if comparator < 0.0:
                raise ValueError(f"row {index} {name} comparator SWD is negative")
            converted[f"{name}_absolute_gain"] = effect
            converted[f"{name}_relative_gain"] = _relative_gain(effect, comparator)
        numeric_rows.append(converted)
        by_target[target].append(converted)

    summaries: Dict[str, object] = {}
    target_summaries: List[Dict[str, object]] = []
    for name in EFFECT_COLUMNS:
        absolute = [float(row[f"{name}_absolute_gain"]) for row in numeric_rows]
        relative = [float(row[f"{name}_relative_gain"]) for row in numeric_rows]
        per_target_relative = {
            target: mean(float(row[f"{name}_relative_gain"]) for row in target_rows)
            for target, target_rows in by_target.items()
        }
        summaries[name] = {
            "absolute_gain": _summary(absolute),
            "relative_gain": _summary(relative),
            "target_fraction_above_practical_relative_threshold": mean(
                value
                >= threshold_values[
                    PRACTICAL_RELATIVE_THRESHOLDS.get(
                        name, "minimum_frozen_all_mean_relative_gain"
                    )
                ]
                for value in per_target_relative.values()
            ),
            "target_fraction_above_numerical_absolute_tolerance": mean(
                mean(
                    float(row[f"{name}_absolute_gain"])
                    for row in by_target[target]
                )
                > NUMERICAL_ABSOLUTE_TOLERANCE
                for target in targets
            ),
        }

        per_seed_relative = {
            seed: mean(
                float(row[f"{name}_relative_gain"])
                for row in numeric_rows
                if int(row["seed"]) == seed
            )
            for seed in seeds
        }
        summaries[name]["seed_mean_relative_gains"] = {
            str(seed): value for seed, value in per_seed_relative.items()
        }
        per_seed_absolute = {
            seed: mean(
                float(row[f"{name}_absolute_gain"])
                for row in numeric_rows
                if int(row["seed"]) == seed
            )
            for seed in seeds
        }
        summaries[name]["seed_mean_absolute_gains"] = {
            str(seed): value for seed, value in per_seed_absolute.items()
        }
        summaries[name]["all_seed_means_numerically_positive"] = all(
            value > NUMERICAL_ABSOLUTE_TOLERANCE
            for value in per_seed_absolute.values()
        )

    for target in targets:
        target_rows = by_target[target]
        target_summaries.append(
            {
                "target": target,
                "seeds": len(target_rows),
                "route_supported": all(
                    bool(row["route_supported"]) for row in target_rows
                ),
                "mean_persistence_relative_gain": mean(
                    float(row["persistence_relative_gain"]) for row in target_rows
                ),
                "mean_matched_control_relative_gain": mean(
                    float(row["matched_control_mean_relative_gain"])
                    for row in target_rows
                ),
                "mean_frozen_all_routes_relative_gain": mean(
                    float(row["frozen_all_routes_relative_gain"])
                    for row in target_rows
                ),
                "mean_response_nrmse": mean(
                    float(row["response_nrmse"]) for row in target_rows
                ),
                "mean_response_cosine": mean(
                    float(row["response_cosine"]) for row in target_rows
                ),
            }
        )

    detectability, detectable_pairs = _detectability(report, targets, seeds)
    evaluable_rows = [
        row
        for row in numeric_rows
        if (str(row["target"]), int(row["seed"])) in detectable_pairs
    ]
    response_nrmse = [float(row["response_nrmse"]) for row in evaluable_rows]
    response_cosine = [float(row["response_cosine"]) for row in evaluable_rows]
    if not response_nrmse:
        # Keep the report serializable and fail all response gates closed.
        response_nrmse = [float(row["response_nrmse"]) for row in numeric_rows]
        response_cosine = [float(row["response_cosine"]) for row in numeric_rows]
    persistence = summaries["persistence"]
    controls = summaries["matched_control_mean"]
    frozen_all = summaries["frozen_all_routes"]
    perturbed_mean = summaries["training_perturbed_mean"]
    assert isinstance(persistence, Mapping)
    assert isinstance(controls, Mapping)
    assert isinstance(frozen_all, Mapping)
    assert isinstance(perturbed_mean, Mapping)

    numerical_checks = {
        "true_beats_persistence_beyond_absolute_tolerance": (
            persistence["absolute_gain"]["mean"]
            > NUMERICAL_ABSOLUTE_TOLERANCE
        ),
        "true_beats_training_perturbed_mean_beyond_absolute_tolerance": (
            perturbed_mean["absolute_gain"]["mean"]
            > NUMERICAL_ABSOLUTE_TOLERANCE
        ),
        "true_beats_matched_controls_beyond_absolute_tolerance": (
            controls["absolute_gain"]["mean"] > NUMERICAL_ABSOLUTE_TOLERANCE
        ),
        "fitted_routes_used_beyond_absolute_tolerance": (
            frozen_all["absolute_gain"]["mean"]
            > NUMERICAL_ABSOLUTE_TOLERANCE
        ),
        "response_nrmse_below_persistence_beyond_absolute_tolerance": (
            mean(response_nrmse) < 1.0 - NUMERICAL_ABSOLUTE_TOLERANCE
        ),
        "response_cosine_beyond_absolute_tolerance": (
            mean(response_cosine) > NUMERICAL_ABSOLUTE_TOLERANCE
        ),
    }
    numerical_checks["all_pass"] = all(numerical_checks.values())

    matched_controls = _condition_count(report)
    expected_control_names = {
        str(name) for name in conditions if str(name).startswith("matched_control_")
    }
    config = report.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Source report has no config mapping")
    minimum_controls_value = config.get("minimum_control_replicates")
    configured_controls_value = config.get("control_replicates")
    if type(minimum_controls_value) is not int or type(configured_controls_value) is not int:
        raise ValueError("Matched-control counts are not integers")
    minimum_controls = minimum_controls_value
    configured_controls = configured_controls_value
    if minimum_controls < 10 or configured_controls < minimum_controls:
        raise ValueError("Source report has an invalid matched-control configuration")
    if matched_controls != configured_controls:
        raise ValueError("Condition roster does not match configured control replicates")
    raw_control_effects = report.get("per_control_descriptive_effects")
    control_effect_values: List[float] = []
    if isinstance(raw_control_effects, list):
        control_names = set()
        for index, row in enumerate(raw_control_effects):
            if not isinstance(row, Mapping):
                raise ValueError(f"Invalid per-control row {index}")
            name = str(row.get("condition", ""))
            if not name.startswith("matched_control_") or name in control_names:
                raise ValueError(f"Invalid or duplicate per-control condition {name!r}")
            control_names.add(name)
            control_effect_values.append(
                _finite(
                    row.get("mean_control_minus_true"),
                    f"per-control row {index} mean_control_minus_true",
                )
            )
    if len(control_effect_values) != matched_controls:
        raise ValueError("Per-control effects and matched-control condition roster disagree")
    if control_names != expected_control_names:
        raise ValueError("Per-control names and matched-control condition names disagree")
    expected_target_seed_rows = len(targets) * len(seeds)
    for index, row in enumerate(raw_control_effects):
        row_count = row.get("target_seed_rows")
        if type(row_count) is not int or row_count != expected_target_seed_rows:
            raise ValueError(f"Per-control row {index} has a wrong target/seed row count")
    paired_control_mean = float(controls["absolute_gain"]["mean"])
    if not math.isclose(
        mean(control_effect_values), paired_control_mean, rel_tol=1e-7, abs_tol=1e-9
    ):
        raise ValueError("Per-control effects do not reproduce the paired control mean")
    control_median = median(control_effect_values)
    control_mad = median(
        [abs(value - control_median) for value in control_effect_values]
    )
    control_null_summary = {
        **_summary(control_effect_values),
        "median_absolute_deviation": control_mad,
        "fraction_beyond_numerical_tolerance": mean(
            value > NUMERICAL_ABSOLUTE_TOLERANCE for value in control_effect_values
        ),
        "true_beats_every_control_beyond_numerical_tolerance": all(
            value > NUMERICAL_ABSOLUTE_TOLERANCE for value in control_effect_values
        ),
        "mean_effect_over_mad": (
            mean(control_effect_values) / control_mad if control_mad > 0.0 else None
        ),
        "inference": False,
    }

    practical_checks = {
        "mean_persistence_relative_gain_at_least_threshold": (
            persistence["relative_gain"]["mean"]
            >= threshold_values["minimum_persistence_mean_relative_gain"]
        ),
        "persistence_target_fraction_at_least_threshold": (
            persistence["target_fraction_above_practical_relative_threshold"]
            >= threshold_values["minimum_persistence_positive_target_fraction"]
        ),
        "persistence_gain_positive_in_every_seed": bool(
            persistence["all_seed_means_numerically_positive"]
        ),
        "mean_training_perturbed_mean_relative_gain_at_least_threshold": (
            perturbed_mean["relative_gain"]["mean"]
            >= threshold_values["minimum_perturbed_mean_relative_gain"]
        ),
        "perturbed_mean_target_fraction_at_least_threshold": (
            perturbed_mean["target_fraction_above_practical_relative_threshold"]
            >= threshold_values["minimum_perturbed_mean_positive_target_fraction"]
        ),
        "mean_matched_control_relative_gain_at_least_threshold": (
            controls["relative_gain"]["mean"]
            >= threshold_values["minimum_matched_control_mean_relative_gain"]
        ),
        "matched_control_target_fraction_at_least_threshold": (
            controls["target_fraction_above_practical_relative_threshold"]
            >= threshold_values["minimum_matched_control_positive_target_fraction"]
        ),
        "at_least_ten_matched_controls": matched_controls >= 10,
        "true_beats_every_matched_control_beyond_numerical_tolerance": bool(
            control_null_summary[
                "true_beats_every_control_beyond_numerical_tolerance"
            ]
        ),
        "mean_frozen_all_routes_relative_gain_at_least_threshold": (
            frozen_all["relative_gain"]["mean"]
            >= threshold_values["minimum_frozen_all_mean_relative_gain"]
        ),
        "frozen_all_target_fraction_at_least_threshold": (
            frozen_all["target_fraction_above_practical_relative_threshold"]
            >= threshold_values["minimum_frozen_all_positive_target_fraction"]
        ),
        "detectable_target_fraction_at_least_threshold": (
            bool(detectability.get("available"))
            and float(detectability["detectable_target_fraction"])
            >= threshold_values["minimum_detectable_target_fraction"]
        ),
        "detectable_target_seed_fraction_at_least_threshold": (
            bool(detectability.get("available"))
            and float(detectability["detectable_target_seed_fraction"])
            >= threshold_values["minimum_detectable_target_seed_fraction"]
        ),
        "route_supported_target_count_at_least_threshold": (
            sum(
                all(bool(row["route_supported"]) for row in by_target[target])
                for target in targets
            )
            >= threshold_values["minimum_route_supported_targets"]
        ),
        "mean_response_nrmse_at_most_threshold": (
            mean(response_nrmse)
            <= threshold_values["maximum_mean_response_nrmse"]
        ),
        "median_response_nrmse_at_most_threshold": (
            median(response_nrmse)
            <= threshold_values["maximum_median_response_nrmse"]
        ),
        "mean_response_cosine_at_least_threshold": (
            mean(response_cosine)
            >= threshold_values["minimum_mean_response_cosine"]
        ),
        "median_response_cosine_at_least_threshold": (
            median(response_cosine)
            >= threshold_values["minimum_median_response_cosine"]
        ),
    }
    practical_checks["all_pass"] = all(practical_checks.values())

    old_checks = report.get("development_checks", {})
    legacy_eligible = bool(
        isinstance(old_checks, Mapping)
        and old_checks.get("eligible_to_freeze_new_confirmation_plan") is True
    )
    corrected_eligible = bool(numerical_checks["all_pass"] and practical_checks["all_pass"])

    gate_summary: Dict[str, object] = {}
    gates = report.get("fitted_true_branch_gates_by_seed", {})
    if isinstance(gates, Mapping):
        names = sorted(
            {
                str(name)
                for seed_gates in gates.values()
                if isinstance(seed_gates, Mapping)
                for name in seed_gates
            }
        )
        for name in names:
            values = [
                _finite(seed_gates[name], f"branch gate {name}")
                for seed_gates in gates.values()
                if isinstance(seed_gates, Mapping) and name in seed_gates
            ]
            if values:
                gate_summary[name] = _summary(values)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Post-hoc, no-retraining practical-effect audit of reused v5.6 "
            "development targets"
        ),
        "source": {
            "report_path": source_report_path,
            "report_sha256": source_report_sha256,
            "source_schema_version": report.get("schema_version"),
            "source_legacy_eligibility": legacy_eligible,
            "source_report_immutable": True,
        },
        "data_contract": {
            **claims,
            "checkpoints_loaded": False,
            "raw_cells_loaded": False,
            "training_performed": False,
            "target_count": len(targets),
            "seed_count": len(seeds),
            "target_seed_rows": len(numeric_rows),
            "matched_control_count": matched_controls,
            "complete_target_seed_grid": True,
        },
        "decision_rule": {
            "numerical_absolute_tolerance": NUMERICAL_ABSOLUTE_TOLERANCE,
            "relative_effect_denominator_floor": RELATIVE_EFFECT_FLOOR,
            "practical_thresholds": threshold_values,
            "threshold_status": (
                "prospective development-viability gates; not inferential or "
                "universal biological constants"
            ),
        },
        "effect_summaries": summaries,
        "matched_control_null_summary": control_null_summary,
        "response": {
            "nrmse": _summary(response_nrmse),
            "nrmse_improvement_over_persistence": 1.0 - mean(response_nrmse),
            "cosine": _summary(response_cosine),
            "rows_used_for_response_summaries": len(response_nrmse),
            "above_floor_rows_available": len(evaluable_rows),
            "only_above_floor_rows_used_for_response_gates": bool(evaluable_rows),
            "fallback_is_fail_closed_by_detectability_gate": not bool(evaluable_rows),
        },
        "detectability": detectability,
        "route_support_target_fraction": mean(
            all(bool(row["route_supported"]) for row in by_target[target])
            for target in targets
        ),
        "fitted_branch_gate_summary": gate_summary,
        "numerical_checks": numerical_checks,
        "practical_checks": practical_checks,
        "decision": {
            "legacy_eligibility_flag": legacy_eligible,
            "legacy_flag_superseded": legacy_eligible != corrected_eligible,
            "eligible_to_freeze_new_confirmation_plan": corrected_eligible,
            "open_sealed_test": False,
            "reason": (
                "All numerical and practical development gates passed."
                if corrected_eligible
                else (
                    "The completed run does not clear explicit numerical and "
                    "practical-effect gates; do not treat floating-point dust as "
                    "predictive or mechanistic evidence."
                )
            ),
        },
        "target_summaries": target_summaries,
        "claims": {
            "inference": False,
            "confidence_interval_claim": False,
            "p_value_claim": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
            "confirmation_claim": False,
        },
        "next_action": (
            "Keep all sealed targets closed. Quantify perturbation-response "
            "detectability and route coverage, then redesign development only if "
            "the observed responses are learnable above persistence."
        ),
    }


def write_target_table(path: Path, audit: Mapping[str, object]) -> None:
    rows = audit["target_summaries"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Audit has no target summaries")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_audit(report_path: Path, output_root: Path) -> Dict[str, object]:
    report_path = Path(report_path)
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing completed v5.6 report: {report_path}")
    source_hash = sha256_file(report_path)
    report = json.loads(report_path.read_text())
    if not isinstance(report, Mapping):
        raise ValueError("Completed v5.6 report is not a JSON object")
    candidate = audit_payload(
        report,
        source_report_sha256=source_hash,
        source_report_path=str(report_path),
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    audit_path = output_root / "wld_v56_practical_effect_audit.json"
    table_path = output_root / "wld_v56_target_effects.tsv"
    if audit_path.is_file() and audit_path.stat().st_size:
        existing = json.loads(audit_path.read_text())
        if not isinstance(existing, Mapping):
            raise ValueError("Existing practical-effect audit is not a JSON object")
        existing_comparable = dict(existing)
        candidate_comparable = dict(candidate)
        existing_comparable.pop("created_utc", None)
        candidate_comparable.pop("created_utc", None)
        if existing_comparable != candidate_comparable:
            raise RuntimeError(
                "Existing audit differs from a fresh validation of the same input; "
                "preserve it and use a new output directory for this implementation."
            )
        # The table is a small derived view. Rebuild it on every successful
        # resume so a stale or truncated TSV cannot survive verification.
        write_target_table(table_path, existing)
        return dict(existing)
    atomic_json(audit_path, candidate)
    write_target_table(table_path, candidate)
    return candidate


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    audit = run_audit(args.report, args.output_root)
    decision = audit["decision"]
    response = audit["response"]
    effects = audit["effect_summaries"]
    print("=" * 78)
    print("WLD V5.6 PRACTICAL-EFFECT AUDIT")
    print("=" * 78)
    print(f"Legacy eligibility flag:       {decision['legacy_eligibility_flag']}")
    print(
        "Corrected practical eligibility: "
        f"{decision['eligible_to_freeze_new_confirmation_plan']}"
    )
    print(
        "Persistence relative gain:      "
        f"{effects['persistence']['relative_gain']['mean']:+.8%}"
    )
    print(
        "Matched-control relative gain:  "
        f"{effects['matched_control_mean']['relative_gain']['mean']:+.8%}"
    )
    print(
        "Frozen-route relative gain:     "
        f"{effects['frozen_all_routes']['relative_gain']['mean']:+.8%}"
    )
    print(f"Mean response NRMSE:            {response['nrmse']['mean']:.9f}")
    print(f"Mean response cosine:           {response['cosine']['mean']:+.9f}")
    print(f"Open sealed test:               {decision['open_sealed_test']}")
    print(f"Reason: {decision['reason']}")
    print(
        "Report: "
        + str(Path(args.output_root) / "wld_v56_practical_effect_audit.json")
    )


if __name__ == "__main__":
    main()
