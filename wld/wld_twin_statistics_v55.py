"""Target-level uncertainty and claim gates for WLD v5.5.

Cells from one perturbation target are not independent biological replicates.
These utilities average algorithmic seeds within target and bootstrap targets,
then calibrate ensemble uncertainty with a disjoint target block.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    effect: float
    lower: float
    upper: float
    targets: int
    seeds: int
    bootstrap_samples: int
    positive_seed_fraction: float


@dataclass(frozen=True)
class MeanBootstrapInterval:
    mean: float
    lower: float
    upper: float
    targets: int
    seeds: int
    bootstrap_samples: int


def paired_target_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    true_key: str,
    comparator_key: str,
    target_key: str = "target",
    seed_key: str = "seed",
    samples: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 41,
) -> BootstrapInterval:
    """Estimate comparator-minus-model effect using targets as sampling units."""

    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5,1)")
    seen = set()
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    seed_effects: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        target = str(row[target_key])
        seed = int(row[seed_key])
        key = (target, seed)
        if key in seen:
            raise ValueError(f"duplicate target/seed result: {key}")
        seen.add(key)
        true_value = float(row[true_key])
        comparator = float(row[comparator_key])
        if not math.isfinite(true_value) or not math.isfinite(comparator):
            raise ValueError("bootstrap inputs must be finite")
        effect = comparator - true_value
        grouped[target].append((seed, effect))
        seed_effects[seed].append(effect)
    if len(grouped) < 2:
        raise ValueError("at least two held-out targets are required")
    seed_grids = [{seed for seed, _effect in values} for values in grouped.values()]
    if any(grid != seed_grids[0] for grid in seed_grids[1:]):
        raise ValueError("target bootstrap requires a complete target/seed grid")

    targets = sorted(grouped)
    per_target = np.asarray(
        [np.mean([effect for _seed, effect in grouped[target]]) for target in targets],
        dtype=np.float64,
    )
    rng = np.random.default_rng(random_seed)
    draws = rng.choice(len(targets), size=(int(samples), len(targets)), replace=True)
    distribution = per_target[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    seeds = sorted(seed_effects)
    positive = np.mean(
        [float(np.mean(seed_effects[seed])) > 0.0 for seed in seeds]
    )
    return BootstrapInterval(
        effect=float(per_target.mean()),
        lower=float(np.quantile(distribution, alpha)),
        upper=float(np.quantile(distribution, 1.0 - alpha)),
        targets=len(targets),
        seeds=len(seeds),
        bootstrap_samples=int(samples),
        positive_seed_fraction=float(positive),
    )


def target_bootstrap_mean(
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    target_key: str = "target",
    seed_key: str = "seed",
    samples: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 43,
) -> MeanBootstrapInterval:
    """Bootstrap a metric mean with perturbation targets as sampling units."""

    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5,1)")
    seen = set()
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        target, seed = str(row[target_key]), int(row[seed_key])
        key = (target, seed)
        if key in seen:
            raise ValueError(f"duplicate target/seed result: {key}")
        seen.add(key)
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("bootstrap metric values must be finite")
        grouped[target].append((seed, value))
    if len(grouped) < 2:
        raise ValueError("at least two held-out targets are required")
    seed_grids = [{seed for seed, _value in values} for values in grouped.values()]
    if any(grid != seed_grids[0] for grid in seed_grids[1:]):
        raise ValueError("target bootstrap requires a complete target/seed grid")
    targets = sorted(grouped)
    per_target = np.asarray(
        [np.mean([value for _seed, value in grouped[target]]) for target in targets],
        dtype=np.float64,
    )
    rng = np.random.default_rng(random_seed)
    draws = rng.choice(len(targets), size=(int(samples), len(targets)), replace=True)
    distribution = per_target[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return MeanBootstrapInterval(
        mean=float(per_target.mean()),
        lower=float(np.quantile(distribution, alpha)),
        upper=float(np.quantile(distribution, 1.0 - alpha)),
        targets=len(targets),
        seeds=len({seed for values in grouped.values() for seed, _value in values}),
        bootstrap_samples=int(samples),
    )


def conformal_quantile(scores: Iterable[float], *, alpha: float) -> float:
    """Finite-sample split-conformal quantile with the conservative rank."""

    values = np.sort(np.asarray(list(scores), dtype=np.float64))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("conformal scores must be a nonempty finite vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    if rank > len(values):
        raise ValueError(
            "Calibration block is too small for a finite split-conformal "
            f"quantile at alpha={alpha}; need rank {rank} from {len(values)} targets"
        )
    rank = max(rank, 1)
    return float(values[rank - 1])


def calibrate_ensemble_intervals(
    calibration_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    alpha: float = 0.2,
    scale_floor: float = 1e-4,
) -> List[Dict[str, float]]:
    """Target-block normalized split-conformal calibration.

    Each row contains ``observed``, ``ensemble_mean`` and ``ensemble_std``.
    Calibration and prediction target names must be disjoint.
    """

    calibration_targets = {str(row["target"]) for row in calibration_rows}
    prediction_targets = {str(row["target"]) for row in prediction_rows}
    overlap = calibration_targets & prediction_targets
    if overlap:
        raise ValueError(f"calibration and prediction targets overlap: {sorted(overlap)}")
    scores = []
    for row in calibration_rows:
        observed = float(row["observed"])
        mean = float(row["ensemble_mean"])
        scale = max(float(row["ensemble_std"]), float(scale_floor))
        if not all(math.isfinite(value) for value in (observed, mean, scale)):
            raise ValueError("uncertainty rows must be finite")
        scores.append(abs(observed - mean) / scale)
    quantile = conformal_quantile(scores, alpha=alpha)
    calibrated = []
    for row in prediction_rows:
        mean = float(row["ensemble_mean"])
        raw_std = float(row["ensemble_std"])
        scale = max(raw_std, float(scale_floor))
        width = quantile * scale
        calibrated.append(
            {
                "target": str(row["target"]),
                "predictive_mean": mean,
                "ensemble_std": raw_std,
                "conformal_quantile": quantile,
                "lower": mean - width,
                "upper": mean + width,
                "interval_width": 2.0 * width,
                "std_floor_used": float(raw_std < scale_floor),
            }
        )
    return calibrated


def evaluate_claims(
    *,
    persistence: BootstrapInterval,
    topology_shuffle: BootstrapInterval,
    frozen_removal: BootstrapInterval,
    frozen_tf_removal: Optional[BootstrapInterval],
    frozen_complex_removal: Optional[BootstrapInterval],
    response_nrmse: MeanBootstrapInterval,
    response_cosine: MeanBootstrapInterval,
    calibrated_coverage: float,
    normalized_interval_width: float,
    external_subject_study_test: bool,
    prospective_update_loop: bool,
    longitudinal_return_data: bool,
    nrmse_maximum: float = 0.75,
    cosine_minimum: float = 0.20,
    coverage_minimum: float = 0.75,
    width_maximum: float = 2.0,
) -> Dict[str, object]:
    """A conservative, explicit truth table for WLD scientific language."""

    tf_path_reliance = (
        frozen_tf_removal.lower > 0.0
        if frozen_tf_removal is not None
        else False
    )
    complex_path_reliance = (
        frozen_complex_removal.lower > 0.0
        if frozen_complex_removal is not None
        else False
    )
    fitted_path_reliance = tf_path_reliance and complex_path_reliance
    topology_specificity = topology_shuffle.lower > 0.0
    useful_prediction = (
        persistence.lower > 0.0
        and math.isfinite(response_nrmse.upper)
        and response_nrmse.upper <= nrmse_maximum
        and math.isfinite(response_cosine.lower)
        and response_cosine.lower >= cosine_minimum
    )
    calibrated_uncertainty = (
        calibrated_coverage >= coverage_minimum
        and normalized_interval_width <= width_maximum
    )
    transient_mechanistic_response = (
        fitted_path_reliance and topology_specificity and useful_prediction
    )
    digital_twin_readiness = (
        transient_mechanistic_response
        and calibrated_uncertainty
        and bool(external_subject_study_test)
        and bool(prospective_update_loop)
        and bool(longitudinal_return_data)
    )
    failures = []
    checks = {
        "frozen TF-route reliance CI above zero": tf_path_reliance,
        "frozen complex-route reliance CI above zero": complex_path_reliance,
        "topology specificity CI above zero": topology_specificity,
        "useful prediction versus persistence": useful_prediction,
        "calibrated and sufficiently sharp uncertainty": calibrated_uncertainty,
        "external subject/study test": bool(external_subject_study_test),
        "prospective physical-counterpart update loop": bool(prospective_update_loop),
        "longitudinal perturb-and-return data": bool(longitudinal_return_data),
    }
    failures.extend(label for label, passed in checks.items() if not passed)
    return {
        "fitted_path_reliance": fitted_path_reliance,
        "tf_path_reliance": tf_path_reliance,
        "complex_path_reliance": complex_path_reliance,
        "topology_specificity": topology_specificity,
        "useful_perturbation_prediction": useful_prediction,
        "response_nrmse_upper_ci": response_nrmse.upper,
        "response_cosine_lower_ci": response_cosine.lower,
        "calibrated_predictive_uncertainty": calibrated_uncertainty,
        "transient_mechanistic_response": transient_mechanistic_response,
        "digital_twin_readiness": digital_twin_readiness,
        "digital_twin_claim": False,
        "ode_kinetics_identified": False,
        "fixed_point_claim": False,
        "basin_claim": False,
        "attractor_claim": False,
        "failure_reasons": failures,
        "claim_note": (
            "Digital-twin and attractor claims remain false until repeated individual "
            "calibration/feedback and longitudinal perturb-and-return tests exist."
        ),
    }
