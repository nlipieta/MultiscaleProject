"""Synthetic contract tests for the WLD v5.7 response-learnability ladder.

These fixtures are deliberately target-level.  They test whether the diagnostic
ladder can distinguish measurement noise, a linearly learnable route map, a
static nonlinear map, stable route-independent responses, and historical WLD
predictions that remained at persistence.  They do not train a WLD model and
never construct a sealed test outcome.
"""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import sparse
from wld_chromatin_modules_v55 import _selected_csr

from wld_response_learnability_v57 import (
    ResponseLearnabilityConfig,
    _ordered_half,
    _rank_cv,
    _split_half_data,
    _validate_production_config,
    deterministic_profile_shuffle,
    deterministic_target_folds,
    evaluate_learnability,
    validate_claims,
)


HISTORICAL_NEAR_PERSISTENCE = {
    "persistence_minus_true": 0.0,
    "training_perturbed_mean_minus_true": 0.00023467,
    "matched_control_mean_minus_true": 0.0,
    "frozen_all_routes_minus_true": 0.0,
    "mean_response_nrmse": 0.9999999615053335,
    "mean_response_cosine": 0.011501414725595774,
    "corrected_practical_gate_passed": False,
    "corrected_eligibility": False,
    "eligible_to_freeze_new_confirmation_plan": False,
    "open_sealed_test": False,
}


@dataclass(frozen=True)
class Fixture:
    train_responses: np.ndarray
    validation_discovery: np.ndarray
    validation_outcomes: np.ndarray
    train_route_features: np.ndarray
    validation_route_features: np.ndarray
    train_targets: tuple[str, ...]
    validation_targets: tuple[str, ...]
    train_screens: tuple[str, ...]
    validation_screens: tuple[str, ...]
    route_supported: np.ndarray


def _standardize_response(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    scale = max(float(np.sqrt(np.mean(value * value))), 1e-12)
    return value / scale


def _names(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index:03d}" for index in range(count))


def _screens(count: int) -> tuple[str, ...]:
    return tuple("screen_1" if index % 2 == 0 else "screen_2" for index in range(count))


def _linear_fixture(seed: int = 7101) -> Fixture:
    rng = np.random.default_rng(seed)
    train_count, validation_count, features, bins = 48, 18, 6, 36
    train_x = rng.normal(size=(train_count, features))
    validation_x = rng.normal(size=(validation_count, features))
    weights = rng.normal(size=(features, bins))
    train_y = _standardize_response(train_x @ weights)
    truth = _standardize_response(validation_x @ weights)
    return Fixture(
        train_responses=train_y + rng.normal(0.0, 0.008, train_y.shape),
        validation_discovery=truth + rng.normal(0.0, 0.008, truth.shape),
        validation_outcomes=truth + rng.normal(0.0, 0.008, truth.shape),
        train_route_features=train_x,
        validation_route_features=validation_x,
        train_targets=_names("TRAIN_LINEAR", train_count),
        validation_targets=_names("DEV_LINEAR", validation_count),
        train_screens=_screens(train_count),
        validation_screens=_screens(validation_count),
        route_supported=np.ones(validation_count, dtype=bool),
    )


def _null_fixture(seed: int = 7102) -> Fixture:
    rng = np.random.default_rng(seed)
    train_count, validation_count, features, bins = 48, 18, 6, 36
    return Fixture(
        train_responses=rng.normal(size=(train_count, bins)),
        validation_discovery=rng.normal(size=(validation_count, bins)),
        validation_outcomes=rng.normal(size=(validation_count, bins)),
        train_route_features=rng.normal(size=(train_count, features)),
        validation_route_features=rng.normal(size=(validation_count, features)),
        train_targets=_names("TRAIN_NULL", train_count),
        validation_targets=_names("DEV_NULL", validation_count),
        train_screens=_screens(train_count),
        validation_screens=_screens(validation_count),
        route_supported=np.ones(validation_count, dtype=bool),
    )


def _stable_high_rank_fixture(seed: int = 7103) -> Fixture:
    rng = np.random.default_rng(seed)
    train_count, validation_count, features, bins = 48, 18, 5, 72
    train_y = rng.normal(size=(train_count, bins))
    truth = rng.normal(size=(validation_count, bins))
    train_y = train_y / np.maximum(np.sqrt(np.mean(train_y * train_y, axis=1, keepdims=True)), 1e-12)
    truth = truth / np.maximum(np.sqrt(np.mean(truth * truth, axis=1, keepdims=True)), 1e-12)
    return Fixture(
        train_responses=train_y,
        validation_discovery=truth + rng.normal(0.0, 0.01, truth.shape),
        validation_outcomes=truth + rng.normal(0.0, 0.01, truth.shape),
        train_route_features=rng.normal(size=(train_count, features)),
        validation_route_features=rng.normal(size=(validation_count, features)),
        train_targets=_names("TRAIN_HIGH_RANK", train_count),
        validation_targets=_names("DEV_HIGH_RANK", validation_count),
        train_screens=_screens(train_count),
        validation_screens=_screens(validation_count),
        route_supported=np.ones(validation_count, dtype=bool),
    )


def _nonlinear_fixture(seed: int = 7104) -> Fixture:
    rng = np.random.default_rng(seed)
    train_count, validation_count, features, bins = 72, 24, 3, 40
    train_x = rng.uniform(-1.7, 1.7, size=(train_count, features))
    validation_x = rng.uniform(-1.7, 1.7, size=(validation_count, features))

    def nonlinear_basis(x: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                np.sin(2.8 * x[:, 0]),
                np.cos(2.5 * x[:, 1]),
                x[:, 0] * x[:, 1],
                np.exp(-1.4 * np.sum((x - np.asarray([0.7, -0.5, 0.3])) ** 2, axis=1)),
                np.exp(-1.8 * np.sum((x - np.asarray([-0.8, 0.6, -0.4])) ** 2, axis=1)),
                np.sin(2.2 * (x[:, 1] + x[:, 2])),
            )
        )

    output_weights = rng.normal(size=(6, bins))
    train_y = _standardize_response(nonlinear_basis(train_x) @ output_weights)
    truth = _standardize_response(nonlinear_basis(validation_x) @ output_weights)
    return Fixture(
        train_responses=train_y + rng.normal(0.0, 0.006, train_y.shape),
        validation_discovery=truth + rng.normal(0.0, 0.006, truth.shape),
        validation_outcomes=truth + rng.normal(0.0, 0.006, truth.shape),
        train_route_features=train_x,
        validation_route_features=validation_x,
        train_targets=_names("TRAIN_NONLINEAR", train_count),
        validation_targets=_names("DEV_NONLINEAR", validation_count),
        train_screens=_screens(train_count),
        validation_screens=_screens(validation_count),
        route_supported=np.ones(validation_count, dtype=bool),
    )


def _route_independent_fixture(seed: int = 7105) -> Fixture:
    rng = np.random.default_rng(seed)
    train_count, validation_count, route_features, latent, bins = 56, 20, 6, 4, 40
    train_latent = rng.normal(size=(train_count, latent))
    validation_latent = rng.normal(size=(validation_count, latent))
    loadings = rng.normal(size=(latent, bins))
    train_y = _standardize_response(train_latent @ loadings)
    truth = _standardize_response(validation_latent @ loadings)
    return Fixture(
        train_responses=train_y + rng.normal(0.0, 0.006, train_y.shape),
        validation_discovery=truth + rng.normal(0.0, 0.006, truth.shape),
        validation_outcomes=truth + rng.normal(0.0, 0.006, truth.shape),
        train_route_features=rng.normal(size=(train_count, route_features)),
        validation_route_features=rng.normal(size=(validation_count, route_features)),
        train_targets=_names("TRAIN_INDEPENDENT", train_count),
        validation_targets=_names("DEV_INDEPENDENT", validation_count),
        train_screens=_screens(train_count),
        validation_screens=_screens(validation_count),
        route_supported=np.ones(validation_count, dtype=bool),
    )


def _screen_confounding_fixture(seed: int = 7106) -> Fixture:
    rng = np.random.default_rng(seed)
    train_count, validation_count, bins = 48, 18, 32
    train_screens = _screens(train_count)
    validation_screens = _screens(validation_count)
    direction = rng.normal(size=bins)
    direction /= np.linalg.norm(direction)
    train_sign = np.asarray([1.0 if screen == "screen_1" else -1.0 for screen in train_screens])
    validation_sign = np.asarray([1.0 if screen == "screen_1" else -1.0 for screen in validation_screens])
    train_y = 2.0 * train_sign[:, None] * direction[None, :]
    truth = 2.0 * validation_sign[:, None] * direction[None, :]
    # Routes expose only the batch/screen nuisance, not target-specific biology.
    train_x = np.column_stack((train_sign, np.zeros((train_count, 3))))
    validation_x = np.column_stack((validation_sign, np.zeros((validation_count, 3))))
    return Fixture(
        train_responses=train_y + rng.normal(0.0, 0.005, train_y.shape),
        validation_discovery=truth + rng.normal(0.0, 0.005, truth.shape),
        validation_outcomes=truth + rng.normal(0.0, 0.005, truth.shape),
        train_route_features=train_x,
        validation_route_features=validation_x,
        train_targets=_names("TRAIN_SCREEN", train_count),
        validation_targets=_names("DEV_SCREEN", validation_count),
        train_screens=train_screens,
        validation_screens=validation_screens,
        route_supported=np.ones(validation_count, dtype=bool),
    )


def _run(fixture: Fixture, *, config: ResponseLearnabilityConfig | None = None) -> Mapping[str, object]:
    return evaluate_learnability(
        fixture.train_responses,
        fixture.validation_discovery,
        fixture.validation_outcomes,
        fixture.train_route_features,
        fixture.validation_route_features,
        train_targets=fixture.train_targets,
        validation_targets=fixture.validation_targets,
        train_screens=fixture.train_screens,
        validation_screens=fixture.validation_screens,
        route_supported=fixture.route_supported,
        historical_wld=HISTORICAL_NEAR_PERSISTENCE,
        config=config,
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{label} is not a mapping")
    return value


def _model(report: Mapping[str, object], name: str) -> Mapping[str, object]:
    return _mapping(_mapping(report.get("models"), "models").get(name), f"model {name}")


def _smoke_config() -> ResponseLearnabilityConfig:
    """Keep the numerical smoke quick while retaining all three fixed seeds."""

    return ResponseLearnabilityConfig(
        reliability_replicates=4,
        null_permutations=4,
        model_seeds=(42, 137, 911),
        rank_grid=(1, 2, 4, 8, 16),
        alpha_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
        rbf_gamma_scales=(0.5, 1.0, 2.0),
        inner_folds=3,
        route_shuffle_replicates=6,
    )


def _decision(report: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(report.get("decision"), "decision")


def _diagnosis(report: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(report.get("diagnosis"), "diagnosis")


def _all_metrics(report: Mapping[str, object], model: str) -> Mapping[str, object]:
    return _mapping(
        _mapping(_model(report, model).get("subsets"), f"{model} subsets").get("all"),
        f"{model} all-target metrics",
    )


def deterministic_helpers_smoke() -> None:
    targets = ("A", "A", "B", "C", "D", "E", "F", "G")
    strata = ("s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2")
    first = deterministic_target_folds(targets, n_folds=3, seed=57, strata=strata)
    second = deterministic_target_folds(targets, n_folds=3, seed=57, strata=strata)
    assert np.array_equal(first, second)
    assert first[0] == first[1]
    assert any(
        not np.array_equal(
            first,
            deterministic_target_folds(targets, n_folds=3, seed=seed, strata=strata),
        )
        for seed in range(58, 70)
    )

    profiles = np.arange(48, dtype=np.float64).reshape(8, 6)
    shuffled = deterministic_profile_shuffle(profiles, seed=57, strata=strata)
    repeated = deterministic_profile_shuffle(profiles, seed=57, strata=strata)
    assert np.array_equal(shuffled, repeated)
    assert not np.array_equal(shuffled, profiles)
    assert any(
        not np.array_equal(
            shuffled,
            deterministic_profile_shuffle(profiles, seed=seed, strata=strata),
        )
        for seed in range(58, 70)
    )
    for label in sorted(set(strata)):
        rows = np.flatnonzero(np.asarray(strata) == label)
        before = sorted(map(tuple, profiles[rows].tolist()))
        after = sorted(map(tuple, shuffled[rows].tolist()))
        assert before == after
    # The configured production defaults must retain the prespecified null size.
    assert ResponseLearnabilityConfig().route_shuffle_replicates >= 20
    assert ResponseLearnabilityConfig().model_seeds == (42, 137, 911)
    _validate_production_config(ResponseLearnabilityConfig())
    try:
        _validate_production_config(_smoke_config())
    except ValueError as error:
        assert "durable v5.7 reports require" in str(error)
    else:
        raise AssertionError("Reduced smoke evidence was accepted for a durable report")


def split_half_population_smoke() -> None:
    class TinyBundle:
        def __init__(self) -> None:
            rng = np.random.default_rng(77)
            self.accessibility = sparse.csr_matrix(
                rng.binomial(1, 0.25, size=(896, 12)).astype(np.float32)
            )
            self.source_rows = np.arange(896, dtype=np.int64)
            self.groups = {
                ("validation", "screen_1", "A"): np.arange(0, 128),
                ("validation", "screen_1", "B"): np.arange(128, 384),
                ("validation", "screen_1", "NTC"): np.arange(384, 896),
            }

        def rows(self, split: str, screen: str, target: str) -> np.ndarray:
            return self.groups.get((split, screen, target), np.zeros(0, dtype=np.int64))

        def target_screens(self, split: str, target: str) -> list[str]:
            return ["screen_1"] if (split, "screen_1", target) in self.groups else []

    bundle = TinyBundle()
    first, second = _ordered_half(bundle, bundle.rows("validation", "screen_1", "B"), 42, 128)
    assert len(first) == len(second) == 128
    assert not set(first).intersection(second)
    config = ResponseLearnabilityConfig(
        reliability_replicates=4,
        null_permutations=4,
        min_cells_per_half=64,
        max_cells_per_half=128,
        route_shuffle_replicates=2,
    )
    discovery, outcome, reliability, nulls = _split_half_data(bundle, ("A", "B"), config)
    assert discovery.shape == outcome.shape == (2, 12)
    assert reliability["A"]["target_cells_per_half"] == 64
    assert reliability["B"]["target_cells_per_half"] == 128
    assert set(nulls) == {"screen_1|half=64", "screen_1|half=128"}

    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "matrix.npz"
        source = sparse.random(
            30,
            17,
            density=0.2,
            random_state=np.random.default_rng(91),
            format="csr",
            dtype=np.float32,
        )
        sparse.save_npz(archive, source, compressed=True)
        rows = np.asarray([0, 3, 4, 17, 29], dtype=np.int64)
        selected = _selected_csr(archive, rows)
        assert selected.shape == (len(rows), source.shape[1])
        assert (selected != source[rows]).nnz == 0


def training_split_rank_smoke() -> None:
    rng = np.random.default_rng(83)
    targets = [f"RANK_{index}" for index in range(40)]
    screens = ["screen_1" if index % 2 else "screen_2" for index in range(40)]
    direction = rng.normal(size=32)
    direction /= np.linalg.norm(direction)
    amplitude = rng.normal(0.0, 3.0, size=(40, 1))
    outcomes = amplitude * direction[None, :]
    discovery = outcomes + rng.normal(0.0, 0.35, size=outcomes.shape)
    config = ResponseLearnabilityConfig(
        reliability_replicates=4,
        null_permutations=4,
        rank_grid=(1, 2, 4, 8),
        inner_folds=4,
        route_shuffle_replicates=2,
    )
    rank, rows, audit = _rank_cv(
        discovery,
        outcomes,
        targets,
        screens,
        config,
        np.ones(40, dtype=bool),
    )
    assert rank < max(config.rank_grid)
    assert audit["direction"] == "training_split_A_to_disjoint_split_B"
    assert all("training_split_a_to_b_mean_nrmse" in row for row in rows)


def null_and_high_rank_smoke(config: ResponseLearnabilityConfig) -> None:
    null = _run(_null_fixture(), config=config)
    assert not bool(_decision(null)["measurement_gate"])
    assert not bool(_decision(null)["open_sealed_test"])
    assert _diagnosis(null)["primary_failure_class"] == "MEASUREMENT_LIMITED"

    high_rank = _run(_stable_high_rank_fixture(), config=config)
    assert bool(_decision(high_rank)["measurement_gate"])
    assert not bool(_decision(high_rank)["low_rank_ceiling_gate"])
    assert (
        _diagnosis(high_rank)["primary_failure_class"]
        == "STABLE_HIGH_RANK_OR_LATENT_BOTTLENECK"
    )


def linear_smoke(config: ResponseLearnabilityConfig) -> Mapping[str, object]:
    report = _run(_linear_fixture(), config=config)
    decision = _decision(report)
    assert bool(decision["measurement_gate"])
    assert bool(decision["low_rank_ceiling_gate"])
    assert bool(decision["route_linear_gate"])
    assert _diagnosis(report)["primary_failure_class"] == "LINEAR_ROUTE_SIGNAL"
    assert "WLD_OPTIMIZATION_OR_PROPAGATION_FAILURE" in _diagnosis(report)["flags"]
    assert not bool(decision["open_sealed_test"])
    assert _all_metrics(report, "route_linear")["nrmse"]["mean"] < 0.20

    # Selected hyperparameters and seeded route-null rows are deterministic.
    # created_utc is intentionally excluded.
    repeated = _run(_linear_fixture(), config=config)
    assert report["models"] == repeated["models"]
    assert report["matched_route_shuffles"] == repeated["matched_route_shuffles"]
    assert report["decision"] == repeated["decision"]
    assert report["diagnosis"] == repeated["diagnosis"]
    return report


def nonlinear_smoke(config: ResponseLearnabilityConfig) -> None:
    report = _run(_nonlinear_fixture(), config=config)
    decision = _decision(report)
    linear_nrmse = float(_all_metrics(report, "route_linear")["nrmse"]["mean"])
    nonlinear_nrmse = float(_all_metrics(report, "static_nonlinear")["nrmse"]["mean"])
    assert bool(decision["measurement_gate"])
    assert bool(decision["low_rank_ceiling_gate"])
    assert not bool(decision["route_linear_gate"])
    assert bool(decision["static_nonlinear_gate"])
    assert nonlinear_nrmse <= 0.98 * linear_nrmse
    assert _diagnosis(report)["primary_failure_class"] == "STATIC_NONLINEAR_LEARNABLE"


def route_independent_smoke(config: ResponseLearnabilityConfig) -> None:
    report = _run(_route_independent_fixture(), config=config)
    decision = _decision(report)
    assert bool(decision["measurement_gate"])
    assert bool(decision["low_rank_ceiling_gate"])
    assert not bool(decision["route_linear_gate"])
    assert not bool(decision["static_nonlinear_gate"])
    assert _diagnosis(report)["primary_failure_class"] in {
        "ROUTE_PRIOR_OR_TARGET_MAPPING_INSUFFICIENT",
        "WHOLE_TARGET_SHIFT_OR_OVERFIT",
    }
    assert "TOPOLOGY_NONSPECIFIC" in _diagnosis(report)["flags"]


def screen_and_target_weighting_smoke(config: ResponseLearnabilityConfig) -> None:
    report = _run(_screen_confounding_fixture(), config=config)
    flags = _diagnosis(report)["flags"]
    assert "TARGET_NONSPECIFIC_SCREEN_RESPONSE" in flags
    generic = _all_metrics(report, "training_screen_mean")
    best_route = min(
        _all_metrics(report, "route_linear")["nrmse"]["mean"],
        _all_metrics(report, "static_nonlinear")["nrmse"]["mean"],
    )
    assert generic["nrmse"]["mean"] <= best_route + config.numerical_tolerance

    # Every metric is an equal mean across the target rows; cell-rich targets
    # cannot be duplicated into a larger contribution after aggregation.
    target_rows = generic["target_metrics"]
    expected = float(np.mean([row["nrmse"] for row in target_rows]))
    assert np.isclose(generic["nrmse"]["mean"], expected, atol=1e-12, rtol=0.0)
    assert generic["target_count"] == len(_screen_confounding_fixture().validation_targets)

    # Duplicate one development target eleven times.  Aggregation must collapse
    # those rows before scoring, so that target still contributes exactly once.
    fixture = _linear_fixture(seed=7113)
    duplicate_count = 11
    duplicated_indices = np.concatenate(
        (np.zeros(duplicate_count, dtype=np.int64), np.arange(len(fixture.validation_targets)))
    )
    duplicated = evaluate_learnability(
        fixture.train_responses,
        fixture.validation_discovery[duplicated_indices],
        fixture.validation_outcomes[duplicated_indices],
        fixture.train_route_features,
        fixture.validation_route_features[duplicated_indices],
        train_targets=fixture.train_targets,
        validation_targets=tuple(fixture.validation_targets[index] for index in duplicated_indices),
        train_screens=fixture.train_screens,
        validation_screens=tuple(fixture.validation_screens[index] for index in duplicated_indices),
        route_supported=None,
        historical_wld=HISTORICAL_NEAR_PERSISTENCE,
        config=config,
    )
    unduplicated = _run(fixture, config=config)
    assert duplicated["rosters"] == unduplicated["rosters"]
    assert duplicated["models"] == unduplicated["models"]
    assert duplicated["decision"] == unduplicated["decision"]


def unsupported_target_smoke(config: ResponseLearnabilityConfig) -> None:
    fixture = _linear_fixture(seed=7110)
    unsupported = len(fixture.validation_targets) - 1
    validation_routes = fixture.validation_route_features.copy()
    validation_routes[unsupported] = 0.0
    supported = fixture.route_supported.copy()
    supported[unsupported] = False
    modified = Fixture(
        train_responses=fixture.train_responses,
        validation_discovery=fixture.validation_discovery,
        validation_outcomes=fixture.validation_outcomes,
        train_route_features=fixture.train_route_features,
        validation_route_features=validation_routes,
        train_targets=fixture.train_targets,
        validation_targets=fixture.validation_targets,
        train_screens=fixture.train_screens,
        validation_screens=fixture.validation_screens,
        route_supported=supported,
    )
    report = _run(modified, config=config)
    target = fixture.validation_targets[unsupported]
    for model_name in ("route_linear", "static_nonlinear"):
        model = _model(report, model_name)
        assert model["unsupported_targets_receive_zero_response"] is True
        rows = _all_metrics(report, model_name)["target_metrics"]
        row = next(value for value in rows if value["target"] == target)
        assert np.isclose(row["nrmse"], 1.0, atol=1e-12, rtol=0.0)
        assert np.isclose(row["cosine"], 0.0, atol=1e-12, rtol=0.0)


def sealed_claim_smoke(valid_report: Mapping[str, object]) -> None:
    validate_claims(valid_report)
    # The durable runner writes this mapping directly to JSON.
    json.dumps(valid_report, allow_nan=False)
    for claim in (
        "fresh_wld_training",
        "test_values_materialized",
        "test_targets_evaluated",
        "confirmatory_inference",
        "digital_twin_claim",
        "attractor_claim",
    ):
        unsafe = copy.deepcopy(valid_report)
        unsafe["claims"][claim] = True
        try:
            validate_claims(unsafe)
        except ValueError as error:
            assert "boundary" in str(error)
        else:
            raise AssertionError(f"Unsafe claim was accepted: {claim}")
    unsafe = copy.deepcopy(valid_report)
    unsafe["decision"]["open_sealed_test"] = True
    try:
        validate_claims(unsafe)
    except ValueError as error:
        assert "boundary" in str(error)
    else:
        raise AssertionError("Sealed-test opening was accepted")


def main() -> None:
    config = _smoke_config()
    deterministic_helpers_smoke()
    split_half_population_smoke()
    training_split_rank_smoke()
    null_and_high_rank_smoke(config)
    linear_report = linear_smoke(config)
    nonlinear_smoke(config)
    route_independent_smoke(config)
    screen_and_target_weighting_smoke(config)
    unsupported_target_smoke(config)
    sealed_claim_smoke(linear_report)

    assert linear_report["historical_wld"]["mean_response_nrmse"] > 0.99
    assert linear_report["historical_wld"]["mean_response_cosine"] < 0.02
    assert linear_report["decision"]["open_sealed_test"] is False
    assert linear_report["claims"]["test_values_materialized"] is False
    assert linear_report["claims"]["test_targets_evaluated"] is False

    print("PASS: null and stable-high-rank failures are distinguished")
    print("PASS: low-rank linear and static nonlinear response fixtures")
    print("PASS: route-independent ceiling and screen-confounding diagnosis")
    print("PASS: deterministic target folds, profile shuffles, and seeded controls")
    print("PASS: disjoint target/NTC halves and count-matched NTC nulls")
    print("PASS: training split-A to disjoint split-B rank selection")
    print("PASS: unsupported targets receive exact zero-response persistence")
    print("PASS: sealed claims and historical near-persistence WLD never open test")


if __name__ == "__main__":
    main()
