"""Synthetic contract tests for the WLD v6.0 virtual-tissue scaffold.

The fixtures validate information flow, provenance, leakage, integration and
source sealing. They contain no biological assay values and make no biological,
digital-twin or attractor claim.
"""

from __future__ import annotations

import copy
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np

from wld_axolotl_data_v60 import (
    MetadataFetchRefused,
    _MetadataRedirectHandler,
    _metadata_content_type,
    audit_sources,
    load_registry,
    validate_registry,
)
from wld_regulatory_twin_v60 import (
    CausalFeatureDeclaration,
    ContextualRouteParameters,
    GenomicCoordinate,
    GraphMaskedResidual,
    HierarchicalField,
    NodeKineticModulation,
    ProvenanceState,
    ProvenanceTensor,
    RealizedTwinContext,
    RegulatoryGraph,
    RegulatoryTwin,
    RouteIntervention,
    TwinConfig,
    TwinContext,
    TwinKinetics,
    TwinState,
    audit_encoder_features,
)


SEALED_ACCESSION = "GSE315993"


def raises(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"{function.__name__} did not raise {error_type.__name__}")


def pt(
    values,
    names: tuple[str, ...],
    *,
    states: object = ProvenanceState.OBSERVED,
    uncertainty=0.0,
    method: str | None = None,
    lineage: tuple[str, ...] = ("synthetic_v60_fixture",),
    measurement_time: float = 0.0,
    source_role: str = "synthetic_fixture",
    source_partition: str = "development",
    source_accessions: tuple[str, ...] = ("SYNTHETIC_V60_FIXTURE",),
) -> ProvenanceTensor:
    """Construct an explicitly lineage-backed synthetic observation."""

    return ProvenanceTensor(
        np.asarray(values, dtype=np.float64),
        states,
        uncertainty,
        source_feature_names=tuple(names),
        source_lineage=tuple(lineage),
        method_lineage=method,
        measurement_time=measurement_time,
        source_role=source_role,
        source_partition=source_partition,
        source_accessions=source_accessions,
    )


def graph() -> RegulatoryGraph:
    value = RegulatoryGraph(
        cue_names=("injury_cue",),
        signal_names=("ERK",),
        tf_names=("FOS",),
        enhancer_names=("enh1",),
        gene_names=("fos_target",),
        cue_signal_mask=np.ones((1, 1), dtype=bool),
        signal_tf_mask=np.ones((1, 1), dtype=bool),
        tf_enhancer_mask=np.ones((1, 1), dtype=bool),
        enhancer_gene_mask=np.ones((1, 1), dtype=bool),
        neighbor_signal_mask=np.ones((1, 1), dtype=bool),
        enhancer_coordinates=(GenomicCoordinate("chr1", 10, 20),),
        gene_coordinates=(GenomicCoordinate("chr1", 30, 40),),
    )
    value.validate()
    assert value.dimensions == {
        "cues": 1,
        "signals": 1,
        "tfs": 1,
        "enhancers": 1,
        "genes": 1,
        "metabolites": 0,
    }
    return value


def route_parameters(*, subject_specific: bool = False) -> ContextualRouteParameters:
    # A candidate edge is inactive unless explicit, lineage-backed sign
    # evidence activates it. The per-context fixture is inferred from measured
    # ATAC—not a donor/subject lookup key.
    sign_values = (
        np.asarray([[[1.0]], [[2.5]]], dtype=np.float64)
        if subject_specific
        else np.asarray(2.0, dtype=np.float64)
    )
    return ContextualRouteParameters(
        sign=HierarchicalField(
            subject_effect=pt(
                sign_values,
                ("ATAC_peak_context",),
                states=ProvenanceState.MODEL_INFERRED,
                method="measured_ATAC_route_regression",
            )
        ),
    )


def context(value_graph: RegulatoryGraph, *, varying: bool = True) -> TwinContext:
    routes = {
        name: route_parameters(subject_specific=(varying and name == "tf_enhancer"))
        for name in value_graph.route_masks
    }
    return TwinContext(
        cues=pt(
            np.ones((2, 1), dtype=np.float64),
            ("injury_cue",),
        ),
        accessibility=pt(
            np.asarray([[0.25], [0.90]], dtype=np.float64),
            ("ATAC_peaks",),
            states=np.asarray(
                [["observed"], ["reference_transferred"]], dtype=object
            ),
            uncertainty=np.asarray([[0.0], [0.08]], dtype=np.float64),
            method="declared_reference_accessibility_transfer",
        ),
        tf_availability=pt(
            np.ones((2, 1), dtype=np.float64), ("protein_abundance",)
        ),
        contact=pt(
            np.ones((2, 1, 1), dtype=np.float64), ("HiC_contact",)
        ),
        routes=routes,
        declarations=(
            CausalFeatureDeclaration("injury_cue", "cue", 0.0),
            CausalFeatureDeclaration("proximal_distance", "spatial_coordinate", 0.0),
        ),
        encoder_feature_names=("ATAC_peaks",),
        spatial_coordinates=pt(
            np.asarray([[0.2], [0.8]], dtype=np.float64),
            ("proximal_distance",),
            states=ProvenanceState.MODEL_INFERRED,
            uncertainty=0.15,
            method="spatial_reference_transfer",
        ),
        spatial_coordinate_names=("proximal_distance",),
        spatial_adjacency=pt(
            np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            ("proximal_distance",),
            method="radius_graph_from_declared_spatial_coordinate",
        ),
        node_kinetics=NodeKineticModulation(
            rna=pt(
                np.asarray([[0.5], [2.0]], dtype=np.float64),
                ("ATAC_kinetic_context",),
                states=ProvenanceState.MODEL_INFERRED,
                method="measured_ATAC_rate_regression",
            )
        ),
    )


def external_state(
    *,
    chromatin=((0.35,), (0.35,)),
    signal=((0.20,), (0.20,)),
    tf=((0.45,), (0.45,)),
    rna=((0.10,), (0.10,)),
    measurement_time: float = 0.0,
) -> TwinState:
    """Build a public initial state with per-variable temporal provenance."""

    return TwinState.from_provenance(
        chromatin=pt(
            chromatin,
            ("initial_ATAC_chromatin",),
            measurement_time=measurement_time,
        ),
        signal=pt(
            signal,
            ("initial_signaling_measurement",),
            measurement_time=measurement_time,
        ),
        tf=pt(
            tf,
            ("initial_TF_measurement",),
            measurement_time=measurement_time,
        ),
        rna=pt(
            rna,
            ("initial_RNA_measurement",),
            measurement_time=measurement_time,
        ),
    )


def state(measurement_time: float = 0.0) -> TwinState:
    return external_state(measurement_time=measurement_time)


def context_variation_smoke() -> None:
    value_graph = graph()
    value_context = context(value_graph)
    realized = value_context.realize(value_graph, batch_size=2, seed=17)
    active = realized.routes["tf_enhancer"].effective(
        value_graph.route_masks["tf_enhancer"], 2
    )
    assert active.shape == (2, 1, 1)
    assert not np.isclose(active[0, 0, 0], active[1, 0, 0])

    model = RegulatoryTwin(value_graph, TwinKinetics(), TwinConfig(seed=17))
    drift = model.derivative(state(), value_context)
    assert not np.isclose(drift.rna[0, 0], drift.rna[1, 0])
    print("PASS: provenance-backed measured context changes realized active edges")


def provenance_smoke() -> None:
    values = pt(
        values=np.asarray([0.0, 2.0, np.nan]),
        names=("ATAC_peak", "transferred_contact", "missing_feature"),
        states=np.asarray(
            [
                ProvenanceState.OBSERVED,
                ProvenanceState.MODEL_INFERRED,
                ProvenanceState.UNKNOWN,
            ],
            dtype=object,
        ),
        uncertainty=np.asarray([0.0, 0.25, 0.0]),
        method="declared_synthetic_inference",
    )
    values.validate()
    first = values.sample(101)
    second = values.sample(101)
    assert np.ma.allequal(first, second)
    assert np.ma.getmaskarray(first).tolist() == [False, False, True]
    effective = values.effective_values()
    assert effective[0] == 0.0
    assert not bool(np.ma.getmaskarray(effective)[0])
    assert bool(np.ma.getmaskarray(effective)[2])
    assert values.known_mask().tolist() == [True, True, False]
    diagnostics = values.diagnostics()
    counts = diagnostics["counts"]
    assert isinstance(counts, Mapping)
    assert counts["observed"] == 1
    assert counts["reference_transferred"] == 0
    assert counts["model_inferred"] == 1
    assert diagnostics["mean_uncertainty"] > 0.0
    assert diagnostics["source_lineage"] == ["synthetic_v60_fixture"]

    # Missing source/method lineage and negative uncertainty fail closed.
    raises(
        ValueError,
        ProvenanceTensor(
            np.ones(1),
            source_feature_names=("ATAC_peak",),
            source_lineage=(),
            measurement_time=0.0,
        ).validate,
    )
    raises(
        ValueError,
        ProvenanceTensor(
            np.ones(1),
            ProvenanceState.MODEL_INFERRED,
            source_feature_names=("ATAC_peak",),
            source_lineage=("fixture",),
            measurement_time=0.0,
        ).validate,
    )
    raises(
        ValueError,
        ProvenanceTensor(
            np.ones(1),
            source_feature_names=("ATAC_peak",),
            source_lineage=("fixture",),
        ).validate,
    )
    raises(
        ValueError,
        pt(
            np.ones(1),
            ("ATAC_peak",),
            states=ProvenanceState.MODEL_INFERRED,
            uncertainty=-0.1,
            method="fixture_inference",
        ).validate,
    )

    value_graph = graph()
    base = context(value_graph)
    unknown_accessibility = pt(
        [[np.nan], [0.8]],
        ("ATAC_peaks",),
        states=np.asarray([["unknown"], ["observed"]], dtype=object),
    )
    raises(
        ValueError,
        replace(base, accessibility=unknown_accessibility).realize,
        value_graph,
        2,
        7,
    )
    measured_zero = replace(
        base,
        accessibility=pt([[0.0], [0.0]], ("ATAC_peaks",)),
    ).realize(value_graph, 2, 7)
    assert np.array_equal(measured_zero.accessibility, np.zeros((2, 1)))
    assert bool(measured_zero.known_masks["accessibility"].all())
    raises(
        ValueError,
        replace(base, cues=np.ones((2, 1))).realize,
        value_graph,
        2,
        7,
    )
    print("PASS: unknown gates fail closed; measured zero and lineage remain explicit")


def leakage_smoke() -> None:
    declarations = (
        CausalFeatureDeclaration("injury_cue", "cue", 0.0),
        CausalFeatureDeclaration("proximal_distance", "spatial_coordinate", 0.0),
    )
    accepted = audit_encoder_features(
        ("ATAC_peaks", "injury_cue", "proximal_distance"), declarations
    )
    assert accepted["direct_identity_or_state_proxies"] == []
    for forbidden in (
        "cell_type",
        "cluster",
        "pseudotime",
        "donor_id",
        "study_id",
        "target_state",
        "future_RNA",
        "future_ATAC",
        "cell_state",
        "RNA_counts",
        "guide_identity",
        "donor",
        "subject",
        "study",
        "identity",
        "lineage",
        "animal_id",
        "patient_id",
        "sample_id",
        "specimen_id",
        "individual_id",
        "participant_id",
        "class_id",
        "cohort_id",
        "replicate_id",
        "GSE315993_ATAC_peaks",
    ):
        raises(ValueError, audit_encoder_features, (forbidden,), declarations)
    raises(ValueError, audit_encoder_features, ("injury_cue",), ())
    raises(
        ValueError,
        audit_encoder_features,
        ("injury_cue",),
        (CausalFeatureDeclaration("injury_cue", "cue", 1.0),),
        prediction_origin_time=0.0,
    )
    value_graph = graph()
    value_context = context(value_graph)
    model = RegulatoryTwin(value_graph, TwinKinetics(), TwinConfig(seed=19))
    raw_state = TwinState(
        chromatin=np.asarray([[0.35], [0.35]]),
        signal=np.asarray([[0.20], [0.20]]),
        tf=np.asarray([[0.45], [0.45]]),
        rna=np.asarray([[0.10], [0.10]]),
    )
    raises(ValueError, model.derivative, raw_state, value_context)
    raises(ValueError, model.rollout, raw_state, value_context, [0.0, 0.1])
    raises(ValueError, model.derivative, state(1.0), value_context)
    raises(ValueError, model.rollout, state(1.0), value_context, [0.0, 0.1])
    identity_state = state()
    assert identity_state.input_provenance is not None
    identity_provenance = dict(identity_state.input_provenance)
    identity_provenance["rna"] = pt(
        identity_state.rna,
        ("animal_id",),
    )
    raises(
        ValueError,
        model.derivative,
        replace(identity_state, input_provenance=identity_provenance),
        value_context,
    )
    sealed_context = replace(
        value_context,
        accessibility=pt(
            [[0.8], [0.3]],
            ("ATAC_peaks",),
            lineage=("GSE315993 sealed external test matrix",),
            source_accessions=("GSE315993",),
        ),
    )
    raises(ValueError, model.derivative, state(), sealed_context)
    sealed_feature_context = replace(
        value_context,
        accessibility=pt(
            [[0.8], [0.3]],
            ("GSE315993_ATAC_peaks",),
        ),
    )
    raises(ValueError, model.derivative, state(), sealed_feature_context)
    sealed_method_context = replace(
        value_context,
        accessibility=pt(
            [[0.8], [0.3]],
            ("ATAC_peaks",),
            method="regression_fitted_on_GSE315993_sealed_test",
        ),
    )
    raises(ValueError, model.derivative, state(), sealed_method_context)
    sealed_initial = state()
    assert sealed_initial.input_provenance is not None
    sealed_provenance = dict(sealed_initial.input_provenance)
    sealed_provenance["rna"] = pt(
        sealed_initial.rna,
        ("initial_RNA_measurement",),
        lineage=("GSE315993 sealed external test matrix",),
        source_accessions=("GSE315993",),
    )
    raises(
        ValueError,
        model.derivative,
        replace(sealed_initial, input_provenance=sealed_provenance),
        value_context,
    )
    print(
        "PASS: identity/outcome proxies and unsealed/future initial states are rejected"
    )


def residual_and_persistence_smoke() -> None:
    value_graph = graph()
    raises(
        ValueError,
        GraphMaskedResidual(
            cue_signal=np.asarray([[1.0]], dtype=np.float64), bound=0.05
        ).validate,
        value_graph,
    )
    supported = GraphMaskedResidual(
        cue_signal=np.asarray([[1e9]], dtype=np.float64),
        bound=0.05,
        source_feature_names=("ATAC_peak",),
        source_lineage=("training_only_fixture",),
        method_lineage="bounded_residual_fit",
        source_role="synthetic_fixture",
        source_partition="development",
        source_accessions=("SYNTHETIC_V60_FIXTURE",),
    )
    supported.validate(value_graph)
    raises(
        ValueError,
        replace(
            supported,
            source_lineage=("GSE315993 sealed external test matrix",),
            source_accessions=("GSE315993",),
        ).validate,
        value_graph,
    )
    raises(
        ValueError,
        replace(
            supported,
            method_lineage="regression_fitted_on_GSE315993_sealed_test",
        ).validate,
        value_graph,
    )
    effective = supported.effective("cue_signal", value_graph.cue_signal_mask, 2)
    assert effective.shape == (2, 1, 1)
    assert float(np.max(np.abs(effective))) <= 0.05 + 1e-15
    assert not hasattr(supported, "rna")

    masked_graph = RegulatoryGraph(
        cue_names=("cue",),
        signal_names=("supported", "unsupported"),
        tf_names=("TF",),
        enhancer_names=("enh",),
        gene_names=("gene",),
        cue_signal_mask=np.asarray([[1, 0]], dtype=bool),
        signal_tf_mask=np.asarray([[1], [0]], dtype=bool),
        tf_enhancer_mask=np.ones((1, 1), dtype=bool),
        enhancer_gene_mask=np.ones((1, 1), dtype=bool),
        enhancer_coordinates=(GenomicCoordinate("chr1", 100, 200),),
        gene_coordinates=(GenomicCoordinate("chr1", 300, 400),),
    )
    raises(
        ValueError,
        GraphMaskedResidual(
            cue_signal=np.asarray([[0.0, 1.0]]),
            bound=0.05,
            source_feature_names=("ATAC_peak_context",),
            source_lineage=("synthetic_v60_fixture",),
            method_lineage="bounded_graph_residual_fixture",
            source_role="synthetic_fixture",
            source_partition="development",
            source_accessions=("SYNTHETIC_V60_FIXTURE",),
        ).validate,
        masked_graph,
    )

    model = RegulatoryTwin(
        value_graph,
        TwinKinetics(),
        TwinConfig(seed=23, persistence_when_all_routes_removed=True),
        residual=supported,
    )
    initial = external_state(
        chromatin=((0.0,), (1.0,)),
        signal=((0.2,), (-0.3,)),
        tf=((0.5,), (0.7,)),
        rna=((0.4,), (0.6,)),
    )
    intervention = RouteIntervention.frozen_all()
    derivative = model.derivative(initial, context(value_graph), intervention)
    for name in ("chromatin", "signal", "tf", "rna"):
        assert np.array_equal(getattr(derivative, name), np.zeros_like(getattr(initial, name)))
    frozen = model.rollout(initial, context(value_graph), [0.0, 0.1, 0.3], intervention)
    for later in frozen.states:
        for name in ("chromatin", "signal", "tf", "rna"):
            assert np.array_equal(getattr(later, name), getattr(initial, name))
    assert frozen.diagnostics["exact_persistence_rollout"] is True
    raises(
        ValueError,
        TwinConfig(persistence_when_all_routes_removed=False).validate,
    )
    print("PASS: bounded residual stays on graph; frozen route removal is exact persistence")


def adversarial_context_smoke() -> None:
    value_graph = graph()
    candidate_only = ContextualRouteParameters().effective(
        value_graph.route_masks["tf_enhancer"], 2
    )
    assert np.array_equal(candidate_only, np.zeros((2, 1, 1)))

    raises(
        ValueError,
        HierarchicalField(
            subject_effect=np.asarray([[[-0.5]], [[0.5]]], dtype=np.float64)
        ).total,
        (1, 1),
        2,
    )

    base = context(value_graph, varying=False)
    # Required gates, kinetic modifiers, and adjacency cannot be raw batch
    # arrays that could serve as subject lookup tables.
    raises(
        ValueError,
        replace(base, spatial_adjacency=np.eye(2)).realize,
        value_graph,
        2,
        9,
    )
    raises(
        ValueError,
        replace(
            base,
            node_kinetics=NodeKineticModulation(signal=np.ones((2, 1))),
        ).realize,
        value_graph,
        2,
        9,
    )

    # Temporal provenance applies to every assimilated path—not only named
    # cues. No future gate, spatial value, edge effect or kinetic modulator may
    # leak across the prediction origin.
    future_required = (
        (
            "cues",
            pt([[1.0], [1.0]], ("injury_cue",), measurement_time=1.0),
        ),
        (
            "accessibility",
            pt([[0.5], [0.5]], ("ATAC_peaks",), measurement_time=1.0),
        ),
        (
            "tf_availability",
            pt([[1.0], [1.0]], ("protein_abundance",), measurement_time=1.0),
        ),
        (
            "contact",
            pt(
                np.ones((2, 1, 1)),
                ("HiC_contact",),
                measurement_time=1.0,
            ),
        ),
        (
            "spatial_coordinates",
            pt(
                [[0.2], [0.8]],
                ("proximal_distance",),
                states=ProvenanceState.MODEL_INFERRED,
                method="future_spatial_inference",
                measurement_time=1.0,
            ),
        ),
    )
    for field_name, future_value in future_required:
        raises(
            ValueError,
            replace(base, **{field_name: future_value}).realize,
            value_graph,
            2,
            9,
        )

    future_adjacency = replace(
        base,
        spatial_adjacency=pt(
            [[0.0, 1.0], [1.0, 0.0]],
            ("proximal_distance",),
            method="future_radius_graph",
            measurement_time=1.0,
        ),
    )
    raises(ValueError, future_adjacency.realize, value_graph, 2, 9)

    future_kinetic = replace(
        base,
        node_kinetics=NodeKineticModulation(
            signal=pt(
                [[1.0], [1.2]],
                ("ATAC_kinetic_context",),
                states=ProvenanceState.MODEL_INFERRED,
                method="future_rate_regression",
                measurement_time=1.0,
            )
        ),
    )
    raises(ValueError, future_kinetic.realize, value_graph, 2, 9)

    future_routes = dict(base.routes)
    future_routes["tf_enhancer"] = ContextualRouteParameters(
        sign=HierarchicalField(
            subject_effect=pt(
                [[[1.0]], [[1.2]]],
                ("ATAC_peak_context",),
                states=ProvenanceState.MODEL_INFERRED,
                method="future_route_regression",
                measurement_time=1.0,
            )
        )
    )
    future_route_context = replace(base, routes=future_routes)
    future_route_model = RegulatoryTwin(
        value_graph, TwinKinetics(), TwinConfig(seed=9)
    )
    raises(ValueError, future_route_model.derivative, state(), future_route_context)

    no_neighbor_route = replace(value_graph, neighbor_signal_mask=None)
    no_neighbor_context = context(no_neighbor_route, varying=False)
    # Add the otherwise valid spatial graph to a topology that has no named
    # neighbor route; it must not become an unrestricted mixing channel.
    no_neighbor_context = replace(
        no_neighbor_context,
        spatial_adjacency=base.spatial_adjacency,
    )
    raises(
        ValueError,
        no_neighbor_context.realize,
        no_neighbor_route,
        2,
        9,
    )

    identical = replace(
        base,
        accessibility=pt([[0.5], [0.5]], ("ATAC_peaks",)),
        node_kinetics=NodeKineticModulation(
            signal=pt(
                [[0.5], [2.0]],
                ("ATAC_kinetic_context",),
                states=ProvenanceState.MODEL_INFERRED,
                method="measured_ATAC_rate_regression",
            )
        ),
    )
    model = RegulatoryTwin(value_graph, TwinKinetics(), TwinConfig(seed=9))
    kinetic_drift = model.derivative(state(), identical)
    assert not np.isclose(kinetic_drift.signal[0, 0], kinetic_drift.signal[1, 0])

    realized = identical.realize(value_graph, 2, 9)
    assert isinstance(realized, RealizedTwinContext)
    raises(TypeError, model.derivative, state(), realized)
    print(
        "PASS: inactive defaults, temporal context, named space and realized-context seal"
    )


def rk4_smoke() -> None:
    value_graph = graph()
    model = RegulatoryTwin(value_graph, TwinKinetics(), TwinConfig(seed=31))
    initial = state()
    trajectory = model.rollout(initial, context(value_graph), [0.0, 0.05, 0.10, 0.20])
    assert len(trajectory.states) == 4
    assert trajectory.times.shape == (4,)
    for step in trajectory.states:
        step.validate(value_graph)
        for name, shape in (
            ("chromatin", (2, 1)),
            ("signal", (2, 1)),
            ("tf", (2, 1)),
            ("rna", (2, 1)),
        ):
            value = np.asarray(getattr(step, name))
            assert value.shape == shape
            assert np.isfinite(value).all()
    assert not np.array_equal(trajectory.final_state.rna, initial.rna)
    assert str(trajectory.diagnostics["integrator"]).startswith("RK4")
    assert trajectory.diagnostics["direct_cue_to_rna_path"] is False
    assert trajectory.diagnostics["direct_residual_to_rna_path"] is False
    assert trajectory.diagnostics["explicit_route_delay_model"] is False
    assert trajectory.diagnostics["route_kinetics_are_instantaneous_rate_multipliers"] is True
    assert trajectory.diagnostics["digital_twin_claim"] is False
    assert trajectory.diagnostics["attractor_claim"] is False
    print("PASS: RK4 preserves named-state shapes and finite graph-routed outputs")


def _find_accession_entry(value: object, accession: str) -> list[Mapping[str, object]]:
    answer: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        candidate = str(
            value.get(
                "accession",
                value.get("study_id", value.get("study", value.get("id", ""))),
            )
        ).upper()
        if candidate == accession.upper() and ("role" in value or "split" in value):
            answer.append(value)
        for child in value.values():
            answer.extend(_find_accession_entry(child, accession))
    elif isinstance(value, list):
        for child in value:
            answer.extend(_find_accession_entry(child, accession))
    return answer


def _pairing_records(value: object) -> list[Mapping[str, object]]:
    answer: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("pairing"), Mapping):
            answer.append(value)
        for child in value.values():
            answer.extend(_pairing_records(child))
    elif isinstance(value, list):
        for child in value:
            answer.extend(_pairing_records(child))
    return answer


def registry_smoke() -> None:
    registry_path = Path(__file__).with_name("wld_v60_axolotl_sources.json")
    registry = load_registry(registry_path)
    validation = validate_registry(registry, registry_path=registry_path)
    assert isinstance(validation, Mapping)
    assert validation["verified_exact_pairing_records"] == 0
    assert validation["exact_deposited_pairing_records"] == 0
    assert validation["metadata_declared_exact_pairing_records"] == 4
    sealed = _find_accession_entry(registry, SEALED_ACCESSION)
    assert len(sealed) == 1
    role = str(sealed[0].get("role", sealed[0].get("split", ""))).lower()
    assert role == "sealed_external_test"

    invalid = copy.deepcopy(registry)
    invalid_sealed = _find_accession_entry(invalid, SEALED_ACCESSION)
    assert len(invalid_sealed) == 1
    invalid_sealed[0]["split"] = "development_reference"
    raises(ValueError, validate_registry, invalid, registry_path=registry_path)

    duplicate_role = copy.deepcopy(registry)
    duplicate = copy.deepcopy(duplicate_role["reference_atlas_sources"][0])
    duplicate["record_id"] = "GSE315993_illegal_atlas_reuse"
    duplicate["study"] = SEALED_ACCESSION
    duplicate["source_url"] = sealed[0]["metadata_url"]
    duplicate_role["reference_atlas_sources"].append(duplicate)
    raises(
        ValueError,
        validate_registry,
        duplicate_role,
        registry_path=registry_path,
    )

    duplicate_prior_role = copy.deepcopy(registry)
    duplicate_prior_role["mechanistic_prior_sources"][0]["source_id"] = "GSE106269"
    raises(
        ValueError,
        validate_registry,
        duplicate_prior_role,
        registry_path=registry_path,
    )

    duplicate_atlas = copy.deepcopy(registry)
    repeated_atlas = copy.deepcopy(duplicate_atlas["reference_atlas_sources"][0])
    repeated_atlas["record_id"] = "duplicate_atlas_record_id"
    duplicate_atlas["reference_atlas_sources"].append(repeated_atlas)
    raises(
        ValueError,
        validate_registry,
        duplicate_atlas,
        registry_path=registry_path,
    )

    pairing_records = _pairing_records(registry)
    assert pairing_records
    pairing_fields = {
        "mode",
        "evidence_type",
        "evidence",
        "identifier_fields",
        "crosswalk_fields",
        "verification_status",
        "schema_materialized",
        "fabricated",
        "expression_similarity_used",
        "cell_label_matching_used",
    }
    for record in pairing_records:
        pairing = record["pairing"]
        assert isinstance(pairing, Mapping)
        assert pairing_fields.issubset(pairing)
        assert pairing.get("fabricated") is False
        assert pairing.get("expression_similarity_used") is False
        assert pairing.get("cell_label_matching_used") is False

    sealed_pairing = sealed[0]["records"][0]["pairing"]
    assert sealed_pairing["mode"] == "same_spot_exact"
    assert sealed_pairing["verification_status"] == "metadata_declared_unverified"
    assert sealed_pairing["schema_materialized"] is False
    assert SEALED_ACCESSION not in {
        str(record.get("study"))
        for record in pairing_records
        if record["pairing"]["mode"] in {"same_cell_exact", "same_spot_exact"}
        and record["pairing"]["verification_status"]
        == "verified_from_materialized_schema"
    }
    exact = next(
        record
        for record in pairing_records
        if record["pairing"]["mode"] in {"same_cell_exact", "same_spot_exact"}
    )
    invalid_pairing = copy.deepcopy(registry)
    invalid_exact = next(
        record
        for record in _pairing_records(invalid_pairing)
        if record.get("study") == exact.get("study")
        and record.get("record_id") == exact.get("record_id")
        and record["pairing"]["mode"] == exact["pairing"]["mode"]
    )
    invalid_exact["pairing"]["fabricated"] = True
    raises(ValueError, validate_registry, invalid_pairing, registry_path=registry_path)

    unpaired = next(
        (
            record
            for record in pairing_records
            if record["pairing"]["mode"] == "unpaired_population"
        ),
        None,
    )
    if unpaired is not None:
        invalid_unpaired = copy.deepcopy(registry)
        changed = next(
            record
            for record in _pairing_records(invalid_unpaired)
            if record.get("study") == unpaired.get("study")
            and record.get("record_id") == unpaired.get("record_id")
            and record["pairing"]["mode"] == "unpaired_population"
        )
        changed["pairing"]["mode"] = "same_cell_exact"
        # Missing deposited identifier evidence must not be silently promoted
        # to exact pairing.
        raises(ValueError, validate_registry, invalid_unpaired, registry_path=registry_path)

    primary_timecourse = next(
        record
        for record in pairing_records
        if record.get("record_id") == "PRJNA589484_scrna_8stage"
    )
    assert primary_timecourse["tissue"] == "upper arm"
    assert primary_timecourse["cell_count"] == 41376
    assert primary_timecourse["stage"] == [
        "homeostatic",
        "trauma",
        "wound healing",
        "early-bud blastema",
        "mid-bud blastema",
        "late-bud blastema",
        "palette",
        "re-differentiated",
    ]
    assert primary_timecourse["time"] == [
        {"value": 0, "unit": "hour", "relation": "homeostatic"},
        {"value": 3, "unit": "hour", "relation": "post_amputation"},
        {"value": 1, "unit": "day", "relation": "post_amputation"},
        {"value": 3, "unit": "day", "relation": "post_amputation"},
        {"value": 7, "unit": "day", "relation": "post_amputation"},
        {"value": 14, "unit": "day", "relation": "post_amputation"},
        {"value": 22, "unit": "day", "relation": "post_amputation"},
        {"value": 33, "unit": "day", "relation": "post_amputation"},
    ]
    print(
        "PASS: structured pairing, unverified exact counts and eight-stage schedule"
    )

    requested: list[str] = []

    def fake_fetch(url, *args, **kwargs):
        requested.append(str(url))
        # The source auditor accepts a compact metadata-fetch record from an
        # injected fetcher; no real network or assay value is used here.
        return {
            "url": str(url),
            "status": 200,
            "content_type": "text/html",
            "bytes_read": 64,
            "headers_checked_before_body": True,
            "measurement_value_bytes_read": 0,
            "sha256": "0" * 64,
            "text": (
                "GSE106269 GSE121737 PRJNA589484 PRJNA682840 GSE243225 "
                "GSE315993 Ambystoma mexicanum"
            ),
        }

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "audit.json"
        audit = audit_sources(
            registry_path=registry_path,
            output_path=output,
            live=True,
            strict_live=True,
            fetcher=fake_fetch,
        )
        assert output.is_file()
    claims = audit["claims"]
    assert claims["metadata_only"] is True
    assert claims["sealed_external_measurement_urls_downloaded"] is False
    assert claims["gse315993_measurement_values_materialized"] is False
    sealed_entry = sealed[0]
    measurement_urls = set()
    for key, value in sealed_entry.items():
        if "url" not in str(key).lower() or "metadata" in str(key).lower():
            continue
        if isinstance(value, str):
            measurement_urls.add(value)
        elif isinstance(value, list):
            measurement_urls.update(str(item) for item in value)
    assert not measurement_urls.intersection(requested)
    assert requested

    # The built-in HTTP redirect policy classifies the Location before
    # urllib can follow it or expose a response body. This complements the
    # injected-fetcher check below, which tests defensive reclassification of
    # an already-resolved effective URL.
    class SyntheticMetadataRequest:
        full_url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE315993"

    measurement_redirects = (
        "/geo/download/suppl/GSE315993_sealed_count_matrix.mtx.gz",
        "/geo/series/GSE315nnn/GSE315993/suppl/spot_coordinates.csv",
        "/geo/series/GSE315nnn/GSE315993/suppl/expression_matrix.tsv.gz",
        "/geo/series/GSE315nnn/GSE315993/suppl/processed_object.rds.gz",
        "/geo/series/GSE315nnn/GSE315993/suppl/spatial_image.png",
        "/geo/query/file.cgi?filename=sealed_values%2Etsv",
    )
    for measurement_url in measurement_redirects:
        redirect_handler = _MetadataRedirectHandler(("www.ncbi.nlm.nih.gov",))
        try:
            redirect_handler.redirect_request(
                SyntheticMetadataRequest(),
                None,
                302,
                "Found",
                {},
                measurement_url,
            )
        except MetadataFetchRefused as error:
            assert error.audit["refusal_reason"] == "redirect_location_classified_as_measurement"
            assert error.audit["headers_checked_before_body"] is True
            assert error.audit["body_bytes_read"] == 0
            assert error.audit["measurement_value_bytes_read"] == 0
            assert len(error.audit["redirect_chain"]) == 1
        else:
            raise AssertionError(
                f"measurement-value redirect was followed: {measurement_url}"
            )
    for content_type in (
        "text/plain", "text/csv", "text/tab-separated-values", "application/csv"
    ):
        assert _metadata_content_type(content_type) is False
    for content_type in ("text/html", "application/json", "application/xml"):
        assert _metadata_content_type(content_type) is True

    def redirected_value_fetch(url, *args, **kwargs):
        if SEALED_ACCESSION in str(url):
            return {
                "url": (
                    "https://www.ncbi.nlm.nih.gov/geo/download/"
                    "suppl/GSE315993_sealed_count_matrix.mtx.gz"
                ),
                "status": 200,
                "content_type": "text/html",
                "content_disposition": "",
                "bytes_read": 0,
                "headers_checked_before_body": True,
                "measurement_value_bytes_read": 0,
                "sha256": "0" * 64,
                "text": "",
            }
        return fake_fetch(url, *args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        refused_output = Path(directory) / "redirect_refusal.json"
        refused = audit_sources(
            registry_path=registry_path,
            output_path=refused_output,
            live=True,
            strict_live=False,
            fetcher=redirected_value_fetch,
        )
        assert refused_output.is_file()
    ledger = refused["live_audit"]["access_ledger"]
    sealed_ledger = next(
        item for item in ledger if item.get("sealed_external_test") is True
    )
    assert sealed_ledger["outcome"] == "refused"
    assert sealed_ledger["refusal_reason"] == "effective_url_classified_as_measurement"
    assert sealed_ledger["headers_checked_before_body"] is True
    assert sealed_ledger["body_bytes_read"] == 0
    assert sealed_ledger["measurement_value_bytes_read"] == 0
    assert refused["claims"]["metadata_only"] is True
    assert refused["claims"]["gse315993_measurement_values_materialized"] is False
    print(
        "PASS: global source roles and pre-follow redirect/value refusal are enforced"
    )


def main() -> None:
    context_variation_smoke()
    provenance_smoke()
    leakage_smoke()
    residual_and_persistence_smoke()
    adversarial_context_smoke()
    rk4_smoke()
    registry_smoke()
    print("PASS: no biological training, digital-twin, attractor or sealed-test claim")


if __name__ == "__main__":
    main()
