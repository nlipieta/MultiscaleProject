"""Synthetic CPU/CUDA contract tests for WLD v5.6 null-aware development."""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

from wld_chromatin_twin_v55 import BranchOverrides, ChromatinTwinPriors
from wld_chromatin_twin_v56 import (
    WLDNullAwareChromatinTwin,
    architecture_contract,
)
from wld_foundation_model_v4 import FoundationPriors, WLDMultistudyFoundationModel
from wld_chromatin_twin_training_v56 import (
    TwinTrainingConfig,
    _make_optimizer,
    _validate_control_generation_audit,
    compile_training_perturbed_mean_baseline,
    evaluate_perturbed_mean_baseline,
    response_focused_loss,
    run_nullaware_development,
    run_twin_development,
    validate_control_priors,
)
from wld_chromatin_modules_v55 import SparseFullChromatinBundle
from wld_v56_topology_controls import build_matched_control_priors


class _FakeEncoder(nn.Module):
    def __init__(self, anchors: int, tfs: int) -> None:
        super().__init__()
        self.anchor_scale = nn.Parameter(torch.ones(anchors))
        self.tfs = int(tfs)

    def forward(
        self,
        *,
        cues: torch.Tensor,
        atac: torch.Tensor,
        rna=None,
        protein=None,
        metabolic=None,
        modality_masks=None,
    ) -> Dict[str, torch.Tensor]:
        context = torch.cat((atac * self.anchor_scale, cues), dim=1)
        tf = torch.nn.functional.softplus(atac[:, : self.tfs])
        return {"biological_context": context, "tf": tf}


class _FakeFoundation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tfs = 3
        self.num_peaks = 4
        self.context_dim = 4
        self.priors = SimpleNamespace(num_cues=1)
        self.encoder = _FakeEncoder(4, 3)
        self.context_network = nn.Sequential(nn.Linear(5, 4), nn.Tanh())


def _model_priors(device: torch.device | str = "cpu") -> ChromatinTwinPriors:
    device = torch.device(device)
    regulator_tf = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # TF-only
            [0.0, 0.0, 0.0],  # complex-only
            [1.0, 0.7, 0.0],  # dual and high-degree
            [0.0, 0.0, 0.0],  # unsupported
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.8],
            [1.0, 0.0, 0.5],
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 1.0],
            [0.8, 0.0, 0.0],
            [0.0, 1.0, 0.5],
            [0.5, 0.0, 0.9],
        ],
        device=device,
    )
    motif = torch.tensor(
        [
            [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.9, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
    )
    regulator_complex = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.7],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9],
            [0.7, 0.0, 0.0],
            [0.0, 0.5, 1.0],
            [0.6, 0.0, 0.8],
        ],
        device=device,
    )
    complex_module = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -0.8, 0.0], [0.0, 0.0, 0.6]],
        device=device,
    )
    module_peak = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.8, -1.0],
        ],
        device=device,
    )
    return ChromatinTwinPriors(
        regulator_tf_support=regulator_tf,
        tf_peak_motif=motif,
        regulator_complex_support=regulator_complex,
        complex_module_effect=complex_module,
        module_peak_loading=module_peak,
        foundation_peak_index=torch.tensor([0, 3, 7, 10], device=device),
    )


def _fixture(
    device: torch.device | str = "cpu",
    *,
    tf_efficacy: float = 1e-3,
    complex_efficacy: float = 1e-3,
) -> Tuple[WLDNullAwareChromatinTwin, ChromatinTwinPriors]:
    device = torch.device(device)
    torch.manual_seed(71)
    priors = _model_priors(device)
    foundation = _FakeFoundation().to(device)
    model = WLDNullAwareChromatinTwin(
        foundation,
        priors,
        tf_initial_efficacy=tf_efficacy,
        complex_initial_efficacy=complex_efficacy,
    ).to(device)
    return model, priors


def _real_foundation_fixture() -> WLDNullAwareChromatinTwin:
    """Construct the real foundation class with zero-width optional modalities."""

    foundation_priors = FoundationPriors(
        peak_to_gene=torch.zeros(4, 5),
        peak_tf_motif=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
        tf_gene_support=torch.zeros(3, 5),
        circuit_tf_tf=torch.zeros(3, 3),
        signal_signal=torch.zeros(2, 2),
        signal_tf=torch.zeros(2, 3),
        cue_signal=torch.tensor([[1.0, 0.0]]),
        tf_peak_effect=torch.zeros(3, 4),
        tf_gene_index=torch.tensor([0, 1, 2]),
        protein_signal=torch.zeros(0, 2),
        metabolic_signal=torch.zeros(0, 2),
        metabolic_tf=torch.zeros(0, 3),
    )
    foundation = WLDMultistudyFoundationModel(
        foundation_priors,
        context_covariate_dim=0,
        context_dim=4,
    )
    return WLDNullAwareChromatinTwin(foundation, _model_priors())


def real_foundation_optimizer_partition_smoke() -> Dict[str, object]:
    """Regress the Colab failure caused by auditing frozen raw coordinates."""

    model = _real_foundation_fixture()
    # Use an intentionally visible decay step so the membership test remains
    # robust at float32 precision; the grouping rule is identical to default.
    config = TwinTrainingConfig(
        learning_rate=1e-2,
        representation_learning_rate=1e-2,
        weight_decay=0.1,
    )
    optimizer, optimized, audit = _make_optimizer(model, config)
    frozen = set(audit["frozen_parameters"])
    grouped = {
        name
        for group in audit["groups"]
        for name in group["parameters"]
    }
    frozen_raw = {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("foundation.field.") and ".raw_" in name.lower()
    }
    frozen_foundation_field = {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("foundation.field.")
    }
    foundation_field_in_optimizer = bool(frozen_foundation_field & grouped)
    assert len(frozen_raw) == 18
    assert frozen_raw <= frozen
    assert frozen_foundation_field <= frozen
    assert not foundation_field_in_optimizer
    named_parameters = dict(model.named_parameters())
    assert all(
        not named_parameters[name].requires_grad
        for name in frozen_foundation_field
    )
    assert len(optimized) == len(grouped)

    decay_by_name = {
        name: float(group["weight_decay"])
        for group in audit["groups"]
        for name in group["parameters"]
    }
    assert all(
        decay_by_name[name] == 0.0
        for name in grouped
        if named_parameters[name].ndim <= 1
    )
    assert any(
        decay_by_name[name] > 0.0
        for name in grouped
        if named_parameters[name].ndim > 1
    )

    for group in audit["groups"]:
        if float(group["weight_decay"]) == 0.0:
            continue
        for name in group["parameters"]:
            lower = name.lower()
            assert ".raw_" not in lower and "gate" not in lower and "efficacy" not in lower

    zero_decay = set(audit["zero_weight_decay_parameters"])
    trainable_raw_or_gate = {
        name
        for name in grouped
        if ".raw_" in name.lower()
        or "gate" in name.lower()
        or "efficacy" in name.lower()
    }
    sensitive_coordinate_decayed = bool(trainable_raw_or_gate - zero_decay)
    assert not sensitive_coordinate_decayed
    assert "field.tf_gate_logit" in zero_decay
    assert "field.complex_gate_logit" in zero_decay

    before = {
        name: named_parameters[name].detach().clone()
        for name in grouped
    }
    for parameter in optimized:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    zero_decay_coordinate_changed = any(
        not torch.equal(named_parameters[name].detach(), before[name])
        for name in zero_decay
    )
    assert not zero_decay_coordinate_changed
    decayed_matrix_moved = any(
        not torch.equal(named_parameters[name].detach(), before[name])
        for name in grouped
        if decay_by_name[name] > 0.0
    )
    assert decayed_matrix_moved
    return {
        "frozen_foundation_raw_coordinates": len(frozen_raw),
        "frozen_foundation_field_parameters": len(frozen_foundation_field),
        "optimized_parameters": len(grouped),
        "frozen_parameters_in_optimizer": foundation_field_in_optimizer,
        "trainable_inverse_link_or_gate_parameters_decayed": sensitive_coordinate_decayed,
        "zero_gradient_decay_changed_sensitive_coordinate": zero_decay_coordinate_changed,
        "zero_gradient_decay_changed_ordinary_matrix": decayed_matrix_moved,
    }


def _intervention(
    regulator: int,
    batch: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    value = torch.zeros(batch, 12, device=device)
    value[:, int(regulator)] = 1.0
    return value


def near_persistence_and_gradient_smoke() -> Dict[str, float]:
    model, _ = _fixture()
    control = torch.full((8, 12), 0.35)
    cues = torch.zeros(8, 1)
    intervention = torch.cat(
        (_intervention(0, 4), _intervention(1, 4)), dim=0
    )
    output = model(control, intervention, cues=cues, steps=2)
    movement = (output["atac_t"] - control).abs().mean()
    gates = model.field.effective_branch_gates()
    assert 0.0 < float(movement.detach()) < 1e-2
    assert torch.allclose(gates["tf"], gates["tf"].new_tensor(1e-3), atol=1e-7)
    assert torch.allclose(
        gates["complex"], gates["complex"].new_tensor(1e-3), atol=1e-7
    )

    model.zero_grad(set_to_none=True)
    (output["atac_t"] - control).square().mean().backward()
    tf_gradient = model.field.tf_gate_logit.grad
    complex_gradient = model.field.complex_gate_logit.grad
    assert tf_gradient is not None and float(tf_gradient.abs()) > 0.0
    assert complex_gradient is not None and float(complex_gradient.abs()) > 0.0
    return {
        "initial_mean_absolute_movement": float(movement.detach()),
        "initial_tf_gate": float(gates["tf"].detach()),
        "initial_complex_gate": float(gates["complex"].detach()),
        "tf_gate_gradient": float(tf_gradient.detach()),
        "complex_gate_gradient": float(complex_gradient.detach()),
    }


def gate_learning_smoke() -> Dict[str, float]:
    student, _ = _fixture(tf_efficacy=1e-3, complex_efficacy=1e-3)
    teacher, _ = _fixture(tf_efficacy=0.15, complex_efficacy=0.15)
    control = torch.full((12, 12), 0.35)
    cues = torch.zeros(12, 1)
    intervention = torch.cat(
        (_intervention(0, 6), _intervention(1, 6)), dim=0
    )
    teacher.eval()
    with torch.no_grad():
        observed = teacher(control, intervention, cues=cues, steps=2)["atac_t"]
    target_delta = observed - control
    denominator = target_delta.square().mean().clamp_min(1e-12)

    for parameter in student.parameters():
        parameter.requires_grad_(False)
    student.field.tf_gate_logit.requires_grad_(True)
    student.field.complex_gate_logit.requires_grad_(True)
    optimizer = torch.optim.Adam(
        (student.field.tf_gate_logit, student.field.complex_gate_logit), lr=0.05
    )

    def objective() -> torch.Tensor:
        prediction = student(control, intervention, cues=cues, steps=2)["atac_t"]
        return ((prediction - control - target_delta).square().mean() / denominator)

    initial = float(objective().detach())
    for _ in range(120):
        loss = objective()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = float(objective().detach())
    gates = student.field.effective_branch_gates()
    tf_gate = float(gates["tf"].detach())
    complex_gate = float(gates["complex"].detach())
    assert tf_gate > 5e-3 and complex_gate > 5e-3
    assert final < 0.25 * initial
    return {
        "initial_normalized_response_error": initial,
        "final_normalized_response_error": final,
        "learned_tf_gate": tf_gate,
        "learned_complex_gate": complex_gate,
    }


def persistence_and_regularization_smoke() -> Dict[str, float]:
    model, _ = _fixture()
    control = torch.linspace(0.05, 0.95, 12).repeat(5, 1)
    cues = torch.zeros(5, 1)
    zero = model(control, torch.zeros(5, 12), cues=cues, steps=3)
    active = model(control, _intervention(2, 5), cues=cues, steps=3)
    frozen = model(
        control,
        _intervention(2, 5),
        cues=cues,
        steps=3,
        overrides=BranchOverrides(tf_scale=0.0, complex_scale=0.0),
    )
    unsupported = model(control, _intervention(3, 5), cues=cues, steps=3)
    assert torch.equal(zero["atac_t"], control)
    assert torch.equal(frozen["atac_t"], control)
    assert torch.equal(unsupported["atac_t"], control)

    active_penalty = model.realized_regularization(active, control)
    frozen_penalty = model.realized_regularization(frozen, control)
    assert float(active_penalty.detach()) > 0.0
    assert float(frozen_penalty.detach()) == 0.0
    model.zero_grad(set_to_none=True)
    active_penalty.backward()
    assert model.field.tf_gate_logit.grad is not None
    assert model.field.complex_gate_logit.grad is not None
    assert bool(torch.isfinite(model.field.tf_gate_logit.grad))
    assert bool(torch.isfinite(model.field.complex_gate_logit.grad))

    config = TwinTrainingConfig()
    _optimizer, _parameters, optimizer_audit = _make_optimizer(model, config)
    zero_decay = set(optimizer_audit["zero_weight_decay_parameters"])
    raw_or_gate = {
        name
        for name, _parameter in model.named_parameters()
        if ".raw_" in name.lower() or "gate" in name.lower()
    }
    assert raw_or_gate <= zero_decay
    assert optimizer_audit["raw_parameter_l2"] is False
    return {
        "active_realized_regularization": float(active_penalty.detach()),
        "frozen_realized_regularization": float(frozen_penalty.detach()),
        "inverse_link_or_gate_parameters_without_weight_decay": len(raw_or_gate),
    }


def route_normalization_smoke() -> Dict[str, object]:
    model, _ = _fixture()
    diagnostics = model.field.evidence_mass_diagnostics()
    results: Dict[str, object] = {}
    for branch in ("tf", "complex"):
        before = diagnostics[f"{branch}_pre_normalization"]
        after = diagnostics[f"{branch}_normalized"]
        supported = before > 0
        assert int(torch.count_nonzero(supported)) >= 3
        assert torch.allclose(after[supported], torch.ones_like(after[supported]), atol=1e-6)
        assert torch.equal(after[~supported], torch.zeros_like(after[~supported]))
        assert float(before[supported].max() - before[supported].min()) > 1e-3
        results[f"{branch}_supported_regulators"] = int(torch.count_nonzero(supported))
        results[f"{branch}_normalized_mass_min"] = float(after[supported].min())
        results[f"{branch}_normalized_mass_max"] = float(after[supported].max())
    return results


def device_smoke() -> Dict[str, object]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", torch.cuda.current_device()))
    for device in devices:
        model, _ = _fixture(device)
        control = torch.full((4, 12), 0.35, device=device)
        output = model(
            control,
            _intervention(0, 4, device),
            cues=torch.zeros(4, 1, device=device),
            steps=2,
        )
        wrong_parameters = {
            name: str(parameter.device)
            for name, parameter in model.named_parameters()
            if parameter.device != device
        }
        wrong_buffers = {
            name: str(buffer.device)
            for name, buffer in model.named_buffers()
            if buffer.device != device
        }
        assert not wrong_parameters, (str(device), wrong_parameters)
        assert not wrong_buffers, (str(device), wrong_buffers)
        assert output["atac_t"].device == device
        model.zero_grad(set_to_none=True)
        (output["atac_t"] - control).square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(gradient.device == device for gradient in gradients)
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    return {
        "tested_devices": [str(device) for device in devices],
        "cuda_forward_backward_tested": torch.cuda.is_available(),
    }


def response_selection_smoke() -> Dict[str, float]:
    config = TwinTrainingConfig()
    config.validate()
    generator = torch.Generator().manual_seed(17)
    control = 0.25 + 0.02 * torch.randn(24, 12, generator=generator)
    response = torch.zeros(12)
    response[0], response[1], response[5] = 0.10, -0.06, 0.04
    observed = (control + response).clamp(0.0, 1.0)
    persistence = control.clone()
    exact = observed.clone()
    # Correct aggregate response but intentionally wrong population spread.
    mean_only = observed.mean(0, keepdim=True).repeat(observed.shape[0], 1)
    projection = torch.nn.functional.normalize(
        torch.randn(12, 8, generator=generator), dim=0
    )

    def score(prediction: torch.Tensor):
        return response_focused_loss(
            prediction,
            observed,
            control,
            projection,
            observed.mean(0),
            control.mean(0),
            config,
        )

    exact_loss, exact_metrics = score(exact)
    persistence_loss, persistence_metrics = score(persistence)
    mean_only_loss, mean_only_metrics = score(mean_only)
    assert float(exact_loss) < float(mean_only_loss) < float(persistence_loss)
    assert abs(float(persistence_loss) - 1.0) < 1e-5
    assert exact_metrics["response_nrmse"] < 1e-7
    assert mean_only_metrics["response_nrmse"] < 1e-5
    assert mean_only_metrics["relative_swd"] > exact_metrics["relative_swd"]
    assert (
        config.response_nrmse_weight + config.response_cosine_weight
        > config.full_state_relative_swd_weight
        > 0.0
    )
    assert persistence_metrics["persistence_swd"] > 0.0
    return {
        "exact_response_score": float(exact_loss),
        "mean_only_response_score": float(mean_only_loss),
        "persistence_score": float(persistence_loss),
        "full_state_relative_swd_weight": config.full_state_relative_swd_weight,
    }


def training_only_perturbed_mean_smoke() -> Dict[str, object]:
    """Prove the generic-response baseline is compiled from train rows only."""

    train_ntc = np.zeros((4, 6), dtype=np.float32)
    train_a = train_ntc.copy()
    train_a[:, (0, 2)] = 1.0
    train_b = train_ntc.copy()
    train_b[:, (1, 2)] = 1.0
    validation_ntc = np.zeros((4, 6), dtype=np.float32)
    validation_c = validation_ntc.copy()
    validation_c[:, (0, 1, 2)] = 1.0
    matrix = sparse.csr_matrix(
        np.vstack((train_ntc, train_a, train_b, validation_ntc, validation_c))
    )
    row_groups = {
        ("train", "screen_1", "NTC"): np.arange(0, 4),
        ("train", "screen_1", "A"): np.arange(4, 8),
        ("train", "screen_1", "B"): np.arange(8, 12),
        ("validation", "screen_1", "NTC"): np.arange(12, 16),
        ("validation", "screen_1", "C"): np.arange(16, 20),
    }
    bundle = SparseFullChromatinBundle(
        accessibility=matrix,
        bins=tuple(f"bin_{index}" for index in range(6)),
        foundation_anchor_indices=np.asarray([0, 1], dtype=np.int64),
        targets=("NTC",) * 4
        + ("A",) * 4
        + ("B",) * 4
        + ("NTC",) * 4
        + ("C",) * 4,
        screens=("screen_1",) * 20,
        splits=("train",) * 12 + ("validation",) * 8,
        source_rows=np.arange(20),
        row_groups=row_groups,
        provenance={
            "v53_manifest_sha256": "fixture-manifest",
            "whole_target_split_sha256": "fixture-split",
            "whole_target_roster_sha256": "fixture-roster",
            "v53_matrix_sha256": "fixture-matrix",
            "v53_cells_sha256": "fixture-cells",
            "v53_bins_sha256": "fixture-bins",
        },
    )
    with tempfile.TemporaryDirectory(prefix="wld-v56-mean-") as directory:
        responses, manifest = compile_training_perturbed_mean_baseline(
            bundle, ("A", "B"), Path(directory)
        )
        expected = np.asarray([0.5, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert np.array_equal(responses["screen_1"], expected)
        assert manifest["construction_split"] == "train"
        assert manifest["validation_values_used"] is False
        assert manifest["test_values_materialized"] is False
        evaluation = evaluate_perturbed_mean_baseline(
            bundle,
            ("C",),
            responses,
            TwinTrainingConfig(validation_cells_per_target=4, projections=4),
            seed=81,
            device=torch.device("cpu"),
        )
    assert evaluation["validation_values_used_for_construction"] is False
    assert evaluation["test_values_materialized"] is False
    return {
        "training_targets": 2,
        "validation_targets": 1,
        "validation_values_used_for_construction": False,
        "test_values_materialized": False,
        "validation_response_nrmse": evaluation["all_targets"]["response_nrmse"],
    }


def _control_priors() -> ChromatinTwinPriors:
    regulators, tfs, complexes, modules, peaks = 12, 4, 4, 4, 16
    regulator_tf = torch.zeros(regulators, tfs)
    regulator_complex = torch.zeros(regulators, complexes)
    for index in range(regulators):
        regulator_tf[index, index % tfs] = 1.0 + 0.1 * (index % 4)
        regulator_complex[index, (index + 1) % complexes] = 0.7 + 0.1 * (index % 4)
    motif = torch.zeros(tfs, peaks)
    module_peak = torch.zeros(modules, peaks)
    for index in range(4):
        motif[index, 4 * index : 4 * index + index + 1] = torch.linspace(
            1.0, 0.7, index + 1
        )
        module_peak[index, 4 * index : 4 * index + index + 1] = torch.linspace(
            0.6, 1.0, index + 1
        )
    complex_module = torch.diag(torch.tensor([1.0, -0.8, 0.6, -1.0]))
    return ChromatinTwinPriors(
        regulator_tf_support=regulator_tf,
        tf_peak_motif=motif,
        regulator_complex_support=regulator_complex,
        complex_module_effect=complex_module,
        module_peak_loading=module_peak,
        foundation_peak_index=torch.tensor([0, 4, 8, 12]),
    )


def topology_control_smoke() -> Dict[str, object]:
    priors = _control_priors()
    strata = ("train",) * 4 + ("validation",) * 4 + ("test",) * 4
    controls, audit = build_matched_control_priors(
        priors, 10, 56_042, strata=strata
    )
    repeated_controls, repeated_audit = build_matched_control_priors(
        priors, 10, 56_042, strata=strata
    )
    assert len(controls) == 10
    assert [row["topology_sha256"] for row in audit["controls"]] == [
        row["topology_sha256"] for row in repeated_audit["controls"]
    ]
    assert audit["matching_contract"]["test_outcomes_or_observations_read"] is False
    for record in audit["controls"]:
        assert record["fixed_regulator_labels"] == 0
        assert record["split_boundaries_crossed"] is False
        permutation = record["source_row_for_control_regulator"]
        assert all(strata[index] == strata[source] for index, source in enumerate(permutation))
        assert set(record["strata"]) == {"train", "validation", "test"}
    locks = validate_control_priors(priors, controls, minimum=10)
    validated = _validate_control_generation_audit(audit, locks)
    assert validated["replicates"] == 10
    assert len({record["topology_sha256"] for record in audit["controls"]}) == 10
    try:
        build_matched_control_priors(priors, 9, 1, strata=strata)
    except ValueError:
        pass
    else:
        raise AssertionError("fewer than ten topology controls were accepted")
    # Keep a strong reference until after digest checks to ensure the repeated
    # construction returned complete prior objects, not only audit metadata.
    assert len(repeated_controls) == len(controls)
    return {
        "controls": len(controls),
        "strata": audit["strata"],
        "deterministic": True,
        "test_outcomes_read": False,
    }


def leakage_and_sealed_boundary_smoke() -> Dict[str, object]:
    model, _ = _fixture()
    control = torch.full((3, 12), 0.35)
    cues = torch.zeros(3, 1)
    encoded = model.encode_control(control, cues=cues)
    first = model(control, _intervention(0, 3), cues=cues, steps=2)
    second = model(control, _intervention(1, 3), cues=cues, steps=2)
    assert torch.equal(encoded["context"], first["context"])
    assert torch.equal(first["context"], second["context"])
    contract = architecture_contract(model)
    assert contract["neural_response_bypass"] is False

    builder_source = inspect.getsource(run_nullaware_development)
    runner_source = inspect.getsource(run_twin_development)
    assert "whole_target_split.json" in builder_source
    assert '"test_outcomes_or_observations_read": False' in builder_source
    assert 'str(split).lower() == "test"' in runner_source
    assert '"test_targets_materialized": False' in runner_source
    assert '"test_targets_evaluated": False' in runner_source
    assert '"digital_twin_claim": False' in runner_source
    assert '"attractor_claim": False' in runner_source
    return {
        "intervention_enters_after_encoder": True,
        "test_target_names_used_only_for_split_stratification": True,
        "test_values_materialized": False,
        "test_values_evaluated": False,
        "untouched_audit_inference": False,
        "digital_twin_claim": False,
        "attractor_claim": False,
    }


def main() -> None:
    report = {
        "schema_version": "wld-v5.6-null-aware-synthetic-contract",
        "near_persistence": near_persistence_and_gradient_smoke(),
        "gate_learning": gate_learning_smoke(),
        "persistence_and_regularization": persistence_and_regularization_smoke(),
        "real_foundation_optimizer_partition": real_foundation_optimizer_partition_smoke(),
        "route_normalization": route_normalization_smoke(),
        "device_contract": device_smoke(),
        "response_selection": response_selection_smoke(),
        "training_only_perturbed_mean": training_only_perturbed_mean_smoke(),
        "topology_controls": topology_control_smoke(),
        "leakage_and_sealed_boundary": leakage_and_sealed_boundary_smoke(),
    }
    output = Path("wld_v56_synthetic_validation.json")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print("PASS: near-persistence initialization with live route-gate gradients")
    print("PASS: supported TF and complex gates learn away from the null")
    print("PASS: realized-effect regularization and zero-decay inverse coordinates")
    print("PASS: real foundation raw coordinates remain frozen outside the optimizer")
    print("PASS: exact zero/frozen/unsupported persistence")
    print("PASS: separately normalized TF and complex evidence mass")
    print("PASS: CPU/CUDA parameter, buffer, forward, backward placement")
    print("PASS: response-focused selection retains full-state/persistence checks")
    print("PASS: screen-matched perturbed mean uses training targets only")
    print("PASS: ten split-stratified footprint/sign/mass-matched controls")
    print("PASS: sealed test, no-bypass, endpoint, and claim boundaries")
    print(f"PASS: wrote {output}")


if __name__ == "__main__":
    main()
