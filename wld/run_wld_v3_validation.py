"""Neutral structural and numerical checks for WLD v3 circuit dynamics.

This is not a biological validation and intentionally is not a toggle-switch
benchmark. It checks that the implementation obeys its mechanistic contract
before it is trained on real temporal or perturbation-resolved observations.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from wld_circuit_dynamics_v3 import (
    CircuitDynamicsModel,
    CircuitIntervention,
    MultiscaleCircuitPriors,
    architecture_contract,
    degree_preserving_signed_permutation,
    temporal_leakage_audit,
    temporal_circuit_objective,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "wld_v3_validation.json"


def example_priors() -> MultiscaleCircuitPriors:
    """Small non-toggle network spanning every supported biological layer."""
    return MultiscaleCircuitPriors(
        peak_to_gene=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.6, 0.4, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        peak_tf_motif=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        tf_gene_support=torch.tensor(
            [
                [1.0, -0.8, 0.0],
                [-0.7, 0.9, 1.0],
            ]
        ),
        circuit_tf_tf=torch.tensor(
            [
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ),
        tf_gene_index=torch.tensor([0, 1], dtype=torch.long),
        signal_signal=torch.tensor(
            [
                [0.0, 0.7],
                [-0.6, 0.0],
            ]
        ),
        signal_tf=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, -1.0],
            ]
        ),
        cue_signal=torch.tensor([[1.0, 0.0]]),
        tf_peak_effect=torch.tensor(
            [
                [0.0, 0.0, 0.8, 0.0],
                [0.0, -0.7, 0.0, 0.6],
            ]
        ),
    )


def stable_feedforward_priors() -> MultiscaleCircuitPriors:
    """Single-equilibrium fixture with no feedback or multistability premise."""
    return MultiscaleCircuitPriors(
        peak_to_gene=torch.tensor([[1.0], [0.5]]),
        peak_tf_motif=torch.tensor([[1.0], [1.0]]),
        tf_gene_support=torch.tensor([[1.0]]),
        circuit_tf_tf=torch.zeros(1, 1),
        tf_gene_index=torch.tensor([0], dtype=torch.long),
        signal_signal=torch.zeros(1, 1),
        signal_tf=torch.tensor([[1.0]]),
        cue_signal=torch.tensor([[1.0]]),
        tf_peak_effect=torch.zeros(1, 2),
    )


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains a non-finite value.")


def validate_sparse_topology(
    model: CircuitDynamicsModel, priors: MultiscaleCircuitPriors
) -> None:
    pairs = (
        ("cue_to_signal", model.field.cue_to_signal, priors.cue_signal),
        ("signal_recurrent", model.field.signal_recurrent, priors.signal_signal),
        ("signal_to_tf", model.field.signal_to_tf, priors.signal_tf),
        ("tf_circuit", model.field.tf_circuit, priors.circuit_tf_tf),
        ("tf_to_peak", model.field.tf_to_peak, priors.tf_peak_effect),
        ("tf_to_gene", model.field.tf_to_gene, priors.tf_gene_support),
    )
    for name, layer, adjacency in pairs:
        expected_edges = int(torch.count_nonzero(adjacency))
        if layer.num_edges != expected_edges:
            raise AssertionError(
                f"{name} has {layer.num_edges} parameters for "
                f"{expected_edges} prior edges."
            )
        if layer.num_edges:
            observed_sign = layer.edge_sign.cpu()
            expected_sign = torch.sign(
                adjacency[layer.source_index, layer.target_index]
            ).cpu()
            if not torch.equal(observed_sign, expected_sign):
                raise AssertionError(f"{name} did not preserve supplied edge signs.")
            if not bool((layer.effective_gain() > 0).all()):
                raise AssertionError(f"{name} has a non-positive edge gain.")
            if hasattr(layer, "effective_threshold"):
                if not bool((layer.effective_threshold() > 0).all()):
                    raise AssertionError(f"{name} has a non-positive threshold.")
                if not bool((layer.effective_hill() > 1).all()):
                    raise AssertionError(f"{name} has a Hill coefficient <= 1.")


def validate_signed_permutation() -> None:
    adjacency = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [-1.0, 0.0, 0.0, 1.0],
        ]
    )
    permuted = degree_preserving_signed_permutation(adjacency, seed=13)
    if torch.equal(adjacency, permuted):
        raise AssertionError("Signed permutation left the scaffold unchanged.")
    for sign in (1, -1):
        before = (torch.sign(adjacency) == sign).to(torch.int64)
        after = (torch.sign(permuted) == sign).to(torch.int64)
        if not torch.equal(before.sum(0), after.sum(0)):
            raise AssertionError("Signed target degree changed during permutation.")
        if not torch.equal(before.sum(1), after.sum(1)):
            raise AssertionError("Signed source degree changed during permutation.")


def validate_general_architecture() -> dict[str, object]:
    torch.manual_seed(42)
    priors = example_priors()
    priors.validate()
    inconsistent = replace(
        priors,
        circuit_tf_tf=torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )
    try:
        inconsistent.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("An inconsistent circuit/target-gene sign was accepted.")
    model = CircuitDynamicsModel(priors)
    contract = architecture_contract(model)
    if contract["neural_bypass_modules"]:
        raise AssertionError("An unrestricted neural bypass was detected.")
    validate_sparse_topology(model, priors)

    closed = torch.zeros(2, priors.num_peaks)
    open_accessibility = torch.ones(2, priors.num_peaks)
    closed_gate = model.field.accessibility_gate(closed)
    open_gate = model.field.accessibility_gate(open_accessibility)
    if not torch.equal(closed_gate, torch.zeros_like(closed_gate)):
        raise AssertionError("Closed chromatin did not eliminate regulatory gates.")
    supported = priors.tf_gene_support != 0
    unsupported = ~supported
    if not bool((open_gate[:, supported] > 0).all()):
        raise AssertionError("Open, localized supported edges did not receive a gate.")
    if not torch.equal(
        open_gate[:, unsupported], torch.zeros_like(open_gate[:, unsupported])
    ):
        raise AssertionError("An unsupported TF-gene edge received accessibility.")

    atac = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [0.8, 0.6, 0.9, 1.0],
            [0.3, 1.0, 0.7, 0.8],
        ]
    )
    cues = torch.tensor([[0.2], [0.8], [1.2]])
    output = model(atac, cues, horizon=0.75, steps=12)
    expected_path_shape = (3, 13, model.state_dim)
    if output["state_path"].shape != expected_path_shape:
        raise AssertionError(
            f"State path is {tuple(output['state_path'].shape)}, "
            f"expected {expected_path_shape}."
        )
    for name in (
        "state_path",
        "terminal_state",
        "terminal_velocity",
        "rna_t",
        "accessibility_t",
    ):
        assert_finite(name, output[name])

    target_rna = (output["rna_t"].detach() * 1.1 + 0.02).clamp_min(0.0)
    target_accessibility = output["accessibility_t"].detach().clamp(0.0, 1.0)
    observed_derivative = torch.zeros_like(output["terminal_velocity"])
    losses = temporal_circuit_objective(
        output,
        target_rna,
        target_accessibility=target_accessibility,
        observed_derivative=observed_derivative,
        terminal_mask=torch.tensor([False, True, True]),
        model=model,
    )
    losses["total"].backward()
    gradient = model.field.tf_circuit.raw_gain.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError("The validated circuit did not receive a finite gradient.")
    if float(gradient.abs().sum()) <= 0.0:
        raise AssertionError("The temporal objective did not train circuit-edge gains.")

    with torch.no_grad():
        reference = model(atac, cues, horizon=0.75, steps=12)
        inhibited = model(
            atac,
            cues,
            horizon=0.75,
            steps=12,
            intervention=CircuitIntervention(
                tf_activity_scale=torch.zeros(priors.num_tfs)
            ),
        )
    intervention_effect = float(
        torch.mean(torch.abs(reference["rna_t"] - inhibited["rna_t"]))
    )
    if intervention_effect <= 1e-7:
        raise AssertionError("A complete TF activity intervention had no RNA effect.")
    try:
        model(
            atac,
            cues,
            horizon=0.25,
            steps=2,
            intervention=CircuitIntervention(
                circuit_edge_scale=-torch.ones(model.field.tf_circuit.num_edges)
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A sign-flipping negative edge scale was accepted.")

    rates = model.field.positive_rates()
    if any(not bool((value > 0).all()) for value in rates.values()):
        raise AssertionError("A decay or chromatin timescale was non-positive.")
    validate_signed_permutation()

    return {
        "hard_sparse_topology": True,
        "fixed_edge_signs": True,
        "accessibility_gate_required": True,
        "circuit_gradient_nonzero": True,
        "intervention_effect": intervention_effect,
        "signed_degree_control": True,
        "neural_bypass_modules": contract["neural_bypass_modules"],
    }


def validate_leakage_contract() -> dict[str, object]:
    accepted = temporal_leakage_audit(
        train_groups=["donor_1", "donor_2"],
        test_groups=["donor_3"],
        initial_feature_names=["ATAC_peaks", "external_cue"],
        initial_time=0.0,
        target_time=24.0,
    )
    rejected = []
    for proxy in ("RNA_counts", "cell_type", "target_state", "pseudotime"):
        try:
            temporal_leakage_audit(
                train_groups=["donor_1"],
                test_groups=["donor_2"],
                initial_feature_names=["ATAC_peaks", proxy],
                initial_time=0.0,
                target_time=24.0,
            )
        except ValueError:
            rejected.append(proxy)
        else:
            raise AssertionError(f"The v3 leakage audit accepted {proxy!r}.")
    for invalid_kwargs in (
        {"train_groups": ["d1"], "test_groups": ["d1"]},
        {"initial_time": 24.0, "target_time": 0.0},
    ):
        kwargs = {
            "train_groups": ["d1"],
            "test_groups": ["d2"],
            "initial_feature_names": ["ATAC_peaks", "external_cue"],
            "initial_time": 0.0,
            "target_time": 24.0,
        }
        kwargs.update(invalid_kwargs)
        try:
            temporal_leakage_audit(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("The v3 leakage audit accepted an invalid design.")
    temporal_leakage_audit(
        train_groups=["d1"],
        test_groups=["d2"],
        initial_feature_names=["ATAC_peaks", "RNA_t0"],
        initial_time=0.0,
        target_time=24.0,
        uses_initial_rna=True,
        initial_rna_time=0.0,
    )
    return {
        "accepted_standard_inputs": accepted["initial_features"],
        "rejected_proxies": rejected,
        "group_separation_required": True,
        "forward_time_required": True,
        "time_zero_rna_requires_explicit_declaration": True,
    }


def validate_neutral_stability() -> dict[str, object]:
    """Test diagnostics on a stable feed-forward system, not a toggle model."""
    torch.manual_seed(7)
    model = CircuitDynamicsModel(stable_feedforward_priors())
    atac = torch.tensor([[0.7, 0.4]])
    cues = torch.tensor([[0.6]])
    initial = model.initial_state(atac, cues)[0]
    terminal, _ = model.integrate_state(
        initial.unsqueeze(0), cues, horizon=25.0, steps=250
    )
    fixed, residual = model.refine_fixed_point(
        terminal[0],
        cues[0],
        iterations=1000,
        learning_rate=0.03,
        tolerance=1e-6,
    )
    if residual > 5e-4:
        raise AssertionError(f"Fixed-point residual remained too large: {residual:.3g}")

    eigenvalues = model.jacobian_eigenvalues(fixed, cues[0])
    max_real_eigenvalue = float(eigenvalues.real.max())
    if max_real_eigenvalue >= -1e-5:
        raise AssertionError(
            "Neutral stability fixture was not stable: "
            f"max Re(lambda)={max_real_eigenvalue:.3g}"
        )

    basin = model.basin_return_fraction(
        fixed,
        cues[0],
        trials=24,
        perturbation_scale=0.02,
        horizon=25.0,
        steps=250,
        tolerance=0.08,
        seed=29,
    )
    return_fraction = float(basin["fraction_returned"])
    if return_fraction < 0.9:
        raise AssertionError(
            f"Only {return_fraction:.1%} of neutral perturbations returned."
        )
    return {
        "fixture": "stable feed-forward single equilibrium",
        "toggle_benchmark": False,
        "fixed_point_residual": residual,
        "max_real_jacobian_eigenvalue": max_real_eigenvalue,
        "basin_return_fraction": return_fraction,
    }


def main() -> None:
    print("Validating WLD v3 mechanistic architecture...", flush=True)
    architecture = validate_general_architecture()
    print(
        "PASS: hard prior topology, fixed signs, enhancer gates, and gradients",
        flush=True,
    )
    print(
        "PASS: interventions and signed degree-preserving negative control",
        flush=True,
    )
    leakage = validate_leakage_contract()
    print("PASS: grouped temporal leakage contract", flush=True)

    stability = validate_neutral_stability()
    print("PASS: fixed-point, Jacobian, and basin diagnostic plumbing", flush=True)
    print("      (neutral stable feed-forward fixture; no toggle benchmark)", flush=True)

    report = {
        "scope": (
            "structural and numerical validation only; biological attractor claims "
            "require grouped temporal or perturbation data"
        ),
        "architecture": architecture,
        "leakage_contract": leakage,
        "neutral_stability": stability,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: wrote {REPORT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
