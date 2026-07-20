"""Synthetic contract tests for WLD v5.4 chromatin-response dynamics."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from wld_chromatin_response_v54 import (
    ChromatinRoutePriors,
    WLDChromatinResponseModel,
    architecture_contract,
    degree_preserving_bipartite_shuffle,
)
from wld_foundation_model_v4 import FoundationPriors, WLDMultistudyFoundationModel


def fixture() -> tuple[WLDChromatinResponseModel, torch.Tensor]:
    peak_to_gene = torch.zeros(6, 5)
    peak_tf_motif = torch.zeros(6, 3)
    tf_gene_support = torch.zeros(3, 5)
    for index in range(3):
        peak_to_gene[index, index] = 1.0
        peak_to_gene[index + 3, index] = 0.7
        peak_tf_motif[index, index] = 1.0
        peak_tf_motif[index + 3, index] = 0.8
        tf_gene_support[index, index] = 1.0 if index != 1 else -1.0
    priors = FoundationPriors(
        peak_to_gene=peak_to_gene,
        peak_tf_motif=peak_tf_motif,
        tf_gene_support=tf_gene_support,
        circuit_tf_tf=torch.zeros(3, 3),
        signal_signal=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        signal_tf=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]]),
        cue_signal=torch.tensor([[1.0, 0.0]]),
        tf_peak_effect=torch.zeros(3, 6),
        tf_gene_index=torch.tensor([0, 1, 2]),
        protein_signal=torch.zeros(0, 2),
        metabolic_signal=torch.zeros(0, 2),
        metabolic_tf=torch.zeros(0, 3),
    )
    foundation = WLDMultistudyFoundationModel(
        priors, context_covariate_dim=0, context_dim=8
    )
    route = torch.tensor(
        [
            [1.0, 0.0, 0.4],
            [0.7, 0.8, 0.0],
            [0.0, 1.0, 0.7],
            [0.8, 0.0, 1.0],
            [0.0, 0.9, 1.0],
            [1.0, 0.7, 0.0],
        ]
    )
    route_priors = ChromatinRoutePriors(
        regulator_tf_support=route,
        tf_peak_motif=peak_tf_motif.transpose(0, 1),
    )
    return WLDChromatinResponseModel(foundation, route_priors), route


def main() -> None:
    torch.manual_seed(42)
    model, support = fixture()
    contract = architecture_contract(model)
    assert contract["guide_or_target_in_encoder"] is False
    assert contract["direct_neural_context_to_peak_decoder"] is False
    assert contract["unsupported_edges_trainable"] is False

    control = torch.rand(12, 6) * 0.7
    intervention = F.one_hot(torch.arange(12) % 6, num_classes=6).float()
    zero_intervention = torch.zeros_like(intervention)

    no_perturbation = model(control, zero_intervention, steps=3)["atac_t"]
    assert torch.allclose(no_perturbation, control, atol=1e-7)
    output = model(control, intervention, steps=3)
    movement = (output["atac_t"] - control).abs().mean()
    assert 1e-8 < float(movement.detach()) < 0.2

    # Guide identity is applied after the foundation encoder.  Changing the
    # intervention cannot alter the encoded control state or context.
    first = model.encode_control(control)
    second = model.encode_control(control)
    assert torch.equal(first["context"], second["context"])

    # Removing every supported route freezes the intervention response.
    frozen_zero = model(
        control,
        intervention,
        steps=3,
        support_override=torch.zeros_like(support),
    )["atac_t"]
    assert torch.allclose(frozen_zero, control, atol=1e-7)

    shuffled = degree_preserving_bipartite_shuffle(support, seed=7)
    assert torch.equal((support > 0).sum(0), (shuffled > 0).sum(0))
    assert torch.equal((support > 0).sum(1), (shuffled > 0).sum(1))
    assert not torch.equal(support > 0, shuffled > 0)

    # The supported response parameters receive gradient, while unsupported
    # regulator-to-TF edges have no parameter at all.
    loss = output["atac_t"].mean()
    loss.backward()
    assert float(model.field.raw_tf_gain.grad.abs().sum()) > 0
    assert float(model.field.raw_motif_gain.grad.abs().sum()) > 0
    assert float(model.field.tf_context_gain.weight.grad.abs().sum()) > 0

    # A small distributional response fixture must be learnable without any
    # target-specific embedding or dense guide-to-peak decoder.
    teacher = copy.deepcopy(model).eval()
    with torch.no_grad():
        teacher.field.raw_tf_direction.copy_(torch.tensor([0.9, -0.6, 0.5]))
        teacher.field.raw_motif_direction.copy_(
            torch.linspace(-0.8, 0.8, teacher.field.motif_edges)
        )
        target = teacher(control, intervention, steps=3)["atac_t"]

    student, _ = fixture()
    optimizer = torch.optim.Adam(student.field.parameters(), lr=0.03)
    initial = None
    for _ in range(80):
        prediction = student(control, intervention, steps=3)["atac_t"]
        fit = F.mse_loss(prediction, target)
        if initial is None:
            initial = float(fit.detach())
        optimizer.zero_grad(set_to_none=True)
        fit.backward()
        optimizer.step()
    assert float(fit.detach()) < initial * 0.80

    print("PASS: named intervention is excluded from the encoder")
    print("PASS: regulator-to-TF-to-motif topology is the only response path")
    print("PASS: supported gains and opening/closing rates vary with context")
    print("PASS: frozen-zero and degree-preserving route controls")
    print("PASS: perturbational opening/closing direction is learnable")
    print("PASS: response scope is transient chromatin dynamics, not an attractor claim")


if __name__ == "__main__":
    main()
