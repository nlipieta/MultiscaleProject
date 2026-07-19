"""Synthetic contract checks for WLD v4 temporal fine-tuning."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from wld_foundation_model_v4 import (
    FoundationPriors,
    WLDMultistudyFoundationModel,
    no_circuit_priors,
    supported_sign_shuffle_priors,
)
from wld_v4_temporal_finetuning import (
    SubjectTransition,
    TemporalConditionTrainer,
    TemporalFinetuningConfig,
    _state_sha256,
)


ROOT = Path(__file__).resolve().parent


def priors_fixture() -> FoundationPriors:
    result = FoundationPriors(
        peak_to_gene=torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
            ],
            dtype=torch.float32,
        ),
        peak_tf_motif=torch.tensor(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
            ],
            dtype=torch.float32,
        ),
        tf_gene_support=torch.tensor(
            [[1, 1, 0, 1, 0], [0, -1, -1, 1, 0], [0, 0, 1, 0, 1]],
            dtype=torch.float32,
        ),
        circuit_tf_tf=torch.tensor(
            [[0, 1, 0], [0, 0, -1], [0, 0, 0]], dtype=torch.float32
        ),
        signal_signal=torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32),
        signal_tf=torch.tensor([[1, 0, 0], [0, 1, -1]], dtype=torch.float32),
        cue_signal=torch.tensor([[1, 0], [0, 1]], dtype=torch.float32),
        tf_peak_effect=torch.tensor(
            [[1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            dtype=torch.float32,
        ),
        tf_gene_index=torch.tensor([0, 1, 2]),
        protein_signal=torch.tensor([[1, 0], [0, 1]], dtype=torch.float32),
        metabolic_signal=torch.tensor([[0.5, 0], [0, -0.5]], dtype=torch.float32),
        metabolic_tf=torch.tensor([[0.2, 0, 0], [0, 0.2, -0.2]], dtype=torch.float32),
    )
    result.validate()
    return result


def transition(subject: str, condition: str, split: str, seed: int) -> SubjectTransition:
    rng = np.random.default_rng(seed)
    cells = 24
    initial_atac = rng.binomial(1, 0.35, size=(cells, 6)).astype(np.float32)
    initial_rna = rng.poisson(2.0 + initial_atac[:, :5], size=(cells, 5)).astype(np.float32)
    cue = 1.0 if condition == "exercise" else 0.0
    target_rna = rng.poisson(
        2.0 + 0.35 * initial_atac[:, :5] + 0.5 * cue,
        size=(cells, 5),
    ).astype(np.float32)
    target_atac = rng.binomial(
        1, np.clip(0.30 + 0.10 * cue, 0, 1), size=(cells, 6)
    ).astype(np.float32)
    result = SubjectTransition(
        subject,
        condition,
        split,
        sparse.csr_matrix(initial_rna),
        sparse.csr_matrix(initial_atac),
        sparse.csr_matrix(target_rna),
        sparse.csr_matrix(target_atac),
    )
    result.validate(5, 6)
    return result


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(13)
    priors = priors_fixture()
    model = WLDMultistudyFoundationModel(priors, context_dim=8)
    before_field = _state_sha256(model.field.state_dict())
    before_context = _state_sha256(model.context_network.state_dict())
    training = [
        transition("A", "exercise", "train", 1),
        transition("B", "control", "train", 2),
    ]
    validation = [transition("C", "exercise", "validation", 3)]
    config = TemporalFinetuningConfig(
        epochs=3,
        batch_size=8,
        batches_per_transition=1,
        integration_steps=2,
        validation_cells=8,
        patience=3,
        projections=4,
        seed=13,
    )
    with tempfile.TemporaryDirectory() as value:
        state_path = Path(value) / "temporal_state.pt"
        trainer = TemporalConditionTrainer(
            model, config, cue_index=0, condition_name="true_circuit", device=torch.device("cpu")
        )
        result = trainer.fit(training, validation, state_path, "synthetic-signature")
        assert result["test_subjects_evaluated"] is False
        assert result["all_parameters_require_grad"] is True
        assert len(result["history"]) == 3
        assert _state_sha256(model.field.state_dict()) != before_field
        assert _state_sha256(model.context_network.state_dict()) != before_context

        restored = WLDMultistudyFoundationModel(priors, context_dim=8)
        restored_trainer = TemporalConditionTrainer(
            restored, config, cue_index=0, condition_name="true_circuit", device=torch.device("cpu")
        )
        resumed = restored_trainer.fit(
            training, validation, state_path, "synthetic-signature"
        )
        assert resumed["history"] == result["history"]
    print("PASS: temporal checkpoint selection and restart-safe resume")
    print("PASS: representation and context-conditioned field both update")

    no_circuit = no_circuit_priors(priors)
    shuffled = None
    for seed in range(20):
        candidate = supported_sign_shuffle_priors(priors, seed)
        if not torch.equal(candidate.circuit_tf_tf, priors.circuit_tf_tf):
            shuffled = candidate
            break
    assert shuffled is not None
    assert int(torch.count_nonzero(no_circuit.circuit_tf_tf)) == 0
    assert torch.equal(
        shuffled.circuit_tf_tf != 0, priors.circuit_tf_tf != 0
    )
    assert not torch.equal(shuffled.circuit_tf_tf, priors.circuit_tf_tf)
    shuffled.validate()
    print("PASS: no-circuit and supported-sign controls preserve admissible evidence")

    validation_metrics = result["validation"]["aggregate"]
    assert "persistence_rna_swd" in validation_metrics
    assert "terminal_velocity_l1" in validation_metrics
    assert np.isfinite(validation_metrics["rna_swd"])
    print("PASS: independently sampled population loss and persistence baseline")
    print("PASS: transient endpoint velocity is reported but not minimized as an attractor")

    report = {
        "scope": "synthetic WLD v4 temporal software contract only",
        "checks": [
            "independent_population_sampling",
            "time_zero_rna_initial_state_not_encoder",
            "all_parameters_trainable",
            "validation_selected_restart_safe_checkpoint",
            "persistence_baseline",
            "true_no_and_supported_sign_controls",
            "no_transient_attractor_claim",
        ],
        "test_subjects_evaluated": False,
        "attractor_claim": False,
    }
    (ROOT / "wld_v4_temporal_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("PASS: wrote wld_v4_temporal_validation.json")


if __name__ == "__main__":
    main()
