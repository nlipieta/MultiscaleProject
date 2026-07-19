"""Fast synthetic contract test for the WLD v4 multi-study foundation track.

The fixture tests software and scientific guardrails.  It is intentionally not
a biological benchmark and never opens the synthetic held-out test partition.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from wld_foundation_model_v4 import (
    FoundationPriors,
    WLDMultistudyFoundationModel,
    architecture_contract,
    audit_foundation_inputs,
)
from wld_multistudy_pretraining import (
    FoundationBatch,
    MultistudyPretrainer,
    ObservationGroup,
    PretrainingConfig,
    StudySpec,
    make_scientific_controls,
    make_sealed_split,
    validate_catalog,
    verify_donor_separation,
)


ROOT = Path(__file__).resolve().parent


def priors_fixture() -> FoundationPriors:
    peak_to_gene = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    peak_tf_motif = torch.tensor(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=torch.float32,
    )
    tf_gene = torch.tensor(
        [
            [1, 1, 0, 1, 0],
            [0, -1, -1, 1, 0],
            [0, 0, 1, 0, 1],
        ],
        dtype=torch.float32,
    )
    circuit = torch.tensor(
        [[0, 1, 0], [0, 0, -1], [0, 0, 0]], dtype=torch.float32
    )
    result = FoundationPriors(
        peak_to_gene=peak_to_gene,
        peak_tf_motif=peak_tf_motif,
        tf_gene_support=tf_gene,
        circuit_tf_tf=circuit,
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


def study_fixture():
    studies = [
        StudySpec("study_a", "human", "GRCh38", "muscle", ("atac", "rna", "cue"), ("a1", "a2"), longitudinal=True),
        StudySpec("study_b", "human", "GRCh38", "blood", ("atac", "rna", "protein", "cue"), ("b1",), perturbation=True),
        StudySpec("study_c", "human", "GRCh38", "brain", ("atac", "rna", "metabolic", "cue"), ("c1",), longitudinal=True),
        StudySpec("study_d", "human", "GRCh38", "skin", ("atac", "rna", "cue"), ("d1",), longitudinal=True),
    ]
    groups = [
        ObservationGroup("a1_t", "study_a", "a1", 0, 1, "stimulus"),
        ObservationGroup("a2_t", "study_a", "a2", 0, 1, "stimulus"),
        ObservationGroup("b1_t", "study_b", "b1", 0, 2, "perturbation"),
        ObservationGroup("c1_t", "study_c", "c1", 0, 1, "stimulus"),
        ObservationGroup("d1_t", "study_d", "d1", 0, 1, "stimulus"),
    ]
    return studies, groups


def batch_fixture(group: ObservationGroup, seed: int, missing: str = "") -> FoundationBatch:
    generator = torch.Generator().manual_seed(seed)
    cells = 12
    atac = torch.rand((cells, 6), generator=generator)
    cues = torch.rand((cells, 2), generator=generator)
    covariates = torch.rand((cells, 3), generator=generator) * 2 - 1
    protein = torch.rand((cells, 2), generator=generator) if missing != "protein" else None
    metabolic = torch.rand((cells, 2), generator=generator) if missing != "metabolic" else None
    # Population target, deliberately permuted: there is no cell pairing.
    rna = torch.relu(
        0.4
        + 0.6 * atac[:, :5]
        + 0.25 * cues[:, :1]
        + 0.15 * covariates[:, :1]
    )
    order = torch.randperm(cells, generator=generator)
    target_rna = (1.05 * rna[order] + 0.05).clamp_min(0)
    target_atac = (0.9 * atac[order] + 0.05).clamp(0, 1)
    return FoundationBatch(
        group_id=group.group_id,
        study_id=group.study_id,
        donor_id=group.donor_id,
        cues=cues,
        horizon=group.target_time - group.source_time,
        source_atac=atac,
        source_protein=protein,
        source_metabolic=metabolic,
        context_covariates=covariates,
        target_rna=target_rna,
        target_atac=target_atac,
    )


def main() -> None:
    torch.manual_seed(7)
    priors = priors_fixture()
    contract_model = WLDMultistudyFoundationModel(
        priors, context_covariate_dim=3, context_dim=8
    )
    contract = architecture_contract(contract_model)
    assert contract["hard_sparse_topology"]
    assert not contract["direct_context_to_rna_decoder"]
    print("PASS: structured encoder and hard-topology context-conditioned ODE")

    studies, groups = study_fixture()
    catalog = validate_catalog(
        studies,
        ["ATAC_peaks", "protein_abundance", "metabolite_abundance", "external_cue"],
    )
    try:
        audit_foundation_inputs(["ATAC_peaks", "study_id", "cell_type"])
    except ValueError:
        pass
    else:
        raise AssertionError("Identity and batch proxies were accepted")
    split = make_sealed_split(
        groups, validation_studies=["study_c"], test_studies=["study_d"]
    )
    donors = verify_donor_separation(groups, split)
    assert split.test_groups == ("d1_t",)
    print("PASS: studies/donors split before selection and IDs excluded from inputs")

    by_id = {value.group_id: value for value in groups}
    batches = {
        value.group_id: batch_fixture(
            value,
            100 + index,
            missing="protein" if index % 2 else "metabolic",
        )
        for index, value in enumerate(groups)
    }
    with torch.no_grad():
        missing_output = contract_model(
            **batches["a1_t"].model_inputs(steps=2, use_source_rna=False)
        )
    assert torch.isfinite(missing_output["rna_t"]).all()
    assert set(missing_output["modality_mask"][:, 1:].unique().tolist()).issubset({0.0, 1.0})
    print("PASS: absent protein/metabolic measurements remain explicitly masked")

    # Demonstrate that kinetics are not frozen: cell-specific covariates produce
    # gradients for rate and circuit-gain adapters.  The zero initialization is
    # a shared-population prior, not a frozen constraint.
    probe = batches["a1_t"]
    output = contract_model(**probe.model_inputs(steps=2, use_source_rna=False))
    loss = output["rna_t"].mean() + output["tf_t"].mean()
    loss.backward()
    rate_grad = contract_model.field.rna_decay.adapter.weight.grad
    gain_grad = contract_model.field.tf_circuit.context_gain.weight.grad
    assert rate_grad is not None and float(rate_grad.abs().sum()) > 0
    assert gain_grad is not None and float(gain_grad.abs().sum()) > 0
    with torch.no_grad():
        contract_model.field.rna_decay.adapter.weight[0, 0] = 0.1
        varied = contract_model(**probe.model_inputs(steps=2, use_source_rna=False))
    rate_variation = float(varied["rna_decay"].var(0, unbiased=False).mean())
    assert rate_variation > 0
    print("PASS: cell-context gradients reach circuit gains and kinetic parameters")

    train_batches = [batches[value] for value in split.train_groups]
    validation_batches = [batches[value] for value in split.validation_groups]
    student = WLDMultistudyFoundationModel(
        priors, context_covariate_dim=3, context_dim=8
    )
    trainer = MultistudyPretrainer(
        student,
        PretrainingConfig(
            epochs=8,
            steps=2,
            learning_rate=3e-3,
            adaptation_penalty=1e-4,
            use_source_rna=False,
            seed=7,
        ),
    )
    development = trainer.fit(train_batches, validation_batches)
    assert development["test_evaluated"] is False
    assert len(development["history"]) == 8
    print("PASS: validation-selected multi-study training; test study remains sealed")

    controls = make_scientific_controls(priors, seed=13)
    assert set(controls) == {
        "supported_circuit", "no_tf_circuit", "supported_sign_shuffle"
    }
    assert int((controls["no_tf_circuit"].circuit_tf_tf != 0).sum()) == 0
    assert torch.equal(
        controls["supported_sign_shuffle"].circuit_tf_tf != 0,
        priors.circuit_tf_tf != 0,
    )
    print("PASS: supported, no-circuit and sign-shuffled controls are mandatory")

    report = {
        "scope": "synthetic software and scientific-contract validation only",
        "biological_pretraining_complete": False,
        "held_out_test_evaluated": False,
        "architecture": contract,
        "catalog_contract": catalog,
        "donor_partitions": donors,
        "context_specific_parameters": {
            "edge_gains": True,
            "production_rates": True,
            "decay_rates": True,
            "chromatin_timescales": True,
            "observed_rate_variation": rate_variation,
        },
        "training": development,
        "required_controls": list(controls),
    }
    path = ROOT / "wld_v4_foundation_validation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: wrote {path.name}")
    print("NOTE: no biological performance or attractor claim was evaluated")


if __name__ == "__main__":
    main()
