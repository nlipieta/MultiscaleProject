"""End-to-end software test for grouped WLD v3 temporal training.

The generated cohort is neutral and synthetic.  It tests data validation,
distributional (unpaired) training, group-sealed selection, held-out evaluation,
checkpoint writing, and the no-circuit control.  It is not biological evidence.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from wld_circuit_dynamics_v3 import CircuitDynamicsModel, MultiscaleCircuitPriors
from wld_temporal_training import (
    TemporalTrainingConfig,
    load_temporal_cohort,
    run_temporal_benchmark,
    save_priors,
)


def smoke_priors() -> MultiscaleCircuitPriors:
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
                [0.0, -1.0],
                [0.0, 0.0],
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


def write_smoke_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    priors = smoke_priors()
    priors.validate()
    save_priors(root / "priors.npz", priors)

    generator = torch.Generator().manual_seed(123)
    ground_truth = CircuitDynamicsModel(priors)
    with torch.no_grad():
        ground_truth.field.tf_circuit.raw_gain.fill_(1.0)

    groups = ("donor_train_a", "donor_train_b", "donor_validation", "donor_test")
    transitions = []
    initial_atac = []
    initial_cues = []
    initial_transition = []
    target_rna = []
    target_atac = []
    target_transition = []

    for group_index, group in enumerate(groups):
        transition_id = f"{group}_cue"
        horizon = 0.45 + 0.05 * group_index
        transitions.append(
            {
                "transition_id": transition_id,
                "group_id": group,
                "horizon": horizon,
                "terminal": group == "donor_test",
            }
        )
        atac = 0.1 + 0.8 * torch.rand(24, priors.num_peaks, generator=generator)
        cues = torch.full((24, priors.num_cues), 0.25 + 0.15 * group_index)
        with torch.no_grad():
            prediction = ground_truth(
                atac,
                cues,
                horizon=horizon,
                steps=6,
            )
        permutation = torch.randperm(24, generator=generator)[:20]
        noisy_rna = (
            prediction["rna_t"][permutation]
            + 0.005 * torch.rand(20, priors.num_genes, generator=generator)
        ).clamp_min(0.0)
        noisy_atac = (
            prediction["accessibility_t"][permutation]
            + 0.003
            * torch.randn(20, priors.num_peaks, generator=generator)
        ).clamp(0.0, 1.0)

        initial_atac.append(atac.numpy())
        initial_cues.append(cues.numpy())
        initial_transition.extend([transition_id] * len(atac))
        target_rna.append(noisy_rna.numpy())
        target_atac.append(noisy_atac.numpy())
        target_transition.extend([transition_id] * len(noisy_rna))

    np.savez_compressed(
        root / "observations.npz",
        initial_atac=np.concatenate(initial_atac).astype(np.float32),
        initial_cues=np.concatenate(initial_cues).astype(np.float32),
        initial_transition=np.asarray(initial_transition, dtype="U64"),
        target_rna=np.concatenate(target_rna).astype(np.float32),
        target_atac=np.concatenate(target_atac).astype(np.float32),
        target_transition=np.asarray(target_transition, dtype="U64"),
    )
    manifest = {
        "schema_version": 1,
        "alignment_mode": "distribution",
        "initial_feature_names": ["ATAC_peaks", "external_cue"],
        "priors_fit_groups": ["donor_train_a", "donor_train_b"],
        "split_groups": {
            "train": ["donor_train_a", "donor_train_b"],
            "validation": ["donor_validation"],
            "test": ["donor_test"],
        },
        "transitions": transitions,
        "scope": "neutral synthetic software fixture; not biological validation",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def run_smoke(output_root: Optional[Path] = None) -> Dict[str, object]:
    temporary = output_root is None
    root = Path(tempfile.mkdtemp(prefix="wld_temporal_smoke_")) if temporary else Path(output_root)
    if root.exists() and not temporary:
        shutil.rmtree(root)
        root.mkdir(parents=True)
    data_root = root / "data"
    result_root = root / "results"
    write_smoke_dataset(data_root)
    cohort = load_temporal_cohort(data_root)
    if cohort.alignment_mode != "distribution":
        raise AssertionError("Smoke cohort did not retain distributional alignment.")

    config = TemporalTrainingConfig(
        epochs=12,
        learning_rate=3e-3,
        integration_steps=4,
        max_initial_cells=24,
        max_target_cells=20,
        projections=8,
        quantiles=8,
        validation_every=2,
        patience=4,
        fixed_point_iterations=40,
        jacobian_max_dimension=32,
        basin_trials=4,
        basin_horizon=1.0,
        basin_steps=8,
        seed=17,
    )
    result = run_temporal_benchmark(
        data_root,
        result_root,
        config,
        conditions=("true_circuit", "no_circuit"),
        device_name="cpu",
    )
    for condition in ("true_circuit", "no_circuit"):
        condition_result = result["conditions"][condition]
        if condition_result["training"]["test_groups_used_during_selection"]:
            raise AssertionError("Test groups leaked into model selection.")
        if not condition_result["test"]["test_evaluated_once_after_model_selection"]:
            raise AssertionError("Test evaluation contract was not recorded.")
        attractor = condition_result["test"]["attractor_diagnostics"]
        if attractor["status"] != "evaluated":
            raise AssertionError("Terminal test diagnostic was not evaluated.")
        if attractor["unconditional_attractor_claim"]:
            raise AssertionError("Terminal diagnostics made an attractor claim.")
        checkpoint = result_root / condition_result["checkpoint"]
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    if result["control_comparison"]["unconditional_success_claim"]:
        raise AssertionError("Control comparison made an unconditional claim.")
    if not (result_root / "wld_temporal_results.json").exists():
        raise FileNotFoundError("Temporal result report was not written.")

    print("PASS: temporal cohort schema and biological-group split", flush=True)
    print("PASS: unpaired distributional training without fabricated cell pairs", flush=True)
    print("PASS: validation-selected checkpoint and sealed held-out test groups", flush=True)
    print("PASS: true-circuit and no-circuit conditions retained without forced claim", flush=True)
    print("PASS: terminal fixed-point, Jacobian, and basin diagnostics without forced claim", flush=True)
    print("PASS: checkpoints and wld_temporal_results.json", flush=True)
    if temporary:
        shutil.rmtree(root)
    return result


def main() -> None:
    run_smoke()


if __name__ == "__main__":
    main()
