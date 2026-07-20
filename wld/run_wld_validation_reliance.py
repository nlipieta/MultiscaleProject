"""Continue the sealed WLD validation run and audit frozen circuit reliance.

This script consumes the already completed GSE240061 validation-only run.  It
never evaluates test subjects J/L.  All three matched conditions are continued
from their validation-selected seed-42 checkpoints, after which the continued
true-circuit checkpoint is evaluated with its TF circuit frozen intact,
knocked out, randomly thinned, and replaced by the supported sign-shuffled
mechanism without retraining the remaining parameters.

The output is a development diagnostic, not an attractor or biological
replication claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path("/content/wld_real_data")
COHORT_ROOT = ROOT / "gse240061_temporal_cohort"
SOURCE_RESULTS = ROOT / "gse240061_temporal_development_v2_seed42"
AUDIT_RESULTS = ROOT / "gse240061_validation_reliance_seed42"
CODE_ROOT = ROOT / "validation_reliance_code"

REPOSITORY = "nlipieta/MultiscaleProject"
DEPENDENCY_COMMIT = "13673a8490c57eca0284ac62c527d8c2fb37cab5"
CODE_HASHES = {
    "wld_circuit_dynamics_v3.py": (
        "2ffcd9d0a60551dd06db2646c60747ba0680e47150fd5f91bf42b7d8eadfe068"
    ),
    "wld_temporal_training.py": (
        "44b76cca948b3da27df7ce1934428047c83fd55e60b8ea1db8ed89851d9d50c4"
    ),
}
EXPECTED_SPLIT = {
    "train": {"E", "G", "N"},
    "validation": {"I"},
    "test": {"J", "L"},
}
CONDITIONS = ("true_circuit", "no_circuit", "sign_shuffled_circuit")
ADDITIONAL_EPOCHS = 600
CONTINUATION_LR = 5e-4
VALIDATION_EVERY = 10
PATIENCE_CHECKS = 20
MIN_IMPROVEMENT = 1e-5
AUDIT_CELLS = 512
DROPOUT_REPEATS = 16
DROPOUT_FRACTION = 0.5
MIN_RELATIVE_SWD_ADVANTAGE = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_verified(name: str) -> Path:
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = CODE_ROOT / name
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/"
        f"{DEPENDENCY_COMMIT}/wld/{name}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "WLD-validation-reliance/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    observed = hashlib.sha256(payload).hexdigest()
    expected = CODE_HASHES[name]
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {name}: expected {expected}, observed {observed}"
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    print(f"   verified: {name} ({observed[:12]}...)", flush=True)
    return destination


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def normalized_split(value: Mapping[str, Sequence[object]]) -> Dict[str, set[str]]:
    return {name: {str(item) for item in value[name]} for name in EXPECTED_SPLIT}


def validate_sealed_inputs(development: Mapping[str, object]) -> None:
    if normalized_split(development["split_groups"]) != EXPECTED_SPLIT:
        raise RuntimeError("The source run does not use the frozen E/G/N, I, J/L split.")
    if development.get("test_groups_evaluated") is not False:
        raise RuntimeError("The source development report does not certify sealed tests.")
    if set(development.get("conditions", {})) != set(CONDITIONS):
        raise RuntimeError("The source run is missing a required matched condition.")
    for condition in CONDITIONS:
        record = development["conditions"][condition]
        if "test" in record:
            raise RuntimeError(f"Source condition {condition!r} contains test metrics.")
        checkpoint = SOURCE_RESULTS / str(record["checkpoint"])
        require_file(checkpoint, f"{condition} checkpoint")


def load_checkpoint_model(condition, cohort, development, device, torch, dynamics, trainer):
    record = development["conditions"][condition]
    checkpoint_path = SOURCE_RESULTS / str(record["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("condition") != condition:
        raise RuntimeError(f"Checkpoint condition mismatch for {condition!r}.")
    if checkpoint.get("test_groups_evaluated") is not False:
        raise RuntimeError(f"Checkpoint {checkpoint_path} is not test-sealed.")
    config = trainer.TemporalTrainingConfig(**checkpoint["config"])
    priors = trainer._condition_priors(
        cohort.priors, condition, cohort.control_priors
    )
    model = dynamics.CircuitDynamicsModel(priors).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, config, checkpoint_path


def continue_condition(
    condition,
    cohort,
    development,
    device,
    torch,
    np,
    dynamics,
    trainer,
):
    model, original_config, source_checkpoint = load_checkpoint_model(
        condition, cohort, development, device, torch, dynamics, trainer
    )
    source_training = development["conditions"][condition]["training"]
    source_epoch = int(source_training["best_epoch"])
    config = replace(
        original_config,
        epochs=source_epoch + ADDITIONAL_EPOCHS,
        learning_rate=CONTINUATION_LR,
        validation_every=VALIDATION_EVERY,
        patience=PATIENCE_CHECKS,
    )
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = trainer._mean_split_loss(
        model, cohort, "validation", config, device
    )
    best_epoch = source_epoch
    stale_checks = 0
    history = [
        {
            "epoch": float(source_epoch),
            "training_loss": None,
            "validation_loss": best_validation,
            "checkpoint_origin": True,
        }
    ]
    train_transitions = cohort.transition_ids("train")

    for local_epoch in range(1, ADDITIONAL_EPOCHS + 1):
        absolute_epoch = source_epoch + local_epoch
        model.train()
        optimizer.zero_grad()
        transition_losses = []
        for transition_id in train_transitions:
            total, _, _ = trainer._transition_loss(
                model,
                cohort,
                transition_id,
                config,
                device,
                epoch=absolute_epoch,
                sample=True,
            )
            transition_losses.append(total)
        training_loss = torch.stack(transition_losses).mean()
        if not torch.isfinite(training_loss):
            raise RuntimeError(
                f"Non-finite continuation loss for {condition!r} at epoch {absolute_epoch}."
            )
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        should_validate = (
            local_epoch % VALIDATION_EVERY == 0
            or local_epoch == ADDITIONAL_EPOCHS
        )
        if not should_validate:
            continue
        validation_loss = trainer._mean_split_loss(
            model, cohort, "validation", config, device
        )
        history.append(
            {
                "epoch": float(absolute_epoch),
                "training_loss": float(training_loss.detach().cpu()),
                "validation_loss": validation_loss,
                "checkpoint_origin": False,
            }
        )
        if validation_loss < best_validation - MIN_IMPROVEMENT:
            best_validation = validation_loss
            best_epoch = absolute_epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_checks = 0
        else:
            stale_checks += 1
            if stale_checks >= PATIENCE_CHECKS:
                break

    model.load_state_dict(best_state)
    checkpoint_path = AUDIT_RESULTS / f"wld_continued_{condition}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "condition": condition,
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "config": asdict(config),
            "source_checkpoint": str(source_checkpoint),
            "source_best_epoch": source_epoch,
            "test_groups_evaluated": False,
        },
        checkpoint_path,
    )
    validation = trainer.evaluate_split_groups(
        model, cohort, config, device, "validation"
    )
    return model, config, {
        "condition": condition,
        "source_best_epoch": source_epoch,
        "continued_best_epoch": best_epoch,
        "additional_epochs_run": int(history[-1]["epoch"] - source_epoch),
        "best_validation_loss": best_validation,
        "stopped_for_plateau": stale_checks >= PATIENCE_CHECKS,
        "history": history,
        "validation": validation,
        "checkpoint": checkpoint_path.name,
        "test_groups_used_during_continuation": False,
    }


def sampled_indices(indices, maximum: int, seed: int, np):
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def pearson(left, right, torch) -> Optional[float]:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-12:
        return None
    return float((torch.dot(left, right) / denominator).cpu())


def fixed_sample(cohort, trainer, np, device):
    transition_ids = cohort.transition_ids("validation")
    if len(transition_ids) != 1:
        raise RuntimeError(
            "This pinned audit expects one validation transition for subject I."
        )
    transition_id = transition_ids[0]
    spec = cohort.transitions[transition_id]
    if spec.group_id != "I":
        raise RuntimeError("The validation transition is not frozen subject I.")
    initial_indices = sampled_indices(
        cohort.initial_indices(transition_id), AUDIT_CELLS, 271828, np
    )
    target_indices = sampled_indices(
        cohort.target_indices(transition_id), AUDIT_CELLS, 314159, np
    )
    atac = cohort.initial_atac[initial_indices].to(device)
    cues = cohort.initial_cues[initial_indices].to(device)
    if cohort.initial_cue_mask is not None:
        cues = cues * cohort.initial_cue_mask[initial_indices].to(device)
    initial_rna = (
        cohort.initial_rna[initial_indices].to(device)
        if cohort.initial_rna is not None
        else None
    )
    target_rna = cohort.target_rna[target_indices].to(device)
    target_log = trainer._rna_log_expression(cohort, target_rna)
    target_atac = (
        cohort.target_atac[target_indices].to(device)
        if cohort.target_atac is not None
        else None
    )
    return {
        "transition_id": transition_id,
        "horizon": spec.horizon,
        "atac": atac,
        "cues": cues,
        "initial_rna": initial_rna,
        "target_log": target_log,
        "target_atac": target_atac,
        "initial_cells": len(initial_indices),
        "target_cells": len(target_indices),
    }


def counterfactual_metrics(
    model,
    sample,
    config,
    trainer,
    torch,
    intervention=None,
) -> Tuple[Dict[str, Optional[float]], object]:
    model.eval()
    with torch.no_grad():
        output = model(
            sample["atac"],
            sample["cues"],
            horizon=sample["horizon"],
            steps=config.integration_steps,
            initial_rna=sample["initial_rna"],
            intervention=intervention,
        )
        predicted_log = torch.log1p(output["rna_t"].clamp_min(0.0))
        target_log = sample["target_log"]
        metrics: Dict[str, Optional[float]] = {
            "log_rna_swd": float(
                trainer.sliced_wasserstein(
                    predicted_log,
                    target_log,
                    config.projections,
                    config.quantiles,
                    config.seed,
                ).cpu()
            ),
            "log_rna_mean_mse": float(
                torch.nn.functional.mse_loss(
                    predicted_log.mean(0), target_log.mean(0)
                ).cpu()
            ),
            "log_rna_mean_pearson": pearson(
                predicted_log.mean(0), target_log.mean(0), torch
            ),
            "terminal_velocity_rms": float(
                output["terminal_velocity"].square().mean().sqrt().cpu()
            ),
        }
        if sample["target_atac"] is not None:
            metrics["atac_swd"] = float(
                trainer.sliced_wasserstein(
                    output["accessibility_t"].clamp(0.0, 1.0),
                    sample["target_atac"],
                    config.projections,
                    config.quantiles,
                    config.seed + 1,
                ).cpu()
            )
        components = model.field.components(
            output["terminal_state"], sample["cues"], intervention
        )
        circuit_drive = model.field.tf_circuit(
            output["tf_t"].clamp_min(0.0), components["circuit_edge_gate"]
        )
        signal_drive = model.field.signal_to_tf(output["signal_t"].clamp_min(0.0))
        circuit_rms = circuit_drive.square().mean().sqrt()
        signal_rms = signal_drive.square().mean().sqrt()
        metrics["terminal_circuit_drive_rms"] = float(circuit_rms.cpu())
        metrics["terminal_signal_to_tf_drive_rms"] = float(signal_rms.cpu())
        metrics["circuit_to_signal_drive_ratio"] = float(
            (circuit_rms / signal_rms.clamp_min(1e-12)).cpu()
        )
        prediction = predicted_log.detach().clone()
    return metrics, prediction


def copy_trainable_parameters(source, target, torch) -> None:
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if set(source_parameters) != set(target_parameters):
        raise RuntimeError("Counterfactual model parameter names differ.")
    with torch.no_grad():
        for name, target_parameter in target_parameters.items():
            source_parameter = source_parameters[name]
            if source_parameter.shape != target_parameter.shape:
                raise RuntimeError(f"Counterfactual parameter shape differs: {name}")
            target_parameter.copy_(source_parameter)


def frozen_reliance_audit(
    true_model,
    config,
    cohort,
    device,
    torch,
    np,
    dynamics,
    trainer,
):
    sample = fixed_sample(cohort, trainer, np, device)
    normal_metrics, normal_prediction = counterfactual_metrics(
        true_model, sample, config, trainer, torch
    )
    edge_count = true_model.field.tf_circuit.num_edges
    if edge_count < 2:
        raise RuntimeError("The true circuit has too few edges for reliance auditing.")
    zero_scale = torch.zeros(edge_count, device=device)
    knockout = dynamics.CircuitIntervention(circuit_edge_scale=zero_scale)
    knockout_metrics, knockout_prediction = counterfactual_metrics(
        true_model, sample, config, trainer, torch, knockout
    )
    knockout_metrics["prediction_log_rms_shift_from_true"] = float(
        (knockout_prediction - normal_prediction).square().mean().sqrt().cpu()
    )

    if "sign_shuffled_circuit" not in cohort.control_priors:
        raise RuntimeError("The supported sign-shuffled control is unavailable.")
    sign_model = dynamics.CircuitDynamicsModel(
        cohort.control_priors["sign_shuffled_circuit"]
    ).to(device)
    copy_trainable_parameters(true_model, sign_model, torch)
    sign_metrics, sign_prediction = counterfactual_metrics(
        sign_model, sample, config, trainer, torch
    )
    sign_metrics["prediction_log_rms_shift_from_true"] = float(
        (sign_prediction - normal_prediction).square().mean().sqrt().cpu()
    )

    dropout_records = []
    generator = torch.Generator(device="cpu").manual_seed(161803)
    for repeat in range(DROPOUT_REPEATS):
        keep = (
            torch.rand(edge_count, generator=generator) >= DROPOUT_FRACTION
        ).float().to(device)
        intervention = dynamics.CircuitIntervention(circuit_edge_scale=keep)
        metrics, prediction = counterfactual_metrics(
            true_model, sample, config, trainer, torch, intervention
        )
        dropout_records.append(
            {
                "repeat": repeat,
                "retained_edges": int(keep.sum().cpu()),
                "log_rna_swd": metrics["log_rna_swd"],
                "swd_delta_from_true": (
                    metrics["log_rna_swd"] - normal_metrics["log_rna_swd"]
                ),
                "prediction_log_rms_shift_from_true": float(
                    (prediction - normal_prediction).square().mean().sqrt().cpu()
                ),
            }
        )

    knockout_delta = (
        knockout_metrics["log_rna_swd"] - normal_metrics["log_rna_swd"]
    )
    sign_delta = sign_metrics["log_rna_swd"] - normal_metrics["log_rna_swd"]
    knockout_relative = knockout_delta / max(knockout_metrics["log_rna_swd"], 1e-12)
    sign_relative = sign_delta / max(sign_metrics["log_rna_swd"], 1e-12)
    dropout_deltas = np.asarray(
        [record["swd_delta_from_true"] for record in dropout_records], dtype=float
    )
    gate_passed = bool(
        knockout_relative >= MIN_RELATIVE_SWD_ADVANTAGE
        and sign_relative >= MIN_RELATIVE_SWD_ADVANTAGE
        and float(np.median(dropout_deltas)) > 0.0
    )
    return {
        "scope": "validation subject I only; test subjects J/L were not evaluated",
        "sample": {
            "transition_id": sample["transition_id"],
            "initial_cells": sample["initial_cells"],
            "target_cells": sample["target_cells"],
            "sampling_seeds": {"initial": 271828, "target": 314159},
        },
        "true_circuit": normal_metrics,
        "frozen_circuit_knockout": knockout_metrics,
        "frozen_supported_sign_shuffle": sign_metrics,
        "random_half_edge_dropout": {
            "fraction_dropped": DROPOUT_FRACTION,
            "repeats": dropout_records,
            "median_swd_delta_from_true": float(np.median(dropout_deltas)),
            "positive_delta_fraction": float(np.mean(dropout_deltas > 0.0)),
        },
        "development_gate": {
            "minimum_relative_swd_advantage": MIN_RELATIVE_SWD_ADVANTAGE,
            "knockout_relative_swd_advantage": knockout_relative,
            "sign_shuffle_relative_swd_advantage": sign_relative,
            "requires_positive_median_dropout_delta": True,
            "passed": gate_passed,
            "biological_replication_claim": False,
        },
    }


def condition_comparison(conditions: Mapping[str, object]) -> dict:
    true_swd = conditions["true_circuit"]["validation"]["by_group"]["I"][
        "log_rna_swd"
    ]
    result = {}
    for control in ("no_circuit", "sign_shuffled_circuit"):
        control_swd = conditions[control]["validation"]["by_group"]["I"][
            "log_rna_swd"
        ]
        result[control] = {
            "control_minus_true_swd": float(control_swd - true_swd),
            "relative_true_advantage": float(
                (control_swd - true_swd) / max(control_swd, 1e-12)
            ),
        }
    return result


def finite(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    print("WLD VALIDATION-ONLY CONVERGENCE + FROZEN CIRCUIT RELIANCE AUDIT", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    development_path = SOURCE_RESULTS / "wld_temporal_development.json"
    require_file(development_path, "successful source development report")
    require_file(COHORT_ROOT / "manifest.json", "temporal cohort manifest")
    require_file(COHORT_ROOT / "observations.npz", "temporal observations")
    require_file(COHORT_ROOT / "priors.npz", "temporal priors")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    validate_sealed_inputs(development)
    print("PASS: source run is complete and J/L remain sealed", flush=True)

    print(f"\nDownloading pinned model code from {DEPENDENCY_COMMIT}...", flush=True)
    downloaded = {name: download_verified(name) for name in CODE_HASHES}
    for path in downloaded.values():
        subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    import numpy as np
    import torch
    import wld_circuit_dynamics_v3 as dynamics
    import wld_temporal_training as trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__} | device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    cohort = trainer.load_temporal_cohort(COHORT_ROOT)
    if normalized_split(cohort.split_groups) != EXPECTED_SPLIT:
        raise RuntimeError("Loaded cohort split differs from the frozen contract.")
    print("PASS: hard priors, supported null, and cohort schema", flush=True)

    AUDIT_RESULTS.mkdir(parents=True, exist_ok=True)
    conditions = {}
    continued_models = {}
    continued_configs = {}
    for condition in CONDITIONS:
        print(
            f"\nContinuing {condition!r} for up to {ADDITIONAL_EPOCHS} additional epochs...",
            flush=True,
        )
        model, config, record = continue_condition(
            condition,
            cohort,
            development,
            device,
            torch,
            np,
            dynamics,
            trainer,
        )
        continued_models[condition] = model
        continued_configs[condition] = config
        conditions[condition] = record
        print(
            f"   best epoch {record['continued_best_epoch']} | "
            f"validation loss {record['best_validation_loss']:.6f} | "
            f"plateau={record['stopped_for_plateau']}",
            flush=True,
        )
        partial = {
            "stage": "validation_only_continuation_partial",
            "conditions": conditions,
            "test_groups_evaluated": False,
        }
        atomic_json(AUDIT_RESULTS / "wld_validation_audit.partial.json", partial)

    comparison = condition_comparison(conditions)
    print("\nRunning frozen true-model counterfactuals on subject I only...", flush=True)
    reliance = frozen_reliance_audit(
        continued_models["true_circuit"],
        continued_configs["true_circuit"],
        cohort,
        device,
        torch,
        np,
        dynamics,
        trainer,
    )
    report = {
        "schema_version": 1,
        "stage": "validation_only_convergence_and_frozen_reliance",
        "split_groups": {
            name: list(cohort.split_groups[name]) for name in EXPECTED_SPLIT
        },
        "source_development": str(development_path),
        "source_development_sha256": sha256(development_path),
        "dependency_commit": DEPENDENCY_COMMIT,
        "dependency_sha256": CODE_HASHES,
        "device": str(device),
        "continuation": {
            "seed": 42,
            "additional_epoch_cap": ADDITIONAL_EPOCHS,
            "learning_rate": CONTINUATION_LR,
            "validation_every": VALIDATION_EVERY,
            "patience_checks": PATIENCE_CHECKS,
            "conditions": conditions,
            "trained_condition_comparison": comparison,
        },
        "frozen_reliance_audit": reliance,
        "test_groups_evaluated": False,
        "attractor_claim": False,
        "claim_boundary": (
            "Validation subject I can support development diagnostics only. "
            "The transient 3.5-hour endpoint cannot establish a fixed point or basin."
        ),
    }
    report_path = AUDIT_RESULTS / "wld_validation_reliance.json"
    atomic_json(report_path, report)
    (AUDIT_RESULTS / "wld_validation_audit.partial.json").unlink(missing_ok=True)

    print("\n" + "=" * 76, flush=True)
    print("VALIDATION-ONLY AUDIT SUMMARY", flush=True)
    print("=" * 76, flush=True)
    for condition in CONDITIONS:
        metrics = conditions[condition]["validation"]["by_group"]["I"]
        print(
            f"{condition:24s} epoch={conditions[condition]['continued_best_epoch']:4d} "
            f"SWD={finite(metrics['log_rna_swd'])} "
            f"mean-MSE={finite(metrics['log_rna_mean_mse'])} "
            f"Pearson={finite(metrics['log_rna_mean_pearson'])}",
            flush=True,
        )
    print("\nTrained-control comparison (positive favors true circuit):", flush=True)
    for control, values in comparison.items():
        print(
            f"  {control}: delta={finite(values['control_minus_true_swd'])}, "
            f"relative={100.0 * values['relative_true_advantage']:.3f}%",
            flush=True,
        )
    gate = reliance["development_gate"]
    print("\nFrozen true-model reliance:", flush=True)
    print(
        "  knockout relative SWD advantage: "
        f"{100.0 * gate['knockout_relative_swd_advantage']:.3f}%",
        flush=True,
    )
    print(
        "  sign-shuffle relative SWD advantage: "
        f"{100.0 * gate['sign_shuffle_relative_swd_advantage']:.3f}%",
        flush=True,
    )
    dropout = reliance["random_half_edge_dropout"]
    print(
        "  half-edge dropout median SWD delta: "
        f"{finite(dropout['median_swd_delta_from_true'])}",
        flush=True,
    )
    print(f"  1% development gate passed: {gate['passed']}", flush=True)
    print("\nTEST STATUS: J/L were not evaluated.", flush=True)
    print("ATTRACTOR STATUS: not assessed by this transient dataset.", flush=True)
    print(f"Report: {report_path}", flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
