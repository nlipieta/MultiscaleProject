"""CPU/CUDA smoke tests for the WLD v5.5 chromatin digital-model contract."""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

from wld_chromatin_modules_v55 import (
    ComplexModuleConfig,
    CuratedComplexCatalog,
    SparseFullChromatinBundle,
    compile_training_complex_modules,
    load_complex_module_atlas,
    load_v53_sparse_full_bundle,
)
from wld_chromatin_twin_training_v55 import (
    _base_supported_target_roster,
    _checkpoint_selection_summary,
    split_validation_targets,
)

from wld_chromatin_twin_v55 import (
    BranchOverrides,
    ChromatinTwinPriors,
    WLDChromatinDigitalTwin,
    architecture_contract,
    degree_preserving_bipartite_shuffle,
    topology_digest,
)
from wld_twin_statistics_v55 import (
    calibrate_ensemble_intervals,
    conformal_quantile,
    evaluate_claims,
    paired_target_bootstrap,
    target_bootstrap_mean,
)


class _FakeEncoder(nn.Module):
    def __init__(self, anchors: int, tfs: int) -> None:
        super().__init__()
        self.anchor_scale = nn.Parameter(torch.ones(anchors))
        self.tfs = tfs

    def forward(
        self,
        *,
        cues: torch.Tensor,
        atac: torch.Tensor,
        rna=None,
        protein=None,
        metabolic=None,
        modality_masks=None,
    ):
        context = torch.cat([atac * self.anchor_scale, cues], dim=1)
        tf = torch.nn.functional.softplus(atac[:, : self.tfs])
        return {"biological_context": context, "tf": tf}


class _FakeFoundation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tfs = 2
        self.num_peaks = 4
        self.context_dim = 3
        self.priors = SimpleNamespace(num_cues=1)
        self.encoder = _FakeEncoder(4, 2)
        self.context_network = nn.Sequential(nn.Linear(5, 3), nn.Tanh())


def fixture(
    device: torch.device | str = "cpu",
) -> tuple[WLDChromatinDigitalTwin, ChromatinTwinPriors]:
    resolved_device = torch.device(device)
    regulator_tf = torch.tensor(
        [
            [1.0, 0.0],  # TF-only
            [0.0, 0.0],  # complex-only
            [0.0, 1.0],  # dual
            [0.0, 0.0],  # unsupported
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        device=resolved_device,
    )
    motif = torch.tensor(
        [
            [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        device=resolved_device,
    )
    regulator_complex = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        device=resolved_device,
    )
    complex_module = torch.tensor(
        [[1.0, 0.0], [0.0, -0.8]],
        device=resolved_device,
    )
    module_peak = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, -0.7, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, -1.0],
        ],
        device=resolved_device,
    )
    priors = ChromatinTwinPriors(
        regulator_tf_support=regulator_tf,
        tf_peak_motif=motif,
        regulator_complex_support=regulator_complex,
        complex_module_effect=complex_module,
        module_peak_loading=module_peak,
        foundation_peak_index=torch.tensor(
            [0, 2, 4, 6],
            device=resolved_device,
        ),
    )
    torch.manual_seed(7)
    foundation = _FakeFoundation().to(resolved_device)
    return WLDChromatinDigitalTwin(foundation, priors).to(resolved_device), priors


def intervention(
    index: int,
    batch: int = 5,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    value = torch.zeros(batch, 6, device=device)
    value[:, index] = 1.0
    return value


def device_contract_smoke() -> dict:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        model, _priors = fixture(device)
        control = torch.full((3, 8), 0.35, device=device)
        cues = torch.zeros(3, 1, device=device)
        zero_intervention = torch.zeros(3, 6, device=device)
        supported_intervention = intervention(0, batch=3, device=device)

        for _name, parameter in model.named_parameters():
            assert parameter.device == device
        for _name, buffer in model.named_buffers():
            assert buffer.device == device

        zero = model(control, zero_intervention, cues=cues, steps=2)
        supported = model(control, supported_intervention, cues=cues, steps=2)
        assert zero["atac_t"].device == device
        assert supported["atac_t"].device == device
        assert torch.equal(zero["atac_t"], control)
        assert float((supported["atac_t"] - control).abs().sum().detach().cpu()) > 1e-6

        model.zero_grad(set_to_none=True)
        supported["atac_t"].sum().backward()
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
        "cuda_constructor_forward_backward_tested": torch.cuda.is_available(),
        "priors_moved_before_model_construction": True,
    }


def architecture_smoke() -> dict:
    torch.manual_seed(11)
    model, priors = fixture()
    control = torch.full((5, 8), 0.35)
    cues = torch.zeros(5, 1)
    original_digest = topology_digest(priors)

    # Intervention identity never enters the encoder.
    encoded = model.encode_control(control, cues=cues)
    tf_output = model(control, intervention(0), cues=cues, steps=3)
    complex_output = model(control, intervention(1), cues=cues, steps=3)
    assert torch.equal(encoded["context"], tf_output["context"])
    assert torch.equal(tf_output["context"], complex_output["context"])

    zero = model(control, torch.zeros(5, 6), cues=cues, steps=3)
    all_removed = model(
        control,
        intervention(2),
        cues=cues,
        steps=3,
        overrides=BranchOverrides(tf_scale=0.0, complex_scale=0.0),
    )
    assert torch.equal(zero["atac_t"], control)
    assert torch.equal(all_removed["atac_t"], control)

    # Persistence is the exact measured tensor, including boundary values and
    # non-anchor bins; it is not persistence of a silently clamped surrogate.
    measured = torch.linspace(0.0, 1.0, steps=40).reshape(5, 8)
    exact_zero = model(measured, torch.zeros(5, 6), cues=cues, steps=3)
    exact_all_removed = model(
        measured,
        intervention(2),
        cues=cues,
        steps=3,
        overrides=BranchOverrides(tf_scale=0.0, complex_scale=0.0),
    )
    assert torch.equal(exact_zero["initial_atac"], measured)
    assert torch.equal(exact_zero["atac_t"], measured)
    assert torch.equal(exact_all_removed["initial_atac"], measured)
    assert torch.equal(exact_all_removed["atac_t"], measured)

    # Index 1 is not a foundation anchor.  Invalid values there must still be
    # rejected before the anchor subset is encoded or the field is integrated.
    valid_values = model.encode_control(control, cues=cues)
    for bad_value, expected_message in (
        (-0.01, "[0,1]"),
        (1.01, "[0,1]"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
    ):
        invalid_response = control.clone()
        invalid_response[:, 1] = bad_value
        try:
            model.encode_control(invalid_response, cues=cues)
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError("invalid non-anchor response ATAC reached the encoder")
        try:
            model.field.integrate(
                invalid_response,
                torch.zeros(5, 6),
                valid_values["context"],
                steps=1,
            )
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError("invalid non-anchor response ATAC reached integration")

    # Branch-selective causal behavior.
    tf_removed = model(
        control, intervention(0), cues=cues, steps=3,
        overrides=BranchOverrides(tf_scale=0.0),
    )
    complex_removed = model(
        control, intervention(0), cues=cues, steps=3,
        overrides=BranchOverrides(complex_scale=0.0),
    )
    assert torch.equal(tf_removed["atac_t"], control)
    assert torch.allclose(complex_removed["atac_t"], tf_output["atac_t"])
    assert float(
        (tf_output["atac_t"][:, :2] - control[:, :2]).abs().sum().detach()
    ) > 1e-6
    assert torch.equal(tf_output["atac_t"][:, 4:], control[:, 4:])

    complex_tf_removed = model(
        control, intervention(1), cues=cues, steps=3,
        overrides=BranchOverrides(tf_scale=0.0),
    )
    complex_branch_removed = model(
        control, intervention(1), cues=cues, steps=3,
        overrides=BranchOverrides(complex_scale=0.0),
    )
    assert torch.allclose(complex_tf_removed["atac_t"], complex_output["atac_t"])
    assert torch.equal(complex_branch_removed["atac_t"], control)
    assert torch.equal(complex_output["atac_t"][:, :4], control[:, :4])
    assert float(
        (complex_output["atac_t"][:, 4:6] - control[:, 4:6]).abs().sum().detach()
    ) > 1e-6

    unsupported = model(control, intervention(3), cues=cues, steps=3)
    assert torch.equal(unsupported["atac_t"], control)
    assert torch.allclose(
        tf_output["peak_drive"],
        tf_output["tf_peak_drive"] + tf_output["complex_peak_drive"],
    )

    # Context changes supported magnitudes but cannot create a route.
    with torch.no_grad():
        model.field.tf_context_gain.weight.fill_(0.2)
        model.field.complex_context_gain.weight.fill_(-0.15)
        model.field.module_context_gain.weight.fill_(0.1)
    context_a = model(control, intervention(2), cues=torch.zeros(5, 1), steps=2)
    context_b = model(control, intervention(2), cues=torch.ones(5, 1), steps=2)
    assert not torch.allclose(context_a["atac_t"], context_b["atac_t"])
    zero_context_b = model(control, torch.zeros(5, 6), cues=torch.ones(5, 1), steps=2)
    assert torch.equal(zero_context_b["atac_t"], control)

    # Supported branches receive gradients; topology buffers do not.
    model.zero_grad(set_to_none=True)
    dual = model(control, intervention(2), cues=cues, steps=2)
    dual["atac_t"].sum().backward()
    gradient_names = {
        name
        for name, parameter in model.field.named_parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
    }
    for required in (
        "raw_tf_gain", "raw_motif_gain", "raw_complex_gain", "raw_cm_gain", "raw_module_gain"
    ):
        assert required in gradient_names, (required, sorted(gradient_names))
    assert all(not buffer.requires_grad for _name, buffer in model.field.named_buffers())

    # An ablation cannot invent evidence.
    invalid = priors.regulator_tf_support.clone()
    invalid[3, 0] = 1.0
    try:
        model(
            control, intervention(3), cues=cues,
            overrides=BranchOverrides(regulator_tf_support=invalid),
        )
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unsupported frozen override was accepted")
    amplified = priors.regulator_tf_support.clone()
    amplified[0, 0] *= 2.0
    try:
        model(
            control, intervention(0), cues=cues,
            overrides=BranchOverrides(regulator_tf_support=amplified),
        )
    except ValueError as error:
        assert "amplifies" in str(error)
    else:
        raise AssertionError("evidence-amplifying frozen override was accepted")

    for invalid_horizon in (float("nan"), float("inf")):
        try:
            model.field.integrate(
                control, torch.zeros(5, 6), valid_values["context"],
                horizon=invalid_horizon,
            )
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError("nonfinite integration horizon was accepted")
    try:
        model.field.integrate(
            control, torch.zeros(5, 6), valid_values["context"], steps=1.5,
        )
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("fractional integration steps were accepted")

    shuffled = degree_preserving_bipartite_shuffle(
        priors.regulator_complex_support, seed=19
    )
    assert torch.equal((shuffled > 0).sum(0), (priors.regulator_complex_support > 0).sum(0))
    assert torch.equal((shuffled > 0).sum(1), (priors.regulator_complex_support > 0).sum(1))
    assert not torch.equal(shuffled > 0, priors.regulator_complex_support > 0)
    assert topology_digest(priors) == original_digest

    contract = architecture_contract(model)
    assert contract["guide_or_target_in_encoder"] is False
    assert contract["direct_context_to_peak_delta_decoder"] is False
    assert contract["unsupported_edges_trainable"] is False
    assert contract["attractor_claim"] is False
    return contract


def statistics_smoke() -> dict:
    rows = []
    for target_number, target in enumerate(("A", "B", "C", "D")):
        for seed in (7, 11, 17):
            true = 0.20 + 0.01 * target_number + 0.001 * seed
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "true_loss": true,
                    "persistence_loss": true + 0.10 + 0.01 * target_number,
                    "shuffle_loss": true + 0.07 + 0.005 * target_number,
                    "frozen_loss": true + 0.05,
                }
            )
    persistence = paired_target_bootstrap(
        rows, true_key="true_loss", comparator_key="persistence_loss", samples=500, random_seed=3
    )
    shuffle = paired_target_bootstrap(
        list(reversed(rows)), true_key="true_loss", comparator_key="shuffle_loss", samples=500, random_seed=3
    )
    frozen = paired_target_bootstrap(
        rows, true_key="true_loss", comparator_key="frozen_loss", samples=500, random_seed=3
    )
    nrmse_interval = target_bootstrap_mean(
        [
            {"target": row["target"], "seed": row["seed"], "nrmse": 0.4}
            for row in rows
        ],
        value_key="nrmse",
        samples=500,
        random_seed=4,
    )
    assert abs(nrmse_interval.lower - 0.4) < 1e-12
    assert abs(nrmse_interval.upper - 0.4) < 1e-12
    assert persistence.lower > 0 and shuffle.lower > 0 and frozen.lower > 0
    assert conformal_quantile(range(1, 11), alpha=0.2) == 9.0
    try:
        conformal_quantile((1.0, 2.0, 3.0), alpha=0.2)
    except ValueError as error:
        assert "too small" in str(error)
    else:
        raise AssertionError("anti-conservative small calibration block was accepted")
    intervals = calibrate_ensemble_intervals(
        [
            {"target": "CAL1", "observed": 1.2, "ensemble_mean": 1.0, "ensemble_std": 0.1},
            {"target": "CAL2", "observed": 1.8, "ensemble_mean": 2.0, "ensemble_std": 0.1},
            {"target": "CAL3", "observed": 2.9, "ensemble_mean": 3.0, "ensemble_std": 0.1},
            {"target": "CAL4", "observed": 4.1, "ensemble_mean": 4.0, "ensemble_std": 0.1},
        ],
        [{"target": "VAL", "ensemble_mean": 2.5, "ensemble_std": 0.2}],
        alpha=0.2,
    )
    assert intervals[0]["lower"] <= 2.5 <= intervals[0]["upper"]
    claims = evaluate_claims(
        persistence=persistence,
        topology_shuffle=shuffle,
        frozen_removal=frozen,
        frozen_tf_removal=frozen,
        frozen_complex_removal=frozen,
        response_nrmse=nrmse_interval,
        response_cosine=nrmse_interval,
        calibrated_coverage=0.8,
        normalized_interval_width=0.9,
        external_subject_study_test=False,
        prospective_update_loop=False,
        longitudinal_return_data=False,
    )
    assert claims["transient_mechanistic_response"] is True
    assert claims["digital_twin_claim"] is False
    assert claims["attractor_claim"] is False
    return {
        "persistence_effect": persistence.__dict__,
        "shuffle_effect": shuffle.__dict__,
        "frozen_effect": frozen.__dict__,
        "nrmse_interval": nrmse_interval.__dict__,
        "claims": claims,
    }


def data_contract_smoke() -> dict:
    matrix = np.asarray(
        [
            [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],
            [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],
            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
            [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    targets = ("A", "A", "A", "B", "B", "B", "NTC", "NTC", "NTC", "V", "V", "NTC")
    splits = ("train",) * 9 + ("validation",) * 3
    screens = ("S",) * len(targets)
    groups = {}
    for index, key in enumerate(zip(splits, screens, targets)):
        groups.setdefault(key, []).append(index)
    groups = {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}

    def bundle(values: np.ndarray) -> SparseFullChromatinBundle:
        return SparseFullChromatinBundle(
            sparse.csr_matrix(values),
            ("P0", "P1", "P2", "P3"),
            np.asarray([0, 2], dtype=np.int64),
            targets,
            screens,
            splits,
            np.arange(len(targets), dtype=np.int64),
            groups,
            {"test_values_materialized": False},
            sealed_test_row_count=5,
        )

    catalog = CuratedComplexCatalog(
        ("C1",),
        ("synthetic complex",),
        (("A", "B", "V"),),
        {"source_sha256": "0" * 64},
    )
    config = ComplexModuleConfig(
        bootstrap_replicates=10,
        bootstrap_chunk_size=5,
        min_target_sign_stability=0.5,
        min_complex_sign_concordance=0.5,
    )
    with tempfile.TemporaryDirectory() as directory:
        first = compile_training_complex_modules(
            bundle(matrix), catalog, ("A", "B", "V"), config=config,
            output_root=Path(directory),
        )
        changed = matrix.copy()
        changed[9:11] = 1.0 - changed[9:11]
        second = compile_training_complex_modules(
            bundle(changed), catalog, ("A", "B", "V"), config=config,
        )
        assert np.array_equal(
            first.module_peak_loading.toarray(), second.module_peak_loading.toarray()
        )
        restored = load_complex_module_atlas(Path(directory))
        assert restored.construction_targets == ("A", "B")
        assert restored.provenance["claims"]["test_values_materialized"] is False

    # Exercise the actual on-disk v5.3 loader, including fail-before-CSR-read
    # behavior for a mislabeled sealed target or hashed control.
    seed = 42

    def ntc_split(barcode: str) -> str:
        digest = hashlib.sha256(f"{seed}|control|S|{barcode}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(2**64)
        return "train" if fraction < 0.70 else ("validation" if fraction < 0.85 else "test")

    ntc_barcodes = {}
    candidate = 0
    while set(ntc_barcodes) != {"train", "validation", "test"}:
        barcode = f"NTC{candidate}"
        ntc_barcodes.setdefault(ntc_split(barcode), barcode)
        candidate += 1

    base_rows = [
        [0, "S", "A0", "A", "train"],
        [1, "S", "B0", "B", "train"],
        [2, "S", "V10", "V1", "validation"],
        [3, "S", "V20", "V2", "validation"],
        [4, "S", "X0", "X", "test"],
        [5, "S", ntc_barcodes["train"], "NTC", "train"],
        [6, "S", ntc_barcodes["validation"], "NTC", "validation"],
        [7, "S", ntc_barcodes["test"], "NTC", "test"],
    ]

    def write_cells(root: Path, rows) -> None:
        with gzip.open(root / "cells.tsv.gz", "wt") as handle:
            handle.write("row\tscreen\tbarcode\ttarget\tsplit\n")
            for row in rows:
                handle.write("\t".join(map(str, row)) + "\n")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sparse.save_npz(
            root / "atac_counts.GRCh38.2kb.npz",
            sparse.csr_matrix(np.eye(8, 4, dtype=np.float32)),
        )
        with gzip.open(root / "bins.GRCh38.2kb.tsv.gz", "wt") as handle:
            handle.write("P0\nP1\nP2\nP3\n")
        (root / "wld_v53_ingestion_manifest.json").write_text(
            json.dumps({"claims": {"test_evaluated": False}})
        )
        (root / "whole_target_split.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "targets": {
                        "train": ["A", "B"],
                        "validation": ["V1", "V2"],
                        "test": ["X"],
                    },
                    "test_evaluated": False,
                }
            )
        )
        write_cells(root, base_rows)
        loaded = load_v53_sparse_full_bundle(
            root, foundation_peaks=("P0", "P2")
        )
        assert set(loaded.split_targets("train")) == {"A", "B"}
        assert set(loaded.split_targets("validation")) == {"V1", "V2"}
        assert "X" not in loaded.targets and loaded.sealed_test_row_count == 2

        # Corrupt the matrix deliberately: the two metadata violations below
        # must be rejected before the sparse archive is opened.
        (root / "atac_counts.GRCh38.2kb.npz").write_bytes(b"not an NPZ")
        wrong_target = [list(row) for row in base_rows]
        wrong_target[4][4] = "train"
        write_cells(root, wrong_target)
        try:
            load_v53_sparse_full_bundle(root, foundation_peaks=("P0", "P2"))
        except RuntimeError as error:
            assert "whole-target split is test" in str(error)
        else:
            raise AssertionError("mislabeled sealed target reached the sparse matrix")

        wrong_control = [list(row) for row in base_rows]
        wrong_control[6][4] = "train"
        write_cells(root, wrong_control)
        try:
            load_v53_sparse_full_bundle(root, foundation_peaks=("P0", "P2"))
        except RuntimeError as error:
            assert "NTC barcode-hash split is validation" in str(error)
        else:
            raise AssertionError("mislabeled NTC reached the sparse matrix")

    blocks = split_validation_targets(
        tuple(f"T{index:02d}" for index in range(12)),
        tuple(f"T{index:02d}" for index in range(12)),
        seed=71,
        calibration_fraction=0.5,
        minimum_selection=2,
        minimum_calibration=4,
        minimum_audit=2,
    )
    assert len(blocks["selection"]) == 6
    assert len(blocks["calibration"]) == 4
    assert len(blocks["audit"]) == 2
    assert not (
        set(blocks["selection"]) & set(blocks["calibration"])
        or set(blocks["selection"]) & set(blocks["audit"])
        or set(blocks["calibration"]) & set(blocks["audit"])
    )

    # A retrained shuffle may disconnect a base-supported target downstream.
    # It must still see the identical train roster, and that target's actual
    # persistence-like prediction must remain in checkpoint selection.
    fixed_roster = _base_supported_target_roster(
        ("A", "B", "C"), ("A", "B", "C"), (True, True, False)
    )
    shuffled_reachability = {"A": True, "B": False, "C": False}
    assert fixed_roster == ("A", "B")
    assert any(not shuffled_reachability[target] for target in fixed_roster)
    selection_summary = _checkpoint_selection_summary(
        {
            "all_targets": {"targets": 2, "selection_score": 0.9},
            "route_supported_targets": {"targets": 1, "selection_score": 0.1},
        }
    )
    assert selection_summary["targets"] == 2
    assert selection_summary["selection_score"] == 0.9
    return {
        "module_construction_targets": list(first.construction_targets),
        "validation_mutation_changed_modules": False,
        "frozen_split_loader_exercised": True,
        "fixed_base_roster_used_by_all_retrained_conditions": True,
        "target_blocks": blocks,
    }


def main() -> None:
    torch.set_num_threads(1)
    report = {
        "scope": "synthetic numerical/contract verification only; no biological claim",
        "device_contract": device_contract_smoke(),
        "architecture": architecture_smoke(),
        "statistics": statistics_smoke(),
        "data_contract": data_contract_smoke(),
        "claims": {
            "biological_claim": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
            "sealed_test_evaluated": False,
        },
    }
    output = Path("wld_v55_synthetic_validation.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("PASS: immutable dual-branch evidence topology and provenance")
    print("PASS: device-local prior construction, forward pass, and gradients")
    print("PASS: exact persistence and full-response ATAC domain validation")
    print("PASS: complex and TF branches have selective causal effects")
    print("PASS: context varies supported gains/rates without creating edges")
    print("PASS: named interventions occur after encoding with no direct bypass")
    print("PASS: target-level bootstrap and target-block uncertainty calibration")
    print("PASS: training-only complex modules and three disjoint target blocks")
    print("PASS: transient/digital-twin/attractor claim boundaries")
    print(f"PASS: wrote {output}")


if __name__ == "__main__":
    main()
