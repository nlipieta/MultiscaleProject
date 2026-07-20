"""Matched topology controls for WLD v5.6 null-aware development.

The v5.5 degree-swap control preserved the immediate bipartite degree
sequences, but it did not preserve the distribution of *end-to-end* genomic
footprints induced by the fixed downstream motif and complex-module maps.
That leaves a possible capacity difference between the biological graph and
its null graph.

This module implements a stricter named-regulator null.  A control jointly
permutes each regulator's TF and complex route profiles *within its immutable
whole-target split* while leaving all downstream evidence tensors unchanged.
Consequently every control preserves, exactly and separately in train,
validation and test (as well as jointly across the regulator roster):

* the TF and complex support weight multisets and column degrees;
* the distribution of reachable genomic-bin footprints;
* the distribution of positive/negative complex-route footprints; and
* the distributions of signed and absolute end-to-end path mass.

What changes is the assignment of those mechanistic profiles to named
perturbation targets.  This is therefore a null for biological target-to-route
specificity, not a claim that every possible graph randomization has been
tested.  At least ten independently seeded controls are required.
"""

from __future__ import annotations

import hashlib
import numbers
from dataclasses import fields, replace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

try:
    from wld_chromatin_twin_v56 import ChromatinTwinPriors
except ModuleNotFoundError as error:  # Allows isolated helper testing before v5.6 lands.
    if error.name != "wld_chromatin_twin_v56":
        raise
    from wld_chromatin_twin_v55 import ChromatinTwinPriors


MINIMUM_CONTROL_REPLICATES = 10


def _tensor_digest(value: Tensor) -> bytes:
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.digest()


def _prior_digest(priors: ChromatinTwinPriors) -> str:
    digest = hashlib.sha256()
    for field in fields(priors):
        digest.update(field.name.encode("utf-8"))
        digest.update(_tensor_digest(getattr(priors, field.name)))
    return digest.hexdigest()


def _sattolo_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a one-cycle derangement of ``range(size)``."""

    if size < 2:
        raise ValueError("A derangement requires at least two regulators")
    permutation = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        swap = int(rng.integers(0, index))
        permutation[index], permutation[swap] = permutation[swap], permutation[index]
    if np.any(permutation == np.arange(size)):
        raise RuntimeError("Sattolo permutation unexpectedly contains a fixed point")
    return permutation


def _end_to_end_summaries(priors: ChromatinTwinPriors) -> Mapping[str, Tensor]:
    """Summarize every named regulator's complete fixed-evidence footprint."""

    regulator_tf = torch.as_tensor(priors.regulator_tf_support, dtype=torch.float64)
    motif = torch.as_tensor(
        priors.tf_peak_motif,
        dtype=torch.float64,
        device=regulator_tf.device,
    )
    regulator_complex = torch.as_tensor(
        priors.regulator_complex_support,
        dtype=torch.float64,
        device=regulator_tf.device,
    )
    complex_module = torch.as_tensor(
        priors.complex_module_effect,
        dtype=torch.float64,
        device=regulator_tf.device,
    )
    module_peak = torch.as_tensor(
        priors.module_peak_loading,
        dtype=torch.float64,
        device=regulator_tf.device,
    )

    # Motifs are unsigned localization evidence.  Complex/module effects carry
    # compiler-fixed signs, so their positive/negative footprint is auditable.
    tf_paths = regulator_tf @ motif.abs()
    complex_paths = regulator_complex @ complex_module @ module_peak
    return {
        "tf_bin_footprint": torch.count_nonzero(tf_paths, dim=1),
        "tf_absolute_mass": tf_paths.abs().sum(dim=1),
        "complex_bin_footprint": torch.count_nonzero(complex_paths, dim=1),
        "complex_positive_bins": torch.count_nonzero(complex_paths > 0, dim=1),
        "complex_negative_bins": torch.count_nonzero(complex_paths < 0, dim=1),
        "complex_absolute_mass": complex_paths.abs().sum(dim=1),
        "complex_signed_mass": complex_paths.sum(dim=1),
    }


def _assert_row_permuted(
    base: Mapping[str, Tensor],
    control: Mapping[str, Tensor],
    permutation: Tensor,
) -> None:
    for name, expected_rows in base.items():
        expected = expected_rows.index_select(0, permutation.to(expected_rows.device))
        observed = control[name].to(expected.device)
        if expected.dtype.is_floating_point:
            matched = torch.allclose(observed, expected, rtol=1e-10, atol=1e-12)
        else:
            matched = torch.equal(observed, expected)
        if not matched:
            raise RuntimeError(f"Control failed end-to-end {name} matching")


def _support_audit(base: Tensor, control: Tensor) -> Dict[str, object]:
    base = torch.as_tensor(base).detach().cpu()
    control = torch.as_tensor(control).detach().cpu()
    base_edges, control_edges = base > 0, control > 0
    weight_multiset_exact = bool(
        torch.equal(
            torch.sort(base[base_edges]).values,
            torch.sort(control[control_edges]).values,
        )
    )
    return {
        "edges": int(torch.count_nonzero(base_edges)),
        "column_degrees_exact": bool(
            torch.equal(base_edges.sum(dim=0), control_edges.sum(dim=0))
        ),
        "row_degree_distribution_exact": bool(
            torch.equal(
                torch.sort(base_edges.sum(dim=1)).values,
                torch.sort(control_edges.sum(dim=1)).values,
            )
        ),
        "weight_multiset_exact": weight_multiset_exact,
        # Exact equality of the weight multiset proves mathematical mass
        # equality without depending on floating-point reduction order.
        "total_mass_exact": weight_multiset_exact,
        "total_mass_exact_from_weight_multiset": weight_multiset_exact,
    }


def _normalize_strata(
    strata: Sequence[object] | Mapping[int, object],
    regulators: int,
) -> Tuple[str, ...]:
    """Return one immutable split label per regulator without reading outcomes."""

    if isinstance(strata, Mapping):
        expected = set(range(regulators))
        observed = set(strata)
        if observed != expected:
            raise ValueError(
                "strata mapping keys must be exactly the aligned regulator indices"
            )
        values = tuple(str(strata[index]) for index in range(regulators))
    else:
        if isinstance(strata, (str, bytes)):
            raise TypeError("strata must contain one split label per regulator")
        values = tuple(str(value) for value in strata)
        if len(values) != regulators:
            raise ValueError("strata length does not match the regulator vocabulary")
    if any(not value.strip() for value in values):
        raise ValueError("strata labels cannot be empty")
    counts = {value: values.count(value) for value in set(values)}
    too_small = sorted(value for value, count in counts.items() if count < 2)
    if too_small:
        raise ValueError(
            "Every split stratum needs at least two regulators for a derangement: "
            f"{too_small}"
        )
    return values


def _stratified_derangement(
    labels: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Derange regulator profiles independently within each split label."""

    permutation = np.arange(len(labels), dtype=np.int64)
    for label in sorted(set(labels)):
        members = np.asarray(
            [index for index, value in enumerate(labels) if value == label],
            dtype=np.int64,
        )
        within = _sattolo_permutation(len(members), rng)
        permutation[members] = members[within]
    if np.any(permutation == np.arange(len(labels))):
        raise RuntimeError("Stratified derangement unexpectedly contains a fixed point")
    if any(labels[index] != labels[source] for index, source in enumerate(permutation)):
        raise RuntimeError("Stratified derangement crossed a split boundary")
    return permutation


def build_matched_control_priors(
    base_priors: ChromatinTwinPriors,
    replicates: int,
    seed: int,
    *,
    strata: Sequence[object] | Mapping[int, object],
) -> Tuple[List[ChromatinTwinPriors], Dict[str, object]]:
    """Build deterministic end-to-end matched named-topology controls.

    Parameters
    ----------
    base_priors:
        The true fixed-evidence prior scaffold.
    replicates:
        Number of independent controls.  At least ten are required so a result
        cannot depend on one or two unusually easy null graphs.
    seed:
        Root seed used only to construct regulator-profile derangements.
    strata:
        Required train/validation/test (or equivalent) labels aligned to the
        regulator rows.  A mapping must use integer regulator indices as keys.
        Only these names/labels are consumed; no test observation or outcome is
        accepted by this API.  Profiles are deranged independently per stratum.

    Returns
    -------
    controls, audit:
        A list of ``ChromatinTwinPriors`` and JSON-serializable provenance for
        every permutation and matching invariant.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, numbers.Integral):
        raise TypeError("replicates must be an integer")
    if int(replicates) < MINIMUM_CONTROL_REPLICATES:
        raise ValueError(
            f"At least {MINIMUM_CONTROL_REPLICATES} matched topology controls are required"
        )
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer")
    base_priors.validate()

    regulator_tf = torch.as_tensor(base_priors.regulator_tf_support)
    regulator_complex = torch.as_tensor(base_priors.regulator_complex_support)
    regulators = int(regulator_tf.shape[0])
    if regulators < 3 or regulator_complex.shape[0] != regulators:
        raise ValueError("TF and complex supports need at least three aligned regulators")
    split_labels = _normalize_strata(strata, regulators)
    stratum_indices = {
        label: torch.as_tensor(
            [index for index, value in enumerate(split_labels) if value == label],
            dtype=torch.long,
        )
        for label in sorted(set(split_labels))
    }

    rng = np.random.default_rng(int(seed))
    base_digest = _prior_digest(base_priors)
    base_summary = _end_to_end_summaries(base_priors)
    controls: List[ChromatinTwinPriors] = []
    records: List[Dict[str, object]] = []
    seen = {base_digest}
    maximum_attempts = max(2_000, int(replicates) * 200)

    for attempt in range(maximum_attempts):
        permutation_array = _stratified_derangement(split_labels, rng)
        permutation = torch.as_tensor(
            permutation_array,
            dtype=torch.long,
            device=regulator_tf.device,
        )
        control = replace(
            base_priors,
            regulator_tf_support=regulator_tf.index_select(0, permutation),
            regulator_complex_support=regulator_complex.index_select(
                0, permutation.to(regulator_complex.device)
            ),
        )
        control.validate()
        digest = _prior_digest(control)
        if digest in seen:
            continue

        summary = _end_to_end_summaries(control)
        _assert_row_permuted(base_summary, summary, permutation.cpu())
        tf_audit = _support_audit(regulator_tf, control.regulator_tf_support)
        complex_audit = _support_audit(
            regulator_complex, control.regulator_complex_support
        )
        if not all(
            bool(value)
            for audit in (tf_audit, complex_audit)
            for key, value in audit.items()
            if key != "edges"
        ):
            raise RuntimeError("Matched control changed a support invariant")

        stratum_audits: Dict[str, object] = {}
        for label, indices in stratum_indices.items():
            control_indices = indices.to(regulator_tf.device)
            source_indices = permutation.index_select(0, control_indices)
            if any(
                split_labels[int(index)] != label
                for index in source_indices.detach().cpu().tolist()
            ):
                raise RuntimeError("A matched control crossed a split stratum")
            stratum_tf = _support_audit(
                regulator_tf.index_select(0, control_indices),
                control.regulator_tf_support.index_select(0, control_indices),
            )
            complex_indices = indices.to(regulator_complex.device)
            stratum_complex = _support_audit(
                regulator_complex.index_select(0, complex_indices),
                control.regulator_complex_support.index_select(0, complex_indices),
            )
            if not all(
                bool(value)
                for audit in (stratum_tf, stratum_complex)
                for key, value in audit.items()
                if key != "edges"
            ):
                raise RuntimeError(
                    f"Matched control changed a support invariant in stratum {label}"
                )
            # The global row-permutation assertion above proves exact values.
            # This local check makes the split-specific claim independently
            # auditable rather than inferred from one global flag.
            for name, expected_rows in base_summary.items():
                expected = expected_rows.index_select(
                    0, source_indices.to(expected_rows.device)
                )
                observed = summary[name].index_select(
                    0, control_indices.to(summary[name].device)
                ).to(expected.device)
                if expected.dtype.is_floating_point:
                    matched = torch.allclose(
                        observed, expected, rtol=1e-10, atol=1e-12
                    )
                else:
                    matched = torch.equal(observed, expected)
                if not matched:
                    raise RuntimeError(
                        f"Control changed {name} inside stratum {label}"
                    )
            stratum_audits[label] = {
                "regulators": int(indices.numel()),
                "tf_support": stratum_tf,
                "complex_support": stratum_complex,
                "end_to_end_row_profile_permutation_exact": True,
            }

        seen.add(digest)
        controls.append(control)
        records.append(
            {
                "replicate": len(controls),
                "generation_attempt": attempt + 1,
                "topology_sha256": digest,
                "permutation_sha256": hashlib.sha256(
                    permutation_array.tobytes()
                ).hexdigest(),
                "source_row_for_control_regulator": permutation_array.tolist(),
                "fixed_regulator_labels": int(
                    np.count_nonzero(permutation_array == np.arange(regulators))
                ),
                "tf_support": tf_audit,
                "complex_support": complex_audit,
                "end_to_end_row_profile_permutation_exact": True,
                "split_boundaries_crossed": False,
                "strata": stratum_audits,
            }
        )
        if len(controls) == int(replicates):
            break

    if len(controls) != int(replicates):
        raise RuntimeError(
            f"Only {len(controls)} distinct matched controls could be generated "
            f"after {maximum_attempts} attempts"
        )

    audit: Dict[str, object] = {
        "schema_version": "wld-v5.6-end-to-end-matched-topology-controls",
        "control_kind": "joint_named_regulator_route_profile_permutation",
        "root_seed": int(seed),
        "replicates": int(replicates),
        "minimum_replicates_enforced": MINIMUM_CONTROL_REPLICATES,
        "regulators": regulators,
        "strata": {
            label: int(indices.numel()) for label, indices in stratum_indices.items()
        },
        "base_topology_sha256": base_digest,
        "matching_contract": {
            "joint_tf_and_complex_profile_permutation": True,
            "profiles_permuted_only_within_whole_target_split": True,
            "zero_fixed_regulator_labels": True,
            "support_column_degrees_exact": True,
            "support_row_degree_distributions_exact": True,
            "support_weight_multisets_exact": True,
            "end_to_end_footprint_distribution_exact": True,
            "end_to_end_complex_sign_distribution_exact": True,
            "end_to_end_signed_and_absolute_mass_distributions_exact": True,
            "downstream_evidence_tensors_unchanged": True,
            "test_outcomes_or_observations_read": False,
        },
        "interpretation": (
            "These controls test whether named regulators are assigned to the "
            "correct joint mechanistic route profiles. They do not randomize "
            "the immutable downstream evidence inside each profile."
        ),
        "controls": records,
    }
    return controls, audit


__all__ = [
    "MINIMUM_CONTROL_REPLICATES",
    "build_matched_control_priors",
]
