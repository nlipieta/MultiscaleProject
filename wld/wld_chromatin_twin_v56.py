"""WLD v5.6 null-aware mechanistic chromatin response model.

This module keeps the v5.5 information-flow contract and prior dataclasses but
repairs two important optimization properties:

* each regulator has unit end-to-end *evidence mass* within the TF branch and
  within the protein-complex branch, so graph degree alone cannot set the size
  of a perturbation; and
* both branches pass through bounded global efficacy gates initialized close
  to zero, making persistence a nearby, learnable solution rather than forcing
  a sizeable response at initialization.

The gates scale only evidence-supported routes.  Neural context may still
condition gains and rates on those routes, but it cannot add a response bin or
decode a perturbation directly.  As in v5.5, zero intervention and frozen
removal of both branches return the measured baseline exactly.

This remains an endpoint transient-response parameterization.  It does not
identify physical time, fixed points, basins, attractors, or a synchronized
biological digital twin.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from wld_chromatin_twin_v55 import (
    BranchOverrides,
    ChromatinTwinPriors,
    HybridChromatinField,
    architecture_contract as _v55_architecture_contract,
)
from wld_foundation_model_v4 import WLDMultistudyFoundationModel


__all__ = [
    "BranchOverrides",
    "ChromatinTwinPriors",
    "NullAwareHybridChromatinField",
    "WLDNullAwareChromatinTwin",
    "architecture_contract",
]


def _logit(probability: float) -> float:
    """Return a numerically stable scalar logit for a strict probability."""

    if not math.isfinite(float(probability)) or not 0.0 < float(probability) < 1.0:
        raise ValueError("gate probability must be finite and strictly between zero and one")
    probability = float(probability)
    return math.log(probability) - math.log1p(-probability)


def _nonnegative_weight(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


class NullAwareHybridChromatinField(HybridChromatinField):
    """Sparse dual-route field with normalized evidence and null-aware gates.

    The inherited v5.5 topology, context-conditioned supported-path gains,
    saturation terms, recovery term, and RK4 integration are retained.  Before
    any intervention is routed, each regulator row is divided by its complete
    downstream evidence mass, computed separately for the TF/motif and
    complex/module branches.  Consequently, an intervention contributes one
    unit of evidence mass to a supported branch before learned gains and signs,
    independent of how many TFs, complexes, modules, or bins it can reach.

    ``tf_initial_efficacy`` and ``complex_initial_efficacy`` are fractions of
    ``maximum_efficacy``.  Low-temperature sigmoid logits keep both gates
    differentiable and bounded while amplifying the gradient available near
    zero.  Their default effective value is 1e-3, near persistence but not at
    a dead-gradient boundary.
    """

    def __init__(
        self,
        priors: ChromatinTwinPriors,
        context_dim: int,
        *,
        max_context_gain: float = 1.5,
        tf_initial_efficacy: float = 1e-3,
        complex_initial_efficacy: float = 1e-3,
        maximum_efficacy: float = 1.0,
        gate_temperature: float = 0.1,
    ) -> None:
        super().__init__(priors, context_dim, max_context_gain=max_context_gain)

        maximum_efficacy = float(maximum_efficacy)
        if not math.isfinite(maximum_efficacy) or maximum_efficacy <= 0.0:
            raise ValueError("maximum_efficacy must be finite and positive")
        tf_initial_efficacy = float(tf_initial_efficacy)
        complex_initial_efficacy = float(complex_initial_efficacy)
        for name, value in (
            ("tf_initial_efficacy", tf_initial_efficacy),
            ("complex_initial_efficacy", complex_initial_efficacy),
        ):
            if not math.isfinite(value) or not 0.0 < value < maximum_efficacy:
                raise ValueError(f"{name} must lie strictly between zero and maximum_efficacy")
        self.maximum_efficacy = maximum_efficacy
        gate_temperature = float(gate_temperature)
        if (
            not math.isfinite(gate_temperature)
            or gate_temperature <= 0.0
            or gate_temperature > 1.0
        ):
            raise ValueError("gate_temperature must be finite and in (0,1]")
        self.gate_temperature = gate_temperature

        # Compute complete downstream absolute evidence mass using the same
        # confidence buffers that the inherited sparse router uses.  All
        # allocations inherit the buffers' device and dtype, which keeps model
        # construction and later .to(device) calls CUDA safe.
        tf_downstream_mass = self.regulator_tf_support.new_zeros(self.num_tfs)
        tf_downstream_mass.index_add_(
            0, self.motif_tf_index, self.motif_confidence
        )
        tf_regulator_mass = self.regulator_tf_support @ tf_downstream_mass

        module_downstream_mass = self.regulator_complex_support.new_zeros(
            self.num_modules
        )
        module_downstream_mass.index_add_(
            0, self.mp_module_index, self.mp_confidence
        )
        complex_downstream_mass = self.regulator_complex_support.new_zeros(
            self.num_complexes
        )
        complex_downstream_mass.index_add_(
            0,
            self.cm_complex_index,
            self.cm_confidence
            * module_downstream_mass.index_select(0, self.cm_module_index),
        )
        complex_regulator_mass = (
            self.regulator_complex_support @ complex_downstream_mass
        )

        self.register_buffer(
            "tf_regulator_evidence_mass", tf_regulator_mass.detach().clone()
        )
        self.register_buffer(
            "complex_regulator_evidence_mass",
            complex_regulator_mass.detach().clone(),
        )
        self.register_buffer(
            "tf_downstream_evidence_mass", tf_downstream_mass.detach().clone()
        )
        self.register_buffer(
            "complex_downstream_evidence_mass",
            complex_downstream_mass.detach().clone(),
        )

        # Zero-mass rows are unsupported and remain exactly zero.  Supported
        # rows have end-to-end evidence mass one in their respective branch.
        tf_denominator = tf_regulator_mass.unsqueeze(1).clamp_min(1e-12)
        complex_denominator = complex_regulator_mass.unsqueeze(1).clamp_min(1e-12)
        normalized_tf = torch.where(
            tf_regulator_mass.unsqueeze(1) > 0.0,
            self.regulator_tf_support / tf_denominator,
            torch.zeros_like(self.regulator_tf_support),
        )
        normalized_complex = torch.where(
            complex_regulator_mass.unsqueeze(1) > 0.0,
            self.regulator_complex_support / complex_denominator,
            torch.zeros_like(self.regulator_complex_support),
        )
        self.regulator_tf_support.copy_(normalized_tf)
        self.regulator_complex_support.copy_(normalized_complex)

        tf_fraction = tf_initial_efficacy / maximum_efficacy
        complex_fraction = complex_initial_efficacy / maximum_efficacy
        self.tf_gate_logit = nn.Parameter(
            self.regulator_tf_support.new_tensor(
                gate_temperature * _logit(tf_fraction)
            )
        )
        self.complex_gate_logit = nn.Parameter(
            self.regulator_complex_support.new_tensor(
                gate_temperature * _logit(complex_fraction)
            )
        )

    def effective_branch_gates(self) -> Dict[str, Tensor]:
        """Return differentiable, bounded realized global branch efficacies."""

        maximum = self.tf_gate_logit.new_tensor(self.maximum_efficacy)
        temperature = self.tf_gate_logit.new_tensor(self.gate_temperature)
        return {
            "tf": maximum * torch.sigmoid(self.tf_gate_logit / temperature),
            "complex": maximum
            * torch.sigmoid(self.complex_gate_logit / temperature),
        }

    def gate_diagnostics(self) -> Dict[str, Tensor]:
        """Expose gate values, logits, and temperature without detaching them."""

        gates = self.effective_branch_gates()
        return {
            "tf": gates["tf"],
            "complex": gates["complex"],
            "tf_logit": self.tf_gate_logit,
            "complex_logit": self.complex_gate_logit,
            "temperature": self.tf_gate_logit.new_tensor(self.gate_temperature),
            "maximum": self.tf_gate_logit.new_tensor(self.maximum_efficacy),
        }

    def evidence_mass_diagnostics(self) -> Dict[str, Tensor]:
        """Return pre-normalization and normalized end-to-end evidence masses."""

        tf_normalized = (
            self.regulator_tf_support @ self.tf_downstream_evidence_mass
        )
        complex_normalized = (
            self.regulator_complex_support @ self.complex_downstream_evidence_mass
        )
        return {
            "tf_pre_normalization": self.tf_regulator_evidence_mass,
            "complex_pre_normalization": self.complex_regulator_evidence_mass,
            "tf_normalized": tf_normalized,
            "complex_normalized": complex_normalized,
        }

    def components(
        self,
        accessibility: Tensor,
        baseline: Tensor,
        intervention: Tensor,
        context: Tensor,
        *,
        overrides: Optional[BranchOverrides] = None,
    ) -> Dict[str, Tensor]:
        """Evaluate gated branch drives and the resulting chromatin derivative.

        Additional tensors are returned without detaching so callers can audit
        realized branch gates and drive magnitudes or regularize them during
        fitting.  No diagnostic tensor participates in a response bypass.
        """

        values = super().components(
            accessibility,
            baseline,
            intervention,
            context,
            overrides=overrides,
        )
        gates = self.effective_branch_gates()
        tf_pre_gate = values["tf_peak_drive"]
        complex_pre_gate = values["complex_peak_drive"]
        tf_drive = gates["tf"] * tf_pre_gate
        complex_drive = gates["complex"] * complex_pre_gate
        peak_drive = tf_drive + complex_drive

        current = accessibility.clamp(0.0, 1.0)
        reference = baseline.clamp(0.0, 1.0)
        opening = self.open_rate(context) * (1.0 - current) * F.relu(peak_drive)
        closing = self.close_rate(context) * current * F.relu(-peak_drive)
        recovery = self.recovery_rate(context) * (reference - current)
        derivative = opening - closing + recovery

        resolved = overrides or BranchOverrides()
        resolved.validate()
        values.update(
            derivative=derivative,
            tf_peak_drive_pre_gate=tf_pre_gate,
            complex_peak_drive_pre_gate=complex_pre_gate,
            tf_peak_drive=tf_drive,
            complex_peak_drive=complex_drive,
            peak_drive=peak_drive,
            opening=opening,
            closing=closing,
            recovery=recovery,
            effective_tf_gate=gates["tf"],
            effective_complex_gate=gates["complex"],
            gate_temperature=gates["tf"].new_tensor(self.gate_temperature),
            realized_tf_gate=gates["tf"] * float(resolved.tf_scale),
            realized_complex_gate=gates["complex"]
            * float(resolved.complex_scale),
            tf_pre_gate_drive_l1=tf_pre_gate.abs().mean(dim=1),
            complex_pre_gate_drive_l1=complex_pre_gate.abs().mean(dim=1),
            tf_realized_drive_l1=tf_drive.abs().mean(dim=1),
            complex_realized_drive_l1=complex_drive.abs().mean(dim=1),
            total_realized_drive_l1=peak_drive.abs().mean(dim=1),
        )
        return values


class WLDNullAwareChromatinTwin(nn.Module):
    """v5.5-compatible wrapper around the null-aware v5.6 field.

    The measured control ATAC vector is encoded before intervention identity is
    supplied to the field.  The forward signature and principal output keys
    match :class:`wld_chromatin_twin_v55.WLDChromatinDigitalTwin`.
    """

    def __init__(
        self,
        foundation: WLDMultistudyFoundationModel,
        priors: ChromatinTwinPriors,
        *,
        tf_initial_efficacy: float = 1e-3,
        complex_initial_efficacy: float = 1e-3,
        maximum_efficacy: float = 1.0,
        max_context_gain: float = 1.5,
        gate_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        priors.validate()
        if foundation.num_tfs != priors.num_tfs:
            raise ValueError("foundation and v5.6 priors disagree on TFs")
        if foundation.num_peaks != priors.num_foundation_peaks:
            raise ValueError("foundation encoder anchors and v5.6 index disagree")
        self.foundation = foundation
        self.register_buffer(
            "foundation_peak_index",
            torch.as_tensor(priors.foundation_peak_index).long(),
        )
        self.field = NullAwareHybridChromatinField(
            priors,
            foundation.context_dim,
            max_context_gain=max_context_gain,
            tf_initial_efficacy=tf_initial_efficacy,
            complex_initial_efficacy=complex_initial_efficacy,
            maximum_efficacy=maximum_efficacy,
            gate_temperature=gate_temperature,
        )

    def encode_control(
        self,
        response_atac: Tensor,
        *,
        cues: Optional[Tensor] = None,
        rna: Optional[Tensor] = None,
        protein: Optional[Tensor] = None,
        metabolic: Optional[Tensor] = None,
        modality_masks: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Encode measured baseline biology without intervention identity."""

        if (
            not isinstance(response_atac, Tensor)
            or response_atac.ndim != 2
            or response_atac.shape[1] != self.field.num_peaks
        ):
            raise ValueError("response ATAC has the wrong shape")
        if not torch.isfinite(response_atac).all():
            raise ValueError("response ATAC must contain only finite values")
        if bool(((response_atac < 0.0) | (response_atac > 1.0)).any()):
            raise ValueError("response ATAC must be within [0,1]")
        anchor_atac = response_atac.index_select(1, self.foundation_peak_index)
        if cues is None:
            cues = response_atac.new_zeros(
                (response_atac.shape[0], self.foundation.priors.num_cues)
            )
        encoded = self.foundation.encoder(
            cues=cues,
            atac=anchor_atac,
            rna=rna,
            protein=protein,
            metabolic=metabolic,
            modality_masks=modality_masks,
        )
        context = self.foundation.context_network(encoded["biological_context"])
        return {
            "encoded": encoded,
            "context": context,
            "foundation_atac": anchor_atac,
        }

    def forward(
        self,
        response_atac: Tensor,
        intervention: Tensor,
        *,
        cues: Optional[Tensor] = None,
        rna: Optional[Tensor] = None,
        protein: Optional[Tensor] = None,
        metabolic: Optional[Tensor] = None,
        modality_masks: Optional[Mapping[str, Tensor]] = None,
        horizon: float = 1.0,
        steps: int = 6,
        overrides: Optional[BranchOverrides] = None,
    ) -> Dict[str, Tensor]:
        values = self.encode_control(
            response_atac,
            cues=cues,
            rna=rna,
            protein=protein,
            metabolic=metabolic,
            modality_masks=modality_masks,
        )
        output = self.field.integrate(
            response_atac,
            intervention,
            values["context"],
            horizon=horizon,
            steps=steps,
            overrides=overrides,
        )
        output.update(
            context=values["context"],
            initial_tf=values["encoded"]["tf"],
            foundation_atac=values["foundation_atac"],
        )
        return output

    def realized_regularization(
        self,
        output: Mapping[str, Tensor],
        baseline: Tensor,
        gate_weight: float = 1.0,
        delta_weight: float = 1.0,
    ) -> Tensor:
        """Penalize realized branch efficacy and endpoint displacement.

        Regularization is applied to the bounded effective gates and to the
        actual predicted change, never to inverse-link/raw parameters.  This
        avoids the v5.5 failure mode in which weight decay toward a raw value of
        zero increased a softplus-transformed gain.  The returned scalar remains
        differentiable and may be added directly to a training objective.
        """

        gate_weight = _nonnegative_weight("gate_weight", gate_weight)
        delta_weight = _nonnegative_weight("delta_weight", delta_weight)
        if not isinstance(output, Mapping) or "atac_t" not in output:
            raise ValueError("output must contain the model's atac_t tensor")
        prediction = output["atac_t"]
        if (
            not isinstance(prediction, Tensor)
            or not isinstance(baseline, Tensor)
            or prediction.shape != baseline.shape
        ):
            raise ValueError("output atac_t and baseline must have the same tensor shape")
        if not torch.isfinite(prediction).all() or not torch.isfinite(baseline).all():
            raise ValueError("regularization tensors must contain only finite values")
        if prediction.device != baseline.device:
            raise ValueError("output atac_t and baseline must be on the same device")

        fallback_gates = self.field.effective_branch_gates()
        realized_tf_gate = output.get("realized_tf_gate", fallback_gates["tf"])
        realized_complex_gate = output.get(
            "realized_complex_gate", fallback_gates["complex"]
        )
        if (
            not isinstance(realized_tf_gate, Tensor)
            or not isinstance(realized_complex_gate, Tensor)
            or realized_tf_gate.numel() != 1
            or realized_complex_gate.numel() != 1
            or not torch.isfinite(realized_tf_gate).all()
            or not torch.isfinite(realized_complex_gate).all()
        ):
            raise ValueError("output contains invalid realized branch gates")
        gate_penalty = torch.stack(
            (realized_tf_gate.reshape(()), realized_complex_gate.reshape(()))
        ).mean()
        delta_penalty = (prediction - baseline).abs().mean()
        return gate_weight * gate_penalty + delta_weight * delta_penalty


def architecture_contract(model: WLDNullAwareChromatinTwin) -> Dict[str, object]:
    """Extend the v5.5 no-bypass contract with v5.6 null-aware guarantees."""

    if not isinstance(model, WLDNullAwareChromatinTwin):
        raise TypeError("v5.6 architecture contract requires WLDNullAwareChromatinTwin")
    contract = dict(_v55_architecture_contract(model))
    contract.update(
        schema_version="wld-v5.6-null-aware-chromatin-twin",
        branch_efficacy_bounded=True,
        branch_efficacy_initialization="near-persistence",
        branch_efficacy_gate_temperature=model.field.gate_temperature,
        regulator_evidence_mass_normalized_separately_by_branch=True,
        realized_gate_regularization=True,
        realized_endpoint_delta_regularization=True,
        raw_inverse_link_regularization_required=False,
        neural_response_bypass=False,
    )
    return contract
