"""WLD v5.5: a leakage-safe mechanistic chromatin digital-model prototype.

The model combines a broadly pretrained, measured-state encoder with two
explicit perturbation routes:

    regulator -> TF/signalling route -> motif-localized genomic bins
    regulator -> curated complex -> empirical accessibility module -> bins

The graph is evidence.  Cell/context-dependent activities, gains and rates are
learned, but a neural context vector cannot add an accessibility delta.  A
zero intervention, or removal of both mechanistic branches, therefore returns
the measured control state exactly.

This is intentionally called a digital-model / digital-twin *prototype*.  It
does not implement continuous synchronization to one physical counterpart and
single-endpoint perturbations cannot identify attractors or kinetic time scale.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from dataclasses import dataclass, fields
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from wld_foundation_model_v4 import WLDMultistudyFoundationModel


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus initialization must be positive")
    return math.log(math.expm1(value))


def _as_float(value: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _validate_support(name: str, value: Tensor, *, signed: bool = False) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-two tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if not signed and bool((value < 0).any()):
        raise ValueError(f"{name} must be non-negative")


def _validate_unit_interval_matrix(name: str, value: Tensor, *, columns: int) -> None:
    """Reject an invalid measured state instead of silently projecting it.

    Validation intentionally covers every response bin, including bins that
    are not selected as foundation-encoder anchors.  This keeps persistence a
    literal identity operation on the measured input rather than persistence
    of a clamped surrogate.
    """

    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{name} has the wrong shape")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError(f"{name} must be within [0,1]")


@dataclass(frozen=True)
class ChromatinTwinPriors:
    """Immutable topology and compiler-estimated effects for WLD v5.5.

    Signed complex/module tensors are fixed empirical effects estimated only
    from training targets.  Their nonzero pattern is the allowed topology.
    The TF motif tensor is non-negative localization evidence; its direction is
    learned from training perturbations because motif presence alone does not
    establish whether accessibility opens or closes.
    """

    regulator_tf_support: Tensor
    tf_peak_motif: Tensor
    regulator_complex_support: Tensor
    complex_module_effect: Tensor
    module_peak_loading: Tensor
    foundation_peak_index: Tensor

    def validate(self) -> None:
        _validate_support("regulator_tf_support", self.regulator_tf_support)
        _validate_support("tf_peak_motif", self.tf_peak_motif)
        _validate_support("regulator_complex_support", self.regulator_complex_support)
        _validate_support("complex_module_effect", self.complex_module_effect, signed=True)
        _validate_support("module_peak_loading", self.module_peak_loading, signed=True)

        regulator_tf = self.regulator_tf_support
        tf_peak = self.tf_peak_motif
        regulator_complex = self.regulator_complex_support
        complex_module = self.complex_module_effect
        module_peak = self.module_peak_loading
        if regulator_tf.shape[0] != regulator_complex.shape[0]:
            raise ValueError("TF and complex routes disagree on regulator count")
        if regulator_tf.shape[1] != tf_peak.shape[0]:
            raise ValueError("regulator routes and motifs disagree on TF count")
        if regulator_complex.shape[1] != complex_module.shape[0]:
            raise ValueError("complex route and complex-module effects disagree")
        if complex_module.shape[1] != module_peak.shape[0]:
            raise ValueError("complex-module and module-peak tensors disagree")
        if tf_peak.shape[1] != module_peak.shape[1]:
            raise ValueError("TF and complex branches disagree on response bins")
        if regulator_tf.shape[0] < 3 or tf_peak.shape[1] < 2:
            raise ValueError("priors are too small for grouped perturbation evaluation")
        if not int(torch.count_nonzero(regulator_tf)):
            raise ValueError("TF route contains no evidence")
        if not int(torch.count_nonzero(tf_peak)):
            raise ValueError("motif route contains no evidence")
        if not int(torch.count_nonzero(regulator_complex)):
            raise ValueError("complex membership contains no evidence")
        if not int(torch.count_nonzero(complex_module)):
            raise ValueError("complex-module atlas contains no training evidence")
        if not int(torch.count_nonzero(module_peak)):
            raise ValueError("module atlas contains no genomic-bin loading")

        index = torch.as_tensor(self.foundation_peak_index)
        if index.ndim != 1 or index.numel() < 1:
            raise ValueError("foundation_peak_index must be a nonempty vector")
        if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("foundation_peak_index must contain integers")
        if bool((index < 0).any()) or bool((index >= self.num_peaks).any()):
            raise ValueError("foundation_peak_index contains an out-of-range bin")
        if int(torch.unique(index).numel()) != int(index.numel()):
            raise ValueError("foundation_peak_index contains duplicates")

        # An empirical complex->module edge is useful only if that module has
        # at least one bin.  Empty routes indicate a corrupt compiler artifact.
        module_has_peak = torch.count_nonzero(module_peak, dim=1) > 0
        used_modules = torch.count_nonzero(complex_module, dim=0) > 0
        if bool((used_modules & ~module_has_peak).any()):
            raise ValueError("a complex points to an empty accessibility module")

    @property
    def num_regulators(self) -> int:
        return int(self.regulator_tf_support.shape[0])

    @property
    def num_tfs(self) -> int:
        return int(self.regulator_tf_support.shape[1])

    @property
    def num_complexes(self) -> int:
        return int(self.regulator_complex_support.shape[1])

    @property
    def num_modules(self) -> int:
        return int(self.module_peak_loading.shape[0])

    @property
    def num_peaks(self) -> int:
        return int(self.tf_peak_motif.shape[1])

    @property
    def num_foundation_peaks(self) -> int:
        return int(torch.as_tensor(self.foundation_peak_index).numel())


@dataclass(frozen=True)
class BranchOverrides:
    """Frozen-evaluation kill switches; never a route-creation API."""

    tf_scale: float = 1.0
    complex_scale: float = 1.0
    regulator_tf_support: Optional[Tensor] = None
    regulator_complex_support: Optional[Tensor] = None

    def validate(self) -> None:
        for name, value in (("tf_scale", self.tf_scale), ("complex_scale", self.complex_scale)):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")


class LowRankContextRate(nn.Module):
    """Positive per-bin rate with low-rank, bounded context variation.

    Context is projected onto compiler-supplied mechanism modules rather than
    through a dense context-to-bin response decoder.  These rates only
    multiply mechanistic drive or recovery toward the initial measurement.
    """

    def __init__(
        self,
        context_dim: int,
        basis: Tensor,
        *,
        initial: float,
        max_log_delta: float = 1.25,
        maximum: float = 20.0,
    ) -> None:
        super().__init__()
        basis = _as_float(basis)
        if basis.ndim != 2 or not torch.isfinite(basis).all() or bool((basis < 0).any()):
            raise ValueError("rate basis must be a finite non-negative matrix")
        if basis.shape[0] < 1 or basis.shape[1] < 1:
            raise ValueError("rate basis cannot be empty")
        basis = basis / basis.sum(dim=0, keepdim=True).clamp_min(1.0)
        self.register_buffer("basis", basis)
        self.raw_base = nn.Parameter(torch.full((basis.shape[1],), _inverse_softplus(initial)))
        self.context_adapter = nn.Linear(context_dim, basis.shape[0], bias=False)
        nn.init.zeros_(self.context_adapter.weight)
        self.max_log_delta = float(max_log_delta)
        self.maximum = float(maximum)

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_adapter.in_features:
            raise ValueError("context has the wrong shape")
        module_delta = torch.tanh(self.context_adapter(context))
        peak_delta = module_delta @ self.basis
        base = F.softplus(self.raw_base).unsqueeze(0)
        return (base * torch.exp(self.max_log_delta * peak_delta)).clamp(1e-6, self.maximum)


class HybridChromatinField(nn.Module):
    """Sparse dual-route accessibility vector field."""

    def __init__(
        self,
        priors: ChromatinTwinPriors,
        context_dim: int,
        *,
        max_context_gain: float = 1.5,
    ) -> None:
        super().__init__()
        priors.validate()
        self.num_regulators = priors.num_regulators
        self.num_tfs = priors.num_tfs
        self.num_complexes = priors.num_complexes
        self.num_modules = priors.num_modules
        self.num_peaks = priors.num_peaks
        self.context_dim = int(context_dim)
        self.max_context_gain = float(max_context_gain)

        regulator_tf = _as_float(priors.regulator_tf_support)
        regulator_tf = regulator_tf / regulator_tf.amax(dim=1, keepdim=True).clamp_min(1.0)
        regulator_complex = _as_float(priors.regulator_complex_support)
        regulator_complex = regulator_complex / regulator_complex.amax(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("regulator_tf_support", regulator_tf)
        self.register_buffer("regulator_complex_support", regulator_complex)

        motif = _as_float(priors.tf_peak_motif)
        motif_edges = torch.nonzero(motif > 0, as_tuple=False)
        motif_confidence = motif[motif_edges[:, 0], motif_edges[:, 1]]
        motif_confidence = motif_confidence / motif_confidence.amax().clamp_min(1.0)
        self.register_buffer("motif_tf_index", motif_edges[:, 0].long())
        self.register_buffer("motif_peak_index", motif_edges[:, 1].long())
        self.register_buffer("motif_confidence", motif_confidence)

        complex_module = _as_float(priors.complex_module_effect)
        cm_edges = torch.nonzero(complex_module != 0, as_tuple=False)
        cm_values = complex_module[cm_edges[:, 0], cm_edges[:, 1]]
        cm_confidence = cm_values.abs() / cm_values.abs().amax().clamp_min(1.0)
        self.register_buffer("cm_complex_index", cm_edges[:, 0].long())
        self.register_buffer("cm_module_index", cm_edges[:, 1].long())
        self.register_buffer("cm_sign", torch.sign(cm_values))
        self.register_buffer("cm_confidence", cm_confidence)

        module_peak = _as_float(priors.module_peak_loading)
        mp_edges = torch.nonzero(module_peak != 0, as_tuple=False)
        mp_values = module_peak[mp_edges[:, 0], mp_edges[:, 1]]
        # Normalize within module so a large module is not automatically a
        # stronger mechanism solely because it contains more bins.
        denominators = module_peak.abs().amax(dim=1).clamp_min(1e-8)
        mp_confidence = mp_values.abs() / denominators[mp_edges[:, 0]]
        self.register_buffer("mp_module_index", mp_edges[:, 0].long())
        self.register_buffer("mp_peak_index", mp_edges[:, 1].long())
        self.register_buffer("mp_sign", torch.sign(mp_values))
        self.register_buffer("mp_confidence", mp_confidence)

        # No parameter exists for an unsupported edge.  TF direction is
        # learned because motif localization alone is unsigned.  Complex and
        # module directions are compiler-fixed train-only empirical evidence.
        self.raw_tf_direction = nn.Parameter(torch.full((self.num_tfs,), 0.35))
        self.raw_tf_gain = nn.Parameter(torch.full((self.num_tfs,), _inverse_softplus(0.40)))
        self.raw_motif_direction = nn.Parameter(torch.full((motif_edges.shape[0],), 0.35))
        self.raw_motif_gain = nn.Parameter(torch.full((motif_edges.shape[0],), _inverse_softplus(0.35)))
        self.raw_complex_gain = nn.Parameter(torch.full((self.num_complexes,), _inverse_softplus(0.40)))
        self.raw_cm_gain = nn.Parameter(torch.full((cm_edges.shape[0],), _inverse_softplus(0.50)))
        self.raw_module_gain = nn.Parameter(torch.full((self.num_modules,), _inverse_softplus(0.50)))

        self.tf_context_gain = nn.Linear(context_dim, self.num_tfs, bias=False)
        self.complex_context_gain = nn.Linear(context_dim, self.num_complexes, bias=False)
        self.module_context_gain = nn.Linear(context_dim, self.num_modules, bias=False)
        nn.init.zeros_(self.tf_context_gain.weight)
        nn.init.zeros_(self.complex_context_gain.weight)
        nn.init.zeros_(self.module_context_gain.weight)

        # A constant channel covers TF-only peaks; module channels carry the
        # mechanistic low-rank structure for complex-linked bins.
        rate_basis = torch.cat(
            [module_peak.new_ones((1, self.num_peaks)), module_peak.abs()], dim=0
        )
        self.open_rate = LowRankContextRate(context_dim, rate_basis, initial=0.45)
        self.close_rate = LowRankContextRate(context_dim, rate_basis, initial=0.45)
        self.recovery_rate = LowRankContextRate(context_dim, rate_basis, initial=0.03)

    @staticmethod
    def _bounded_context_gain(layer: nn.Linear, context: Tensor, maximum: float) -> Tensor:
        return torch.exp(maximum * torch.tanh(layer(context)))

    @staticmethod
    def _resolve_support(base: Tensor, override: Optional[Tensor], reference: Tensor, name: str) -> Tensor:
        if override is None:
            return base
        value = torch.as_tensor(override, dtype=reference.dtype, device=reference.device)
        if value.shape != base.shape or not torch.isfinite(value).all() or bool((value < 0).any()):
            raise ValueError(f"{name} override has invalid values or shape")
        # Frozen ablations may remove/attenuate evidence, never invent an edge.
        if bool(((value != 0) & (base == 0)).any()):
            raise ValueError(f"{name} override creates unsupported edges")
        if bool((value > base + 1e-7).any()):
            raise ValueError(f"{name} override amplifies frozen evidence")
        return value

    def reachability(self) -> Dict[str, Tensor]:
        tf = (self.regulator_tf_support > 0).float() @ self._dense_motif_mask()
        cm = (self.regulator_complex_support > 0).float() @ self._dense_complex_module_mask()
        complex_route = (cm > 0).float() @ self._dense_module_peak_mask()
        return {"tf": tf > 0, "complex": complex_route > 0, "total": (tf > 0) | (complex_route > 0)}

    def _dense_motif_mask(self) -> Tensor:
        value = self.regulator_tf_support.new_zeros((self.num_tfs, self.num_peaks))
        value[self.motif_tf_index, self.motif_peak_index] = 1.0
        return value

    def _dense_complex_module_mask(self) -> Tensor:
        value = self.regulator_complex_support.new_zeros((self.num_complexes, self.num_modules))
        value[self.cm_complex_index, self.cm_module_index] = 1.0
        return value

    def _dense_module_peak_mask(self) -> Tensor:
        value = self.regulator_complex_support.new_zeros((self.num_modules, self.num_peaks))
        value[self.mp_module_index, self.mp_peak_index] = 1.0
        return value

    def components(
        self,
        accessibility: Tensor,
        baseline: Tensor,
        intervention: Tensor,
        context: Tensor,
        *,
        overrides: Optional[BranchOverrides] = None,
    ) -> Dict[str, Tensor]:
        if accessibility.shape != (intervention.shape[0], self.num_peaks):
            raise ValueError("accessibility has the wrong shape")
        if baseline.shape != accessibility.shape:
            raise ValueError("baseline has the wrong shape")
        if intervention.shape[1:] != (self.num_regulators,):
            raise ValueError("intervention must have shape [batch, regulators]")
        if context.shape != (intervention.shape[0], self.context_dim):
            raise ValueError("context has the wrong shape")
        if not torch.isfinite(intervention).all() or bool((intervention < 0).any()):
            raise ValueError("intervention must be finite and non-negative")
        resolved = overrides or BranchOverrides()
        resolved.validate()
        regulator_tf = self._resolve_support(
            self.regulator_tf_support, resolved.regulator_tf_support, intervention, "TF support"
        )
        regulator_complex = self._resolve_support(
            self.regulator_complex_support,
            resolved.regulator_complex_support,
            intervention,
            "complex support",
        )

        tf_route = intervention @ regulator_tf
        tf_message = (
            tf_route
            * torch.tanh(self.raw_tf_direction).unsqueeze(0)
            * F.softplus(self.raw_tf_gain).unsqueeze(0)
            * self._bounded_context_gain(self.tf_context_gain, context, self.max_context_gain)
        )
        motif_message = (
            tf_message[:, self.motif_tf_index]
            * torch.tanh(self.raw_motif_direction).unsqueeze(0)
            * F.softplus(self.raw_motif_gain).unsqueeze(0)
            * self.motif_confidence.unsqueeze(0)
        )
        tf_peak_drive = accessibility.new_zeros((accessibility.shape[0], self.num_peaks))
        tf_peak_drive.index_add_(1, self.motif_peak_index, motif_message)
        tf_peak_drive = tf_peak_drive * float(resolved.tf_scale)

        complex_route = intervention @ regulator_complex
        complex_message = (
            complex_route
            * F.softplus(self.raw_complex_gain).unsqueeze(0)
            * self._bounded_context_gain(self.complex_context_gain, context, self.max_context_gain)
        )
        cm_message = (
            complex_message[:, self.cm_complex_index]
            * self.cm_sign.unsqueeze(0)
            * self.cm_confidence.unsqueeze(0)
            * F.softplus(self.raw_cm_gain).unsqueeze(0)
        )
        module_message = accessibility.new_zeros((accessibility.shape[0], self.num_modules))
        module_message.index_add_(1, self.cm_module_index, cm_message)
        module_message = (
            module_message
            * F.softplus(self.raw_module_gain).unsqueeze(0)
            * self._bounded_context_gain(self.module_context_gain, context, self.max_context_gain)
        )
        mp_message = (
            module_message[:, self.mp_module_index]
            * self.mp_sign.unsqueeze(0)
            * self.mp_confidence.unsqueeze(0)
        )
        complex_peak_drive = accessibility.new_zeros((accessibility.shape[0], self.num_peaks))
        complex_peak_drive.index_add_(1, self.mp_peak_index, mp_message)
        complex_peak_drive = complex_peak_drive * float(resolved.complex_scale)

        peak_drive = tf_peak_drive + complex_peak_drive
        current = accessibility.clamp(0.0, 1.0)
        reference = baseline.clamp(0.0, 1.0)
        opening = self.open_rate(context) * (1.0 - current) * F.relu(peak_drive)
        closing = self.close_rate(context) * current * F.relu(-peak_drive)
        recovery = self.recovery_rate(context) * (reference - current)
        derivative = opening - closing + recovery
        return {
            "derivative": derivative,
            "tf_route": tf_route,
            "tf_message": tf_message,
            "tf_peak_drive": tf_peak_drive,
            "complex_route": complex_route,
            "complex_message": complex_message,
            "module_message": module_message,
            "complex_peak_drive": complex_peak_drive,
            "peak_drive": peak_drive,
            "opening": opening,
            "closing": closing,
            "recovery": recovery,
        }

    def integrate(
        self,
        baseline: Tensor,
        intervention: Tensor,
        context: Tensor,
        *,
        horizon: float = 1.0,
        steps: int = 6,
        overrides: Optional[BranchOverrides] = None,
    ) -> Dict[str, Tensor]:
        _validate_unit_interval_matrix(
            "baseline response ATAC", baseline, columns=self.num_peaks
        )
        if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(float(horizon)) or float(horizon) <= 0:
            raise ValueError("horizon must be finite and positive")
        # The measured baseline has already been validated.  Starting from the
        # tensor itself is what makes a zero drive an exact persistence model.
        state = baseline
        dt = float(horizon) / int(steps)

        def derivative(value: Tensor) -> Tensor:
            return self.components(value, baseline, intervention, context, overrides=overrides)["derivative"]

        for _ in range(int(steps)):
            k1 = derivative(state)
            k2 = derivative((state + 0.5 * dt * k1).clamp(0.0, 1.0))
            k3 = derivative((state + 0.5 * dt * k2).clamp(0.0, 1.0))
            k4 = derivative((state + dt * k3).clamp(0.0, 1.0))
            state = (state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0).clamp(0.0, 1.0)
        final_components = self.components(state, baseline, intervention, context, overrides=overrides)
        return {"atac_t": state, "initial_atac": baseline, **final_components}


class WLDChromatinDigitalTwin(nn.Module):
    """Population foundation plus personalized dual-route chromatin dynamics."""

    def __init__(self, foundation: WLDMultistudyFoundationModel, priors: ChromatinTwinPriors) -> None:
        super().__init__()
        priors.validate()
        if foundation.num_tfs != priors.num_tfs:
            raise ValueError("foundation and v5.5 priors disagree on TFs")
        if foundation.num_peaks != priors.num_foundation_peaks:
            raise ValueError("foundation encoder anchors and v5.5 index disagree")
        self.foundation = foundation
        self.register_buffer("foundation_peak_index", torch.as_tensor(priors.foundation_peak_index).long())
        self.field = HybridChromatinField(priors, foundation.context_dim)

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
        _validate_unit_interval_matrix(
            "response ATAC", response_atac, columns=self.field.num_peaks
        )
        anchor_atac = response_atac.index_select(1, self.foundation_peak_index)
        if cues is None:
            cues = response_atac.new_zeros((response_atac.shape[0], self.foundation.priors.num_cues))
        encoded = self.foundation.encoder(
            cues=cues,
            atac=anchor_atac,
            rna=rna,
            protein=protein,
            metabolic=metabolic,
            modality_masks=modality_masks,
        )
        context = self.foundation.context_network(encoded["biological_context"])
        return {"encoded": encoded, "context": context, "foundation_atac": anchor_atac}

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


def degree_preserving_bipartite_shuffle(
    support: Tensor,
    *,
    seed: int,
    swaps_per_edge: int = 20,
) -> Tensor:
    """Shuffle a bipartite topology while preserving both degree sequences."""

    values = _as_float(support)
    if values.ndim != 2 or not torch.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("support must be a finite non-negative matrix")
    edges = [tuple(map(int, pair)) for pair in np.argwhere((values > 0).cpu().numpy())]
    if len(edges) < 2:
        raise ValueError("at least two edges are required")
    edge_set = set(edges)
    rng = np.random.default_rng(seed)
    changes = 0
    for _ in range(max(1, len(edges) * int(swaps_per_edge))):
        first, second = map(int, rng.choice(len(edges), size=2, replace=False))
        r1, c1 = edges[first]
        r2, c2 = edges[second]
        if r1 == r2 or c1 == c2 or (r1, c2) in edge_set or (r2, c1) in edge_set:
            continue
        edge_set.remove((r1, c1)); edge_set.remove((r2, c2))
        edge_set.add((r1, c2)); edge_set.add((r2, c1))
        edges[first], edges[second] = (r1, c2), (r2, c1)
        changes += 1
    if not changes:
        raise RuntimeError("could not produce a degree-preserving shuffle")
    result = torch.zeros_like(values)
    confidences = values[values > 0].cpu().numpy().copy()
    rng.shuffle(confidences)
    for edge, confidence in zip(sorted(edge_set), confidences):
        result[edge] = float(confidence)
    if not torch.equal((values > 0).sum(1), (result > 0).sum(1)):
        raise RuntimeError("shuffle changed row degrees")
    if not torch.equal((values > 0).sum(0), (result > 0).sum(0)):
        raise RuntimeError("shuffle changed column degrees")
    if torch.equal(values > 0, result > 0):
        raise RuntimeError("degree-preserving shuffle returned the original topology")
    return result


def topology_digest(priors: ChromatinTwinPriors) -> str:
    priors.validate()
    digest = hashlib.sha256()
    for field in fields(priors):
        value = torch.as_tensor(getattr(priors, field.name)).detach().cpu().contiguous().numpy()
        digest.update(field.name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def architecture_contract(model: WLDChromatinDigitalTwin) -> Dict[str, object]:
    parameter_names = tuple(name.lower() for name, _ in model.named_parameters())
    forbidden = ("target_embedding", "guide_embedding", "direct_peak_decoder", "residual_peak_decoder")
    found = sorted(token for token in forbidden if any(token in name for name in parameter_names))
    if found:
        raise RuntimeError(f"Forbidden bypass parameters found: {found}")
    return {
        "named_regulator_nodes": model.field.num_regulators,
        "named_tf_nodes": model.field.num_tfs,
        "named_complex_nodes": model.field.num_complexes,
        "training_derived_accessibility_modules": model.field.num_modules,
        "response_bins": model.field.num_peaks,
        "foundation_encoder_anchor_bins": int(model.foundation_peak_index.numel()),
        "regulator_tf_edges": int(torch.count_nonzero(model.field.regulator_tf_support)),
        "tf_motif_bin_edges": int(model.field.motif_tf_index.numel()),
        "regulator_complex_edges": int(torch.count_nonzero(model.field.regulator_complex_support)),
        "complex_module_edges": int(model.field.cm_complex_index.numel()),
        "module_bin_edges": int(model.field.mp_module_index.numel()),
        "guide_or_target_in_encoder": False,
        "direct_context_to_peak_delta_decoder": False,
        "unsupported_edges_trainable": False,
        "context_conditions_supported_paths_only": True,
        "independent_tf_and_complex_kill_switches": True,
        "zero_all_routes_returns_persistence": True,
        "cell_context_parameters_variable": True,
        "claim_scope": "transient chromatin intervention digital-model prototype",
        "continuous_physical_twin_synchronization": False,
        "attractor_claim": False,
    }
