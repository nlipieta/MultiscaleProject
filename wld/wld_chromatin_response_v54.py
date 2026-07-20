"""WLD v5.4: perturbation-supervised chromatin response dynamics.

This module is deliberately narrower than an attractor model.  It predicts the
unpaired accessibility distribution produced by a named CRISPR intervention
from control-cell accessibility.  Information can reach a peak only through
an externally supported regulator-to-TF route and a localized TF motif.  The
intervention identity selects a named node *after* encoding and is never an
encoder feature.

Topology is fixed evidence.  Cell-dependent TF activity, supported-edge gains,
opening/closing rates and recovery rates remain context conditioned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

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


@dataclass(frozen=True)
class ChromatinRoutePriors:
    """Hard evidence topology for the chromatin intervention layer."""

    regulator_tf_support: Tensor
    tf_peak_motif: Tensor

    def validate(self) -> None:
        route = torch.as_tensor(self.regulator_tf_support)
        motif = torch.as_tensor(self.tf_peak_motif)
        if route.ndim != 2 or motif.ndim != 2:
            raise ValueError("route and motif priors must be rank-two")
        if route.shape[1] != motif.shape[0]:
            raise ValueError("regulator routes and TF motifs disagree on TF count")
        if route.shape[0] < 3 or motif.shape[1] < 2:
            raise ValueError("route prior is too small for grouped perturbation training")
        if not torch.isfinite(route).all() or not torch.isfinite(motif).all():
            raise ValueError("route priors contain non-finite values")
        if bool((route < 0).any()) or bool((motif < 0).any()):
            raise ValueError("route topology/confidence must be non-negative")
        if not int(torch.count_nonzero(route)) or not int(torch.count_nonzero(motif)):
            raise ValueError("route and motif priors must each contain evidence")

    @property
    def num_regulators(self) -> int:
        return int(self.regulator_tf_support.shape[0])

    @property
    def num_tfs(self) -> int:
        return int(self.regulator_tf_support.shape[1])

    @property
    def num_peaks(self) -> int:
        return int(self.tf_peak_motif.shape[1])


class ContextRate(nn.Module):
    """Positive rate with bounded cell-context variation."""

    def __init__(
        self,
        context_dim: int,
        output_dim: int,
        initial: float,
        max_log_delta: float = 1.5,
    ) -> None:
        super().__init__()
        self.max_log_delta = float(max_log_delta)
        self.raw_base = nn.Parameter(
            torch.full((output_dim,), _inverse_softplus(initial))
        )
        self.adapter = nn.Linear(context_dim, output_dim, bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, context: Tensor) -> Tensor:
        base = F.softplus(self.raw_base).unsqueeze(0)
        scale = torch.exp(
            self.max_log_delta * torch.tanh(self.adapter(context))
        )
        return (base * scale).clamp(1e-5, 20.0)


class GraphRoutedChromatinField(nn.Module):
    """Accessibility vector field with no direct neural or guide decoder.

    A named perturbation first traverses the regulator-to-TF evidence matrix.
    It then traverses motif-supported TF-to-peak edges.  Learned signs are
    permitted here because the CRISPR-sciATAC response is the missing
    perturbational evidence for opening versus closing; edge existence is not
    learned.
    """

    def __init__(
        self,
        priors: ChromatinRoutePriors,
        context_dim: int,
        *,
        max_context_gain: float = 1.5,
    ) -> None:
        super().__init__()
        priors.validate()
        route = torch.as_tensor(priors.regulator_tf_support, dtype=torch.float32)
        route = route / route.amax(dim=1, keepdim=True).clamp_min(1.0)
        motif = torch.as_tensor(priors.tf_peak_motif, dtype=torch.float32)
        edges = torch.nonzero(motif > 0, as_tuple=False)
        confidence = motif[edges[:, 0], edges[:, 1]]
        confidence = confidence / confidence.amax().clamp_min(1.0)

        self.num_regulators = int(route.shape[0])
        self.num_tfs = int(route.shape[1])
        self.num_peaks = int(motif.shape[1])
        self.context_dim = int(context_dim)
        self.max_context_gain = float(max_context_gain)
        self.register_buffer("regulator_tf_support", route)
        self.register_buffer("motif_tf_index", edges[:, 0].long())
        self.register_buffer("motif_peak_index", edges[:, 1].long())
        self.register_buffer("motif_confidence", confidence)

        # Direction is learned only on supported paths using perturbational
        # chromatin observations.  There is no parameter for an unsupported
        # regulator-TF or TF-peak edge.
        self.raw_tf_direction = nn.Parameter(
            torch.linspace(-0.15, 0.15, self.num_tfs)
        )
        self.raw_tf_gain = nn.Parameter(
            torch.full((self.num_tfs,), _inverse_softplus(0.15))
        )
        self.raw_motif_direction = nn.Parameter(
            torch.linspace(-0.10, 0.10, edges.shape[0])
        )
        self.raw_motif_gain = nn.Parameter(
            torch.full((edges.shape[0],), _inverse_softplus(0.05))
        )
        self.tf_context_gain = nn.Linear(context_dim, self.num_tfs, bias=False)
        nn.init.zeros_(self.tf_context_gain.weight)

        self.open_rate = ContextRate(context_dim, self.num_peaks, 0.08)
        self.close_rate = ContextRate(context_dim, self.num_peaks, 0.08)
        self.recovery_rate = ContextRate(context_dim, self.num_peaks, 0.02)

    @property
    def motif_edges(self) -> int:
        return int(self.motif_tf_index.numel())

    def _route(
        self,
        intervention: Tensor,
        support_override: Optional[Tensor],
    ) -> Tensor:
        if intervention.ndim != 2 or intervention.shape[1] != self.num_regulators:
            raise ValueError("intervention must have shape [batch, regulators]")
        if not torch.isfinite(intervention).all() or bool((intervention < 0).any()):
            raise ValueError("intervention must be finite and non-negative")
        support = self.regulator_tf_support
        if support_override is not None:
            support = torch.as_tensor(
                support_override,
                dtype=intervention.dtype,
                device=intervention.device,
            )
            if support.shape != self.regulator_tf_support.shape:
                raise ValueError("support override has the wrong shape")
            if not torch.isfinite(support).all() or bool((support < 0).any()):
                raise ValueError("support override must be finite and non-negative")
        return intervention @ support

    def components(
        self,
        accessibility: Tensor,
        baseline: Tensor,
        intervention: Tensor,
        context: Tensor,
        *,
        support_override: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        expected = (intervention.shape[0], self.num_peaks)
        if accessibility.shape != expected or baseline.shape != expected:
            raise ValueError("accessibility and baseline have the wrong shape")
        if context.shape != (intervention.shape[0], self.context_dim):
            raise ValueError("context has the wrong shape")

        route = self._route(intervention, support_override)
        tf_gain = torch.exp(
            self.max_context_gain * torch.tanh(self.tf_context_gain(context))
        )
        tf_message = (
            route
            * torch.tanh(self.raw_tf_direction).unsqueeze(0)
            * F.softplus(self.raw_tf_gain).unsqueeze(0)
            * tf_gain
        )
        edge_message = (
            tf_message[:, self.motif_tf_index]
            * torch.tanh(self.raw_motif_direction).unsqueeze(0)
            * F.softplus(self.raw_motif_gain).unsqueeze(0)
            * self.motif_confidence.unsqueeze(0)
        )
        peak_drive = accessibility.new_zeros((accessibility.shape[0], self.num_peaks))
        peak_drive.index_add_(1, self.motif_peak_index, edge_message)

        current = accessibility.clamp(0.0, 1.0)
        reference = baseline.clamp(0.0, 1.0)
        opening = self.open_rate(context) * (1.0 - current) * F.relu(peak_drive)
        closing = self.close_rate(context) * current * F.relu(-peak_drive)
        recovery = self.recovery_rate(context) * (reference - current)
        derivative = opening - closing + recovery
        return {
            "derivative": derivative,
            "regulator_tf_route": route,
            "tf_message": tf_message,
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
        support_override: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if steps < 1 or horizon <= 0:
            raise ValueError("positive horizon and at least one integration step are required")
        state = baseline.clamp(0.0, 1.0)
        dt = float(horizon) / int(steps)

        def derivative(value: Tensor) -> Tensor:
            return self.components(
                value,
                baseline,
                intervention,
                context,
                support_override=support_override,
            )["derivative"]

        for _ in range(int(steps)):
            k1 = derivative(state)
            k2 = derivative((state + 0.5 * dt * k1).clamp(0.0, 1.0))
            k3 = derivative((state + 0.5 * dt * k2).clamp(0.0, 1.0))
            k4 = derivative((state + dt * k3).clamp(0.0, 1.0))
            state = (state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).clamp(0.0, 1.0)
        final_components = self.components(
            state,
            baseline,
            intervention,
            context,
            support_override=support_override,
        )
        return {
            "atac_t": state,
            "initial_atac": baseline,
            **final_components,
        }


class WLDChromatinResponseModel(nn.Module):
    """Broadly pretrained ATAC context plus a graph-routed response field."""

    def __init__(
        self,
        foundation: WLDMultistudyFoundationModel,
        route_priors: ChromatinRoutePriors,
    ) -> None:
        super().__init__()
        route_priors.validate()
        if foundation.num_tfs != route_priors.num_tfs:
            raise ValueError("foundation and chromatin route priors disagree on TFs")
        if foundation.num_peaks != route_priors.num_peaks:
            raise ValueError("foundation and chromatin route priors disagree on peaks")
        self.foundation = foundation
        self.field = GraphRoutedChromatinField(
            route_priors, context_dim=foundation.context_dim
        )

    def encode_control(self, control_atac: Tensor) -> Dict[str, Tensor]:
        if control_atac.ndim != 2 or control_atac.shape[1] != self.foundation.num_peaks:
            raise ValueError("control ATAC has the wrong shape")
        cues = control_atac.new_zeros(
            (control_atac.shape[0], self.foundation.priors.num_cues)
        )
        encoded = self.foundation.encoder(cues=cues, atac=control_atac)
        context = self.foundation.context_network(encoded["biological_context"])
        return {"encoded": encoded, "context": context}

    def forward(
        self,
        control_atac: Tensor,
        intervention: Tensor,
        *,
        horizon: float = 1.0,
        steps: int = 6,
        support_override: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        values = self.encode_control(control_atac)
        output = self.field.integrate(
            control_atac,
            intervention,
            values["context"],
            horizon=horizon,
            steps=steps,
            support_override=support_override,
        )
        output["context"] = values["context"]
        output["initial_tf"] = values["encoded"]["tf"]
        return output


def degree_preserving_bipartite_shuffle(
    support: Tensor,
    *,
    seed: int,
    swaps_per_edge: int = 20,
) -> Tensor:
    """Shuffle route topology while preserving regulator and TF degrees."""

    values = torch.as_tensor(support, dtype=torch.float32)
    if values.ndim != 2 or bool((values < 0).any()):
        raise ValueError("support must be a non-negative matrix")
    mask = (values > 0).cpu().numpy().astype(np.uint8)
    edges = [tuple(map(int, edge)) for edge in np.argwhere(mask)]
    if len(edges) < 2:
        raise ValueError("at least two edges are required for a shuffle")
    edge_set = set(edges)
    rng = np.random.default_rng(seed)
    successful = 0
    for _ in range(max(len(edges) * int(swaps_per_edge), 1)):
        first, second = rng.choice(len(edges), size=2, replace=False)
        r1, t1 = edges[int(first)]
        r2, t2 = edges[int(second)]
        if r1 == r2 or t1 == t2 or (r1, t2) in edge_set or (r2, t1) in edge_set:
            continue
        edge_set.remove((r1, t1))
        edge_set.remove((r2, t2))
        edge_set.add((r1, t2))
        edge_set.add((r2, t1))
        edges[int(first)] = (r1, t2)
        edges[int(second)] = (r2, t1)
        successful += 1
    if not successful:
        raise RuntimeError("could not create a degree-preserving route shuffle")
    shuffled = torch.zeros_like(values)
    confidences = values[values > 0].detach().cpu().numpy().copy()
    rng.shuffle(confidences)
    for (row, column), confidence in zip(sorted(edge_set), confidences):
        shuffled[row, column] = float(confidence)
    original_row = (values > 0).sum(dim=1)
    original_col = (values > 0).sum(dim=0)
    if not torch.equal(original_row, (shuffled > 0).sum(dim=1)):
        raise RuntimeError("shuffle changed regulator degrees")
    if not torch.equal(original_col, (shuffled > 0).sum(dim=0)):
        raise RuntimeError("shuffle changed TF degrees")
    return shuffled


def architecture_contract(model: WLDChromatinResponseModel) -> Dict[str, object]:
    support = model.field.regulator_tf_support
    return {
        "named_regulator_nodes": model.field.num_regulators,
        "named_tf_nodes": model.field.num_tfs,
        "foundation_peaks": model.field.num_peaks,
        "regulator_tf_edges": int(torch.count_nonzero(support)),
        "motif_supported_tf_peak_edges": model.field.motif_edges,
        "guide_or_target_in_encoder": False,
        "direct_neural_context_to_peak_decoder": False,
        "unsupported_edges_trainable": False,
        "context_conditioned_supported_gain": True,
        "context_conditioned_open_close_recovery": True,
        "claim_scope": "transient perturbational chromatin response; not attractor dynamics",
    }
