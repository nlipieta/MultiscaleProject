"""Mechanistically constrained multiscale circuit dynamics for WLD v3.

This module is deliberately separate from the PBMC snapshot reconstruction
runner.  It defines a temporal model for experiments with real time,
perturbation, lineage, metabolic-labeling, or velocity information.

The state follows the causal ordering in the Attractor State manuscript:

    measured cues -> signaling proteins -> core TF circuit
                                      <-> chromatin accessibility
                                      -> RNA production

Important properties
--------------------
* Trainable interaction parameters exist only for supplied prior edges.
* Activating/repressive signs are fixed by the supplied prior.
* TF effects are gated by TF-motif x accessible-enhancer x gene-link support.
* Hill kinetics and positive degradation rates define the vector field.
* Chromatin is a slower state variable, not a predeclared toggle or label.
* There is no unrestricted neural residual that can bypass the circuit.

The model does not make an attractor claim by construction.  Fixed points,
Jacobian stability, basin return, and held-out interventions must all be
evaluated on an experimental design that identifies temporal dynamics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def _feature_tokens(name: str) -> Tuple[str, ...]:
    """Tokenize feature names without substring false positives."""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return tuple(token for token in re.split(r"[^a-z0-9]+", normalized.lower()) if token)


def temporal_leakage_audit(
    train_groups: Sequence[object],
    test_groups: Sequence[object],
    initial_feature_names: Sequence[str],
    initial_time: float,
    target_time: float,
    uses_initial_rna: bool = False,
    initial_rna_time: Optional[float] = None,
) -> Dict[str, object]:
    """Reject identity/state proxies and invalid temporal evaluation designs.

    The standard v3 initialization contract is measured ATAC plus external
    cues. A real RNA measurement is allowed only when explicitly declared and
    timestamped at the same initial time; it is then a conditioning
    measurement for future-state prediction, not a derived latent-state test.
    """
    train = {str(value) for value in train_groups}
    test = {str(value) for value in test_groups}
    overlap = sorted(train.intersection(test))
    if overlap:
        raise ValueError(f"Train/test biological groups overlap: {overlap}")
    if not train or not test:
        raise ValueError("Train and test biological groups must both be present.")
    if not math.isfinite(initial_time) or not math.isfinite(target_time):
        raise ValueError("Initial and target times must be finite.")
    if target_time <= initial_time:
        raise ValueError("target_time must be later than initial_time.")

    blocked_singletons = {
        "cluster",
        "label",
        "outcome",
        "pseudotime",
        "target",
        "trajectory",
    }
    blocked_sequences = {
        ("cell", "identity"),
        ("cell", "type"),
        ("future", "state"),
        ("target", "state"),
    }
    rna_tokens = {"rna", "expression", "transcriptome", "transcriptomic"}
    bad = []
    for feature in initial_feature_names:
        tokens = _feature_tokens(feature)
        pairs = set(zip(tokens, tokens[1:]))
        is_proxy = bool(blocked_singletons.intersection(tokens)) or bool(
            blocked_sequences.intersection(pairs)
        )
        is_undeclared_rna = bool(rna_tokens.intersection(tokens)) and not uses_initial_rna
        if is_proxy or is_undeclared_rna:
            bad.append(str(feature))
    if bad:
        raise ValueError(f"Direct identity/state proxies found in initial inputs: {bad}")

    if uses_initial_rna:
        if initial_rna_time is None or not math.isclose(
            float(initial_rna_time), float(initial_time), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "Declared initial RNA must be a genuine measurement at initial_time."
            )
    elif initial_rna_time is not None:
        raise ValueError("initial_rna_time was supplied while uses_initial_rna is False.")

    return {
        "group_overlap": [],
        "initial_features": [str(value) for value in initial_feature_names],
        "initial_time": float(initial_time),
        "target_time": float(target_time),
        "uses_initial_rna": bool(uses_initial_rna),
    }


def _inverse_softplus_scalar(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus initialization must be positive.")
    return math.log(math.expm1(value))


def _inverse_softplus(value: Tensor) -> Tensor:
    value = value.clamp_min(1e-6)
    return torch.where(value > 20.0, value, torch.log(torch.expm1(value)))


@dataclass(frozen=True)
class MultiscaleCircuitPriors:
    """Hard structural priors for the multiscale temporal model.

    Matrix orientation is always ``[source, target]`` except for the two peak
    annotation matrices, whose first dimension is the peak index.

    peak_to_gene:
        Non-negative enhancer/promoter-to-gene evidence ``[peaks, genes]``.
    peak_tf_motif:
        Non-negative localized motif/occupancy evidence ``[peaks, TFs]``.
    tf_gene_support:
        Signed TF-to-gene regulatory support ``[TFs, genes]``.  Its non-zero
        pattern is intersected with peak-level evidence at run time.
    circuit_tf_tf:
        Signed, validated core circuit ``[TFs, TFs]``.
    tf_gene_index:
        Gene index for each TF node ``[TFs]``.  It lets chromatin at the target
        TF gene gate every TF-to-TF circuit edge.
    signal_signal:
        Signed protein/signaling graph ``[signals, signals]``.
    signal_tf:
        Signed signaling-to-TF graph ``[signals, TFs]``.
    cue_signal:
        Signed measured-cue-to-signaling graph ``[cues, signals]``.
    tf_peak_effect:
        Signed TF-to-peak opening/closing support ``[TFs, peaks]``.  Use zero
        when no defensible pioneer/chromatin-remodeling evidence is available.
    """

    peak_to_gene: Tensor
    peak_tf_motif: Tensor
    tf_gene_support: Tensor
    circuit_tf_tf: Tensor
    tf_gene_index: Tensor
    signal_signal: Tensor
    signal_tf: Tensor
    cue_signal: Tensor
    tf_peak_effect: Tensor

    def validate(self) -> None:
        matrices = {
            "peak_to_gene": self.peak_to_gene,
            "peak_tf_motif": self.peak_tf_motif,
            "tf_gene_support": self.tf_gene_support,
            "circuit_tf_tf": self.circuit_tf_tf,
            "signal_signal": self.signal_signal,
            "signal_tf": self.signal_tf,
            "cue_signal": self.cue_signal,
            "tf_peak_effect": self.tf_peak_effect,
        }
        for name, value in matrices.items():
            if value.ndim != 2:
                raise ValueError(f"{name} must be rank two.")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")

        num_peaks, num_genes = self.peak_to_gene.shape
        motif_peaks, num_tfs = self.peak_tf_motif.shape
        num_signals = self.signal_signal.shape[0]
        num_cues = self.cue_signal.shape[0]

        if motif_peaks != num_peaks:
            raise ValueError("peak_to_gene and peak_tf_motif disagree on peaks.")
        if self.tf_gene_support.shape != (num_tfs, num_genes):
            raise ValueError("tf_gene_support has incompatible dimensions.")
        if self.circuit_tf_tf.shape != (num_tfs, num_tfs):
            raise ValueError("circuit_tf_tf must be square with one row per TF.")
        if self.signal_signal.shape != (num_signals, num_signals):
            raise ValueError("signal_signal must be square.")
        if self.signal_tf.shape != (num_signals, num_tfs):
            raise ValueError("signal_tf has incompatible dimensions.")
        if self.cue_signal.shape != (num_cues, num_signals):
            raise ValueError("cue_signal has incompatible dimensions.")
        if self.tf_peak_effect.shape != (num_tfs, num_peaks):
            raise ValueError("tf_peak_effect has incompatible dimensions.")
        if self.tf_gene_index.ndim != 1 or self.tf_gene_index.shape[0] != num_tfs:
            raise ValueError("tf_gene_index must have one entry per TF.")
        if self.tf_gene_index.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("tf_gene_index must contain integer indices.")
        if bool((self.tf_gene_index < 0).any()) or bool(
            (self.tf_gene_index >= num_genes).any()
        ):
            raise ValueError("tf_gene_index contains an out-of-range gene index.")
        if bool((self.peak_to_gene < 0).any()):
            raise ValueError("peak_to_gene must be non-negative.")
        if bool((self.peak_tf_motif < 0).any()):
            raise ValueError("peak_tf_motif must be non-negative.")

        localized_support = torch.einsum(
            "pt,pg->tg", self.peak_tf_motif.float(), self.peak_to_gene.float()
        ) > 0
        dead_tf_gene = (self.tf_gene_support != 0) & ~localized_support
        if bool(dead_tf_gene.any()):
            raise ValueError(
                "Every TF-gene edge must have localized motif/occupancy and "
                "peak-to-gene support."
            )

        circuit_edges = torch.nonzero(self.circuit_tf_tf != 0, as_tuple=False)
        if circuit_edges.numel():
            circuit_sources = circuit_edges[:, 0]
            circuit_targets = circuit_edges[:, 1]
            target_genes = self.tf_gene_index[circuit_targets].long()
            target_gene_support = self.tf_gene_support[
                circuit_sources, target_genes
            ]
            if bool((target_gene_support == 0).any()):
                raise ValueError(
                    "Every TF-circuit edge must regulate the target TF gene."
                )
            circuit_sign = torch.sign(
                self.circuit_tf_tf[circuit_sources, circuit_targets]
            )
            if bool((circuit_sign != torch.sign(target_gene_support)).any()):
                raise ValueError(
                    "Circuit-edge signs must match TF-to-target-TF-gene signs."
                )

        peak_effect_edges = torch.nonzero(
            self.tf_peak_effect != 0, as_tuple=False
        )
        if peak_effect_edges.numel():
            effect_tfs = peak_effect_edges[:, 0]
            effect_peaks = peak_effect_edges[:, 1]
            if bool((self.peak_tf_motif[effect_peaks, effect_tfs] <= 0).any()):
                raise ValueError(
                    "Every TF-to-peak effect must have localized binding evidence."
                )

    @property
    def num_peaks(self) -> int:
        return int(self.peak_to_gene.shape[0])

    @property
    def num_genes(self) -> int:
        return int(self.peak_to_gene.shape[1])

    @property
    def num_tfs(self) -> int:
        return int(self.peak_tf_motif.shape[1])

    @property
    def num_signals(self) -> int:
        return int(self.signal_signal.shape[0])

    @property
    def num_cues(self) -> int:
        return int(self.cue_signal.shape[0])


class SparseSignedLinearLayer(nn.Module):
    """Signed linear messages with parameters only on supplied edges."""

    def __init__(self, signed_adjacency: Tensor, initial_gain: float = 0.25):
        super().__init__()
        if signed_adjacency.ndim != 2:
            raise ValueError("signed_adjacency must be rank two.")
        edge_index = torch.nonzero(signed_adjacency != 0, as_tuple=False)
        self.source_dim = int(signed_adjacency.shape[0])
        self.target_dim = int(signed_adjacency.shape[1])
        self.register_buffer("source_index", edge_index[:, 0].long())
        self.register_buffer("target_index", edge_index[:, 1].long())
        values = signed_adjacency[edge_index[:, 0], edge_index[:, 1]].float()
        confidence = values.abs()
        if confidence.numel():
            confidence = confidence / confidence.amax().clamp_min(1.0)
        self.register_buffer("edge_sign", torch.sign(values))
        self.register_buffer("edge_confidence", confidence)
        self.raw_gain = nn.Parameter(
            torch.full(
                (edge_index.shape[0],),
                _inverse_softplus_scalar(initial_gain),
                dtype=torch.float32,
            )
        )

    @property
    def num_edges(self) -> int:
        return int(self.source_index.numel())

    def effective_gain(self) -> Tensor:
        return F.softplus(self.raw_gain)

    def forward(self, source: Tensor, edge_scale: Optional[Tensor] = None) -> Tensor:
        if source.ndim != 2 or source.shape[1] != self.source_dim:
            raise ValueError("source has the wrong shape.")
        result = source.new_zeros((source.shape[0], self.target_dim))
        if self.num_edges == 0:
            return result
        messages = (
            source[:, self.source_index]
            * self.edge_sign
            * self.edge_confidence
            * self.effective_gain()
        )
        if edge_scale is not None:
            messages = messages * _validated_edge_scale(
                edge_scale, source.shape[0], self.num_edges, source
            )
        result.index_add_(1, self.target_index, messages)
        return result


class SparseSignedHillLayer(nn.Module):
    """Edge-specific signed Hill kinetics on a hard sparse topology."""

    def __init__(
        self,
        signed_adjacency: Tensor,
        initial_gain: float = 0.25,
        initial_threshold: float = 0.5,
        initial_hill: float = 1.5,
    ) -> None:
        super().__init__()
        if signed_adjacency.ndim != 2:
            raise ValueError("signed_adjacency must be rank two.")
        if initial_hill <= 1.0:
            raise ValueError("initial_hill must be greater than one.")
        edge_index = torch.nonzero(signed_adjacency != 0, as_tuple=False)
        self.source_dim = int(signed_adjacency.shape[0])
        self.target_dim = int(signed_adjacency.shape[1])
        self.register_buffer("source_index", edge_index[:, 0].long())
        self.register_buffer("target_index", edge_index[:, 1].long())
        values = signed_adjacency[edge_index[:, 0], edge_index[:, 1]].float()
        confidence = values.abs()
        if confidence.numel():
            confidence = confidence / confidence.amax().clamp_min(1.0)
        self.register_buffer("edge_sign", torch.sign(values))
        self.register_buffer("edge_confidence", confidence)
        edge_count = edge_index.shape[0]
        self.raw_gain = nn.Parameter(
            torch.full(
                (edge_count,),
                _inverse_softplus_scalar(initial_gain),
                dtype=torch.float32,
            )
        )
        self.raw_threshold = nn.Parameter(
            torch.full(
                (edge_count,),
                _inverse_softplus_scalar(initial_threshold),
                dtype=torch.float32,
            )
        )
        self.raw_hill = nn.Parameter(
            torch.full(
                (edge_count,),
                _inverse_softplus_scalar(initial_hill - 1.0),
                dtype=torch.float32,
            )
        )

    @property
    def num_edges(self) -> int:
        return int(self.source_index.numel())

    def effective_gain(self) -> Tensor:
        return F.softplus(self.raw_gain)

    def effective_threshold(self) -> Tensor:
        return F.softplus(self.raw_threshold).clamp_min(1e-4)

    def effective_hill(self) -> Tensor:
        return 1.0 + F.softplus(self.raw_hill)

    def forward(self, source: Tensor, edge_gate: Optional[Tensor] = None) -> Tensor:
        if source.ndim != 2 or source.shape[1] != self.source_dim:
            raise ValueError("source has the wrong shape.")
        result = source.new_zeros((source.shape[0], self.target_dim))
        if self.num_edges == 0:
            return result

        concentration = source[:, self.source_index].clamp_min(0.0)
        threshold = self.effective_threshold()
        hill = self.effective_hill()
        log_ratio = hill * (
            torch.log(concentration.clamp_min(1e-8)) - torch.log(threshold)
        )
        occupancy = torch.sigmoid(log_ratio)
        messages = (
            occupancy
            * self.edge_sign
            * self.edge_confidence
            * self.effective_gain()
        )
        if edge_gate is not None:
            messages = messages * _validated_edge_scale(
                edge_gate, source.shape[0], self.num_edges, source
            )
        result.index_add_(1, self.target_index, messages)
        return result

    def dense_effective_adjacency(self) -> Tensor:
        dense = self.raw_gain.new_zeros((self.source_dim, self.target_dim))
        if self.num_edges:
            dense[self.source_index, self.target_index] = (
                self.edge_sign * self.edge_confidence * self.effective_gain()
            )
        return dense


def _validated_edge_scale(
    value: Tensor,
    batch_size: int,
    edge_count: int,
    reference: Tensor,
) -> Tensor:
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if not torch.isfinite(value).all() or bool((value < 0).any()):
        raise ValueError("edge scales must be finite and non-negative.")
    if value.ndim == 1:
        if value.shape[0] != edge_count:
            raise ValueError("edge scale has the wrong number of edges.")
        return value.unsqueeze(0)
    if value.shape != (batch_size, edge_count):
        raise ValueError("edge scale must have shape [edges] or [batch, edges].")
    return value


class EnhancerAccessibilityGate(nn.Module):
    """Cell-specific TF-to-gene gates from localized accessible enhancers."""

    def __init__(
        self,
        peak_to_gene: Tensor,
        peak_tf_motif: Tensor,
        tf_gene_support: Tensor,
    ) -> None:
        super().__init__()
        p2g = peak_to_gene.float().clamp_min(0.0)
        motif = peak_tf_motif.float().clamp_min(0.0)
        support = (tf_gene_support != 0).float()
        normalizer = torch.einsum("pt,pg->tg", motif, p2g).clamp_min(1e-8)
        self.register_buffer("peak_to_gene", p2g)
        self.register_buffer("peak_tf_motif", motif)
        self.register_buffer("tf_gene_support", support)
        self.register_buffer("normalizer", normalizer)

    def forward(self, accessibility: Tensor) -> Tensor:
        if accessibility.ndim != 2 or accessibility.shape[1] != self.peak_to_gene.shape[0]:
            raise ValueError("accessibility must have shape [batch, peaks].")
        accessibility = accessibility.clamp(0.0, 1.0)
        localized = torch.einsum(
            "bp,pt,pg->btg",
            accessibility,
            self.peak_tf_motif,
            self.peak_to_gene,
        )
        gate = (localized / self.normalizer.unsqueeze(0)).clamp(0.0, 1.0)
        return gate * self.tf_gene_support.unsqueeze(0)


@dataclass(frozen=True)
class CircuitIntervention:
    """Explicit activity or edge perturbation applied during integration."""

    tf_activity_scale: Optional[Tensor] = None
    circuit_edge_scale: Optional[Tensor] = None
    signaling_edge_scale: Optional[Tensor] = None


class MultiscaleCircuitVectorField(nn.Module):
    """Hard-constrained vector field for signaling, TF, chromatin, and RNA."""

    def __init__(self, priors: MultiscaleCircuitPriors) -> None:
        super().__init__()
        priors.validate()
        self.num_signals = priors.num_signals
        self.num_tfs = priors.num_tfs
        self.num_peaks = priors.num_peaks
        self.num_genes = priors.num_genes
        self.num_cues = priors.num_cues
        self.state_dim = (
            self.num_signals + self.num_tfs + self.num_peaks + self.num_genes
        )
        self.register_buffer("tf_gene_index", priors.tf_gene_index.long())

        self.accessibility_gate = EnhancerAccessibilityGate(
            priors.peak_to_gene,
            priors.peak_tf_motif,
            priors.tf_gene_support,
        )
        self.cue_to_signal = SparseSignedLinearLayer(priors.cue_signal)
        self.signal_recurrent = SparseSignedHillLayer(priors.signal_signal)
        self.signal_to_tf = SparseSignedHillLayer(priors.signal_tf)
        self.tf_circuit = SparseSignedHillLayer(priors.circuit_tf_tf)
        self.tf_to_peak = SparseSignedHillLayer(priors.tf_peak_effect)
        self.tf_to_gene = SparseSignedHillLayer(priors.tf_gene_support)

        self.signal_basal = nn.Parameter(torch.full((self.num_signals,), -1.5))
        self.tf_basal = nn.Parameter(torch.full((self.num_tfs,), -1.5))
        self.chromatin_basal = nn.Parameter(torch.zeros(self.num_peaks))
        self.rna_basal = nn.Parameter(torch.full((self.num_genes,), -1.5))

        self.signal_decay_raw = nn.Parameter(torch.zeros(self.num_signals))
        self.tf_decay_raw = nn.Parameter(torch.zeros(self.num_tfs))
        self.rna_decay_raw = nn.Parameter(torch.zeros(self.num_genes))
        self.chromatin_timescale_raw = nn.Parameter(
            torch.full((self.num_peaks,), -3.0)
        )

    def split_state(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError("state must have shape [batch, state_dim].")
        s_end = self.num_signals
        z_end = s_end + self.num_tfs
        a_end = z_end + self.num_peaks
        return (
            state[:, :s_end],
            state[:, s_end:z_end],
            state[:, z_end:a_end],
            state[:, a_end:],
        )

    def join_state(
        self, signal: Tensor, tf: Tensor, accessibility: Tensor, rna: Tensor
    ) -> Tensor:
        return torch.cat([signal, tf, accessibility, rna], dim=-1)

    def positive_rates(self) -> Dict[str, Tensor]:
        return {
            "signal_decay": F.softplus(self.signal_decay_raw).clamp_min(1e-4),
            "tf_decay": F.softplus(self.tf_decay_raw).clamp_min(1e-4),
            "rna_decay": F.softplus(self.rna_decay_raw).clamp_min(1e-4),
            "chromatin_timescale": F.softplus(
                self.chromatin_timescale_raw
            ).clamp_min(1e-4),
        }

    def components(
        self,
        state: Tensor,
        cues: Tensor,
        intervention: Optional[CircuitIntervention] = None,
    ) -> Dict[str, Tensor]:
        signal, tf, accessibility, rna = self.split_state(state)
        if cues.ndim != 2 or cues.shape != (state.shape[0], self.num_cues):
            raise ValueError("cues must have shape [batch, num_cues].")
        if not torch.isfinite(cues).all() or bool((cues < 0).any()):
            raise ValueError("cues must be finite, non-negative magnitudes.")
        intervention = intervention or CircuitIntervention()
        rates = self.positive_rates()

        tf_effective = tf.clamp_min(0.0)
        if intervention.tf_activity_scale is not None:
            scale = torch.as_tensor(
                intervention.tf_activity_scale,
                dtype=state.dtype,
                device=state.device,
            )
            if scale.ndim == 1:
                if scale.shape[0] != self.num_tfs:
                    raise ValueError("tf_activity_scale has the wrong length.")
                scale = scale.unsqueeze(0)
            if scale.shape not in ((1, self.num_tfs), (state.shape[0], self.num_tfs)):
                raise ValueError("tf_activity_scale has the wrong shape.")
            if not torch.isfinite(scale).all() or bool((scale < 0).any()):
                raise ValueError(
                    "tf_activity_scale must be finite and non-negative."
                )
            tf_effective = tf_effective * scale

        cue_drive = self.cue_to_signal(cues)
        signal_drive = self.signal_recurrent(
            signal.clamp_min(0.0), intervention.signaling_edge_scale
        )
        signal_production = F.softplus(self.signal_basal + cue_drive + signal_drive)
        d_signal = signal_production - rates["signal_decay"] * signal

        tf_gene_gate = self.accessibility_gate(accessibility)
        if self.tf_circuit.num_edges:
            circuit_gate = tf_gene_gate[
                :,
                self.tf_circuit.source_index,
                self.tf_gene_index[self.tf_circuit.target_index],
            ]
            if intervention.circuit_edge_scale is not None:
                circuit_gate = circuit_gate * _validated_edge_scale(
                    intervention.circuit_edge_scale,
                    state.shape[0],
                    self.tf_circuit.num_edges,
                    state,
                )
        else:
            circuit_gate = state.new_zeros((state.shape[0], 0))

        tf_drive = self.signal_to_tf(signal.clamp_min(0.0))
        tf_drive = tf_drive + self.tf_circuit(tf_effective, circuit_gate)
        tf_production = F.softplus(self.tf_basal + tf_drive)
        d_tf = tf_production - rates["tf_decay"] * tf

        chromatin_drive = self.tf_to_peak(tf_effective)
        chromatin_target = torch.sigmoid(self.chromatin_basal + chromatin_drive)
        d_accessibility = rates["chromatin_timescale"] * (
            chromatin_target - accessibility
        )

        if self.tf_to_gene.num_edges:
            rna_edge_gate = tf_gene_gate[
                :, self.tf_to_gene.source_index, self.tf_to_gene.target_index
            ]
        else:
            rna_edge_gate = state.new_zeros((state.shape[0], 0))
        rna_drive = self.tf_to_gene(tf_effective, rna_edge_gate)
        rna_production = F.softplus(self.rna_basal + rna_drive)
        d_rna = rna_production - rates["rna_decay"] * rna

        return {
            "derivative": self.join_state(
                d_signal, d_tf, d_accessibility, d_rna
            ),
            "signal_production": signal_production,
            "tf_production": tf_production,
            "chromatin_target": chromatin_target,
            "rna_production": rna_production,
            "tf_gene_gate": tf_gene_gate,
            "circuit_edge_gate": circuit_gate,
        }

    def forward(
        self,
        state: Tensor,
        cues: Tensor,
        intervention: Optional[CircuitIntervention] = None,
    ) -> Tensor:
        return self.components(state, cues, intervention)["derivative"]


def rk4_integrate_controlled(
    field: MultiscaleCircuitVectorField,
    initial_state: Tensor,
    cues: Tensor,
    horizon: float,
    steps: int,
    intervention: Optional[CircuitIntervention] = None,
) -> Tuple[Tensor, Tensor]:
    """Differentiable fixed-step RK4 for one piecewise-constant cue interval."""
    if horizon <= 0 or steps < 1:
        raise ValueError("horizon and steps must be positive.")
    dt = horizon / steps
    state = initial_state
    path = [state]
    for _ in range(steps):
        k1 = field(state, cues, intervention)
        k2 = field(state + 0.5 * dt * k1, cues, intervention)
        k3 = field(state + 0.5 * dt * k2, cues, intervention)
        k4 = field(state + dt * k3, cues, intervention)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        path.append(state)
    return state, torch.stack(path, dim=1)


class CircuitDynamicsModel(nn.Module):
    """ATAC-initialized, hard-constrained temporal WLD model."""

    def __init__(self, priors: MultiscaleCircuitPriors) -> None:
        super().__init__()
        priors.validate()
        self.field = MultiscaleCircuitVectorField(priors)
        self.num_signals = priors.num_signals
        self.num_tfs = priors.num_tfs
        self.num_peaks = priors.num_peaks
        self.num_genes = priors.num_genes
        self.num_cues = priors.num_cues
        self.state_dim = self.field.state_dim

        motif = priors.peak_tf_motif.float().clamp_min(0.0)
        motif = motif / motif.sum(dim=0, keepdim=True).clamp_min(1.0)
        self.register_buffer("motif_activity_map", motif)
        self.initial_signal_bias = nn.Parameter(torch.full((self.num_signals,), -1.5))
        self.initial_tf_bias = nn.Parameter(torch.full((self.num_tfs,), -1.5))
        self.initial_tf_scale_raw = nn.Parameter(torch.zeros(self.num_tfs))

    def initial_state(
        self,
        atac: Tensor,
        cues: Tensor,
        initial_rna: Optional[Tensor] = None,
    ) -> Tensor:
        """Infer the initial state without a dense identity encoder.

        ``initial_rna`` is optional and is valid only when it is genuinely
        observed at the initial time of a future-state prediction task.  It is
        never substituted with a future target.  With the default ``None``,
        RNA is initialized at the circuit-implied instantaneous equilibrium.
        """
        if atac.ndim != 2 or atac.shape[1] != self.num_peaks:
            raise ValueError("atac must have shape [batch, num_peaks].")
        if cues.ndim != 2 or cues.shape != (atac.shape[0], self.num_cues):
            raise ValueError("cues must have shape [batch, num_cues].")
        if not torch.isfinite(atac).all() or bool(
            ((atac < 0) | (atac > 1)).any()
        ):
            raise ValueError("atac must be finite and normalized to [0, 1].")
        if not torch.isfinite(cues).all() or bool((cues < 0).any()):
            raise ValueError("cues must be finite, non-negative magnitudes.")
        accessibility = atac
        cue_drive = self.field.cue_to_signal(cues)
        signal = F.softplus(self.initial_signal_bias + cue_drive)
        motif_activity = accessibility @ self.motif_activity_map
        tf = F.softplus(
            self.initial_tf_bias
            + F.softplus(self.initial_tf_scale_raw) * motif_activity
        )

        if initial_rna is not None:
            if initial_rna.shape != (atac.shape[0], self.num_genes):
                raise ValueError("initial_rna has the wrong shape.")
            if not torch.isfinite(initial_rna).all() or bool(
                (initial_rna < 0).any()
            ):
                raise ValueError("initial_rna must be finite and non-negative.")
            rna = initial_rna
        else:
            zero_rna = atac.new_zeros((atac.shape[0], self.num_genes))
            probe = self.field.join_state(signal, tf, accessibility, zero_rna)
            components = self.field.components(probe, cues)
            decay = self.field.positive_rates()["rna_decay"]
            rna = components["rna_production"] / decay
        return self.field.join_state(signal, tf, accessibility, rna)

    def split_path(self, path: Tensor) -> Dict[str, Tensor]:
        if path.ndim != 3 or path.shape[-1] != self.state_dim:
            raise ValueError("path must have shape [batch, time, state_dim].")
        s_end = self.num_signals
        z_end = s_end + self.num_tfs
        a_end = z_end + self.num_peaks
        return {
            "signal_path": path[:, :, :s_end],
            "tf_path": path[:, :, s_end:z_end],
            "accessibility_path": path[:, :, z_end:a_end],
            "rna_path": path[:, :, a_end:],
        }

    def integrate_state(
        self,
        initial_state: Tensor,
        cues: Tensor,
        horizon: float,
        steps: int,
        intervention: Optional[CircuitIntervention] = None,
    ) -> Tuple[Tensor, Tensor]:
        return rk4_integrate_controlled(
            self.field,
            initial_state,
            cues,
            horizon,
            steps,
            intervention,
        )

    def forward(
        self,
        atac: Tensor,
        cues: Tensor,
        horizon: float,
        steps: int,
        initial_rna: Optional[Tensor] = None,
        intervention: Optional[CircuitIntervention] = None,
    ) -> Dict[str, Tensor]:
        initial = self.initial_state(atac, cues, initial_rna)
        terminal, state_path = self.integrate_state(
            initial, cues, horizon, steps, intervention
        )
        paths = self.split_path(state_path)
        terminal_components = self.field.components(terminal, cues, intervention)
        return {
            "initial_state": initial,
            "terminal_state": terminal,
            "state_path": state_path,
            "terminal_velocity": terminal_components["derivative"],
            "terminal_tf_gene_gate": terminal_components["tf_gene_gate"],
            "terminal_circuit_edge_gate": terminal_components["circuit_edge_gate"],
            **paths,
            "signal_t": paths["signal_path"][:, -1],
            "tf_t": paths["tf_path"][:, -1],
            "accessibility_t": paths["accessibility_path"][:, -1],
            "rna_t": paths["rna_path"][:, -1],
        }

    def _raw_to_state(self, raw: Tensor) -> Tensor:
        signal, tf, accessibility, rna = self.field.split_state(raw)
        return self.field.join_state(
            F.softplus(signal),
            F.softplus(tf),
            torch.sigmoid(accessibility),
            F.softplus(rna),
        )

    def _state_to_raw(self, state: Tensor) -> Tensor:
        signal, tf, accessibility, rna = self.field.split_state(state)
        accessibility = accessibility.clamp(1e-5, 1.0 - 1e-5)
        return self.field.join_state(
            _inverse_softplus(signal),
            _inverse_softplus(tf),
            torch.logit(accessibility),
            _inverse_softplus(rna),
        )

    def refine_fixed_point(
        self,
        initial_state: Tensor,
        cues: Tensor,
        intervention: Optional[CircuitIntervention] = None,
        iterations: int = 1000,
        learning_rate: float = 1e-2,
        tolerance: float = 1e-6,
    ) -> Tuple[Tensor, float]:
        """Minimize vector-field norm without selecting held-out outcomes."""
        if initial_state.shape != (self.state_dim,):
            raise ValueError("initial_state must be a single state vector.")
        if cues.shape != (self.num_cues,):
            raise ValueError("cues must be a single cue vector.")
        raw = nn.Parameter(self._state_to_raw(initial_state.unsqueeze(0)).squeeze(0))
        optimizer = torch.optim.Adam([raw], lr=learning_rate)
        residual = float("inf")
        for _ in range(iterations):
            optimizer.zero_grad()
            state = self._raw_to_state(raw.unsqueeze(0))
            velocity = self.field(state, cues.unsqueeze(0), intervention)
            loss = velocity.square().mean()
            loss.backward()
            optimizer.step()
            residual = float(loss.detach().sqrt().cpu())
            if residual <= tolerance:
                break
        fixed = self._raw_to_state(raw.detach().unsqueeze(0)).squeeze(0)
        return fixed, residual

    def jacobian_eigenvalues(
        self,
        state: Tensor,
        cues: Tensor,
        intervention: Optional[CircuitIntervention] = None,
        max_dimension: int = 256,
    ) -> Tensor:
        """Return full-state Jacobian eigenvalues for a candidate equilibrium."""
        if state.shape != (self.state_dim,) or cues.shape != (self.num_cues,):
            raise ValueError("state or cues has the wrong shape.")
        if self.state_dim > max_dimension:
            raise ValueError(
                "Full Jacobian is too large for this diagnostic. Restrict the "
                "model to a validated regulatory core or raise max_dimension explicitly."
            )
        with torch.enable_grad():
            point = state.detach().clone().requires_grad_(True)

            def single_field(value: Tensor) -> Tensor:
                return self.field(
                    value.unsqueeze(0), cues.unsqueeze(0), intervention
                ).squeeze(0)

            jacobian = torch.autograd.functional.jacobian(single_field, point)
        return torch.linalg.eigvals(jacobian.detach())

    @torch.no_grad()
    def basin_return_fraction(
        self,
        fixed_state: Tensor,
        cues: Tensor,
        intervention: Optional[CircuitIntervention] = None,
        trials: int = 32,
        perturbation_scale: float = 0.05,
        horizon: float = 10.0,
        steps: int = 200,
        tolerance: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Tensor]:
        """Perturb a candidate fixed point and measure return to its neighborhood."""
        if trials < 1 or perturbation_scale <= 0 or tolerance <= 0:
            raise ValueError("trials, perturbation_scale, and tolerance must be positive.")
        if fixed_state.shape != (self.state_dim,) or cues.shape != (self.num_cues,):
            raise ValueError("fixed_state or cues has the wrong shape.")
        generator = torch.Generator(device=fixed_state.device).manual_seed(seed)
        base = fixed_state.unsqueeze(0).repeat(trials, 1)
        noise = torch.randn(
            base.shape,
            dtype=base.dtype,
            device=base.device,
            generator=generator,
        )
        perturbed = base + perturbation_scale * (base.abs() + 1.0) * noise
        signal, tf, accessibility, rna = self.field.split_state(perturbed)
        perturbed = self.field.join_state(
            signal.clamp_min(0.0),
            tf.clamp_min(0.0),
            accessibility.clamp(0.0, 1.0),
            rna.clamp_min(0.0),
        )
        repeated_cues = cues.unsqueeze(0).repeat(trials, 1)
        terminal, _ = self.integrate_state(
            perturbed,
            repeated_cues,
            horizon,
            steps,
            intervention,
        )
        normalized_distance = torch.linalg.vector_norm(terminal - base, dim=-1) / math.sqrt(
            self.state_dim
        )
        return {
            "fraction_returned": (normalized_distance <= tolerance).float().mean(),
            "normalized_distance": normalized_distance,
            "terminal_states": terminal,
        }


def temporal_circuit_objective(
    output: Dict[str, Tensor],
    target_rna: Tensor,
    target_accessibility: Optional[Tensor] = None,
    observed_derivative: Optional[Tensor] = None,
    terminal_mask: Optional[Tensor] = None,
    model: Optional[CircuitDynamicsModel] = None,
    rna_weight: float = 1.0,
    accessibility_weight: float = 0.5,
    derivative_weight: float = 0.25,
    terminal_weight: float = 0.1,
    gain_weight: float = 1e-4,
) -> Dict[str, Tensor]:
    """Loss for genuine temporal or perturbation-resolved observations.

    ``terminal_mask`` must come from the experimental design (for example a
    measured recovery plateau), not from clusters or pseudotime inferred on
    the complete evaluation data.
    """
    if target_rna.shape != output["rna_t"].shape:
        raise ValueError("target_rna has the wrong shape.")
    if not torch.isfinite(target_rna).all() or bool((target_rna < 0).any()):
        raise ValueError("target_rna must be finite and non-negative.")
    rna_loss = F.mse_loss(
        torch.log1p(output["rna_t"].clamp_min(0.0)),
        torch.log1p(target_rna.clamp_min(0.0)),
    )
    zero = rna_loss.new_zeros(())

    accessibility_loss = zero
    if target_accessibility is not None:
        if target_accessibility.shape != output["accessibility_t"].shape:
            raise ValueError("target_accessibility has the wrong shape.")
        if not torch.isfinite(target_accessibility).all() or bool(
            ((target_accessibility < 0) | (target_accessibility > 1)).any()
        ):
            raise ValueError(
                "target_accessibility must be finite and normalized to [0, 1]."
            )
        accessibility_loss = F.binary_cross_entropy(
            output["accessibility_t"].clamp(1e-6, 1.0 - 1e-6),
            target_accessibility.clamp(0.0, 1.0),
        )

    derivative_loss = zero
    if observed_derivative is not None:
        if observed_derivative.shape != output["terminal_velocity"].shape:
            raise ValueError("observed_derivative has the wrong shape.")
        if not torch.isfinite(observed_derivative).all():
            raise ValueError("observed_derivative must be finite.")
        derivative_loss = F.mse_loss(
            output["terminal_velocity"], observed_derivative
        )

    terminal_loss = zero
    if terminal_mask is not None:
        terminal_mask = terminal_mask.bool()
        if terminal_mask.shape != (target_rna.shape[0],):
            raise ValueError("terminal_mask must have one value per observation.")
        if bool(terminal_mask.any()):
            terminal_loss = output["terminal_velocity"][terminal_mask].square().mean()

    gain_regularization = zero
    if model is not None:
        layers = (
            model.field.signal_recurrent,
            model.field.signal_to_tf,
            model.field.tf_circuit,
            model.field.tf_to_peak,
            model.field.tf_to_gene,
        )
        gains = [layer.effective_gain() for layer in layers if layer.num_edges]
        if gains:
            gain_regularization = torch.cat(gains).mean()

    total = (
        rna_weight * rna_loss
        + accessibility_weight * accessibility_loss
        + derivative_weight * derivative_loss
        + terminal_weight * terminal_loss
        + gain_weight * gain_regularization
    )
    return {
        "total": total,
        "rna": rna_loss,
        "accessibility": accessibility_loss,
        "derivative": derivative_loss,
        "terminal_velocity": terminal_loss,
        "edge_gain": gain_regularization,
    }


def degree_preserving_signed_permutation(
    adjacency: Tensor,
    seed: int,
    swaps_per_edge: int = 20,
) -> Tensor:
    """Permute positive and negative edges separately, preserving all degrees."""
    if adjacency.ndim != 2:
        raise ValueError("adjacency must be rank two.")
    original = adjacency.detach().cpu().numpy()
    result = np.zeros_like(original)
    rng = np.random.default_rng(seed)
    occupied_result = set()

    for sign in (1, -1):
        coordinates = np.argwhere(np.sign(original) == sign)
        edges = [tuple(map(int, row)) for row in coordinates]
        magnitudes = [float(abs(original[row, col])) for row, col in edges]
        edge_set = set(edges)
        opposite_original = {
            tuple(map(int, row))
            for row in np.argwhere(np.sign(original) == -sign)
        }
        target_swaps = len(edges) * swaps_per_edge
        accepted = 0
        for _ in range(max(target_swaps * 20, 1000)):
            if accepted >= target_swaps or len(edges) < 2:
                break
            first, second = rng.choice(len(edges), size=2, replace=False)
            source_a, target_a = edges[first]
            source_b, target_b = edges[second]
            if source_a == source_b or target_a == target_b:
                continue
            new_a = (source_a, target_b)
            new_b = (source_b, target_a)
            forbidden = opposite_original.union(occupied_result)
            if (
                new_a in edge_set
                or new_b in edge_set
                or new_a in forbidden
                or new_b in forbidden
            ):
                continue
            edge_set.remove(edges[first])
            edge_set.remove(edges[second])
            edges[first], edges[second] = new_a, new_b
            edge_set.add(new_a)
            edge_set.add(new_b)
            accepted += 1
        for (source, target), magnitude in zip(edges, magnitudes):
            result[source, target] = sign * magnitude
        occupied_result.update(edges)

        original_mask = (np.sign(original) == sign).astype(int)
        result_mask = (np.sign(result) == sign).astype(int)
        if not np.array_equal(original_mask.sum(axis=0), result_mask.sum(axis=0)):
            raise RuntimeError("Permutation changed a signed target degree.")
        if not np.array_equal(original_mask.sum(axis=1), result_mask.sum(axis=1)):
            raise RuntimeError("Permutation changed a signed source degree.")

    if np.count_nonzero(original) >= 2 and np.array_equal(original, result):
        raise RuntimeError("Signed permutation could not change any edge.")
    return torch.as_tensor(result, dtype=adjacency.dtype, device=adjacency.device)


def architecture_contract(model: CircuitDynamicsModel) -> Dict[str, object]:
    """Machine-readable statement of the v3 mechanistic constraints."""
    forbidden = []
    for name, module in model.named_modules():
        if "residual" in name.lower() or isinstance(
            module, (nn.Linear, nn.MultiheadAttention)
        ):
            forbidden.append(name or module.__class__.__name__)
    return {
        "state_blocks": ["signaling", "TF activity", "chromatin", "RNA"],
        "initial_encoder": "localized motif accessibility; no dense identity encoder",
        "unsupported_edges": "structurally absent",
        "edge_signs": "fixed",
        "edge_kinetics": "positive gain, positive threshold, Hill coefficient > 1",
        "chromatin_role": "slow continuous state",
        "neural_bypass_modules": forbidden,
        "attractor_claim": "requires temporal/perturbation validation",
    }
