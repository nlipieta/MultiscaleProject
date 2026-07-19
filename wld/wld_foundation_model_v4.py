"""WLD v4: structured multimodal foundation model with circuit ODE dynamics.

The v4 model separates three ideas which were conflated in earlier prototypes:

1. Evidence defines *where information is allowed to travel*.
2. A structured multimodal encoder estimates biologically named initial states.
3. A context-conditioned ODE estimates how those states change in a cell.

The neural context code may modulate supported edge gains, production and decay
rates, but it is never decoded directly to RNA.  Every RNA prediction therefore
passes through TF-gene support gated by localized motifs, peak-gene contacts and
cell-specific accessibility.  Missing modalities are masked rather than filled
with inferred values and presented as measurements.

This module is a software architecture, not a pretrained biological model.
Biological claims require multi-study training and study/donor-sealed tests.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from wld_circuit_dynamics_v3 import EnhancerAccessibilityGate


FORBIDDEN_FEATURE_TOKENS = {
    "barcode",
    "celltype",
    "cell_type",
    "cluster",
    "donorid",
    "donor_id",
    "future",
    "integrated",
    "label",
    "leiden",
    "louvain",
    "outcome",
    "pseudotime",
    "seurat",
    "studyid",
    "study_id",
    "subjectid",
    "subject_id",
    "targetstate",
    "target_state",
    "umap",
}


def _tokens(value: object) -> set[str]:
    text = str(value).strip().lower()
    split = {item for item in re.split(r"[^a-z0-9]+", text) if item}
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return split | ({compact} if compact else set())


def audit_foundation_inputs(
    feature_names: Sequence[object],
    *,
    uses_time_zero_rna: bool = False,
    predicts_future: bool = True,
) -> Dict[str, object]:
    """Reject identity/outcome proxies from the encoder contract.

    RNA may be used during general multimodal pretraining or as a declared
    time-zero measurement.  Future RNA and labels are never legal inputs.
    """

    bad = []
    for name in feature_names:
        tokens = _tokens(name)
        if tokens.intersection(FORBIDDEN_FEATURE_TOKENS):
            bad.append(str(name))
    if bad:
        raise ValueError(f"Direct identity, batch or outcome proxies found: {bad}")
    if uses_time_zero_rna and not predicts_future:
        raise ValueError("Time-zero RNA is meaningful only for a future-state task.")
    return {
        "accepted_features": [str(value) for value in feature_names],
        "uses_time_zero_rna": bool(uses_time_zero_rna),
        "future_target_required_for_time_zero_rna": True,
    }


def _tensor(value: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _normalized_columns(value: Tensor) -> Tensor:
    value = value.float()
    return value / value.abs().sum(dim=0, keepdim=True).clamp_min(1.0)


def _inverse_softplus_scalar(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus initial values must be positive")
    return math.log(math.expm1(value))


@dataclass(frozen=True)
class FoundationPriors:
    """Hard topology and cross-modal projection evidence for WLD v4."""

    peak_to_gene: Tensor
    peak_tf_motif: Tensor
    tf_gene_support: Tensor
    circuit_tf_tf: Tensor
    signal_signal: Tensor
    signal_tf: Tensor
    cue_signal: Tensor
    tf_peak_effect: Tensor
    tf_gene_index: Tensor
    protein_signal: Tensor
    metabolic_signal: Tensor
    metabolic_tf: Tensor

    def validate(self) -> None:
        matrices = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "tf_gene_index"
        }
        for name, value in matrices.items():
            if not isinstance(value, Tensor) or value.ndim != 2:
                raise ValueError(f"{name} must be a rank-two tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

        peaks, genes = self.peak_to_gene.shape
        motif_peaks, tfs = self.peak_tf_motif.shape
        signals = self.signal_signal.shape[0]
        if peaks != motif_peaks:
            raise ValueError("peak_to_gene and peak_tf_motif disagree on peaks")
        expected = {
            "tf_gene_support": (tfs, genes),
            "circuit_tf_tf": (tfs, tfs),
            "signal_signal": (signals, signals),
            "signal_tf": (signals, tfs),
            "cue_signal": (self.cue_signal.shape[0], signals),
            "tf_peak_effect": (tfs, peaks),
            "protein_signal": (self.protein_signal.shape[0], signals),
            "metabolic_signal": (self.metabolic_signal.shape[0], signals),
            "metabolic_tf": (self.metabolic_tf.shape[0], tfs),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} has shape {tuple(getattr(self, name).shape)}; expected {shape}")
        if self.tf_gene_index.ndim != 1 or self.tf_gene_index.shape[0] != tfs:
            raise ValueError("tf_gene_index must have one element per TF")
        if self.tf_gene_index.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("tf_gene_index must contain integers")
        if bool((self.tf_gene_index < 0).any()) or bool(
            (self.tf_gene_index >= genes).any()
        ):
            raise ValueError("tf_gene_index contains an out-of-range gene")
        if bool((self.peak_to_gene < 0).any()) or bool(
            (self.peak_tf_motif < 0).any()
        ):
            raise ValueError("peak/contact and motif evidence must be non-negative")

        localized = torch.einsum(
            "pt,pg->tg", self.peak_tf_motif.float(), self.peak_to_gene.float()
        ) > 0
        unsupported = (self.tf_gene_support != 0) & ~localized
        if bool(unsupported.any()):
            raise ValueError(
                "Every TF-gene edge must have motif/occupancy and peak-gene support"
            )
        circuit_edges = torch.nonzero(self.circuit_tf_tf != 0, as_tuple=False)
        if circuit_edges.numel():
            source = circuit_edges[:, 0]
            target_tf = circuit_edges[:, 1]
            target_gene = self.tf_gene_index[target_tf].long()
            support = self.tf_gene_support[source, target_gene]
            if bool((support == 0).any()):
                raise ValueError("Every TF-circuit edge must regulate the target TF gene")
            if bool(
                (
                    torch.sign(self.circuit_tf_tf[source, target_tf])
                    != torch.sign(support)
                ).any()
            ):
                raise ValueError(
                    "Circuit-edge signs must match TF-to-target-TF-gene signs"
                )
        peak_effect_edges = torch.nonzero(
            self.tf_peak_effect != 0, as_tuple=False
        )
        if peak_effect_edges.numel():
            effect_tf = peak_effect_edges[:, 0]
            effect_peak = peak_effect_edges[:, 1]
            if bool((self.peak_tf_motif[effect_peak, effect_tf] <= 0).any()):
                raise ValueError(
                    "Every TF-to-peak effect must have localized binding evidence"
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

    @property
    def num_proteins(self) -> int:
        return int(self.protein_signal.shape[0])

    @property
    def num_metabolites(self) -> int:
        return int(self.metabolic_signal.shape[0])


class ContextualSparseHillLayer(nn.Module):
    """Signed Hill messages with cell-specific positive gain modulation.

    Edge existence and the curated base sign remain explicit.  Context can
    amplify or suppress an edge, including making it effectively absent, but
    cannot create a new unsupported connection.  Context-dependent sign models
    can be added only when a prior table explicitly supplies both directions.
    """

    def __init__(
        self,
        signed_adjacency: Tensor,
        context_dim: int,
        initial_gain: float = 0.25,
        max_log_gain_delta: float = 2.0,
    ) -> None:
        super().__init__()
        if signed_adjacency.ndim != 2:
            raise ValueError("signed_adjacency must be rank two")
        edges = torch.nonzero(signed_adjacency != 0, as_tuple=False)
        self.source_dim = int(signed_adjacency.shape[0])
        self.target_dim = int(signed_adjacency.shape[1])
        self.context_dim = int(context_dim)
        self.max_log_gain_delta = float(max_log_gain_delta)
        self.register_buffer("source_index", edges[:, 0].long())
        self.register_buffer("target_index", edges[:, 1].long())
        values = signed_adjacency[edges[:, 0], edges[:, 1]].float()
        confidence = values.abs()
        if confidence.numel():
            confidence = confidence / confidence.amax().clamp_min(1.0)
        self.register_buffer("edge_sign", torch.sign(values))
        self.register_buffer("edge_confidence", confidence)
        count = int(edges.shape[0])
        self.raw_gain = nn.Parameter(
            torch.full((count,), _inverse_softplus_scalar(initial_gain))
        )
        self.raw_threshold = nn.Parameter(
            torch.full((count,), _inverse_softplus_scalar(0.5))
        )
        self.raw_hill = nn.Parameter(
            torch.full((count,), _inverse_softplus_scalar(0.5))
        )
        self.context_gain = nn.Linear(context_dim, count, bias=False)
        nn.init.zeros_(self.context_gain.weight)

    @property
    def num_edges(self) -> int:
        return int(self.source_index.numel())

    def gain_scale(self, context: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context has the wrong shape")
        if not self.num_edges:
            return context.new_zeros((context.shape[0], 0))
        return torch.exp(
            self.max_log_gain_delta * torch.tanh(self.context_gain(context))
        )

    def forward(
        self,
        source: Tensor,
        context: Tensor,
        edge_gate: Optional[Tensor] = None,
        intervention_scale: Optional[Tensor] = None,
    ) -> Tensor:
        if source.ndim != 2 or source.shape[1] != self.source_dim:
            raise ValueError("source has the wrong shape")
        result = source.new_zeros((source.shape[0], self.target_dim))
        if not self.num_edges:
            return result
        concentration = source[:, self.source_index].clamp_min(0.0)
        threshold = F.softplus(self.raw_threshold).clamp_min(1e-4)
        hill = 1.0 + F.softplus(self.raw_hill)
        occupancy = torch.sigmoid(
            hill
            * (
                torch.log(concentration.clamp_min(1e-8))
                - torch.log(threshold)
            )
        )
        messages = (
            occupancy
            * self.edge_sign
            * self.edge_confidence
            * F.softplus(self.raw_gain)
            * self.gain_scale(context)
        )
        if edge_gate is not None:
            messages = messages * _edge_scale(
                edge_gate, source.shape[0], self.num_edges, source
            )
        if intervention_scale is not None:
            messages = messages * _edge_scale(
                intervention_scale, source.shape[0], self.num_edges, source
            )
        result.index_add_(1, self.target_index, messages)
        return result


def _edge_scale(value: Tensor, batch: int, edges: int, reference: Tensor) -> Tensor:
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if not torch.isfinite(value).all() or bool((value < 0).any()):
        raise ValueError("edge scales must be finite and non-negative")
    if value.ndim == 1:
        if value.shape[0] != edges:
            raise ValueError("edge scale has the wrong number of edges")
        return value.unsqueeze(0)
    if value.shape != (batch, edges):
        raise ValueError("edge scale must have shape [edges] or [batch, edges]")
    return value


class ContextRateAdapter(nn.Module):
    """Bounded cell/context variation around a shared population parameter."""

    def __init__(
        self,
        context_dim: int,
        output_dim: int,
        initial: float,
        max_delta: float,
        positive: bool,
    ) -> None:
        super().__init__()
        self.positive = bool(positive)
        self.max_delta = float(max_delta)
        if positive:
            self.base = nn.Parameter(
                torch.full((output_dim,), _inverse_softplus_scalar(initial))
            )
        else:
            self.base = nn.Parameter(torch.full((output_dim,), float(initial)))
        self.adapter = nn.Linear(context_dim, output_dim, bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, context: Tensor) -> Tensor:
        raw = self.base.unsqueeze(0) + self.max_delta * torch.tanh(
            self.adapter(context)
        )
        return F.softplus(raw).clamp_min(1e-4) if self.positive else raw


class StructuredMultimodalEncoder(nn.Module):
    """Project measured modalities into named signal/TF/chromatin variables."""

    def __init__(self, priors: FoundationPriors, context_covariate_dim: int) -> None:
        super().__init__()
        priors.validate()
        self.num_peaks = priors.num_peaks
        self.num_genes = priors.num_genes
        self.num_tfs = priors.num_tfs
        self.num_signals = priors.num_signals
        self.num_cues = priors.num_cues
        self.num_proteins = priors.num_proteins
        self.num_metabolites = priors.num_metabolites
        self.context_covariate_dim = int(context_covariate_dim)

        motif = _normalized_columns(priors.peak_tf_motif.clamp_min(0.0))
        rna_tf = _normalized_columns(priors.tf_gene_support.abs().transpose(0, 1))
        self.register_buffer("motif_projection", motif)
        self.register_buffer("rna_tf_projection", rna_tf)
        self.register_buffer(
            "protein_signal_projection", _normalized_columns(priors.protein_signal)
        )
        self.register_buffer(
            "metabolic_signal_projection", _normalized_columns(priors.metabolic_signal)
        )
        self.register_buffer(
            "metabolic_tf_projection", _normalized_columns(priors.metabolic_tf)
        )
        self.register_buffer(
            "cue_signal_projection", _normalized_columns(priors.cue_signal)
        )
        self.signal_bias = nn.Parameter(torch.full((priors.num_signals,), -1.5))
        self.tf_bias = nn.Parameter(torch.full((priors.num_tfs,), -1.5))
        self.motif_scale = nn.Parameter(torch.zeros(priors.num_tfs))
        self.rna_scale = nn.Parameter(torch.full((priors.num_tfs,), -2.0))

    @property
    def biological_context_dim(self) -> int:
        # signal, TF, motif, RNA-TF, metabolic-TF, four modality masks,
        # and declared measured context covariates.
        return (
            2 * self.num_signals
            + 3 * self.num_tfs
            + 4
            + self.context_covariate_dim
        )

    @staticmethod
    def _mask(
        value: Optional[Tensor],
        mask: Optional[Tensor],
        batch: int,
        width: int,
        reference: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if value is None:
            value = reference.new_zeros((batch, width))
            resolved = reference.new_zeros((batch, 1))
        else:
            value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
            if value.shape != (batch, width):
                raise ValueError(f"modality has shape {tuple(value.shape)}; expected {(batch, width)}")
            if not torch.isfinite(value).all():
                raise ValueError("modality contains non-finite values")
            resolved = reference.new_ones((batch, 1))
        if mask is not None:
            resolved = torch.as_tensor(mask, dtype=reference.dtype, device=reference.device)
            if resolved.ndim == 1:
                resolved = resolved.unsqueeze(1)
            if resolved.shape != (batch, 1) or bool(
                ((resolved < 0) | (resolved > 1)).any()
            ):
                raise ValueError("modality mask must have shape [batch, 1] in [0,1]")
        return value * resolved, resolved

    def forward(
        self,
        *,
        cues: Tensor,
        atac: Optional[Tensor] = None,
        rna: Optional[Tensor] = None,
        protein: Optional[Tensor] = None,
        metabolic: Optional[Tensor] = None,
        context_covariates: Optional[Tensor] = None,
        modality_masks: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if cues.ndim != 2 or cues.shape[1] != self.num_cues:
            raise ValueError("cues must have shape [batch, cues]")
        if not torch.isfinite(cues).all() or bool((cues < 0).any()):
            raise ValueError("cues must be finite and non-negative")
        batch = cues.shape[0]
        masks = dict(modality_masks or {})
        atac, atac_mask = self._mask(
            atac, masks.get("atac"), batch, self.num_peaks, cues
        )
        rna, rna_mask = self._mask(
            rna, masks.get("rna"), batch, self.num_genes, cues
        )
        protein, protein_mask = self._mask(
            protein, masks.get("protein"), batch, self.num_proteins, cues
        )
        metabolic, metabolic_mask = self._mask(
            metabolic, masks.get("metabolic"), batch, self.num_metabolites, cues
        )
        if bool((atac < 0).any()) or bool((atac > 1).any()):
            raise ValueError("ATAC must be represented in [0,1]")
        if bool((rna < 0).any()) or bool((protein < 0).any()) or bool(
            (metabolic < 0).any()
        ):
            raise ValueError("abundance modalities must be non-negative")

        motif = atac @ self.motif_projection
        rna_tf = torch.log1p(rna) @ self.rna_tf_projection
        protein_signal = protein @ self.protein_signal_projection
        metabolic_signal = metabolic @ self.metabolic_signal_projection
        metabolic_tf = metabolic @ self.metabolic_tf_projection
        cue_signal = cues @ self.cue_signal_projection
        signal = F.softplus(
            self.signal_bias
            + cue_signal
            + protein_signal
            + metabolic_signal
        )
        tf = F.softplus(
            self.tf_bias
            + F.softplus(self.motif_scale) * motif
            + F.softplus(self.rna_scale) * rna_tf
            + metabolic_tf
        )

        if context_covariates is None:
            context_covariates = cues.new_zeros(
                (batch, self.context_covariate_dim)
            )
        if context_covariates.shape != (batch, self.context_covariate_dim):
            raise ValueError("context_covariates has the wrong shape")
        if not torch.isfinite(context_covariates).all():
            raise ValueError("context_covariates contains non-finite values")
        biological_context = torch.cat(
            [
                signal,
                protein_signal + metabolic_signal,
                tf,
                motif,
                rna_tf,
                atac_mask,
                rna_mask,
                protein_mask,
                metabolic_mask,
                context_covariates,
            ],
            dim=1,
        )
        return {
            "signal": signal,
            "tf": tf,
            "accessibility": atac,
            "biological_context": biological_context,
            "modality_mask": torch.cat(
                [atac_mask, rna_mask, protein_mask, metabolic_mask], dim=1
            ),
            "rna_observed": rna,
        }


@dataclass(frozen=True)
class FoundationIntervention:
    circuit_edge_scale: Optional[Tensor] = None
    signaling_edge_scale: Optional[Tensor] = None
    signal_tf_edge_scale: Optional[Tensor] = None
    tf_gene_edge_scale: Optional[Tensor] = None
    tf_peak_edge_scale: Optional[Tensor] = None


class ContextConditionedCircuitField(nn.Module):
    """Mechanistic vector field whose kinetics vary by measured cell context."""

    def __init__(
        self,
        priors: FoundationPriors,
        context_dim: int,
    ) -> None:
        super().__init__()
        priors.validate()
        self.num_signals = priors.num_signals
        self.num_tfs = priors.num_tfs
        self.num_peaks = priors.num_peaks
        self.num_genes = priors.num_genes
        self.state_dim = (
            self.num_signals + self.num_tfs + self.num_peaks + self.num_genes
        )
        self.register_buffer("tf_gene_index", priors.tf_gene_index.long())
        self.accessibility_gate = EnhancerAccessibilityGate(
            priors.peak_to_gene,
            priors.peak_tf_motif,
            priors.tf_gene_support,
        )
        self.cue_signal = ContextualSparseHillLayer(
            priors.cue_signal, context_dim
        )
        self.signal_recurrent = ContextualSparseHillLayer(
            priors.signal_signal, context_dim
        )
        self.signal_to_tf = ContextualSparseHillLayer(
            priors.signal_tf, context_dim
        )
        self.tf_circuit = ContextualSparseHillLayer(
            priors.circuit_tf_tf, context_dim
        )
        self.tf_to_peak = ContextualSparseHillLayer(
            priors.tf_peak_effect, context_dim
        )
        self.tf_to_gene = ContextualSparseHillLayer(
            priors.tf_gene_support, context_dim
        )

        self.signal_basal = ContextRateAdapter(
            context_dim, priors.num_signals, -1.5, 1.5, False
        )
        self.tf_basal = ContextRateAdapter(
            context_dim, priors.num_tfs, -1.5, 1.5, False
        )
        self.chromatin_basal = ContextRateAdapter(
            context_dim, priors.num_peaks, 0.0, 2.0, False
        )
        self.rna_basal = ContextRateAdapter(
            context_dim, priors.num_genes, -1.5, 1.5, False
        )
        self.signal_decay = ContextRateAdapter(
            context_dim, priors.num_signals, 0.7, 1.0, True
        )
        self.tf_decay = ContextRateAdapter(
            context_dim, priors.num_tfs, 0.7, 1.0, True
        )
        self.rna_decay = ContextRateAdapter(
            context_dim, priors.num_genes, 0.7, 1.0, True
        )
        self.chromatin_timescale = ContextRateAdapter(
            context_dim, priors.num_peaks, 0.05, 1.0, True
        )

    def split_state(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError("state has the wrong shape")
        s = self.num_signals
        t = s + self.num_tfs
        p = t + self.num_peaks
        return state[:, :s], state[:, s:t], state[:, t:p], state[:, p:]

    @staticmethod
    def join_state(signal: Tensor, tf: Tensor, atac: Tensor, rna: Tensor) -> Tensor:
        return torch.cat([signal, tf, atac, rna], dim=1)

    def components(
        self,
        state: Tensor,
        cues: Tensor,
        context: Tensor,
        intervention: Optional[FoundationIntervention] = None,
    ) -> Dict[str, Tensor]:
        signal, tf, accessibility, rna = self.split_state(state)
        intervention = intervention or FoundationIntervention()
        cue_drive = self.cue_signal(cues, context)
        signal_drive = self.signal_recurrent(
            signal.clamp_min(0.0),
            context,
            intervention_scale=intervention.signaling_edge_scale,
        )
        signal_production = F.softplus(
            self.signal_basal(context) + cue_drive + signal_drive
        )
        d_signal = signal_production - self.signal_decay(context) * signal

        gates = self.accessibility_gate(accessibility.clamp(0.0, 1.0))
        if self.tf_circuit.num_edges:
            circuit_gate = gates[
                :,
                self.tf_circuit.source_index,
                self.tf_gene_index[self.tf_circuit.target_index],
            ]
        else:
            circuit_gate = state.new_zeros((state.shape[0], 0))
        tf_drive = self.signal_to_tf(
            signal.clamp_min(0.0),
            context,
            intervention_scale=intervention.signal_tf_edge_scale,
        )
        tf_drive = tf_drive + self.tf_circuit(
            tf.clamp_min(0.0),
            context,
            edge_gate=circuit_gate,
            intervention_scale=intervention.circuit_edge_scale,
        )
        tf_production = F.softplus(self.tf_basal(context) + tf_drive)
        d_tf = tf_production - self.tf_decay(context) * tf

        chromatin_drive = self.tf_to_peak(
            tf.clamp_min(0.0),
            context,
            intervention_scale=intervention.tf_peak_edge_scale,
        )
        chromatin_target = torch.sigmoid(
            self.chromatin_basal(context) + chromatin_drive
        )
        d_accessibility = self.chromatin_timescale(context) * (
            chromatin_target - accessibility
        )

        if self.tf_to_gene.num_edges:
            rna_gate = gates[
                :, self.tf_to_gene.source_index, self.tf_to_gene.target_index
            ]
        else:
            rna_gate = state.new_zeros((state.shape[0], 0))
        rna_drive = self.tf_to_gene(
            tf.clamp_min(0.0),
            context,
            edge_gate=rna_gate,
            intervention_scale=intervention.tf_gene_edge_scale,
        )
        rna_production = F.softplus(self.rna_basal(context) + rna_drive)
        d_rna = rna_production - self.rna_decay(context) * rna
        return {
            "derivative": self.join_state(d_signal, d_tf, d_accessibility, d_rna),
            "signal_production": signal_production,
            "tf_production": tf_production,
            "chromatin_target": chromatin_target,
            "rna_production": rna_production,
            "circuit_gate": circuit_gate,
            "rna_gate": rna_gate,
        }

    def forward(
        self,
        state: Tensor,
        cues: Tensor,
        context: Tensor,
        intervention: Optional[FoundationIntervention] = None,
    ) -> Tensor:
        return self.components(state, cues, context, intervention)["derivative"]


class WLDMultistudyFoundationModel(nn.Module):
    """Structured multimodal encoder followed by a hard-topology neural ODE."""

    def __init__(
        self,
        priors: FoundationPriors,
        *,
        context_covariate_dim: int = 0,
        context_dim: int = 32,
    ) -> None:
        super().__init__()
        priors.validate()
        self.priors = priors
        self.encoder = StructuredMultimodalEncoder(
            priors, context_covariate_dim
        )
        self.context_network = nn.Sequential(
            nn.LayerNorm(self.encoder.biological_context_dim),
            nn.Linear(self.encoder.biological_context_dim, context_dim),
            nn.Tanh(),
            nn.Linear(context_dim, context_dim),
            nn.Tanh(),
        )
        self.field = ContextConditionedCircuitField(priors, context_dim)
        self.num_genes = priors.num_genes
        self.num_peaks = priors.num_peaks
        self.num_tfs = priors.num_tfs
        self.num_signals = priors.num_signals
        self.context_dim = int(context_dim)

    def initial_state(
        self,
        encoded: Mapping[str, Tensor],
        cues: Tensor,
        context: Tensor,
        initial_rna: Optional[Tensor],
    ) -> Tensor:
        signal = encoded["signal"]
        tf = encoded["tf"]
        accessibility = encoded["accessibility"]
        if initial_rna is not None:
            initial_rna = torch.as_tensor(
                initial_rna, dtype=cues.dtype, device=cues.device
            )
            if initial_rna.shape != (cues.shape[0], self.num_genes):
                raise ValueError("initial_rna has the wrong shape")
            if not torch.isfinite(initial_rna).all() or bool(
                (initial_rna < 0).any()
            ):
                raise ValueError("initial_rna must be finite and non-negative")
            rna = initial_rna
        else:
            zero = cues.new_zeros((cues.shape[0], self.num_genes))
            probe = self.field.join_state(signal, tf, accessibility, zero)
            production = self.field.components(
                probe, cues, context
            )["rna_production"]
            rna = production / self.field.rna_decay(context)
        return self.field.join_state(signal, tf, accessibility, rna)

    def forward(
        self,
        *,
        cues: Tensor,
        horizon: float,
        steps: int,
        atac: Optional[Tensor] = None,
        rna_encoder_input: Optional[Tensor] = None,
        protein: Optional[Tensor] = None,
        metabolic: Optional[Tensor] = None,
        context_covariates: Optional[Tensor] = None,
        modality_masks: Optional[Mapping[str, Tensor]] = None,
        initial_rna: Optional[Tensor] = None,
        intervention: Optional[FoundationIntervention] = None,
    ) -> Dict[str, Tensor]:
        if horizon <= 0 or not math.isfinite(float(horizon)) or steps < 1:
            raise ValueError("horizon must be finite and positive and steps >= 1")
        encoded = self.encoder(
            cues=cues,
            atac=atac,
            rna=rna_encoder_input,
            protein=protein,
            metabolic=metabolic,
            context_covariates=context_covariates,
            modality_masks=modality_masks,
        )
        context = self.context_network(encoded["biological_context"])
        state = self.initial_state(encoded, cues, context, initial_rna)
        path = [state]
        dt = float(horizon) / int(steps)
        for _ in range(int(steps)):
            k1 = self.field(state, cues, context, intervention)
            k2 = self.field(state + 0.5 * dt * k1, cues, context, intervention)
            k3 = self.field(state + 0.5 * dt * k2, cues, context, intervention)
            k4 = self.field(state + dt * k3, cues, context, intervention)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            path.append(state)
        stacked = torch.stack(path, dim=1)
        signal, tf, accessibility, rna = self.field.split_state(state)
        terminal = self.field.components(state, cues, context, intervention)
        return {
            "signal_t": signal.clamp_min(0.0),
            "tf_t": tf.clamp_min(0.0),
            "accessibility_t": accessibility.clamp(0.0, 1.0),
            "rna_t": rna.clamp_min(0.0),
            "terminal_velocity": terminal["derivative"],
            "path": stacked,
            "context": context,
            "modality_mask": encoded["modality_mask"],
            "circuit_gain_scale": self.field.tf_circuit.gain_scale(context),
            "tf_gene_gain_scale": self.field.tf_to_gene.gain_scale(context),
            "rna_decay": self.field.rna_decay(context),
            "tf_decay": self.field.tf_decay(context),
        }


def no_circuit_priors(priors: FoundationPriors) -> FoundationPriors:
    values = {field.name: getattr(priors, field.name) for field in fields(priors)}
    values["circuit_tf_tf"] = torch.zeros_like(priors.circuit_tf_tf)
    result = FoundationPriors(**values)
    result.validate()
    return result


def supported_sign_shuffle_priors(
    priors: FoundationPriors, seed: int
) -> FoundationPriors:
    """Shuffle signs only among supported TF-circuit edges."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    circuit = priors.circuit_tf_tf.clone()
    edges = torch.nonzero(circuit != 0, as_tuple=False)
    if edges.shape[0] > 1:
        signs = torch.sign(circuit[edges[:, 0], edges[:, 1]])
        order = torch.randperm(edges.shape[0], generator=generator)
        shuffled = signs[order] * circuit[edges[:, 0], edges[:, 1]].abs()
        circuit[edges[:, 0], edges[:, 1]] = shuffled
    # A circuit sign must remain compatible with its target-TF gene edge.  The
    # corresponding direct support sign is changed in lockstep while topology,
    # motif evidence and contacts remain fixed.
    tf_gene = priors.tf_gene_support.clone()
    for source, target_tf in edges.tolist():
        target_gene = int(priors.tf_gene_index[target_tf])
        tf_gene[source, target_gene] = (
            circuit[source, target_tf].sign()
            * tf_gene[source, target_gene].abs()
        )
    values = {field.name: getattr(priors, field.name) for field in fields(priors)}
    values["circuit_tf_tf"] = circuit
    values["tf_gene_support"] = tf_gene
    result = FoundationPriors(**values)
    result.validate()
    return result


def architecture_contract(model: WLDMultistudyFoundationModel) -> Dict[str, object]:
    """Machine-readable statement of the v4 scientific architecture."""

    # The context MLP ends in context_dim and the only gene-dimensional
    # adapters live inside the explicitly named mechanistic rate field.  This
    # targeted check remains valid even when context_dim happens to equal the
    # number of modeled genes.
    direct_decoders = [
        name
        for name, _module in model.named_modules()
        if any(token in name.lower() for token in ("decoder", "residual_dynamics"))
    ]
    if direct_decoders:
        raise RuntimeError(f"Direct neural output bypass found: {direct_decoders}")
    return {
        "structured_modalities": ["ATAC", "RNA_optional", "protein_optional", "metabolic_optional", "cue"],
        "named_latent_state": ["signal", "TF", "chromatin", "RNA"],
        "context_conditioned_edge_gains": True,
        "context_conditioned_production_and_decay": True,
        "context_conditioned_chromatin_timescale": True,
        "hard_sparse_topology": True,
        "fixed_evidence_sign_orientation": True,
        "direct_context_to_rna_decoder": False,
        "missing_modalities_masked": True,
        "neural_residual_dynamics": False,
    }


__all__ = [
    "FoundationIntervention",
    "FoundationPriors",
    "WLDMultistudyFoundationModel",
    "architecture_contract",
    "audit_foundation_inputs",
    "no_circuit_priors",
    "supported_sign_shuffle_priors",
]
