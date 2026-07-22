"""Atlas-conditioned mechanistic virtual-tissue scaffold for WLD v6.0.

This module is intentionally pure NumPy.  It defines a small, auditable core
for the biological contract

    cue -> signal -> TF -> enhancer -> gene.

The graph stores only node identities, genomic coordinates and candidate-edge
masks.  Accessibility, TF availability, enhancer--gene contact, edge strength,
edge sign and kinetic gain are supplied for each context.  Consequently, a
candidate edge is not assumed to be active, activating, or equally fast in two
cell types, tissues, subjects or states.

This is a software scaffold, not a trained digital twin or an attractor claim.
It deliberately has no cell-type/cluster/pseudotime encoder input and no direct
cue-to-RNA or neural-to-RNA path.  All RNA change must traverse the typed graph.
The current route ``kinetics`` fields are instantaneous rate multipliers, not
learned or declared time delays; downstream contracts must not claim explicit
delay identification until a history-aware delay model is implemented.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


Array = np.ndarray
ArrayLike = Union[Array, Sequence[float], float]


class ProvenanceState(str, Enum):
    """Permitted evidence states for every assimilated feature."""

    OBSERVED = "observed"
    REFERENCE_TRANSFERRED = "reference_transferred"
    MODEL_INFERRED = "model_inferred"
    UNKNOWN = "unknown"


DEFAULT_PROVENANCE_WEIGHTS: Dict[str, float] = {
    ProvenanceState.OBSERVED.value: 1.0,
    ProvenanceState.REFERENCE_TRANSFERRED.value: 0.65,
    ProvenanceState.MODEL_INFERRED.value: 0.35,
    ProvenanceState.UNKNOWN.value: 0.0,
}

ALLOWED_DEVELOPMENT_SOURCE_ROLES = {
    "development", "training", "reference_atlas", "model_inference",
    "synthetic_fixture",
}
FORBIDDEN_SOURCE_ROLE_TOKENS = {
    "sealed", "test", "testing", "validation", "holdout", "heldout",
    "evaluation",
}
SEALED_SOURCE_ACCESSIONS = {"GSE315993"}


def _state_name(value: object) -> str:
    if isinstance(value, ProvenanceState):
        return value.value
    return str(value).strip().lower()


def _validate_source_access(
    *,
    source_role: object,
    source_partition: object,
    source_accessions: Sequence[object],
    source_lineage: Sequence[object],
    source_feature_names: Sequence[object],
    method_lineage: object,
    label: str,
) -> None:
    """Reject sealed/evaluation evidence at every model-consumption boundary."""

    role = str(source_role or "").strip().lower()
    if role not in ALLOWED_DEVELOPMENT_SOURCE_ROLES:
        raise ValueError(
            f"{label} requires an explicit development-safe source_role; got {role!r}"
        )
    partition = str(source_partition or "").strip()
    if not partition:
        raise ValueError(f"{label} requires an explicit source_partition")
    if isinstance(source_accessions, str) or not source_accessions:
        raise ValueError(f"{label} requires non-empty source_accessions")
    accessions = tuple(str(item).strip().upper() for item in source_accessions)
    if any(not item for item in accessions) or len(set(accessions)) != len(accessions):
        raise ValueError(f"{label} source_accessions must be non-empty and unique")
    forbidden_accessions = sorted(SEALED_SOURCE_ACCESSIONS.intersection(accessions))
    if forbidden_accessions:
        raise ValueError(
            f"{label} cannot consume sealed source accessions: {forbidden_accessions}"
        )
    lineage_text = " ".join(str(item) for item in source_lineage)
    feature_text = " ".join(str(item) for item in source_feature_names)
    method_text = str(method_lineage or "")
    access_text = " ".join(
        (role, partition, " ".join(accessions), lineage_text, feature_text, method_text)
    )
    access_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", access_text.lower())
        if token
    }
    if FORBIDDEN_SOURCE_ROLE_TOKENS.intersection(access_tokens):
        raise ValueError(
            f"{label} source metadata identifies a sealed/evaluation partition"
        )
    compact_metadata = re.sub(r"[^A-Z0-9]+", "", access_text.upper())
    if any(accession in compact_metadata for accession in SEALED_SOURCE_ACCESSIONS):
        raise ValueError(f"{label} metadata names a sealed source accession")


def _finite_array(value: ArrayLike, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _broadcast(value: ArrayLike, shape: Tuple[int, ...], name: str) -> Array:
    array = _finite_array(value, name)
    try:
        return np.array(np.broadcast_to(array, shape), dtype=np.float64, copy=True)
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {array.shape} cannot broadcast to {shape}"
        ) from exc


def _jsonable(value: object) -> object:
    """Convert diagnostics to strict JSON-compatible builtins."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("diagnostics contain a non-finite float")
    return value


@dataclass(frozen=True)
class ProvenanceTensor:
    """A measured or inferred tensor with element-wise provenance.

    ``states`` and ``uncertainty`` may be scalars or broadcastable to
    ``values``.  Sampling is deterministic for a supplied seed.  Unknown
    entries are always zeroed and carry zero evidence weight; they are never
    silently imputed and relabeled as observations.
    """

    values: Array
    states: object = ProvenanceState.OBSERVED.value
    uncertainty: ArrayLike = 0.0
    weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PROVENANCE_WEIGHTS)
    )
    source_feature_names: Tuple[str, ...] = ()
    source_lineage: Tuple[str, ...] = ()
    method_lineage: Optional[str] = None
    measurement_time: Optional[float] = None
    source_role: Optional[str] = None
    source_partition: Optional[str] = None
    source_accessions: Tuple[str, ...] = ()

    def _normalized(self) -> Tuple[Array, Array, Array, Dict[str, float]]:
        values = np.asarray(self.values, dtype=np.float64)
        raw_states = np.asarray(self.states, dtype=object)
        try:
            states = np.array(np.broadcast_to(raw_states, values.shape), dtype=object)
        except ValueError as exc:
            raise ValueError("provenance states do not broadcast to values") from exc
        states = np.vectorize(_state_name, otypes=[object])(states)
        allowed = {item.value for item in ProvenanceState}
        invalid = sorted({str(item) for item in np.unique(states)} - allowed)
        if invalid:
            raise ValueError(f"unknown provenance states: {invalid}")
        uncertainty = _broadcast(self.uncertainty, values.shape, "uncertainty")
        if bool((uncertainty < 0.0).any()):
            raise ValueError("uncertainty must be non-negative")
        weights = {_state_name(key): float(value) for key, value in self.weights.items()}
        if set(weights) != allowed:
            raise ValueError("weights must define exactly the four provenance states")
        if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in weights.values()):
            raise ValueError("provenance weights must be finite and lie in [0, 1]")
        if weights[ProvenanceState.UNKNOWN.value] != 0.0:
            raise ValueError("unknown provenance must have zero weight")
        if self.measurement_time is None or not math.isfinite(
            float(self.measurement_time)
        ):
            raise ValueError(
                "ProvenanceTensor requires an explicit finite measurement_time"
            )
        if isinstance(self.source_feature_names, str) or not self.source_feature_names or any(
            not str(name).strip() for name in self.source_feature_names
        ):
            raise ValueError("ProvenanceTensor requires source_feature_names")
        if len(set(self.source_feature_names)) != len(self.source_feature_names):
            raise ValueError("source_feature_names must be unique")
        if isinstance(self.source_lineage, str) or not self.source_lineage or any(
            not str(item).strip() for item in self.source_lineage
        ):
            raise ValueError("ProvenanceTensor requires non-empty source_lineage")
        _validate_source_access(
            source_role=self.source_role,
            source_partition=self.source_partition,
            source_accessions=self.source_accessions,
            source_lineage=self.source_lineage,
            source_feature_names=self.source_feature_names,
            method_lineage=self.method_lineage,
            label="ProvenanceTensor",
        )
        derived = np.isin(
            states,
            [
                ProvenanceState.REFERENCE_TRANSFERRED.value,
                ProvenanceState.MODEL_INFERRED.value,
            ],
        )
        if bool(derived.any()) and not str(self.method_lineage or "").strip():
            raise ValueError(
                "reference-transferred/model-inferred values require method_lineage"
            )
        known = states != ProvenanceState.UNKNOWN.value
        if not np.isfinite(values[known]).all():
            raise ValueError("known feature values must be finite")
        return values, states, uncertainty, weights

    def validate(self) -> None:
        self._normalized()

    def validate_for_prediction_origin(
        self, prediction_origin_time: float, label: str = "feature"
    ) -> None:
        """Reject measurements acquired after the prediction origin."""

        self.validate()
        if not math.isfinite(float(prediction_origin_time)):
            raise ValueError("prediction_origin_time must be finite")
        assert self.measurement_time is not None
        if float(self.measurement_time) > float(prediction_origin_time):
            raise ValueError(
                f"{label} was measured at {self.measurement_time}, after prediction "
                f"origin {prediction_origin_time}"
            )

    def sample(self, seed: int) -> np.ma.MaskedArray:
        values, states, uncertainty, _ = self._normalized()
        rng = np.random.default_rng(int(seed))
        safe = np.where(np.isfinite(values), values, 0.0)
        sampled = safe + rng.normal(size=values.shape) * uncertainty
        unknown = states == ProvenanceState.UNKNOWN.value
        return np.ma.MaskedArray(sampled, mask=unknown, copy=False)

    def effective_values(self, seed: Optional[int] = None) -> np.ma.MaskedArray:
        """Return physical values without confidence-weight rescaling.

        Evidence weights are returned separately by :meth:`evidence_weights`.
        Unknown entries remain explicitly masked rather than being encoded as
        numerical zero.
        """

        values, states, _, _ = self._normalized()
        if seed is None:
            sampled = np.where(np.isfinite(values), values, 0.0)
            return np.ma.MaskedArray(
                sampled,
                mask=states == ProvenanceState.UNKNOWN.value,
                copy=False,
            )
        return self.sample(int(seed))

    def physical_values(self, seed: Optional[int] = None) -> np.ma.MaskedArray:
        """Explicit alias for unscaled, provenance-masked physical values."""

        return self.effective_values(seed)

    def known_mask(self) -> Array:
        """Return a mask that distinguishes measured zero from missing data."""

        _, states, _, _ = self._normalized()
        return np.asarray(states != ProvenanceState.UNKNOWN.value, dtype=bool)

    def evidence_weights(self) -> Array:
        """Return the element-wise evidence weight without altering values."""

        _, states, _, weights = self._normalized()
        return np.asarray(
            np.vectorize(weights.__getitem__, otypes=[float])(states),
            dtype=np.float64,
        )

    def diagnostics(self) -> Dict[str, object]:
        _, states, uncertainty, weights = self._normalized()
        return {
            "shape": list(states.shape),
            "counts": {
                item.value: int(np.sum(states == item.value)) for item in ProvenanceState
            },
            "weights": dict(weights),
            "mean_uncertainty": float(np.mean(uncertainty)) if uncertainty.size else 0.0,
            "source_feature_names": list(self.source_feature_names),
            "source_lineage": list(self.source_lineage),
            "method_lineage": self.method_lineage,
            "measurement_time": float(self.measurement_time),
            "source_role": self.source_role,
            "source_partition": self.source_partition,
            "source_accessions": list(self.source_accessions),
            "physical_values_scaled_by_evidence_weight": False,
        }


@dataclass(frozen=True)
class CausalFeatureDeclaration:
    """Declare a causal input that will genuinely exist at inference time."""

    name: str
    kind: str  # ``cue`` or ``spatial_coordinate``
    measured_time: float
    available_at_inference: bool = True
    causal: bool = True

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ValueError("causal feature name cannot be empty")
        if self.kind not in {"cue", "spatial_coordinate"}:
            raise ValueError("causal feature kind must be cue or spatial_coordinate")
        if not math.isfinite(float(self.measured_time)):
            raise ValueError("causal feature measured_time must be finite")
        if not self.available_at_inference or not self.causal:
            raise ValueError(
                "causal inputs must be declared causal and available at inference"
            )


def _feature_tokens(name: object) -> Tuple[str, ...]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name)).lower()
    return tuple(token for token in re.split(r"[^a-z0-9]+", text) if token)


def audit_encoder_features(
    feature_names: Sequence[object],
    declarations: Sequence[CausalFeatureDeclaration] = (),
    *,
    prediction_origin_time: float = 0.0,
) -> Dict[str, object]:
    """Reject identity/state/outcome proxies and undeclared causal inputs."""

    if not math.isfinite(float(prediction_origin_time)):
        raise ValueError("prediction_origin_time must be finite")
    declaration_map: Dict[str, CausalFeatureDeclaration] = {}
    for declaration in declarations:
        declaration.validate()
        if float(declaration.measured_time) > float(prediction_origin_time):
            raise ValueError(
                f"causal feature {declaration.name!r} was measured after the "
                "prediction origin"
            )
        if declaration.name in declaration_map:
            raise ValueError(f"duplicate causal declaration: {declaration.name}")
        declaration_map[declaration.name] = declaration

    blocked_singletons = {
        "celltype", "cluster", "pseudotime", "label", "outcome", "leiden",
        "louvain", "trajectory", "seurat", "umap", "barcode", "batch",
        "condition", "tissue", "target", "guide", "donorid", "subjectid",
        "studyid", "integrated", "timepoint", "post", "cellstate",
        "statescore", "donor", "subject", "study", "identity", "lineage",
        "embedding",
        # Identity keys remain leakage even when the biological role is hidden
        # behind a generic table/metadata name.  In particular, a suffix such
        # as ``_id`` is tokenized to the singleton ``id`` and therefore cannot
        # be used to smuggle a donor lookup into a hierarchical context map.
        "id", "identifier", "animal", "patient", "sample", "specimen",
        "individual", "participant", "person", "case", "class", "cohort",
        "group", "replicate", "library", "plate", "well", "lane", "run",
        "biosample",
    }
    blocked_pairs = {
        ("cell", "type"), ("cell", "identity"), ("future", "state"),
        ("target", "state"), ("target", "rna"), ("rna", "target"),
        ("future", "rna"), ("future", "expression"),
        ("donor", "id"), ("subject", "id"), ("study", "id"),
        ("guide", "id"), ("target", "id"), ("tissue", "type"),
        ("cell", "state"), ("state", "score"), ("rna", "counts"),
        ("future", "atac"),
        ("subject", "embedding"), ("lineage", "identity"),
    }
    causal_tokens = {
        "cue", "stimulus", "stimulation", "ligand", "perturbation",
        "coordinate", "coordinates", "position", "spatial",
    }
    bad: list[str] = []
    undeclared: list[str] = []
    accepted: list[str] = []
    for raw_name in feature_names:
        name = str(raw_name)
        tokens = _feature_tokens(name)
        pairs = set(zip(tokens, tokens[1:]))
        compact = "".join(tokens)
        if (
            blocked_singletons.intersection(tokens)
            or blocked_pairs.intersection(pairs)
            or any(
                accession.lower() in compact
                for accession in SEALED_SOURCE_ACCESSIONS
            )
            or compact in {
                "celltype", "cellidentity", "futurestate", "targetstate",
                "targetrna", "rnatarget", "donorid", "subjectid", "studyid",
                "guideid", "targetid", "tissuetype",
                "cellstate", "statescore", "rnacounts", "futureatac",
                "animalid", "patientid", "sampleid", "specimenid",
                "individualid", "participantid", "personid", "caseid",
                "classid", "cohortid", "groupid", "replicateid",
                "libraryid", "plateid", "wellid", "laneid", "runid",
                "biosampleid",
            }
        ):
            bad.append(name)
            continue
        looks_causal = bool(causal_tokens.intersection(tokens))
        if looks_causal and name not in declaration_map:
            undeclared.append(name)
            continue
        accepted.append(name)
    if bad:
        raise ValueError(f"direct identity/state/outcome proxies found: {bad}")
    if undeclared:
        raise ValueError(f"causal inputs require explicit declarations: {undeclared}")
    return {
        "accepted_features": accepted,
        "causal_declarations": [
            {
                "name": item.name,
                "kind": item.kind,
                "measured_time": float(item.measured_time),
                "available_at_inference": True,
            }
            for item in declarations
        ],
        "direct_identity_or_state_proxies": [],
        "prediction_origin_time": float(prediction_origin_time),
    }


@dataclass(frozen=True)
class GenomicCoordinate:
    """Hard-fixed genomic identity/coordinate, never a trainable feature."""

    chrom: str
    start: int
    end: int
    strand: str = "."

    def validate(self) -> None:
        if not str(self.chrom).strip():
            raise ValueError("chromosome cannot be empty")
        if int(self.start) < 0 or int(self.end) <= int(self.start):
            raise ValueError("genomic coordinate must satisfy 0 <= start < end")
        if self.strand not in {"+", "-", "."}:
            raise ValueError("strand must be '+', '-' or '.'")


def _mask(value: ArrayLike, shape: Tuple[int, int], name: str) -> Array:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.isin(array, [0, 1, False, True]).all():
        raise ValueError(f"{name} must be a binary candidate mask, not edge weights")
    return np.asarray(array, dtype=bool)


@dataclass(frozen=True)
class RegulatoryGraph:
    """Typed candidate topology with no context-frozen biological weights."""

    cue_names: Tuple[str, ...]
    signal_names: Tuple[str, ...]
    tf_names: Tuple[str, ...]
    enhancer_names: Tuple[str, ...]
    gene_names: Tuple[str, ...]
    cue_signal_mask: Array
    signal_tf_mask: Array
    tf_enhancer_mask: Array
    enhancer_gene_mask: Array
    enhancer_coordinates: Tuple[GenomicCoordinate, ...]
    gene_coordinates: Tuple[GenomicCoordinate, ...]
    signal_signal_mask: Optional[Array] = None
    tf_tf_mask: Optional[Array] = None
    metabolism_names: Tuple[str, ...] = ()
    metabolism_signal_mask: Optional[Array] = None
    metabolism_tf_mask: Optional[Array] = None
    neighbor_signal_mask: Optional[Array] = None

    @property
    def dimensions(self) -> Dict[str, int]:
        return {
            "cues": len(self.cue_names),
            "signals": len(self.signal_names),
            "tfs": len(self.tf_names),
            "enhancers": len(self.enhancer_names),
            "genes": len(self.gene_names),
            "metabolites": len(self.metabolism_names),
        }

    @property
    def route_masks(self) -> Dict[str, Array]:
        d = self.dimensions
        routes = {
            "cue_signal": _mask(self.cue_signal_mask, (d["cues"], d["signals"]), "cue_signal_mask"),
            "signal_tf": _mask(self.signal_tf_mask, (d["signals"], d["tfs"]), "signal_tf_mask"),
            "tf_enhancer": _mask(self.tf_enhancer_mask, (d["tfs"], d["enhancers"]), "tf_enhancer_mask"),
            "enhancer_gene": _mask(self.enhancer_gene_mask, (d["enhancers"], d["genes"]), "enhancer_gene_mask"),
        }
        if self.signal_signal_mask is not None:
            routes["signal_signal"] = _mask(self.signal_signal_mask, (d["signals"], d["signals"]), "signal_signal_mask")
        if self.tf_tf_mask is not None:
            routes["tf_tf"] = _mask(self.tf_tf_mask, (d["tfs"], d["tfs"]), "tf_tf_mask")
        if self.metabolism_signal_mask is not None:
            routes["metabolism_signal"] = _mask(self.metabolism_signal_mask, (d["metabolites"], d["signals"]), "metabolism_signal_mask")
        if self.metabolism_tf_mask is not None:
            routes["metabolism_tf"] = _mask(self.metabolism_tf_mask, (d["metabolites"], d["tfs"]), "metabolism_tf_mask")
        if self.neighbor_signal_mask is not None:
            routes["neighbor_signal"] = _mask(
                self.neighbor_signal_mask,
                (d["signals"], d["signals"]),
                "neighbor_signal_mask",
            )
        return routes

    def validate(self) -> None:
        d = self.dimensions
        for key, names in {
            "cue": self.cue_names, "signal": self.signal_names, "TF": self.tf_names,
            "enhancer": self.enhancer_names, "gene": self.gene_names,
            "metabolism": self.metabolism_names,
        }.items():
            if len(set(names)) != len(names) or any(not str(name).strip() for name in names):
                raise ValueError(f"{key} node names must be non-empty and unique")
        if any(d[key] <= 0 for key in ("cues", "signals", "tfs", "enhancers", "genes")):
            raise ValueError("the primary cue-to-gene graph cannot contain an empty layer")
        if len(self.enhancer_coordinates) != d["enhancers"]:
            raise ValueError("enhancer_coordinates must have one entry per enhancer")
        if len(self.gene_coordinates) != d["genes"]:
            raise ValueError("gene_coordinates must have one entry per gene")
        for coordinate in self.enhancer_coordinates + self.gene_coordinates:
            coordinate.validate()
        self.route_masks
        if d["metabolites"] == 0 and (
            self.metabolism_signal_mask is not None or self.metabolism_tf_mask is not None
        ):
            raise ValueError("metabolic masks require named metabolic nodes")


@dataclass(frozen=True)
class HierarchicalField:
    """Global + lineage + tissue + state + subject edge effects.

    Zero-valued raw scalars/arrays are permitted as inactive defaults.  Every
    nonzero component must be a provenance-backed tensor derived from measured
    features; raw per-batch label lookup arrays are rejected.
    """

    global_effect: Union[ArrayLike, ProvenanceTensor] = 0.0
    lineage_effect: Union[ArrayLike, ProvenanceTensor] = 0.0
    tissue_effect: Union[ArrayLike, ProvenanceTensor] = 0.0
    state_effect: Union[ArrayLike, ProvenanceTensor] = 0.0
    subject_effect: Union[ArrayLike, ProvenanceTensor] = 0.0

    def total(
        self,
        edge_shape: Tuple[int, int],
        batch_size: int,
        seed: int = 0,
        prediction_origin_time: float = 0.0,
    ) -> Array:
        target = (int(batch_size),) + edge_shape
        total = np.zeros(target, dtype=np.float64)
        for offset, name in enumerate((
            "global_effect", "lineage_effect", "tissue_effect", "state_effect",
            "subject_effect",
        )):
            raw = getattr(self, name)
            if isinstance(raw, ProvenanceTensor):
                raw.validate_for_prediction_origin(
                    prediction_origin_time, f"hierarchical {name}"
                )
                audit_encoder_features(raw.source_feature_names)
                sampled = raw.effective_values(int(seed) + offset)
                if bool(np.ma.getmaskarray(sampled).any()):
                    raise ValueError(
                        f"hierarchical {name} cannot contain unknown effects"
                    )
                value = np.asarray(sampled.data, dtype=np.float64)
            else:
                value = _finite_array(raw, f"hierarchical.{name}")
                if bool((value != 0.0).any()):
                    raise ValueError(
                        f"nonzero hierarchical {name} must be a ProvenanceTensor "
                        "derived from leakage-audited measured features"
                    )
            if value.shape == edge_shape:
                value = value[None, :, :]
            total += _broadcast(value, target, name)
        return total


@dataclass(frozen=True)
class ContextualRouteParameters:
    """Context-variable strength, sign and instantaneous kinetic gain.

    ``kinetics`` changes the current rate.  It is not a propagation delay.
    """

    strength: HierarchicalField = field(default_factory=HierarchicalField)
    sign: HierarchicalField = field(default_factory=HierarchicalField)
    kinetics: HierarchicalField = field(default_factory=HierarchicalField)

    def effective(
        self,
        mask: Array,
        batch_size: int,
        seed: int = 0,
        prediction_origin_time: float = 0.0,
    ) -> Array:
        shape = tuple(int(item) for item in mask.shape)
        strength = np.exp(
            np.clip(
                self.strength.total(
                    shape, batch_size, seed, prediction_origin_time
                ),
                -12.0,
                12.0,
            )
        )
        sign = np.tanh(
            self.sign.total(
                shape, batch_size, seed + 101, prediction_origin_time
            )
        )
        kinetics = np.exp(
            np.clip(
                self.kinetics.total(
                    shape, batch_size, seed + 202, prediction_origin_time
                ),
                -12.0,
                12.0,
            )
        )
        return np.asarray(mask[None, :, :] * strength * sign * kinetics, dtype=np.float64)


@dataclass(frozen=True)
class GraphMaskedResidual:
    """Bounded residual weights that can only occupy candidate graph edges."""

    cue_signal: Optional[Array] = None
    signal_tf: Optional[Array] = None
    tf_enhancer: Optional[Array] = None
    enhancer_gene: Optional[Array] = None
    signal_signal: Optional[Array] = None
    tf_tf: Optional[Array] = None
    metabolism_signal: Optional[Array] = None
    metabolism_tf: Optional[Array] = None
    neighbor_signal: Optional[Array] = None
    bound: float = 0.1
    source_feature_names: Tuple[str, ...] = ()
    source_lineage: Tuple[str, ...] = ()
    method_lineage: Optional[str] = None
    source_role: Optional[str] = None
    source_partition: Optional[str] = None
    source_accessions: Tuple[str, ...] = ()

    def validate(self, graph: RegulatoryGraph) -> None:
        if not math.isfinite(float(self.bound)) or self.bound < 0.0:
            raise ValueError("residual bound must be finite and non-negative")
        masks = graph.route_masks
        nonzero_residual = False
        for name, mask in masks.items():
            value = getattr(self, name)
            if value is None:
                continue
            array = _finite_array(value, f"residual.{name}")
            if array.shape != mask.shape:
                raise ValueError(f"residual.{name} must have shape {mask.shape}")
            if bool((np.abs(array[~mask]) > 0.0).any()):
                raise ValueError(f"residual.{name} has weights outside the candidate graph")
            nonzero_residual = nonzero_residual or bool((array != 0.0).any())
        absent = set(item.name for item in fields(self)) - set(masks) - {
            "bound", "source_feature_names", "source_lineage", "method_lineage",
            "source_role", "source_partition", "source_accessions",
        }
        for name in absent:
            if getattr(self, name) is not None:
                raise ValueError(f"residual.{name} was supplied without a graph mask")
        if nonzero_residual and self.bound > 0.0:
            if (
                isinstance(self.source_feature_names, str)
                or isinstance(self.source_lineage, str)
                or not self.source_feature_names
                or not self.source_lineage
                or not str(self.method_lineage or "").strip()
            ):
                raise ValueError(
                    "nonzero graph residuals require source features, source "
                    "lineage and method lineage"
                )
            _validate_source_access(
                source_role=self.source_role,
                source_partition=self.source_partition,
                source_accessions=self.source_accessions,
                source_lineage=self.source_lineage,
                source_feature_names=self.source_feature_names,
                method_lineage=self.method_lineage,
                label="graph residual",
            )
            audit_encoder_features(self.source_feature_names)

    def effective(self, name: str, mask: Array, batch_size: int) -> Array:
        value = getattr(self, name)
        if value is None or self.bound == 0.0:
            return np.zeros((batch_size,) + mask.shape, dtype=np.float64)
        bounded = float(self.bound) * np.tanh(np.asarray(value, dtype=np.float64))
        return np.broadcast_to(mask[None, :, :] * bounded[None, :, :], (batch_size,) + mask.shape).copy()


@dataclass(frozen=True)
class NodeKineticModulation:
    """Provenance-backed context multipliers for named node kinetics.

    A missing field means multiplier one.  Any supplied field must be a
    ``ProvenanceTensor`` derived from leakage-audited measured features; raw
    label-indexed per-batch arrays are intentionally not accepted.
    """

    chromatin: Optional[ProvenanceTensor] = None
    signal: Optional[ProvenanceTensor] = None
    tf: Optional[ProvenanceTensor] = None
    rna: Optional[ProvenanceTensor] = None
    metabolism: Optional[ProvenanceTensor] = None

    def realize(
        self,
        graph: RegulatoryGraph,
        batch_size: int,
        seed: int,
        prediction_origin_time: float,
    ) -> Dict[str, Array]:
        d = graph.dimensions
        dimensions = {
            "chromatin": d["enhancers"],
            "signal": d["signals"],
            "tf": d["tfs"],
            "rna": d["genes"],
        }
        if d["metabolites"]:
            dimensions["metabolism"] = d["metabolites"]
        answer: Dict[str, Array] = {}
        for offset, (name, width) in enumerate(dimensions.items()):
            value = getattr(self, name)
            if value is None:
                answer[name] = np.ones((batch_size, width), dtype=np.float64)
                continue
            if not isinstance(value, ProvenanceTensor):
                raise ValueError(f"node kinetic {name} must be a ProvenanceTensor")
            value.validate_for_prediction_origin(
                prediction_origin_time, f"node kinetic {name}"
            )
            audit_encoder_features(value.source_feature_names)
            sampled = value.effective_values(int(seed) + offset)
            if bool(np.ma.getmaskarray(sampled).any()):
                raise ValueError(f"required node kinetic {name} contains unknown values")
            physical = _broadcast(
                np.asarray(sampled.data, dtype=np.float64),
                (batch_size, width),
                f"node_kinetics.{name}",
            )
            if bool((physical < 0.0).any()):
                raise ValueError(f"node kinetic {name} must be non-negative")
            answer[name] = physical
        if not d["metabolites"] and self.metabolism is not None:
            raise ValueError("metabolic node kinetics require named metabolic nodes")
        return answer


@dataclass(frozen=True)
class TwinContext:
    """Cell/context-variable measurements and route parameters."""

    cues: ProvenanceTensor
    accessibility: ProvenanceTensor
    tf_availability: ProvenanceTensor
    contact: ProvenanceTensor
    routes: Mapping[str, ContextualRouteParameters]
    declarations: Tuple[CausalFeatureDeclaration, ...]
    encoder_feature_names: Tuple[str, ...] = ()
    spatial_coordinates: Optional[ProvenanceTensor] = None
    spatial_coordinate_names: Tuple[str, ...] = ()
    spatial_adjacency: Optional[ProvenanceTensor] = None
    metabolism_input: Optional[ProvenanceTensor] = None
    node_kinetics: NodeKineticModulation = field(
        default_factory=NodeKineticModulation
    )
    prediction_origin_time: float = 0.0

    @staticmethod
    def _effective(value: ProvenanceTensor, seed: int) -> np.ma.MaskedArray:
        if not isinstance(value, ProvenanceTensor):
            raise ValueError("assimilated context features require ProvenanceTensor")
        value.validate()
        return value.effective_values(seed)

    @staticmethod
    def _resolved(
        value: ProvenanceTensor, seed: int
    ) -> Tuple[Array, Array, Array]:
        """Return effective values, explicit known mask and evidence weights."""

        if not isinstance(value, ProvenanceTensor):
            raise ValueError("assimilated context features require ProvenanceTensor")
        value.validate()
        masked = value.effective_values(seed)
        return (
            np.asarray(masked.data, dtype=np.float64),
            value.known_mask(),
            value.evidence_weights(),
        )

    def realize(self, graph: RegulatoryGraph, batch_size: int, seed: int) -> "RealizedTwinContext":
        graph.validate()
        declarations = {item.name: item for item in self.declarations}
        for name in graph.cue_names:
            item = declarations.get(name)
            if item is None or item.kind != "cue":
                raise ValueError(f"cue {name!r} requires an explicit cue declaration")
        for name in self.spatial_coordinate_names:
            item = declarations.get(name)
            if item is None or item.kind != "spatial_coordinate":
                raise ValueError(f"spatial coordinate {name!r} requires an explicit declaration")
        audit_encoder_features(
            self.encoder_feature_names + graph.cue_names + self.spatial_coordinate_names,
            self.declarations,
            prediction_origin_time=self.prediction_origin_time,
        )
        provenance_inputs = [
            self.cues,
            self.accessibility,
            self.tf_availability,
            self.contact,
        ]
        if self.spatial_coordinates is not None:
            provenance_inputs.append(self.spatial_coordinates)
        if self.metabolism_input is not None:
            provenance_inputs.append(self.metabolism_input)
        for value in provenance_inputs:
            if not isinstance(value, ProvenanceTensor):
                raise ValueError(
                    "every assimilated context input must be a ProvenanceTensor"
                )
            value.validate_for_prediction_origin(
                self.prediction_origin_time, "context input"
            )
            audit_encoder_features(
                value.source_feature_names,
                self.declarations,
                prediction_origin_time=self.prediction_origin_time,
            )
        d = graph.dimensions
        known_masks: Dict[str, Array] = {}
        evidence_weights: Dict[str, Array] = {}

        def resolve(
            name: str,
            value: ProvenanceTensor,
            feature_seed: int,
            shape: Tuple[int, ...],
        ) -> Array:
            effective, known, weights = self._resolved(value, feature_seed)
            known_masks[name] = _broadcast(known, shape, f"{name}_known_mask").astype(bool)
            evidence_weights[name] = _broadcast(
                weights, shape, f"{name}_evidence_weights"
            )
            if not bool(known_masks[name].all()):
                raise ValueError(
                    f"required ODE gate {name} contains unknown values; "
                    "missingness cannot be encoded as zero"
                )
            return _broadcast(effective, shape, name)

        cues = resolve("cues", self.cues, seed + 1, (batch_size, d["cues"]))
        accessibility = resolve(
            "accessibility", self.accessibility, seed + 2,
            (batch_size, d["enhancers"]),
        )
        tf_availability = resolve(
            "tf_availability", self.tf_availability, seed + 3,
            (batch_size, d["tfs"]),
        )
        contact = resolve(
            "contact", self.contact, seed + 4,
            (batch_size, d["enhancers"], d["genes"]),
        )
        if bool((accessibility < 0.0).any()) or bool((accessibility > 1.0).any()):
            raise ValueError("accessibility must lie in [0, 1]")
        if bool((tf_availability < 0.0).any()):
            raise ValueError("TF availability must be non-negative")
        if bool((contact < 0.0).any()):
            raise ValueError("contact evidence must be non-negative")
        missing_routes = sorted(set(graph.route_masks) - set(self.routes))
        unknown_routes = sorted(set(self.routes) - set(graph.route_masks))
        if missing_routes or unknown_routes:
            raise ValueError(
                f"context route mismatch; missing={missing_routes}, unknown={unknown_routes}"
            )
        # Validate temporal provenance for every contextual edge component at
        # assimilation time, not only when the derivative first uses a route.
        for offset, (route_name, route_mask) in enumerate(
            graph.route_masks.items()
        ):
            self.routes[route_name].effective(
                route_mask,
                batch_size,
                seed + 1000 + offset * 100,
                self.prediction_origin_time,
            )
        spatial_coordinates: Optional[Array] = None
        adjacency: Optional[Array] = None
        if self.spatial_coordinates is not None:
            raw_spatial = self._effective(self.spatial_coordinates, seed + 5)
            spatial_coordinates = resolve(
                "spatial_coordinates", self.spatial_coordinates, seed + 5,
                tuple(raw_spatial.shape),
            )
            if spatial_coordinates.ndim != 2 or spatial_coordinates.shape[0] != batch_size:
                raise ValueError("spatial_coordinates must have shape [batch, dimensions]")
            if len(self.spatial_coordinate_names) != spatial_coordinates.shape[1]:
                raise ValueError("spatial_coordinate_names disagree with spatial coordinates")
        elif self.spatial_coordinate_names:
            raise ValueError("spatial coordinate names were supplied without values")
        if self.spatial_adjacency is not None:
            if "neighbor_signal" not in graph.route_masks:
                raise ValueError(
                    "spatial adjacency requires a named neighbor_signal candidate mask"
                )
            if not isinstance(self.spatial_adjacency, ProvenanceTensor):
                raise ValueError("spatial_adjacency must be a ProvenanceTensor")
            self.spatial_adjacency.validate_for_prediction_origin(
                self.prediction_origin_time, "spatial adjacency"
            )
            if not str(self.spatial_adjacency.method_lineage or "").strip():
                raise ValueError(
                    "spatial_adjacency requires explicit graph-construction lineage"
                )
            audit_encoder_features(
                self.spatial_adjacency.source_feature_names,
                self.declarations,
                prediction_origin_time=self.prediction_origin_time,
            )
            masked_adjacency = self.spatial_adjacency.effective_values(seed + 50)
            adjacency_known = self.spatial_adjacency.known_mask()
            if bool((~adjacency_known).any()):
                raise ValueError("spatial adjacency cannot contain unknown edges")
            adjacency = _finite_array(
                masked_adjacency.data, "spatial_adjacency"
            )
            if adjacency.shape != (batch_size, batch_size):
                raise ValueError("spatial_adjacency must have shape [batch, batch]")
            if bool((adjacency < 0.0).any()):
                raise ValueError("spatial_adjacency must be non-negative")
            rows = adjacency.sum(axis=1, keepdims=True)
            isolated = rows[:, 0] == 0.0
            if bool(isolated.any()):
                indices = np.flatnonzero(isolated)
                adjacency[indices, indices] = 1.0
                rows = adjacency.sum(axis=1, keepdims=True)
            adjacency = np.divide(adjacency, rows, out=np.zeros_like(adjacency), where=rows > 0)
            known_masks["spatial_adjacency"] = adjacency_known.astype(bool)
            evidence_weights["spatial_adjacency"] = (
                self.spatial_adjacency.evidence_weights()
            )
        metabolism_input: Optional[Array] = None
        if d["metabolites"]:
            if self.metabolism_input is not None:
                metabolism_input = resolve(
                    "metabolism_input", self.metabolism_input, seed + 6,
                    (batch_size, d["metabolites"]),
                )
        elif self.metabolism_input is not None:
            raise ValueError("metabolism_input requires named metabolic nodes")
        node_kinetic_multipliers = self.node_kinetics.realize(
            graph,
            batch_size,
            seed + 100,
            self.prediction_origin_time,
        )
        return RealizedTwinContext(
            cues=cues,
            accessibility=accessibility,
            tf_availability=tf_availability,
            contact=contact,
            routes=dict(self.routes),
            spatial_coordinates=spatial_coordinates,
            spatial_adjacency=adjacency,
            metabolism_input=metabolism_input,
            known_masks=known_masks,
            evidence_weights=evidence_weights,
            node_kinetic_multipliers=node_kinetic_multipliers,
            realization_seed=int(seed),
            prediction_origin_time=float(self.prediction_origin_time),
        )

    def provenance_diagnostics(self) -> Dict[str, object]:
        answer: Dict[str, object] = {}
        for name in (
            "cues", "accessibility", "tf_availability", "contact",
            "spatial_coordinates", "spatial_adjacency", "metabolism_input",
        ):
            value = getattr(self, name)
            answer[name] = value.diagnostics() if isinstance(value, ProvenanceTensor) else {
                "provenance": "observed" if value is not None else "not_supplied"
            }
        answer["node_kinetics"] = {
            name: (
                value.diagnostics()
                if isinstance(value, ProvenanceTensor)
                else {"provenance": "default_unit_multiplier"}
            )
            for name, value in {
                "chromatin": self.node_kinetics.chromatin,
                "signal": self.node_kinetics.signal,
                "tf": self.node_kinetics.tf,
                "rna": self.node_kinetics.rna,
                "metabolism": self.node_kinetics.metabolism,
            }.items()
        }
        return answer


@dataclass(frozen=True)
class RealizedTwinContext:
    cues: Array
    accessibility: Array
    tf_availability: Array
    contact: Array
    routes: Mapping[str, ContextualRouteParameters]
    spatial_coordinates: Optional[Array]
    spatial_adjacency: Optional[Array]
    metabolism_input: Optional[Array]
    known_masks: Mapping[str, Array]
    evidence_weights: Mapping[str, Array]
    node_kinetic_multipliers: Mapping[str, Array]
    realization_seed: int
    prediction_origin_time: float


@dataclass(frozen=True)
class TwinState:
    """Named dynamic state; no opaque latent is decoded directly to RNA.

    Raw array construction is reserved for internal RK stages and derivative
    values.  Any state entering a public simulation boundary must be created
    with :meth:`from_provenance` (or carry the equivalent ``input_provenance``
    mapping) so a future RNA/ATAC measurement cannot be relabeled as time zero.
    """

    chromatin: Array
    signal: Array
    tf: Array
    rna: Array
    metabolism: Optional[Array] = None
    input_provenance: Optional[Mapping[str, ProvenanceTensor]] = None

    @classmethod
    def from_provenance(
        cls,
        *,
        chromatin: ProvenanceTensor,
        signal: ProvenanceTensor,
        tf: ProvenanceTensor,
        rna: ProvenanceTensor,
        metabolism: Optional[ProvenanceTensor] = None,
    ) -> "TwinState":
        """Construct an external initial state from explicit evidence tensors."""

        tensors: Dict[str, ProvenanceTensor] = {
            "chromatin": chromatin,
            "signal": signal,
            "tf": tf,
            "rna": rna,
        }
        if metabolism is not None:
            tensors["metabolism"] = metabolism
        arrays: Dict[str, Array] = {}
        for name, tensor in tensors.items():
            if not isinstance(tensor, ProvenanceTensor):
                raise TypeError(f"initial state {name} must be a ProvenanceTensor")
            tensor.validate()
            physical = tensor.physical_values()
            if bool(np.ma.getmaskarray(physical).any()):
                raise ValueError(
                    f"initial state {name} contains unknown values; missing state "
                    "cannot be encoded as zero"
                )
            arrays[name] = np.asarray(physical.data, dtype=np.float64).copy()
        return cls(
            chromatin=arrays["chromatin"],
            signal=arrays["signal"],
            tf=arrays["tf"],
            rna=arrays["rna"],
            metabolism=arrays.get("metabolism"),
            input_provenance=dict(tensors),
        )

    @property
    def batch_size(self) -> int:
        array = np.asarray(self.rna)
        if array.ndim != 2:
            raise ValueError("RNA state must be rank two")
        return int(array.shape[0])

    def validate(self, graph: RegulatoryGraph) -> None:
        d = graph.dimensions
        batch = self.batch_size
        expected = {
            "chromatin": (batch, d["enhancers"]),
            "signal": (batch, d["signals"]),
            "tf": (batch, d["tfs"]),
            "rna": (batch, d["genes"]),
        }
        for name, shape in expected.items():
            array = _finite_array(getattr(self, name), f"state.{name}")
            if array.shape != shape:
                raise ValueError(f"state.{name} has shape {array.shape}; expected {shape}")
        if bool((np.asarray(self.chromatin) < 0.0).any()) or bool((np.asarray(self.chromatin) > 1.0).any()):
            raise ValueError("chromatin state must lie in [0, 1]")
        if d["metabolites"]:
            if self.metabolism is None or np.asarray(self.metabolism).shape != (batch, d["metabolites"]):
                raise ValueError("metabolism state has incompatible shape")
            _finite_array(self.metabolism, "state.metabolism")
        elif self.metabolism is not None:
            raise ValueError("metabolism state supplied without metabolic nodes")

    def validate_as_initial(
        self, graph: RegulatoryGraph, prediction_origin_time: float
    ) -> None:
        """Validate an externally supplied state at a causal prediction origin."""

        self.validate(graph)
        if not isinstance(self.input_provenance, Mapping):
            raise ValueError(
                "public initial state requires per-variable input_provenance; "
                "use TwinState.from_provenance"
            )
        expected = {"chromatin", "signal", "tf", "rna"}
        if graph.dimensions["metabolites"]:
            expected.add("metabolism")
        observed = set(self.input_provenance)
        if observed != expected:
            raise ValueError(
                "initial-state provenance keys disagree with the named state; "
                f"expected={sorted(expected)}, observed={sorted(observed)}"
            )
        for name in sorted(expected):
            tensor = self.input_provenance[name]
            if not isinstance(tensor, ProvenanceTensor):
                raise TypeError(
                    f"initial-state provenance for {name} must be a ProvenanceTensor"
                )
            tensor.validate_for_prediction_origin(
                prediction_origin_time, f"initial state {name}"
            )
            audit_encoder_features(
                tensor.source_feature_names,
                prediction_origin_time=prediction_origin_time,
            )
            physical = tensor.physical_values()
            if bool(np.ma.getmaskarray(physical).any()):
                raise ValueError(f"initial state {name} contains unknown values")
            state_value = np.asarray(getattr(self, name), dtype=np.float64)
            evidence_value = np.asarray(physical.data, dtype=np.float64)
            if state_value.shape != evidence_value.shape or not np.array_equal(
                state_value, evidence_value
            ):
                raise ValueError(
                    f"initial state {name} differs from its provenance tensor"
                )

    def provenance_diagnostics(self) -> Dict[str, object]:
        if not isinstance(self.input_provenance, Mapping):
            return {"external_initial_state_provenance": "not_supplied"}
        return {
            name: tensor.diagnostics()
            for name, tensor in sorted(self.input_provenance.items())
        }

    def copy(self) -> "TwinState":
        return TwinState(
            chromatin=np.array(self.chromatin, dtype=np.float64, copy=True),
            signal=np.array(self.signal, dtype=np.float64, copy=True),
            tf=np.array(self.tf, dtype=np.float64, copy=True),
            rna=np.array(self.rna, dtype=np.float64, copy=True),
            metabolism=None if self.metabolism is None else np.array(self.metabolism, dtype=np.float64, copy=True),
            input_provenance=(
                None
                if self.input_provenance is None
                else dict(self.input_provenance)
            ),
        )


@dataclass(frozen=True)
class TwinKinetics:
    chromatin_open: ArrayLike = 0.15
    chromatin_close: ArrayLike = 0.08
    signal_decay: ArrayLike = 0.20
    tf_decay: ArrayLike = 0.12
    rna_decay: ArrayLike = 0.08
    metabolism_decay: ArrayLike = 0.05
    signal_production: ArrayLike = 1.0
    tf_production: ArrayLike = 1.0
    rna_production: ArrayLike = 1.0

    def validate(self, graph: RegulatoryGraph) -> None:
        d = graph.dimensions
        checks = {
            "chromatin_open": (d["enhancers"],),
            "chromatin_close": (d["enhancers"],),
            "signal_decay": (d["signals"],),
            "tf_decay": (d["tfs"],),
            "rna_decay": (d["genes"],),
            "signal_production": (d["signals"],),
            "tf_production": (d["tfs"],),
            "rna_production": (d["genes"],),
        }
        if d["metabolites"]:
            checks["metabolism_decay"] = (d["metabolites"],)
        for name, shape in checks.items():
            value = _broadcast(getattr(self, name), shape, f"kinetics.{name}")
            if bool((value < 0.0).any()):
                raise ValueError(f"kinetics.{name} must be non-negative")


@dataclass(frozen=True)
class TwinConfig:
    seed: int = 42
    persistence_when_all_routes_removed: bool = True
    clip_chromatin: bool = True
    chromatin_noise: float = 0.0
    signal_noise: float = 0.0
    tf_noise: float = 0.0
    rna_noise: float = 0.0
    metabolism_noise: float = 0.0

    def validate(self) -> None:
        if not self.persistence_when_all_routes_removed:
            raise ValueError(
                "all-route removal must remain exact persistence; this invariant "
                "cannot be disabled"
            )
        for name in ("chromatin_noise", "signal_noise", "tf_noise", "rna_noise", "metabolism_noise"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RouteIntervention:
    """Frozen inference-time route ablation/scaling; never retrains weights."""

    remove_routes: Tuple[str, ...] = ()
    route_scales: Mapping[str, float] = field(default_factory=dict)
    remove_all: bool = False

    @classmethod
    def frozen_all(cls) -> "RouteIntervention":
        return cls(remove_all=True)

    def validate(self, graph: RegulatoryGraph) -> None:
        known = set(graph.route_masks)
        unknown = (set(self.remove_routes) | set(self.route_scales)) - known
        if unknown:
            raise ValueError(f"unknown route interventions: {sorted(unknown)}")
        for name, scale in self.route_scales.items():
            if not math.isfinite(float(scale)) or float(scale) < 0.0:
                raise ValueError(f"route scale for {name} must be finite and non-negative")

    def scale(self, name: str) -> float:
        if self.remove_all or name in self.remove_routes:
            return 0.0
        return float(self.route_scales.get(name, 1.0))

    def removes_every_route(self, graph: RegulatoryGraph) -> bool:
        return self.remove_all or all(self.scale(name) == 0.0 for name in graph.route_masks)


def _zeros_like_state(state: TwinState) -> TwinState:
    return TwinState(
        chromatin=np.zeros_like(state.chromatin, dtype=np.float64),
        signal=np.zeros_like(state.signal, dtype=np.float64),
        tf=np.zeros_like(state.tf, dtype=np.float64),
        rna=np.zeros_like(state.rna, dtype=np.float64),
        metabolism=None if state.metabolism is None else np.zeros_like(state.metabolism, dtype=np.float64),
    )


def _state_linear(a: TwinState, b: TwinState, scale: float) -> TwinState:
    return TwinState(
        chromatin=np.asarray(a.chromatin) + scale * np.asarray(b.chromatin),
        signal=np.asarray(a.signal) + scale * np.asarray(b.signal),
        tf=np.asarray(a.tf) + scale * np.asarray(b.tf),
        rna=np.asarray(a.rna) + scale * np.asarray(b.rna),
        metabolism=None if a.metabolism is None else np.asarray(a.metabolism) + scale * np.asarray(b.metabolism),
    )


@dataclass(frozen=True)
class TwinTrajectory:
    times: Array
    states: Tuple[TwinState, ...]
    diagnostics: Mapping[str, object]

    @property
    def final_state(self) -> TwinState:
        return self.states[-1]

    def to_jsonable_diagnostics(self) -> Dict[str, object]:
        return _jsonable(dict(self.diagnostics))  # type: ignore[return-value]


class RegulatoryTwin:
    """Mechanistically routed virtual-tissue vector field and RK4 simulator."""

    def __init__(
        self,
        graph: RegulatoryGraph,
        kinetics: TwinKinetics = TwinKinetics(),
        config: TwinConfig = TwinConfig(),
        residual: Optional[GraphMaskedResidual] = None,
    ) -> None:
        graph.validate()
        kinetics.validate(graph)
        config.validate()
        if residual is not None:
            residual.validate(graph)
        self.graph = graph
        self.kinetics = kinetics
        self.config = config
        self.residual = residual or GraphMaskedResidual(bound=0.0)

    def _validate_realized(
        self, context: RealizedTwinContext, batch_size: int
    ) -> None:
        """Validate even pre-realized contexts before evaluating the field."""

        d = self.graph.dimensions
        expected = {
            "cues": (batch_size, d["cues"]),
            "accessibility": (batch_size, d["enhancers"]),
            "tf_availability": (batch_size, d["tfs"]),
            "contact": (batch_size, d["enhancers"], d["genes"]),
        }
        for name, shape in expected.items():
            value = getattr(context, name)
            if value is None:
                raise ValueError(f"realized context is missing {name}")
            array = _finite_array(value, f"realized.{name}")
            if array.shape != shape:
                raise ValueError(
                    f"realized.{name} has shape {array.shape}; expected {shape}"
                )
            if name not in context.known_masks or name not in context.evidence_weights:
                raise ValueError(f"realized context lost provenance mask for {name}")
            known = np.asarray(context.known_masks[name])
            evidence = _finite_array(
                context.evidence_weights[name], f"realized.{name}_evidence_weights"
            )
            if known.shape != shape or evidence.shape != shape:
                raise ValueError(f"realized provenance arrays disagree with {name}")
            if not np.isin(known, [0, 1, False, True]).all():
                raise ValueError(f"realized known mask for {name} is not binary")
            if bool((evidence < 0.0).any()) or bool((evidence > 1.0).any()):
                raise ValueError(f"realized evidence weights for {name} must lie in [0, 1]")
            if bool((evidence[~known.astype(bool)] != 0.0).any()):
                raise ValueError(f"unknown realized {name} entries must retain zero evidence weight")
        if bool((context.accessibility < 0.0).any()) or bool(
            (context.accessibility > 1.0).any()
        ):
            raise ValueError("realized accessibility must lie in [0, 1]")
        if bool((context.tf_availability < 0.0).any()) or bool(
            (context.contact < 0.0).any()
        ):
            raise ValueError("realized availability/contact must be non-negative")
        if context.metabolism_input is not None:
            metabolic_shape = (batch_size, d["metabolites"])
            metabolic_input = _finite_array(
                context.metabolism_input, "realized.metabolism_input"
            )
            if metabolic_input.shape != metabolic_shape:
                raise ValueError("realized metabolism input has incompatible shape")
            if (
                "metabolism_input" not in context.known_masks
                or "metabolism_input" not in context.evidence_weights
            ):
                raise ValueError("realized metabolism input lost provenance masks")
            metabolic_known = np.asarray(context.known_masks["metabolism_input"])
            metabolic_evidence = _finite_array(
                context.evidence_weights["metabolism_input"],
                "realized.metabolism_input_evidence_weights",
            )
            if (
                metabolic_known.shape != metabolic_shape
                or metabolic_evidence.shape != metabolic_shape
                or not bool(metabolic_known.all())
            ):
                raise ValueError("realized metabolism input has invalid provenance")
        if set(context.routes) != set(self.graph.route_masks):
            raise ValueError("realized context route names disagree with graph")
        kinetic_shapes = {
            "chromatin": (batch_size, d["enhancers"]),
            "signal": (batch_size, d["signals"]),
            "tf": (batch_size, d["tfs"]),
            "rna": (batch_size, d["genes"]),
        }
        if d["metabolites"]:
            kinetic_shapes["metabolism"] = (batch_size, d["metabolites"])
        if set(context.node_kinetic_multipliers) != set(kinetic_shapes):
            raise ValueError("realized node kinetic names disagree with graph")
        for name, shape in kinetic_shapes.items():
            multiplier = _finite_array(
                context.node_kinetic_multipliers[name],
                f"realized.node_kinetics.{name}",
            )
            if multiplier.shape != shape or bool((multiplier < 0.0).any()):
                raise ValueError(f"realized node kinetic {name} is invalid")
        if context.spatial_coordinates is not None:
            coordinate = _finite_array(
                context.spatial_coordinates, "realized.spatial_coordinates"
            )
            if coordinate.ndim != 2 or coordinate.shape[0] != batch_size:
                raise ValueError("realized spatial coordinates have incompatible shape")
            if "spatial_coordinates" not in context.known_masks or (
                "spatial_coordinates" not in context.evidence_weights
            ):
                raise ValueError("realized spatial coordinates lost provenance masks")
            spatial_known = np.asarray(context.known_masks["spatial_coordinates"])
            spatial_evidence = _finite_array(
                context.evidence_weights["spatial_coordinates"],
                "realized.spatial_coordinate_evidence_weights",
            )
            if spatial_known.shape != coordinate.shape or spatial_evidence.shape != coordinate.shape:
                raise ValueError("spatial provenance arrays disagree with coordinates")
            if bool((spatial_evidence[~spatial_known.astype(bool)] != 0.0).any()):
                raise ValueError("unknown spatial coordinates must retain zero evidence weight")
        if context.spatial_adjacency is not None:
            adjacency = _finite_array(
                context.spatial_adjacency, "realized.spatial_adjacency"
            )
            if adjacency.shape != (batch_size, batch_size) or bool(
                (adjacency < 0.0).any()
            ):
                raise ValueError("realized spatial adjacency is invalid")
            if not np.allclose(adjacency.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
                raise ValueError("realized spatial adjacency must be row stochastic")

    def _weights(
        self,
        context: RealizedTwinContext,
        intervention: RouteIntervention,
        batch: int,
    ) -> Dict[str, Array]:
        answer: Dict[str, Array] = {}
        for name, mask in self.graph.route_masks.items():
            base = context.routes[name].effective(
                mask,
                batch,
                context.realization_seed + len(answer) * 1000,
                context.prediction_origin_time,
            )
            residual = self.residual.effective(name, mask, batch)
            answer[name] = float(intervention.scale(name)) * (base + residual)
        return answer

    def derivative(
        self,
        state: TwinState,
        context: TwinContext,
        intervention: Optional[RouteIntervention] = None,
        *,
        seed: Optional[int] = None,
    ) -> TwinState:
        """Evaluate the named-state ODE drift.

        Optional process noise is applied only by :meth:`rollout`; this method
        always returns the deterministic drift used by RK4.
        """

        if not isinstance(context, TwinContext):
            raise TypeError(
                "public derivative requires TwinContext; pre-realized contexts "
                "cannot bypass causal declarations or provenance validation"
            )
        state.validate_as_initial(self.graph, context.prediction_origin_time)
        intervention = intervention or RouteIntervention()
        intervention.validate(self.graph)
        batch = state.batch_size
        realized = context.realize(
            self.graph,
            batch,
            self.config.seed if seed is None else int(seed),
        )
        return self._derivative_realized(state, realized, intervention)

    def _derivative_realized(
        self,
        state: TwinState,
        realized: RealizedTwinContext,
        intervention: RouteIntervention,
    ) -> TwinState:
        """Private RK4 drift for a context validated by ``TwinContext``."""

        state.validate(self.graph)
        batch = state.batch_size
        self._validate_realized(realized, batch)
        if intervention.removes_every_route(self.graph):
            return _zeros_like_state(state)
        weights = self._weights(realized, intervention, batch)
        d = self.graph.dimensions

        cue_signal = np.einsum("bc,bcs->bs", realized.cues, weights["cue_signal"])
        signal_drive = cue_signal
        if "signal_signal" in weights:
            signal_drive = signal_drive + np.einsum("bs,bsk->bk", np.tanh(state.signal), weights["signal_signal"])
        if state.metabolism is not None and "metabolism_signal" in weights:
            signal_drive = signal_drive + np.einsum("bm,bms->bs", np.tanh(state.metabolism), weights["metabolism_signal"])
        if realized.spatial_adjacency is not None and "neighbor_signal" in weights:
            neighbor_state = realized.spatial_adjacency @ np.tanh(
                np.asarray(state.signal)
            )
            signal_drive = signal_drive + np.einsum(
                "bs,bsk->bk", neighbor_state, weights["neighbor_signal"]
            )

        signal_tf = np.einsum("bs,bst->bt", np.tanh(state.signal), weights["signal_tf"])
        tf_drive = signal_tf
        if "tf_tf" in weights:
            tf_drive = tf_drive + np.einsum("bt,btu->bu", np.tanh(state.tf), weights["tf_tf"])
        if state.metabolism is not None and "metabolism_tf" in weights:
            tf_drive = tf_drive + np.einsum("bm,bmt->bt", np.tanh(state.metabolism), weights["metabolism_tf"])

        available_tf = np.tanh(np.asarray(state.tf)) * realized.tf_availability
        enhancer_regulation = np.einsum("bt,bte->be", available_tf, weights["tf_enhancer"])
        chromatin = np.asarray(state.chromatin, dtype=np.float64)
        open_rate = _broadcast(self.kinetics.chromatin_open, (d["enhancers"],), "chromatin_open")
        close_rate = _broadcast(self.kinetics.chromatin_close, (d["enhancers"],), "chromatin_close")
        chromatin_kinetic = realized.node_kinetic_multipliers["chromatin"]
        d_chromatin = (
            chromatin_kinetic * open_rate[None, :]
            * np.maximum(enhancer_regulation, 0.0) * (1.0 - chromatin)
            + chromatin_kinetic * close_rate[None, :]
            * np.minimum(enhancer_regulation, 0.0) * chromatin
        )
        active_enhancer = np.clip(chromatin * realized.accessibility, 0.0, 1.0) * np.tanh(enhancer_regulation)
        enhancer_gene = weights["enhancer_gene"] * realized.contact
        gene_drive = np.einsum("be,beg->bg", active_enhancer, enhancer_gene)

        signal_decay = _broadcast(self.kinetics.signal_decay, (d["signals"],), "signal_decay")
        tf_decay = _broadcast(self.kinetics.tf_decay, (d["tfs"],), "tf_decay")
        rna_decay = _broadcast(self.kinetics.rna_decay, (d["genes"],), "rna_decay")
        signal_prod = _broadcast(self.kinetics.signal_production, (d["signals"],), "signal_production")
        tf_prod = _broadcast(self.kinetics.tf_production, (d["tfs"],), "tf_production")
        rna_prod = _broadcast(self.kinetics.rna_production, (d["genes"],), "rna_production")
        signal_kinetic = realized.node_kinetic_multipliers["signal"]
        tf_kinetic = realized.node_kinetic_multipliers["tf"]
        rna_kinetic = realized.node_kinetic_multipliers["rna"]
        d_signal = signal_kinetic * (
            signal_prod[None, :] * np.tanh(signal_drive)
            - signal_decay[None, :] * np.asarray(state.signal)
        )
        d_tf = tf_kinetic * (
            tf_prod[None, :] * np.tanh(tf_drive)
            - tf_decay[None, :] * np.asarray(state.tf)
        )
        d_rna = rna_kinetic * (
            rna_prod[None, :] * np.tanh(gene_drive)
            - rna_decay[None, :] * np.asarray(state.rna)
        )

        d_metabolism: Optional[Array] = None
        if state.metabolism is not None:
            if realized.metabolism_input is None:
                d_metabolism = np.zeros_like(state.metabolism, dtype=np.float64)
            else:
                decay = _broadcast(
                    self.kinetics.metabolism_decay,
                    (d["metabolites"],),
                    "metabolism_decay",
                )
                d_metabolism = realized.node_kinetic_multipliers["metabolism"] * (
                    decay[None, :]
                    * (realized.metabolism_input - np.asarray(state.metabolism))
                )
        return TwinState(d_chromatin, d_signal, d_tf, d_rna, d_metabolism)

    def _clip_state(self, state: TwinState) -> TwinState:
        return TwinState(
            chromatin=np.clip(state.chromatin, 0.0, 1.0) if self.config.clip_chromatin else np.asarray(state.chromatin),
            signal=np.asarray(state.signal),
            tf=np.asarray(state.tf),
            rna=np.asarray(state.rna),
            metabolism=None if state.metabolism is None else np.asarray(state.metabolism),
        )

    def rollout(
        self,
        initial_state: TwinState,
        context: TwinContext,
        times: Sequence[float],
        intervention: Optional[RouteIntervention] = None,
        *,
        seed: Optional[int] = None,
    ) -> TwinTrajectory:
        """RK4 rollout with optional seeded SDE-style process perturbations."""

        if not isinstance(context, TwinContext):
            raise TypeError(
                "public rollout requires TwinContext; pre-realized contexts "
                "cannot bypass causal declarations or provenance validation"
            )
        initial_state.validate_as_initial(
            self.graph, context.prediction_origin_time
        )
        time_array = _finite_array(times, "times")
        if time_array.ndim != 1 or time_array.size < 2 or bool((np.diff(time_array) <= 0.0).any()):
            raise ValueError("times must be a strictly increasing one-dimensional sequence")
        intervention = intervention or RouteIntervention()
        intervention.validate(self.graph)
        run_seed = int(self.config.seed if seed is None else seed)
        realized = context.realize(self.graph, initial_state.batch_size, run_seed)
        rng = np.random.default_rng(run_seed)
        current = initial_state.copy()
        trajectory = [current.copy()]
        persistent = self.config.persistence_when_all_routes_removed and intervention.removes_every_route(self.graph)
        for left, right in zip(time_array[:-1], time_array[1:]):
            dt = float(right - left)
            if persistent:
                trajectory.append(current.copy())
                continue
            k1 = self._derivative_realized(current, realized, intervention)
            k2 = self._derivative_realized(
                _state_linear(current, k1, 0.5 * dt), realized, intervention
            )
            k3 = self._derivative_realized(
                _state_linear(current, k2, 0.5 * dt), realized, intervention
            )
            k4 = self._derivative_realized(
                _state_linear(current, k3, dt), realized, intervention
            )
            combined = TwinState(
                chromatin=(k1.chromatin + 2 * k2.chromatin + 2 * k3.chromatin + k4.chromatin) / 6.0,
                signal=(k1.signal + 2 * k2.signal + 2 * k3.signal + k4.signal) / 6.0,
                tf=(k1.tf + 2 * k2.tf + 2 * k3.tf + k4.tf) / 6.0,
                rna=(k1.rna + 2 * k2.rna + 2 * k3.rna + k4.rna) / 6.0,
                metabolism=None if k1.metabolism is None else (k1.metabolism + 2 * k2.metabolism + 2 * k3.metabolism + k4.metabolism) / 6.0,
            )
            current = _state_linear(current, combined, dt)
            root_dt = math.sqrt(dt)
            current = TwinState(
                chromatin=current.chromatin + root_dt * self.config.chromatin_noise * rng.normal(size=current.chromatin.shape),
                signal=current.signal + root_dt * self.config.signal_noise * rng.normal(size=current.signal.shape),
                tf=current.tf + root_dt * self.config.tf_noise * rng.normal(size=current.tf.shape),
                rna=current.rna + root_dt * self.config.rna_noise * rng.normal(size=current.rna.shape),
                metabolism=None if current.metabolism is None else current.metabolism + root_dt * self.config.metabolism_noise * rng.normal(size=current.metabolism.shape),
            )
            current = self._clip_state(current)
            finite_names = ["chromatin", "signal", "tf", "rna"]
            if current.metabolism is not None:
                finite_names.append("metabolism")
            if not all(
                np.isfinite(np.asarray(getattr(current, name))).all()
                for name in finite_names
            ):
                raise FloatingPointError("non-finite state encountered during rollout")
            trajectory.append(current.copy())
        diagnostics = self.diagnostics(initial_state, context, intervention)
        diagnostics.update({
            "integrator": "RK4 with optional seeded post-step SDE-style noise",
            "steps": int(time_array.size - 1),
            "seed": run_seed,
            "exact_persistence_rollout": bool(persistent),
        })
        return TwinTrajectory(time_array.copy(), tuple(trajectory), _jsonable(diagnostics))

    def diagnostics(
        self,
        state: TwinState,
        context: TwinContext,
        intervention: Optional[RouteIntervention] = None,
    ) -> Dict[str, object]:
        if not isinstance(context, TwinContext):
            raise TypeError(
                "public diagnostics requires TwinContext; pre-realized contexts "
                "cannot bypass causal declarations or provenance validation"
            )
        state.validate_as_initial(self.graph, context.prediction_origin_time)
        intervention = intervention or RouteIntervention()
        intervention.validate(self.graph)
        realized = context.realize(self.graph, state.batch_size, self.config.seed)
        self._validate_realized(realized, state.batch_size)
        edge_counts = {name: int(mask.sum()) for name, mask in self.graph.route_masks.items()}
        norms = {
            name: float(np.mean(np.abs(np.asarray(getattr(state, name)))))
            for name in ("chromatin", "signal", "tf", "rna")
        }
        if state.metabolism is not None:
            norms["metabolism"] = float(np.mean(np.abs(state.metabolism)))
        answer = {
            "scope": "atlas-conditioned mechanistic virtual-tissue scaffold",
            "dimensions": self.graph.dimensions,
            "candidate_edge_counts": edge_counts,
            "hard_fixed": ["node identities", "genomic coordinates", "candidate masks"],
            "context_variable": [
                "accessibility", "TF availability", "contact", "edge strength",
                "edge sign", "kinetic gain", "lineage/tissue/state/subject effects",
            ],
            "hierarchy": ["global", "lineage", "tissue", "state", "subject"],
            "state_mean_absolute_values": norms,
            "initial_state_provenance": state.provenance_diagnostics(),
            "provenance": context.provenance_diagnostics(),
            "realized_known_fraction": {
                name: float(np.mean(mask)) if np.asarray(mask).size else 0.0
                for name, mask in realized.known_masks.items()
            },
            "measured_zero_distinct_from_unknown": True,
            "removed_routes": sorted(self.graph.route_masks) if intervention.remove_all else sorted(intervention.remove_routes),
            "all_routes_removed": intervention.removes_every_route(self.graph),
            "all_routes_removed_returns_persistence": bool(self.config.persistence_when_all_routes_removed),
            "bounded_graph_masked_residual": True,
            "explicit_route_delay_model": False,
            "route_kinetics_are_instantaneous_rate_multipliers": True,
            "direct_cue_to_rna_path": False,
            "direct_residual_to_rna_path": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
        }
        return _jsonable(answer)  # type: ignore[return-value]


__all__ = [
    "ProvenanceState",
    "DEFAULT_PROVENANCE_WEIGHTS",
    "ProvenanceTensor",
    "CausalFeatureDeclaration",
    "audit_encoder_features",
    "GenomicCoordinate",
    "RegulatoryGraph",
    "HierarchicalField",
    "ContextualRouteParameters",
    "GraphMaskedResidual",
    "NodeKineticModulation",
    "TwinContext",
    "RealizedTwinContext",
    "TwinState",
    "TwinKinetics",
    "TwinConfig",
    "RouteIntervention",
    "TwinTrajectory",
    "RegulatoryTwin",
]
