"""Restart-safe real-measurement corpus builder for WLD v6.1.

The module has a deliberately narrow job: acquire allowlisted, unsealed public
axolotl development measurements, preserve raw sparse values and deposited
identifiers, freeze biological-group partitions, and fit a feature registry
from training groups only.  It does not construct a ``TwinContext`` or train a
model.  GSE315993 is rejected before URL resolution or response-body access.
"""

from __future__ import annotations

import csv
import base64
import binascii
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urljoin, urlparse

import numpy as np
from scipy import sparse
from scipy.io import mmread


SCHEMA_VERSION = "wld-v6.1-axolotl-corpus-v1"
REGISTRY_SCHEMA_VERSIONS = {"6.1.0", SCHEMA_VERSION}
SEALED_ACCESSION = "GSE315993"
DEFAULT_USER_AGENT = "WLD-v6.1-axolotl-corpus/1.0 (+public-development-data)"
FORBIDDEN_ENCODER_TOKENS = {
    "animal", "barcode", "batch", "biosample", "celltype", "cell_type",
    "cluster", "condition", "counts", "donor", "embedding", "expression",
    "guide", "identity", "integrated", "label", "lineage", "outcome",
    "pseudotime", "response", "rna", "sample", "specimen", "stage", "state",
    "study", "target", "timepoint", "tissue", "umap",
}
ALLOWED_PARTITIONS = {"train", "validation", "reference"}
ALLOWED_ADAPTERS = {
    "bulk_gene_table", "dense_sc_barcode_maps", "visium_bundle",
    "acquisition_plan_only",
}
VALUE_ARTIFACT_ROLES = {
    "matrix", "counts", "barcode_map", "barcodes", "features",
    "coordinates", "histology", "atac", "cutandtag", "rna",
}


class SealedSourceRefused(RuntimeError):
    """Raised before any sealed measurement request can access a body."""


class CorpusIntegrityError(RuntimeError):
    """Raised when immutable source or derived-artifact integrity fails."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _stable_gzip_lines(path: Path, rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, delimiter="\t", lineterminator="\n")
                writer.writerows(rows)
    os.replace(temporary, path)


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") \
        if path.suffix == ".gz" else path.open("rt", encoding="utf-8", errors="replace", newline="")


def _decode_repeated(value: str) -> str:
    result = unicodedata.normalize("NFKC", str(value))
    for _ in range(6):
        decoded = unquote(result)
        if decoded == result:
            break
        result = decoded
    return result


def _flatten_scalars(value: object) -> List[str]:
    if isinstance(value, Mapping):
        result: List[str] = []
        for key, item in value.items():
            result.extend(_flatten_scalars(key))
            result.extend(_flatten_scalars(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten_scalars(item))
        return result
    return [_decode_repeated(str(value))]


def _flatten_values(value: object) -> List[str]:
    """Flatten payload values without mapping keys interrupting split tokens."""

    if isinstance(value, Mapping):
        result: List[str] = []
        for item in value.values():
            result.extend(_flatten_values(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten_values(item))
        return result
    return [_decode_repeated(str(value))]


def _compact_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFKC", value).upper())


def _decoded_text_candidates(value: str) -> List[str]:
    candidates = [_decode_repeated(value)]
    compact = re.sub(r"\s+", "", value)
    if len(compact) >= 8 and len(compact) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", compact):
        try:
            decoded = base64.b64decode(compact, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
        else:
            candidates.append(_decode_repeated(decoded))
    return candidates


def assert_unsealed(value: object, *, purpose: str) -> None:
    scalars = _flatten_scalars(value)
    candidates = [candidate for scalar in scalars for candidate in _decoded_text_candidates(scalar)]
    value_scalars = _flatten_values(value)
    value_candidates = [
        candidate for scalar in value_scalars for candidate in _decoded_text_candidates(scalar)
    ]
    compact_individual = [_compact_token(candidate) for candidate in candidates]
    compact_concatenated = _compact_token("".join(candidates))
    compact_values = _compact_token("".join(value_candidates))
    if (
        any(SEALED_ACCESSION in token for token in compact_individual)
        or SEALED_ACCESSION in compact_concatenated
        or SEALED_ACCESSION in compact_values
    ):
        raise SealedSourceRefused(
            f"Refused sealed accession {SEALED_ACCESSION} before {purpose}"
        )


def _safe_component(value: object, *, label: str) -> str:
    token = str(value)
    if (
        not token or token in {".", ".."} or Path(token).name != token
        or "/" in token or "\\" in token
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", token)
    ):
        raise ValueError(f"Unsafe {label}: {token!r}")
    assert_unsealed(token, purpose=f"{label} validation")
    return token


def _feature_tokens(name: str) -> List[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return [token for token in re.split(r"[^a-z0-9]+", expanded.lower()) if token]


def audit_encoder_feature_names(
    feature_names: Sequence[str],
    causal_declarations: Sequence[str] = (),
) -> None:
    """Reject identity/state/outcome proxies while allowing declared cues."""

    # A declaration documents a causal input; it never whitelists an identity,
    # state or outcome proxy.  Safe declarations such as ``external_cue`` pass
    # because they do not match the forbidden vocabulary in the first place.
    declared = {str(value) for value in causal_declarations}
    forbidden = []
    for name in feature_names:
        tokens = _feature_tokens(name)
        joined = "_".join(tokens)
        identity_proxy = (
            any(token in FORBIDDEN_ENCODER_TOKENS for token in tokens)
            or joined in FORBIDDEN_ENCODER_TOKENS
            or joined.replace("_", "") in FORBIDDEN_ENCODER_TOKENS
            or any(value in joined for value in ("future_rna", "future_atac", "target_rna", "target_state"))
        )
        if identity_proxy:
            forbidden.append(name)
    undeclared = [name for name in declared if name not in set(map(str, feature_names))]
    if undeclared:
        raise ValueError(f"Causal declarations are not encoder inputs: {sorted(undeclared)}")
    if forbidden:
        raise ValueError(f"Direct identity/state proxies found in encoder inputs: {forbidden}")


def validate_context_record(record: Mapping[str, object]) -> None:
    """Validate observed/unknown/transferred context semantics at corpus level."""

    state = record.get("provenance_state")
    if state not in {"observed", "reference_transferred", "model_inferred", "unknown"}:
        raise ValueError(f"Unsupported provenance_state {state!r}")
    known = record.get("known")
    if not isinstance(known, bool):
        raise ValueError("Context known mask must be boolean")
    uncertainty = record.get("uncertainty")
    if uncertainty is None or not np.isfinite(float(uncertainty)) or float(uncertainty) < 0:
        raise ValueError("Context uncertainty must be finite and non-negative")
    if state == "unknown":
        if known or float(record.get("evidence_weight", 0.0)) != 0.0:
            raise ValueError("Unknown context must be masked with zero evidence weight")
    elif not known:
        raise ValueError("Known provenance states cannot carry a false known mask")
    if state in {"reference_transferred", "model_inferred"}:
        if not record.get("method_lineage") or float(uncertainty) <= 0:
            raise ValueError("Transferred/inferred context requires method lineage and positive uncertainty")
    if state == "observed" and record.get("method_lineage") == "anatomy_label_as_coordinate":
        raise ValueError("An anatomy label is not an observed spatial coordinate")
    assert_unsealed(record.get("source_accessions", ()), purpose="context provenance validation")


def validate_initial_provenance(
    records: Sequence[Mapping[str, object]], prediction_origin_time: float
) -> None:
    for record in records:
        measurement_time = float(record.get("measurement_time", float("inf")))
        if measurement_time > float(prediction_origin_time):
            raise ValueError("A future measurement cannot initialize an earlier prediction")
        lineage = record.get("method_lineage", "")
        if any(token in "_".join(_feature_tokens(str(lineage))) for token in ("future_rna", "future_atac")):
            raise ValueError("Initial context lineage depends on a future outcome")
        audit_encoder_feature_names([str(record.get("feature_name", ""))])
        assert_unsealed(record, purpose="initial-state provenance validation")


def _safe_url(url: str, allowed_hosts: Sequence[str], *, purpose: str) -> None:
    assert_unsealed(url, purpose=f"URL resolution for {purpose}")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"Only absolute HTTPS source URLs are allowed: {url!r}")
    hosts = {str(host).lower() for host in allowed_hosts}
    if parsed.hostname.lower() not in hosts:
        raise ValueError(f"Source host is not allowlisted: {parsed.hostname}")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: Sequence[str]) -> None:
        super().__init__()
        self.allowed_hosts = tuple(allowed_hosts)
        self.chain: List[Dict[str, object]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        effective = urljoin(req.full_url, str(newurl))
        _safe_url(effective, self.allowed_hosts, purpose="redirect following")
        self.chain.append({"from": req.full_url, "to": effective, "status": int(code)})
        return super().redirect_request(req, fp, code, msg, headers, effective)


def _content_disposition_candidates(headers: Mapping[str, str]) -> List[str]:
    value = str(headers.get("Content-Disposition", ""))
    parameters: List[Tuple[str, str]] = []
    for part in value.split(";")[1:]:
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        parameters.append((key.strip().lower(), raw.strip().strip('"')))
    candidates = [raw for key, raw in parameters if key in {"filename", "filename*", "name"}]
    continuations = sorted(
        ((int(match.group(1)), raw) for key, raw in parameters
         if (match := re.fullmatch(r"filename\*(\d+)\*?", key))),
        key=lambda item: item[0],
    )
    if continuations and [index for index, _ in continuations] == list(range(len(continuations))):
        joined = "".join(raw for _, raw in continuations)
        candidates.append(re.sub(r"^UTF-8''", "", joined, flags=re.I))
    return [_decode_repeated(candidate) for candidate in candidates if candidate]


def _content_disposition_filename(headers: Mapping[str, str]) -> str:
    candidates = _content_disposition_candidates(headers)
    return candidates[0] if candidates else ""


def _validate_artifact_payload(path: Path, content_kind: str = "") -> str:
    """Fail before publication when a recognized container is truncated."""

    if re.search(r"\.gz(?:\.part)?$", path.name.lower()):
        try:
            with gzip.open(path, "rb") as handle:
                for _ in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    pass
        except (OSError, EOFError) as error:
            raise CorpusIntegrityError(f"Unreadable gzip payload: {path}") from error
        return "gzip_eof_crc"
    return "byte_stream"


def _download_receipt_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + ".download.json")


def _download_https(
    url: str,
    destination: Path,
    *,
    allowed_hosts: Sequence[str],
    expected_sha256: Optional[str] = None,
    content_kind: str = "",
) -> Dict[str, object]:
    """Download atomically with bounded retries, safe redirects and resume."""

    _safe_url(url, allowed_hosts, purpose="network request")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Optional[BaseException] = None
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        handler = _SafeRedirectHandler(allowed_hosts)
        opener = urllib.request.build_opener(handler)
        request = urllib.request.Request(url, headers=headers)
        try:
            response = opener.open(request, timeout=120)
        except urllib.error.HTTPError as error:
            if error.code == 416 and offset:
                content_range = str(error.headers.get("Content-Range", ""))
                match = re.fullmatch(r"bytes \*/(\d+)", content_range)
                if match and int(match.group(1)) == offset:
                    validation = _validate_artifact_payload(partial, content_kind)
                    digest = sha256_file(partial)
                    if expected_sha256 and digest.lower() != expected_sha256.lower():
                        raise CorpusIntegrityError("Publisher checksum mismatch after HTTP 416 resume")
                    record = {
                        "requested_url": url,
                        "effective_url": url,
                        "redirect_chain": handler.chain,
                        "sha256": digest,
                        "byte_size": offset,
                        "checksum_provenance": "publisher_supplied" if expected_sha256 else "locally_computed_not_publisher_supplied",
                        "attachment_filename": None,
                        "content_location": None,
                        "content_range": content_range,
                        "expected_total_bytes": offset,
                        "container_validation": validation,
                    }
                    atomic_json(_download_receipt_path(destination), record)
                    os.replace(partial, destination)
                    return record
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
            continue

        try:
            with response:
                effective_url = response.geturl()
                _safe_url(effective_url, allowed_hosts, purpose="effective response")
                disposition = str(response.headers.get("Content-Disposition", ""))
                if disposition:
                    assert_unsealed(disposition, purpose="complete attachment header inspection")
                    for candidate in _content_disposition_candidates(response.headers):
                        assert_unsealed(candidate, purpose="decoded attachment filename inspection")
                attachment = _content_disposition_filename(response.headers)
                content_location = str(response.headers.get("Content-Location", ""))
                if content_location:
                    resolved_location = urljoin(effective_url, content_location)
                    _safe_url(resolved_location, allowed_hosts, purpose="Content-Location inspection")
                status = int(getattr(response, "status", response.getcode()))
                content_length = response.headers.get("Content-Length")
                expected_body = int(content_length) if content_length not in (None, "") else None
                content_range = str(response.headers.get("Content-Range", ""))
                expected_total: Optional[int] = None
                if offset and status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                    if not match or int(match.group(1)) != offset:
                        partial.unlink(missing_ok=True)
                        raise CorpusIntegrityError("Invalid resumed Content-Range")
                    if int(match.group(2)) < int(match.group(1)):
                        raise CorpusIntegrityError("Invalid resumed byte interval")
                    if expected_body is not None and expected_body != int(match.group(2)) - int(match.group(1)) + 1:
                        raise CorpusIntegrityError("Content-Length disagrees with Content-Range")
                    expected_total = None if match.group(3) == "*" else int(match.group(3))
                    mode = "ab"
                else:
                    if offset:
                        partial.unlink(missing_ok=True)
                        offset = 0
                    mode = "wb"
                    expected_total = expected_body
                body_bytes = 0
                next_progress = 64 * 1024 * 1024
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        body_bytes += len(block)
                        if body_bytes >= next_progress:
                            print(f"   downloaded {destination.name}: {(offset + body_bytes) / 1024**2:.1f} MB", flush=True)
                            next_progress += 64 * 1024 * 1024
                if expected_body is not None and body_bytes != expected_body:
                    raise CorpusIntegrityError("Response body length disagrees with Content-Length")
                if expected_total is not None and partial.stat().st_size != expected_total:
                    raise CorpusIntegrityError("Completed file length disagrees with server total")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
            continue

        validation = _validate_artifact_payload(partial, content_kind)
        digest = sha256_file(partial)
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise CorpusIntegrityError(
                f"Publisher checksum mismatch for {url}: expected {expected_sha256}, got {digest}"
            )
        record = {
            "requested_url": url,
            "effective_url": effective_url,
            "redirect_chain": handler.chain,
            "sha256": digest,
            "byte_size": partial.stat().st_size,
            "checksum_provenance": "publisher_supplied" if expected_sha256 else "locally_computed_not_publisher_supplied",
            "attachment_filename": attachment or None,
            "content_location": content_location or None,
            "content_range": content_range or None,
            "expected_total_bytes": expected_total,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "container_validation": validation,
        }
        atomic_json(_download_receipt_path(destination), record)
        os.replace(partial, destination)
        return record
    raise RuntimeError(f"Download retry loop exhausted for {url}") from last_error


def _validate_download_record(
    record: Mapping[str, object],
    destination: Path,
    *,
    allowed_hosts: Sequence[str],
    expected_requested_url: str,
) -> Dict[str, object]:
    assert_unsealed(record, purpose="download provenance persistence")
    requested = str(record.get("requested_url", ""))
    effective = str(record.get("effective_url", ""))
    if requested != expected_requested_url:
        raise CorpusIntegrityError("Downloader changed the requested source URL")
    _safe_url(requested, allowed_hosts, purpose="download-record validation")
    _safe_url(effective, allowed_hosts, purpose="download-record effective URL validation")
    chain = record.get("redirect_chain", [])
    if not isinstance(chain, list):
        raise CorpusIntegrityError("Downloader redirect_chain must be a list")
    for hop in chain:
        if not isinstance(hop, Mapping):
            raise CorpusIntegrityError("Downloader redirect hop is malformed")
        for key in ("from", "to"):
            if hop.get(key):
                _safe_url(str(hop[key]), allowed_hosts, purpose="download-record redirect validation")
    for key in ("attachment_filename", "content_location"):
        if record.get(key):
            assert_unsealed(record[key], purpose=f"download-record {key} validation")
    observed_hash = sha256_file(destination)
    observed_size = destination.stat().st_size
    if str(record.get("sha256", "")) != observed_hash:
        raise CorpusIntegrityError("Downloader-reported SHA-256 disagrees with the materialized file")
    if int(record.get("byte_size", -1)) != observed_size:
        raise CorpusIntegrityError("Downloader-reported byte size disagrees with the materialized file")
    expected_total = record.get("expected_total_bytes")
    if expected_total is not None and int(expected_total) != observed_size:
        raise CorpusIntegrityError("Downloader expected total disagrees with the materialized file")
    return dict(record)


def load_registry(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("v6.1 source registry must be a JSON object")
    return payload


def validate_registry(registry: Mapping[str, object]) -> Dict[str, object]:
    if registry.get("schema_version") not in REGISTRY_SCHEMA_VERSIONS:
        raise ValueError(
            f"Registry schema_version must be one of {sorted(REGISTRY_SCHEMA_VERSIONS)!r}"
        )
    exclusions = registry.get("sealed_exclusions")
    if not isinstance(exclusions, list) or SEALED_ACCESSION not in exclusions:
        raise ValueError(f"Registry must explicitly seal {SEALED_ACCESSION}")
    allowed_hosts = registry.get("allowed_hosts")
    if not isinstance(allowed_hosts, list) or not allowed_hosts:
        raise ValueError("Registry allowed_hosts must be a nonempty list")
    studies = registry.get("studies")
    if not isinstance(studies, list) or not studies:
        raise ValueError("Registry studies must be a nonempty list")
    seen = set()
    for study in studies:
        if not isinstance(study, Mapping):
            raise ValueError("Each study record must be an object")
        accession = str(study.get("accession", ""))
        if not accession or accession in seen:
            raise ValueError(f"Missing or duplicate study accession {accession!r}")
        _safe_component(accession, label="study accession")
        seen.add(accession)
        if accession == SEALED_ACCESSION:
            raise ValueError("The sealed study cannot be a measurement study record")
        assert_unsealed(study, purpose="study registry validation")
        adapter = study.get("adapter")
        if adapter not in ALLOWED_ADAPTERS:
            raise ValueError(f"{accession}: unsupported adapter {adapter!r}")
        partition = study.get("partition", "train")
        if partition not in ALLOWED_PARTITIONS:
            raise ValueError(f"{accession}: invalid partition {partition!r}")
        _study_partition(study)  # also rejects policy/partition contradictions
        role = str(study.get("role", ""))
        if not role or any(token in role.lower() for token in ("external_test", "heldout", "held_out", "sealed_test")):
            raise ValueError(f"{accession}: v6.1 accepts development/reference/acquisition roles only")
        if not study.get("source_assembly") or not study.get("measurement_level"):
            raise ValueError(f"{accession}: source_assembly and measurement_level are required")
        artifacts = study.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError(f"{accession}: artifacts must be a list")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise ValueError(f"{accession}: artifact must be an object")
            for key in ("artifact_id", "role", "filename", "url"):
                if not artifact.get(key):
                    raise ValueError(f"{accession}: artifact missing {key}")
            _safe_component(artifact["artifact_id"], label=f"{accession} artifact_id")
            _safe_component(artifact["filename"], label=f"{accession} artifact filename")
            for scope_key in ("subcohort", "matrix_scope"):
                if artifact.get(scope_key) is not None:
                    _safe_component(artifact[scope_key], label=f"{accession} {scope_key}")
            assert_unsealed(artifact, purpose="registry validation")
            _safe_url(str(artifact["url"]), allowed_hosts, purpose="registry validation")
            checksum = artifact.get("sha256", artifact.get("checksum"))
            if checksum is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(checksum)):
                raise ValueError(f"{accession}: invalid SHA-256 for {artifact['artifact_id']}")
    return {
        "valid": True,
        "studies": len(studies),
        "sealed_exclusions": list(exclusions),
        "registry_sha256": sha256_json(registry),
    }


@dataclass
class CorpusBlock:
    matrix: sparse.csr_matrix
    features: List[str]
    observations: List[Dict[str, object]]
    modality: str = "rna"
    pairing: Optional[Dict[str, object]] = None
    feature_metadata: Optional[List[Dict[str, object]]] = None

    def validate(self) -> None:
        if self.matrix.shape != (len(self.observations), len(self.features)):
            raise ValueError(
                f"Block shape {self.matrix.shape} disagrees with "
                f"{len(self.observations)} observations/{len(self.features)} features"
            )
        if len(set(self.features)) != len(self.features):
            raise ValueError("Feature identifiers must be unique inside a block")
        if self.feature_metadata is not None:
            if len(self.feature_metadata) != len(self.features):
                raise ValueError("Feature metadata must align one-to-one with features")
            for feature, metadata in zip(self.features, self.feature_metadata):
                if str(metadata.get("feature_id", feature)) != feature:
                    raise ValueError("Feature metadata primary ID disagrees with matrix feature")
        ids = [str(value.get("observation_id", "")) for value in self.observations]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError("Observation identifiers must be nonempty and unique")
        if self.matrix.data.size:
            if not np.isfinite(self.matrix.data).all() or (self.matrix.data < 0).any():
                raise ValueError("Raw assay values must be finite and non-negative")
        for observation in self.observations:
            if observation.get("partition") not in ALLOWED_PARTITIONS:
                raise ValueError("Every observation needs a valid corpus partition")
            if not observation.get("group_id"):
                raise ValueError("Every observation needs a biological group_id")


def semantic_sparse_hash(matrix: sparse.spmatrix) -> str:
    value = matrix.tocsr(copy=True)
    value.sort_indices()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    for array in (value.indptr, value.indices, value.data):
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _canonical_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _study_partition(study: Mapping[str, object]) -> str:
    explicit = study.get("partition")
    policy = study.get("split_policy", {})
    mode = str(policy.get("mode", "")) if isinstance(policy, Mapping) else ""
    policy_partition = None
    if "validation" in mode:
        policy_partition = "validation"
    elif "reference" in mode:
        policy_partition = "reference"
    elif mode and "group_first" not in mode and "defer" not in mode:
        policy_partition = "train"
    if explicit in ALLOWED_PARTITIONS:
        if policy_partition is not None and str(explicit) != policy_partition:
            raise ValueError(
                f"Study partition {explicit!r} contradicts split_policy mode {mode!r}"
            )
        return str(explicit)
    return policy_partition or "train"


def _dense_prefix_map(study: Mapping[str, object]) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Materialize deterministic biological-sample partitions before parsing cells."""

    records = study.get("sample_map", [])
    if not isinstance(records, list) or not records:
        raise ValueError("dense_sc_barcode_maps requires a nonempty sample_map")
    policy = study.get("split_policy", {})
    fraction = float(policy.get("validation_fraction", 0.0)) if isinstance(policy, Mapping) else 0.0
    seed = int(policy.get("seed", 0)) if isinstance(policy, Mapping) else 0
    by_stage: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("dense sample_map entries must be objects")
        by_stage[str(record.get("stage", "unspecified"))].append(record)
    validation_groups = set()
    if fraction > 0:
        for stage, stage_records in sorted(by_stage.items()):
            # A stage with one biological unit cannot be split into independent
            # train/validation cells. Keep it in training and report its limit.
            if len(stage_records) < 2:
                continue
            ranked = sorted(
                stage_records,
                key=lambda record: hashlib.sha256(
                    f"{seed}:{stage}:{record.get('experimental_unit_id')}".encode()
                ).hexdigest(),
            )
            count = max(1, int(round(len(ranked) * fraction)))
            count = min(count, len(ranked) - 1)
            validation_groups.update(str(record["experimental_unit_id"]) for record in ranked[:count])
    result: Dict[Tuple[str, str], Dict[str, object]] = {}
    for record in records:
        scope = str(record.get("matrix_scope", "default"))
        prefix = str(record.get("barcode_prefix", ""))
        if not prefix:
            raise ValueError("dense sample_map record lacks barcode_prefix")
        item = dict(record)
        group = str(item.get("experimental_unit_id", f"{study['accession']}:{scope}:{prefix}"))
        item["group_id"] = group
        item["partition"] = "validation" if group in validation_groups else "train"
        key = (scope, prefix)
        if key in result:
            raise ValueError(f"Duplicate dense sample prefix mapping {key}")
        result[key] = item
    return result


def _sample_annotation(
    sample_name: str,
    study: Mapping[str, object],
    *,
    fallback_partition: str,
) -> Dict[str, object]:
    sample_map = study.get("sample_map", [])
    matches: List[Tuple[int, Mapping[str, object]]] = []
    for record in sample_map if isinstance(sample_map, list) else []:
        if not isinstance(record, Mapping):
            continue
        score = _sample_record_score(sample_name, record)
        if score:
            matches.append((score, record))
    if matches:
        matches.sort(key=lambda pair: (-pair[0], str(pair[1].get("sample_accession", ""))))
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            raise ValueError(
                f"Matrix column {sample_name!r} ambiguously matches multiple sample records"
            )
        result = dict(matches[0][1])
        result.pop("pattern", None)
        result.setdefault("sample", result.get("sample_accession", sample_name))
        result.setdefault("group_id", f"{study['accession']}:{result['sample']}")
        # Whole-study partitions are immutable.  Per-sample partition fields
        # cannot relabel a development-validation study as training data.
        result["partition"] = fallback_partition
        return result
    return {
        "sample": sample_name,
        "group_id": f"{study['accession']}:{sample_name}",
        "partition": fallback_partition,
    }


def _token_in_sample(token: str, sample_name: str) -> bool:
    token = str(token).strip()
    if not token:
        return False
    # Dots are retained as numeric/dose boundaries, preventing ``1uM`` from
    # matching the ``0.1uM`` component. Slashes, spaces and underscores remain
    # legitimate component separators around deposited sample labels.
    pattern = r"(?<![A-Za-z0-9.])" + re.escape(token) + r"(?![A-Za-z0-9.])"
    return re.search(pattern, sample_name, flags=re.I) is not None


def _sample_record_score(sample_name: str, record: Mapping[str, object]) -> int:
    exact = record.get("sample")
    pattern = record.get("pattern")
    accession = record.get("sample_accession")
    tokens = record.get("matrix_column_tokens", [])
    if exact is not None and str(exact) == sample_name:
        return 10_000
    if accession is not None and _token_in_sample(str(accession), sample_name):
        return 9_000 + len(str(accession))
    if pattern is not None and re.search(str(pattern), sample_name):
        return 8_000 + len(str(pattern))
    if isinstance(tokens, list):
        hits = [str(token) for token in tokens if _token_in_sample(str(token), sample_name)]
        if hits:
            return len(hits) * 100 + sum(map(len, hits))
    return 0


def _read_delimited_header(path: Path) -> Tuple[Optional[str], List[str]]:
    with _open_text(path) as handle:
        first = ""
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                first = line.rstrip("\r\n")
                break
    if not first:
        raise ValueError(f"No non-comment table header found in {path}")
    if "\t" in first:
        delimiter: Optional[str] = "\t"
        header = next(csv.reader([first], delimiter=delimiter))
    elif "," in first:
        delimiter = ","
        header = next(csv.reader([first], delimiter=delimiter))
    else:
        delimiter = None
        header = first.split()
    return delimiter, header


def _iter_delimited_rows(path: Path, delimiter: Optional[str]):
    with _open_text(path) as handle:
        if delimiter is None:
            for line in handle:
                if line.strip() and not line.lstrip().startswith("#"):
                    yield line.strip().split()
        else:
            yield from csv.reader(
                (line for line in handle if line.strip() and not line.lstrip().startswith("#")),
                delimiter=delimiter,
            )


def _context_json(
    annotation: Mapping[str, object],
    study: Mapping[str, object],
) -> str:
    """Preserve deposited biological context outside the encoder."""

    excluded = {
        "matrix_column_tokens", "pattern", "partition", "group_id",
    }
    sample_context = {
        str(key): value for key, value in annotation.items()
        if key not in excluded
    }
    study_context = {
        str(key): study[key]
        for key in (
            "accession", "assay", "anatomy", "organism", "role",
            "source_assembly", "measurement_level", "tissue",
        )
        if key in study
    }
    explicit = study.get("study_context")
    if isinstance(explicit, Mapping):
        study_context["study_context"] = dict(explicit)
    return json.dumps(
        {"sample": sample_context, "study": study_context},
        sort_keys=True,
        separators=(",", ":"),
    )


def _tabular_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _physical_time_hours(annotation: Mapping[str, object]) -> Optional[float]:
    if annotation.get("physical_time_hours") is not None:
        return float(annotation["physical_time_hours"])
    if annotation.get("time_dpa") is not None:
        return 24.0 * float(annotation["time_dpa"])
    if annotation.get("time_post_transfection_days") is not None:
        return 24.0 * float(annotation["time_post_transfection_days"])
    return None


def _time_axis_basis(annotation: Mapping[str, object]) -> Optional[str]:
    if annotation.get("physical_time_hours") is not None:
        return "deposited_physical_time_hours"
    if annotation.get("time_dpa") is not None:
        return "hours_post_amputation_from_deposited_time_dpa"
    if annotation.get("time_post_transfection_days") is not None:
        return "hours_post_transfection_from_deposited_days"
    return None


def parse_bulk_gene_table(path: Path, study: Mapping[str, object]) -> CorpusBlock:
    delimiter, header = _read_delimited_header(path)
    if len(header) < 2:
        raise ValueError(f"Bulk table has fewer than two columns: {path}")
    metadata_names = {
        "geneid", "gene", "geneidversion", "chr", "chromosome", "start",
        "end", "strand", "length", "description", "symbol", "gene_name",
    }
    configured = study.get("sample_map", [])
    configured_patterns = [
        record for record in configured if isinstance(record, Mapping)
    ] if isinstance(configured, list) else []
    sample_indices: List[int] = []
    matched_record_ids: List[str] = []
    for index, name in enumerate(header[1:], start=1):
        token = _canonical_header(name)
        if token in {_canonical_header(value) for value in metadata_names}:
            continue
        if configured_patterns:
            scores = [(_sample_record_score(name, record), record) for record in configured_patterns]
            scores = [pair for pair in scores if pair[0] > 0]
            if not scores:
                continue
            scores.sort(key=lambda pair: (-pair[0], str(pair[1].get("sample_accession", pair[1].get("sample", "")))))
            if len(scores) > 1 and scores[0][0] == scores[1][0]:
                raise ValueError(f"Matrix column {name!r} ambiguously matches multiple sample records")
            matched_id = str(scores[0][1].get("sample_accession", scores[0][1].get("sample", "")))
            if not matched_id or matched_id in matched_record_ids:
                raise ValueError(f"Sample record {matched_id!r} maps to multiple matrix columns")
            matched_record_ids.append(matched_id)
        sample_indices.append(index)
    if configured_patterns:
        expected_ids = {
            str(record.get("sample_accession", record.get("sample", "")))
            for record in configured_patterns
        }
        if set(matched_record_ids) != expected_ids:
            missing = sorted(expected_ids.difference(matched_record_ids))
            raise ValueError(
                f"Deposited matrix columns did not resolve one-to-one to sample_map; missing={missing}"
            )
    elif not sample_indices:
        # FeatureCounts reserves its first six columns for annotation.  For
        # other matrices, every column after the identifier is a candidate.
        canonical = [_canonical_header(value) for value in header]
        start = 6 if canonical[:6] == ["geneid", "chr", "start", "end", "strand", "length"] else 1
        sample_indices = list(range(start, len(header)))
    sample_names = [header[index] for index in sample_indices]
    rows: Dict[str, np.ndarray] = {}
    reader = _iter_delimited_rows(path, delimiter)
    next(reader, None)
    for row_number, row in enumerate(reader, start=2):
        if not row or not row[0].strip():
            continue
        if max(sample_indices) >= len(row):
            raise ValueError(f"Short row {row_number} in {path.name}")
        try:
            values = np.asarray([float(row[index]) for index in sample_indices], dtype=np.float64)
        except ValueError as error:
            raise ValueError(f"Non-numeric count at row {row_number} in {path.name}") from error
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid raw abundance at row {row_number} in {path.name}")
        feature = row[0].strip()
        if feature in rows:
            rows[feature] += values
        else:
            rows[feature] = values
    features = sorted(rows)
    dense = np.vstack([rows[feature] for feature in features]).T if features else np.zeros((len(sample_names), 0))
    accession = str(study["accession"])
    default_partition = _study_partition(study)
    observations = []
    for sample in sample_names:
        annotation = _sample_annotation(sample, study, fallback_partition=default_partition)
        observations.append({
            "observation_id": f"{accession}:{sample}",
            "deposited_id": sample,
            "study_accession": accession,
            "sample_id": str(annotation.get("sample", sample)),
            "group_id": str(annotation["group_id"]),
            "partition": str(annotation["partition"]),
            "condition": annotation.get("condition", annotation.get("construct")),
            "cue": annotation.get("cue"),
            "construct": annotation.get("construct"),
            "perturbation": annotation.get("perturbation", annotation.get("cue", annotation.get("construct"))),
            "dose": annotation.get("dose"),
            "physical_time_hours": _physical_time_hours(annotation),
            "time_axis_basis": _time_axis_basis(annotation),
            "time_dpa": annotation.get("time_dpa"),
            "time_post_transfection_days": annotation.get("time_post_transfection_days"),
            "replicate": annotation.get("replicate", annotation.get("biological_replicate")),
            "blastemas_pooled": annotation.get("blastemas_pooled"),
            "amputation_level": annotation.get("amputation_level"),
            "anatomy": annotation.get("anatomy", annotation.get("amputation_level", study.get("anatomy"))),
            "measurement_level": study["measurement_level"],
            "source_assembly": study["source_assembly"],
            "encoder_eligible": False,
            "model_role": "supervision_only",
            "context_provenance": "observed_deposited_sample_annotation",
            "context_json": _context_json(annotation, study),
        })
    block = CorpusBlock(sparse.csr_matrix(dense), features, observations, "rna")
    block.validate()
    return block


def _read_barcode_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with _open_text(path) as handle:
        for row_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"Malformed barcode-map row {row_number} in {path.name}")
            cell_name, barcode = parts[0].strip(), parts[1].strip()
            if not cell_name or not barcode or cell_name in mapping:
                raise ValueError(f"Invalid/duplicate cell name at row {row_number} in {path.name}")
            mapping[cell_name] = barcode
    return mapping


def _canonical_cell_name(value: str) -> str:
    # The deposited GSE121737 sidecars use both S_1 and S1 sample spellings.
    # This is an identifier spelling normalization, not biological pairing.
    return re.sub(r"^([A-Za-z])_(\d+)", r"\1\2", value.strip().strip('"'))


def _matrix_market(path: Path) -> sparse.csr_matrix:
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as handle:
        return mmread(handle).tocsr()


def _parse_dense_gene_by_cell(
    matrix_path: Path,
    cell_mapping: Mapping[str, str],
) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    delimiter, header = _read_delimited_header(matrix_path)
    cells = [value.strip().strip('"') for value in header[1:]]
    canonical_mapping = {_canonical_cell_name(key): key for key in cell_mapping}
    if len(canonical_mapping) != len(cell_mapping):
        raise ValueError("Barcode-map identifiers collide after deposited spelling normalization")
    mapping_keys = set(canonical_mapping)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError(f"Dense matrix has invalid cell header: {matrix_path.name}")
    canonical_header_cells = [_canonical_cell_name(cell) for cell in cells]
    overlap = sum(cell in mapping_keys for cell in canonical_header_cells) / len(cells)
    if overlap < 0.95:
        # Some deposited dense tables are cells x genes.  Verify that case from
        # row identifiers, then stream it without guessing from dimensions.
        row_ids = []
        reader = _iter_delimited_rows(matrix_path, delimiter)
        next(reader, None)
        for index, row in enumerate(reader):
                if row:
                    row_ids.append(_canonical_cell_name(row[0]))
                if index >= 199:
                    break
        row_overlap = sum(value in mapping_keys for value in row_ids) / max(1, len(row_ids))
        if row_overlap < 0.95:
            raise ValueError(
                f"Neither matrix axis matches the deposited barcode map for {matrix_path.name} "
                f"(header={overlap:.1%}, rows={row_overlap:.1%})"
            )
        features = cells
        if len(set(features)) != len(features):
            raise ValueError("Cell-by-gene matrix has duplicate feature identifiers")
        observed_cells: List[str] = []
        data_parts: List[np.ndarray] = []
        index_parts: List[np.ndarray] = []
        indptr = [0]
        reader = _iter_delimited_rows(matrix_path, delimiter)
        next(reader, None)
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(features) + 1:
                raise ValueError(f"Dense matrix row width changed at line {row_number}")
            canonical = _canonical_cell_name(row[0])
            if canonical not in canonical_mapping:
                raise ValueError(f"Cell {row[0]!r} is absent from deposited barcode map")
            values = np.asarray(row[1:], dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid abundance at line {row_number}")
            nz = np.flatnonzero(values).astype(np.int64, copy=False)
            index_parts.append(nz)
            data_parts.append(values[nz])
            indptr.append(indptr[-1] + nz.size)
            observed_cells.append(canonical_mapping[canonical])
        if len(observed_cells) != len(set(observed_cells)):
            raise ValueError("Cell-by-gene matrix repeats deposited cell identifiers")
        indices = np.concatenate(index_parts) if index_parts else np.asarray([], dtype=np.int64)
        data = np.concatenate(data_parts) if data_parts else np.asarray([], dtype=np.float64)
        matrix = sparse.csr_matrix(
            (data, indices, np.asarray(indptr, dtype=np.int64)),
            shape=(len(observed_cells), len(features)),
        )
        return matrix, features, observed_cells
    features: List[str] = []
    row_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    data_parts: List[np.ndarray] = []
    reader = _iter_delimited_rows(matrix_path, delimiter)
    next(reader, None)
    for feature_index, row in enumerate(reader):
            if len(row) != len(cells) + 1:
                raise ValueError(f"Dense matrix row width changed at feature {feature_index}")
            feature = row[0].strip().strip('"')
            if not feature:
                raise ValueError("Dense matrix contains an empty feature identifier")
            values = np.asarray(row[1:], dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid abundance for feature {feature}")
            nz = np.flatnonzero(values)
            if nz.size:
                row_parts.append(nz.astype(np.int64, copy=False))
                col_parts.append(np.full(nz.size, feature_index, dtype=np.int64))
                data_parts.append(values[nz])
            features.append(feature)
    if len(set(features)) != len(features):
        # Duplicate gene identifiers are aggregated deterministically.
        unique = sorted(set(features))
        unique_index = {value: index for index, value in enumerate(unique)}
        remap = np.asarray([unique_index[value] for value in features], dtype=np.int64)
    else:
        unique = features
        remap = np.arange(len(features), dtype=np.int64)
    if data_parts:
        row_index = np.concatenate(row_parts)
        old_columns = np.concatenate(col_parts)
        column_index = remap[old_columns]
        data = np.concatenate(data_parts)
        matrix = sparse.coo_matrix(
            (data, (row_index, column_index)), shape=(len(cells), len(unique))
        ).tocsr()
    else:
        matrix = sparse.csr_matrix((len(cells), len(unique)), dtype=np.float64)
    deposited_cells = [canonical_mapping[value] for value in canonical_header_cells]
    return matrix, unique, deposited_cells


def parse_dense_sc_barcode_maps(
    files: Mapping[str, Path], study: Mapping[str, object]
) -> List[Tuple[str, CorpusBlock]]:
    artifacts = study.get("artifacts", [])
    by_subcohort: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for artifact in artifacts:
        subcohort = str(artifact.get("subcohort", artifact.get("matrix_scope", "default")))
        role = str(artifact["role"])
        canonical_role = {
            "cell_barcode_map": "barcode_map",
            "single_cell_gene_counts": "matrix",
        }.get(role, role)
        by_subcohort[subcohort][canonical_role] = files[str(artifact["artifact_id"])]
    result = []
    prefix_map = _dense_prefix_map(study)
    accession = str(study["accession"])
    for subcohort, parts in sorted(by_subcohort.items()):
        if "matrix" not in parts or "barcode_map" not in parts:
            raise ValueError(f"{accession}/{subcohort}: matrix and barcode_map are required")
        barcode_map = _read_barcode_map(parts["barcode_map"])
        matrix, features, cells = _parse_dense_gene_by_cell(parts["matrix"], barcode_map)
        observations = []
        for cell in cells:
            prefix = re.split(r"[_-]bc", cell, maxsplit=1, flags=re.I)[0]
            record = prefix_map.get((subcohort, prefix))
            if not isinstance(record, Mapping):
                raise ValueError(f"No biological sample mapping for cell prefix {prefix!r}")
            group = str(record.get("group_id", record.get("experimental_unit_id", f"{accession}:{subcohort}:{prefix}")))
            partition = str(record.get("partition", "train"))
            observations.append({
                "observation_id": f"{accession}:{subcohort}:{cell}",
                "deposited_id": cell,
                "deposited_barcode": barcode_map[cell],
                "study_accession": accession,
                "sample_id": prefix,
                "group_id": group,
                "partition": partition,
                "condition": record.get("condition"),
                "physical_time_hours": _physical_time_hours(record),
                "time_axis_basis": _time_axis_basis(record),
                "time_dpa": record.get("time_dpa"),
                "replicate": record.get("replicate", record.get("biological_replicate")),
                "morphological_stage": record.get("morphological_stage", record.get("stage")),
                "anatomy": record.get("anatomy", study.get("anatomy")),
                "measurement_level": "single_cell",
                "source_assembly": study["source_assembly"],
                "encoder_eligible": False,
                "model_role": "supervision_or_representation_only",
                "context_provenance": "observed_deposited_sample_annotation",
                "context_json": _context_json(record, study),
            })
        block = CorpusBlock(
            matrix,
            features,
            observations,
            "rna",
            pairing={
                "mode": "unpaired_population",
                "fabricated": False,
                "expression_similarity_used": False,
                "cell_label_matching_used": False,
                "evidence": "destructive single-cell samples; deposited prefixes define biological samples only",
            },
        )
        block.validate()
        result.append((subcohort, block))
    return result


def _read_lines(path: Path) -> List[str]:
    with _open_text(path) as handle:
        return [line.rstrip("\r\n").split("\t")[0] for line in handle if line.strip()]


def _read_visium_coordinates(path: Path) -> Dict[str, Dict[str, object]]:
    with _open_text(path) as handle:
        rows = [line.rstrip("\r\n").split(",") for line in handle if line.strip()]
    if not rows:
        raise ValueError("Visium coordinates are empty")
    header_tokens = {_canonical_header(value) for value in rows[0]}
    has_header = "barcode" in header_tokens
    result: Dict[str, Dict[str, object]] = {}
    if has_header:
        header = [_canonical_header(value) for value in rows[0]]
        data_rows = rows[1:]
        barcode_index = header.index("barcode")
        row_index = next((header.index(name) for name in ("pxlrowinfullres", "imagerow", "arrayrow") if name in header), None)
        col_index = next((header.index(name) for name in ("pxlcolinfullres", "imagecol", "arraycol") if name in header), None)
        in_tissue_index = header.index("intissue") if "intissue" in header else None
        array_row_index = header.index("arrayrow") if "arrayrow" in header else None
        array_col_index = header.index("arraycol") if "arraycol" in header else None
    else:
        data_rows = rows
        barcode_index, row_index, col_index = 0, 4, 5
        in_tissue_index, array_row_index, array_col_index = 1, 2, 3
    if row_index is None or col_index is None:
        raise ValueError("Cannot find coordinate columns in Visium positions file")
    for row in data_rows:
        barcode = row[barcode_index]
        if barcode in result:
            raise ValueError(f"Duplicate Visium coordinate barcode {barcode}")
        record = {
            "spatial_row": float(row[row_index]),
            "spatial_col": float(row[col_index]),
        }
        if in_tissue_index is not None:
            record["in_tissue"] = int(float(row[in_tissue_index]))
        if array_row_index is not None:
            record["array_row"] = float(row[array_row_index])
        if array_col_index is not None:
            record["array_col"] = float(row[array_col_index])
        result[barcode] = record
    return result


def parse_visium_bundle(files: Mapping[str, Path], study: Mapping[str, object]) -> CorpusBlock:
    roles = {}
    for artifact in study.get("artifacts", []):
        role = str(artifact["role"])
        canonical_role = {
            "spot_gene_counts": "matrix",
            "spot_barcodes": "barcodes",
            "gene_features": "features",
            "spot_coordinates": "coordinates",
        }.get(role, role)
        roles[canonical_role] = files[str(artifact["artifact_id"])]
    required = {"matrix", "barcodes", "features", "coordinates"}
    missing = required.difference(roles)
    if missing:
        raise ValueError(f"Visium bundle missing {sorted(missing)}")
    matrix = _matrix_market(roles["matrix"])
    barcodes = _read_lines(roles["barcodes"])
    feature_rows = []
    feature_metadata = []
    with _open_text(roles["features"]) as handle:
        for source_index, line in enumerate(handle):
            if line.strip():
                parts = line.rstrip("\r\n").split("\t")
                feature_id = parts[0].strip()
                if not feature_id:
                    raise ValueError("Visium feature has an empty primary identifier")
                feature_rows.append(feature_id)
                feature_metadata.append({
                    "feature_id": feature_id,
                    "feature_name": parts[1].strip() if len(parts) > 1 else "",
                    "feature_type": parts[2].strip() if len(parts) > 2 else "",
                    "source_index": source_index,
                })
    if len(feature_rows) != len(set(feature_rows)):
        raise ValueError("Visium primary feature identifiers are not unique")
    if matrix.shape == (len(feature_rows), len(barcodes)):
        matrix = matrix.transpose().tocsr()
    elif matrix.shape != (len(barcodes), len(feature_rows)):
        raise ValueError("Visium matrix/barcode/feature dimensions disagree")
    coordinates = _read_visium_coordinates(roles["coordinates"])
    missing_coordinates = set(barcodes).difference(coordinates)
    if missing_coordinates:
        raise ValueError(
            f"{len(missing_coordinates)} expression barcodes lack deposited coordinates"
        )
    unused_coordinate_rows = set(coordinates).difference(barcodes)
    accession = str(study["accession"])
    sample_records = study.get("sample_map", [])
    sample_record = sample_records[0] if isinstance(sample_records, list) and sample_records else {}
    specimen = str(
        study.get("specimen_id", sample_record.get("sample_accession", "single_spatial_specimen"))
    )
    observations = [{
        "observation_id": f"{accession}:{barcode}",
        "deposited_id": barcode,
        "study_accession": accession,
        "sample_id": specimen,
        "group_id": f"{accession}:{specimen}",
        "partition": _study_partition(study),
        "physical_time_hours": study.get(
            "physical_time_hours",
            24.0 * float(sample_record["time_dpa"]) if sample_record.get("time_dpa") is not None else None,
        ),
        "anatomy": study.get("anatomy", sample_record.get("tissue")),
        "spatial_row": coordinates[barcode]["spatial_row"],
        "spatial_col": coordinates[barcode]["spatial_col"],
        "in_tissue": coordinates[barcode].get("in_tissue"),
        "array_row": coordinates[barcode].get("array_row"),
        "array_col": coordinates[barcode].get("array_col"),
        "spatial_provenance": "observed_deposited_same_spot_coordinate",
        "measurement_level": "spatial_spot",
        "source_assembly": study["source_assembly"],
        "encoder_eligible": False,
        "model_role": "spatial_reference_only",
        "context_provenance": "observed_deposited_same_spot_and_sample_annotation",
        "context_json": _context_json(sample_record, study),
    } for barcode in barcodes]
    block = CorpusBlock(
        matrix,
        feature_rows,
        observations,
        "rna",
        pairing={
            "mode": "same_spot_exact",
            "verification_status": "verified_from_materialized_schema",
            "schema_materialized": True,
            "identifier_field": "deposited_spot_barcode",
            "expression_barcodes_all_have_coordinates": True,
            "unused_coordinate_rows": len(unused_coordinate_rows),
            "crosswalk_sha256": sha256_json([
                {
                    "barcode": barcode,
                    "spatial_row": coordinates[barcode]["spatial_row"],
                    "spatial_col": coordinates[barcode]["spatial_col"],
                }
                for barcode in sorted(barcodes)
            ]),
            "fabricated": False,
            "expression_similarity_used": False,
            "cell_label_matching_used": False,
        },
        feature_metadata=feature_metadata,
    )
    block.validate()
    return block


def _save_block(
    root: Path,
    name: str,
    block: CorpusBlock,
    *,
    study: Mapping[str, object],
    source_lock: Mapping[str, object],
) -> Dict[str, object]:
    block.validate()
    _safe_component(name, label="bundle name")
    bundle = root / "bundles" / name
    modality_root = bundle / "modalities" / block.modality
    modality_root.mkdir(parents=True, exist_ok=True)
    matrix_path = modality_root / "counts.csr.npz"
    temporary = matrix_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        sparse.save_npz(handle, block.matrix.tocsr(), compressed=True)
    os.replace(temporary, matrix_path)
    feature_path = modality_root / "features.tsv.gz"
    feature_records = []
    for index, feature in enumerate(block.features):
        record = dict(block.feature_metadata[index]) if block.feature_metadata is not None else {}
        record["feature_id"] = feature
        record.setdefault("source_index", index)
        feature_records.append(record)
    feature_fields = ["feature_id", "source_index"] + sorted({
        key for record in feature_records for key in record
        if key not in {"feature_id", "source_index"}
    })
    _stable_gzip_lines(
        feature_path,
        [feature_fields]
        + [[_tabular_value(record.get(field)) for field in feature_fields]
           for record in feature_records],
    )
    observation_fields = sorted({key for row in block.observations for key in row})
    observation_path = bundle / "observations.tsv.gz"
    _stable_gzip_lines(
        observation_path,
        [observation_fields]
        + [[_tabular_value(row.get(field)) for field in observation_fields]
           for row in block.observations],
    )
    pairing_path = bundle / "pairing.json"
    atomic_json(pairing_path, block.pairing or {
        "mode": "single_modality_not_applicable",
        "fabricated": False,
    })
    provenance_path = bundle / "provenance.json"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "provenance_state": "observed",
        "bundle_id": name,
        "study_accession": study["accession"],
        "source_assembly": study["source_assembly"],
        "modality": block.modality,
        "measurement_level": study["measurement_level"],
        "observation_group_field": "group_id",
        "partition_field": "partition",
        "measurement_time_fields": [
            "physical_time_hours", "time_dpa", "time_post_transfection_days",
        ],
        "study_context": study.get("study_context", {}),
        "deposited_sample_map": study.get("sample_map", []),
        "source_artifacts": [
            {
                "artifact_id": item["artifact_id"],
                "role": item["role"],
                "filename": item["filename"],
                "sha256": item["sha256"],
            }
            for item in source_lock["artifacts"]
        ],
        "raw_unnormalized": True,
        "rna_model_role": "supervision_or_reference_only",
        "unknown_is_not_observed_zero": True,
        "model_inferred_values_added": False,
        "identity_and_state_labels_encoder_eligible": False,
    }
    assert_unsealed(provenance, purpose="bundle provenance persistence")
    atomic_json(provenance_path, provenance)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": name,
        "study_accession": study["accession"],
        "modality": block.modality,
        "model_role": "supervision_or_reference_only",
        "raw_unnormalized": True,
        "sparse": True,
        "source_assembly": study["source_assembly"],
        "measurement_level": study["measurement_level"],
        "shape": list(block.matrix.shape),
        "nnz": int(block.matrix.nnz),
        "matrix_semantic_sha256": semantic_sparse_hash(block.matrix),
        "matrix_file_sha256": sha256_file(matrix_path),
        "feature_sha256": sha256_file(feature_path),
        "observation_sha256": sha256_file(observation_path),
        "pairing_sha256": sha256_file(pairing_path),
        "provenance_sha256": sha256_file(provenance_path),
        "source_lock_sha256": source_lock["sha256"],
        "files": {
            "matrix": str(matrix_path.relative_to(root)),
            "features": str(feature_path.relative_to(root)),
            "observations": str(observation_path.relative_to(root)),
            "pairing": str(pairing_path.relative_to(root)),
            "provenance": str(provenance_path.relative_to(root)),
        },
    }
    manifest_path = bundle / "bundle_manifest.json"
    atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.relative_to(root))
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def _load_observations(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _build_split_manifest(root: Path, bundle_manifests: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    groups: Dict[str, str] = {}
    observations_by_group: Dict[str, int] = defaultdict(int)
    for manifest in bundle_manifests:
        rows = _load_observations(root / str(manifest["files"]["observations"]))
        for row in rows:
            group, partition = row["group_id"], row["partition"]
            previous = groups.setdefault(group, partition)
            if previous != partition:
                raise ValueError(f"Biological group {group} crosses {previous}/{partition}")
            observations_by_group[group] += 1
    split = {
        "schema_version": SCHEMA_VERSION,
        "split_unit": "namespaced_biological_specimen_or_sample",
        "split_before_feature_selection": True,
        "groups": [
            {"group_id": group, "partition": groups[group], "observations": observations_by_group[group]}
            for group in sorted(groups)
        ],
        "partitions": {
            partition: sorted(group for group, value in groups.items() if value == partition)
            for partition in sorted(ALLOWED_PARTITIONS)
        },
        "sealed_accessions_present": False,
    }
    split["content_sha256"] = sha256_json(split)
    path = root / "folds" / "development" / "split_manifest.json"
    atomic_json(path, split)
    split["path"] = str(path.relative_to(root))
    split["file_sha256"] = sha256_file(path)
    return split


def _read_features(path: Path) -> List[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row["feature_id"] for row in reader]


def _build_feature_registry(
    root: Path,
    bundle_manifests: Sequence[Mapping[str, object]],
    split_manifest: Mapping[str, object],
    *,
    max_features_per_namespace: int = 5000,
) -> Dict[str, object]:
    fit_groups = set(split_manifest["partitions"]["train"])
    statistics: Dict[Tuple[str, str], MutableMapping[str, float]] = defaultdict(
        lambda: {"n": 0.0, "sum": 0.0, "sumsq": 0.0, "nnz": 0.0}
    )
    namespaces: Dict[Tuple[str, str], str] = {}
    for manifest in bundle_manifests:
        rows = _load_observations(root / str(manifest["files"]["observations"]))
        selected_rows = [index for index, row in enumerate(rows) if row["group_id"] in fit_groups]
        if not selected_rows:
            continue
        matrix = sparse.load_npz(root / str(manifest["files"]["matrix"])).tocsr()[selected_rows]
        features = _read_features(root / str(manifest["files"]["features"]))
        namespace = f"{manifest['source_assembly']}::{manifest['modality']}"
        sums = np.asarray(matrix.sum(axis=0)).ravel()
        sumsq = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
        nnz = np.asarray((matrix != 0).sum(axis=0)).ravel()
        for index, feature in enumerate(features):
            key = (namespace, feature)
            namespaces[key] = namespace
            item = statistics[key]
            item["n"] += matrix.shape[0]
            item["sum"] += float(sums[index])
            item["sumsq"] += float(sumsq[index])
            item["nnz"] += float(nnz[index])
    by_namespace: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for key, item in statistics.items():
        namespace, feature = key
        n = item["n"]
        mean = item["sum"] / n if n else 0.0
        variance = max(0.0, item["sumsq"] / n - mean * mean) if n else 0.0
        prevalence = item["nnz"] / n if n else 0.0
        if prevalence > 0.0:
            by_namespace[namespace].append({
                "feature_id": feature,
                "train_mean": mean,
                "train_variance": variance,
                "train_prevalence": prevalence,
                "train_observations": int(n),
            })
    registries = {}
    for namespace, records in sorted(by_namespace.items()):
        ordered = sorted(records, key=lambda value: (-float(value["train_variance"]), str(value["feature_id"])))
        registries[namespace] = ordered[:max_features_per_namespace]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fit_partition": "train",
        "fit_groups": sorted(fit_groups),
        "fit_groups_sha256": sha256_json(sorted(fit_groups)),
        "parent_split_manifest_sha256": split_manifest["file_sha256"],
        "split_before_feature_selection": True,
        "normalization_fitted": False,
        "assembly_namespaces_merged": False,
        "selection_rule": "positive train prevalence; rank by train variance then lexical ID; cap 5000 per assembly/modality namespace",
        "registries": registries,
    }
    payload["content_sha256"] = sha256_json(payload)
    path = root / "folds" / "development" / "feature_registry.json"
    atomic_json(path, payload)
    payload["path"] = str(path.relative_to(root))
    payload["file_sha256"] = sha256_file(path)
    return payload


def _study_enabled(study: Mapping[str, object], include_timecourse: bool, include_spatial: bool) -> bool:
    tier = study.get("tier", "core")
    if (tier == "timecourse" or study.get("adapter") == "dense_sc_barcode_maps") and not include_timecourse:
        return False
    if (tier == "spatial" or study.get("adapter") == "visium_bundle") and not include_spatial:
        return False
    return bool(study.get("default_download", True))


def _acquire_study(
    root: Path,
    study: Mapping[str, object],
    *,
    allowed_hosts: Sequence[str],
    fetcher: Optional[Callable[[str, Path, Sequence[str], Optional[str]], Mapping[str, object]]] = None,
) -> Tuple[Dict[str, Path], Dict[str, object]]:
    accession = str(study["accession"])
    assert_unsealed(accession, purpose="source directory creation")
    _safe_component(accession, label="study accession")
    source_root = root / "raw" / accession
    raw_root = (root / "raw").resolve()
    if source_root.resolve().parent != raw_root:
        raise ValueError(f"Study source directory escaped raw root: {source_root}")
    lock_path = source_root / "source.lock.json"
    artifacts = study.get("artifacts", [])
    existing = json.loads(lock_path.read_text()) if lock_path.exists() else None
    acquired: Dict[str, Path] = {}
    lock_records = []
    for artifact in artifacts:
        artifact_id = str(artifact["artifact_id"])
        filename = str(artifact["filename"])
        _safe_component(artifact_id, label="artifact_id")
        _safe_component(filename, label="artifact filename")
        assert_unsealed(filename, purpose="destination creation")
        destination = source_root / filename
        if destination.resolve().parent != source_root.resolve() or source_root.resolve().parent != raw_root:
            raise ValueError(f"Artifact destination escaped source root: {destination}")
        receipt_path = _download_receipt_path(destination)
        partial_path = destination.with_suffix(destination.suffix + ".part")
        old = None
        if existing:
            old = next((item for item in existing.get("artifacts", []) if item["artifact_id"] == artifact_id), None)
        # The production downloader writes a validated receipt before its
        # atomic final rename.  A disconnect in that tiny window can therefore
        # promote the complete partial without reading the network again.
        if not destination.exists() and partial_path.exists() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            recovered = _validate_download_record(
                receipt,
                partial_path,
                allowed_hosts=allowed_hosts,
                expected_requested_url=str(artifact["url"]),
            )
            _validate_artifact_payload(partial_path, str(artifact.get("content_kind", "")))
            if old and (recovered["sha256"] != old["sha256"] or recovered["byte_size"] != old["byte_size"]):
                raise CorpusIntegrityError("Completed partial disagrees with immutable source lock")
            os.replace(partial_path, destination)
        if destination.exists() and old:
            digest = sha256_file(destination)
            if digest != old["sha256"] or destination.stat().st_size != old["byte_size"]:
                raise CorpusIntegrityError(
                    f"Cached immutable source changed: {destination}. Remove/quarantine it explicitly before retrying."
                )
            record = dict(old)
            record["reused"] = True
        elif destination.exists() and receipt_path.is_file():
            record = _validate_download_record(
                json.loads(receipt_path.read_text()),
                destination,
                allowed_hosts=allowed_hosts,
                expected_requested_url=str(artifact["url"]),
            )
            record["reused"] = True
        else:
            if destination.exists() or old:
                raise CorpusIntegrityError(
                    f"Unreceipted source file/lock disagreement for {destination}; refusing silent replacement"
                )
            source_root.mkdir(parents=True, exist_ok=True)
            url = str(artifact["url"])
            if fetcher is None:
                downloaded = _download_https(
                    url, destination, allowed_hosts=allowed_hosts,
                    expected_sha256=artifact.get("sha256", artifact.get("checksum")),
                    content_kind=str(artifact.get("content_kind", "")),
                )
            else:
                assert_unsealed(url, purpose="fixture fetch")
                downloaded = dict(fetcher(
                    url, destination, allowed_hosts,
                    artifact.get("sha256", artifact.get("checksum")),
                ))
            record = {
                "artifact_id": artifact_id,
                "role": artifact["role"],
                "filename": filename,
                **downloaded,
                "reused": False,
            }
        record = {
            **record,
            "artifact_id": artifact_id,
            "role": artifact["role"],
            "filename": filename,
        }
        record = _validate_download_record(
            record,
            destination,
            allowed_hosts=allowed_hosts,
            expected_requested_url=str(artifact["url"]),
        )
        record["container_validation"] = _validate_artifact_payload(
            destination, str(artifact.get("content_kind", ""))
        )
        # A per-artifact receipt closes the multi-file study restart gap.  It
        # is retained after the study lock as independently auditable evidence.
        receipt_record = dict(record)
        receipt_record.pop("reused", None)
        atomic_json(receipt_path, receipt_record)
        acquired[artifact_id] = destination
        lock_records.append(record)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "study_accession": accession,
        "immutable": True,
        "artifacts": sorted(lock_records, key=lambda value: value["artifact_id"]),
    }
    comparison = json.loads(json.dumps(lock))
    for item in comparison["artifacts"]:
        item.pop("reused", None)
    if existing:
        old_comparison = json.loads(json.dumps(existing))
        for item in old_comparison.get("artifacts", []):
            item.pop("reused", None)
        if comparison != old_comparison:
            raise CorpusIntegrityError(f"Immutable source lock changed for {accession}")
    else:
        atomic_json(lock_path, comparison)
    lock["path"] = str(lock_path.relative_to(root))
    lock["sha256"] = sha256_file(lock_path)
    return acquired, lock


def _parse_study(
    files: Mapping[str, Path], study: Mapping[str, object]
) -> List[Tuple[str, CorpusBlock]]:
    adapter = study["adapter"]
    accession = str(study["accession"])
    if adapter == "bulk_gene_table":
        matrix_artifact = next(
            artifact for artifact in study["artifacts"]
            if artifact["role"] in {
                "matrix", "counts", "rna", "raw_gene_counts", "deposited_gene_counts",
            }
        )
        return [(accession, parse_bulk_gene_table(files[str(matrix_artifact["artifact_id"])], study))]
    if adapter == "dense_sc_barcode_maps":
        return [(f"{accession}_{name}", block) for name, block in parse_dense_sc_barcode_maps(files, study)]
    if adapter == "visium_bundle":
        return [(accession, parse_visium_bundle(files, study))]
    if adapter == "acquisition_plan_only":
        return []
    raise ValueError(f"Unsupported adapter {adapter}")


def _write_acquisition_plan(root: Path, studies: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    staged = []
    for study in studies:
        if study.get("adapter") == "acquisition_plan_only" or not study.get("default_download", True):
            staged.append({
                "accession": study["accession"],
                "assay": study.get("assay"),
                "role": study.get("role"),
                "reason": study.get("staging_reason", "explicit high-volume stage required"),
                "source_assembly": study.get("source_assembly"),
                "measurement_level": study.get("measurement_level"),
                "values_downloaded": False,
            })
    plan = {
        "schema_version": SCHEMA_VERSION,
        "staged_sources": staged,
        "bulk_accessibility_cell_specific": False,
        "requires_explicit_high_volume_stage": True,
    }
    path = root / "staged" / "chromatin_acquisition_plan.json"
    atomic_json(path, plan)
    plan["path"] = str(path.relative_to(root))
    plan["sha256"] = sha256_file(path)
    return plan


def build_from_registry(
    registry_path: Path,
    output_root: Path,
    include_timecourse: bool = True,
    include_spatial: bool = True,
    dry_run: bool = False,
    *,
    fetcher: Optional[Callable[[str, Path, Sequence[str], Optional[str]], Mapping[str, object]]] = None,
) -> Dict[str, object]:
    """Build or resume the v6.1 development corpus."""

    registry = load_registry(Path(registry_path))
    registry_audit = validate_registry(registry)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    registry_lock_path = root / "registry" / "source_registry.lock.json"
    registry_lock = {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": registry_audit["registry_sha256"],
        "sealed_exclusions": [SEALED_ACCESSION],
        "registry": registry,
    }
    if registry_lock_path.exists():
        existing = json.loads(registry_lock_path.read_text())
        if existing != registry_lock:
            raise CorpusIntegrityError("Pinned registry changed under an existing corpus root")
    else:
        atomic_json(registry_lock_path, registry_lock)
    plan = _write_acquisition_plan(root, registry["studies"])
    enabled = [
        study for study in registry["studies"]
        if study["adapter"] != "acquisition_plan_only"
        and _study_enabled(study, include_timecourse, include_spatial)
    ]
    if dry_run:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run_complete",
            "registry_sha256": registry_audit["registry_sha256"],
            "planned_studies": [study["accession"] for study in enabled],
            "acquisition_plan": plan,
            "bundle_manifests": [],
            "claims": {
                "development_assay_values_downloaded": False,
                "gse315993_measurement_values_materialized": False,
                "sealed_values_fetched": False,
                "sealed_study_evaluated": False,
                "model_trained": False,
                "biological_prediction_claim": False,
                "digital_twin_claim": False,
                "attractor_claim": False,
            },
        }
        atomic_json(root / "wld_v61_axolotl_corpus_dry_run_report.json", report)
        return report
    profile = {
        "schema_version": SCHEMA_VERSION,
        "include_timecourse": bool(include_timecourse),
        "include_spatial": bool(include_spatial),
        "enabled_studies": [str(study["accession"]) for study in enabled],
    }
    profile_path = root / "registry" / "build_profile.lock.json"
    if profile_path.exists():
        if json.loads(profile_path.read_text()) != profile:
            raise CorpusIntegrityError(
                "Build profile changed under an existing corpus root; use a separate output root"
            )
    else:
        atomic_json(profile_path, profile)
    source_locks = []
    bundle_manifests = []
    ledger = []
    for study in enabled:
        files, source_lock = _acquire_study(
            root, study, allowed_hosts=registry["allowed_hosts"], fetcher=fetcher
        )
        source_locks.append(source_lock)
        for record in source_lock["artifacts"]:
            ledger.append({
                "study_accession": study["accession"],
                "artifact_id": record["artifact_id"],
                "body_bytes_read": 0 if record.get("reused") else record["byte_size"],
                "sealed_measurement_value_bytes_read": 0,
                "status": "reused" if record.get("reused") else "downloaded",
                "sha256": record["sha256"],
            })
        for bundle_name, block in _parse_study(files, study):
            bundle_manifests.append(_save_block(
                root, bundle_name, block, study=study,
                source_lock=source_lock,
            ))
    ledger_path = root / "access_ledger.json"
    atomic_json(ledger_path, {
        "schema_version": SCHEMA_VERSION,
        "events": ledger,
        "sealed_measurement_value_bytes_read": 0,
    })
    split = _build_split_manifest(root, bundle_manifests)
    feature_registry = _build_feature_registry(root, bundle_manifests, split)
    harmonization = {
        "schema_version": SCHEMA_VERSION,
        "assemblies": sorted({str(manifest["source_assembly"]) for manifest in bundle_manifests}),
        "assemblies_merged": False,
        "symbol_only_mapping_performed": False,
        "ambiguous_mappings_guessed": False,
        "physical_time_and_morphological_stage_separate": True,
        "missing_features_are_unknown_not_observed_zero": True,
    }
    harmonization_path = root / "folds" / "development" / "harmonization_manifest.json"
    atomic_json(harmonization_path, harmonization)
    harmonization_record = {
        "path": str(harmonization_path.relative_to(root)),
        "file_sha256": sha256_file(harmonization_path),
    }
    ledger_record = {
        "path": str(ledger_path.relative_to(root)),
        "file_sha256": sha256_file(ledger_path),
    }
    for value, label in (
        (source_locks, "source-lock lineage"),
        (bundle_manifests, "bundle lineage"),
        (split, "split lineage"),
        (feature_registry, "feature-registry lineage"),
        (ledger, "access-ledger lineage"),
    ):
        assert_unsealed(value, purpose=f"final {label} audit")
    claims = {
        "development_assay_values_downloaded": bool(bundle_manifests),
        "gse315993_measurement_values_materialized": False,
        "sealed_values_fetched": False,
        "sealed_study_evaluated": False,
        "test_measurement_values_materialized": False,
        "model_trained": False,
        "model_checkpoint_written": False,
        "biological_prediction_claim": False,
        "digital_twin_claim": False,
        "attractor_claim": False,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "registry_sha256": registry_audit["registry_sha256"],
        "registry_lock_sha256": sha256_file(registry_lock_path),
        "build_profile": {
            "path": str(profile_path.relative_to(root)),
            "file_sha256": sha256_file(profile_path),
            **profile,
        },
        "enabled_studies": profile["enabled_studies"],
        "source_locks": source_locks,
        "bundle_manifests": bundle_manifests,
        "split_manifest": split,
        "feature_registry": feature_registry,
        "harmonization_manifest": harmonization_record,
        "access_ledger": ledger_record,
        "acquisition_plan": plan,
        "context_policy": {
            "identity_and_state_labels_outside_encoder": True,
            "rna_model_role": "supervision_only",
            "future_state_initialization_forbidden": True,
            "inferred_context_requires_uncertainty": True,
            "bulk_accessibility_never_relabelled_cell_observed": True,
        },
        "claims": claims,
    }
    report_path = root / "wld_v61_axolotl_corpus_report.json"
    atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def verify_corpus(output_root: Path) -> Dict[str, object]:
    root = Path(output_root)
    report_path = root / "wld_v61_axolotl_corpus_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing v6.1 corpus report: {report_path}")
    report = json.loads(report_path.read_text())
    claims = report.get("claims")
    if not isinstance(claims, Mapping):
        raise CorpusIntegrityError("Corpus report is missing structured claims")
    for key in (
        "gse315993_measurement_values_materialized", "sealed_values_fetched",
        "sealed_study_evaluated", "test_measurement_values_materialized",
        "model_trained", "model_checkpoint_written", "biological_prediction_claim",
        "digital_twin_claim", "attractor_claim",
    ):
        if claims.get(key) is not False:
            raise CorpusIntegrityError(f"Forbidden claim/access flag {key} is not false")
    if report.get("status") == "dry_run_complete":
        return {
            "verified": True,
            "dry_run": True,
            "bundles_verified": 0,
            "claims": dict(claims),
        }
    registry_lock_path = root / "registry" / "source_registry.lock.json"
    if not registry_lock_path.is_file() or sha256_file(registry_lock_path) != report.get("registry_lock_sha256"):
        raise CorpusIntegrityError("Registry-lock checksum failed")
    registry_lock = json.loads(registry_lock_path.read_text())
    registry_audit_payload = json.loads(json.dumps(registry_lock))
    registry_audit_payload.pop("sealed_exclusions", None)
    if isinstance(registry_audit_payload.get("registry"), dict):
        registry_audit_payload["registry"].pop("sealed_exclusions", None)
    assert_unsealed(registry_audit_payload, purpose="registry-lock content verification")
    if registry_lock.get("registry_sha256") != report.get("registry_sha256"):
        raise CorpusIntegrityError("Registry-lock lineage mismatch")
    for label, record, hash_key in (
        ("build profile", report.get("build_profile"), "file_sha256"),
        ("harmonization manifest", report.get("harmonization_manifest"), "file_sha256"),
        ("access ledger", report.get("access_ledger"), "file_sha256"),
        ("acquisition plan", report.get("acquisition_plan"), "sha256"),
    ):
        if not isinstance(record, Mapping):
            raise CorpusIntegrityError(f"Missing structured {label} record")
        path = root / str(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != record.get(hash_key):
            raise CorpusIntegrityError(f"{label.capitalize()} checksum failed")
        assert_unsealed(json.loads(path.read_text()), purpose=f"{label} content verification")
    # Measurement directories may never contain the sealed accession, even in
    # filenames or derived manifests.  Registry/report metadata can name the
    # exclusion itself and are deliberately outside this scan.
    for restricted in (root / "raw", root / "bundles", root / "folds"):
        if restricted.exists():
            for path in restricted.rglob("*"):
                assert_unsealed(path.relative_to(root), purpose="corpus verification")
    source_lock_hashes: Dict[str, str] = {}
    for lock in report.get("source_locks", []):
        assert_unsealed(lock, purpose="reported source-lock verification")
        lock_path = root / str(lock["path"])
        if sha256_file(lock_path) != lock["sha256"]:
            raise CorpusIntegrityError(f"Source-lock checksum failed: {lock_path}")
        locked = json.loads(lock_path.read_text())
        assert_unsealed(locked, purpose="source-lock content verification")
        source_root = lock_path.parent
        for artifact in locked["artifacts"]:
            path = source_root / artifact["filename"]
            if not path.is_file() or path.stat().st_size != artifact["byte_size"]:
                raise CorpusIntegrityError(f"Missing/truncated source artifact {path}")
            if sha256_file(path) != artifact["sha256"]:
                raise CorpusIntegrityError(f"Source artifact checksum failed: {path}")
        accession = str(lock.get("study_accession", ""))
        if accession in source_lock_hashes:
            raise CorpusIntegrityError(f"Duplicate source lock for {accession}")
        source_lock_hashes[accession] = str(lock["sha256"])
    if set(source_lock_hashes) != set(map(str, report.get("enabled_studies", []))):
        raise CorpusIntegrityError("Completed source-lock set disagrees with enabled studies")
    verified_groups: Dict[str, str] = {}
    for manifest in report.get("bundle_manifests", []):
        assert_unsealed(manifest, purpose="reported bundle verification")
        manifest_path = root / str(manifest["manifest_path"])
        if sha256_file(manifest_path) != manifest["manifest_sha256"]:
            raise CorpusIntegrityError(f"Bundle manifest checksum failed: {manifest_path}")
        accession = str(manifest.get("study_accession", ""))
        if manifest.get("source_lock_sha256") != source_lock_hashes.get(accession):
            raise CorpusIntegrityError(f"Bundle/source-lock lineage failed for {manifest['bundle_id']}")
        for file_key, hash_key in {
            "matrix": "matrix_file_sha256",
            "features": "feature_sha256",
            "observations": "observation_sha256",
            "pairing": "pairing_sha256",
            "provenance": "provenance_sha256",
        }.items():
            path = root / str(manifest["files"].get(file_key, ""))
            if not path.is_file() or sha256_file(path) != manifest.get(hash_key):
                raise CorpusIntegrityError(
                    f"{file_key} checksum failed for {manifest['bundle_id']}"
                )
        matrix = sparse.load_npz(root / str(manifest["files"]["matrix"])).tocsr()
        if list(matrix.shape) != manifest["shape"] or semantic_sparse_hash(matrix) != manifest["matrix_semantic_sha256"]:
            raise CorpusIntegrityError(f"Sparse matrix integrity failed for {manifest['bundle_id']}")
        rows = _load_observations(root / str(manifest["files"]["observations"]))
        assert_unsealed(rows, purpose="observation provenance verification")
        pairing = json.loads((root / str(manifest["files"]["pairing"])).read_text())
        assert_unsealed(pairing, purpose="pairing provenance verification")
        provenance = json.loads((root / str(manifest["files"]["provenance"])).read_text())
        assert_unsealed(provenance, purpose="bundle provenance verification")
        if provenance.get("study_accession") != accession or provenance.get("raw_unnormalized") is not True:
            raise CorpusIntegrityError(f"Bundle provenance lineage failed for {manifest['bundle_id']}")
        features = _read_features(root / str(manifest["files"]["features"]))
        assert_unsealed(features, purpose="feature lineage verification")
        if matrix.shape != (len(rows), len(features)):
            raise CorpusIntegrityError(f"Bundle matrix/metadata cardinality failed for {manifest['bundle_id']}")
        for row in rows:
            previous = verified_groups.setdefault(row["group_id"], row["partition"])
            if previous != row["partition"]:
                raise CorpusIntegrityError(f"Biological group crosses partitions: {row['group_id']}")
    split_path = root / str(report["split_manifest"]["path"])
    if sha256_file(split_path) != report["split_manifest"]["file_sha256"]:
        raise CorpusIntegrityError("Split-manifest checksum failed")
    split = json.loads(split_path.read_text())
    assert_unsealed(split, purpose="split-manifest content verification")
    if {item["group_id"]: item["partition"] for item in split["groups"]} != verified_groups:
        raise CorpusIntegrityError("Split manifest disagrees with observation groups")
    feature_path = root / str(report["feature_registry"]["path"])
    if sha256_file(feature_path) != report["feature_registry"]["file_sha256"]:
        raise CorpusIntegrityError("Feature-registry checksum failed")
    feature_registry = json.loads(feature_path.read_text())
    assert_unsealed(feature_registry, purpose="feature-registry content verification")
    if feature_registry["fit_groups"] != split["partitions"]["train"]:
        raise CorpusIntegrityError("Feature registry was not fit from exactly the training groups")
    ledger_path = root / str(report["access_ledger"]["path"])
    ledger = json.loads(ledger_path.read_text())
    assert_unsealed(ledger, purpose="access-ledger content verification")
    if int(ledger.get("sealed_measurement_value_bytes_read", -1)) != 0:
        raise CorpusIntegrityError("Access ledger reports sealed measurement bytes")
    if any(int(event.get("sealed_measurement_value_bytes_read", -1)) != 0 for event in ledger.get("events", [])):
        raise CorpusIntegrityError("An access event reports sealed measurement bytes")
    if claims.get("development_assay_values_downloaded") is not bool(report.get("bundle_manifests")):
        raise CorpusIntegrityError("Development-assay claim is inconsistent with parsed bundles")
    return {
        "verified": True,
        "dry_run": False,
        "source_artifacts_verified": sum(len(lock["artifacts"]) for lock in report.get("source_locks", [])),
        "bundles_verified": len(report.get("bundle_manifests", [])),
        "biological_groups_verified": len(verified_groups),
        "train_feature_namespaces": sorted(feature_registry.get("registries", {})),
        "claims": dict(claims),
    }


__all__ = [
    "CorpusBlock", "CorpusIntegrityError", "SCHEMA_VERSION", "SEALED_ACCESSION",
    "SealedSourceRefused", "assert_unsealed", "atomic_json", "audit_encoder_feature_names",
    "build_from_registry",
    "load_registry", "parse_bulk_gene_table", "parse_dense_sc_barcode_maps",
    "parse_visium_bundle", "semantic_sparse_hash", "sha256_file", "sha256_json",
    "validate_context_record", "validate_initial_provenance", "validate_registry", "verify_corpus",
]
