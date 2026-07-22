"""Metadata-only source audit for the WLD v6 axolotl virtual-tissue prototype.

This module validates a small, human-readable registry.  It never downloads a
count matrix, image, fragment file, or other measurement value.  Optional live
checks issue bounded HTTP GETs only to official accession metadata pages.
GSE315993 is an external validation series: its public metadata may be checked,
but its matrices, images, coordinates, and processed values remain sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Union
from urllib.parse import unquote, urljoin, urlparse


DEFAULT_REGISTRY = Path(__file__).with_name("wld_v60_axolotl_sources.json")
DEFAULT_REPORT = Path("wld_v60_axolotl_source_audit.json")

REQUIRED_CANONICAL_FIELDS = (
    "study", "biosample", "donor", "species", "assembly", "tissue",
    "anatomy", "stage", "condition", "perturbation", "dose", "time",
    "assay", "modality", "pairing", "spatial_relation", "source_url",
    "checksum", "license",
)
REQUIRED_AXOLOTL_STUDIES = {
    "GSE106269", "GSE121737", "PRJNA589484", "PRJNA682840",
    "GSE243225", "GSE315993",
}
REQUIRED_ATLAS_SOURCES = {
    "GSE149683", "10.17632/yv4fzv6cnm.1", "GSE158013", "GSE194122",
}
OFFICIAL_HOST_ALLOWLIST = {
    "www.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "data.mendeley.com", "jaspar.elixir.no", "omnipathdb.org",
    "www.omnipathdb.org", "mips.helmholtz-muenchen.de",
    "data.4dnucleome.org",
}
PAIRING_MODES = {
    "same_cell_exact", "same_spot_exact",
    "lineage_experiment_no_per_cell_crosswalk", "unpaired_population",
    "single_modality_not_applicable",
}
EXACT_PAIRING_EVIDENCE = {
    "same_cell_exact": "deposited_barcode_identity",
    "same_spot_exact": "deposited_spot_barcode_to_coordinate",
}
PAIRING_VERIFICATION_STATUSES = {
    "verified_from_materialized_schema", "metadata_declared_unverified",
    "not_applicable",
}
NULL_SENTINELS = {"unknown", "n/a", "na", "not available", "not reported"}
VALUE_FILE_RE = re.compile(
    r"(?:/suppl(?:/|$)|/(?:download|downloads)(?:/|$)|"
    r"(?:^|[/=])(?:fragments?|count[_-]?matrix)(?=$|[./?&#=/])|"
    r"\.(?:mtx|h5|h5ad|loom|fastq|fq|fasta|fa|bam|cram|sam|bed|"
    r"bigwig|bw|csv|tsv|txt|rds|rda|rdata|png|jpe?g|tiff?|gif|"
    r"webp|bmp|svg|tar|tgz|zip|7z|rar|gz|bgz|bz2|xz|zst)"
    r"(?=$|[?&#=/]))",
    re.IGNORECASE,
)
METADATA_CONTENT_TYPES = {
    "text/html", "text/xml", "application/json", "application/xml",
    "application/xhtml+xml",
}
MEASUREMENT_TEXT_CONTENT_TYPES = {
    "text/plain", "text/csv", "text/tab-separated-values",
    "application/csv", "application/tsv",
}
REQUIRED_MECHANISTIC_PRIOR_TYPES = {
    "sequence_motif_binding", "curated_signed_tf_regulation",
    "signaling_and_protein_interactions", "protein_complex_membership",
    "contact_and_hic_reference",
}


class MetadataFetchRefused(RuntimeError):
    """A response refused before measurement-value body materialization."""

    def __init__(self, message: str, **audit: object) -> None:
        super().__init__(message)
        self.audit = dict(audit)


class _MetadataRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every redirect target before urllib follows its Location."""

    def __init__(self, allowed_hosts: Sequence[str]) -> None:
        super().__init__()
        self.allowed_hosts = tuple(allowed_hosts)
        self.redirect_chain = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        effective_url = urljoin(req.full_url, str(newurl))
        hop = {
            "from_url": req.full_url,
            "to_url": effective_url,
            "status": int(code),
        }
        self.redirect_chain.append(hop)
        try:
            _allowed_url(effective_url, self.allowed_hosts)
        except ValueError as error:
            raise MetadataFetchRefused(
                f"Redirect Location is not an allowlisted metadata URL: {error}",
                requested_url=req.full_url,
                effective_url=effective_url,
                redirect_chain=list(self.redirect_chain),
                refusal_reason="redirect_location_host_or_scheme_refused",
                headers_checked_before_body=True,
                body_bytes_read=0,
                measurement_value_bytes_read=0,
            ) from error
        if _is_value_url(effective_url):
            raise MetadataFetchRefused(
                "Redirect Location points to a measurement-value resource",
                requested_url=req.full_url,
                effective_url=effective_url,
                redirect_chain=list(self.redirect_chain),
                refusal_reason="redirect_location_classified_as_measurement",
                headers_checked_before_body=True,
                body_bytes_read=0,
                measurement_value_bytes_read=0,
            )
        return super().redirect_request(req, fp, code, msg, headers, effective_url)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_registry(
    path: Union[str, Path] = DEFAULT_REGISTRY,
) -> Dict[str, object]:
    """Load a registry without contacting any remote host."""

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Axolotl source registry must be a JSON object")
    return payload


def _allowed_url(url: str, allowed_hosts: Sequence[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"Only absolute HTTPS metadata URLs are allowed: {url!r}")
    if parsed.hostname.lower() not in {value.lower() for value in allowed_hosts}:
        raise ValueError(f"Metadata URL host is not allowlisted: {parsed.hostname}")


def _is_value_url(url: str) -> bool:
    parsed = urlparse(url)
    # Decode the query as well as the path so ``?filename=counts%2Etsv`` is
    # refused before urllib follows a redirect.  Metadata endpoints ending in
    # HTML, XML, or JSON remain admissible.
    path = unquote(parsed.path).lower()
    query = unquote(parsed.query).lower()
    resource = path + (f"?{query}" if query else "")
    return bool(
        VALUE_FILE_RE.search(resource)
        or "download=1" in query
        or re.search(
            r"(?:^|&)(?:format|filetype|type)="
            r"(?:fastq|fq|fasta|fa|bam|cram|sam|bed|mtx|h5|h5ad|loom|"
            r"csv|tsv|txt|rds|rda|rdata|png|jpe?g|tiff?|gif|webp|bmp|svg|"
            r"tar|tgz|zip|7z|rar|gz|bgz|bz2|xz|zst)(?:&|$)",
            query,
        )
    )


def _metadata_content_type(content_type: str) -> bool:
    token = content_type.split(";", 1)[0].strip().lower()
    if token in MEASUREMENT_TEXT_CONTENT_TYPES:
        return False
    return (
        token in METADATA_CONTENT_TYPES
        or token.endswith("+json")
        or token.endswith("+xml")
    )


def _attachment_header(content_disposition: str) -> bool:
    token = content_disposition.strip().lower()
    return bool(token and ("attachment" in token or "filename=" in token))


def _validate_nulls(value: object, path: str) -> None:
    if isinstance(value, str) and value.strip().lower() in NULL_SENTINELS:
        raise ValueError(f"Unknown value must be JSON null, not {value!r}, at {path}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_nulls(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_nulls(item, f"{path}.{key}")


def _validate_pairing(pairing: object, record_id: str) -> None:
    if not isinstance(pairing, dict):
        raise ValueError(f"{record_id}: pairing must be a structured object")
    required = {
        "mode", "evidence_type", "evidence", "identifier_fields",
        "crosswalk_fields", "verification_status", "schema_materialized", "fabricated",
        "expression_similarity_used", "cell_label_matching_used",
    }
    missing = required.difference(pairing)
    if missing:
        raise ValueError(f"{record_id}: pairing missing {sorted(missing)}")
    mode = pairing["mode"]
    if mode not in PAIRING_MODES:
        raise ValueError(f"{record_id}: unsupported pairing mode {mode!r}")
    for flag in ("fabricated", "expression_similarity_used", "cell_label_matching_used"):
        if pairing[flag] is not False:
            raise ValueError(f"{record_id}: pairing flag {flag} must be false")
    identifier_fields = pairing["identifier_fields"]
    crosswalk_fields = pairing["crosswalk_fields"]
    if not isinstance(identifier_fields, list) or not all(
        isinstance(value, str) and value for value in identifier_fields
    ):
        raise ValueError(f"{record_id}: identifier_fields must be a string list")
    if not isinstance(crosswalk_fields, list) or not all(
        isinstance(value, str) and value for value in crosswalk_fields
    ):
        raise ValueError(f"{record_id}: crosswalk_fields must be a string list")
    verification = pairing["verification_status"]
    if verification not in PAIRING_VERIFICATION_STATUSES:
        raise ValueError(f"{record_id}: unsupported verification_status {verification!r}")
    if not isinstance(pairing["schema_materialized"], bool):
        raise ValueError(f"{record_id}: schema_materialized must be boolean")
    if mode in EXACT_PAIRING_EVIDENCE:
        expected = EXACT_PAIRING_EVIDENCE[mode]
        if pairing["evidence_type"] != expected or not pairing["evidence"]:
            raise ValueError(
                f"{record_id}: {mode} requires {expected} and nonempty evidence"
            )
        if not identifier_fields or not crosswalk_fields:
            raise ValueError(
                f"{record_id}: exact pairing requires identifier and crosswalk fields"
            )
        if verification == "verified_from_materialized_schema":
            if pairing["schema_materialized"] is not True:
                raise ValueError(
                    f"{record_id}: verified exact pairing requires a materialized schema"
                )
        elif verification == "metadata_declared_unverified":
            if pairing["schema_materialized"] is not False:
                raise ValueError(
                    f"{record_id}: metadata-declared pairing must remain unmaterialized"
                )
        else:
            raise ValueError(f"{record_id}: exact pairing cannot be not_applicable")
    elif mode == "lineage_experiment_no_per_cell_crosswalk":
        if pairing["evidence_type"] != "deposited_study_design":
            raise ValueError(f"{record_id}: lineage experiment needs study-design evidence")
        if identifier_fields or crosswalk_fields:
            raise ValueError(
                f"{record_id}: lineage experiment cannot imply a per-cell crosswalk"
            )
        if verification != "metadata_declared_unverified" or pairing["schema_materialized"]:
            raise ValueError(f"{record_id}: lineage relation must remain declared/unverified")
    else:
        if pairing["evidence_type"] is not None:
            raise ValueError(f"{record_id}: {mode} must not invent pairing evidence")
        if identifier_fields or crosswalk_fields:
            raise ValueError(f"{record_id}: {mode} cannot contain pairing fields")
        if verification != "not_applicable" or pairing["schema_materialized"]:
            raise ValueError(f"{record_id}: {mode} must be unmaterialized/not_applicable")


def _validate_record(
    record: object,
    allowed_hosts: Sequence[str],
    record_ids: set,
    *,
    expected_study: Optional[str] = None,
) -> None:
    if not isinstance(record, dict):
        raise ValueError("Every registry record must be an object")
    missing = set(REQUIRED_CANONICAL_FIELDS).difference(record)
    if missing:
        raise ValueError(f"Record missing canonical fields: {sorted(missing)}")
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Every registry record requires a nonempty record_id")
    if record_id in record_ids:
        raise ValueError(f"Duplicate record_id: {record_id}")
    record_ids.add(record_id)
    if expected_study is not None and record["study"] != expected_study:
        raise ValueError(f"{record_id}: study differs from parent study")
    if not isinstance(record["species"], str) or not record["species"]:
        raise ValueError(f"{record_id}: species must be present")
    _allowed_url(str(record["source_url"]), allowed_hosts)
    if _is_value_url(str(record["source_url"])):
        raise ValueError(f"{record_id}: source_url points to measurement values")
    checksum = record["checksum"]
    if checksum is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", str(checksum)):
        raise ValueError(f"{record_id}: checksum must be null or sha256:<64 hex>")
    _validate_pairing(record["pairing"], record_id)
    _validate_nulls(record, record_id)


def validate_registry(
    registry: Mapping[str, object],
    registry_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Validate schema, provenance, pairing, and sealed-data invariants."""

    if registry.get("schema_version") != "6.0.0":
        raise ValueError("Expected WLD axolotl registry schema_version 6.0.0")
    if tuple(registry.get("canonical_fields", ())) != REQUIRED_CANONICAL_FIELDS:
        raise ValueError("Registry canonical_fields differ from the v6 contract")
    policy = registry.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Registry policy is missing")
    required_policy = {
        "metadata_only": True,
        "large_count_downloads_allowed": False,
        "unknown_values_must_be_null": True,
        "fabricated_pairing_forbidden": True,
        "cell_identity_labels_are_encoder_inputs": False,
        "gse315993_measurement_values_sealed": True,
        "gse315993_metadata_audit_allowed": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) is not expected:
            raise ValueError(f"Registry policy {key} must be {expected}")

    allowed_hosts = registry.get("official_host_allowlist")
    if not isinstance(allowed_hosts, list) or not allowed_hosts:
        raise ValueError("official_host_allowlist must be a nonempty list")
    if not set(allowed_hosts).issubset(OFFICIAL_HOST_ALLOWLIST):
        raise ValueError("Registry attempts to allow an unapproved metadata host")

    studies = registry.get("studies")
    if not isinstance(studies, list):
        raise ValueError("Registry studies must be a list")
    studies_by_id: Dict[str, Mapping[str, object]] = {}
    record_ids: set = set()
    record_count = 0
    verified_exact_pairing_records = 0
    metadata_declared_exact_pairing_records = 0
    metadata_declared_nonexact_records = 0
    unpaired_records = 0
    for study in studies:
        if not isinstance(study, dict) or not isinstance(study.get("study"), str):
            raise ValueError("Each study requires an accession")
        accession = str(study["study"])
        if accession in studies_by_id:
            raise ValueError(f"Duplicate study: {accession}")
        studies_by_id[accession] = study
        metadata_url = str(study.get("metadata_url", ""))
        _allowed_url(metadata_url, allowed_hosts)
        if _is_value_url(metadata_url):
            raise ValueError(f"{accession}: metadata_url points to measurement values")
        if study.get("measurement_value_urls") != []:
            raise ValueError(f"{accession}: v6 registry must not contain value URLs")
        records = study.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{accession}: records must be nonempty")
        for record in records:
            _validate_record(record, allowed_hosts, record_ids, expected_study=accession)
            record_count += 1
            mode = record["pairing"]["mode"]
            verification = record["pairing"]["verification_status"]
            verified_exact_pairing_records += int(
                mode in EXACT_PAIRING_EVIDENCE
                and verification == "verified_from_materialized_schema"
            )
            metadata_declared_exact_pairing_records += int(
                mode in EXACT_PAIRING_EVIDENCE
                and verification == "metadata_declared_unverified"
            )
            metadata_declared_nonexact_records += int(
                mode not in EXACT_PAIRING_EVIDENCE
                and verification == "metadata_declared_unverified"
            )
            unpaired_records += int(mode == "unpaired_population")

    missing_studies = REQUIRED_AXOLOTL_STUDIES.difference(studies_by_id)
    if missing_studies:
        raise ValueError(f"Missing required axolotl sources: {sorted(missing_studies)}")
    for accession, study in studies_by_id.items():
        for record in study["records"]:
            if record["species"] != "Ambystoma mexicanum":
                raise ValueError(f"{accession}: axolotl study has wrong organism")

    bulk = studies_by_id["PRJNA682840"]
    if bulk.get("bulk_atac_cell_composition_limitation") is not True:
        raise ValueError("PRJNA682840 must disclose bulk cell-composition confounding")
    if not any("cell-composition" in text.lower() for text in bulk.get("limitations", [])):
        raise ValueError("PRJNA682840 lacks explicit cell-composition limitation text")

    for accession in ("GSE243225", "GSE315993"):
        if studies_by_id[accession].get("spatial_biological_replication_sufficient") is not False:
            raise ValueError(f"{accession} must disclose limited spatial replication")
    sealed = studies_by_id["GSE315993"]
    if sealed.get("split") != "sealed_external_test":
        raise ValueError("GSE315993 must remain the sealed external test")
    if sealed.get("measurement_values_sealed") is not True or sealed.get("metadata_audit_only") is not True:
        raise ValueError("GSE315993 seal is incomplete")
    sealed_pairing = sealed["records"][0]["pairing"]
    if not (
        sealed_pairing["mode"] == "same_spot_exact"
        and sealed_pairing["verification_status"] == "metadata_declared_unverified"
        and sealed_pairing["schema_materialized"] is False
    ):
        raise ValueError(
            "GSE315993 spot pairing must remain planned, unverified and unmaterialized"
        )
    lineage_pairing = studies_by_id["GSE106269"]["records"][0]["pairing"]
    if not (
        lineage_pairing["mode"] == "lineage_experiment_no_per_cell_crosswalk"
        and lineage_pairing["identifier_fields"] == []
        and lineage_pairing["crosswalk_fields"] == []
    ):
        raise ValueError("GSE106269 cannot imply exact per-cell lineage pairing")

    atlas_sources = registry.get("reference_atlas_sources")
    if not isinstance(atlas_sources, list):
        raise ValueError("reference_atlas_sources must be a list")
    atlas_ids = set()
    for record in atlas_sources:
        _validate_record(record, allowed_hosts, record_ids)
        if record["study"] == "GSE315993" or "GSE315993" in json.dumps(record):
            raise ValueError("GSE315993 cannot reappear in reference_atlas_sources")
        if record.get("candidate_only") is not True or record.get("download_in_v60") is not False:
            raise ValueError(f"{record.get('record_id')}: atlas source must remain candidate-only")
        atlas_study = record["study"]
        if atlas_study in atlas_ids:
            raise ValueError(f"Duplicate reference-atlas study: {atlas_study}")
        atlas_ids.add(atlas_study)
        mode = record["pairing"]["mode"]
        verification = record["pairing"]["verification_status"]
        verified_exact_pairing_records += int(
            mode in EXACT_PAIRING_EVIDENCE
            and verification == "verified_from_materialized_schema"
        )
        metadata_declared_exact_pairing_records += int(
            mode in EXACT_PAIRING_EVIDENCE
            and verification == "metadata_declared_unverified"
        )
    if REQUIRED_ATLAS_SOURCES.difference(atlas_ids):
        raise ValueError(
            f"Missing atlas sources: {sorted(REQUIRED_ATLAS_SOURCES.difference(atlas_ids))}"
        )
    role_overlap = set(studies_by_id).intersection(atlas_ids)
    if role_overlap:
        raise ValueError(f"Study accessions cannot span source roles: {sorted(role_overlap)}")
    for accession in studies_by_id:
        if any(accession in json.dumps(record) for record in atlas_sources):
            raise ValueError(
                f"Axolotl study accession {accession} reappears in reference_atlas_sources"
            )

    mechanistic_sources = registry.get("mechanistic_prior_sources")
    if not isinstance(mechanistic_sources, list) or not mechanistic_sources:
        raise ValueError("mechanistic_prior_sources must be a nonempty list")
    prior_types = set()
    mechanistic_ids = set()
    for source in mechanistic_sources:
        if not isinstance(source, dict):
            raise ValueError("Each mechanistic prior source must be an object")
        required = {
            "source_id", "prior_type", "transfer_status", "axolotl_observation",
            "metadata_only", "download_in_v60", "source_url", "checksum",
            "license", "limitations",
        }
        missing = required.difference(source)
        if missing:
            raise ValueError(f"Mechanistic prior source missing {sorted(missing)}")
        source_id = source["source_id"]
        if source_id in mechanistic_ids:
            raise ValueError(f"Duplicate mechanistic source_id: {source_id}")
        mechanistic_ids.add(source_id)
        if source["prior_type"] not in REQUIRED_MECHANISTIC_PRIOR_TYPES:
            raise ValueError(f"Unsupported mechanistic prior type: {source['prior_type']}")
        prior_types.add(source["prior_type"])
        if source["transfer_status"] != "reference_transferred_candidate_support_only":
            raise ValueError(f"{source_id}: mechanistic source must remain candidate support")
        if source["axolotl_observation"] is not False:
            raise ValueError(f"{source_id}: reference prior cannot be an axolotl observation")
        if source["metadata_only"] is not True or source["download_in_v60"] is not False:
            raise ValueError(f"{source_id}: v6 must not download mechanistic prior data")
        _allowed_url(str(source["source_url"]), allowed_hosts)
        if _is_value_url(str(source["source_url"])):
            raise ValueError(f"{source_id}: source_url points to measurement values")
        if "GSE315993" in json.dumps(source):
            raise ValueError("GSE315993 cannot reappear in mechanistic_prior_sources")
        _validate_nulls(source, f"mechanistic_prior_sources.{source_id}")
    if prior_types != REQUIRED_MECHANISTIC_PRIOR_TYPES:
        raise ValueError("Mechanistic prior coverage is incomplete")
    role_overlap = set(studies_by_id).intersection(mechanistic_ids)
    if role_overlap:
        raise ValueError(f"Study accessions cannot span source roles: {sorted(role_overlap)}")
    for accession in studies_by_id:
        if any(accession in json.dumps(source) for source in mechanistic_sources):
            raise ValueError(
                f"Axolotl study accession {accession} reappears in mechanistic_prior_sources"
            )

    scrna_record = studies_by_id["PRJNA589484"]["records"][0]
    expected_scrna_times = [
        {"value": 0, "unit": "hour", "relation": "homeostatic"},
        {"value": 3, "unit": "hour", "relation": "post_amputation"},
        {"value": 1, "unit": "day", "relation": "post_amputation"},
        {"value": 3, "unit": "day", "relation": "post_amputation"},
        {"value": 7, "unit": "day", "relation": "post_amputation"},
        {"value": 14, "unit": "day", "relation": "post_amputation"},
        {"value": 22, "unit": "day", "relation": "post_amputation"},
        {"value": 33, "unit": "day", "relation": "post_amputation"},
    ]
    if scrna_record["time"] != expected_scrna_times:
        raise ValueError("PRJNA589484 must retain the primary eight-stage design")
    if scrna_record.get("cell_count") != 41376:
        raise ValueError("PRJNA589484 must retain the reported 41,376-cell total")
    if scrna_record.get("tissue") != "upper arm":
        raise ValueError("PRJNA589484 must retain the source-described upper-arm tissue")
    if verified_exact_pairing_records:
        raise ValueError(
            "Metadata-only v6 cannot report exact pairing as schema-verified"
        )

    return {
        "valid": True,
        "registry_path": str(Path(registry_path).resolve()) if registry_path else None,
        "study_count": len(studies),
        "record_count": record_count,
        "reference_atlas_source_count": len(atlas_sources),
        "mechanistic_prior_source_count": len(mechanistic_sources),
        "mechanistic_prior_types": sorted(prior_types),
        "studies": sorted(studies_by_id),
        "verified_exact_pairing_records": verified_exact_pairing_records,
        "metadata_declared_exact_pairing_records": metadata_declared_exact_pairing_records,
        "metadata_declared_nonexact_records": metadata_declared_nonexact_records,
        "exact_deposited_pairing_records": verified_exact_pairing_records,
        "unpaired_population_records": unpaired_records,
        "bulk_atac_cell_composition_limitation_present": True,
        "spatial_replication_limits_present": True,
        "sealed_external_test_study": "GSE315993",
    }


def _fetch_metadata_url(
    url: str,
    *,
    timeout: float,
    retries: int,
    max_bytes: int,
    allowed_hosts: Sequence[str] = tuple(OFFICIAL_HOST_ALLOWLIST),
) -> Dict[str, object]:
    """Fetch one bounded metadata page with GET; never follow to a data host.

    A test fetcher may return the same dictionary without ``text``.  Required
    keys are url, status, content_type, bytes_read, and sha256.
    """

    _allowed_url(url, allowed_hosts)
    if _is_value_url(url):
        raise MetadataFetchRefused(
            "Requested URL points to a measurement-value resource",
            requested_url=url,
            effective_url=url,
            redirect_chain=[],
            refusal_reason="requested_url_classified_as_measurement",
            headers_checked_before_body=True,
            body_bytes_read=0,
            measurement_value_bytes_read=0,
        )
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        redirect_handler = _MetadataRedirectHandler(allowed_hosts)
        opener = urllib.request.build_opener(redirect_handler)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "WLD-v6-metadata-audit/1.0",
                "Accept": (
                    "text/html,application/xhtml+xml,application/json,"
                    "application/xml,text/xml"
                ),
                "Range": f"bytes=0-{max_bytes - 1}",
            },
            method="GET",
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                final_url = response.geturl()
                _allowed_url(final_url, allowed_hosts)
                content_type = response.headers.get_content_type()
                content_disposition = response.headers.get("Content-Disposition", "")
                content_length_raw = response.headers.get("Content-Length")
                common_audit = {
                    "requested_url": url,
                    "effective_url": final_url,
                    "content_type": content_type,
                    "content_disposition": content_disposition,
                    "headers_checked_before_body": True,
                    "body_bytes_read": 0,
                    "measurement_value_bytes_read": 0,
                    "redirect_chain": list(redirect_handler.redirect_chain),
                }
                # Redirects are resolved when urlopen returns.  Reclassify the
                # effective URL and response headers before reading one byte.
                if _is_value_url(final_url):
                    raise MetadataFetchRefused(
                        "Effective URL points to a measurement-value resource",
                        refusal_reason="effective_url_classified_as_measurement",
                        **common_audit,
                    )
                if not _metadata_content_type(content_type):
                    raise MetadataFetchRefused(
                        f"Refusing non-metadata content type: {content_type}",
                        refusal_reason="non_metadata_content_type",
                        **common_audit,
                    )
                if _attachment_header(content_disposition):
                    raise MetadataFetchRefused(
                        f"Refusing attachment response: {content_disposition}",
                        refusal_reason="attachment_content_disposition",
                        **common_audit,
                    )
                if content_length_raw:
                    try:
                        content_length = int(content_length_raw)
                    except ValueError:
                        content_length = None
                    if content_length is not None and content_length > max_bytes:
                        raise MetadataFetchRefused(
                            f"Metadata Content-Length exceeded {max_bytes} bytes",
                            refusal_reason="content_length_exceeds_cap",
                            **common_audit,
                        )
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise MetadataFetchRefused(
                        f"Metadata response exceeded {max_bytes} bytes",
                        refusal_reason="streamed_metadata_exceeds_cap",
                        requested_url=url,
                        effective_url=final_url,
                        content_type=content_type,
                        content_disposition=content_disposition,
                        headers_checked_before_body=True,
                        body_bytes_read=len(payload),
                        measurement_value_bytes_read=0,
                        redirect_chain=list(redirect_handler.redirect_chain),
                    )
                return {
                    "url": final_url,
                    "status": int(getattr(response, "status", 200)),
                    "content_type": content_type,
                    "content_disposition": content_disposition,
                    "bytes_read": len(payload),
                    "headers_checked_before_body": True,
                    "measurement_value_bytes_read": 0,
                    "redirect_chain": list(redirect_handler.redirect_chain),
                    "sha256": _sha256_bytes(payload),
                    "text": payload.decode("utf-8", errors="replace"),
                }
        except MetadataFetchRefused:
            raise
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"Metadata GET failed after {retries + 1} attempts: {url}: {last_error}")


def audit_sources(
    registry_path: Union[str, Path] = DEFAULT_REGISTRY,
    output_path: Union[str, Path] = DEFAULT_REPORT,
    *,
    live: bool = False,
    strict_live: bool = False,
    timeout: float = 10.0,
    retries: int = 2,
    max_metadata_bytes: int = 512 * 1024,
    fetcher: Optional[Callable[..., Mapping[str, object]]] = None,
) -> Dict[str, object]:
    """Validate the registry and optionally GET only bounded metadata pages."""

    if timeout <= 0 or retries < 0 or max_metadata_bytes < 1024:
        raise ValueError("timeout/retries/max_metadata_bytes are invalid")
    registry_file = Path(registry_path)
    registry = load_registry(registry_file)
    validation = validate_registry(registry, registry_file)
    allowed_hosts = tuple(registry["official_host_allowlist"])
    studies = registry["studies"]
    metadata_urls = [str(study["metadata_url"]) for study in studies]
    sealed_metadata_url = next(
        str(study["metadata_url"]) for study in studies if study["study"] == "GSE315993"
    )

    live_report: Dict[str, object] = {
        "enabled": bool(live),
        "http_method": "GET" if live else None,
        "scope": "axolotl_accession_metadata_only",
        "reference_atlas_sources_live_checked": False,
        "mechanistic_prior_sources_live_checked": False,
        "requested_urls": [],
        "fetched_urls": [],
        "failed_urls": [],
        "checks": [],
        "access_ledger": [],
        "sealed_metadata_urls_fetched": [],
        "sealed_measurement_urls_fetched": [],
        "sealed_measurement_urls_refused": [],
        "bytes_fetched": 0,
        "measurement_value_bytes_read": 0,
        "max_bytes_per_response": max_metadata_bytes,
    }
    if live:
        active_fetcher = fetcher or _fetch_metadata_url
        for study, url in zip(studies, metadata_urls):
            live_report["requested_urls"].append(url)
            sealed_request = study["study"] == "GSE315993"
            ledger_entry: Dict[str, object] = {
                "study": study["study"],
                "split": study["split"],
                "requested_url": url,
                "request_purpose": "accession_metadata_audit",
                "sealed_external_test": sealed_request,
                "effective_url": None,
                "headers_checked_before_body": False,
                "body_bytes_read": 0,
                "measurement_value_bytes_read": 0,
                "outcome": "pending",
                "refusal_reason": None,
            }
            try:
                fetched = dict(active_fetcher(
                    url,
                    timeout=timeout,
                    retries=retries,
                    max_bytes=max_metadata_bytes,
                    allowed_hosts=allowed_hosts,
                ))
                for key in ("url", "status", "content_type", "bytes_read", "sha256"):
                    if key not in fetched:
                        raise RuntimeError(f"Fetcher result missing {key}")
                final_url = str(fetched["url"])
                _allowed_url(final_url, allowed_hosts)
                bytes_read = int(fetched["bytes_read"])
                if bytes_read < 0 or bytes_read > max_metadata_bytes:
                    raise RuntimeError("Fetcher violated metadata byte cap")
                content_type = str(fetched["content_type"])
                content_disposition = str(fetched.get("content_disposition", ""))
                ledger_entry.update({
                    "effective_url": final_url,
                    "content_type": content_type,
                    "content_disposition": content_disposition,
                    "headers_checked_before_body": bool(
                        fetched.get("headers_checked_before_body", False)
                    ),
                    "body_bytes_read": bytes_read,
                })
                # The built-in fetcher performs these checks before reading.
                # Recheck injected fetchers and conservatively classify any
                # already-read unsafe response bytes as measurement values.
                if _is_value_url(final_url):
                    raise MetadataFetchRefused(
                        "Fetcher returned a measurement-value URL",
                        effective_url=final_url,
                        refusal_reason="effective_url_classified_as_measurement",
                        body_bytes_read=bytes_read,
                        measurement_value_bytes_read=bytes_read,
                    )
                if not _metadata_content_type(content_type):
                    raise MetadataFetchRefused(
                        "Fetcher returned a non-metadata content type",
                        effective_url=final_url,
                        refusal_reason="non_metadata_content_type",
                        body_bytes_read=bytes_read,
                        measurement_value_bytes_read=bytes_read,
                    )
                if _attachment_header(content_disposition):
                    raise MetadataFetchRefused(
                        "Fetcher returned an attachment response",
                        effective_url=final_url,
                        refusal_reason="attachment_content_disposition",
                        body_bytes_read=bytes_read,
                        measurement_value_bytes_read=bytes_read,
                    )
                live_report["fetched_urls"].append(final_url)
                if url == sealed_metadata_url:
                    live_report["sealed_metadata_urls_fetched"].append(final_url)
                text = fetched.get("text")
                content_check = {
                    "study": study["study"],
                    "url": final_url,
                    "accession_present": None,
                    "organism_present": None,
                    "content_checked": isinstance(text, str),
                }
                if isinstance(text, str):
                    folded = text.casefold()
                    content_check["accession_present"] = str(study["study"]).casefold() in folded
                    content_check["organism_present"] = str(study["expected_organism"]).casefold() in folded
                    if strict_live and not (
                        content_check["accession_present"] and content_check["organism_present"]
                    ):
                        raise RuntimeError("Metadata page omitted expected accession or organism")
                elif strict_live:
                    raise RuntimeError(
                        "Strict live audit requires response text for accession/organism checks"
                    )
                live_report["checks"].append(content_check)
                ledger_entry["outcome"] = "metadata_fetched"
            except MetadataFetchRefused as error:
                ledger_entry.update(error.audit)
                ledger_entry["outcome"] = "refused"
                ledger_entry["error"] = str(error)
                live_report["failed_urls"].append({"url": url, "error": str(error)})
            except Exception as error:  # report remote failure without weakening static seal
                ledger_entry["outcome"] = "failed"
                ledger_entry["error"] = str(error)
                live_report["failed_urls"].append({"url": url, "error": str(error)})
            finally:
                live_report["access_ledger"].append(ledger_entry)
        if strict_live and live_report["failed_urls"]:
            raise RuntimeError(f"Strict live metadata audit failed: {live_report['failed_urls']}")

    requested = set(live_report["requested_urls"])
    if requested.difference(metadata_urls):
        raise RuntimeError("Audit requested a URL outside accession metadata pages")
    ledger = live_report["access_ledger"]
    live_report["bytes_fetched"] = sum(
        int(entry.get("body_bytes_read", 0)) for entry in ledger
    )
    live_report["measurement_value_bytes_read"] = sum(
        int(entry.get("measurement_value_bytes_read", 0)) for entry in ledger
    )
    live_report["sealed_measurement_urls_fetched"] = [
        entry.get("effective_url")
        for entry in ledger
        if entry.get("sealed_external_test")
        and int(entry.get("measurement_value_bytes_read", 0)) > 0
    ]
    live_report["sealed_measurement_urls_refused"] = [
        entry.get("effective_url")
        for entry in ledger
        if entry.get("sealed_external_test")
        and entry.get("outcome") == "refused"
        and entry.get("effective_url")
        and _is_value_url(str(entry.get("effective_url")))
    ]

    measurement_value_bytes = int(live_report["measurement_value_bytes_read"])
    sealed_value_bytes = sum(
        int(entry.get("measurement_value_bytes_read", 0))
        for entry in ledger
        if entry.get("sealed_external_test")
    )
    metadata_only_observed = measurement_value_bytes == 0
    sealed_values_materialized = sealed_value_bytes > 0

    report: Dict[str, object] = {
        "report_version": "6.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_file.resolve()),
        "registry_sha256": _sha256_file(registry_file),
        "validation": validation,
        "live_audit": live_report,
        "claims": {
            "metadata_only": metadata_only_observed,
            "large_count_matrices_downloaded": measurement_value_bytes > 0,
            "raw_or_processed_measurement_values_downloaded": measurement_value_bytes > 0,
            "sealed_external_measurement_urls_downloaded": sealed_values_materialized,
            "gse315993_measurement_values_materialized": sealed_values_materialized,
            "test_measurement_values_materialized": sealed_values_materialized,
            "reference_atlas_sources_live_checked": False,
            "mechanistic_prior_sources_live_checked": False,
            "fresh_model_training": False,
            "digital_twin_claim": False,
            "attractor_claim": False,
        },
        "limitations": {
            "bulk_atac_is_cell_composition_confounded": True,
            "gse243225_has_single_spatial_specimen": True,
            "gse315993_has_one_deposited_sample_per_stage": True,
            "metadata_validation_is_not_measurement_validation": True,
        },
    }
    _atomic_json(Path(output_path), report)
    if measurement_value_bytes:
        raise RuntimeError(
            "Unsafe fetcher materialized measurement-value bytes; see durable access ledger"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--live", action="store_true", help="GET only bounded accession metadata pages")
    parser.add_argument("--strict-live", action="store_true", help="fail if a live metadata check is unavailable or inconsistent")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-metadata-bytes", type=int, default=512 * 1024)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_sources(
        args.registry,
        args.output,
        live=args.live,
        strict_live=args.strict_live,
        timeout=args.timeout,
        retries=args.retries,
        max_metadata_bytes=args.max_metadata_bytes,
    )
    print("PASS: WLD v6 axolotl metadata-only source audit")
    print(f"Studies: {report['validation']['studies']}")
    print(f"GSE315993 values materialized: {report['claims']['gse315993_measurement_values_materialized']}")
    print(f"Report: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
