"""Run the WLD v6.0 synthetic and metadata-only virtual-tissue contract.

This entry point does not download assay matrices or train a biological model.
It validates the graph-constrained architecture, audits the declared public
source metadata, and keeps the external spatial study GSE315993 sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

from wld_axolotl_data_v60 import audit_sources, load_registry, validate_registry


REPORT_SCHEMA = "wld-v6.0-virtual-tissue-software-metadata-validation"
REPORT_NAME = "wld_v60_virtual_tissue_validation.json"
AUDIT_NAME = "wld_v60_axolotl_source_audit.json"
SEALED_ACCESSION = "GSE315993"

REQUIRED_TRUE_CLAIMS = ("metadata_only",)
REQUIRED_FALSE_CLAIMS = (
    "large_count_matrices_downloaded",
    "sealed_external_measurement_urls_downloaded",
    "gse315993_measurement_values_materialized",
    "test_measurement_values_materialized",
    "fresh_model_training",
    "digital_twin_claim",
    "attractor_claim",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def nested_mappings(value: object) -> Iterable[Mapping[str, object]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from nested_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_mappings(child)


def validate_sealed_registry_entry(registry: Mapping[str, object]) -> None:
    matches = []
    for entry in nested_mappings(registry):
        accession = str(
            entry.get(
                "accession",
                entry.get("study_id", entry.get("study", entry.get("id", ""))),
            )
        ).strip().upper()
        if accession == SEALED_ACCESSION and ("role" in entry or "split" in entry):
            matches.append(entry)
    if len(matches) != 1:
        raise RuntimeError(
            f"Registry must contain exactly one {SEALED_ACCESSION} entry; found {len(matches)}"
        )
    entry = matches[0]
    role = str(entry.get("role", entry.get("split", ""))).strip().lower()
    if role != "sealed_external_test":
        raise RuntimeError(
            f"{SEALED_ACCESSION} is not locked to sealed_external_test"
        )
    # Accept different schema spellings but require an explicit false for every
    # present measurement-access switch. Absence is left to validate_registry,
    # which owns the canonical source schema.
    measurement_flags = (
        "measurement_values_allowed",
        "download_measurements",
        "values_allowed",
        "materialize_values",
    )
    for name in measurement_flags:
        if name in entry and entry[name] is not False:
            raise RuntimeError(f"{SEALED_ACCESSION} permits sealed measurement access via {name}")
    records = entry.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError(f"{SEALED_ACCESSION} must have one sealed registry record")
    pairing = records[0].get("pairing")
    if not isinstance(pairing, Mapping):
        raise RuntimeError(f"{SEALED_ACCESSION} has no structured pairing record")
    if (
        pairing.get("mode") != "same_spot_exact"
        or pairing.get("verification_status") != "metadata_declared_unverified"
        or pairing.get("schema_materialized") is not False
    ):
        raise RuntimeError(
            f"{SEALED_ACCESSION} must remain a declared but unverified spatial relation"
        )


def validate_registry_report(validation: Mapping[str, object]) -> None:
    """Refuse inflated exact-pairing claims in the metadata-only stage."""

    if validation.get("verified_exact_pairing_records") != 0:
        raise RuntimeError("Metadata-only v6 cannot report schema-verified exact pairing")
    if validation.get("exact_deposited_pairing_records") != 0:
        raise RuntimeError("Legacy exact-pairing count must include verified records only")
    declared = validation.get("metadata_declared_exact_pairing_records")
    if not isinstance(declared, int) or declared < 1:
        raise RuntimeError("Registry did not retain declared/unverified pairing separately")


def validate_audit_claims(audit: Mapping[str, object]) -> Mapping[str, object]:
    claims = audit.get("claims")
    if not isinstance(claims, Mapping):
        raise RuntimeError("The source audit has no claims mapping")
    wrong_true = [name for name in REQUIRED_TRUE_CLAIMS if claims.get(name) is not True]
    wrong_false = [
        name for name in REQUIRED_FALSE_CLAIMS if claims.get(name) is not False
    ]
    if wrong_true or wrong_false:
        raise RuntimeError(
            "The source audit crossed its metadata-only claim boundary: "
            + ", ".join(wrong_true + wrong_false)
        )
    live_audit = audit.get("live_audit")
    if not isinstance(live_audit, Mapping):
        raise RuntimeError("The source audit has no access ledger")
    if live_audit.get("measurement_value_bytes_read") != 0:
        raise RuntimeError("The metadata audit materialized measurement-value bytes")
    ledger = live_audit.get("access_ledger")
    if not isinstance(ledger, list):
        raise RuntimeError("The source audit access ledger is malformed")
    for entry in ledger:
        if not isinstance(entry, Mapping):
            raise RuntimeError("The source audit access ledger contains a non-record")
        if int(entry.get("measurement_value_bytes_read", 0)) != 0:
            raise RuntimeError("An access-ledger entry materialized measurement values")
    return claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--strict-live", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-metadata-bytes", type=int, default=524288)
    args = parser.parse_args()

    sibling = Path(__file__).resolve().parent
    smoke_path = sibling / "run_wld_v60_virtual_tissue_smoke.py"
    core_path = sibling / "wld_regulatory_twin_v60.py"
    source_path = sibling / "wld_axolotl_data_v60.py"
    contract_path = sibling / "wld_v60_virtual_tissue_contract.md"
    for path, label in (
        (smoke_path, "v6.0 synthetic contract"),
        (core_path, "v6.0 regulatory-twin architecture"),
        (source_path, "v6.0 source-audit implementation"),
        (contract_path, "v6.0 written contract"),
        (args.registry, "v6.0 source registry"),
    ):
        require_file(path, label)

    print("WLD V6.0 ATLAS-CONDITIONED VIRTUAL-TISSUE CONTRACT", flush=True)
    print(f"Python {platform.python_version()}", flush=True)
    print("Synthetic architecture checks and accession metadata only.", flush=True)
    print("No assay matrices, biological training, digital-twin claim or attractor claim.\n", flush=True)

    print("1. Validating the frozen source registry and GSE315993 seal...", flush=True)
    registry = load_registry(args.registry)
    if not isinstance(registry, Mapping):
        raise RuntimeError("load_registry did not return a mapping")
    validation = validate_registry(registry, registry_path=args.registry)
    if not isinstance(validation, Mapping):
        raise RuntimeError("validate_registry did not return a structured report")
    validate_sealed_registry_entry(registry)
    validate_registry_report(validation)
    print("PASS: source roles, schema and sealed external spatial test", flush=True)

    print("\n2. Running the synthetic mechanistic and leakage contract...", flush=True)
    subprocess.run([sys.executable, str(smoke_path)], check=True)

    print("\n3. Running the metadata-only public-source audit...", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_root / AUDIT_NAME
    audit = audit_sources(
        registry_path=args.registry,
        output_path=audit_path,
        live=args.live,
        strict_live=args.strict_live,
        timeout=args.timeout,
        retries=args.retries,
        max_metadata_bytes=args.max_metadata_bytes,
    )
    if not isinstance(audit, Mapping):
        raise RuntimeError("audit_sources did not return a structured report")
    claims = validate_audit_claims(audit)

    implementation_paths = (
        core_path,
        source_path,
        smoke_path,
        Path(__file__).resolve(),
        args.registry,
        contract_path,
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "scope": "synthetic software contract and accession metadata audit only",
        "registry_validation": validation,
        "source_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "live_requested": bool(args.live),
            "strict_live": bool(args.strict_live),
        },
        "provenance": {
            "sealed_accession": SEALED_ACCESSION,
            "implementation_sha256": {
                path.name: sha256_file(path) for path in implementation_paths
            },
        },
        "claims": {
            **dict(claims),
            "software_contract_validated": True,
            "assay_values_downloaded": False,
            "sealed_values_fetched": False,
            "model_trained": False,
            "biological_prediction_claim": False,
            "sealed_study_evaluated": False,
            "model_checkpoint_written": False,
        },
    }
    report_path = args.output_root / REPORT_NAME
    atomic_json(report_path, report)

    print("\n" + "=" * 78, flush=True)
    print("VERIFIED COMPLETE: WLD V6.0 SOFTWARE + METADATA CONTRACT", flush=True)
    print("=" * 78, flush=True)
    print("Synthetic graph and leakage contract: True", flush=True)
    print(f"Metadata audit requested live:        {bool(args.live)}", flush=True)
    print("Assay values downloaded:              False", flush=True)
    print(f"{SEALED_ACCESSION} values fetched:             False", flush=True)
    print("Biological model trained:             False", flush=True)
    print("Biological prediction claim:          False", flush=True)
    print("Digital-twin claim:                   False", flush=True)
    print("Attractor claim:                      False", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"Source audit: {audit_path}", flush=True)


if __name__ == "__main__":
    main()
