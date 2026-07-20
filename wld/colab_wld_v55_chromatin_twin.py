# WLD v5.5 — pinned, restart-safe mechanistic chromatin-twin development cell
from google.colab import drive

drive.mount("/content/drive")

import datetime as dt
import hashlib
import json
import os
import py_compile
import shutil
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "nlipieta/MultiscaleProject"
SOURCE_REF = "d3367efbae31d9d460fb9ec7c98acb46c10e4fa2"
BACKUP = Path("/content/drive/MyDrive/WLD_Backup")
CODE = Path("/content/wld_v55_code")
PACKAGES = Path("/content/wld_v55_packages_py312_np1264")
PHASE_B = BACKUP / "wld_phase_b"
CORPUS = BACKUP / "wld_corpus_pretraining"
V53_BUNDLE = BACKUP / "wld_v53_crispr_sciatac_ingestion" / "bundle"
PRIOR_SOURCES = BACKUP / "wld_real_data" / "prior_sources"
OUTPUT = BACKUP / "wld_v55_chromatin_twin_r3"
SOURCE_DIR = OUTPUT / "frozen_sources"
CORUM = SOURCE_DIR / "corum_human_complexes_v5_3.txt"
CORUM_MANIFEST = SOURCE_DIR / "corum_human_complexes_v5_3.json"
LAUNCHER_MANIFEST = OUTPUT / "launcher_provenance.json"
LOG = OUTPUT / "wld_v55_complete.log"
REPORT = OUTPUT / "development" / "wld_v55_chromatin_twin_report.json"

CORUM_RELEASE = "5.3"
CORUM_EXPECTED_BYTES = 6_268_940
CORUM_SHA256 = "5e40556eb59bb767f3396ba8c080f64c777903ffca9102345fdc35662a2e0282"
CORUM_RELEASE_URL = (
    "https://mips.helmholtz-muenchen.de/fastapi-corum/public/releases/current"
)
CORUM_DOWNLOAD_URL = (
    "https://mips.helmholtz-muenchen.de/fastapi-corum/"
    "public/file/download_current_file?file_id=human&file_format=txt"
)

FILES = {
    "wld_circuit_dynamics_v3.py": "2ffcd9d0a60551dd06db2646c60747ba0680e47150fd5f91bf42b7d8eadfe068",
    "wld_foundation_model_v4.py": "0999e2f5de11883dfb05e18d2cddc272b4501aba16d43cef1600afaa27cf7071",
    "wld_foundation_data.py": "446d52ea61f882ba6aaeeec275077213a3d94dc4b705d5d8427081111134720a",
    "wld_phase_b_priors.py": "d3b216b1d11c7ec3f767126787abc5df9abc2cc22f6b6c12d1bc2bc6566d58ce",
    "wld_chromatin_response_v54.py": "7743c61e415dbe3fc9bb941448d99c07a1d6ec3469235c585964ef58a7340579",
    "wld_chromatin_training_v54.py": "24b0a314d38730045745fc6361f1912581e7730b50c88d97acf1ac5c615cdfa4",
    "wld_chromatin_twin_v55.py": "c9ec8a16c1355dd59f02fcf8492dff3bf4ed089575b4413d2d4568dbb4c16e4e",
    "wld_chromatin_modules_v55.py": "eecda349e2bba4a03071fb9018cff87ff6c480c3a68e6b621b0c242512f3f91a",
    "wld_twin_statistics_v55.py": "b0bc34f52d77bbe396b8f0111907321415daadf3b2dab293c8902069a360f25e",
    "wld_chromatin_twin_training_v55.py": "a3315be4afe6a325474ac6842af0524e48169761324b25a401e5355d3fb7e18a",
    "run_wld_v55_twin_smoke.py": "81c696116160bdd662f3659d334bb1a0db2b16acf3e2dc3d58745e5eab864351",
    "run_wld_v55_twin_colab.py": "b6c9678b6016960cefe38a98dfaab46ab9dd413754c164ab76e682e3f34add29",
    "wld_v55_digital_twin_contract.md": "89d4943e731044b45fb56c82ab8be88512d5c88067852cc06e706908f38dcbb6",
}

RUN_CONFIG = {
    "epochs": 28,
    "targets_per_epoch": 28,
    "batch_size": 48,
    "patience": 6,
    "shuffle_replicates": 2,
    "bootstrap_replicates": 100,
    "seeds": [42, 137, 911],
    "device": "cuda",
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fetch_bytes(
    url,
    *,
    attempts=6,
    timeout=180,
    user_agent="WLD-v5.5-Colab/1.0",
    ssl_context=None,
):
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Cache-Control": "no-cache",
                    "Accept": "*/*",
                },
            )
            open_kwargs = {"timeout": timeout}
            if ssl_context is not None:
                open_kwargs["context"] = ssl_context
            with urllib.request.urlopen(request, **open_kwargs) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                delay = min(2 ** attempt, 20)
                print(f"   retry {attempt + 1}/{attempts - 1} after {error}")
                time.sleep(delay)
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}") from last_error


def is_certificate_verification_failure(error):
    """Recognize the wrapped urllib/SSL failure emitted by Colab."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        if "CERTIFICATE_VERIFY_FAILED" in str(current):
            return True
        pending.extend(
            [
                getattr(current, "reason", None),
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
            ]
        )
    return False


def fetch_corum_bytes(url, *, attempts=6, timeout=180):
    """Fetch CORUM with an exact-content-locked fallback for its broken TLS chain.

    The fallback is deliberately restricted to the hard-coded official table URL.
    It is never used for mutable release metadata. Transport authentication may be
    relaxed only after Colab reports a certificate-verification failure; the
    caller must still validate the table's frozen byte count and SHA-256 before
    writing it.
    """
    if url != CORUM_DOWNLOAD_URL:
        raise ValueError(f"Unapproved URL for CORUM TLS fallback: {url}")
    try:
        return fetch_bytes(url, attempts=2, timeout=timeout), "verified_tls"
    except RuntimeError as error:
        if not is_certificate_verification_failure(error):
            raise

    print(
        "   WARNING: CORUM's server did not provide a certificate chain that "
        "Colab can verify. Retrying only this pinned CORUM URL; the downloaded "
        "biological table must still match its frozen byte count and SHA-256."
    )
    unverified_context = ssl.create_default_context()
    unverified_context.check_hostname = False
    unverified_context.verify_mode = ssl.CERT_NONE
    return (
        fetch_bytes(
            url,
            attempts=attempts,
            timeout=timeout,
            ssl_context=unverified_context,
        ),
        "exact_content_lock_after_corum_tls_chain_failure",
    )


def download_verified_source(name, expected_hash):
    destination = CODE / name
    if destination.is_file() and sha256_file(destination) == expected_hash:
        print(f"   PASS cached: {name}")
        return
    url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_REF}/wld/"
        f"{urllib.parse.quote(name)}"
    )
    data = fetch_bytes(url, timeout=180)
    observed = sha256_bytes(data)
    if observed != expected_hash:
        raise RuntimeError(
            f"Pinned source hash mismatch for {name}: {observed} != {expected_hash}"
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    print(f"   PASS downloaded: {name}")


def ensure_corum_v53():
    if CORUM.is_file():
        observed = sha256_file(CORUM)
        if observed != CORUM_SHA256 or CORUM.stat().st_size != CORUM_EXPECTED_BYTES:
            raise RuntimeError(
                "The frozen CORUM file exists but fails its v5.3 content lock; "
                f"move it aside and rerun: {CORUM}"
            )
        if not CORUM_MANIFEST.is_file():
            atomic_json(
                CORUM_MANIFEST,
                {
                    "schema_version": "wld-v5.5-frozen-corum-source",
                    "release": CORUM_RELEASE,
                    "download_url": CORUM_DOWNLOAD_URL,
                    "manifest_recovered_utc": dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat(),
                    "bytes": CORUM.stat().st_size,
                    "sha256": observed,
                },
            )
        manifest = json.loads(CORUM_MANIFEST.read_text())
        if (
            manifest.get("release") != CORUM_RELEASE
            or manifest.get("sha256") != CORUM_SHA256
            or int(manifest.get("bytes", -1)) != CORUM_EXPECTED_BYTES
        ):
            raise RuntimeError("Frozen CORUM manifest does not match release 5.3")
        print("   PASS cached: CORUM human complexes release 5.3")
        return

    try:
        release_bytes = fetch_bytes(CORUM_RELEASE_URL, attempts=2, timeout=90)
    except RuntimeError as error:
        if not is_certificate_verification_failure(error):
            raise
        release = None
        release_transport = "unavailable_after_certificate_verification_failure"
        print(
            "   WARNING: verified CORUM release metadata is unavailable because "
            "of the server's certificate chain. It will not be trusted through "
            "the fallback; the exact table content lock remains mandatory."
        )
    else:
        release = json.loads(release_bytes.decode("utf-8"))
        release_transport = "verified_tls"
        if str(release.get("version")) != CORUM_RELEASE:
            raise RuntimeError(
                "CORUM's current release is no longer 5.3. The launcher refuses "
                "to silently substitute a different biological prior."
            )
    data, download_transport = fetch_corum_bytes(
        CORUM_DOWNLOAD_URL,
        attempts=8,
        timeout=300,
    )
    observed = sha256_bytes(data)
    if observed != CORUM_SHA256 or len(data) != CORUM_EXPECTED_BYTES:
        raise RuntimeError(
            "Official CORUM release 5.3 bytes changed: "
            f"sha256={observed}, bytes={len(data)}"
        )
    temporary = CORUM.with_suffix(CORUM.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(CORUM)
    atomic_json(
        CORUM_MANIFEST,
        {
            "schema_version": "wld-v5.5-frozen-corum-source",
            "release": CORUM_RELEASE,
            "release_metadata": release,
            "release_metadata_transport": release_transport,
            "download_url": CORUM_DOWNLOAD_URL,
            "download_transport": download_transport,
            "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "bytes": len(data),
            "sha256": observed,
        },
    )
    print("   PASS downloaded and froze CORUM human complexes release 5.3")


def environment_probe(child_env):
    command = [
        sys.executable,
        "-c",
        (
            "import h5py,numpy,scipy,torch; "
            "assert numpy.__version__=='1.26.4'; "
            "assert scipy.__version__=='1.16.3'; "
            "assert h5py.__version__=='3.16.0'; "
            "assert torch.cuda.is_available(), "
            "'Select Runtime > Change runtime type > T4 GPU'; "
            "print('NumPy',numpy.__version__,'| SciPy',scipy.__version__,"
            "'| h5py',h5py.__version__,'| PyTorch',torch.__version__); "
            "print('GPU:',torch.cuda.get_device_name(0))"
        ),
    ]
    return subprocess.run(command, env=child_env, text=True, capture_output=True)


CODE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

print("WLD V5.5 PINNED MECHANISTIC CHROMATIN-TWIN LAUNCHER")
print(f"Repository source commit: {SOURCE_REF}")
print("Test targets, muscle J/L, and external test studies remain sealed.\n")

print("1. Downloading and SHA-verifying the exact v5.5 implementation...")
for filename, expected in FILES.items():
    download_verified_source(filename, expected)

print("\n2. Freezing the curated protein-complex source...")
ensure_corum_v53()

print("\n3. Checking the isolated numerical environment...")
child_env = os.environ.copy()
child_env["PYTHONPATH"] = str(CODE) + os.pathsep + str(PACKAGES)
child_env["PYTHONNOUSERSITE"] = "1"
child_env["PYTHONUNBUFFERED"] = "1"
child_env["MPLBACKEND"] = "Agg"
child_env["OMP_NUM_THREADS"] = "2"
child_env["MKL_NUM_THREADS"] = "2"

probe = environment_probe(child_env)
if probe.returncode:
    # This directory is ephemeral and uniquely owned by this launcher.
    if PACKAGES.exists():
        shutil.rmtree(PACKAGES)
    PACKAGES.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "--target",
            str(PACKAGES),
            "--no-cache-dir",
            "numpy==1.26.4",
            "scipy==1.16.3",
            "h5py==3.16.0",
        ],
        check=True,
    )
    probe = environment_probe(child_env)
if probe.returncode:
    print(probe.stdout)
    print(probe.stderr)
    raise RuntimeError(
        "Could not create the isolated WLD environment. Confirm that this is a "
        "fresh Colab Python 3.12 runtime with a T4 GPU, then rerun this cell."
    )
print(probe.stdout.strip())
print("   PASS: compatible isolated packages and CUDA")

print("\n4. Compiling the pinned implementation...")
for filename in FILES:
    if filename.endswith(".py"):
        py_compile.compile(str(CODE / filename), doraise=True)
print("   PASS: Python compilation")

required = {
    "Phase B prior manifest": PHASE_B / "priors" / "homo_sapiens_grch38" / "prior_manifest.json",
    "Phase B numeric priors": PHASE_B / "priors" / "homo_sapiens_grch38" / "foundation_priors.npz",
    "Phase B feature vocabulary": PHASE_B / "priors" / "homo_sapiens_grch38" / "feature_vocab.json",
    "expanded-corpus checkpoint": CORPUS / "wld_corpus_pretrained_model.pt",
    "expanded-corpus report": CORPUS / "wld_corpus_pretraining_report.json",
    "v5.3 ingestion manifest": V53_BUNDLE / "wld_v53_ingestion_manifest.json",
    "v5.3 whole-target split": V53_BUNDLE / "whole_target_split.json",
    "v5.3 full response matrix": V53_BUNDLE / "atac_counts.GRCh38.2kb.npz",
    "v5.3 cell metadata": V53_BUNDLE / "cells.tsv.gz",
    "v5.3 response bins": V53_BUNDLE / "bins.GRCh38.2kb.tsv.gz",
}
missing = [
    f"{label}: {path}"
    for label, path in required.items()
    if not path.is_file() or path.stat().st_size == 0
]
if missing:
    raise FileNotFoundError(
        "Missing durable upstream WLD artifacts. Restore the Drive backup first:\n"
        + "\n".join(missing)
    )
interaction_candidates = [
    PRIOR_SOURCES / "omnipath_core_human.tsv",
    *sorted(PRIOR_SOURCES.glob("omnipath_core_human.tsv.*")),
    *sorted(PRIOR_SOURCES.glob("omnipath_webservice_interactions*.tsv.xz")),
]
if not any(path.is_file() and path.stat().st_size for path in interaction_candidates):
    raise FileNotFoundError(
        f"No frozen OmniPath core interaction source under {PRIOR_SOURCES}"
    )
print("   PASS: Phase B, corpus, v5.3, OmniPath, and CORUM inputs found")

launcher_lock = {
    "schema_version": "wld-v5.5-colab-launcher-lock",
    "repository": REPOSITORY,
    "source_ref": SOURCE_REF,
    "source_sha256": FILES,
    "corum_release": CORUM_RELEASE,
    "corum_sha256": CORUM_SHA256,
    "run_config": RUN_CONFIG,
    "claims": {
        "test_targets_evaluated": False,
        "muscle_j_l_evaluated": False,
        "external_test_studies_evaluated": False,
        "digital_twin_claim": False,
        "attractor_claim": False,
    },
}
if LAUNCHER_MANIFEST.is_file():
    existing_lock = json.loads(LAUNCHER_MANIFEST.read_text())
    if existing_lock != launcher_lock:
        raise RuntimeError(
            "The existing v5.5 output directory belongs to a different locked "
            "launcher. Preserve it and use a new output directory."
        )
else:
    atomic_json(LAUNCHER_MANIFEST, launcher_lock)
print("   PASS: immutable launcher provenance lock")

command = [
    sys.executable,
    "-u",
    str(CODE / "run_wld_v55_twin_colab.py"),
    "--phase-b-root",
    str(PHASE_B),
    "--corpus-root",
    str(CORPUS),
    "--v53-bundle",
    str(V53_BUNDLE),
    "--prior-sources",
    str(PRIOR_SOURCES),
    "--corum-file",
    str(CORUM),
    "--output-root",
    str(OUTPUT),
    "--epochs",
    str(RUN_CONFIG["epochs"]),
    "--targets-per-epoch",
    str(RUN_CONFIG["targets_per_epoch"]),
    "--batch-size",
    str(RUN_CONFIG["batch_size"]),
    "--patience",
    str(RUN_CONFIG["patience"]),
    "--shuffle-replicates",
    str(RUN_CONFIG["shuffle_replicates"]),
    "--bootstrap-replicates",
    str(RUN_CONFIG["bootstrap_replicates"]),
    "--seeds",
    ",".join(map(str, RUN_CONFIG["seeds"])),
    "--device",
    RUN_CONFIG["device"],
]

print("\n5. Starting restart-safe v5.5 development...")
print("   Fits: 3 seeds x (true dual routes + 2 degree-shuffled controls) = 9")
print("   The first run can take several hours and may span Colab reconnects.")
print("   Rerun this exact cell after a disconnect; completed work is retained.")
print("   This is development on unseen validation targets, not an attractor test.\n")

with LOG.open("a", encoding="utf-8") as log_handle:
    boundary = (
        "\n" + "=" * 78 + "\n"
        + f"LAUNCH {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
        + "=" * 78 + "\n"
    )
    log_handle.write(boundary)
    log_handle.flush()
    process = subprocess.Popen(
        command,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
        log_handle.flush()
    return_code = process.wait()

if return_code:
    print("\n" + "=" * 78)
    print("FINAL 180 CHILD-PROCESS LOG LINES")
    print("=" * 78)
    print("\n".join(LOG.read_text(errors="replace").splitlines()[-180:]))
    raise RuntimeError(
        f"WLD v5.5 exited with code {return_code}. "
        f"The complete log is at {LOG}. Rerun this same cell to resume."
    )

if not REPORT.is_file() or REPORT.stat().st_size == 0:
    raise RuntimeError(f"v5.5 exited without its final report: {REPORT}")
result = json.loads(REPORT.read_text())
claims = result.get("claim_evaluation", {})
sealed = result.get("claims", {})
if (
    claims.get("digital_twin_claim") is not False
    or claims.get("attractor_claim") is not False
    or sealed.get("test_targets_evaluated") is not False
):
    raise RuntimeError("The completed report crossed a declared scientific boundary")

print("\n" + "=" * 78)
print("VERIFIED COMPLETE: WLD V5.5 MECHANISTIC CHROMATIN DEVELOPMENT")
print("=" * 78)
print(f"Report: {REPORT}")
print(f"Log:    {LOG}")
print(f"Fitted path reliance:           {claims.get('fitted_path_reliance')}")
print(f"Topology specificity:           {claims.get('topology_specificity')}")
print(f"Useful perturbation prediction: {claims.get('useful_perturbation_prediction')}")
print("Test targets evaluated:         False")
print("Muscle J/L evaluated:           False")
print("Digital-twin claim:             False")
print("Attractor claim:                False")
