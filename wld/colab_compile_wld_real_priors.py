# Run this in the SAME Colab runtime that contains gse240061_export.
# It does not reinstall NumPy/SciPy/PyTorch and does not rerun the 3.5 GB export.

import gzip
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/content/wld_real_data")
EXPORT = ROOT / "gse240061_export"
PRIORS = ROOT / "gse240061_priors"
SOURCES = ROOT / "prior_sources"
SOURCES.mkdir(parents=True, exist_ok=True)

required_export = [
    "rna.mtx.gz", "atac.mtx.gz", "genes.tsv", "peaks.tsv",
    "barcodes.tsv", "metadata.tsv", "split.json",
]
missing = [name for name in required_export if not (EXPORT / name).exists()]
if missing:
    raise FileNotFoundError(
        f"Missing export files {missing}. Run the successful GSE240061 export "
        "cell again in this runtime before this cell."
    )


def run(command, *, env=None, check=True, stdout=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    return subprocess.run(
        list(map(str, command)), env=env, check=check, stdout=stdout
    )


def download(url, path, min_bytes):
    path = pathlib.Path(path)
    if path.exists() and path.stat().st_size >= min_bytes:
        print(f"Already present: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    command = [
        "curl", "-fL", "--retry", "8", "--retry-delay", "3",
        "--connect-timeout", "30", "--user-agent", "WLD-prior-compiler/1.0",
        "--continue-at", "-", "--output", path, url,
    ]
    result = run(command, check=False)
    if result.returncode != 0:
        print("Resume was rejected; restarting only this source file.")
        path.unlink(missing_ok=True)
        run([item for item in command if item not in ("--continue-at", "-")])
    if not path.exists() or path.stat().st_size < min_bytes:
        raise RuntimeError(f"Download is unexpectedly small: {path}")
    return path


print("1. Installing isolated command-line motif tools...")
TOOL_PREFIX = pathlib.Path("/content/wld_motif_tools")
MAMBA_ROOT = pathlib.Path("/content/wld_micromamba_root")
MAMBA_ARCHIVE = pathlib.Path("/content/micromamba-linux-64.tar.bz2")
MAMBA = pathlib.Path("/content/bin/micromamba")

if not MAMBA.exists():
    download(
        "https://micro.mamba.pm/api/micromamba/linux-64/latest",
        MAMBA_ARCHIVE,
        1_000_000,
    )
    run(["tar", "-xjf", MAMBA_ARCHIVE, "-C", "/content", "bin/micromamba"])

tool_env = os.environ.copy()
tool_env["MAMBA_ROOT_PREFIX"] = str(MAMBA_ROOT)
if not (TOOL_PREFIX / "bin/fimo").exists() or not (TOOL_PREFIX / "bin/bedtools").exists():
    action = "install" if TOOL_PREFIX.exists() else "create"
    run(
        [
            MAMBA, action, "-y", "-p", TOOL_PREFIX,
            "-c", "conda-forge", "-c", "bioconda",
            "meme=5.5.7", "bedtools",
        ],
        env=tool_env,
    )
tool_env["PATH"] = f"{TOOL_PREFIX / 'bin'}:{tool_env['PATH']}"
run(["fimo", "--version"], env=tool_env)
run(["bedtools", "--version"], env=tool_env)


print("\n2. Downloading/resuming frozen biological sources...")
PCHIC = ROOT / "GSE126100_interactions.csv.gz"
if not PCHIC.exists():
    download(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126100/suppl/GSE126100_interactions.csv.gz",
        PCHIC,
        1_000,
    )

GTF_GZ = download(
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.primary_assembly.annotation.gtf.gz",
    SOURCES / "gencode.v44.primary_assembly.annotation.gtf.gz",
    30_000_000,
)
FASTA_GZ = download(
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz",
    SOURCES / "GRCh38.primary_assembly.genome.fa.gz",
    700_000_000,
)
JASPAR = download(
    "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
    SOURCES / "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
    400_000,
)
COLLECTRI = download(
    "https://omnipathdb.org/interactions?genesymbols=1&fields=sources,references,curation_effort&datasets=collectri&organisms=9606&license=academic",
    SOURCES / "omnipath_collectri_human.tsv",
    1_000_000,
)
OMNIPATH = download(
    "https://omnipathdb.org/interactions?genesymbols=1&fields=sources,references,curation_effort&datasets=omnipath&organisms=9606&license=academic",
    SOURCES / "omnipath_core_human.tsv",
    1_000_000,
)

for path in (PCHIC, GTF_GZ, FASTA_GZ):
    with gzip.open(path, "rb") as handle:
        handle.read(1)
print("PASS: gzip sources are readable")


print("\n3. Expanding the frozen GRCh38 reference (about 3.1 GB)...")
FASTA = SOURCES / "GRCh38.primary_assembly.genome.fa"
if not FASTA.exists() or FASTA.stat().st_size < 2_500_000_000:
    temporary = FASTA.with_suffix(".fa.tmp")
    with temporary.open("wb") as output:
        run(["pigz", "-dc", FASTA_GZ], stdout=output)
    temporary.replace(FASTA)
print(f"Reference FASTA: {FASTA.stat().st_size / 1e9:.2f} GB")


print("\n4. Downloading the pinned, tested WLD prior compiler...")
COMMIT = "16a2656857e0e5003d9ea31b382b65cf03efec31"
COMPILER_SHA256 = "878fc75dcfdbd29fda4a37d2679069e018ced4808c4571601ea76696ed6bc0af"
COMPILER = SOURCES / "compile_wld_muscle_priors.py"
COMPILER_URL = (
    f"https://raw.githubusercontent.com/nlipieta/MultiscaleProject/{COMMIT}/"
    "wld/compile_wld_muscle_priors.py"
)
download(COMPILER_URL, COMPILER, 30_000)
actual_hash = hashlib.sha256(COMPILER.read_bytes()).hexdigest()
if actual_hash != COMPILER_SHA256:
    raise RuntimeError(
        f"Compiler hash mismatch: expected {COMPILER_SHA256}, got {actual_hash}"
    )
print("PASS: compiler hash verified")


print("\n5. Compiling contact x motif x signed-regulation x signaling priors...")
print("This streams the 814 MB ATAC matrix and then runs FIMO; allow several minutes.")
LOG = ROOT / "wld_prior_compilation.log"
command = [
    sys.executable, COMPILER,
    "--export", EXPORT,
    "--pchic", PCHIC,
    "--gencode-gtf", GTF_GZ,
    "--collectri", COLLECTRI,
    "--omnipath", OMNIPATH,
    "--jaspar-meme", JASPAR,
    "--genome-fasta", FASTA,
    "--output", PRIORS,
    "--max-candidate-peaks", "5000",
    "--expected-max-tfs", "64",
    "--max-signal-depth", "4",
    "--fimo-p-threshold", "1e-4",
    "--overwrite",
]
process_env = tool_env.copy()
process_env["PYTHONUNBUFFERED"] = "1"
with LOG.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(
        list(map(str, command)),
        env=process_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
    return_code = process.wait()
if return_code:
    raise RuntimeError(
        f"Prior compilation failed with exit code {return_code}. Full log: {LOG}"
    )


print("\n6. Recording exact tool versions and checking outputs...")
manifest_path = PRIORS / "prior_manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["compiler"] = {"git_commit": COMMIT, "sha256": COMPILER_SHA256}
manifest["software_tools"] = {
    "fimo": subprocess.check_output(
        [TOOL_PREFIX / "bin/fimo", "--version"], text=True
    ).strip(),
    "bedtools": subprocess.check_output(
        [TOOL_PREFIX / "bin/bedtools", "--version"], text=True
    ).strip(),
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

for name in (
    "peak_gene_links.tsv", "motif_hits.tsv",
    "tf_gene_edges.tsv", "signaling_edges.tsv", "prior_manifest.json",
):
    path = PRIORS / name
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty prior output: {path}")
    print(f"  {name:28s} {path.stat().st_size / 1e6:8.2f} MB")

print("\nPASS: real WLD biological scaffold compiled.")
print(json.dumps(manifest["output_counts"], indent=2))
print("Leakage audit:", json.dumps(manifest["leakage_contract"], indent=2))
print("Full manifest:", manifest_path)
print("Full log:", LOG)
