# WLD v5.3 — CRISPR-sciATAC ingestion and training-only accessibility atlas
# Paste this entire file into one Colab cell. It is restart-safe.
#
# First execution downloads ~1.1 GB of processed ATAC BED data, lifts hg19 to
# GRCh38, aligns cell barcodes to the exact 105-target guide panel, creates
# whole-target splits, and selects 2-kb accessibility bins using TRAINING
# TARGETS/CELLS ONLY. It does not train a model or evaluate sealed targets.

from google.colab import drive
drive.mount("/content/drive")

import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


BACKUP = Path("/content/drive/MyDrive/WLD_Backup")
PANEL_ROOT = BACKUP / "wld_v52_crispr_sciatac_panel"
PANEL_REPORT = PANEL_ROOT / "wld_v52_sequence_repair.json"
PANEL_ATLAS = PANEL_ROOT / "wld_v52_expanded_regulator_atlas.json"
PRIOR_VOCAB = (
    BACKUP
    / "wld_phase_b"
    / "priors"
    / "homo_sapiens_grch38"
    / "feature_vocab.json"
)
ROOT = BACKUP / "wld_v53_crispr_sciatac_ingestion"
RAW = ROOT / "raw_hg19"
LIFTED = ROOT / "lifted_grch38"
BUNDLE = ROOT / "bundle"
TOOLS = ROOT / "tools"
SCRATCH = Path("/content/wld_v53_scratch")
PACKAGES = Path("/content/wld_v53_packages")
LOG = ROOT / "wld_v53_ingestion.log"
for path in (ROOT, RAW, LIFTED, BUNDLE, TOOLS, SCRATCH, PACKAGES):
    path.mkdir(parents=True, exist_ok=True)

for required in (PANEL_REPORT, PANEL_ATLAS, PRIOR_VOCAB):
    if not required.is_file() or required.stat().st_size == 0:
        raise FileNotFoundError(f"Missing required v5.2 artifact: {required}")

panel_report = json.loads(PANEL_REPORT.read_text())
if panel_report.get("exact_design_match") is not True:
    raise RuntimeError("v5.3 requires the exact validated 105-target v5.2 panel")
TARGETS = sorted(set(map(str.upper, panel_report.get("targets", []))))
if len(TARGETS) != 105:
    raise RuntimeError(f"Expected 105 v5.2 targets, found {len(TARGETS)}")

SCREENS = {
    "screen1": {
        "atac_sample": "GSM4887677",
        "guide_sample": "GSM4887678",
        "atac_file": "GSM4887677_screen1_snATAC.bed.gz",
        "guide_file": "GSM4887678_screen1_snATACguide.IDs.mat.txt.gz",
    },
    "screen2": {
        "atac_sample": "GSM4887679",
        "guide_sample": "GSM4887680",
        "atac_file": "GSM4887679_screen2_snATAC.bed.gz",
        "guide_file": "GSM4887680_screen2_snATACguide.IDs.mat.txt.gz",
    },
}

BIN_SIZE = 2000
MAX_DATA_BINS = 20000
MIN_CELL_FRAGMENTS = 500
SEED = 42
CANONICAL_CHROMS = {f"chr{value}" for value in range(1, 23)} | {"chrX", "chrY"}
CONTROL_CANONICAL = {
    "CONTROL", "CTRL", "NEGATIVE", "NEGATIVECONTROL", "NONTARGET",
    "NONTARGETING", "NT", "NTC", "SCRAMBLE", "SCRAMBLED",
}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def stable_fraction(value):
    digest = hashlib.sha256(f"{SEED}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def download_resume(url, path, magic):
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            if handle.read(len(magic)) == magic:
                print(f"   cached: {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
                return
        path.rename(path.with_suffix(path.suffix + ".invalid"))
    partial = path.with_suffix(path.suffix + ".partial")
    command = [
        "curl", "-fL", "--retry", "8", "--retry-delay", "3",
        "--connect-timeout", "30", "--continue-at", "-",
        "--user-agent", "WLD-v5.3-ingestion/1.0",
        "--output", str(partial), url,
    ]
    subprocess.run(command, check=True)
    with partial.open("rb") as handle:
        observed = handle.read(len(magic))
    if observed != magic:
        raise RuntimeError(f"Unexpected signature {observed!r} for {url}")
    os.replace(partial, path)
    print(f"   downloaded: {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def validate_gzip(path):
    completed = subprocess.run(["gzip", "-t", str(path)], check=False)
    if completed.returncode:
        raise RuntimeError(f"Corrupt gzip stream: {path}")


def split_fields(line):
    line = line.rstrip("\r\n")
    if "\t" in line:
        return line.split("\t")
    if "," in line:
        return next(csv.reader([line]))
    return line.split()


def extract_target(value, targets):
    text = str(value).strip().upper().strip('"\'')
    canon = canonical(text)
    if canon in CONTROL_CANONICAL or "NONTARGET" in canon or "NEGCTRL" in canon:
        return "NTC"
    if text in targets:
        return text
    stripped = re.sub(r"(?:[-_:](?:SG|G)?RNA)?[-_:](?:G)?\d+$", "", text)
    stripped = re.sub(r"^(?:CRISPR[-_:]?)?(?:SGRNA|GRNA|GUIDE|SG)[-_:]", "", stripped)
    stripped = stripped.strip(" _-:.")
    if stripped in targets:
        return stripped
    for token in re.findall(r"[A-Z][A-Z0-9.-]{1,24}", text):
        if token in targets:
            return token
    return None


def normalized_key(value, mode):
    value = str(value).strip()
    if mode == "raw":
        return value
    if mode == "canonical":
        return re.sub(r"[^A-Za-z0-9]+", "", value).upper()
    raise ValueError(mode)


def load_assignment_candidates(path, targets):
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows.append(split_fields(line))
    if not rows:
        raise RuntimeError(f"Empty guide assignment: {path}")
    width = Counter(len(row) for row in rows).most_common(1)[0][0]
    rows = [row for row in rows if len(row) == width]
    annotated = []
    for row in rows:
        found = {extract_target(field, targets) for field in row}
        found.discard(None)
        if len(found) == 1:
            annotated.append((row, next(iter(found))))
    if len(annotated) < 100:
        raise RuntimeError(
            f"Only {len(annotated)} assignment rows had one unambiguous target in {path.name}"
        )

    candidates = []
    for column in range(width):
        values = [(row[column].strip(), target) for row, target in annotated if row[column].strip()]
        if not values:
            continue
        target_like = sum(extract_target(value, targets) is not None for value, _ in values)
        unique = len({value for value, _ in values})
        # CRISPR-sciATAC cell identifiers are themselves DNA barcode sequences,
        # so DNA-like values must not be rejected. The correct column is chosen
        # downstream by exact measured overlap with sampled deposited BED rows.
        if unique < 50 or target_like / len(values) > 0.2:
            continue
        candidates.append(
            {
                "column": column,
                "rows": values,
                "unique": unique,
                "unique_ratio": unique / len(values),
                "width": width,
            }
        )
    if not candidates:
        raise RuntimeError(f"No cell-identifier candidate column in {path.name}")
    return candidates, {
        "file": path.name,
        "rows": len(rows),
        "annotated_rows": len(annotated),
        "width": width,
        "candidate_columns": [
            {
                "column": value["column"],
                "unique": value["unique"],
                "unique_ratio": value["unique_ratio"],
            }
            for value in candidates
        ],
    }


def sample_bed(path, limit=100000):
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = split_fields(line)
            if len(fields) < 4:
                continue
            try:
                int(fields[1]); int(fields[2])
            except (ValueError, IndexError):
                continue
            rows.append(fields)
            if len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No BED records found in {path}")
    width = Counter(len(row) for row in rows).most_common(1)[0][0]
    return [row for row in rows if len(row) == width], width


def resolve_alignment(assignment_candidates, bed_rows, bed_width):
    best = None
    for assignment in assignment_candidates:
        for mode in ("raw", "canonical"):
            mapping = {}
            collisions = set()
            for value, target in assignment["rows"]:
                key = normalized_key(value, mode)
                if key in mapping and mapping[key] != target:
                    collisions.add(key)
                else:
                    mapping[key] = target
            for key in collisions:
                mapping.pop(key, None)
            for bed_column in range(3, bed_width):
                matched = 0
                targets = Counter()
                for row in bed_rows:
                    key = normalized_key(row[bed_column], mode)
                    target = mapping.get(key)
                    if target is not None:
                        matched += 1
                        targets[target] += 1
                record = {
                    "assignment_column": assignment["column"],
                    "bed_column": bed_column,
                    "mode": mode,
                    "sample_records": len(bed_rows),
                    "sample_matches": matched,
                    "sample_match_fraction": matched / len(bed_rows),
                    "sample_targets": len(targets),
                    "mapping": mapping,
                }
                if best is None or (
                    record["sample_matches"], record["sample_targets"]
                ) > (best["sample_matches"], best["sample_targets"]):
                    best = record
    if best is None or best["sample_matches"] < 100 or best["sample_match_fraction"] < 0.2:
        summary = None if best is None else {
            key: value for key, value in best.items() if key != "mapping"
        }
        raise RuntimeError(f"Guide/BED barcode alignment failed: {summary}")
    return best


def deterministic_target_split(targets_by_screen):
    screens = sorted(targets_by_screen)
    for left_index, left in enumerate(screens):
        for right in screens[left_index + 1:]:
            shared = set(targets_by_screen[left]) & set(targets_by_screen[right])
            if shared:
                raise RuntimeError(
                    "A perturbation target occurs in more than one screen and could "
                    f"cross split boundaries: {left}/{right} shared={sorted(shared)}"
                )
    result = {"train": [], "validation": [], "test": []}
    for screen, values in sorted(targets_by_screen.items()):
        ordered = sorted(values, key=lambda value: hashlib.sha256(f"{SEED}|{screen}|{value}".encode()).hexdigest())
        total = len(ordered)
        validation = max(3, int(round(total * 0.15)))
        test = max(3, int(round(total * 0.15)))
        if total - validation - test < 3:
            raise RuntimeError(f"Too few targets for a grouped split in {screen}: {total}")
        train_end = total - validation - test
        result["train"].extend(ordered[:train_end])
        result["validation"].extend(ordered[train_end:train_end + validation])
        result["test"].extend(ordered[train_end + validation:])
    return {key: sorted(value) for key, value in result.items()}


def install_liftover():
    binary = TOOLS / "liftOver"
    chain_gz = TOOLS / "hg19ToHg38.over.chain.gz"
    chain = TOOLS / "hg19ToHg38.over.chain"
    download_resume(
        "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver",
        binary,
        b"\x7fELF",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    download_resume(
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
        chain_gz,
        b"\x1f\x8b",
    )
    if not chain.is_file() or chain.stat().st_size == 0:
        temporary = chain.with_suffix(chain.suffix + ".partial")
        with gzip.open(chain_gz, "rb") as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        os.replace(temporary, chain)
    return binary, chain


def normalize_and_lift(screen, raw_path, alignment, cell_lookup, binary, chain):
    lifted_gz = LIFTED / f"{screen}.GRCh38.bed.gz"
    marker = LIFTED / f"{screen}.liftover.json"
    if lifted_gz.is_file() and marker.is_file():
        validate_gzip(lifted_gz)
        print(f"   PASS cached liftover: {lifted_gz.name}", flush=True)
        return json.loads(marker.read_text())

    normalized = SCRATCH / f"{screen}.hg19.normalized.bed"
    lifted_bed = SCRATCH / f"{screen}.GRCh38.bed"
    unmapped = SCRATCH / f"{screen}.unmapped.bed"
    for path in (normalized, lifted_bed, unmapped):
        path.unlink(missing_ok=True)

    input_records = 0
    aligned_records = 0
    with gzip.open(raw_path, "rt", errors="replace") as src, normalized.open("wt") as dst:
        for line in src:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = split_fields(line)
            if len(fields) <= alignment["bed_column"]:
                continue
            input_records += 1
            if input_records % 10_000_000 == 0:
                print(
                    f"      {screen}: scanned {input_records:,} BED records; "
                    f"aligned {aligned_records:,}",
                    flush=True,
                )
            try:
                start, end = int(fields[1]), int(fields[2])
            except (ValueError, IndexError):
                continue
            if start < 0 or end <= start:
                continue
            key = normalized_key(fields[alignment["bed_column"]], alignment["mode"])
            cell_index = cell_lookup.get(key)
            if cell_index is None:
                continue
            dst.write(f"{fields[0]}\t{start}\t{end}\tC{cell_index}\n")
            aligned_records += 1
    if aligned_records < 1000:
        raise RuntimeError(f"Only {aligned_records} aligned fragments in {screen}")

    subprocess.run(
        [
            str(binary), "-minMatch=0.95", "-bedPlus=3",
            str(normalized), str(chain), str(lifted_bed), str(unmapped),
        ],
        check=True,
    )
    mapped_records = sum(1 for _ in lifted_bed.open("rt"))
    unmapped_records = sum(
        1 for line in unmapped.open("rt") if line.strip() and not line.startswith("#")
    )
    map_fraction = mapped_records / max(aligned_records, 1)
    if map_fraction < 0.70:
        raise RuntimeError(f"Unexpectedly low hg19-to-GRCh38 mapping in {screen}: {map_fraction:.3f}")

    partial_gz = lifted_gz.with_suffix(lifted_gz.suffix + ".partial")
    partial_gz.unlink(missing_ok=True)
    with lifted_bed.open("rb") as src, gzip.open(partial_gz, "wb", compresslevel=5) as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    os.replace(partial_gz, lifted_gz)
    validate_gzip(lifted_gz)
    record = {
        "screen": screen,
        "source": str(raw_path),
        "source_sha256": sha256(raw_path),
        "input_bed_records": input_records,
        "guide_aligned_records": aligned_records,
        "mapped_records": mapped_records,
        "unmapped_records": unmapped_records,
        "map_fraction": map_fraction,
        "output": str(lifted_gz),
        "output_sha256": sha256(lifted_gz),
    }
    atomic_json(marker, record)
    for path in (normalized, lifted_bed, unmapped):
        path.unlink(missing_ok=True)
    return record


def iter_lifted(paths):
    for path in paths:
        with gzip.open(path, "rt", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 4 or not fields[3].startswith("C"):
                    continue
                try:
                    yield fields[0], int(fields[1]), int(fields[2]), int(fields[3][1:])
                except ValueError:
                    continue


def parse_bin(value):
    match = re.fullmatch(r"([^:]+):(\d+)-(\d+)", str(value))
    if not match:
        return None
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    center = (start + end) // 2
    begin = (center // BIN_SIZE) * BIN_SIZE
    return f"{chrom}:{begin}-{begin + BIN_SIZE}"


def build_sparse_bundle(lifted_paths, cells, target_splits):
    target_to_split = {
        target: split for split, targets in target_splits.items() for target in targets
    }
    split_by_cell = {}
    for index, cell in enumerate(cells):
        if cell["target"] == "NTC":
            fraction = stable_fraction(f"control|{cell['screen']}|{cell['barcode']}")
            split = "train" if fraction < 0.70 else ("validation" if fraction < 0.85 else "test")
        else:
            split = target_to_split[cell["target"]]
        split_by_cell[index] = split
        cell["split"] = split

    fragment_counts = Counter()
    print("   pass 1/3: counting GRCh38 fragments per aligned cell...", flush=True)
    for chrom, start, end, cell_index in iter_lifted(lifted_paths):
        if chrom not in CANONICAL_CHROMS or cell_index >= len(cells):
            continue
        fragment_counts[cell_index] += 1

    eligible_old = [
        index for index in range(len(cells)) if fragment_counts[index] >= MIN_CELL_FRAGMENTS
    ]
    if len(eligible_old) < 1000:
        raise RuntimeError(f"Only {len(eligible_old)} cells met the fragment threshold")
    eligible_set = set(eligible_old)

    training_bins = Counter()
    print("   pass 2/3: selecting bins from QC-passing training cells only...", flush=True)
    for chrom, start, end, cell_index in iter_lifted(lifted_paths):
        if (
            chrom in CANONICAL_CHROMS
            and cell_index in eligible_set
            and split_by_cell[cell_index] == "train"
        ):
            center = (start + end) // 2
            begin = (center // BIN_SIZE) * BIN_SIZE
            training_bins[f"{chrom}:{begin}-{begin + BIN_SIZE}"] += 1

    selected = [name for name, _count in training_bins.most_common(MAX_DATA_BINS)]
    vocab = json.loads(PRIOR_VOCAB.read_text())
    foundation = set()
    for value in vocab.get("peaks", vocab.get("atac", [])):
        parsed = parse_bin(value)
        if parsed and parsed.split(":", 1)[0] in CANONICAL_CHROMS:
            foundation.add(parsed)
    selected = sorted(set(selected) | foundation)
    if len(selected) < 1000:
        raise RuntimeError(f"Only {len(selected)} training-selected accessibility bins")
    bin_index = {name: index for index, name in enumerate(selected)}

    row_lookup = {old: new for new, old in enumerate(eligible_old)}
    rows = array("I")
    columns = array("I")
    print("   pass 3/3: assembling the sparse cell-by-bin matrix...", flush=True)
    for chrom, start, end, cell_index in iter_lifted(lifted_paths):
        row = row_lookup.get(cell_index)
        if row is None or chrom not in CANONICAL_CHROMS:
            continue
        center = (start + end) // 2
        begin = (center // BIN_SIZE) * BIN_SIZE
        column = bin_index.get(f"{chrom}:{begin}-{begin + BIN_SIZE}")
        if column is not None:
            rows.append(row)
            columns.append(column)

    sys.path.insert(0, str(PACKAGES))
    try:
        import numpy as np
        import scipy
        from scipy import sparse
    except Exception as exc:
        raise RuntimeError(f"Isolated NumPy/SciPy environment unavailable: {exc}")
    row_array = np.frombuffer(rows, dtype=np.uint32).astype(np.int64, copy=False)
    column_array = np.frombuffer(columns, dtype=np.uint32).astype(np.int64, copy=False)
    data = np.ones(row_array.shape[0], dtype=np.uint16)
    matrix = sparse.coo_matrix(
        (data, (row_array, column_array)),
        shape=(len(eligible_old), len(selected)),
        dtype=np.uint32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    temporary_matrix = BUNDLE / "atac_counts.GRCh38.2kb.npz.partial.npz"
    final_matrix = BUNDLE / "atac_counts.GRCh38.2kb.npz"
    sparse.save_npz(temporary_matrix, matrix, compressed=True)
    os.replace(temporary_matrix, final_matrix)

    with gzip.open(BUNDLE / "bins.GRCh38.2kb.tsv.gz", "wt") as handle:
        for name in selected:
            handle.write(name + "\n")
    with gzip.open(BUNDLE / "cells.tsv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row", "cell_id", "screen", "barcode", "target", "split", "fragments"])
        for row, old in enumerate(eligible_old):
            cell = cells[old]
            writer.writerow([
                row, f"C{old}", cell["screen"], cell["barcode"], cell["target"],
                cell["split"], fragment_counts[old],
            ])

    split_counts = Counter(cells[old]["split"] for old in eligible_old)
    target_counts = {
        split: len(targets) for split, targets in target_splits.items()
    }
    return {
        "cells": len(eligible_old),
        "bins": len(selected),
        "nonzero": int(matrix.nnz),
        "matrix": str(final_matrix),
        "matrix_sha256": sha256(final_matrix),
        "cell_counts_by_split": dict(sorted(split_counts.items())),
        "target_counts_by_split": target_counts,
        "feature_selection": (
            "top 2-kb GRCh38 bins by fragments from training targets and "
            "training control cells only, union frozen foundation bins"
        ),
        "test_values_used_for_feature_selection": False,
    }


def ensure_numerical_environment():
    probe = [
        sys.executable, "-c",
        (
            "import numpy, scipy; "
            "assert numpy.__version__ == '1.26.4'; "
            "print(numpy.__version__, scipy.__version__)"
        ),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGES)
    env["PYTHONNOUSERSITE"] = "1"
    check = subprocess.run(probe, env=env, text=True, capture_output=True)
    if check.returncode:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
                "--target", str(PACKAGES), "--no-cache-dir",
                "numpy==1.26.4", "scipy==1.16.3",
            ],
            check=True,
        )
        check = subprocess.run(probe, env=env, text=True, capture_output=True)
    if check.returncode:
        raise RuntimeError(check.stdout + check.stderr)
    # Prepend the isolated packages for the later sparse write.
    sys.path.insert(0, str(PACKAGES))
    print(f"   PASS: NumPy/SciPy {check.stdout.strip()}", flush=True)


print("WLD V5.3 CRISPR-SCIATAC INGESTION")
print("J/L and external test studies remain untouched. No model training.\n")

print("1. Verifying isolated numerical environment...", flush=True)
ensure_numerical_environment()

print("\n2. Downloading/resuming the two processed hg19 ATAC files...", flush=True)
for screen, values in SCREENS.items():
    sample = values["atac_sample"]
    filename = values["atac_file"]
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{sample[:7]}nnn/"
        f"{sample}/suppl/{filename}"
    )
    destination = RAW / filename
    download_resume(url, destination, b"\x1f\x8b")
    validate_gzip(destination)

print("\n3. Resolving exact cell-to-guide/BED barcode schemas...", flush=True)
alignment_reports = {}
cells = []
cell_lookups = {}
targets_by_screen = {}
for screen, values in SCREENS.items():
    guide_path = PANEL_ROOT / "raw_metadata" / values["guide_file"]
    raw_path = RAW / values["atac_file"]
    candidates, assignment_audit = load_assignment_candidates(guide_path, set(TARGETS))
    bed_rows, bed_width = sample_bed(raw_path)
    alignment = resolve_alignment(candidates, bed_rows, bed_width)
    mapping = alignment.pop("mapping")
    # Create globally unique cell IDs while retaining original barcodes outside
    # the encoder. Mapping keys are unique under the selected normalization.
    lookup = {}
    screen_targets = set()
    for key, target in sorted(mapping.items()):
        index = len(cells)
        cells.append({"screen": screen, "barcode": key, "target": target})
        lookup[key] = index
        if target != "NTC":
            screen_targets.add(target)
    if len(screen_targets) < 10:
        raise RuntimeError(f"Only {len(screen_targets)} targets aligned in {screen}")
    cell_lookups[screen] = lookup
    targets_by_screen[screen] = screen_targets
    alignment_reports[screen] = {
        "assignment": assignment_audit,
        "alignment": alignment,
        "mapped_cells": len(lookup),
        "mapped_targets": len(screen_targets),
    }
    print(
        f"   {screen}: {len(lookup)} cells, {len(screen_targets)} targets, "
        f"sample BED match {alignment['sample_match_fraction']:.3f}",
        flush=True,
    )

observed_targets = set().union(*targets_by_screen.values())
if observed_targets != set(TARGETS):
    missing = sorted(set(TARGETS) - observed_targets)
    extra = sorted(observed_targets - set(TARGETS))
    raise RuntimeError(f"Assignment panel mismatch; missing={missing}, extra={extra}")

target_splits = deterministic_target_split(targets_by_screen)
atomic_json(
    BUNDLE / "whole_target_split.json",
    {
        "seed": SEED,
        "unit": "perturbation target",
        "targets": target_splits,
        "controls": "barcode-hash split 70/15/15",
        "feature_selection_uses": "train targets and train control cells only",
        "test_evaluated": False,
    },
)
atomic_json(BUNDLE / "schema_alignment.json", alignment_reports)
print(
    "   PASS whole-target split: "
    + ", ".join(f"{name}={len(values)}" for name, values in target_splits.items()),
    flush=True,
)

print("\n4. Installing UCSC liftOver and lifting aligned fragments to GRCh38...", flush=True)
binary, chain = install_liftover()
liftover_reports = {}
lifted_paths = []
for screen, values in SCREENS.items():
    report = normalize_and_lift(
        screen,
        RAW / values["atac_file"],
        alignment_reports[screen]["alignment"],
        cell_lookups[screen],
        binary,
        chain,
    )
    liftover_reports[screen] = report
    lifted_paths.append(Path(report["output"]))
    print(
        f"   {screen}: {report['mapped_records']} mapped fragments "
        f"({report['map_fraction']:.3%})",
        flush=True,
    )

print("\n5. Building the training-only GRCh38 accessibility atlas and sparse matrix...", flush=True)
bundle = build_sparse_bundle(lifted_paths, cells, target_splits)
print(
    f"   PASS: {bundle['cells']} cells x {bundle['bins']} bins; "
    f"{bundle['nonzero']} nonzero entries",
    flush=True,
)

manifest = {
    "schema_version": "wld-v5.3-crispr-sciatac-ingestion",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "accession": "GSE161002",
    "source_build": "hg19",
    "model_build": "GRCh38",
    "bin_size": BIN_SIZE,
    "targets": len(TARGETS),
    "new_chromatin_regulator_nodes": panel_report.get("new_chromatin_regulator_node_count"),
    "alignment": alignment_reports,
    "liftover": liftover_reports,
    "bundle": bundle,
    "leakage_contract": {
        "split_before_feature_selection": True,
        "whole_target_split": True,
        "guide_identity_in_encoder": False,
        "target_identity_in_encoder": False,
        "cell_type_label_in_encoder": False,
        "test_values_used_for_feature_selection": False,
    },
    "claims": {
        "model_trained": False,
        "test_evaluated": False,
        "muscle_J_L_evaluated": False,
        "attractor_claim": False,
    },
}
manifest_path = BUNDLE / "wld_v53_ingestion_manifest.json"
atomic_json(manifest_path, manifest)

print("\n" + "=" * 78)
print("VERIFIED COMPLETE: WLD V5.3 CRISPR-SCIATAC INGESTION")
print("=" * 78)
print(f"Exact targets aligned:              {len(TARGETS)}")
print(f"Cells retained:                     {bundle['cells']}")
print(f"Training-selected 2-kb bins:        {bundle['bins']}")
print(f"Sparse nonzero entries:             {bundle['nonzero']}")
print(f"Target split counts:                {bundle['target_counts_by_split']}")
print(f"Cell split counts:                  {bundle['cell_counts_by_split']}")
print("Source coordinates:                 hg19")
print("Model coordinates:                  GRCh38")
print("Split before feature selection:     True")
print("Guide/target identity in encoder:   False")
print("Model trained:                      False")
print("Sealed targets or J/L evaluated:    False")
print("Attractor claim:                    False")
print(f"Manifest: {manifest_path}")
