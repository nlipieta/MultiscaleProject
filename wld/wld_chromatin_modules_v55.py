"""WLD v5.5 training-only chromatin-complex module compiler.

The v5.3 matrix is kept sparse at its complete response-bin resolution. Test
CSR values are streamed past and never converted to NumPy/SciPy arrays.
Complex accessibility modules are estimated only from screen-matched training
target-versus-NTC populations; validation/test values cannot enter them.
"""
from __future__ import annotations

import csv, gzip, hashlib, io, json, os, re, zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

SCHEMA_VERSION = "wld-v5.5-training-only-complex-accessibility-modules"
EMPTY = np.zeros(0, dtype=np.int64)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _member(names: Sequence[str], stem: str) -> str:
    matches = [n for n in names if PurePosixPath(n).name == stem + ".npy"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {stem}.npy, found {matches}")
    return matches[0]


def _small(z: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.load(io.BytesIO(z.read(name)), allow_pickle=False)


def _header(f) -> Tuple[Tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(f)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = reader(f)
    return tuple(map(int, shape)), bool(fortran), np.dtype(dtype)


def _read_exact(f, size: int) -> bytes:
    chunks, left = [], int(size)
    while left:
        block = f.read(left)
        if not block:
            raise EOFError("Truncated sparse NPZ member")
        chunks.append(block); left -= len(block)
    return b"".join(chunks)


def _discard(f, size: int) -> None:
    left = int(size)
    while left:
        block = f.read(min(left, 8 << 20))
        if not block:
            raise EOFError("Truncated sparse NPZ member while skipping sealed rows")
        left -= len(block)


def _merged(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for a, b in sorted((int(a), int(b)) for a, b in intervals if b > a):
        if a < 0 or b < a:
            raise ValueError("Invalid CSR interval")
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _stream_ranges(z: zipfile.ZipFile, name: str,
                   intervals: Sequence[Tuple[int, int]]) -> np.ndarray:
    intervals = _merged(intervals)
    with z.open(name) as f:
        shape, fortran, dtype = _header(f)
        if fortran or len(shape) != 1 or (intervals and intervals[-1][1] > shape[0]):
            raise RuntimeError(f"Invalid one-dimensional array {name}")
        out = np.empty(sum(b - a for a, b in intervals), dtype=dtype)
        src = dst = 0
        for a, b in intervals:
            _discard(f, (a - src) * dtype.itemsize)
            left = b - a
            while left:
                n = min(left, 1_000_000)
                out[dst:dst+n] = np.frombuffer(_read_exact(f, n*dtype.itemsize), dtype)
                dst += n; left -= n
            src = b
    return out


def _stream_positions(z: zipfile.ZipFile, name: str,
                      positions: Sequence[int]) -> Dict[int, int]:
    """Read only selected values from a one-dimensional NPZ member.

    This is used for CSR row pointers so sealed-row library sizes are never
    materialized as a full NumPy vector merely to locate train/development rows.
    The compressed member is traversed sequentially, but unselected values are
    discarded as uninterpreted bytes.
    """
    selected=sorted({int(position) for position in positions})
    with z.open(name) as f:
        shape,fortran,dtype=_header(f)
        if fortran or len(shape)!=1 or (selected and (selected[0]<0 or selected[-1]>=shape[0])):
            raise RuntimeError(f"Invalid one-dimensional array {name}")
        result: Dict[int,int]={}; source=0
        for position in selected:
            _discard(f,(position-source)*dtype.itemsize)
            result[position]=int(np.frombuffer(_read_exact(f,dtype.itemsize),dtype=dtype)[0])
            source=position+1
    return result


def _selected_csr(path: Path, rows: np.ndarray) -> sparse.csr_matrix:
    """Select increasing CSR rows without materializing excluded data values."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or (len(rows) > 1 and np.any(np.diff(rows) <= 0)):
        raise ValueError("Selected source rows must be unique and increasing")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shape = tuple(map(int, _small(z, _member(names, "shape")).tolist()))
        fmt = _small(z, _member(names, "format")).item()
        fmt = fmt.decode() if isinstance(fmt, bytes) else str(fmt)
        if fmt.lower() != "csr" or len(shape) != 2:
            raise RuntimeError("v5.3 accessibility archive must be two-dimensional CSR")
        if len(rows) and (rows[0] < 0 or rows[-1] >= shape[0]):
            raise IndexError("Metadata row outside sparse matrix")
        pointer_name=_member(names,"indptr")
        pointers=_stream_positions(z,pointer_name,[value for row in rows for value in (int(row),int(row)+1)])
        ranges=[(pointers[int(row)],pointers[int(row)+1]) for row in rows]
        indices = _stream_ranges(z, _member(names, "indices"), ranges)
        data = _stream_ranges(z, _member(names, "data"), ranges)
    lengths = np.asarray([b-a for a, b in ranges], dtype=np.int64)
    out_ptr = np.r_[0, np.cumsum(lengths)]
    if len(data) != out_ptr[-1] or len(indices) != len(data):
        raise RuntimeError("Selected CSR payload lengths disagree")
    out = sparse.csr_matrix((data, indices, out_ptr), shape=(len(rows), shape[1]))
    out.sum_duplicates(); out.eliminate_zeros()
    return out


def _frozen_target_rosters(path: Path) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, str], Mapping[str, object]]:
    """Read and validate the authoritative whole-perturbation-target split."""
    payload = json.loads(Path(path).read_text())
    raw = payload.get("targets")
    required = ("train", "validation", "test")
    if not isinstance(raw, Mapping) or set(raw) != set(required):
        raise RuntimeError(
            "whole_target_split.json must contain exactly train, validation, and test target rosters"
        )
    rosters: Dict[str, Tuple[str, ...]] = {}
    target_to_split: Dict[str, str] = {}
    for split in required:
        values = raw[split]
        if not isinstance(values, list):
            raise RuntimeError(f"Frozen {split} target roster must be a JSON list")
        normalized = tuple(_symbol(value) for value in values)
        if any(not target or target == "NTC" for target in normalized):
            raise RuntimeError(f"Frozen {split} target roster contains an invalid target")
        if len(set(normalized)) != len(normalized):
            raise RuntimeError(f"Frozen {split} target roster contains duplicate targets")
        rosters[split] = normalized
        for target in normalized:
            previous = target_to_split.get(target)
            if previous is not None:
                raise RuntimeError(
                    f"Frozen target rosters overlap: {target} occurs in {previous} and {split}"
                )
            target_to_split[target] = split
    if not target_to_split:
        raise RuntimeError("Frozen whole-target split is empty")
    if payload.get("test_evaluated") is not False:
        raise RuntimeError("whole_target_split.json no longer records a sealed target test")
    return rosters, target_to_split, payload


def _hashed_ntc_split(seed: object, screen: str, barcode: str) -> str:
    """Reproduce the v5.3 control|screen|barcode 70/15/15 assignment."""
    if isinstance(seed, bool):
        raise RuntimeError("Frozen whole-target split seed must be an integer")
    try:
        integer_seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Frozen whole-target split seed must be an integer") from exc
    if str(seed).strip() not in {str(integer_seed), f"{integer_seed}.0"}:
        raise RuntimeError("Frozen whole-target split seed must be an integer")
    screen, barcode = str(screen).strip(), str(barcode).strip()
    if not screen or not barcode:
        raise RuntimeError("NTC hash validation requires nonempty screen and barcode fields")
    digest = hashlib.sha256(
        f"{integer_seed}|control|{screen}|{barcode}".encode()
    ).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return "train" if fraction < 0.70 else ("validation" if fraction < 0.85 else "test")


@dataclass
class SparseFullChromatinBundle:
    accessibility: sparse.csr_matrix
    bins: Tuple[str, ...]
    foundation_anchor_indices: np.ndarray
    targets: Tuple[str, ...]
    screens: Tuple[str, ...]
    splits: Tuple[str, ...]
    source_rows: np.ndarray
    row_groups: Dict[Tuple[str, str, str], np.ndarray]
    provenance: Dict[str, object] = field(default_factory=dict)
    sealed_test_row_count: int = 0

    def __post_init__(self) -> None:
        self.accessibility = self.accessibility.tocsr()
        self.foundation_anchor_indices = np.asarray(self.foundation_anchor_indices, dtype=np.int64)
        self.source_rows = np.asarray(self.source_rows, dtype=np.int64)
        n, p = self.accessibility.shape
        if p != len(self.bins) or any(len(x) != n for x in (self.targets,self.screens,self.splits,self.source_rows)):
            raise ValueError("Sparse bundle dimensions disagree")
        if any(str(s).lower() == "test" for s in self.splits):
            raise ValueError("Sealed test rows cannot be materialized")
        if len(self.foundation_anchor_indices) and (
            self.foundation_anchor_indices.min() < 0 or self.foundation_anchor_indices.max() >= p
            or len(np.unique(self.foundation_anchor_indices)) != len(self.foundation_anchor_indices)
        ):
            raise ValueError("Invalid foundation anchor indices")
        if len(self.accessibility.data) and (np.any(self.accessibility.data < 0)
                                             or not np.isfinite(self.accessibility.data).all()):
            raise ValueError("Accessibility values must be finite and nonnegative")

    def rows(self, split: str, screen: str, target: str) -> np.ndarray:
        return self.row_groups.get((str(split).lower(), str(screen), _symbol(target)), EMPTY)

    def split_targets(self, split: str) -> List[str]:
        return sorted({t for (s,_q,t),r in self.row_groups.items()
                       if s == str(split).lower() and t != "NTC" and len(r)})

    def target_screens(self, split: str, target: str) -> List[str]:
        return sorted({q for (s,q,t),r in self.row_groups.items()
                       if s == str(split).lower() and t == _symbol(target) and len(r)})


def load_v53_sparse_full_bundle(bundle_root: Path, *, prior_root: Optional[Path]=None,
                                 foundation_peaks: Optional[Sequence[str]]=None,
                                 materialized_splits: Sequence[str]=("train","validation")) -> SparseFullChromatinBundle:
    root = Path(bundle_root)
    allowed = tuple(sorted({str(x).lower() for x in materialized_splits}))
    if not allowed or not set(allowed).issubset({"train", "validation"}):
        raise ValueError("Only train and validation splits may be materialized")
    paths = {"manifest":root/"wld_v53_ingestion_manifest.json",
             "split":root/"whole_target_split.json",
             "matrix":root/"atac_counts.GRCh38.2kb.npz",
             "cells":root/"cells.tsv.gz", "bins":root/"bins.GRCh38.2kb.tsv.gz"}
    for label,path in paths.items():
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing v5.3 {label}: {path}")
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("claims",{}).get("test_evaluated") is not False:
        raise RuntimeError("v5.3 manifest no longer records a sealed test")
    rosters, target_to_split, split_contract = _frozen_target_rosters(paths["split"])
    if "seed" not in split_contract:
        raise RuntimeError(
            "whole_target_split.json lacks the seed required to reproduce NTC hash splits"
        )
    with gzip.open(paths["bins"],"rt") as f:
        bins = tuple(x.strip() for x in f if x.strip())
    if len(set(bins)) != len(bins): raise RuntimeError("Duplicate response bins")
    if foundation_peaks is None and prior_root is not None:
        vocab = json.loads((Path(prior_root)/"feature_vocab.json").read_text())
        foundation_peaks = vocab.get("peaks", vocab.get("atac", []))
    foundation_peaks = tuple(map(str, foundation_peaks or ()))
    index = {x:i for i,x in enumerate(bins)}
    missing = [x for x in foundation_peaks if x not in index]
    if missing: raise RuntimeError(f"Missing {len(missing)} foundation anchors")
    anchors = np.asarray([index[x] for x in foundation_peaks], dtype=np.int64)
    metadata, sealed = [], 0
    declared_counts: Dict[str, int] = defaultdict(int)
    frozen_counts: Dict[str, int] = defaultdict(int)
    non_ntc_validated = ntc_hash_validated = 0
    frozen_test_target_rows = frozen_test_ntc_rows = 0
    with gzip.open(paths["cells"],"rt",newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not {"row","screen","barcode","target","split"}.issubset(reader.fieldnames or []):
            raise RuntimeError("v5.3 cell metadata schema is incomplete")
        for line_number, row in enumerate(reader, start=2):
            declared_split = str(row["split"]).strip().lower()
            if declared_split not in {"train", "validation", "test"}:
                raise RuntimeError(
                    f"Invalid declared split {row['split']!r} in cell metadata line {line_number}"
                )
            target = _symbol(row["target"])
            if not target:
                raise RuntimeError(f"Empty perturbation target in cell metadata line {line_number}")
            declared_counts[declared_split] += 1
            if target == "NTC":
                frozen_split = _hashed_ntc_split(
                    split_contract["seed"], row["screen"], row["barcode"]
                )
                ntc_hash_validated += 1
                if frozen_split == "test":
                    frozen_test_ntc_rows += 1
            else:
                frozen_split = target_to_split.get(target, "")
                if not frozen_split:
                    raise RuntimeError(
                        f"Metadata target {target!r} in line {line_number} is absent from "
                        "the frozen whole-target split"
                    )
                non_ntc_validated += 1
                if frozen_split == "test":
                    frozen_test_target_rows += 1
            # Validate all rows before reading any sparse values.  In particular,
            # a sealed test target mislabeled as train can never enter source_rows.
            if declared_split != frozen_split:
                kind = "NTC barcode-hash" if target == "NTC" else "whole-target"
                raise RuntimeError(
                    f"Cell metadata line {line_number} declares {declared_split} for {target}, "
                    f"but its frozen {kind} split is {frozen_split}; no matrix values were read"
                )
            frozen_counts[frozen_split] += 1
            if frozen_split == "test":
                sealed += 1
            elif frozen_split in allowed:
                selected = dict(row)
                selected["_frozen_split"] = frozen_split
                metadata.append(selected)
    metadata.sort(key=lambda x:int(x["row"]))
    source_rows = np.asarray([int(x["row"]) for x in metadata], dtype=np.int64)
    matrix = _selected_csr(paths["matrix"], source_rows)
    if matrix.shape[1] != len(bins): raise RuntimeError("Matrix/bin mismatch")
    targets=tuple(_symbol(x["target"]) for x in metadata)
    screens=tuple(str(x["screen"]) for x in metadata)
    splits=tuple(x["_frozen_split"] for x in metadata)
    grouped: Dict[Tuple[str,str,str],List[int]] = defaultdict(list)
    for i,key in enumerate(zip(splits,screens,targets)): grouped[key].append(i)
    groups={k:np.asarray(v,dtype=np.int64) for k,v in grouped.items()}
    for split in set(splits):
        if not any(s==split and t=="NTC" and len(r) for (s,_q,t),r in groups.items()):
            raise RuntimeError(f"No {split} NTC controls")
    matrix_hash=manifest.get("bundle",{}).get("matrix_sha256") or sha256_file(paths["matrix"])
    roster_payload = {split:list(rosters[split]) for split in ("train","validation","test")}
    roster_digest = hashlib.sha256(
        json.dumps(roster_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance={"v53_manifest_sha256":sha256_file(paths["manifest"]),
                "whole_target_split_sha256":sha256_file(paths["split"]),
                "whole_target_roster_sha256":roster_digest,
                "whole_target_roster_counts":{k:len(v) for k,v in rosters.items()},
                "v53_matrix_sha256":matrix_hash,
                "v53_cells_sha256":sha256_file(paths["cells"]),
                "v53_bins_sha256":sha256_file(paths["bins"]),
                "materialized_splits":list(allowed),
                "split_assignment_authority":"whole_target_split.json for perturbations; seeded barcode hash for NTC",
                "non_ntc_rows_validated_against_frozen_roster":non_ntc_validated,
                "ntc_rows_validated_against_seeded_hash":ntc_hash_validated,
                "declared_cell_split_counts":dict(sorted(declared_counts.items())),
                "frozen_cell_split_counts":dict(sorted(frozen_counts.items())),
                "sealed_test_target_rows":frozen_test_target_rows,
                "sealed_test_ntc_rows":frozen_test_ntc_rows,
                "materialized_source_rows":len(metadata),
                "test_metadata_rows_read_only_for_split_integrity":sealed,
                "test_metadata_fragments_field_used":False,
                "test_csr_data_or_indices_materialized":False,
                "test_csr_row_pointer_values_materialized":False,
                "csr_row_pointer_selection":"selected train/development endpoints streamed; unselected pointers discarded as bytes",
                "test_values_materialized":False}
    return SparseFullChromatinBundle(matrix,bins,anchors,targets,screens,splits,
                                     source_rows,groups,provenance,sealed)


@dataclass(frozen=True)
class CuratedComplexCatalog:
    complex_ids: Tuple[str, ...]
    complex_names: Tuple[str, ...]
    members: Tuple[Tuple[str, ...], ...]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not len(self.complex_ids)==len(self.complex_names)==len(self.members):
            raise ValueError("Complex catalog dimensions disagree")
        if len(set(self.complex_ids)) != len(self.complex_ids):
            raise ValueError("Complex IDs are not unique")
        if any(len(x)<2 for x in self.members):
            raise ValueError("Curated complexes need at least two members")


def _norm_header(x: object) -> str:
    return re.sub(r"[^a-z0-9]+","",str(x or "").lower())


def _table_payload(path: Path) -> Tuple[str,str]:
    if path.suffix.lower()==".zip":
        with zipfile.ZipFile(path) as z:
            candidates=[n for n in z.namelist() if PurePosixPath(n).suffix.lower() in {".txt",".tsv",".csv"}]
            if not candidates: raise RuntimeError("CORUM archive has no table")
            selected=sorted(candidates,key=lambda n:(0 if "human" in n.lower() else 1,
                                                     0 if "complex" in n.lower() else 1,n))[0]
            payload=z.read(selected)
    elif path.suffix.lower()==".gz":
        selected=path.name
        with gzip.open(path,"rb") as f: payload=f.read()
    else: selected,payload=path.name,path.read_bytes()
    try: return payload.decode("utf-8-sig"),selected
    except UnicodeDecodeError: return payload.decode("latin-1"),selected


def _column(headers: Sequence[str], aliases: Iterable[str], fuzzy: str="") -> Optional[str]:
    columns={_norm_header(x):x for x in headers}
    for alias in aliases:
        if alias in columns: return columns[alias]
    candidates=[original for normal,original in columns.items()
                if fuzzy and fuzzy in normal and "gene" in normal]
    return candidates[0] if len(candidates)==1 else None


def _human(x: object) -> bool:
    value=re.sub(r"[^a-z0-9]+"," ",str(x or "").lower()).strip()
    return value=="9606" or value=="human" or value.startswith("human ") or "homo sapiens" in value


def _genes(x: object) -> Tuple[str,...]:
    raw=str(x or "").strip()
    values=re.split(r"[;,|]+",raw)
    if len(values)==1 and re.search(r"\s",raw): values=raw.split()
    result=set()
    for value in values:
        value=_symbol(re.sub(r"\s*\([^)]*\)\s*$","",value))
        if value and value not in {"NA","N/A","NONE","NULL","-"}: result.add(value)
    return tuple(sorted(result))


def parse_corum_complexes(path: Path, *, allowed_genes: Optional[Iterable[str]]=None,
                          human_only: bool=True, min_members: int=2) -> CuratedComplexCatalog:
    """Parse legacy/current CORUM exports with flexible column aliases."""
    path=Path(path); text,member=_table_payload(path)
    # CORUM's tabular exports use semicolons *inside* the gene-members field;
    # prefer an explicit tab header before asking Sniffer to choose.
    first_line=text.splitlines()[0] if text.splitlines() else ""
    if "\t" in first_line: delimiter="\t"
    else:
        try: delimiter=csv.Sniffer().sniff(text[:65536],delimiters=",;").delimiter
        except csv.Error: delimiter=","
    reader=csv.DictReader(io.StringIO(text),delimiter=delimiter)
    headers=tuple(reader.fieldnames or ())
    ids=_column(headers,{"complexid","complexidentifier","corumcomplexid","complexidcorum"})
    names=_column(headers,{"complexname","name","complex","complexnamecorum"})
    organism=_column(headers,{"organism","species","organismname","taxon","taxid","taxonomyid"})
    genes=_column(headers,{"subunitsgenename","subunitsgenenames","subunitsgenesymbol",
                           "subunitsgenesymbols","genesymbol","genesymbols","genenames",
                           "membersgenename","membersgenesymbol","genes","subunitsgene"},"subunit")
    if genes is None or (ids is None and names is None):
        raise RuntimeError(f"Unrecognized CORUM columns: {headers}")
    implicit_human=("human" in member.lower() or "human" in path.name.lower())
    if human_only and organism is None and not implicit_human:
        raise RuntimeError("Human filtering requires an organism column or human-only archive")
    allowed=None if allowed_genes is None else {_symbol(x) for x in allowed_genes}
    merged: Dict[str,set[str]]=defaultdict(set); labels={}; raw=human=0
    for row in reader:
        raw+=1
        if human_only and organism is not None and not _human(row.get(organism,"")): continue
        human+=1
        label=str(row.get(names,"") if names else "").strip()
        cid=str(row.get(ids,"") if ids else "").strip() or "NAME:"+_symbol(label)
        values=set(_genes(row.get(genes,"")))
        if allowed is not None: values &= allowed
        if values: merged[cid] |= values; labels.setdefault(cid,label or cid)
    retained=sorted(x for x,v in merged.items() if len(v)>=min_members)
    if not retained: raise RuntimeError("No eligible human CORUM complexes")
    return CuratedComplexCatalog(tuple(retained),tuple(labels[x] for x in retained),
        tuple(tuple(sorted(merged[x])) for x in retained),
        {"source_path":str(path.resolve()),"source_sha256":sha256_file(path),
         "archive_member":member,"delimiter":delimiter,"implicit_human_archive":implicit_human,
         "columns":{"complex_id":ids,
         "complex_name":names,"organism":organism,"gene_members":genes},"human_only":human_only,
         "raw_rows":raw,"human_rows":human,"retained_complexes":len(retained)})


@dataclass(frozen=True)
class ComplexModuleConfig:
    bootstrap_replicates: int=100
    bootstrap_chunk_size: int=10
    min_abs_accessibility_effect: float=.02
    min_target_sign_stability: float=.80
    min_complex_sign_concordance: float=.65
    top_k_peaks_per_target: int=512
    top_k_peaks_per_module: int=1024
    min_training_members_per_complex: int=1
    seed: int=42

    def validate(self) -> None:
        if self.bootstrap_replicates<2 or self.bootstrap_chunk_size<1: raise ValueError("Invalid bootstrap config")
        if self.min_abs_accessibility_effect<0: raise ValueError("Negative effect threshold")
        if not .5<=self.min_target_sign_stability<=1 or not .5<=self.min_complex_sign_concordance<=1:
            raise ValueError("Stability thresholds must be in [0.5,1]")
        if min(self.top_k_peaks_per_target,self.top_k_peaks_per_module,self.min_training_members_per_complex)<1:
            raise ValueError("Top-k/member limits must be positive")


@dataclass
class ComplexAccessibilityModuleAtlas:
    regulator_vocab: Tuple[str,...]
    complex_ids: Tuple[str,...]
    complex_names: Tuple[str,...]
    module_vocab: Tuple[str,...]
    bins: Tuple[str,...]
    foundation_anchor_indices: np.ndarray
    regulator_complex_support: sparse.csr_matrix
    complex_module_effect: sparse.csr_matrix
    module_peak_loading: sparse.csr_matrix
    construction_targets: Tuple[str,...]
    target_peak_effect: sparse.csr_matrix
    target_peak_sign_stability: sparse.csr_matrix
    complex_members: Tuple[Tuple[str,...],...]
    construction_members: Tuple[Tuple[str,...],...]
    target_summaries: Tuple[Mapping[str,object],...]
    complex_summaries: Tuple[Mapping[str,object],...]
    provenance: Dict[str,object]

    def __post_init__(self) -> None:
        r,c,m,p,t=len(self.regulator_vocab),len(self.complex_ids),len(self.module_vocab),len(self.bins),len(self.construction_targets)
        shapes={"regulator_complex_support":(r,c),"complex_module_effect":(c,m),
                "module_peak_loading":(m,p),"target_peak_effect":(t,p),
                "target_peak_sign_stability":(t,p)}
        for name,shape in shapes.items():
            value=getattr(self,name)
            if not sparse.issparse(value) or value.shape!=shape: raise ValueError(f"Invalid {name} shape")
            value=value.tocsr()
            if len(value.data) and not np.isfinite(value.data).all(): raise ValueError(f"Nonfinite {name}")
            setattr(self,name,value)
        if not len(self.complex_names)==len(self.complex_members)==len(self.construction_members)==len(self.complex_summaries)==c:
            raise ValueError("Complex metadata dimensions disagree")
        if len(self.target_summaries)!=t: raise ValueError("Target summaries mismatch")
        self.foundation_anchor_indices=np.asarray(self.foundation_anchor_indices,dtype=np.int64)
        if any(str(x).lower()!="train" for x in self.provenance.get("construction_splits",[])):
            raise ValueError("Atlas contains a non-training construction split")


def _binary(matrix: sparse.csr_matrix, rows: np.ndarray) -> sparse.csr_matrix:
    value=matrix[np.asarray(rows,dtype=np.int64)].astype(np.float32,copy=True).tocsr()
    value.data.fill(1.); value.eliminate_zeros(); return value


def _means(matrix: sparse.csr_matrix, weights: np.ndarray) -> np.ndarray:
    return np.asarray(matrix.T.dot(np.asarray(weights,dtype=np.float32).T).T,dtype=np.float32)


def _top(indices: np.ndarray, magnitude: np.ndarray, limit: int) -> np.ndarray:
    indices=np.asarray(indices,dtype=np.int64)
    if len(indices)>limit:
        indices=indices[np.argpartition(magnitude[indices],-limit)[-limit:]]
    return indices[np.lexsort((indices,-magnitude[indices]))]


def _signature(bundle: SparseFullChromatinBundle, target: str, config: ComplexModuleConfig,
               rng: np.random.Generator, controls: Dict[str,sparse.csr_matrix]):
    observations=[]; audit=[]
    for screen in bundle.target_screens("train",target):
        target_rows=bundle.rows("train",screen,target)
        control_rows=bundle.rows("train",screen,"NTC")
        if not len(control_rows): raise RuntimeError(f"No train NTC for {target}/{screen}")
        tx=_binary(bundle.accessibility,target_rows)
        cx=controls.get(screen)
        if cx is None:
            cx=_binary(bundle.accessibility,control_rows)
            controls[screen]=cx
        weight=float(min(len(target_rows),len(control_rows)))
        observations.append((tx,cx,weight))
        audit.append({"screen":screen,"target_cells":len(target_rows),"control_cells":len(control_rows)})
    if not observations: raise RuntimeError(f"No train rows for {target}")
    total=sum(x[2] for x in observations); p=len(bundle.bins)
    effect=np.zeros(p,dtype=np.float32)
    for tx,cx,w in observations:
        effect+=(w/total)*(np.asarray(tx.mean(0)).ravel()-np.asarray(cx.mean(0)).ravel())
    sign=np.sign(effect); agree=np.zeros(p,dtype=np.int32); done=0
    while done<config.bootstrap_replicates:
        b=min(config.bootstrap_chunk_size,config.bootstrap_replicates-done)
        response=np.zeros((b,p),dtype=np.float32)
        for tx,cx,w in observations:
            tw=rng.multinomial(tx.shape[0],np.full(tx.shape[0],1/tx.shape[0]),size=b)/tx.shape[0]
            cw=rng.multinomial(cx.shape[0],np.full(cx.shape[0],1/cx.shape[0]),size=b)/cx.shape[0]
            response+=(w/total)*(_means(tx,tw)-_means(cx,cw))
        agree+=np.sum(response*sign[None,:]>0,axis=0); done+=b
    stability=agree.astype(np.float32)/config.bootstrap_replicates
    eligible=np.flatnonzero((np.abs(effect)>=config.min_abs_accessibility_effect)&
                            (stability>=config.min_target_sign_stability))
    selected=_top(eligible,np.abs(effect),config.top_k_peaks_per_target)
    out_effect=np.zeros(p,dtype=np.float32); out_stability=np.zeros(p,dtype=np.float32)
    out_effect[selected]=effect[selected]; out_stability[selected]=stability[selected]
    summary={"target":target,"construction_split":"train","screens":audit,
             "stable_selected_peaks":len(selected),
             "mean_absolute_selected_effect":float(np.abs(effect[selected]).mean()) if len(selected) else 0.,
             "mean_selected_sign_stability":float(stability[selected].mean()) if len(selected) else 0.}
    return out_effect,out_stability,summary


def compile_training_complex_modules(bundle: SparseFullChromatinBundle,
                                     catalog: CuratedComplexCatalog,
                                     regulator_vocab: Sequence[str], *,
                                     config: ComplexModuleConfig=ComplexModuleConfig(),
                                     output_root: Optional[Path]=None) -> ComplexAccessibilityModuleAtlas:
    """Build modules from train targets only; validation/test values are unread."""
    config.validate()
    if any(x.lower()=="test" for x in bundle.splits): raise RuntimeError("Test rows materialized")
    regulators=tuple(_symbol(x) for x in regulator_vocab)
    if len(set(regulators))!=len(regulators): raise ValueError("Duplicate regulator symbols")
    targets=tuple(bundle.split_targets("train"))
    if not targets: raise RuntimeError("No train perturbation targets")
    unknown=set(targets)-set(regulators)
    if unknown: raise RuntimeError(f"Train targets absent from regulator vocabulary: {sorted(unknown)}")
    rng=np.random.default_rng(config.seed); controls={}; effects=[]; stabilities=[]; target_audit=[]
    for target in targets:
        e,s,a=_signature(bundle,target,config,rng,controls)
        effects.append(e); stabilities.append(s); target_audit.append(a)
    target_effect=sparse.csr_matrix(np.stack(effects),dtype=np.float32)
    target_stability=sparse.csr_matrix(np.stack(stabilities),dtype=np.float32)
    ti={x:i for i,x in enumerate(targets)}; ri={x:i for i,x in enumerate(regulators)}
    eligible=[]; construction=[]
    for i,members in enumerate(catalog.members):
        used=tuple(sorted(set(members)&set(targets)))
        if len(used)>=config.min_training_members_per_complex: eligible.append(i); construction.append(used)
    if not eligible: raise RuntimeError("No curated complex has sufficient train members")
    ids=tuple(catalog.complex_ids[i] for i in eligible)
    names=tuple(catalog.complex_names[i] for i in eligible)
    members=tuple(catalog.members[i] for i in eligible)
    modules=[]; strengths=[]; complex_audit=[]
    for cid,name,all_members,used in zip(ids,names,members,construction):
        rows=np.asarray([ti[x] for x in used],dtype=np.int64)
        e=target_effect[rows].toarray(); s=target_stability[rows].toarray()
        evidence=s*(e!=0); denominator=evidence.sum(0)
        aggregate=np.divide((e*evidence).sum(0),denominator,out=np.zeros(len(bundle.bins),dtype=np.float32),where=denominator>0)
        positive=(evidence*(e>0)).sum(0); negative=(evidence*(e<0)).sum(0)
        concordance=np.divide(np.maximum(positive,negative),denominator,
                              out=np.zeros(len(bundle.bins),dtype=np.float32),where=denominator>0)
        keep=np.flatnonzero((denominator>0)&(np.abs(aggregate)>=config.min_abs_accessibility_effect)&
                            (concordance>=config.min_complex_sign_concordance))
        keep=_top(keep,np.abs(aggregate),config.top_k_peaks_per_module)
        module=np.zeros(len(bundle.bins),dtype=np.float32); module[keep]=aggregate[keep]; modules.append(module)
        strength=float(concordance[keep].mean()) if len(keep) else 0.; strengths.append(strength)
        complex_audit.append({"complex_id":cid,"complex_name":name,"curated_members":list(all_members),
            "construction_members":list(used),"construction_member_count":len(used),
            "stable_selected_peaks":len(keep),"mean_selected_sign_concordance":strength,
            "mean_absolute_selected_effect":float(np.abs(aggregate[keep]).mean()) if len(keep) else 0.})
    # A curated membership without a stable train-derived bin effect remains
    # provenance, not a runnable mechanistic edge. Keep only evidenced modules.
    active=[i for i,value in enumerate(modules) if np.count_nonzero(value)]
    if not active: raise RuntimeError("No complex produced a stable training-derived accessibility module")
    ids=tuple(ids[i] for i in active); names=tuple(names[i] for i in active)
    members=tuple(members[i] for i in active); construction=[construction[i] for i in active]
    modules=[modules[i] for i in active]; strengths=[strengths[i] for i in active]
    complex_audit=[complex_audit[i] for i in active]
    rr=[];cc=[]
    for column,values in enumerate(members):
        for value in values:
            if value in ri: rr.append(ri[value]);cc.append(column)
    support=sparse.csr_matrix((np.ones(len(rr),dtype=np.float32),(rr,cc)),shape=(len(regulators),len(ids)))
    provenance={"schema_version":SCHEMA_VERSION,"construction_splits":["train"],
        "construction_targets":list(targets),"validation_values_used":False,
        "test_values_materialized":False,"test_values_used":False,"screen_matched_controls":True,
        "response_definition":"binary target pseudobulk minus same-screen NTC pseudobulk",
        "config":{name:getattr(config,name) for name in config.__dataclass_fields__},
        "bundle":dict(bundle.provenance),"curated_complexes":dict(catalog.provenance),
        "sealed_test_row_count":bundle.sealed_test_row_count}
    atlas=ComplexAccessibilityModuleAtlas(regulators,ids,names,tuple("COMPLEX_MODULE:"+x for x in ids),
        bundle.bins,bundle.foundation_anchor_indices.copy(),support,
        sparse.diags(np.asarray(strengths,dtype=np.float32),format="csr"),
        sparse.csr_matrix(np.stack(modules),dtype=np.float32),targets,target_effect,target_stability,
        members,tuple(construction),tuple(target_audit),tuple(complex_audit),provenance)
    if output_root is not None: save_complex_module_atlas(atlas,output_root)
    return atlas


def _csr_payload(prefix: str, value: sparse.spmatrix) -> Dict[str,np.ndarray]:
    value=value.tocsr()
    return {prefix+"_data":value.data,prefix+"_indices":value.indices,
            prefix+"_indptr":value.indptr,prefix+"_shape":np.asarray(value.shape,dtype=np.int64)}


def _csr_load(values: Mapping[str,np.ndarray], prefix: str) -> sparse.csr_matrix:
    keys=[prefix+x for x in ("_data","_indices","_indptr","_shape")]
    if any(x not in values for x in keys): raise RuntimeError(f"Incomplete numeric matrix {prefix}")
    shape=tuple(map(int,values[prefix+"_shape"].tolist()))
    return sparse.csr_matrix((values[prefix+"_data"],values[prefix+"_indices"],
                              values[prefix+"_indptr"]),shape=shape)


def _tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str,object]]) -> None:
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter="\t",lineterminator="\n")
        writer.writeheader()
        for row in rows: writer.writerow({x:row.get(x,"") for x in fields})
    os.replace(tmp,path)


def save_complex_module_atlas(atlas: ComplexAccessibilityModuleAtlas,
                              output_root: Path) -> Dict[str,object]:
    """Write restart-safe NPZ+JSON+TSV artifacts and their SHA-256 digests."""
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    numeric=root/"complex_accessibility_modules.npz"; tmp=numeric.with_suffix(".npz.tmp")
    payload={"foundation_anchor_indices":atlas.foundation_anchor_indices}
    for name in ("regulator_complex_support","complex_module_effect","module_peak_loading",
                 "target_peak_effect","target_peak_sign_stability"):
        payload.update(_csr_payload(name,getattr(atlas,name)))
    with tmp.open("wb") as f: np.savez_compressed(f,**payload)
    os.replace(tmp,numeric)
    vocab=root/"complex_accessibility_vocab.json"
    _atomic_json(vocab,{"schema_version":SCHEMA_VERSION,"regulators":list(atlas.regulator_vocab),
        "complex_ids":list(atlas.complex_ids),"complex_names":list(atlas.complex_names),
        "complex_members":[list(x) for x in atlas.complex_members],
        "construction_members":[list(x) for x in atlas.construction_members],
        "modules":list(atlas.module_vocab),"bins":list(atlas.bins),
        "construction_targets":list(atlas.construction_targets),
        "target_summaries":[dict(x) for x in atlas.target_summaries],
        "complex_summaries":[dict(x) for x in atlas.complex_summaries]})
    complexes=root/"complex_accessibility_modules.tsv"
    complex_fields=("complex_id","complex_name","curated_members","construction_members",
        "construction_member_count","stable_selected_peaks","mean_selected_sign_concordance",
        "mean_absolute_selected_effect")
    _tsv(complexes,complex_fields,({**dict(x),"curated_members":";".join(x["curated_members"]),
          "construction_members":";".join(x["construction_members"])} for x in atlas.complex_summaries))
    targets=root/"complex_module_construction_targets.tsv"
    target_fields=("target","construction_split","screens","stable_selected_peaks",
                   "mean_absolute_selected_effect","mean_selected_sign_stability")
    _tsv(targets,target_fields,({**dict(x),"screens":json.dumps(x["screens"],sort_keys=True)}
                                for x in atlas.target_summaries))
    artifacts={p.name:{"sha256":sha256_file(p),"bytes":p.stat().st_size}
               for p in (numeric,vocab,complexes,targets)}
    manifest={**atlas.provenance,"dimensions":{"regulators":len(atlas.regulator_vocab),
        "complexes":len(atlas.complex_ids),"modules":len(atlas.module_vocab),
        "full_response_bins":len(atlas.bins),"foundation_anchor_bins":len(atlas.foundation_anchor_indices),
        "construction_targets":len(atlas.construction_targets),
        "regulator_complex_edges":atlas.regulator_complex_support.nnz,
        "module_peak_edges":atlas.module_peak_loading.nnz},"artifacts":artifacts,
        "claims":{"validation_values_used_for_module_construction":False,
        "test_values_materialized":False,"test_values_used":False,"model_trained":False,
        "attractor_claim":False}}
    _atomic_json(root/"complex_accessibility_module_manifest.json",manifest)
    return manifest


def load_complex_module_atlas(output_root: Path, *, verify_hashes: bool=True) -> ComplexAccessibilityModuleAtlas:
    """Verify saved artifacts and reconstruct the complete sparse atlas."""
    root=Path(output_root); manifest_path=root/"complex_accessibility_module_manifest.json"
    vocab_path=root/"complex_accessibility_vocab.json"; numeric_path=root/"complex_accessibility_modules.npz"
    for path in (manifest_path,vocab_path,numeric_path):
        if not path.is_file() or not path.stat().st_size: raise FileNotFoundError(path)
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema_version")!=SCHEMA_VERSION: raise RuntimeError("Unsupported module-atlas schema")
    if verify_hashes:
        for name,record in manifest.get("artifacts",{}).items():
            path=root/name
            if not path.is_file() or sha256_file(path)!=record.get("sha256"):
                raise RuntimeError(f"Missing or digest-mismatched atlas artifact: {name}")
    vocab=json.loads(vocab_path.read_text())
    required={"regulators","complex_ids","complex_names","complex_members","construction_members",
              "modules","bins","construction_targets","target_summaries","complex_summaries"}
    if vocab.get("schema_version")!=SCHEMA_VERSION or not required.issubset(vocab):
        raise RuntimeError("Incomplete module-atlas vocabulary")
    with np.load(numeric_path,allow_pickle=False) as z: values={x:z[x] for x in z.files}
    if "foundation_anchor_indices" not in values: raise RuntimeError("Missing foundation anchors")
    atlas=ComplexAccessibilityModuleAtlas(tuple(vocab["regulators"]),tuple(vocab["complex_ids"]),
        tuple(vocab["complex_names"]),tuple(vocab["modules"]),tuple(vocab["bins"]),
        values["foundation_anchor_indices"],_csr_load(values,"regulator_complex_support"),
        _csr_load(values,"complex_module_effect"),_csr_load(values,"module_peak_loading"),
        tuple(vocab["construction_targets"]),_csr_load(values,"target_peak_effect"),
        _csr_load(values,"target_peak_sign_stability"),
        tuple(tuple(x) for x in vocab["complex_members"]),
        tuple(tuple(x) for x in vocab["construction_members"]),
        tuple(dict(x) for x in vocab["target_summaries"]),
        tuple(dict(x) for x in vocab["complex_summaries"]),dict(manifest))
    dimensions=manifest.get("dimensions",{})
    observed={"regulators":len(atlas.regulator_vocab),"complexes":len(atlas.complex_ids),
              "modules":len(atlas.module_vocab),"full_response_bins":len(atlas.bins),
              "foundation_anchor_bins":len(atlas.foundation_anchor_indices),
              "construction_targets":len(atlas.construction_targets),
              "regulator_complex_edges":atlas.regulator_complex_support.nnz,
              "module_peak_edges":atlas.module_peak_loading.nnz}
    for name,value in observed.items():
        if name in dimensions and int(dimensions[name])!=value: raise RuntimeError(f"Dimension mismatch: {name}")
    return atlas


load_v53_bundle=load_v53_sparse_full_bundle
compile_complex_module_atlas=compile_training_complex_modules

__all__=["SparseFullChromatinBundle","CuratedComplexCatalog","ComplexModuleConfig",
    "ComplexAccessibilityModuleAtlas","load_v53_sparse_full_bundle","load_v53_bundle",
    "parse_corum_complexes","compile_training_complex_modules","compile_complex_module_atlas",
    "save_complex_module_atlas","load_complex_module_atlas","sha256_file"]
