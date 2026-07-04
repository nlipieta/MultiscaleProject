"""Pool per-pathway CSVs into one CROSS-PATHWAY training set.

Each source CSV is already in the model's node-column + `label` schema (from
`geo`, `scperturb`, `census`, or hand-built). This aligns them onto the shared
KG node columns, tags each row with provenance (`dataset`, `pathway`, `assay`),
and concatenates. Each row keeps its own cue + memory + program label, so a
model trained on the pooled set learns every program jointly over the shared
graph -- the point of the KG substrate.

Because single-cell pathways contribute thousands of rows while bulk pathways
contribute a handful, `--cap-per-class` subsamples over-represented (pathway,
label) groups so the pooled set is not dominated by one assay. Provenance
columns are ignored by `dataset.load_csv` (it reads only KG-node columns +
`label`), so they are safe to carry for analysis / stratified splitting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .kg import DATA_DIR, load_kg


def combine(sources: list[tuple[str, str, str]], out: Path,
            cap_per_class: int | None = None, seed: int = 0) -> None:
    """sources: list of (csv_path, pathway, assay)."""
    kg = load_kg()
    node_cols = list(kg.node_ids)
    rng = np.random.default_rng(seed)
    frames = []

    for path, pathway, assay in sources:
        df = pd.read_csv(path)
        if "label" not in df.columns:
            raise SystemExit(f"{path} has no 'label' column")
        df = df[df["label"].notna() & (df["label"].astype(str) != "")]
        # align onto the canonical node columns (missing -> 0)
        aligned = pd.DataFrame(0.0, index=df.index, columns=node_cols)
        for c in node_cols:
            if c in df.columns:
                aligned[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        aligned["label"] = df["label"].astype(str).values
        aligned["dataset"] = Path(path).stem
        aligned["pathway"] = pathway
        aligned["assay"] = assay

        # optional per-(pathway,label) cap to curb single-cell dominance
        if cap_per_class is not None:
            capped = []
            for lab, grp in aligned.groupby("label"):
                if len(grp) > cap_per_class:
                    idx = rng.choice(grp.index.to_numpy(), cap_per_class, replace=False)
                    grp = grp.loc[idx]
                capped.append(grp)
            aligned = pd.concat(capped)
        frames.append(aligned)
        print(f"  + {Path(path).stem:<28} {pathway:<16} {assay:<10} "
              f"{len(aligned):>6} rows  ({df['label'].nunique()} labels)")

    pooled = pd.concat(frames, ignore_index=True)
    cols = node_cols + ["label", "dataset", "pathway", "assay"]
    pooled[cols].to_csv(out, index=False)

    print(f"\nCross-pathway set -> {out}  ({len(pooled)} rows)")
    print("  per program class:")
    for lab, n in pooled["label"].value_counts().items():
        paths = pooled[pooled.label == lab]["pathway"].unique()
        print(f"    {lab:<14} {n:>6}   from: {', '.join(paths)}")
    print("  per pathway:")
    for pw, n in pooled["pathway"].value_counts().items():
        print(f"    {pw:<16} {n:>6}")


# Registered cross-pathway composition (source CSVs must exist; build them with
# chromatin-geo / chromatin-scperturb first).
DEFAULT_SOURCES = [
    ("data/gse172380_adm.csv",           "ADM_pancreas",        "scRNA"),       # ADM
    ("data/gse120064_hypertrophy.csv",   "cardiac_stretch",     "scRNA"),       # Hypertrophy
    ("data/gse135893_fibrosis.csv",      "lung_fibrosis",       "scRNA"),       # Fibrosis
    ("data/emtab9702_macrophage.csv",    "trained_immunity",    "scRNA"),       # InnateMemory
    ("data/gse168776_myogenesis.csv",    "myogenesis",          "scRNA"),       # MyogenicDiff
    ("data/gse113049_regeneration.csv",  "lung_regeneration",   "scRNA"),       # Regeneration
    ("data/gse143437_regeneration.csv",  "muscle_regeneration", "scRNA"),       # Regeneration
    ("data/eb_pluripotency.csv",         "pluripotency",        "scRNA"),       # Pluripotency
    ("data/GSE21608.csv",                "TGFb_lineage",        "microarray"),  # TGF-beta
    # breadth: second, independent source per program (different tissue/organism)
    ("data/gse188819_adm.csv",           "pancreatitis2",       "scRNA"),       # ADM (caerulein)
    ("data/hcm_hypertrophy.csv",         "cardiac_hcm_human",   "scRNA"),       # Hypertrophy (human)
    ("data/gse254185_fibrosis.csv",      "kidney_fibrosis",     "scRNA"),       # Fibrosis (kidney)
    ("data/gse149451_myogenesis.csv",    "myogenesis_human",    "scRNA"),       # MyogenicDiff (human)
    ("data/gse184241_innate.csv",        "trained_immunity2",   "sortseq"),     # InnateMemory (BCG, human)
    ("data/dcm_fibrosis.csv",             "cardiac_fibrosis",    "snRNA"),       # Fibrosis (cardiac, 143k)
    ("data/gse147405_emt.csv",            "emt_tnf",             "scRNA"),       # EMT (NEW program)
    ("data/gse115301_senescence.csv",     "senescence_ois",      "scRNA"),       # Senescence (NEW program)
]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Pool per-pathway CSVs into a cross-pathway set")
    ap.add_argument("--add", action="append", default=[],
                    help="source as path:pathway:assay (repeatable). "
                         "Omit to use the registered DEFAULT_SOURCES.")
    ap.add_argument("--cap-per-class", type=int, default=None,
                    help="subsample any (pathway,label) group above this many rows")
    ap.add_argument("--out", default=str(DATA_DIR / "cross_pathway.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.add:
        sources = []
        for spec in args.add:
            parts = spec.split(":")
            if len(parts) != 3:
                raise SystemExit(f"--add must be path:pathway:assay, got {spec!r}")
            sources.append(tuple(parts))
    else:
        sources = DEFAULT_SOURCES

    combine(sources, Path(args.out), cap_per_class=args.cap_per_class, seed=args.seed)


if __name__ == "__main__":
    main()
