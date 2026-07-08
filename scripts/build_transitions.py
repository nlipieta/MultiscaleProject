"""Build the cell-state TRANSITION ATLAS (data/transitions.yaml).

One record per program/transition, harvesting the provenance fields that already live in the
codebase and marking the rest as gaps:

  terminal_program        <- KG program node
  intermediate_regulators <- KG cascade (incoming ACTIVATES edges, base graph)
  terminal_markers        <- data/marker_panel.yaml
  cue                     <- geo.py dataset registries (or CUE_HINTS for script-ingested ones)
  initial_state           <- the Quiescent-mapped cell types in the dataset program_map
  datasets / evidence     <- the pooled cross_pathway_eval.csv (sources, cell counts, label type)
  chromatin_changes       <- CHROMATIN (Multiome/scATAC where available) else <gap>
  time_course             <- TIME_COURSE (temporal datasets) else <gap>
  reversibility           <- <gap: not tested>
  persistence_after_cue_removal <- <gap: hysteresis not run per-program>

Re-run whenever programs/datasets change:  uv run python scripts/build_transitions.py
"""
from __future__ import annotations

import collections
from pathlib import Path

import pandas as pd
import yaml

from chromatin_toggle import geo
from chromatin_toggle.kg import DATA_DIR, load_kg
from chromatin_toggle.oracle import QUIESCENT

# cues for programs ingested via scripts (no geo registry entry)
CUE_HINTS = {
    "Adipogenesis": "adipogenic induction cocktail (D0->D5)",
    "EndMT": "IL1-beta + TGF-beta2",
    "MacrophageActivation": "LPS + IFN-gamma (M1 classical activation)",
    "Regeneration": "injury / partial hepatectomy / notexin (organ-dependent)",
    "Erythropoiesis": "hematopoietic differentiation (HSC -> erythroid)",
    "Megakaryopoiesis": "hematopoietic differentiation (HSC -> megakaryocytic)",
    "IntestinalDiff": "developmental crypt specification",
    "TrophoblastDiff": "trophoblast differentiation (CTB -> STB/EVT)",
    "GerminalCenter": "germinal-center reaction (in vivo)",
    "Tfh": "follicular microenvironment (in vivo)",
    "Treg": "follicular/regulatory microenvironment (in vivo)",
    "EMT": "TGF-beta / TNF-alpha",
    "Exhaustion": "chronic TCR stimulation (tumor)",
    "InnateMemory": "beta-glucan / oxLDL / BCG (trained immunity)",
    "MyogenicDiff": "serum withdrawal (myogenic differentiation)",
    "NeuronalDiff": "proneural differentiation (developmental; no extrinsic cue)",
    "Osteogenesis": "osteogenic differentiation (in vivo ossification; no extrinsic cue)",
    "Pluripotency": "pluripotency / EB differentiation (no extrinsic cue)",
    "Senescence": "oncogene-induced / irradiation / etoposide stress",
}
# experimental-result fields we have measured
TIME_COURSE = {
    "EMT": "gse147405 (0d/8h/1d/3d/7d); EMT-only Spearman(P,time) rho +0.164 (structure) "
           "vs -0.087 (edges removed) -- structure drives graded emergence",
}
CHROMATIN = {
    "Erythropoiesis": "10x Multiome GSE194122 -- scATAC gene-activity as a 2nd input channel (POC)",
    "Megakaryopoiesis": "10x Multiome GSE194122 -- scATAC gene-activity as a 2nd input channel (POC)",
}
# pathway tags whose labels are sample/condition-level (leakage axis) vs per-cell
SAMPLE_LEVEL = {"adipo_3t3l1", "endmt_huvec", "macrophage_m1", "liver_regeneration",
                "lung_fibrosis", "kidney_fibrosis", "cardiac_fibrosis", "cardiac_stretch",
                "cardiac_hcm_human", "trained_immunity", "trained_immunity2"}


def main():
    kg = load_kg(panel=False)          # base cascade only (regulators, not panel markers)
    node_ids = kg.node_ids
    programs = list(kg.program_nodes)

    regulators = collections.defaultdict(list)
    for s, _rel, d, _w in kg.edges:
        if node_ids[d] in programs:
            regulators[node_ids[d]].append(node_ids[s])

    cue_of, initial_of, ds_keys = {}, {}, collections.defaultdict(list)
    for regd in (geo.SCRNA, geo.H5AD, geo.MTX):
        for key, ds in regd.items():
            pm = dict(getattr(ds, "program_map", {}) or {})
            cue = getattr(ds, "cue", None)
            progs = {v for v in pm.values() if v != QUIESCENT}
            inits = sorted(k for k, v in pm.items() if v == QUIESCENT)
            for p in progs:
                ds_keys[p].append(key)
                if cue and p not in cue_of:
                    cue_of[p] = cue
                if inits and p not in initial_of:
                    initial_of[p] = inits

    panel = (yaml.safe_load((DATA_DIR / "marker_panel.yaml").read_text()) or {}).get("panel", {})

    # pool provenance: sources + cell counts per program
    pool = DATA_DIR / "cross_pathway_eval.csv"
    per_prog_paths = collections.defaultdict(collections.Counter)
    if pool.exists():
        df = pd.read_csv(pool, usecols=["label", "pathway"])
        for lab, sub in df.groupby("label"):
            per_prog_paths[lab] = collections.Counter(sub["pathway"])

    atlas = {}
    for p in programs:
        paths = per_prog_paths.get(p, {})
        n_cells = int(sum(paths.values()))
        n_src = len(paths)
        sample_lvl = any(pw in SAMPLE_LEVEL for pw in paths)
        label_type = ("sample/condition-level" if sample_lvl and n_src else
                      "per-cell" if n_src else "not in main pool (POC dataset only)")
        atlas[p] = {
            "terminal_program": p,
            "cue": cue_of.get(p) or CUE_HINTS.get(p, "<unspecified>"),
            "initial_state": (", ".join(initial_of[p]) if p in initial_of
                              else "baseline / progenitor (Quiescent)"),
            "intermediate_regulators": regulators.get(p, []),
            "terminal_markers": panel.get(p, [])[:10],
            "chromatin_changes": CHROMATIN.get(p, "<gap: no scATAC/Multiome yet>"),
            "time_course": TIME_COURSE.get(p, "<gap: no time-course dataset>"),
            "reversibility": "<gap: not tested>",
            "persistence_after_cue_removal": "<gap: hysteresis not run on resistance for this program>",
            "evidence_quality": f"{n_src} source(s), {n_cells} cells (capped); labels: {label_type}",
            "pool_sources": sorted(paths),
        }

    out = DATA_DIR / "transitions.yaml"
    header = ("# Cell-state transition atlas -- auto-built by scripts/build_transitions.py.\n"
              "# One record per program/transition. '<gap: ...>' marks provenance we do not yet have\n"
              "# (data to hunt next). Regenerate after adding programs/datasets.\n")
    out.write_text(header + yaml.safe_dump(atlas, sort_keys=True, width=100))
    gaps = sum(1 for p in atlas for k, v in atlas[p].items()
               if isinstance(v, str) and v.startswith("<gap"))
    print(f"wrote {out}  ({len(atlas)} transitions, {gaps} field-gaps flagged)")
    # gap summary per field
    for field in ("chromatin_changes", "time_course", "reversibility", "persistence_after_cue_removal"):
        have = [p for p in atlas if not str(atlas[p][field]).startswith("<gap")]
        print(f"  {field:<32} filled for {len(have)}/{len(atlas)}: {', '.join(have) or '(none)'}")


if __name__ == "__main__":
    main()
