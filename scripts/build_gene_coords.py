"""(Re)build gene-coordinate tables for the KG genes, in BOTH genome builds, so ATAC peak->gene
gene-activity uses coordinates that match each dataset's build.

Bug this fixes: data/gene_coords.tsv was GRCh38-only and predated the Erythropoiesis/Megakaryopoiesis
genes -> the erythroid regulators (GATA1/KLF1/TAL1/HBB/...) had NO coords, so their accessibility was
0; and GSE207308 SHARE-seq peaks are hg19 (GRCh37) while the coords were hg38 -> build mismatch.

Fetches from Ensembl REST (batch POST /lookup/symbol) on both servers:
  rest.ensembl.org        -> GRCh38 -> data/gene_coords.tsv         (10x Multiome GSE194122, etc.)
  grch37.rest.ensembl.org -> GRCh37 -> data/gene_coords_hg19.tsv    (SHARE-seq GSE207308)

Usage:  uv run python scripts/build_gene_coords.py
"""
from __future__ import annotations

import json
import time
import urllib.request

from chromatin_toggle.kg import DATA_DIR, load_kg

SERVERS = {"data/gene_coords.tsv": "https://rest.ensembl.org",
           "data/gene_coords_hg19.tsv": "https://grch37.rest.ensembl.org"}


def _post_symbols(server, symbols):
    """Ensembl batch symbol lookup -> {SYMBOL_UPPER: (chrom, start, end, strand)}."""
    out = {}
    for i in range(0, len(symbols), 200):                       # server cap ~1000; keep chunks small
        chunk = symbols[i:i + 200]
        req = urllib.request.Request(
            f"{server}/lookup/symbol/homo_sapiens",
            data=json.dumps({"symbols": chunk}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    d = json.loads(r.read())
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        for sym, info in d.items():
            if info and "seq_region_name" in info:
                out[sym.upper()] = (str(info["seq_region_name"]), int(info["start"]),
                                    int(info["end"]), int(info.get("strand", 1)))
        time.sleep(0.3)                                         # be polite to the API
    return out


def main():
    kg = load_kg()
    syms = sorted({str(s) for s in kg.gene_map.values()})
    print(f"{len(syms)} KG gene symbols to resolve")
    ery = ["GATA1", "KLF1", "TAL1", "HBB", "HBA1", "ALAS2", "EPOR", "GYPA", "SLC4A1", "SPTA1",
           "TFRC", "AHSP"]                                      # the ones that were missing
    for out_path, server in SERVERS.items():
        print(f"\n[{server}] -> {out_path}")
        coords = _post_symbols(server, syms)
        lines = ["symbol\tchrom\tstart\tend\tstrand"]
        for s in syms:
            c = coords.get(s.upper())
            if c:
                lines.append(f"{s}\t{c[0]}\t{c[1]}\t{c[2]}\t{c[3]}")
        (DATA_DIR / out_path.split("/")[-1]).write_text("\n".join(lines) + "\n")
        got_ery = [g for g in ery if g in coords]
        print(f"  resolved {len(coords)}/{len(syms)} genes; erythroid regulators found: {got_ery}")


if __name__ == "__main__":
    main()
