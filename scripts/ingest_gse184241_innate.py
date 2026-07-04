"""Ingest GSE184241 (BCG trained immunity) -> model-schema CSV.
Labels are embedded in column names Plate_{p}_{v1|v2|v3}_{RPMI|LPS}_{n}:
v1 = pre-BCG baseline -> Quiescent; v2/v3 = post-BCG -> InnateMemory.
cue = LPS for LPS-restim columns, else 0.
"""
import gzip, sys, urllib.request
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/work/MultiscaleProject/src")
from chromatin_toggle.kg import load_kg

SC = "/private/tmp/claude-502/-Users-work-Downloads/27eeaf1a-8f01-4d74-a1a0-792b2f079e5b/scratchpad"
url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE184nnn/GSE184241/suppl/GSE184241_combined_raw_counts.txt.gz"
path = f"{SC}/g184.txt.gz"
try:
    open(path, "rb").close()
except FileNotFoundError:
    urllib.request.urlretrieve(url, path)

df = pd.read_csv(path, sep=r"\s+", index_col=0, engine="python")  # space-delimited genes x cells
df.columns = [str(c).strip('"') for c in df.columns]
df.index = [str(i).strip('"') for i in df.index]
cols = list(df.columns)
def label(c):
    p = c.split("_")
    v = next((x for x in p if x in ("v1", "v2", "v3")), None)
    if v == "v1":
        return "Quiescent"
    if v in ("v2", "v3"):
        return "InnateMemory"
    return None
def cue(c):
    return 1.0 if "_LPS_" in c or c.endswith("_LPS") else 0.0
labels = [label(c) for c in cols]
keep = [i for i, l in enumerate(labels) if l is not None]
print(f"{len(cols)} cells, {len(keep)} labelled")

kg = load_kg()
node_cols = list(kg.node_ids)
M = df.to_numpy(dtype=float)                        # genes x cells
gene_up = {str(g).upper(): i for i, g in enumerate(df.index)}
totals = M.sum(axis=0)                               # per cell
totals = np.where(totals > 0, totals, 1.0)
mat = np.zeros((len(cols), len(node_cols)))
mapped = []
for node, sym in kg.gene_map.items():
    gi = gene_up.get(sym.upper())
    if gi is not None and node in kg.node_index:
        norm = np.log1p(M[gi] / totals * 1e4)
        lo, hi = norm.min(), norm.max()
        mat[:, node_cols.index(node)] = 0.0 if hi <= lo else (norm - lo) / (hi - lo)
        mapped.append(node)
print("mapped nodes:", len(mapped))

lps_i = kg.node_index.get("LPS")
header = node_cols + ["label"]
lines = [",".join(header)]
cnt = {}
for i in keep:
    row = [f"{mat[i, j]}" for j in range(len(node_cols))]
    if lps_i is not None:
        row[lps_i] = f"{cue(cols[i])}"
    row.append(labels[i])
    lines.append(",".join(row))
    cnt[labels[i]] = cnt.get(labels[i], 0) + 1
open("/Users/work/MultiscaleProject/data/gse184241_innate.csv", "w").write("\n".join(lines) + "\n")
print("wrote:", cnt)
