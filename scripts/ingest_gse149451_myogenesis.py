"""Ingest GSE149451 (human hiPSC PAX7 myogenesis) -> model-schema CSV.
Louvain clusters are unlabeled; assign MyogenicDiff vs Quiescent by scoring
differentiation markers (MYOG/MYH3/MYOD1) vs progenitor markers (PAX7/MYF5/MKI67).
"""
import gzip, numpy as np, pandas as pd, sys
sys.path.insert(0, "/Users/work/MultiscaleProject/src")
from chromatin_toggle.kg import load_kg

SC = "/private/tmp/claude-502/-Users-work-Downloads/73c4d86d-b9b6-462c-b513-e8d3a315f446/scratchpad"
X = pd.read_csv(f"{SC}/g149_X.csv.gz", index_col=0)          # cells x genes (symbols)
obs = pd.read_csv(f"{SC}/g149_CellData.csv.gz", index_col=0)  # louvain per cell
X = X.loc[obs.index]                                          # align
lou = obs["louvain"].to_numpy()

diff_m = [g for g in ["MYOG", "MYH3", "MYOD1"] if g in X.columns]
undiff_m = [g for g in ["PAX7", "MYF5", "MKI67"] if g in X.columns]
print("diff markers:", diff_m, "| undiff markers:", undiff_m)
cluster_prog = {}
for c in sorted(set(lou)):
    m = lou == c
    d = X.loc[m, diff_m].to_numpy().mean()
    u = X.loc[m, undiff_m].to_numpy().mean()
    cluster_prog[c] = "MyogenicDiff" if d > u else "Quiescent"
    print(f"cluster {c}: n={m.sum():4d}  diff={d:.3f}  undiff={u:.3f} -> {cluster_prog[c]}")

kg = load_kg()
node_cols = list(kg.node_ids)
# map KG gene nodes (human symbol) to X columns; min-max scale (data pre-normalized)
mat = np.zeros((X.shape[0], len(node_cols)))
mapped = []
for node, sym in kg.gene_map.items():
    if node in kg.node_index and sym in X.columns:
        col = X[sym].to_numpy(dtype=float)
        lo, hi = col.min(), col.max()
        mat[:, node_cols.index(node)] = 0.0 if hi <= lo else (col - lo) / (hi - lo)
        mapped.append(node)
print("mapped nodes:", len(mapped))

labels = [cluster_prog[c] for c in lou]
header = node_cols + ["label"]
lines = [",".join(header)]
cnt = {}
for i in range(X.shape[0]):
    row = [f"{mat[i, j]}" for j in range(len(node_cols))] + [labels[i]]
    lines.append(",".join(row))
    cnt[labels[i]] = cnt.get(labels[i], 0) + 1
open("/Users/work/MultiscaleProject/data/gse149451_myogenesis.csv", "w").write("\n".join(lines) + "\n")
print("wrote", X.shape[0], "cells:", cnt)
