#!/usr/bin/env bash
# Full re-ingest at the CURRENT KG (194 nodes: wide marker panel + T-cell Exhaustion).
# Rebuilds every per-dataset CSV so exhaustion/panel genes are captured across ALL datasets
# (avoids the "PDCD1 nonzero <=> exhaustion dataset" leakage), ingests exhaustion, then re-pools
# to a 13-class set. Continue-on-error.
#
# Colab (recommended, big disk):  pip install -q anndata scipy scanpy pyyaml pandas scikit-learn
#                                 RUN="" bash scripts/reingest_wide.sh
# Local (uv):                     RUN="uv run --extra census" bash scripts/reingest_wide.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
RUN="${RUN:-}"                                   # empty on Colab (entry points on PATH); uv locally
FILT='warn|deprecat|Building|Built|Uninstalled|Installed|Resolved|Audited|Download'
G()  { echo "=== geo $* @ $(date +%H:%M) ==="; ${RUN} chromatin-geo "$@" 2>&1 | grep -viE "$FILT" | tail -3 || echo "  (FAILED: $*)"; }
PY() { echo "=== py $* @ $(date +%H:%M) ==="; ${RUN} python "$@" 2>&1 | grep -viE "$FILT" | tail -3 || echo "  (FAILED: $*)"; }

# --- 19 existing datasets: re-ingest so they capture all 194 KG nodes ---
G --gse GSE172380 --scrna --out data/gse172380_adm.csv
G --gse GSE120064 --scrna --out data/gse120064_hypertrophy.csv
G --gse GSE113049 --scrna --out data/gse113049_regeneration.csv
G --gse GSE143437 --scrna --out data/gse143437_regeneration.csv
G --gse GSE188819 --scrna --out data/gse188819_adm.csv
G --gse GSE147405 --scrna --out data/gse147405_emt.csv
G --gse GSE115301 --scrna --out data/gse115301_senescence.csv
G --gse GSE135893 --mtx  --out data/gse135893_fibrosis.csv
G --emtab9702 --min-counts 500 --out data/emtab9702_macrophage.csv
G --gse GSE21608 --out data/GSE21608.csv
G --h5ad-dataset GSE168776 --out data/gse168776_myogenesis.csv
G --h5ad-dataset EB_pluripotency --out data/eb_pluripotency.csv
G --h5ad-dataset HCM_cardiac --out data/hcm_hypertrophy.csv
G --h5ad-dataset GSE254185 --out data/gse254185_fibrosis.csv
G --h5ad-dataset DCM_fibroblast --out data/dcm_fibrosis.csv
PY scripts/ingest_gse149451_myogenesis.py
PY scripts/ingest_gse184241_innate.py
G --h5ad-dataset NeuronalDiff_organoid --out data/neuronal_diff.csv
rm -f data/geo_cache/0fff1010-a9fe-4586-b2c7-6359e39d5594.h5ad     # free disk (local)
G --h5ad-dataset Osteogenesis_craniofacial --out data/osteogenesis.csv
rm -f data/geo_cache/4d76b7b4-4d67-4016-b881-ab86e7f4d7f5.h5ad

# --- 13th program: T-cell exhaustion (5 cancers) ---
PY scripts/ingest_gse156728_exhaustion.py BC,PACA,MM,ESCA,RC data/gse156728_exhaustion.csv

# --- re-pool -> 13-class, 194-node wide set ---
echo "=== combine (13-class) @ $(date +%H:%M) ==="
${RUN} chromatin-combine --cap-per-class 600 --harmonize --out data/cross_pathway_eval.csv \
    2>&1 | grep -viE "$FILT" | tail -24
echo "=== sanity: cols + programs ==="
${RUN} python -c "import pandas as pd; d=pd.read_csv('data/cross_pathway_eval.csv'); print('cols',d.shape[1],'rows',len(d)); print(sorted(d['label'].unique()))"
echo "13-CLASS RE-INGEST COMPLETE @ $(date +%H:%M)"
