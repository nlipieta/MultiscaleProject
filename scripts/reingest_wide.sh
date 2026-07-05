#!/usr/bin/env bash
# Part B re-ingest: rebuild every per-dataset CSV with the WIDE gene panel (kg.load_kg
# now auto-injects the marker panel, so ingestion captures ~148 genes instead of ~42),
# then re-pool. Most raw data is in data/geo_cache (no re-download). The two deleted
# h5ads (neuronal, osteo) re-download to cache, get ingested, then are removed to save disk.
# Run as ONE tracked background job. Continue-on-error so one bad dataset doesn't abort all.
set -u
cd /Users/work/MultiscaleProject
export PYTHONUNBUFFERED=1
FILT='warn|deprecat|Building|Built|Uninstalled|Installed|Resolved|Audited|Download'
G() { echo "=== $* @ $(date +%H:%M) ==="; uv run --extra census chromatin-geo "$@" 2>&1 | grep -viE "$FILT" | tail -4 || echo "  (FAILED: $*)"; }
PY() { echo "=== $1 @ $(date +%H:%M) ==="; uv run --extra census python "$@" 2>&1 | grep -viE "$FILT" | tail -4 || echo "  (FAILED: $*)"; }

# --- scRNA (genes x cells) ---
G --gse GSE172380 --scrna --out data/gse172380_adm.csv
G --gse GSE120064 --scrna --out data/gse120064_hypertrophy.csv
G --gse GSE113049 --scrna --out data/gse113049_regeneration.csv
G --gse GSE143437 --scrna --out data/gse143437_regeneration.csv
G --gse GSE188819 --scrna --out data/gse188819_adm.csv
G --gse GSE147405 --scrna --out data/gse147405_emt.csv
G --gse GSE115301 --scrna --out data/gse115301_senescence.csv
# --- 10x MTX ---
G --gse GSE135893 --mtx --out data/gse135893_fibrosis.csv
# --- SORT-seq trained immunity ---
G --emtab9702 --min-counts 500 --out data/emtab9702_macrophage.csv
# --- bulk series-matrix ---
G --gse GSE21608 --out data/GSE21608.csv
# --- h5ad (CELLxGENE / GEO), cached ---
G --h5ad-dataset GSE168776 --out data/gse168776_myogenesis.csv
G --h5ad-dataset EB_pluripotency --out data/eb_pluripotency.csv
G --h5ad-dataset HCM_cardiac --out data/hcm_hypertrophy.csv
G --h5ad-dataset GSE254185 --out data/gse254185_fibrosis.csv
G --h5ad-dataset DCM_fibroblast --out data/dcm_fibrosis.csv
# --- script-based ---
PY scripts/ingest_gse149451_myogenesis.py
PY scripts/ingest_gse184241_innate.py
# --- deleted h5ads: re-download -> ingest -> delete to manage disk ---
G --h5ad-dataset NeuronalDiff_organoid --out data/neuronal_diff.csv
rm -f data/geo_cache/0fff1010-a9fe-4586-b2c7-6359e39d5594.h5ad
G --h5ad-dataset Osteogenesis_craniofacial --out data/osteogenesis.csv
rm -f data/geo_cache/4d76b7b4-4d67-4016-b881-ab86e7f4d7f5.h5ad

# --- re-pool (wide) ---
echo "=== combine (wide) @ $(date +%H:%M) ==="
uv run --extra census chromatin-combine --cap-per-class 600 --harmonize \
    --out data/cross_pathway_eval.csv 2>&1 | grep -viE "$FILT" | tail -20

echo "WIDE RE-INGEST COMPLETE @ $(date +%H:%M)"
