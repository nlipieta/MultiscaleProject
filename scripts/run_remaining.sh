#!/usr/bin/env bash
# Runs the remaining leak-free GNN experiments sequentially, each to its own log.
# Launch as ONE tracked background job so it notifies once at the end (~1.5h).
set -u
cd /Users/work/MultiscaleProject
export PYTHONUNBUFFERED=1
A=artifacts
RUN() { echo "=== $1 @ $(date +%H:%M) ==="; shift; uv run --extra census python -m "$@" 2>&1 \
        | grep -viE "warn|deprecat|Building|Built|Uninstalled|Installed|Resolved|Audited"; }

# 1) Baseline seed 2 (grouped, converged) -> n=3 with seeds 0,1,2
RUN "baseline seed2" chromatin_toggle.baselines --group-split --class-weight \
    --models majority logreg rforest gboost --steps 6 --hidden 64 --epochs 40 --seed 2 \
    > $A/baseline_grouped_ep40_s2.txt 2>&1
echo "done baseline seed2"

# 2) Ablation table (reviewer #4), converged
RUN "ablation" chromatin_toggle.ablate --group-split --class-weight \
    --steps 6 --hidden 64 --epochs 40 --seed 0 \
    > $A/ablation_grouped_ep40_s0.txt 2>&1
echo "done ablation"

# 3) In-silico perturbation (reviewer #5)
RUN "perturbation" chromatin_toggle.perturb --steps 6 --hidden 64 --epochs 40 --seed 0 \
    > $A/perturb_hypertrophy_s0.txt 2>&1
echo "done perturbation"

echo "ALL REMAINING RUNS COMPLETE @ $(date +%H:%M)"
