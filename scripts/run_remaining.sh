#!/usr/bin/env bash
# Leaner remaining experiments (machine under heavy external load -> smaller config).
# Launch as ONE tracked background job; notifies once at the end.
set -u
cd /Users/work/MultiscaleProject
export PYTHONUNBUFFERED=1
A=artifacts
FILT='warn|deprecat|Building|Built|Uninstalled|Installed|Resolved|Audited|Download'

# 1) Leaner multi-seed baseline: 3-fold, 5k cells, epochs 35, grouped, seeds 0/1/2.
for S in 0 1 2; do
  echo "=== baseline seed $S @ $(date +%H:%M) ==="
  uv run --extra census python -m chromatin_toggle.baselines --group-split --class-weight \
      --models majority logreg rforest gboost --kfolds 3 --subsample 5000 \
      --steps 6 --hidden 64 --epochs 35 --seed $S 2>&1 | grep -viE "$FILT" \
      > $A/baseline_lean_s$S.txt 2>&1
  echo "done baseline seed $S"
done

# 2) Ablation (leaner): 3-fold not needed (fixed split), 5k cells, epochs 35.
echo "=== ablation @ $(date +%H:%M) ==="
uv run --extra census python -m chromatin_toggle.ablate --group-split --class-weight \
    --subsample 5000 --steps 6 --hidden 64 --epochs 35 --seed 0 2>&1 | grep -viE "$FILT" \
    > $A/ablation_lean_s0.txt 2>&1
echo "done ablation"

# 3) In-silico perturbation (reviewer #5): epochs 35.
echo "=== perturbation @ $(date +%H:%M) ==="
uv run --extra census python -m chromatin_toggle.perturb --subsample 5000 \
    --steps 6 --hidden 64 --epochs 35 --seed 0 2>&1 | grep -viE "$FILT" \
    > $A/perturb_hypertrophy_s0.txt 2>&1
echo "done perturbation"

echo "ALL REMAINING RUNS COMPLETE @ $(date +%H:%M)"
