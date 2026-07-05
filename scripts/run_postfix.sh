#!/usr/bin/env bash
# Post-fix consolidated batch (hybrid residual now default-on in the model).
# Launch as ONE tracked background job; notifies once at the end.
set -u
cd /Users/work/MultiscaleProject
export PYTHONUNBUFFERED=1
A=artifacts
FILT='warn|deprecat|Building|Built|Uninstalled|Installed|Resolved|Audited|Download'

# 1) FIX MEASUREMENT: hybrid KG-GNN vs baselines, n=3 (lean: 3-fold, 5k, ep35, grouped).
for S in 0 1 2; do
  echo "=== hybrid baseline seed $S @ $(date +%H:%M) ==="
  uv run --extra census python -m chromatin_toggle.baselines --group-split --class-weight \
      --models majority logreg rforest gboost --kfolds 3 --subsample 5000 \
      --steps 6 --hidden 64 --epochs 35 --seed $S 2>&1 | grep -viE "$FILT" \
      > $A/baseline_hybrid_s$S.txt 2>&1
  echo "done hybrid baseline seed $S"
done

# 2) Ablation (post-fix; now includes -hybrid_residual knockout).
echo "=== ablation @ $(date +%H:%M) ==="
uv run --extra census python -m chromatin_toggle.ablate --group-split --class-weight \
    --subsample 5000 --steps 6 --hidden 64 --epochs 35 --seed 0 2>&1 | grep -viE "$FILT" \
    > $A/ablation_postfix_s0.txt 2>&1
echo "done ablation"

# 3) In-silico perturbation (reviewer #5), post-fix model.
echo "=== perturbation @ $(date +%H:%M) ==="
uv run --extra census python -m chromatin_toggle.perturb --subsample 5000 \
    --steps 6 --hidden 64 --epochs 35 --seed 0 2>&1 | grep -viE "$FILT" \
    > $A/perturb_postfix_s0.txt 2>&1
echo "done perturbation"

echo "ALL POST-FIX RUNS COMPLETE @ $(date +%H:%M)"
