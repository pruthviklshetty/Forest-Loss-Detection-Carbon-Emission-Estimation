#!/usr/bin/env bash
# Phase 8 training sweep: plain U-Net on the pooled multi-region split (3 seeds)
# + 4 leave-one-region-out folds (1 seed each). Sequential; logs per run.
set -e
PY=./.venv/Scripts/python.exe
LOG=results/metrics/phase8_train_log.txt
mkdir -p "$(dirname "$LOG")"
echo "=== Phase 8 training sweep $(date) ===" | tee "$LOG"

for S in 42 43 44; do
  EXP="p8_pooled_unet_s${S}"
  echo "--- POOLED seed $S -> $EXP ---" | tee -a "$LOG"
  $PY -m src.train --config configs/train_pooled.yaml --seed $S --experiment "$EXP" 2>&1 | tee -a "$LOG"
  $PY -m src.eval.evaluate --experiment "$EXP" 2>&1 | tee -a "$LOG"
done

for R in wayanad kodagu nilgiris anamalai; do
  EXP="p8_loro_${R}"
  echo "--- LORO test=$R -> $EXP ---" | tee -a "$LOG"
  $PY -m src.train --config configs/train_pooled.yaml --scheme loro --loro-region $R --seed 42 --experiment "$EXP" 2>&1 | tee -a "$LOG"
  $PY -m src.eval.evaluate --experiment "$EXP" 2>&1 | tee -a "$LOG"
done

echo "=== done $(date) ===" | tee -a "$LOG"
