#!/usr/bin/env bash
# Phase 10 - full 2021->2023 pipeline. Writes only to period-scoped paths
# (data/*/2021_2023/, results/*/p10_*, results/carbon_validation/2021_2023/);
# the 2019->2021 artifacts are never touched.
#
# Order: download -> labels -> dataset -> LEAKAGE CHECK (gates training) ->
#        3-seed training + eval -> aggregate -> region inference / area / carbon.
set -e
PY=./.venv/Scripts/python.exe
CFG=configs/period_2021_2023.yaml
TCFG=configs/train_p10.yaml
LOG=results/metrics/phase10_pipeline_log.txt
export PYTHONPATH=.
mkdir -p "$(dirname "$LOG")"
echo "=== Phase 10 pipeline $(date) ===" | tee "$LOG"

echo "--- 1. download (4 regions, 2021 + 2023 windows + GFC) ---" | tee -a "$LOG"
$PY -m src.preprocessing.download_data --config "$CFG" --skip-existing 2>&1 | tee -a "$LOG"

echo "--- 2. build labels (lossyear {21,22}) ---" | tee -a "$LOG"
$PY -m src.preprocessing.build_labels --config "$CFG" 2>&1 | tee -a "$LOG"

echo "--- 3. build dataset (pooled + LORO splits) ---" | tee -a "$LOG"
$PY -m src.preprocessing.build_dataset --config "$CFG" 2>&1 | tee -a "$LOG"

echo "--- 4. LEAKAGE CHECK (must exit 0 before training) ---" | tee -a "$LOG"
if ! $PY scripts/verify_no_leakage.py --period 2021_2023 2>&1 | tee -a "$LOG"; then
  echo "LEAKAGE CHECK FAILED - aborting before any training run" | tee -a "$LOG"
  exit 1
fi

echo "--- 5. training: plain U-Net, pooled split, seeds 42/43/44 ---" | tee -a "$LOG"
for S in 42 43 44; do
  EXP="p10_pooled_unet_s${S}"
  echo "  [seed $S -> $EXP]" | tee -a "$LOG"
  $PY -m src.train --config "$TCFG" --seed $S --experiment "$EXP" 2>&1 | tee -a "$LOG"
  $PY -m src.eval.evaluate --experiment "$EXP" 2>&1 | tee -a "$LOG"
done

echo "--- 6. aggregate seeds + compare to 2019->2021 ---" | tee -a "$LOG"
$PY scripts/aggregate_p10.py 2>&1 | tee -a "$LOG"

echo "=== pipeline data + training done $(date) ===" | tee -a "$LOG"
echo "next: pick carry-forward seed (median val Dice), then infer_region / area_report / run_carbon --period 2021_2023"
