# Phase 7 - Evaluation, Validation & Short Paper

> Metric history: pre-leakage-audit single run (`c9947eb`) -> post-audit single
> 80-epoch run (`82f6948`) -> early stopping (`defa73f`) -> 3-seed protocol
> (this state). Segmentation headlines are **mean +/- sd over seeds 42/43/44**;
> area / CO2 are off the **carry-forward U-Net = seed 43** (median best
> validation Dice). Every superseded number is preserved in a before/after
> table below and in git history.

## What was done

- **Final metrics** consolidated from `results/metrics/seed_runs.json`,
  the per-seed `*_s*.json`, `results/deforestation/`, `results/carbon_validation/`.
- **Qualitative figures** (>= 5 test patches):
  `phase3_baseline_unet_examples.png`, `phase4_compare_examples.png`,
  `phase5_deforestation_map.png`, `phase6_carbon_calibration.png`.
- **CO2 plausibility check** vs Global Forest Watch's Wayanad-district figure
  -> `results/carbon_validation/co2_sanity_check.md`.
- **Lit-review carbon-regression comparators**: Li et al. 2020 (Sci Rep
  10:9952), Muhammad et al. 2024 (Front. Env. Sci. 12:1448648).
- **Short paper**: `report.md`. Section 5.6 is the leakage audit, Section 5.7
  the seed-variance analysis.

## Final numbers

### Segmentation - held-out test (18 patches), mean +/- sd over 3 seeds

| Metric | U-Net (baseline) | Attn U-Net + MNv2 |
|---|---|---|
| test IoU | **0.158 +/- 0.016** | 0.113 +/- 0.023 |
| test Dice / F1 | 0.273 +/- 0.024 | 0.203 +/- 0.038 |
| test precision | 0.332 +/- 0.018 | 0.206 +/- 0.031 |
| test recall | 0.231 +/- 0.026 | 0.202 +/- 0.052 |
| best val Dice | 0.250 +/- 0.006 | 0.237 +/- 0.009 |

Per-seed test IoU: U-Net 0.165 / 0.170 / 0.140; Attn 0.128 / 0.086 / 0.125.
Mean +/- 1 sd intervals: U-Net [0.142, 0.174], Attn [0.090, 0.136] -
**do not overlap** -> U-Net advantage **supported** at that criterion
(Welch t = 2.75, p ~ 0.06; n = 3).

### Pipeline - carry-forward U-Net (seed 43, op threshold 0.92)

| Stage | Result |
|---|---|
| Area - held-out test region | predicted 37.3 ha vs GFC 51.5 ha (0.73x, -27%) |
| Area - full region | predicted 165.7 ha vs GFC 237.4 ha (0.70x) |
| Carbon - test region, 3-bin baseline | 19,938 t CO2 |
| Carbon - test region, regression (primary, exponential) | 17,918 t CO2 (0.83x GFC-ref-area; mean AGC 131 tC/ha) |
| Carbon - GFC ref area (full), regression vs 3-bin | 94,273 vs 116,699 t CO2 (~19% lower) - model/seed-independent |
| CO2 plausibility (GFC reference area) | ~397 t CO2/ha = ~60% of GFW ~665 t CO2e/ha - model/seed-independent |

## Model-selection rule (Phase 4 -> Phase 5)

The carry-forward checkpoint is the plain U-Net seed with the **median best
validation Dice**: values 0.244 (s42), 0.252 (s43), 0.255 (s44) -> **seed 43**.
Selection is on validation only; test metrics are never used. Seed 43's test
scores (IoU 0.170, Dice 0.290) are recorded for traceability. The plain U-Net
rather than the attention model carries forward for reasons independent of test
score: simpler architecture, no pretrained-RGB-encoder mismatch against the
8-band stack, fewer downstream moving parts. Both models' per-seed results have
equal prominence in `results/metrics/seed_runs.json`, `report.md` Section 5.1
and `results/metrics/phase4_comparison.md`.

## Leakage audit and correction

**Finding.** `scripts/verify_no_leakage.py` compared every val/test patch's
pixel extent with every train patch's. Stride-128 overlap crops (256 px wide)
were assigned to "train" using only the 512 px super-block of each crop's
top-left corner; a crop at a 128 px offset from a block boundary extended into
the next block, and where that block was val/test, 128 px strips of held-out
territory became training data. **8/16 validation and 9/18 test patches** were
affected, each with 50-75% of its area also in training. All from overlap
crops; no canonical patch involved.

**Fix** (`build_dataset.py`, `overlap_crop_all_train()`): keep an overlap crop
only if its **entire** 256 x 256 footprint lies in train-assigned blocks; a
crop touching any val/test block is dropped, not reassigned. Verified
byte-identical before/after: canonical grid, block-to-split assignment,
`norm_stats.json` (same SHA), every val/test patch payload. Overlap crops
228 -> 185; train patches 304 -> 261. `verify_no_leakage.py` exits 0.

## Seed-variance analysis

Post-audit, Section 5.6's own numbers showed run-to-run test-IoU swings of
~0.03, so a single training run per model still could not support the
comparison. Each model was retrained under **3 seeds (42/43/44)** with early
stopping, otherwise byte-identical configs; `scripts/aggregate_seeds.py` ->
`results/metrics/seed_runs.json`.

- **Seed sd on test IoU is 0.016 (U-Net) / 0.023 (Attn) - comparable to the
  mean U-Net - Attn gap (0.045).** Any single-run comparison of the two models
  is unfalsifiable; headline metrics are therefore mean +/- sd.
- The mean +/- 1 sd test-IoU intervals do **not** overlap -> the U-Net's
  advantage is **supported** by that criterion. It is a lenient bar: with n = 3
  a Welch t-test gives t = 2.75, p ~ 0.06 - probably real, not established at
  p < 0.05.
- **Early stopping confirmed rather than resolved the overfitting.** Best
  validation Dice lands at epochs 1/7/8 (U-Net seeds) and 15/17/35 (Attn seeds)
  - always early, regardless of the 80-epoch `T_max`. A direct consequence of
  261 training patches at ~0.3% positive prevalence.

## Before / after (held-out test split)

| Quantity | Pre-audit (leaked, 1 run) | Post-audit 1 run (`82f6948`) | Now: 3-seed / seed 43 |
|---|---|---|---|
| Train patches | 304 (76 + 228 ov) | 261 (76 + 185 ov) | 261 |
| Val/test w/ train pixels | 8/16, 9/18 | 0/16, 0/18 | 0/16, 0/18 |
| U-Net best val Dice | 0.317 | 0.245 | 0.250 +/- 0.006 |
| U-Net test IoU / Dice | 0.196 / 0.327 | 0.161 / 0.278 | **0.158 +/- 0.016 / 0.273 +/- 0.024** |
| Attn U-Net test IoU / Dice | 0.168 / 0.287 | 0.081 / 0.149 | **0.113 +/- 0.023 / 0.203 +/- 0.038** |
| Proposed - baseline test IoU | -0.028 | -0.080 | -0.045 (intervals non-overlapping) |
| Predicted test area vs GFC | 49.9 ha (0.97x) | 39.6 ha (0.77x) | **37.3 ha (0.73x)** |
| Predicted full-region area vs GFC | 279.8 ha (1.18x) | 164.5 ha (0.69x) | 165.7 ha (0.70x) |
| Predicted test CO2 (primary reg.) | 19,756 t | 21,645 t | **17,918 t** |
| GFC-ref-area CO2 (primary reg.) | 21,507 t | 21,507 t | 21,507 t (unchanged) |
| Emission factor on GFC area | ~397 t CO2/ha (~60% GFW) | ~397 (~60%) | ~397 (~60%) (unchanged) |

## Honest position (in the paper)

- The segmentation architecture is **not** a contribution: the attention model
  scores below the plain U-Net, with non-overlapping +/-1 sd intervals.
- **Model/seed-independent results** are the ones to build on: the
  bins -> regression carbon upgrade (~19% lower region-wide CO2) and the ~60%
  emission-factor ratio vs GFW.
- The pipeline **under-predicts held-out loss area by ~27%** and CO2 by ~17%;
  the CO2 error is smaller only because area and density biases partly offset.
- Simplifications stated up front as scope: single sensor, one region, one
  2-year window, literature-calibrated (not pixel-matched) carbon,
  aboveground/CO2-only committed emission, 16/18-patch eval splits on which
  both models overfit within a few epochs, 30 m labels on a 10 m grid,
  run-to-run seed variance ~ the architecture effect size.
- The leakage audit, the seed protocol and their open before/after reporting
  are themselves methodological strengths;
  `scripts/verify_no_leakage.py` and `scripts/aggregate_seeds.py` are reusable.

## Outputs

- `report.md` (Sections 5.6 audit, 5.7 seed variance).
- `results/metrics/seed_runs.json`, per-seed `*_s*.json`.
- `results/carbon_validation/summary.md`, `co2_sanity_check.md`.
- `results/figures/` - training curves, triptychs, deforestation map, carbon
  calibration / CO2 bars.
- `scripts/verify_no_leakage.py`, `scripts/aggregate_seeds.py`.
- `docs/refs/john_zhang_2022.md`, `docs/refs/carbon_refs.md`.
