# Phase 7 - Evaluation, Validation & Short Paper

## What was done

- **Final metrics table** consolidated from the Phase 3-6 result JSONs (no
  re-runs; all numbers trace to `results/metrics/*.json`,
  `results/deforestation/`, `results/carbon_validation/`).
- **Qualitative figures** (>= 5 representative test patches): reused
  `results/figures/phase3_baseline_unet_examples.png` (6 patches: input T+1
  false colour / ground truth / predicted probability),
  `phase4_compare_examples.png`, `phase5_deforestation_map.png`,
  `phase6_carbon_calibration.png`.
- **CO2 plausibility check** against Global Forest Watch's Wayanad-district
  figure -> `results/carbon_validation/co2_sanity_check.md`, summarised into
  `results/carbon_validation/summary.md`.
- **Lit review sources** for the two carbon-regression comparators pulled and
  cited: Li et al. 2020 (Sci Rep 10:9952), Muhammad et al. 2024 (Front. Env.
  Sci. 12:1448648).
- **Short paper**: `report.md` (workshop length), real numbers throughout, no
  placeholders.

## Final numbers (all held-out unless noted)

| Stage | Result |
|---|---|
| Segmentation - U-Net (baseline) | test IoU 0.196, Dice 0.327, P 0.323, R 0.331 |
| Segmentation - Attn U-Net + MNv2 | test IoU 0.168, Dice 0.287 (no gain; shared schedule) |
| vs John & Zhang 2022 | their test IoU 0.90-0.95 / F1 0.955-0.977; ~3-5x higher, different task & label prevalence |
| Area - held-out test region | predicted 49.9 ha vs GFC 51.5 ha (0.97x) |
| Area - full region | predicted 279.8 ha vs GFC 237.4 ha (1.18x; train-contaminated) |
| Carbon - test region, 3-bin baseline | 24,713 t CO2 |
| Carbon - test region, regression (primary, exponential) | 19,756 t CO2 (mean 108 tC/ha) |
| Carbon - GFC reference area, regression | 21,507 t CO2 |
| CO2 plausibility | study factor ~386-397 t CO2/ha = ~60% of GFW ~638 t CO2e/ha (aboveground/CO2-only vs all-pools/all-gases) |

## Honest position (carried into the paper)

- The segmentation architecture is **not** a contribution: the attention model
  did not beat the plain U-Net under a fair shared schedule.
- The contributions are the **integrated, externally-checked pipeline** and the
  **bins -> calibrated-regression carbon upgrade**.
- Simplifications are stated up front as scope: single sensor, one region, one
  2-year window, literature-calibrated (not pixel-matched) carbon,
  aboveground/CO2-only committed-emission accounting, small (16/18-patch)
  eval splits, 30 m labels on a 10 m grid.
- The pipeline's demonstrated strength is that a weak per-pixel segmenter
  (IoU ~0.2) still yields an **area estimate within 3%** and a **CO2 estimate
  within a well-understood ~0.6x factor** of independent references on
  held-out data.

## Outputs

- `report.md` - the short paper.
- `results/carbon_validation/summary.md`, `co2_sanity_check.md`.
- `results/figures/` - training curves, qualitative triptychs, deforestation
  map, carbon calibration and CO2 bar charts.
- `docs/refs/john_zhang_2022.md`, `docs/refs/carbon_refs.md` - external-value
  sources with DOIs.
