# Phase 7 - Evaluation, Validation & Short Paper

> All metric numbers here are the **post-leakage-audit re-run**. The
> "Leakage audit and correction" section below carries the full before/after
> and the mechanism. Pre-audit numbers also remain in git history (`c9947eb`).

## What was done

- **Final metrics table** consolidated from the Phase 3-6 result JSONs (traces
  to `results/metrics/*.json`, `results/deforestation/`,
  `results/carbon_validation/`).
- **Qualitative figures** (>= 5 representative test patches):
  `results/figures/phase3_baseline_unet_examples.png` (input T+1 false colour /
  ground truth / predicted probability),
  `phase4_compare_examples.png`, `phase5_deforestation_map.png`,
  `phase6_carbon_calibration.png`.
- **CO2 plausibility check** against Global Forest Watch's Wayanad-district
  figure -> `results/carbon_validation/co2_sanity_check.md`, summarised into
  `results/carbon_validation/summary.md`.
- **Lit-review sources** for the two carbon-regression comparators: Li et al.
  2020 (Sci Rep 10:9952), Muhammad et al. 2024 (Front. Env. Sci. 12:1448648).
- **Short paper**: `report.md`, real numbers throughout, includes the leakage
  audit as Section 5.6.

## Final numbers (post-audit, all held-out unless noted)

| Stage | Result |
|---|---|
| Segmentation - U-Net (baseline) | test IoU 0.161, Dice 0.278, P 0.318, R 0.246; best val Dice 0.245 @ e8 |
| Segmentation - Attn U-Net + MNv2 | test IoU 0.081, Dice 0.149; best val Dice 0.246 @ e36 (clearly worse on test) |
| Model gap (proposed - baseline) | test IoU -0.080, Dice -0.128 |
| vs John & Zhang 2022 | their test IoU 0.90-0.95 / F1 0.955-0.977; far higher - different task & label prevalence |
| Area - held-out test region | predicted 39.6 ha vs GFC 51.5 ha (0.77x, -23%) |
| Area - full region | predicted 164.5 ha vs GFC 237.4 ha (0.69x) |
| Carbon - test region, 3-bin baseline | 22,343 t CO2 |
| Carbon - test region, regression (primary, exponential) | 21,645 t CO2 (mean AGC 149 tC/ha) |
| Carbon - GFC reference area (full), regression vs 3-bin | 94,273 vs 116,699 t CO2 (~19% lower) |
| CO2 plausibility (GFC reference area) | pipeline factor ~397 t CO2/ha = ~60% of GFW ~665 t CO2e/ha (aboveground/CO2-only vs all pools/gases) |

## Model-selection wording (Phase 4 -> Phase 5)

Validation marginally preferred the attention model (Dice 0.2458 vs 0.2453);
the held-out test set preferred the baseline (IoU 0.161 vs 0.081). Both gaps
are within the noise of 16 val / 18 test patches, and selecting on the test set
would be selection on held-out data. The plain U-Net was carried forward for
reasons **independent of test performance**: simpler architecture, no
pretrained-RGB-encoder mismatch against the 8-band stack, fewer downstream
moving parts. Both models' test numbers are reported with equal prominence in
`results/metrics/phase4_comparison.md` and `docs/phase4_notes.md`.

## Leakage audit and correction

**Finding.** After the first full run, `scripts/verify_no_leakage.py` compared
every val/test patch's pixel extent with every train patch's. The train
split's stride-128 overlapping crops are 256 px wide but were assigned to
"train" using only the 512 px super-block containing each crop's **top-left
corner**. A crop starting at a 128 px offset from a block boundary extends into
the next block; when that block was a val or test block, 128 px strips of
held-out territory were used as training data.

- **8 of 16 validation patches and 9 of 18 test patches** received training
  pixels, each with **50-75% of its area** also present in training.
- All contamination came from overlap crops; **no canonical patch** was
  involved (the canonical grid is a clean partition).

**Fix** (`build_dataset.py`, `overlap_crop_all_train()`): an overlap crop is
kept only if its **entire** 256 x 256 footprint lies inside train-assigned
blocks; a crop touching any val/test block is dropped, not reassigned. Verified
byte-identical before/after: the canonical grid, block-to-split assignment,
`norm_stats.json` (same SHA), and every val/test patch payload. Overlap crops
228 -> 185; train patches 304 -> 261. `verify_no_leakage.py` now exits 0. Both
models retrained with byte-identical configs (seed 42, 80 epochs, lr 3e-4
cosine, batch 8, Dice+BCE `pos_weight` 40); threshold re-tuned on val; Phases 5
and 6 fully re-run.

**Before / after (held-out test split):**

| Quantity | Pre-audit (leaked) | Post-audit (clean) |
|---|---|---|
| Train patches | 304 (76 + 228 overlap) | 261 (76 + 185 overlap) |
| Val/test patches with train pixels | 8/16 val, 9/18 test | 0/16, 0/18 |
| Best val Dice - U-Net | 0.317 (@ e54) | 0.245 (@ e8) |
| Best val Dice - Attn U-Net | 0.323 (@ e63) | 0.246 (@ e36) |
| U-Net test IoU / Dice | 0.196 / 0.327 | 0.161 / 0.278 |
| Attn U-Net test IoU / Dice | 0.168 / 0.287 | 0.081 / 0.149 |
| Proposed - baseline test IoU | -0.028 | -0.080 |
| U-Net operating threshold | 0.92 | 0.88 |
| U-Net test IoU @ threshold 0.5 | 0.193 | 0.077 (collapses) |
| Predicted test-region area vs GFC | 49.9 ha (0.97x) | 39.6 ha (0.77x) |
| Predicted full-region area vs GFC | 279.8 ha (1.18x) | 164.5 ha (0.69x) |
| Predicted test-region CO2 (primary reg.) | 19,756 t | 21,645 t |
| GFC-reference-area CO2 (primary reg.) | 21,507 t | 21,507 t (unchanged) |
| Pipeline emission factor on GFC area | ~397 t CO2/ha (~60% of GFW) | ~397 t CO2/ha (~60% of GFW) |

**What the leak had done:** inflated validation Dice (partly memorised
pixels), softened the architecture comparison (the attention model's poorer
generalisation was hidden), and made the area estimate look like a near-match
when it is really a ~20-30% under-prediction. The Hansen-GFC-referenced carbon
numbers and the 3-bin-vs-regression comparison are unaffected - they never
depended on the model.

## Honest position (carried into the paper)

- The segmentation architecture is **not** a contribution: the attention model
  is clearly worse than the plain U-Net on the leak-free held-out set.
- The **model-independent** results are the ones to build on: the
  bins -> calibrated-regression carbon upgrade (~19% lower region-wide CO2,
  toward the moist-deciduous field range) and the ~60% emission-factor ratio
  against Global Forest Watch.
- The pipeline **under-predicts held-out loss area by ~23%**; the near-equal
  test-region CO2 total is a coincidence of offsetting area/density biases.
- Simplifications stated up front as scope: single sensor, one region, one
  2-year window, literature-calibrated (not pixel-matched) carbon,
  aboveground/CO2-only committed-emission accounting, 16/18-patch eval splits
  on which both models overfit, 30 m labels on a 10 m grid.
- The leakage audit and its open before/after reporting are themselves a
  methodological strength; `scripts/verify_no_leakage.py` is reusable.

## Outputs

- `report.md` - the short paper (Section 5.6 is the audit).
- `results/carbon_validation/summary.md`, `co2_sanity_check.md`.
- `results/figures/` - training curves, qualitative triptychs, deforestation
  map, carbon calibration and CO2 bar charts.
- `scripts/verify_no_leakage.py` - the audit check.
- `docs/refs/john_zhang_2022.md`, `docs/refs/carbon_refs.md` - external-value
  sources with DOIs.
