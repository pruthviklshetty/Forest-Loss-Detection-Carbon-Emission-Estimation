# Phase 5 - Change Detection & Area Computation

## What was built

- [`src/change_detection/infer_region.py`](../src/change_detection/infer_region.py)
  - runs a trained model over the **whole study region**. The model consumes
  the 8-band bi-temporal stack and outputs the forest-loss (T -> T+1 change)
  mask directly, so the pixel-by-pixel "compare the two dates" step *is* one
  forward pass. Overlapping 256 px tiles (stride 128) are averaged, thresholded
  at the val-tuned operating threshold (**0.92**), and masked to valid land.
  Writes georeferenced `results/deforestation/baseline_unet_{prob,loss}.tif`.
- [`src/change_detection/area_report.py`](../src/change_detection/area_report.py)
  - converts predicted vs Hansen-GFC loss pixels to hectares
  (10 m GSD -> 0.01 ha/pixel) and breaks it down by the Phase 2 split, so the
  held-out figure is separated from the train-contaminated one. Writes the
  summary and two figures.

## Results - area lost 2019-2020

| Region | Hansen GFC (ha) | Predicted (ha) | Pred - GFC | Pred / GFC | pixel IoU |
|---|---|---|---|---|---|
| **test only (held out)** | **51.5** | **49.9** | **-1.6** | **0.97x** | 0.200 |
| val only | 29.8 | 34.2 | +4.4 | 1.15x | 0.199 |
| train only | 131.7 | 173.3 | +41.6 | 1.32x | 0.390 |
| full region | 237.4 | 279.8 | +42.4 | 1.18x | 0.313 |

`train_only` and `full_region` include pixels the model trained on and
overstate agreement (high IoU, inflated area). **The honest number is
`test_only`.**

## Honest read

- On the **held-out test region the aggregate area estimate is very good**:
  49.9 ha predicted vs 51.5 ha reference, a 3% under-estimate. The model's
  per-pixel IoU is only ~0.20 (it localises loss roughly, as scattered blobs),
  but on the test split false positives and false negatives nearly cancel, so
  the hectares total lands close. See `results/figures/phase5_deforestation_map.png`
  (agreement panel): hits, misses and false alarms are all small scattered
  specks with no systematic spatial bias.
- This is the intended message of the pipeline: even a weak segmenter yields a
  usable hectares-lost number when aggregated over an area. It should not be
  over-read - `val_only` (+15%) shows the cancellation is not guaranteed on
  every subset, and both val and test are small (~17 patches / ~30-50 ha).
- The predicted raster is thresholded at the fixed val-tuned 0.92; no
  per-region recalibration was done.

## Outputs

- `results/deforestation/baseline_unet_prob.tif`, `baseline_unet_loss.tif`
  (georeferenced, git-ignored - regenerate with `infer_region.py`)
- `results/deforestation/baseline_unet_area_summary.json`
- `results/deforestation/summary.md`
- `results/figures/phase5_deforestation_map.png`
- `results/figures/phase5_hectares.png`

## Needed before Phase 6

Nothing external. Phase 6 = carbon module: 3-bin NDVI baseline (documented as
this study's assumed scheme) + an NDVI->carbon-density regression calibrated
against a small literature reference table, then apply both to the Phase 5
hectares to get tonnes CO2.
