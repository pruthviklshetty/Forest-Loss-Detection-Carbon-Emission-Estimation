# Phase 5 - Change Detection & Area Computation

> **Numbers below are the post-leakage-audit re-run** (see the audit section in
> `docs/phase7_notes.md`). Pre-audit: predicted test area 49.9 ha (0.97x GFC),
> full region 279.8 ha (1.18x). Preserved in git history (`c9947eb`).

## What was built

- [`src/change_detection/infer_region.py`](../src/change_detection/infer_region.py)
  - runs a trained model over the **whole study region**. The model consumes
  the 8-band bi-temporal stack and outputs the forest-loss (T -> T+1 change)
  mask directly, so the pixel-by-pixel "compare the two dates" step *is* one
  forward pass. Overlapping 256 px tiles (stride 128) are averaged, thresholded
  at the val-tuned operating threshold (**0.88**), and masked to valid land.
  Writes georeferenced `results/deforestation/baseline_unet_{prob,loss}.tif`.
- [`src/change_detection/area_report.py`](../src/change_detection/area_report.py)
  - converts predicted vs Hansen-GFC loss pixels to hectares
  (10 m GSD -> 0.01 ha/pixel) and breaks it down by the Phase 2 split, so the
  held-out figure is separated from the (partly train-derived) one.

## Results - area lost 2019-2020

| Region | Hansen GFC (ha) | Predicted (ha) | Pred - GFC | Pred / GFC | pixel IoU |
|---|---|---|---|---|---|
| **test only (held out)** | **51.5** | **39.6** | **-11.9** | **0.77x** | 0.161 |
| val only | 29.8 | 26.0 | -3.7 | 0.87x | 0.145 |
| train only | 131.7 | 86.5 | -45.3 | 0.66x | 0.100 |
| full region | 237.4 | 164.5 | -72.8 | 0.69x | 0.125 |

## Honest read

- On the **held-out test region the model under-predicts area by ~23%**
  (39.6 ha predicted vs 51.5 ha reference). The aggregate ratio (0.77) is
  closer to 1 than the per-pixel IoU (0.16) would imply - errors partly cancel
  when summed - but this is a modest improvement, not the near-match the
  pre-audit run showed (0.97x). See `results/figures/phase5_deforestation_map.png`
  (agreement panel): hits, misses and false alarms are scattered specks, but
  misses (recall ~0.25) outnumber false alarms, so the total comes out low.
- The under-prediction is **consistent across all four splits** (0.66-0.87x),
  which points to a genuine recall deficit at the operating threshold, not
  noise. A recall-oriented loss or a lower threshold would trade precision to
  close it.
- `train_only` and `full_region` include pixels the model trained on; even so
  they now *under*-predict, because the leak-free model recovers far fewer loss
  pixels than the leaked one did.
- The predicted raster is thresholded at the fixed val-tuned 0.88; no
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
