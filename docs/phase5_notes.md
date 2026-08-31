# Phase 5 - Change Detection & Area Computation

> **Numbers are off the carry-forward U-Net checkpoint = seed 43** (median best
> validation Dice; selection on validation only). Leak-free split, early
> stopping. Earlier values in git history: pre-audit 49.9 ha / 0.97x
> (`c9947eb`); post-audit single 80-epoch run 39.6 ha / 0.77x (`82f6948`).

## What was built

- [`src/change_detection/infer_region.py`](../src/change_detection/infer_region.py)
  - runs a trained model over the **whole study region**. The model consumes
  the 8-band bi-temporal stack and outputs the forest-loss (T -> T+1 change)
  mask directly, so the pixel-by-pixel "compare the two dates" step *is* one
  forward pass. Overlapping 256 px tiles (stride 128) are averaged, thresholded
  at the val-tuned operating threshold (**0.92**, seed 43), and masked to valid land.
  Writes georeferenced `results/deforestation/baseline_unet_{prob,loss}.tif`.
- [`src/change_detection/area_report.py`](../src/change_detection/area_report.py)
  - converts predicted vs Hansen-GFC loss pixels to hectares
  (10 m GSD -> 0.01 ha/pixel) and breaks it down by the Phase 2 split, so the
  held-out figure is separated from the (partly train-derived) one.

## Results - area lost 2019-2020

| Region | Hansen GFC (ha) | Predicted (ha) | Pred - GFC | Pred / GFC | pixel IoU |
|---|---|---|---|---|---|
| **test only (held out)** | **51.5** | **37.3** | **-14.2** | **0.73x** | 0.169 |
| val only | 29.8 | 26.3 | -3.5 | 0.88x | 0.149 |
| train only | 131.7 | 87.0 | -44.7 | 0.66x | 0.103 |
| full region | 237.4 | 165.7 | -71.7 | 0.70x | 0.132 |

## Honest read

- On the **held-out test region the model under-predicts area by ~27%**
  (37.3 ha predicted vs 51.5 ha reference). The aggregate ratio (0.73) is
  closer to 1 than the per-pixel IoU (0.17) would imply - false positives and
  negatives partly cancel when summed - but it is a modest improvement, not a
  near-match. See `results/figures/phase5_deforestation_map.png` (agreement
  panel): hits, misses and false alarms are scattered specks, but misses
  (recall ~0.25) outnumber false alarms, so the total comes out low.
- The under-prediction is **consistent across all four splits** (0.66-0.88x),
  which points to a genuine recall deficit at the operating threshold, not
  noise. A recall-oriented loss or a lower threshold would trade precision to
  close it.
- `train_only` and `full_region` include pixels the model trained on; even so
  they under-predict, because the leak-free model recovers far fewer loss
  pixels than the leaked one did.
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
