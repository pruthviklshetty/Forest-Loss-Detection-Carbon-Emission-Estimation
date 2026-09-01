# Phase 5/8 - Change Detection & Area Computation (multi-region)

Model: **p8_pooled_unet_s44**. One forward pass on the 8-band bi-temporal stack yields the newly-deforested mask directly; overlapping 256 px tiles (stride 128) are averaged, thresholded at the val-tuned **0.72**, and masked to valid land. Pixel -> area: 10 m GSD, 0.01 ha per pixel. Strict pixel IoU is primary; tolerance IoU (+/-3 px GFC-cell GT dilation, strict union) is secondary, consistent with `src/eval/evaluate.py`.

## Held-out test blocks, per region

| Region | GFC ref (ha) | Predicted (ha) | Pred - GFC (ha) | Pred/GFC | strict IoU | tol IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| wayanad | 51.5 | 78.7 | +27.2 | 1.529 | 0.174 | 0.275 | 0.297 | 0.246 | 0.376 |
| kodagu | 29.9 | 48.1 | +18.2 | 1.609 | 0.113 | 0.375 | 0.204 | 0.165 | 0.266 |
| nilgiris | 17.4 | 33.0 | +15.6 | 1.899 | 0.091 | 0.207 | 0.167 | 0.128 | 0.243 |
| anamalai | 42.0 | 33.2 | -8.8 | 0.791 | 0.424 | 0.564 | 0.595 | 0.674 | 0.533 |
| **POOLED** | 140.7 | 193.0 | +52.3 | 1.372 | 0.193 | 0.343 | 0.323 | 0.279 | 0.383 |

**Headline (pooled held-out test blocks):** predicted **193.0 ha** vs Hansen GFC **140.7 ha** (1.372x; +52.3 ha), strict pixel IoU 0.193 (tolerance 0.343).

`full_region` and `train_only` include pixels the model was trained on and overstate agreement; the test blocks are the honest number.

Figures: `results/figures/phase5_deforestation_map.png`, `results/figures/phase5_hectares.png`.
