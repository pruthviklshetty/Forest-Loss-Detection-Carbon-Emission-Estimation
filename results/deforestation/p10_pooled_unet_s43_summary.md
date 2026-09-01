# Phase 5/8 - Change Detection & Area Computation (multi-region)

Model: **p10_pooled_unet_s43**. One forward pass on the 8-band bi-temporal stack yields the newly-deforested mask directly; overlapping 256 px tiles (stride 128) are averaged, thresholded at the val-tuned **0.64**, and masked to valid land. Pixel -> area: 10 m GSD, 0.01 ha per pixel. Strict pixel IoU is primary; tolerance IoU (+/-3 px GFC-cell GT dilation, strict union) is secondary, consistent with `src/eval/evaluate.py`.

## Held-out test blocks, per region

| Region | GFC ref (ha) | Predicted (ha) | Pred - GFC (ha) | Pred/GFC | strict IoU | tol IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| wayanad | 44.5 | 69.4 | +24.9 | 1.559 | 0.158 | 0.274 | 0.273 | 0.224 | 0.349 |
| kodagu | 21.1 | 7.1 | -14.0 | 0.338 | 0.099 | 0.167 | 0.180 | 0.357 | 0.121 |
| nilgiris | 14.3 | 13.4 | -1.0 | 0.932 | 0.093 | 0.204 | 0.171 | 0.177 | 0.165 |
| anamalai | 24.6 | 23.1 | -1.6 | 0.937 | 0.291 | 0.375 | 0.451 | 0.466 | 0.437 |
| **POOLED** | 104.6 | 112.9 | +8.3 | 1.08 | 0.168 | 0.270 | 0.287 | 0.276 | 0.298 |

**Headline (pooled held-out test blocks):** predicted **112.9 ha** vs Hansen GFC **104.6 ha** (1.08x; +8.3 ha), strict pixel IoU 0.168 (tolerance 0.270).

`full_region` and `train_only` include pixels the model was trained on and overstate agreement; the test blocks are the honest number.

Figures: `results/figures/phase5_deforestation_map.png`, `results/figures/phase5_hectares.png`.
