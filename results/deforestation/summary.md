# Phase 5 - Change Detection & Area Computation

Model: **baseline_unet**. Model choice is independent of test performance: validation was a near-tie between the two models, and the plain U-Net was carried forward for a simpler architecture with no pretrained-RGB-encoder mismatch against the 8-band stack. One forward pass on the 8-band bi-temporal stack yields the newly-deforested mask directly; overlapping 256 px tiles (stride 128) are averaged, thresholded at the val-tuned **0.92**, and masked to valid land.

Pixel -> area: 10 m GSD, 0.01 ha per pixel.

Pixel IoU is reported strict (primary) and tolerance (+/-3 px GFC-cell GT dilation, strict union; secondary, consistent with `src/eval/evaluate.py`).

| Region | GFC ref (ha) | Predicted (ha) | Pred - GFC (ha) | Pred/GFC | strict IoU | tol IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| test_only | 51.5 | 37.3 | -14.2 | 0.725 | 0.169 | 0.260 | 0.289 | 0.344 | 0.249 |
| val_only | 29.8 | 26.3 | -3.4 | 0.885 | 0.149 | 0.219 | 0.259 | 0.276 | 0.244 |
| train_only | 131.7 | 87.0 | -44.7 | 0.66 | 0.103 | 0.175 | 0.187 | 0.235 | 0.155 |
| canonical_all | 213.0 | 150.6 | -62.3 | 0.707 | 0.125 | 0.202 | 0.223 | 0.269 | 0.190 |
| full_region | 237.4 | 165.7 | -71.7 | 0.698 | 0.132 | 0.208 | 0.233 | 0.284 | 0.198 |

**Headline (held-out test region):** predicted **37.3 ha** vs Hansen GFC **51.5 ha** (0.72x; -14.2 ha), strict pixel IoU 0.169 (tolerance 0.260).

`full_region` and `train_only` include pixels the model was trained on and overstate agreement; `test_only` is the honest number.

Figures: `results/figures/phase5_deforestation_map.png`, `results/figures/phase5_hectares.png`.
