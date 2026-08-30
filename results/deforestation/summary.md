# Phase 5 - Change Detection & Area Computation

Model: **baseline_unet** (chosen over the Attention U-Net on test IoU/Dice). One forward pass on the 8-band bi-temporal stack yields the newly-deforested mask directly; overlapping 256 px tiles (stride 128) are averaged, thresholded at the val-tuned **0.92**, and masked to valid land.

Pixel -> area: 10 m GSD, 0.01 ha per pixel.

| Region | GFC ref (ha) | Predicted (ha) | Pred - GFC (ha) | Pred/GFC | IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| test_only | 51.5 | 49.9 | -1.6 | 0.969 | 0.200 | 0.333 | 0.339 | 0.328 |
| val_only | 29.8 | 34.2 | +4.4 | 1.148 | 0.199 | 0.332 | 0.310 | 0.356 |
| train_only | 131.7 | 173.3 | +41.6 | 1.316 | 0.390 | 0.561 | 0.494 | 0.650 |
| canonical_all | 213.0 | 257.4 | +44.4 | 1.208 | 0.317 | 0.481 | 0.439 | 0.531 |
| full_region | 237.4 | 279.8 | +42.5 | 1.179 | 0.313 | 0.477 | 0.441 | 0.520 |

**Headline (held-out test region):** predicted **49.9 ha** vs Hansen GFC **51.5 ha** (0.97x; -1.6 ha), pixel IoU 0.200.

`full_region` and `train_only` include pixels the model was trained on and overstate agreement; `test_only` is the honest number.

Figures: `results/figures/phase5_deforestation_map.png`, `results/figures/phase5_hectares.png`.
