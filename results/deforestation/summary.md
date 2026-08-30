# Phase 5 - Change Detection & Area Computation

Model: **baseline_unet**. Model choice is independent of test performance: validation was a near-tie between the two models, and the plain U-Net was carried forward for a simpler architecture with no pretrained-RGB-encoder mismatch against the 8-band stack. One forward pass on the 8-band bi-temporal stack yields the newly-deforested mask directly; overlapping 256 px tiles (stride 128) are averaged, thresholded at the val-tuned **0.88**, and masked to valid land.

Pixel -> area: 10 m GSD, 0.01 ha per pixel.

| Region | GFC ref (ha) | Predicted (ha) | Pred - GFC (ha) | Pred/GFC | IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| test_only | 51.5 | 39.6 | -11.9 | 0.768 | 0.161 | 0.277 | 0.318 | 0.245 |
| val_only | 29.8 | 26.0 | -3.7 | 0.874 | 0.145 | 0.254 | 0.272 | 0.238 |
| train_only | 131.7 | 86.5 | -45.3 | 0.656 | 0.100 | 0.182 | 0.230 | 0.151 |
| canonical_all | 213.0 | 152.0 | -60.9 | 0.714 | 0.122 | 0.217 | 0.260 | 0.186 |
| full_region | 237.4 | 164.5 | -72.8 | 0.693 | 0.125 | 0.222 | 0.271 | 0.188 |

**Headline (held-out test region):** predicted **39.6 ha** vs Hansen GFC **51.5 ha** (0.77x; -11.9 ha), pixel IoU 0.161.

`full_region` and `train_only` include pixels the model was trained on and overstate agreement; `test_only` is the honest number.

Figures: `results/figures/phase5_deforestation_map.png`, `results/figures/phase5_hectares.png`.
