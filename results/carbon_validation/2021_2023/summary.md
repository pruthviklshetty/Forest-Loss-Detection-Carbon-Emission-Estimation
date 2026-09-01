# Phase 6/8 - Carbon Estimation (multi-region)

NDVI is from each region's Year-T (2019) composite (pre-clearing canopy). Aboveground carbon density per pixel -> tonnes C over the deforested area -> tonnes CO2 (x 3.667). The exponential regression `AGC = exp(a*NDVI + b)` is fit once on 8 Western-Ghats field-inventory anchors; the 3-bin cut points are each region's own forest terciles. Literature-calibrated, not pixel-matched biomass.

## Predicted loss on held-out test blocks (primary = exponential regression)

| Region | Area (ha) | 3-bin tCO2 | reg-linear tCO2 | reg-exp tCO2 (primary) | mean AGC (tC/ha) |
|---|---|---|---|---|---|
| wayanad | 69.4 | 33,218 | 26,933 | 24,945 | 98 |
| kodagu | 7.1 | 3,344 | 5,520 | 5,642 | 216 |
| nilgiris | 13.4 | 7,185 | 8,537 | 9,151 | 187 |
| anamalai | 23.1 | 9,874 | 14,841 | 12,927 | 153 |
| POOLED | 113.0 | 53,621 | 55,832 | 52,665 | 127 |

**Headline (pooled held-out test blocks, primary regression):** predicted loss of 113.0 ha -> **52,665 t CO2** (mean 127 tC/ha). The Hansen-GFC reference area for the same blocks gives 67,438 t CO2 (mean 176 tC/ha) - the model estimate is 0.78x the reference.

Two biases act in opposite directions: the model under-predicts the cleared *area* (1.08x: 113.0 vs 104.6 ha) but tends to over-predict the mean carbon density of the pixels it does flag (127 vs 176 tC/ha), so the CO2 ratio (0.78x) is less extreme than the area ratio. Neither the area nor the CO2 total is independent evidence of accuracy.

Figures: `results/figures/phase6_carbon_calibration.png`, `results/figures/phase6_co2_estimates.png`.
