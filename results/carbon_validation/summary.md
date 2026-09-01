# Phase 6/8 - Carbon Estimation (multi-region)

NDVI is from each region's Year-T (2019) composite (pre-clearing canopy). Aboveground carbon density per pixel -> tonnes C over the deforested area -> tonnes CO2 (x 3.667). The exponential regression `AGC = exp(a*NDVI + b)` is fit once on 8 Western-Ghats field-inventory anchors; the 3-bin cut points are each region's own forest terciles. Literature-calibrated, not pixel-matched biomass.

## Predicted loss on held-out test blocks (primary = exponential regression)

| Region | Area (ha) | 3-bin tCO2 | reg-linear tCO2 | reg-exp tCO2 (primary) | mean AGC (tC/ha) |
|---|---|---|---|---|---|
| wayanad | 78.7 | 42,920 | 45,599 | 39,305 | 136 |
| kodagu | 48.1 | 18,722 | 12,721 | 12,633 | 72 |
| nilgiris | 33.0 | 18,456 | 23,048 | 21,270 | 176 |
| anamalai | 33.2 | 18,135 | 26,565 | 24,997 | 205 |
| POOLED | 193.0 | 98,234 | 107,932 | 98,205 | 139 |

**Headline (pooled held-out test blocks, primary regression):** predicted loss of 193.0 ha -> **98,205 t CO2** (mean 139 tC/ha). The Hansen-GFC reference area for the same blocks gives 74,005 t CO2 (mean 143 tC/ha) - the model estimate is 1.33x the reference.

Two biases act in opposite directions: the model under-predicts the cleared *area* (1.37x: 193.0 vs 140.8 ha) but tends to over-predict the mean carbon density of the pixels it does flag (139 vs 143 tC/ha), so the CO2 ratio (1.33x) is less extreme than the area ratio. Neither the area nor the CO2 total is independent evidence of accuracy.

Figures: `results/figures/phase6_carbon_calibration.png`, `results/figures/phase6_co2_estimates.png`.
