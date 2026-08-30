# Phase 6 - Carbon Estimation

NDVI is from the Year-T (2019) composite (pre-clearing canopy). Aboveground carbon density per pixel -> tonnes C over the deforested area -> tonnes CO2 (x 3.667).

**3-bin baseline** (this study's assumed scheme, not IPCC): NDVI < 0.644 -> 100, 0.644-0.734 -> 150, >= 0.734 -> 200 tC/ha.

**Regression (primary = exponential)**: AGC = exp(6.366*NDVI + 0.510), calibrated on 8 Western-Ghats field-inventory anchors (Padmakumar et al. 2018; Kothandaraman et al. 2020). Literature-calibrated, not pixel-matched biomass; n is small.

| Pixel set | Area (ha) | 3-bin tCO2 | reg-linear tCO2 | reg-exp tCO2 (primary) |
|---|---|---|---|---|
| predicted_test | 49.9 | 24,713 | 22,604 | 19,756 |
| gfc_test | 51.5 | 25,819 | 24,800 | 21,507 |
| predicted_full_region | 279.8 | 136,145 | 119,792 | 107,962 |
| gfc_full_region | 237.4 | 116,699 | 104,726 | 94,273 |

**Headline (held-out test region, primary regression):** predicted loss of 49.9 ha -> **19,756 t CO2** (mean 108 tC/ha). Against the Hansen-GFC reference area for the same test region: 21,507 t CO2.

The regression gives a lower, NDVI-weighted estimate than the flat 3-bin scheme because most cleared pixels sit at moderate NDVI, below the 150 tC/ha 'moderate' bin constant.

Figures: `results/figures/phase6_carbon_calibration.png`, `results/figures/phase6_co2_estimates.png`.
