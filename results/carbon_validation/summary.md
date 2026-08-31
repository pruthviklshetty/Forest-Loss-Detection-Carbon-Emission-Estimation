# Phase 6 - Carbon Estimation

NDVI is from the Year-T (2019) composite (pre-clearing canopy). Aboveground carbon density per pixel -> tonnes C over the deforested area -> tonnes CO2 (x 3.667).

**3-bin baseline** (this study's assumed scheme, not IPCC): NDVI < 0.644 -> 100, 0.644-0.734 -> 150, >= 0.734 -> 200 tC/ha.

**Regression (primary = exponential)**: AGC = exp(6.366*NDVI + 0.510), calibrated on 8 Western-Ghats field-inventory anchors (Padmakumar et al. 2018; Kothandaraman et al. 2020). Literature-calibrated, not pixel-matched biomass; n is small.

| Pixel set | Area (ha) | 3-bin tCO2 | reg-linear tCO2 | reg-exp tCO2 (primary) |
|---|---|---|---|---|
| predicted_test | 37.3 | 19,938 | 21,235 | 17,918 |
| gfc_test | 51.5 | 25,819 | 24,800 | 21,507 |
| predicted_full_region | 165.7 | 89,314 | 94,541 | 82,261 |
| gfc_full_region | 237.4 | 116,699 | 104,726 | 94,273 |

**Headline (held-out test region, primary regression):** predicted loss of 37.3 ha -> **17,918 t CO2** (mean 131 tC/ha). The Hansen-GFC reference area for the same region gives 21,507 t CO2 (mean 114 tC/ha) - the model estimate is 0.83x the reference.

Two biases act in opposite directions: the model under-predicts the cleared *area* (0.72x: 37.3 vs 51.5 ha) but over-predicts the mean carbon density of the pixels it does flag (131 vs 114 tC/ha, because it favours denser higher-NDVI forest), so the CO2 ratio (0.83x) is less extreme than the area ratio. Neither the area nor the CO2 total is independent evidence of accuracy. The model-free comparison is 3-bin vs regression on the same pixels: on the GFC reference area the exponential regression gives 94,273 t CO2 vs 116,699 t from the 3-bin scheme (~19% lower), because most cleared pixels sit at moderate NDVI where the flat 150 tC/ha 'moderate' constant over-credits them.

Figures: `results/figures/phase6_carbon_calibration.png`, `results/figures/phase6_co2_estimates.png`.

## Phase 7 - external plausibility check

Sanity-checked against Global Forest Watch's Wayanad-district figure (3.82 kha
loss, 2.54 Mt CO2e over 2001-2023; Harris et al. 2021, all pools and gases ->
implied factor ~665 t CO2e/ha). On the GFC reference area this pipeline's
aboveground-only, CO2-only emission factor is 94,273 / 237.4 = ~397 t CO2/ha,
i.e. ~60% of GFW's ~665 t CO2e/ha - the expected fraction, and independent of
the segmentation model. On the model-predicted area the factor is ~496
t CO2/ha (82,261 / 165.7), elevated by the model's density bias. Full working:
`results/carbon_validation/co2_sanity_check.md`.
