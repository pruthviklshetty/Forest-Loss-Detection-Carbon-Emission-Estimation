# Phase 6 - Carbon Estimation Module

## What was built

- [`src/carbon/ndvi.py`](../src/carbon/ndvi.py) - NDVI, and the **3-bin
  baseline** (sparse/moderate/dense -> 100/150/200 tC/ha). Explicitly labelled
  *this study's assumed classification scheme* - **not** attributed to IPCC.
  NDVI cut points are the forest tercile boundaries of the Year-T composite
  (0.639 / 0.726).
- [`src/carbon/regression_model.py`](../src/carbon/regression_model.py) - the
  **contribution**: an NDVI -> continuous aboveground-carbon-density regression.
  Two fits, `linear` and `exponential` (primary). Calibrated on **8
  Western-Ghats field-inventory anchors** built from published regional carbon
  densities - **not** a pixel-aligned GEDI/GFW raster download.
- [`src/carbon/run_carbon.py`](../src/carbon/run_carbon.py) - applies both
  schemes to the Year-T (pre-clearing) NDVI of the deforested pixels and
  converts tonnes C -> tonnes CO2 (x 44/12).
- [`data/carbon/reference_table.csv`](../data/carbon/reference_table.csv) +
  [`docs/refs/carbon_refs.md`](refs/carbon_refs.md) - the calibration table
  with full citations.

## Calibration (real)

Anchors pair a published Western-Ghats aboveground carbon density (AGB x 0.47,
IPCC 2006 default carbon fraction) with the matching forest-cover NDVI
percentile of this study's own Year-T composite:

| Class | NDVI | AGC (tC/ha) | Source |
|---|---|---|---|
| dry / open forest | 0.446 | 30.5 | Padmakumar et al. 2018 |
| degraded moist deciduous | 0.583 | 75.6 | Kothandaraman et al. 2020 |
| moist deciduous | 0.666 | 101.8 | Kothandaraman et al. 2020 |
| semi-evergreen | 0.719 | 132.5 | Kothandaraman et al. 2020 |
| semi-evergreen (dense) | 0.746 | 171.3 | Kothandaraman et al. 2020 |
| evergreen | 0.775 | 236.0 | Kothandaraman et al. 2020 |
| evergreen (dense) | 0.819 | 332.9 | Kothandaraman et al. 2020 |
| evergreen (old-growth) | 0.838 | 408.1 | Kothandaraman et al. 2020 |

- **linear:** AGC = 874.1 x NDVI - 425.0, r2 = 0.769 (clipped at 0; poor below
  NDVI ~0.49).
- **exponential (primary):** AGC = exp(6.366 x NDVI + 0.510), r2 = 0.948;
  monotone and non-negative by construction.
- r2 is curve fit on 8 points, **not** held-out accuracy - stated, not hidden.

## Results - tonnes CO2 from 2019-2020 forest loss

| Pixel set | Area (ha) | 3-bin | reg-linear | **reg-exp (primary)** |
|---|---|---|---|---|
| **predicted, test region** | 49.9 | 24,713 | 22,604 | **19,756** |
| Hansen GFC, test region (ref) | 51.5 | 25,819 | 24,800 | **21,507** |
| predicted, full region | 279.8 | 136,145 | 119,792 | 107,962 |
| Hansen GFC, full region (ref) | 237.4 | 116,699 | 104,726 | 94,273 |

**Headline (held-out test region, primary regression):** 49.9 ha of predicted
new loss -> **~19,800 t CO2** (mean 108 tC/ha aboveground). Against the
Hansen-GFC reference area for the same test region: ~21,500 t CO2 - the
prediction is ~8% low, tracking the Phase 5 area ratio (0.97x) plus NDVI
weighting.

## Honest read

- The **regression sits ~20-25% below the flat 3-bin scheme** for this
  region's loss, because most cleared pixels are moderate-NDVI (mean 0.59) and
  the 3-bin "moderate" constant (150 tC/ha) over-credits them; the
  continuous curve places them near 100-110 tC/ha, closer to the moist-
  deciduous field values.
- This is the intended contribution: a continuous, regionally-calibrated,
  vegetation-density-aware estimate replacing three hard-coded constants.
- Simplifications (stated as scope): literature-calibrated not pixel-matched;
  8 anchors; aboveground carbon only; committed-emission accounting (full
  release, no regrowth credit); single sensor / region / window.

## Needed before Phase 7

Nothing new. Phase 7 = finalise the metrics table, 5+ qualitative triptychs,
sanity-check the test-region CO2 figure against one published FAO / Global
Forest Watch value for the area, and write the short paper.
