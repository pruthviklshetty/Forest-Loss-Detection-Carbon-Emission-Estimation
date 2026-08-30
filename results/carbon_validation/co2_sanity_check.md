# CO2 estimate - sanity check against a published figure

*Post-leakage-audit numbers. The pre-audit version of this file is in git
history (commit c9947eb). See `docs/phase7_notes.md` for the before/after.*

## Published reference (Global Forest Watch / WRI)

Global Forest Watch, Wayanad district dashboard (UMD/Hansen tree-cover loss +
the Harris et al. 2021 global forest carbon-flux model, 30% canopy threshold):

- **2001-2023 cumulative:** 3.82 kha tree-cover loss, **2.54 Mt CO2e**.
- Implied **committed emission factor: 2.54 Mt / 3.82 kha = ~665 t CO2e/ha**,
  and ~166 ha/yr of loss for the whole Wayanad district (~2,131 km2). GFW's
  figure covers **all carbon pools** (above- and below-ground biomass,
  deadwood, litter, soil) and **all gases** (CO2, CH4, N2O), as gross
  committed emissions.

Sources: globalforestwatch.org/dashboards/country/IND/17/14 ; Harris, N.L. et
al. (2021) "Global maps of twenty-first century forest carbon fluxes",
*Nature Climate Change* 11, 234-240.

## This study (Wayanad plateau tile, ~848 km2, 2019-2020)

| Quantity | GFC-labelled area | Model-predicted area (leak-free) |
|---|---|---|
| Forest loss, 2-yr total | 237.4 ha | 164.5 ha |
| Forest loss, per year | 118.7 ha/yr | 82.3 ha/yr |
| CO2, primary regression, 2-yr total | 94,273 t | 94,778 t |
| CO2, primary regression, per year | 47,137 t/yr | 47,389 t/yr |
| Implied emission factor | **~397 t CO2/ha** | ~576 t CO2/ha |
| Mean aboveground carbon density | 108 tC/ha | 157 tC/ha |

(aboveground carbon only; committed emission; CO2 only.)

## Comparison and honest delta

- **Area.** The 848 km2 tile (~40% of the district area) carries ~119 ha/yr of
  GFC loss vs the district's ~166 ha/yr - i.e. ~70% of the district's forest
  loss on 40% of its area, consistent with the tile being centred on the
  loss-active plateau. The leak-free model predicts only ~82 ha/yr, a ~30%
  under-count of the tile's own GFC loss.
- **Emission factor on the GFC reference area.** ~397 t CO2/ha (aboveground,
  CO2-only) is **~60% of GFW's ~665 t CO2e/ha**. That ratio is expected:
  aboveground biomass is typically 55-75% of total forest carbon in moist
  tropical/subtropical forest, and GFW additionally counts soil and non-CO2
  gases. This number does not depend on the segmentation model and was
  unchanged by the leakage fix.
- **Emission factor on the model-predicted area.** ~576 t CO2/ha - much closer
  to GFW, but for the wrong reason: the leak-free model preferentially flags
  denser, higher-NDVI forest (mean AGC 157 vs the GFC area's 108 tC/ha), which
  inflates the per-hectare factor. The near-equal 2-year CO2 totals (94,778 vs
  94,273 t) are a coincidence of an area under-count offsetting a density
  over-count, not a validation.
- **Other differences** (not corrected for): GFC v1.12 vs the GFW dashboard
  version; a fixed 2019-2020 window vs a 22-year average; simple
  committed-emission accounting vs the Harris et al. (2021) model.

## Verdict

On the **GFC reference area** the pipeline's per-hectare emission factor is a
sensible ~60% of the published all-pools figure - a genuine plausibility
result. The **model-predicted** CO2 total also lands near the reference, but
through offsetting biases (area low, density high), so it is not independent
evidence of accuracy. This is a plausibility check, not a validation against
ground truth.
