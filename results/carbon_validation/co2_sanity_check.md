# CO2 estimate - sanity check against a published figure

*Numbers off the carry-forward U-Net = seed 43 (median best validation Dice).
Superseded versions in git history: pre-audit (`c9947eb`), post-audit single
run (`82f6948`).*

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

| Quantity | GFC-labelled area | Model-predicted area (U-Net seed 43) |
|---|---|---|
| Forest loss, 2-yr total | 237.4 ha | 165.7 ha |
| Forest loss, per year | 118.7 ha/yr | 82.9 ha/yr |
| CO2, primary regression, 2-yr total | 94,273 t | 82,261 t |
| CO2, primary regression, per year | 47,137 t/yr | 41,131 t/yr |
| Implied emission factor | **~397 t CO2/ha** | ~496 t CO2/ha |
| Mean aboveground carbon density | 108 tC/ha | 135 tC/ha |

(aboveground carbon only; committed emission; CO2 only.)

## Comparison and honest delta

- **Area.** The 848 km2 tile (~40% of the district area) carries ~119 ha/yr of
  GFC loss vs the district's ~166 ha/yr - i.e. ~70% of the district's forest
  loss on 40% of its area, consistent with the tile being centred on the
  loss-active plateau. The model predicts ~83 ha/yr, a ~30% under-count of the
  tile's own GFC loss.
- **Emission factor on the GFC reference area.** ~397 t CO2/ha (aboveground,
  CO2-only) is **~60% of GFW's ~665 t CO2e/ha**. Expected: aboveground biomass
  is typically 55-75% of total forest carbon in moist tropical/subtropical
  forest, and GFW additionally counts soil and non-CO2 gases. This number
  depends only on GFC and the regression - not the segmentation model, the
  seed or the leakage fix.
- **Emission factor on the model-predicted area.** ~496 t CO2/ha - elevated by
  the model's density bias: it preferentially flags denser, higher-NDVI forest
  as loss (mean AGC 135 vs the GFC area's 108 tC/ha). The predicted 2-year CO2
  total (82,261 t) is 0.87x the GFC-reference-area total (94,273 t); the CO2
  ratio is less extreme than the area ratio (0.70x) because the density
  over-count partly offsets the area under-count.
- **Other differences** (not corrected for): GFC v1.12 vs the GFW dashboard
  version; a fixed 2019-2020 window vs a 22-year average; simple
  committed-emission accounting vs the Harris et al. (2021) model.

## Verdict

On the **GFC reference area** the pipeline's per-hectare emission factor is a
sensible ~60% of the published all-pools figure - a genuine plausibility
result, independent of the model and seed. The **model-predicted** area (~-30%)
and CO2 total (~-13%) both under-shoot, with the CO2 shortfall smaller because
area and density biases partly offset. This is a plausibility check, not a
validation against ground truth.
