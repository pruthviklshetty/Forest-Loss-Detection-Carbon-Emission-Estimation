# CO2 estimate - sanity check against a published figure

## Published reference (Global Forest Watch / WRI)

Global Forest Watch, Wayanad district dashboard (UMD/Hansen tree-cover loss +
the Harris et al. 2021 global forest carbon-flux model, 30% canopy threshold):

- **2001-2023 cumulative:** 3.82 kha tree-cover loss, **2.54 Mt CO2e**.
- Implied **annual average: ~173 ha/yr, ~110,400 t CO2e/yr** for the whole
  Wayanad district (~2,131 km2). GFW's figure covers **all carbon pools**
  (above- and below-ground biomass, deadwood, litter, soil) and **all gases**
  (CO2, CH4, N2O), as gross committed emissions.

Sources: globalforestwatch.org/dashboards/country/IND/17/14 ; Harris, N.L. et
al. (2021) "Global maps of twenty-first century forest carbon fluxes",
*Nature Climate Change* 11, 234-240.

## This study (Wayanad plateau tile, ~848 km2, 2019-2020)

| Quantity | GFC-labelled area | Model-predicted area |
|---|---|---|
| Forest loss, 2-yr total | 237.4 ha | 279.8 ha |
| Forest loss, per year | 118.7 ha/yr | 139.9 ha/yr |
| CO2, primary regression, 2-yr total | 94,273 t | 107,962 t |
| CO2, primary regression, per year | 47,137 t/yr | 53,981 t/yr |
| Implied emission factor | ~397 t CO2/ha | ~386 t CO2/ha |

(aboveground carbon only; committed emission; CO2 only.)

## Comparison and honest delta

- **Area.** This 848 km2 tile (~40% of the district area) shows ~119 ha/yr of
  GFC loss vs the district's ~173 ha/yr average - i.e. the tile captures
  roughly two-thirds of the district's forest loss on 40% of its area,
  consistent with it being deliberately centred on the loss-active plateau.
- **Emission factor.** This study's ~386-397 t CO2/ha (aboveground, CO2-only)
  is **~60% of GFW's ~638 t CO2e/ha** (110,400 / 173). That ratio is expected:
  aboveground biomass is typically 55-75% of total forest carbon in
  moist tropical/subtropical forest, and GFW additionally counts soil and
  non-CO2 gases. Scaling this study's aboveground figure by a mid-range
  root:shoot + soil multiplier would close most of the gap.
- **Other differences** (not corrected for): GFC v1.12 vs the GFW dashboard
  version; a fixed 2019-2020 window vs a 22-year average; simple
  committed-emission accounting vs the Harris et al. (2021) model; the model's
  ~18% area over-prediction on the full region (it is only +/-3% on the
  held-out test split).

## Verdict

The pipeline's CO2 estimate is **the right order of magnitude** and its
per-hectare emission factor is a sensible fraction of the published all-pools
figure, with every difference attributable to a stated scope choice
(aboveground-only, CO2-only, one tile, one 2-year window, literature-calibrated
carbon). It is a plausibility check, not a validation against ground truth.
