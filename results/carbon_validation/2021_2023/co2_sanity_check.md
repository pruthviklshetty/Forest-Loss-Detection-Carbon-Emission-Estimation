# CO2 estimate - sanity check against a published figure (2021 -> 2023 period)

*Numbers off the Phase 10 carry-forward U-Net = `p10_pooled_unet_s43` (median
best validation Dice: 0.270 / **0.311** / 0.323 over seeds 42/43/44).
Companion to `results/carbon_validation/co2_sanity_check.md` (the 2019 -> 2021
period); both are kept.*

## Published reference (Global Forest Watch / WRI) - unchanged

Global Forest Watch, Wayanad district dashboard (UMD/Hansen tree-cover loss +
the Harris et al. 2021 global forest carbon-flux model, 30% canopy threshold):

- **2001-2023 cumulative:** 3.82 kha tree-cover loss, **2.54 Mt CO2e**.
- Implied **committed emission factor: 2.54 Mt / 3.82 kha = ~665 t CO2e/ha**,
  covering **all carbon pools** (above- and below-ground biomass, deadwood,
  litter, soil) and **all gases** (CO2, CH4, N2O), as gross committed
  emissions. This cumulative 2001-2023 figure spans both study periods, so it
  is the same yardstick used for 2019 -> 2021.

## This study, 2021 -> 2023, four Western Ghats blocks

From `results/carbon_validation/2021_2023/carbon_estimates.json` (exponential
regression = primary; aboveground carbon only, committed CO2 only):

| Pixel set (pooled over 4 regions) | area (ha) | exp t CO2 | mean AGC (tC/ha) |
|---|---|---|---|
| predicted loss, held-out test blocks | 113.0 | 52,665 | 127 |
| GFC loss, held-out test blocks | 104.6 | 67,438 | 176 |
| predicted loss, full region | 732.6 | 370,040 | 138 |
| GFC loss, full region | 543.9 | 341,879 | 171 |

- **Emission factor on the GFC full-region loss area:** 341,879 t CO2 / 543.9 ha
  = **~629 t CO2/ha = ~95% of the GFW ~665 t CO2e/ha** all-pools figure. For
  comparison the 2019 -> 2021 four-region pool gave ~517 t CO2/ha (~78%) and
  Wayanad alone ~397 t CO2/ha (~60%). The 2021-22 loss sits in denser,
  higher-NDVI forest (mean AGC 171 vs the 2019-20 pool's ~141 tC/ha), which
  pushes the aboveground-only factor close to GFW's all-pools value - the ~60%
  Wayanad ratio does not generalise across periods any more than it does
  across regions.
- **Regression vs 3-bin baseline** on the GFC full-region area: exp 341,879 vs
  3-bin 279,173 t CO2 - the exponential regression is **~22% lower**, close to
  the ~19% it was lower for Wayanad alone (2019 -> 2021) and unlike the ~6%
  agreement for the 2019 -> 2021 four-region pool. The regression correction
  depends on where the cleared pixels sit on the NDVI-AGC curve, which differs
  by period.
- **Model vs reference on the same test blocks:** predicted 52,665 vs
  GFC-reference-area 67,438 t CO2 = **0.78x** - the model *under*-predicts test
  CO2 for this period, where the 2019 -> 2021 carry-forward *over*-predicted
  (predicted / GFC-ref CO2 was ~1.33x). The sign is a seed/threshold effect
  (this checkpoint's val-tuned threshold is 0.64 and it under-predicts area at
  1.08x pooled; the 2019 -> 2021 carry-forward used 0.72 and over-predicted at
  1.37x). Neither the area nor the CO2 total is independent evidence of
  accuracy.

## Position

Consistent with the 2019 -> 2021 finding: the CO2 total tracks the predicted
*area* (which is not calibrated - pooled pred/GFC 1.08x here, 1.37x there) times
a per-hectare density that is literature-calibrated, not pixel-matched. The
model-independent number worth carrying is the emission factor's fraction of
the GFW all-pools value, and even that is period- and region-dependent
(~60% Wayanad 2019-20, ~78% four-region 2019-20, ~95% four-region 2021-22).
