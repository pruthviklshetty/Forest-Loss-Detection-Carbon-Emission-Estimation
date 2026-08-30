# Carbon-density reference table - sources

The Phase 6 NDVI -> carbon-density regression is calibrated against
**literature-reported regional aboveground carbon densities**, not
pixel-aligned biomass rasters. Each anchor pairs a published Western-Ghats
field-inventory carbon density with the matching forest-cover NDVI percentile
of this study's own Year-T Sentinel-2 composite. Table: `data/carbon/reference_table.csv`.

Aboveground carbon density is taken as AGB x 0.47 (IPCC 2006 Guidelines for
National GHG Inventories, Vol. 4, Ch. 4, Table 4.3 default carbon fraction of
oven-dry matter), applied uniformly so the anchors are comparable.

## Sources

1. **Padmakumar, B., Sreekanth, N.P., Shanthiprabha, V., Paul, J., Sreedharan,
   K., Augustine, T., Jayasooryan, K.K., Rameshan, M., Mohan, M., Ramasamy,
   E.V., Thomas, A.P. (2018).** Tree biomass and carbon density estimation in
   the tropical dry forest of Southern Western Ghats, India. *iForest -
   Biogeosciences and Forestry* 11(4): 534-541.
   https://doi.org/10.3832/ifor2190-011
   - Chinnar Wildlife Sanctuary, Kerala. 8 plots x 0.1 ha. Carbon fraction 0.47.
   - Mean AGB 64.13 Mg/ha; mean aboveground carbon **30.46 tC/ha**.
   - Used as the low-NDVI / dry-open-forest anchor.

2. **Kothandaraman, S., Dar, J.A., Sundarapandian, S., Dayanandan, S., Khan,
   M.L. (2020).** Ecosystem-level carbon storage and its links to diversity,
   structural and environmental drivers in tropical forests of Western Ghats,
   India. *Scientific Reports* 10: 13444.
   https://doi.org/10.1038/s41598-020-70313-6
   - Kanyakumari Wildlife Sanctuary, Tamil Nadu. 70 plots x 0.04 ha.
   - Reported AGB by forest type (Mg/ha): TDD I 160.8, TDD II 216.7, TSE II
     282.0, TSE I 364.5, TEF I 502.1, TEF II 708.2, TEF III 868.2.
   - Aboveground carbon at CF 0.47: ~75.6 / 101.8 / 132.5 / 171.3 / 236.0 /
     332.9 / 408.1 tC/ha - the six mid-to-high NDVI anchors.
   - (The paper's own "tree carbon" figures use CF 0.4453 on AGB+BGB and are
     therefore higher/not directly comparable; we recompute from their AGB.)

3. **Jose, K., Najeeb, N., Suryawanshi, K., Hebbalalu, S.S., Page, N.,
   Chaturvedi, R.K. (2025).** Woody species diversity, structure, and carbon
   stock in a tropical semi-evergreen forest in Western Ghats, India.
   *Environmental Research Communications* 7(4): 045027.
   https://doi.org/10.1088/2515-7620/adcdd0
   - Netravali Wildlife Sanctuary, Goa. 1.08 ha plot. Carbon fraction 0.5.
   - AGB 289.4 Mg/ha; aboveground biomass carbon 128.87 tC/ha (CF 0.5), ~136
     tC/ha at CF 0.47.
   - Held out of the fit; used only as a cross-check that the semi-evergreen
     anchors (~130-170 tC/ha) are in the right range.

## Stated simplifications

- Calibrated to regional literature means, not biomass measured at the study
  pixels - so the curve captures the NDVI->carbon *trend* for this forest
  region, not local pixel-level biomass.
- 8 calibration anchors: the reported r2 describes curve fit, not held-out
  predictive accuracy.
- Aboveground only. Belowground biomass, deadwood, litter and soil carbon are
  excluded; committed-emission accounting assumes the aboveground carbon of a
  cleared pixel is fully emitted, with no regrowth credit.
