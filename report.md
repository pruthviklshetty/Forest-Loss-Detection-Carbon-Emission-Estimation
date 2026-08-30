# An Integrated Deep-Learning Pipeline for Forest-Loss Detection and Vegetation-Density-Aware Carbon-Emission Estimation from Bi-temporal Sentinel-2 Imagery

*Short paper / workshop manuscript. All quantitative values are produced by the
scripts in this repository and are reproducible from the committed configs and
data-provenance files; no numbers are placeholders.*

---

## Abstract

We present an end-to-end pipeline that takes two dated Sentinel-2 composites
and returns a hectares-lost and a tonnes-CO2 figure for a study region in one
system. Forest loss between the two dates is segmented directly from an 8-band
bi-temporal stack with an attention-gated U-Net (MobileNetV2 encoder) and a
plain U-Net baseline, trained and evaluated under an identical protocol on a
Western Ghats (Wayanad, Kerala) tile with Hansen Global Forest Change (GFC)
labels for calendar 2019-2020. The predicted change mask is converted to
hectares with the 10 m ground sample distance, and each cleared pixel's
pre-clearing NDVI is mapped to an aboveground carbon density with (i) a coarse
3-bin baseline and (ii) a continuous regression calibrated against published
Western Ghats field-inventory carbon densities. On the held-out test region the
plain U-Net reaches IoU 0.20 / Dice 0.33; the attention model does not improve
on it under the shared schedule (IoU 0.17 / Dice 0.29). Despite the modest
pixel-level accuracy, the **aggregate area estimate is within 3% of the GFC
reference on the held-out region** (49.9 ha predicted vs 51.5 ha), and the
regression yields ~19,800 t CO2 for that region versus ~24,700 t CO2 from the
3-bin scheme. The per-hectare emission factor is ~60% of the Global Forest
Watch all-pools figure for Wayanad, consistent with an aboveground-only,
CO2-only accounting. The contribution is the **integrated, honestly-validated
pipeline** and the **bins-to-regression carbon upgrade**, not the segmentation
architecture.

---

## 1. Introduction

Operational deforestation monitoring and its translation into carbon-emission
estimates are usually handled by separate systems: a change-detection product
(e.g. Hansen GFC) on one side, and a carbon-flux model (e.g. Global Forest
Watch / Harris et al. 2021) on the other, each with its own inputs, resolution
and assumptions. For a specific study region and a specific two-year window, a
researcher who wants "how many hectares were lost and how much CO2 did that
release" must stitch these together by hand.

This paper builds that stitch as a single reproducible pipeline and validates
it honestly, end to end, at a scope that is actually finishable: **one sensor
(Sentinel-2), one region (a Wayanad plateau tile), one window (2019 -> 2021
dry-season composites), two segmentation models, and a carbon step that is
upgraded from fixed bins to a calibrated regression.**

**What is *not* claimed as novel.** Attention U-Net for deforestation
segmentation is established by John and Zhang (2022); MobileNetV2 encoders for
satellite segmentation are standard; NDVI-based carbon binning is weaker than
regression approaches calibrated on biomass data. The genuine contributions
are:

1. an integrated raw-imagery -> change-mask -> hectares -> tonnes-CO2 pipeline
   with every stage validated against an external reference, and
2. replacing three hard-coded NDVI carbon constants with a continuous,
   regionally-calibrated, vegetation-density-aware regression.

---

## 2. Related work

**Deforestation segmentation.** *Ronneberger et al. (2015)* introduced the
U-Net encoder-decoder with skip connections that underlies most forest-cover
segmentation work. *Oktay et al. (2018)* added additive attention gates on the
skips ("Attention U-Net"), letting the decoder suppress irrelevant encoder
activations. *Sandler et al. (2018)* proposed MobileNetV2, the inverted-
residual backbone we use as a pretrained encoder. *John and Zhang (2022)*
combined these ideas for deforestation, applying an Attention U-Net to
single-image Sentinel-2 semantic segmentation of deforestation polygons in the
Amazon and Atlantic Forest, reporting test F1 of 0.955-0.977 and IoU of
0.90-0.95. **This work differs** in three ways that matter for the numbers:
(a) the task is *bi-temporal change* (detect what was cleared *between* two
dates) rather than segmenting already-visible clearing in one image; (b) the
positive class here is ~0.3% of pixels (fragmented smallholder / plantation
loss with 30 m GFC labels) versus their abundant, hand-digitised positive
class; (c) both of our models share one training schedule for a clean ablation,
where John and Zhang tuned learning rate and epochs per model.

**Forest-cover reference data.** *Hansen et al. (2013)* produced the Global
Forest Change product; its `lossyear` raster is a year-of-loss code, which we
convert to a binary T-vs-T+1 mask as a documented sub-step. *Harris et al.
(2021)* built the global forest carbon-flux layers that Global Forest Watch
serves at the admin level; we use its Wayanad figure only as an external
plausibility check.

**Carbon / biomass regression from optical data.** *Li et al. (2020)* compared
linear regression, random forest and XGBoost on Landsat-8 + Sentinel-1A
predictors (including NDVI) for subtropical aboveground biomass, with a best
R2 of 0.75 (RMSE 20.9 Mg/ha). *Muhammad et al. (2024)* used Sentinel-2 bands
and 44 vegetation indices (including NDVI) with random forest for dry-temperate
biomass, best R2 0.83 (RMSE ~30 Mg/ha). Both show that a regression on
spectral / index predictors is the standard, and that it clearly outperforms
discrete binning. **This work differs** by *not* fitting to co-located biomass
plots (which were unavailable and are the highest-risk data task): instead the
regression is calibrated against a small table of *published* regional
aboveground carbon densities, with the limitation stated rather than hidden.
The Western Ghats field inventories underpinning that table are *Padmakumar et
al. (2018)*, *Kothandaraman et al. (2020)* and *Jose et al. (2025)*.

---

## 3. Study area and data

**Region.** A ~848 km2 tile of the Wayanad plateau, Western Ghats, Kerala,
India; bounding box `[76.00, 11.55, 76.28, 11.80]` (WGS84), working projection
UTM 43N (EPSG:32643), 10 m grid (3064 x 2778 px). The landscape is a mosaic of
semi-evergreen and moist-deciduous forest, plantation and smallholder
agriculture; forest loss here is fragmented, which is a deliberately harder
setting than Amazon arc-of-deforestation clearing.

**Imagery.** Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`), two dry-season
windows: **T = 1 Jan - 15 Apr 2019** (84 scenes) and **T+1 = 1 Jan - 15 Apr
2021** (84 scenes). Per window, a cloud-masked (Cloud Score+ `cs_cdf >= 0.60`)
per-band median composite of bands B3/B4/B8/B11 -> green / red / NIR / SWIR1,
scaled to [0,1] reflectance; SWIR1 is bilinearly resampled 20 m -> 10 m. Blue
is dropped (haze-dominated at 10 m); SWIR1 is kept for its canopy-moisture /
bare-soil contrast. The two windows are stacked into an 8-band tensor
`[T_g, T_r, T_nir, T_swir1, T1_g, T1_r, T1_nir, T1_swir1]`. 0.57% of pixels are
persistently cloudy and are zero-filled and marked invalid (excluded from loss
and metrics).

**Labels.** Hansen GFC `UMD/hansen/global_forest_change_2024_v1_12`. Conversion:
`forest2000 = treecover2000 >= 30%`; `land = datamask == 1`;
`loss_window = lossyear in {19, 20}` (calendar 2019 and 2020, which fall
between the two acquisitions); `label = 1` where all three hold. This yields
**23,736 positive pixels = 237.4 ha** of forest loss over the tile, a 0.37%
positive rate within forest2000.

**Patches and split.** 110 non-overlapping 256 x 256 patches; a spatially
blocked 70/15/15 split (2x2-patch super-blocks, seed 42) gives 76 / 16 / 18
canonical patches. Train is enlarged to 304 patches with stride-128 overlapping
crops inside train blocks only; val and test stay non-overlapping. Per-band
train statistics are used for optional z-scoring. Class imbalance is severe by
construction and is handled at training time.

---

## 4. Methods

### 4.1 Segmentation

Both models take the 8-band stack and output a single-channel forest-loss logit
map; the T-vs-T+1 comparison is thus learned in one forward pass.

- **Baseline U-Net** (Ronneberger-style): double-conv blocks, max-pool down,
  transpose-conv up, 4 skip levels, `base_channels = 32`, 7.76 M parameters. No
  attention, no pretrained encoder.
- **Proposed Attention U-Net + MobileNetV2**: an ImageNet-pretrained
  MobileNetV2 encoder (from `segmentation_models.pytorch`, adapted to 8 input
  channels) with a hand-built decoder in which every skip passes through an
  Oktay additive attention gate whose gating signal is the upsampled coarser
  decoder feature. 6.70 M parameters.

**Shared training protocol** (`configs/train_baseline.yaml` ==
`configs/train_attention.yaml` for data, optimiser, loss and seed): 80 epochs,
Adam, lr 3e-4 cosine-annealed, batch 8, mixed precision, gradient clip 1.0,
loss = soft-Dice + BCE with `pos_weight = 40`, all operations masked by the
per-pixel `valid` map. The operating threshold is chosen on the validation
split by maximising Dice over a `[0.10, 0.98]` sweep; metrics are then computed
once on the held-out test split.

### 4.2 Change detection and area

The chosen model is run over the whole tile with overlapping 256 px tiles
(stride 128, probabilities averaged), thresholded at the val-tuned value, and
masked to valid land. Pixel counts convert to hectares at 0.01 ha/pixel (10 m
GSD). Numbers are reported per split so the held-out figure is separable from
the train-contaminated full-region figure.

### 4.3 Carbon estimation

For each cleared pixel we take its **Year-T (pre-clearing) NDVI** = (NIR - red)
/ (NIR + red).

- **3-bin baseline (this study's assumed scheme, not from an IPCC table):**
  NDVI < 0.639 -> 100 tC/ha; 0.639-0.726 -> 150 tC/ha; >= 0.726 -> 200 tC/ha.
  Cut points are the forest tercile boundaries of the Year-T composite.
- **Regression (contribution):** eight calibration anchors pair a published
  Western Ghats aboveground carbon density (AGB x 0.47, IPCC 2006 default
  carbon fraction; from Padmakumar et al. 2018 and Kothandaraman et al. 2020,
  with Jose et al. 2025 as an unfitted cross-check) with the matching
  forest-cover NDVI percentile of this study's own Year-T composite. A linear
  fit (`AGC = 874.1 * NDVI - 425.0`, R2 0.77, clipped at 0) is reported as the
  simplest variant; an **exponential fit** (`AGC = exp(6.366 * NDVI + 0.510)`,
  R2 0.95, monotone and non-negative) is the primary model. R2 describes curve
  fit on eight anchors, not held-out accuracy. This is explicitly a
  *literature-calibrated* mapping, not a fit to co-located biomass.

Aboveground carbon density -> tonnes C over the cleared area -> tonnes CO2 (x
44/12). The accounting is aboveground-only, CO2-only, committed emission (full
release, no regrowth credit).

---

## 5. Results

### 5.1 Segmentation (held-out test split, 18 patches)

| Model | Params | Op. thr | IoU | Dice / F1 | Precision | Recall | Pixel acc. |
|---|---|---|---|---|---|---|---|
| **U-Net (baseline)** | 7.76 M | 0.92 | **0.196** | **0.327** | 0.323 | 0.331 | 0.9940 |
| Attention U-Net + MNv2 | 6.70 M | 0.94 | 0.168 | 0.287 | 0.340 | 0.249 | 0.9946 |
| delta (proposed - baseline) | | | -0.028 | -0.040 | +0.017 | -0.082 | +0.001 |

Pixel accuracy is ~0.994 for both and is uninformative (99.6% of valid pixels
are negative). Under the shared schedule the attention model is marginally
ahead on validation (Dice 0.323 vs 0.317) but behind on test, trading recall
for precision; with 16 / 18 patches per split the honest reading is **no
measurable architecture benefit in this setting**, consistent with John and
Zhang's own small Attention-vs-U-Net F1 gains (+0.002 to +0.018). All
subsequent stages use the plain U-Net.

Against John and Zhang (2022) (test IoU 0.90-0.95, F1 0.955-0.977), our IoU/F1
are ~3-5x lower. This is expected and not a like-for-like failure: different
task (bi-temporal change vs single-image segmentation), ~0.3% vs abundant
positive prevalence, 30 m GFC labels vs hand-digitised polygons, and fragmented
smallholder loss vs Amazon clear-cutting.

### 5.2 Change detection and area

| Region | GFC reference (ha) | Predicted (ha) | Pred / GFC | Pixel IoU |
|---|---|---|---|---|
| **test (held out)** | **51.5** | **49.9** | **0.97** | 0.200 |
| val | 29.8 | 34.2 | 1.15 | 0.199 |
| train | 131.7 | 173.3 | 1.32 | 0.390 |
| full region | 237.4 | 279.8 | 1.18 | 0.313 |

On the held-out region the **aggregate area estimate is within 3%** of the GFC
reference even though per-pixel IoU is only 0.20: false positives and false
negatives are small scattered specks with no systematic spatial bias, so they
nearly cancel in the total. This does not generalise unconditionally - the
validation region runs +15% - and both splits are small (~30-50 ha).

### 5.3 Carbon and CO2

| Pixel set | Area (ha) | 3-bin (t CO2) | reg-linear (t CO2) | reg-exp / primary (t CO2) |
|---|---|---|---|---|
| **predicted, test region** | 49.9 | 24,713 | 22,604 | **19,756** |
| GFC reference, test region | 51.5 | 25,819 | 24,800 | 21,507 |
| predicted, full region | 279.8 | 136,145 | 119,792 | 107,962 |
| GFC reference, full region | 237.4 | 116,699 | 104,726 | 94,273 |

**Held-out headline:** 49.9 ha of predicted new loss -> **~19,800 t CO2**
(mean aboveground carbon density 108 tC/ha), versus ~21,500 t CO2 on the GFC
reference area for the same region. The regression sits ~20-25% below the flat
3-bin scheme because most cleared pixels have moderate NDVI (mean 0.59) and the
3-bin "moderate" constant (150 tC/ha) over-credits them; the continuous curve
places them near 100-110 tC/ha, closer to the moist-deciduous field values.

### 5.4 CO2 plausibility check (Global Forest Watch, Wayanad district)

GFW reports 3.82 kha of tree-cover loss and 2.54 Mt CO2e for Wayanad district
(2001-2023, 30% canopy, all pools, all gases; Harris et al. 2021) -> ~173 ha/yr
and ~110,400 t CO2e/yr for the whole district. This study's 848 km2 tile shows
~119 ha/yr (GFC) / ~140 ha/yr (predicted) and ~47,000-54,000 t CO2/yr
(aboveground, CO2-only). The tile therefore accounts for roughly two-thirds of
the district's loss area on ~40% of its area (it is centred on the loss-active
plateau), and its per-hectare emission factor (~386-397 t CO2/ha) is **~60% of
GFW's ~638 t CO2e/ha** - the expected fraction once soil, belowground biomass
and non-CO2 gases (all in GFW, none here) are accounted for. Every residual
difference maps to a stated scope choice.

### 5.5 Qualitative

`results/figures/phase3_baseline_unet_examples.png` shows input (T+1 false
colour) / ground truth / predicted probability for six representative test
patches; `results/figures/phase4_compare_examples.png` adds the attention
model; `results/figures/phase5_deforestation_map.png` shows the full-region
predicted-loss map and the hit / miss / false-alarm agreement panel against
GFC; `results/figures/phase6_carbon_calibration.png` shows the NDVI -> carbon
curves.

---

## 6. Discussion and scope

These choices are **scope decisions taken up front**, not limitations
discovered afterward:

- **Single sensor.** Sentinel-2 only; no Landsat or SAR fusion.
- **One region, one window.** A single ~848 km2 Wayanad tile, 2019 -> 2021.
  Depth over breadth; the numbers should not be read as regionally
  transferable.
- **Literature-calibrated carbon, not pixel-matched biomass.** The regression
  is anchored to eight published regional carbon densities and this scene's own
  NDVI percentiles, not to co-located biomass plots or a GEDI/GFW raster.
  Its R2 is a curve fit, not predictive accuracy.
- **Aboveground, CO2-only, committed emission.** No belowground / deadwood /
  litter / soil pools, no non-CO2 gases, no regrowth credit.
- **Small evaluation set.** 16 validation and 18 test patches make threshold
  selection and the val-vs-test comparison noisy; the area estimate's
  near-cancellation of errors is demonstrated, not guaranteed.
- **Coarse labels.** Hansen GFC is a 30 m annual product used on a 10 m grid;
  it bounds achievable IoU well below values reported for hand-digitised
  benchmarks.

The pipeline's value is that despite a per-pixel IoU near 0.2, it produces an
**area estimate within 3% and a CO2 estimate within a well-understood factor**
of independent references on held-out data - i.e. a weak segmenter is still
useful once aggregated and honestly bounded.

---

## 7. Conclusion

We built and validated, end to end, a single pipeline from two Sentinel-2
composites to a hectares-lost and a tonnes-CO2 figure for a Western Ghats tile,
and upgraded the carbon step from three hard-coded NDVI constants to a
regionally-calibrated regression. The attention-gated MobileNetV2 model gave no
test-set gain over a plain U-Net under a fair shared schedule, echoing the
small architecture effect in the literature. The held-out area estimate is
within 3% of Hansen GFC and the CO2 estimate is a sensible fraction of the
Global Forest Watch figure for Wayanad. Natural next steps: co-located biomass
calibration, multi-region evaluation, and a larger annotated change set.

---

## References

1. Chave, J. et al. (2014). Improved allometric models to estimate the
   aboveground biomass of tropical trees. *Global Change Biology* 20(10),
   3177-3190.
2. Hansen, M.C. et al. (2013). High-resolution global maps of 21st-century
   forest cover change. *Science* 342(6160), 850-853.
3. Harris, N.L. et al. (2021). Global maps of twenty-first century forest
   carbon fluxes. *Nature Climate Change* 11, 234-240.
4. IPCC (2006). 2006 IPCC Guidelines for National Greenhouse Gas Inventories,
   Volume 4 (AFOLU), Chapter 4, Table 4.3 (default carbon fraction 0.47).
5. John, D. and Zhang, C. (2022). An attention-based U-Net for detecting
   deforestation within satellite sensor imagery. *International Journal of
   Applied Earth Observation and Geoinformation* 107, 102685.
6. Jose, K. et al. (2025). Woody species diversity, structure, and carbon stock
   in a tropical semi-evergreen forest in Western Ghats, India. *Environmental
   Research Communications* 7(4), 045027.
7. Kothandaraman, S. et al. (2020). Ecosystem-level carbon storage and its
   links to diversity, structural and environmental drivers in tropical forests
   of Western Ghats, India. *Scientific Reports* 10, 13444.
8. Li, Y., Li, M., Li, C. and Liu, Z. (2020). Forest aboveground biomass
   estimation using Landsat 8 and Sentinel-1A data with machine learning
   algorithms. *Scientific Reports* 10, 9952.
9. Muhammad, B. et al. (2024). Estimation of above-ground biomass in dry
   temperate forests using Sentinel-2 data and random forest: a case study of
   the Swat area of Pakistan. *Frontiers in Environmental Science* 12, 1448648.
10. Oktay, O. et al. (2018). Attention U-Net: learning where to look for the
    pancreas. *MIDL 2018* / arXiv:1804.03999.
11. Padmakumar, B. et al. (2018). Tree biomass and carbon density estimation in
    the tropical dry forest of Southern Western Ghats, India. *iForest -
    Biogeosciences and Forestry* 11(4), 534-541.
12. Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: convolutional
    networks for biomedical image segmentation. *MICCAI 2015*, 234-241.
13. Sandler, M. et al. (2018). MobileNetV2: inverted residuals and linear
    bottlenecks. *CVPR 2018*, 4510-4520.

---

## Reproducibility

Phase-by-phase notes and every real number are in `docs/phase{1..7}_notes.md`;
provenance JSON/CSV in `data/*/manifest.json`, `data/processed/split.json`,
`results/metrics/*.json`, `results/deforestation/`, `results/carbon_validation/`.
Each phase is a separate git commit. Reference-value sources with DOIs are in
`docs/refs/`.
