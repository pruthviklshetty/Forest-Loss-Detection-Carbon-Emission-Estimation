# An Integrated Deep-Learning Pipeline for Forest-Loss Detection and Vegetation-Density-Aware Carbon-Emission Estimation from Bi-temporal Sentinel-2 Imagery

*Short paper / workshop manuscript. All quantitative values are produced by the
scripts in this repository and are reproducible from the committed configs and
data-provenance files; no numbers are placeholders. Section 5.6 documents a
data-leakage audit and the full re-run that followed it.*

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
plain U-Net reaches **IoU 0.16 / Dice 0.28**; the attention model is clearly
worse under the shared schedule (**IoU 0.08 / Dice 0.15**). The pipeline
**under-predicts the held-out test area by ~23%** (39.6 ha vs a 51.5 ha GFC
reference) - closer to correct than the 0.16 pixel IoU alone suggests, but not
a near-match. The primary regression yields ~21,600 t CO2 for the predicted
test-region loss (vs ~22,300 t from the 3-bin scheme); this lands within ~1% of
the GFC-reference-area figure, but through offsetting errors (area low, mean
carbon density high), so it is a coincidence rather than a validation. On the
GFC reference area the pipeline's per-hectare emission factor is ~60% of the
Global Forest Watch all-pools figure for Wayanad, the expected fraction for an
aboveground-only, CO2-only accounting. The contributions are the **integrated,
honestly-audited pipeline** and the **bins-to-regression carbon upgrade**, not
the segmentation architecture.

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
upgraded from fixed bins to a calibrated regression.** A leakage audit late in
the project (Section 5.6) forced a full re-training and re-evaluation; the
before/after comparison is reported openly.

**What is *not* claimed as novel.** Attention U-Net for deforestation
segmentation is established by John and Zhang (2022); MobileNetV2 encoders for
satellite segmentation are standard; NDVI-based carbon binning is weaker than
regression approaches calibrated on biomass data. The genuine contributions
are:

1. an integrated raw-imagery -> change-mask -> hectares -> tonnes-CO2 pipeline
   with each stage checked against an external reference, and
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
blocked 70/15/15 split (2x2-patch = 512 px super-blocks, seed 42) gives 76 / 16
/ 18 canonical patches. The train split is enlarged with stride-128 overlapping
crops, **but only crops whose entire 256 x 256 footprint lies inside
train-assigned blocks are kept** (see Section 5.6): 185 such crops, for 261
train patches. Validation and test stay as the canonical non-overlapping
patches, byte-identical before and after the audit. Per-band train statistics
(from the 76 canonical train patches) are available for optional z-scoring.
Class imbalance is severe by construction and is handled at training time.

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
per-pixel `valid` map. The checkpoint is the epoch of best validation Dice; the
operating threshold is then chosen on validation by maximising Dice over a
`[0.10, 0.98]` sweep. Metrics are computed once on the held-out test split.

### 4.2 Change detection and area

The chosen model is run over the whole tile with overlapping 256 px tiles
(stride 128, probabilities averaged), thresholded at the val-tuned value, and
masked to valid land. Pixel counts convert to hectares at 0.01 ha/pixel (10 m
GSD). Numbers are reported per split so the held-out figure is separable from
the (partly train-derived) full-region figure.

### 4.3 Carbon estimation

For each cleared pixel we take its **Year-T (pre-clearing) NDVI** = (NIR - red)
/ (NIR + red).

- **3-bin baseline (this study's assumed scheme, not from an IPCC table):**
  NDVI < 0.644 -> 100 tC/ha; 0.644-0.734 -> 150 tC/ha; >= 0.734 -> 200 tC/ha.
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
  *literature-calibrated* mapping, not a fit to co-located biomass. The
  calibration is independent of the train/val/test split and was unchanged by
  the audit.

Aboveground carbon density -> tonnes C over the cleared area -> tonnes CO2 (x
44/12). The accounting is aboveground-only, CO2-only, committed emission (full
release, no regrowth credit).

---

## 5. Results

All numbers below are from the **post-audit re-run** (leak-free split; see
Section 5.6 for the before/after comparison).

### 5.1 Segmentation (held-out test split, 18 patches)

| Model | Params | Op. thr | IoU | Dice / F1 | Precision | Recall | Pixel acc. |
|---|---|---|---|---|---|---|---|
| **U-Net (baseline)** | 7.76 M | 0.88 | **0.161** | **0.278** | 0.318 | 0.246 | 0.9944 |
| Attention U-Net + MNv2 | 6.70 M | 0.78 | 0.081 | 0.149 | 0.133 | 0.171 | 0.9914 |
| delta (proposed - baseline) | | | -0.080 | -0.128 | -0.185 | -0.075 | -0.003 |

Pixel accuracy is ~0.99 for both and is uninformative (99.6% of valid pixels
are negative); IoU and Dice are the operative metrics. Best validation Dice is
a near-tie (**baseline 0.245 @ epoch 8, attention 0.246 @ epoch 36**), while the
held-out test set clearly favours the baseline (IoU 0.161 vs 0.081). On the
leak-free split neither model shows convincing learning: validation Dice bounces
in ~[0.17, 0.25] across all 80 epochs while training loss keeps falling (U-Net
1.74 -> 0.84), i.e. both models overfit the 261-patch train set against a
16-patch validation set. The attention model, with a pretrained natural-RGB
encoder driven by an 8-band reflectance stack, overfits harder and generalises
worse. All subsequent stages use the plain U-Net (see Section 5.3 on model
selection).

Against John and Zhang (2022) (test IoU 0.90-0.95, F1 0.955-0.977), our IoU/F1
are far lower. This is expected and not a like-for-like failure: different task
(bi-temporal change vs single-image segmentation), ~0.3% vs abundant positive
prevalence, 30 m GFC labels vs hand-digitised polygons, and fragmented
smallholder loss vs Amazon clear-cutting. Their own Attention-vs-U-Net gain is
small (F1 +0.002 to +0.018), so the direction we see (attention worse under a
fair shared schedule on a tiny dataset) is not in tension with their finding.

### 5.2 Change detection and area

| Region | GFC reference (ha) | Predicted (ha) | Pred / GFC | Pixel IoU |
|---|---|---|---|---|
| **test (held out)** | **51.5** | **39.6** | **0.77** | 0.161 |
| val | 29.8 | 26.0 | 0.87 | 0.145 |
| train | 131.7 | 86.5 | 0.66 | 0.100 |
| full region | 237.4 | 164.5 | 0.69 | 0.125 |

On the held-out region the model **under-predicts area by ~23%** (39.6 ha vs
51.5 ha). The aggregate ratio (0.77) is closer to 1 than the pixel IoU (0.16)
would imply - errors partly cancel when summed - but this is a modest
improvement, not the near-match one might hope for, and it does not hold on the
full region (0.69). The direction is consistent across all four splits
(under-prediction), which points to a genuine recall deficit (the model at its
operating threshold recovers ~25% of loss pixels) rather than noise.

### 5.3 Model selection

Validation marginally preferred the attention model (Dice 0.2458 vs 0.2453 - a
0.0005 gap); the held-out test set preferred the baseline decisively (IoU 0.161
vs 0.081). **Both differences sit inside the noise of a 16-validation /
18-test-patch split**, and choosing on the test set would itself be selection
on the held-out data. The plain U-Net is carried into Sections 5.2, 5.4 and 5.5
for reasons **independent of its test score**: a simpler architecture, no
pretrained-RGB-encoder mismatch against an 8-band reflectance stack, and fewer
moving parts for the downstream region-wide inference. Both models' test numbers
are reported with equal prominence in Section 5.1.

### 5.4 Carbon and CO2

| Pixel set | Area (ha) | 3-bin (t CO2) | reg-linear (t CO2) | reg-exp / primary (t CO2) | mean AGC (tC/ha) |
|---|---|---|---|---|---|
| **predicted, test region** | 39.6 | 22,343 | 25,591 | **21,645** | 149 |
| GFC reference, test region | 51.5 | 25,819 | 24,800 | 21,507 | 114 |
| predicted, full region | 164.5 | 94,574 | 109,486 | 94,778 | 157 |
| GFC reference, full region | 237.4 | 116,699 | 104,726 | 94,273 | 108 |

**Held-out headline:** 39.6 ha of predicted new loss -> **~21,600 t CO2**
(primary regression), against ~21,500 t CO2 on the GFC reference area for the
same region - a ~1% match. That match is **coincidental**: the model
under-predicts the *area* by ~23% but over-predicts the *mean carbon density*
of the cleared pixels (149 vs 114 tC/ha, because it preferentially flags
denser, higher-NDVI forest as loss), and the two errors offset in the CO2
total. The linear-regression variant does not show the same cancellation
(25,591 t, +19%), and the full-region prediction under-shoots the reference
by ~0% only because the same two biases scale together. The robust, model-free
comparison is 3-bin vs regression on the *same* pixel set: on the GFC reference
area the exponential regression gives 94,273 t CO2 versus 116,699 t from the
3-bin scheme (~19% lower), because most cleared pixels have moderate NDVI and
the flat 150 tC/ha "moderate" constant over-credits them.

### 5.5 CO2 plausibility check (Global Forest Watch, Wayanad district)

GFW reports 3.82 kha of tree-cover loss and 2.54 Mt CO2e for Wayanad district
(2001-2023, 30% canopy, all pools, all gases; Harris et al. 2021) -> an
implied committed emission factor of **~665 t CO2e/ha** and ~166 ha/yr of loss
district-wide. This study's 848 km2 tile shows ~119 ha/yr (GFC) / ~82 ha/yr
(predicted) - the tile is ~40% of the district area but carries ~70% of its
GFC loss, being centred on the loss-active plateau.

- On the **GFC reference area**, this pipeline's factor is 94,273 t CO2 /
  237.4 ha = **~397 t CO2/ha**, i.e. **~60% of GFW's ~665 t CO2e/ha** - the
  expected fraction for an aboveground-only, CO2-only accounting once GFW's
  soil, belowground and non-CO2 pools are set aside. This figure is unchanged
  by the leakage fix (it depends only on GFC and the regression).
- On the **model-predicted area** the factor rises to 94,778 / 164.5 =
  ~576 t CO2/ha, because the leak-free model is biased toward denser forest
  (Section 5.4). This bias was partly masked in the pre-audit results.

The pipeline is therefore within a well-understood factor of the published
figure on the reference area; the model-predicted number additionally carries
the density bias.

### 5.6 Data-leakage audit and correction

**Finding.** After the first full run, an audit
(`scripts/verify_no_leakage.py`) checked every validation and test patch's
pixel extent against every train patch's extent. The train split's stride-128
overlapping crops are 256 px wide but were placed on a 128 px stride and
assigned to "train" using only the super-block containing each crop's top-left
corner. Crops starting at a 128 px offset from a 512 px block boundary
therefore extended into the neighbouring block; when that neighbour was a
validation or test block, 128 px strips of held-out territory were used as
training data. **8 of 16 validation patches and 9 of 18 test patches were
affected, each with 50-75% of its area also present in training.** No canonical
patch was involved; the leak was entirely from the overlap crops.

**Fix.** `build_dataset.py` now keeps an overlap crop only if its **entire**
256 x 256 footprint lies inside train-assigned blocks; any crop touching a
val/test block is dropped, not reassigned. The canonical grid, the
block-to-split assignment, the norm statistics and the val/test patches are
byte-identical before and after (verified by hash). Overlap crops fell from 228
to 185 (train patches 304 -> 261). The audit script now exits 0. Both models
were retrained with byte-identical configs (same seed, schedule, loss,
`pos_weight`) and the threshold was re-tuned on validation; Sections 5.1-5.5,
Phase 5 inference and Phase 6 carbon were fully re-run.

**Before / after (held-out test split).**

| Quantity | Pre-audit (leaked) | Post-audit (clean) |
|---|---|---|
| Train patches | 304 (76 + 228 overlap) | 261 (76 + 185 overlap) |
| Val/test patches with train pixels | 8/16 val, 9/18 test | 0/16, 0/18 |
| Best val Dice - U-Net | 0.317 (@ e54) | 0.245 (@ e8) |
| Best val Dice - Attn U-Net | 0.323 (@ e63) | 0.246 (@ e36) |
| U-Net test IoU / Dice | 0.196 / 0.327 | **0.161 / 0.278** |
| Attn U-Net test IoU / Dice | 0.168 / 0.287 | **0.081 / 0.149** |
| Proposed - baseline test IoU | -0.028 | -0.080 |
| Predicted test area vs GFC | 49.9 ha (0.97x) | 39.6 ha (0.77x) |
| Predicted full-region area vs GFC | 279.8 ha (1.18x) | 164.5 ha (0.69x) |
| Predicted test CO2 (primary reg.) | 19,756 t | 21,645 t |
| GFC-reference-area CO2 (primary reg.) | 21,507 t | 21,507 t (unchanged) |
| Pipeline emission factor on GFC area | ~397 t CO2/ha (~60% of GFW) | ~397 t CO2/ha (~60% of GFW) |

The leak had inflated validation Dice (partly memorised pixels), softened the
architecture comparison (the attention model's collapse was hidden), and made
the area estimate look like a near-match when it is really a ~20-30%
under-prediction. The GFC-referenced carbon numbers and the 3-bin-vs-regression
comparison are unaffected because they never depended on the model.

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
  selection and the val-vs-test comparison noisy, and both models overfit
  against a validation set this size.
- **Coarse labels.** Hansen GFC is a 30 m annual product used on a 10 m grid;
  it bounds achievable IoU well below values reported for hand-digitised
  benchmarks.

The pipeline's honest value after the audit is narrower than the pre-audit
draft implied: with a per-pixel IoU near 0.16 it produces an **aggregate area
estimate ~20-30% low** and a **CO2 total that happens to fall near the
GFC-reference figure through offsetting biases**. What is solid is (a) the
3-bin -> regression carbon upgrade, which is model-independent and cuts the
region-wide CO2 estimate by ~19% toward the moist-deciduous field range, (b)
the GFC-referenced emission factor being a sensible ~60% of the GFW all-pools
value, and (c) the leakage audit itself as a reusable check
(`scripts/verify_no_leakage.py`).

---

## 7. Conclusion

We built and end-to-end validated a single pipeline from two Sentinel-2
composites to a hectares-lost and a tonnes-CO2 figure for a Western Ghats tile,
and upgraded the carbon step from three hard-coded NDVI constants to a
regionally-calibrated regression. Under a fair shared schedule the
attention-gated MobileNetV2 model was clearly worse on the held-out set than a
plain U-Net; a leakage audit forced a full re-run, after which the pipeline
under-predicts held-out loss area by ~23% and its CO2 total matches the
GFC-reference figure only coincidentally. The model-independent results - the
bins-to-regression carbon reduction and the ~60% emission-factor ratio against
Global Forest Watch - are the parts to build on. Natural next steps: co-located
biomass calibration, multi-region evaluation, a larger annotated change set,
and a recall-oriented loss or threshold to correct the area under-prediction.

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
the leakage audit is in `docs/phase7_notes.md` and `scripts/verify_no_leakage.py`.
Provenance JSON/CSV in `data/*/manifest.json`, `data/processed/split.json`,
`results/metrics/*.json`, `results/deforestation/`, `results/carbon_validation/`.
Each phase, the audit, and the post-audit re-run are separate git commits.
Reference-value sources with DOIs are in `docs/refs/`.
