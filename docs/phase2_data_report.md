# Phase 2 - Data Report

**Region:** Wayanad Plateau, Western Ghats, Kerala, India  
**BBox (WGS84 W,S,E,N):** `[76.0, 11.55, 76.28, 11.8]`  
**Grid:** 3064 x 2778 px @ 10 m, EPSG:32643

## 1. Sentinel-2 acquisition

Collection `COPERNICUS/S2_SR_HARMONIZED`, cloud-masked with Cloud Score+ (`cs_cdf >= 0.60`), per-band **median** composite, reflectance scaled by 10000 and clipped to [0,1]. Bands B3/B4/B8/B11 -> green/red/nir/swir1; SWIR1 bilinearly upsampled 20 m -> 10 m.

| Window | Dates | Scenes in window | Cloudy-pixel % (min / median / max) |
|---|---|---|---|
| T | 2019-01-01 .. 2019-04-15 | 84 | 0.0 / 5.3 / 89.6 |
| T+1 | 2021-01-01 .. 2021-04-15 | 84 | 0.0 / 9.8 / 99.9 |

## 2. Ground truth: Hansen GFC -> binary forest-loss label

Asset `UMD/hansen/global_forest_change_2024_v1_12`. `lossyear` is a year-of-loss code, not a T-vs-T+1 mask; conversion (documented in `src/preprocessing/build_labels.py`):

```
forest2000  = treecover2000 >= 30.0%
land        = datamask == 1
loss_window = lossyear in [19, 20]   # calendar 2019 & 2020
label = 1  where  forest2000 AND land AND loss_window   (new loss T -> T+1)
label = 0  elsewhere on land ;  valid = land
```
Loss codes `[19, 20]` chosen because the T composite (Jan-Apr 2019) is the canopy state entering 2019 and T+1 (Jan-Apr 2021) the state entering 2021, so calendar-2019/2020 stand-replacement loss is exactly what happened between acquisitions.

| Quantity | Value |
|---|---|
| Valid land pixels | 8,457,916 (99.367% of grid) |
| Forest (canopy >= 30.0%) at 2000 | 6,360,971 (75.207% of land) |
| Forest-loss positives (2019-20) | 23,736 (0.2806% of land, 0.3732% of forest2000) |
| GFC reference area lost | **237.36 ha** |

`lossyear` histogram over land: `{'0': 8141198, '1': 956, '2': 568, '3': 1963, '4': 2208, '5': 3530, '6': 3789, '7': 22543, '8': 7498, '9': 2874, '10': 79, '11': 12395, '12': 9826, '13': 5130, '14': 7558, '15': 2675, '16': 13857, '17': 15141, '18': 20673, '19': 15299, '20': 12558, '21': 9321, '22': 13881, '23': 90859, '24': 41537}`

## 3. Patches and split

295 non-overlapping 256x256 patches, 8 bands `['T_green', 'T_red', 'T_nir', 'T_swir1', 'T1_green', 'T1_red', 'T1_nir', 'T1_swir1']`. Split: spatially blocked (whole super-blocks per split); overlapping extra patches added to TRAIN blocks only (2x2-patch blocks, seed 42).

Train blocks additionally get overlapping patches at stride 128 px to enlarge the train set; val/test stay canonical non-overlapping. Positive-rate and area columns below are from canonical patches only.

| Split | Patches (canon + overlap) | Patches w/ loss | Positive px (% of valid) | Area lost (ha) |
|---|---|---|---|---|
| train | 261 (76 + 185) | 226 | 0.2652% | 131.7 |
| val | 16 (16 + 0) | 14 | 0.2854% | 29.8 |
| test | 18 (18 + 0) | 18 | 0.4391% | 51.5 |

Severe class imbalance is expected and is handled at train time (Dice + BCE, positive weighting).

## 4. Normalisation

Bands are stored as [0,1] reflectance. Per-band mean/std over valid train pixels (4,967,118) for optional z-scoring at train time:

| band | T_green | T_red | T_nir | T_swir1 | T1_green | T1_red | T1_nir | T1_swir1 |
|---|---|---|---|---|---|---|---|---|
| mean | 0.0590 | 0.0529 | 0.2400 | 0.1864 | 0.0597 | 0.0508 | 0.2480 | 0.1823 |
| std  | 0.0166 | 0.0252 | 0.0441 | 0.0430 | 0.0152 | 0.0224 | 0.0461 | 0.0414 |

## 5. Figures

- `results/figures/phase2_aoi_overview.png`
- `results/figures/phase2_patch_examples.png`
- `results/figures/phase2_split_balance.png`

## 6. Caveats carried into Phase 3

- **Small dataset.** Canonical: 76 train / 16 val / 18 test. Train is enlarged to 261 via stride-128 overlapping crops kept only where the whole 256 px footprint lies in train blocks (a 2026-08 leakage audit tightened this rule; see docs/phase7_notes.md). Still small, so strong augmentation is used and widening the AOI stays an option.
- **Extreme class imbalance.** Positive rate ~0.27-0.44% of valid pixels. Needs Dice/Tversky + weighted BCE and threshold tuning; pixel accuracy will be near-trivial and is reported only for completeness.
- **Edge margin.** The 256-px grid covers 2816 x 2560 of the 3064 x 2778 raster; the right/bottom margin holds 24.4 ha of GFC loss not in any patch (213.0 of 237.4 ha retained).
- **Blocked split is not perfectly balanced.** Test positive rate (0.439%) exceeds train (0.265%); seed 42 is fixed in config and not re-picked to avoid gaming the split.
- **Bi-temporal signal is subtle.** In the highest-loss patches the T and T+1 false-colour composites look similar at loss sites, and some patches retain thin-haze / BRDF differences from median compositing. This is a genuinely hard change-detection setting; Phase 4's comparison to John & Zhang (2022) must account for it.
