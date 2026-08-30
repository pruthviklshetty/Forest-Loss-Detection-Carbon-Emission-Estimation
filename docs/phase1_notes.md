# Phase 1 — Environment & Scope

## What was built

- Repository skeleton matching the build brief (`data/`, `src/`, `notebooks/`,
  `results/`, `configs/`), with package `__init__.py` files and `.gitkeep`
  placeholders so empty dirs are tracked.
- `requirements.txt` — exact-version pins, target interpreter CPython 3.11.
- `.venv/` — Python 3.11.0 virtual environment (git-ignored).
- `scripts/check_env.py` — install checkpoint: imports every load-bearing
  dependency, prints versions, and runs functional smoke tests (torch forward
  pass, `smp.Unet(mobilenet_v2, in_channels=8)` forward pass, rasterio GeoTIFF
  round-trip, pyproj UTM transform). Exits non-zero on any failure.
- `configs/region.yaml` — the study-region and bi-temporal-window definition
  all later phases read from.
- `README.md`, `.gitignore`.

## Study region and window (locked)

| Field | Value |
|---|---|
| Region | Wayanad Plateau, Western Ghats, Kerala, India |
| BBox (WGS84, W,S,E,N) | `76.00, 11.55, 76.28, 11.80` |
| Approx. area | ~848 km² (~30.5 × 27.8 km) |
| Working projection | EPSG:32643 (UTM 43N) |
| Year T | 2019, acquisition window **2019-01-01 … 2019-04-15** (post-NE-monsoon dry season) |
| Year T+1 | 2021, acquisition window **2021-01-01 … 2021-04-15** |
| Sensor | Sentinel-2 L2A `COPERNICUS/S2_SR_HARMONIZED` (10 m); L1C fallback decided in Phase 2 |
| Bands | B3, B4, B8, B11 (green / red / NIR / SWIR1) per timepoint → 8-band stack |
| Ground truth | Hansen GFC `UMD/hansen/global_forest_change_2023_v1_11`, canopy threshold 30%, loss-year codes 19 & 20 |

### Band-choice rationale
Red + NIR give NDVI (segmentation cue + Phase 6 carbon input). Green adds
canopy-vigour / water separation. SWIR1 (B11) is sensitive to canopy moisture
and bare soil, separating fresh clearings and burn scars from intact canopy.
Blue (B2) dropped — haze-dominated at 10 m. SWIR1 preferred over SWIR2 for
stronger vegetation-moisture contrast. B11 resampled 20 m → 10 m.

### GFC loss-year rationale
T image (early 2019) = forest state entering 2019; T+1 image (early 2021) =
state entering 2021. Calendar-2019 and calendar-2020 loss (lossyear 19, 20)
falls between the two acquisitions, so those two codes define the positive
class. Revisit if Phase 2 acquisition dates shift.

## Known risks carried into Phase 2

- Western Ghats dry-season cloud may still limit clear Sentinel-2 scenes;
  a wider window or L1C fallback may be needed.
- Wayanad forest loss is fragmented (smallholder), a fainter signal than
  Amazon arc-of-deforestation clearing — logged as a scope limitation.
- Earth Engine authentication must be set up before the Phase 2 pull.

## Install checkpoint result

`pip install -r requirements.txt` into a fresh Python 3.11.0 `.venv` completed
with exit code 0; every pin resolved exactly as written (no version drift).
`python scripts/check_env.py` — **all 18 imports and all 5 smoke tests pass**:

```
torch 2.5.1+cpu | torchvision 0.20.1+cpu | segmentation_models_pytorch 0.3.4
timm 0.9.7 | numpy 1.26.4 | scipy 1.14.1 | sklearn 1.5.2 | pandas 2.2.3
rasterio 1.3.11 | shapely 2.0.6 | pyproj 3.7.0 | cv2 4.10.0 | PIL 10.4.0
ee 0.1.408 | requests 2.32.3 | yaml 6.0.2 | matplotlib 3.9.2 | tqdm 4.66.5

smoke: torch conv2d fwd OK | smp.Unet(mobilenet_v2, in_channels=8) -> (1,1,256,256) OK
       rasterio GeoTIFF round-trip OK | pyproj 4326->32643 OK
```

**Note:** the PyPI wheel gives `torch==2.5.1+cpu`. For Phase 3+ local training
on the RTX 3050, reinstall torch/torchvision from the CUDA index
(`--index-url https://download.pytorch.org/whl/cu121`) per the `requirements.txt`
header. CPU torch is sufficient for this checkpoint.
