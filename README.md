# Forest Loss Detection & Dynamic Carbon Emission Estimation

An end-to-end pipeline that goes from raw bi-temporal Sentinel-2 imagery to a
hectares-lost and tonnes-CO₂-emitted estimate for a single study region:

1. **Segment** forest vs. non-forest with a standard U-Net (baseline) and an
   Attention U-Net + MobileNetV2 encoder (proposed).
2. **Detect** deforestation by comparing predicted masks at two time points.
3. **Convert** newly-cleared pixel counts to hectares using the 10 m GSD.
4. **Estimate** CO₂ emissions with an NDVI-derived, vegetation-density-aware
   carbon model — a calibrated regression, upgrading the fixed 3-bin scheme.

### Honest positioning

Attention U-Net for deforestation segmentation (John & Zhang, 2022) and
MobileNetV2 encoders for satellite segmentation are both established in prior
work. Simple NDVI-binned carbon density is weaker than calibrated regression.
The contribution here is the **integrated, honestly-validated pipeline** from
bi-temporal imagery to a CO₂ figure, plus the bins→regression carbon upgrade.
Individual components are not claimed as novel.

## Study region (Phase 1)

Wayanad Plateau, Western Ghats, Kerala, India — bbox `[76.00, 11.55, 76.28,
11.80]` (WGS84), ~848 km². Bi-temporal windows: **T = Jan–Apr 2019**,
**T+1 = Jan–Apr 2021** (dry season, minimal cloud). Sentinel-2 bands
**B3, B4, B8, B11** (green / red / NIR / SWIR1) per timepoint → 8-band stack.
Full details and rationale in [`configs/region.yaml`](configs/region.yaml).

## Environment

- Python **3.11** (`py -3.11`).
- Pinned deps in [`requirements.txt`](requirements.txt).

```bash
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
# local NVIDIA GPU: install CUDA torch first (see requirements.txt header)
.venv\Scripts\python -m pip install -r requirements.txt
```

Verify the install:

```bash
.venv\Scripts\python scripts\check_env.py
```

## Repository layout

```
data/raw/          downloaded Sentinel-2 tiles
data/masks/         Hansen GFC-derived ground-truth masks
data/processed/     256x256 patch tensors, train/val/test splits
src/preprocessing/  download, normalize, patchify, band-stacking
src/models/         U-Net baseline, Attention U-Net + MobileNetV2
src/change_detection/  bi-temporal mask comparison, hectare conversion
src/carbon/         NDVI computation, 3-bin baseline, regression model
src/eval/           metrics, comparison runner, figure generation
src/regions.py      multi-region config access (Phase 8)
results/metrics/    real IoU / Dice / pixel-accuracy per model (JSON/CSV)
results/figures/    input / ground-truth / prediction triptychs
results/carbon_validation/  estimated vs. published CO₂ comparison
configs/            YAML configs per experiment
frontend/           static results dashboard (Vite + React, reads results/*.json)
backend/            Phase 9 live inference service (FastAPI, serves the Phase 8 checkpoint)
webapp/             Phase 9 live inference UI (Vite + React, talks to backend/)
```

## Phase status

Numbers below are the **leak-free split, early stopping, 3-seed protocol**
(segmentation = mean ± sd over seeds 42/43/44; area/CO₂ = carry-forward U-Net
seed 43, chosen on validation only). Full history and before/after in
`docs/phase7_notes.md` and `report.md` §5.6–5.7; superseded values in git
history (`c9947eb`, `82f6948`).

| Phase | Description | Status |
|------:|-------------|--------|
| 1 | Environment & scope | done |
| 2 | Data acquisition & preprocessing | done — 261 train / 16 val / 18 test patches |
| 3 | Baseline standard U-Net | done — test strict IoU 0.158 ± 0.016 (tolerance IoU 0.248 ± 0.018), Dice 0.273 ± 0.024 |
| 4 | Attention U-Net + MobileNetV2 | done — test IoU 0.113 ± 0.023; did not improve on the plain U-Net, kept as a recorded negative result |
| 5 | Change detection & area computation | done — test region 37.3 ha predicted vs 51.5 ha GFC (0.73×, −27%); pixel IoU 0.169 strict / 0.260 tolerance |
| 6 | Carbon estimation module | done — GFC ref area: regression 94.3 ktCO₂ vs 3-bin 116.7 ktCO₂ (~19% lower) |
| 7 | Evaluation, validation & short paper | done — `report.md`, leakage audit, seed-variance analysis, CO₂ sanity-check vs GFW |
| audit | Leakage fix + multi-seed re-run | done — `scripts/verify_no_leakage.py` exits 0; `scripts/aggregate_seeds.py` |
| 8 | Multi-region training data (4 Western Ghats blocks) | done — pooled test IoU 0.176 ± 0.026 (within seed variance of the single-region 0.158 ± 0.016); leave-one-region-out mean 0.092 (poor transfer). `docs/phase8_notes.md`, `report.md` §5.8 |
| 9 | Live inference app | code committed (`backend/`, `webapp/`); needs `pip install` + `npm install`, an Earth Engine service-account key, local end-to-end test, then Render deploy |

## Live inference app (Phase 9)

`backend/` is a FastAPI service that serves the Phase 8 carry-forward checkpoint:
a request names a Western Ghats region (or a custom bbox inside the domain
extent) and two January–April date windows; the backend pulls the Sentinel-2
composites from Earth Engine, runs the tiled model, and returns the loss mask,
cleared hectares and committed aboveground CO₂. `webapp/` is the Vite + React UI
that drives it. The model domain (Western Ghats moist forest, Jan–Apr, change
between two increasing years) is enforced server-side — out-of-domain requests
are refused, not warned. Earth Engine auth is a **service-account JSON key** from
`EE_SERVICE_ACCOUNT_KEY` (never committed). Setup and deployment notes are in
[`backend/README.md`](backend/README.md) and [`webapp/README.md`](webapp/README.md).

## Ground rules

No fabricated metrics, citations, or dataset statistics. Every results table
comes from a script that was actually run, with script and output saved. If
external data (Earth Engine, Hansen GFC, literature carbon values) is not
obtainable as planned, the pipeline stops and a concrete alternative is
proposed — never a silent synthetic substitute.
