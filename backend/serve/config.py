"""Serving configuration - paths, the carry-forward checkpoint, and guard-rail
caps. Values that also exist in configs/region.yaml (domain extent, time
windows, bands) are read from there, never re-typed here.
"""

from __future__ import annotations

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
# make the project's `src` package importable no matter where uvicorn is launched
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
REGION_CFG = REPO / "configs" / "region.yaml"

# --- Served checkpoint (config value) -------------------------------------
# `SERVE_CHECKPOINT_STEM` selects which trained model the app serves. The
# model's config, norm_stats path and val-tuned threshold are still read from
# the checkpoint / its eval JSON at load time - nothing about the model is
# hard-coded here.
#   p10_pooled_unet_s43  -> 2021->2023 (Phase 10), the more recent data  [default]
#   p8_pooled_unet_s44   -> 2019->2021 (Phase 8), the paper's basis
CHECKPOINT_STEM = os.environ.get("SERVE_CHECKPOINT_STEM", "p10_pooled_unet_s43")

# Training window shown on the results page, keyed by checkpoint-stem prefix.
# (The stems do not carry their acquisition dates; this is the one lookup.)
_TRAINING_WINDOWS = {
    "p8_": {"period": "2019_2021", "t": ["2019-01-01", "2019-04-15"],
            "t1": ["2021-01-01", "2021-04-15"], "gfc_lossyear": [19, 20]},
    "p10_": {"period": "2021_2023", "t": ["2021-01-01", "2021-04-15"],
             "t1": ["2023-01-01", "2023-04-15"], "gfc_lossyear": [21, 22]},
}


def _for_stem(stem: str) -> dict:
    for pref, win in _TRAINING_WINDOWS.items():
        if stem.startswith(pref):
            return win
    return {"period": "unknown", "t": None, "t1": None, "gfc_lossyear": None}


TRAINING_WINDOW = _for_stem(CHECKPOINT_STEM)
_PERIOD = TRAINING_WINDOW["period"]
_SEED_RUNS_NAME = ("phase8_seed_runs.json" if _PERIOD == "2019_2021"
                   else "phase10_seed_runs.json")
_CARBON_DIR = (REPO / "results" / "carbon_validation" if _PERIOD == "2019_2021"
               else REPO / "results" / "carbon_validation" / _PERIOD)

CHECKPOINT = REPO / "results" / "checkpoints" / f"{CHECKPOINT_STEM}_best.pt"
EVAL_JSON = REPO / "results" / "metrics" / f"{CHECKPOINT_STEM}.json"

# Result JSON the model card is assembled from (no metric is copied into code).
# SEED_RUNS is the served checkpoint's own 3-seed aggregate; REFERENCE_SEED_RUNS
# is always the 2019->2021 (Phase 8) aggregate - the model card falls back to it
# for analyses (leave-one-region-out, the more-data finding) that were only ever
# measured on that period, labelling them as such rather than showing nulls.
SEED_RUNS = REPO / "results" / "metrics" / _SEED_RUNS_NAME
REFERENCE_SEED_RUNS = REPO / "results" / "metrics" / "phase8_seed_runs.json"
REFERENCE_PERIOD = "2019_2021"
SERVED_PERIOD = _PERIOD
# back-compat alias (modelcard.py historically imported this name)
PHASE8_SEED_RUNS = SEED_RUNS
AREA_SUMMARY = REPO / "results" / "deforestation" / f"{CHECKPOINT_STEM}_area_summary.json"
CARBON_ESTIMATES = _CARBON_DIR / "carbon_estimates.json"
# Pre-fitted NDVI->carbon regression coefficients (committed). The backend loads
# these rather than re-fitting - the fit needs data/raw/s2_T.tif, which is
# gitignored and absent from a git-based deploy.
REGRESSION_COEFS = _CARBON_DIR / "regression_coefs.json"

# Per-job scratch (masks served from here). Root is configurable for ephemeral
# / read-only-repo deploys; per-job dirs are pruned to the newest N once a job
# finishes so a long-running instance does not fill the disk.
JOBS_DIR = pathlib.Path(os.environ.get("SERVE_JOBS_DIR", REPO / "backend" / "_jobs"))
MAX_RETAINED_JOBS = int(os.environ.get("SERVE_MAX_RETAINED_JOBS", "20"))

# --- Earth Engine (service account only; NEVER earthengine authenticate) ---
# Resolution order (serve.eepull.init_ee):
#   1. GEE_KEY_JSON  - the full JSON key *contents* as a string (containers /
#      Railway, which have no persistent secret filesystem). Written to a
#      private temp file OUTSIDE the repo at startup.
#   2. GEE_KEY_PATH / EE_SERVICE_ACCOUNT_KEY - path to the JSON key file
#      (local dev; unchanged).
#   3. earth_engine.service_account_key in configs/region.yaml (git-ignored path).
EE_KEY_JSON = os.environ.get("GEE_KEY_JSON")
EE_KEY_PATH = (os.environ.get("EE_SERVICE_ACCOUNT_KEY")
               or os.environ.get("GEE_KEY_PATH"))
# Optional explicit project override; otherwise the key's project_id / the
# region.yaml project is used.
EE_PROJECT = os.environ.get("EE_PROJECT")

# --- Guard rails ---
# A full ~850 km2 preset region is a slow job: two Earth Engine composite pulls
# plus tiled inference run ~15-25 min. The timeout is generous so presets work;
# a sub-bbox of ~100-200 km2 finishes in 2-4 min and is what to use for quick
# iteration.
MAX_AREA_KM2 = float(os.environ.get("SERVE_MAX_AREA_KM2", "900"))   # raw-bbox path only
JOB_TIMEOUT_S = float(os.environ.get("SERVE_JOB_TIMEOUT_S", "1800"))
# Composite cloud/no-data cover above this (either date) raises a visible flag.
CLOUD_FLAG_PCT = float(os.environ.get("SERVE_CLOUD_FLAG_PCT", "35"))
# Minimum clear Sentinel-2 scenes per window before the result is flagged thin.
MIN_SCENES = int(os.environ.get("SERVE_MIN_SCENES", "8"))

# Tiling - identical to src/change_detection/infer_region.py.
TILE_PX = 256
TILE_STRIDE = 128

# --- Point-and-radius AOI ---
# One model tile is 256 px x 10 m = 2.56 km on a side; Hansen GFC labels are
# 30 m. An AOI smaller than one tile carries no meaningful signal and is
# refused. Derived bboxes are snapped to a whole number of tiles.
TILE_KM = TILE_PX * 10 / 1000.0            # 2.56
MAX_RADIUS_KM = float(os.environ.get("SERVE_MAX_RADIUS_KM", "20"))
# Below this AOI size the expected true-loss pixel count at ~0.3% prevalence is
# very low; results carry a "provisional / zero is expected" caveat.
SMALL_AREA_KM2 = float(os.environ.get("SERVE_SMALL_AREA_KM2", "25"))

# --- Geocoding (Nominatim / OpenStreetMap, no API key) ---
NOMINATIM_URL = os.environ.get("SERVE_NOMINATIM_URL",
                               "https://nominatim.openstreetmap.org/search")
# Nominatim's usage policy requires an identifying User-Agent.
GEOCODE_USER_AGENT = os.environ.get(
    "SERVE_GEOCODE_UA",
    "forest-loss-live-app/0.9 (Western Ghats forest-loss demo)")
GEOCODE_MIN_INTERVAL_S = float(os.environ.get("SERVE_GEOCODE_MIN_INTERVAL_S", "1.1"))
GEOCODE_CACHE_MAX = 512
