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

# The Phase 8 carry-forward checkpoint (median best-validation-Dice pooled seed).
# Its config, norm_stats path and val-tuned threshold are read from the file /
# its eval JSON at load time - nothing about the model is hard-coded here.
CHECKPOINT = REPO / "results" / "checkpoints" / "p8_pooled_unet_s44_best.pt"
EVAL_JSON = REPO / "results" / "metrics" / "p8_pooled_unet_s44.json"

# Result JSON the model card is assembled from (no metric is copied into code).
PHASE8_SEED_RUNS = REPO / "results" / "metrics" / "phase8_seed_runs.json"
AREA_SUMMARY = REPO / "results" / "deforestation" / "p8_pooled_unet_s44_area_summary.json"
CARBON_ESTIMATES = REPO / "results" / "carbon_validation" / "carbon_estimates.json"

# Per-job scratch (masks served from here; cleaned on process restart).
JOBS_DIR = pathlib.Path(os.environ.get("SERVE_JOBS_DIR", REPO / "backend" / "_jobs"))

# --- Earth Engine (service account only; NEVER earthengine authenticate) ---
# Path to the service-account JSON key. Must point at a git-ignored file.
# `EE_SERVICE_ACCOUNT_KEY` is the documented name; `GEE_KEY_PATH` is also
# accepted for convenience.
EE_KEY_PATH = (os.environ.get("EE_SERVICE_ACCOUNT_KEY")
               or os.environ.get("GEE_KEY_PATH"))  # required at runtime
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
