"""Earth Engine access for the live app.

Authentication is **service-account only**: the JSON key path comes from the
``EE_SERVICE_ACCOUNT_KEY`` environment variable (fallback:
``earth_engine.service_account_key`` in configs/region.yaml, which must point at
a git-ignored file). ``earthengine authenticate`` / interactive auth is never
used here.

The Sentinel-2 composite is built with the *same* recipe as
``src/preprocessing/download_data.s2_composite`` (S2_SR_HARMONIZED + Cloud
Score+ ``cs_cdf >= 0.60`` median, 4 bands, reflectance in [0,1]); this module
imports that function directly rather than re-implementing it. A 5th band
``obs`` (1 where the composite has a clear observation, 0 where every scene was
masked) is added so the tiler can build a valid-pixel mask and report the
cloud / no-data cover of each date.
"""

from __future__ import annotations

import json
import pathlib

import ee
import yaml

from .config import EE_KEY_PATH, EE_PROJECT, REGION_CFG

# reuse the exact training composite recipe
from src.preprocessing.download_data import (  # noqa: E402
    S2_BAND_NAMES, s2_composite,
)
from src.preprocessing.eeutil import download_image_tiled  # noqa: E402

_INITED = False


def _cfg() -> dict:
    with open(REGION_CFG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def init_ee() -> str:
    """Initialise Earth Engine with a service account. Returns the project id."""
    global _INITED
    if _INITED:
        return ee.data._cloud_api_user_project or "(initialised)"

    cfg = _cfg()
    key_path = EE_KEY_PATH or (cfg.get("earth_engine") or {}).get("service_account_key")
    if not key_path:
        raise RuntimeError(
            "No Earth Engine service-account key. Set EE_SERVICE_ACCOUNT_KEY to "
            "the path of a JSON key file (see backend/README.md).")
    key_path = str(pathlib.Path(key_path).expanduser())
    if not pathlib.Path(key_path).is_file():
        raise RuntimeError(f"EE service-account key not found at {key_path!r}.")

    with open(key_path, "r", encoding="utf-8") as fh:
        info = json.load(fh)
    email = info.get("client_email")
    if not email:
        raise RuntimeError(f"{key_path!r} is not a service-account key "
                           "(no client_email).")
    project = EE_PROJECT or info.get("project_id") or cfg["earth_engine"]["project"]
    creds = ee.ServiceAccountCredentials(email, key_path)
    ee.Initialize(creds, project=project)
    _INITED = True
    return project


def _aoi(bbox_wsen) -> ee.Geometry:
    return ee.Geometry.Rectangle(list(bbox_wsen), "EPSG:4326", geodesic=False)


def fetch_composite(bbox_wsen, start: str, end: str, out_path, crs: str = "EPSG:32643",
                    scale_m: int = 10) -> dict:
    """Download one 5-band composite (green, red, nir, swir1, obs) for the bbox
    and window. Returns provenance incl. scene count and cloud / no-data cover %.
    """
    aoi = _aoi(bbox_wsen)
    comp, stats = s2_composite(aoi, start, end)
    obs = comp.select("green").mask().rename("obs")
    img = comp.addBands(obs)

    # authoritative cloud / no-data cover: fraction of AOI with no clear obs
    frac = obs.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=scale_m,
        maxPixels=1e10, bestEffort=True,
    ).get("obs")
    clear_fraction = frac.getInfo()
    if clear_fraction is None:
        clear_fraction = 0.0
    cover_pct = round(100.0 * (1.0 - float(clear_fraction)), 2)

    download_image_tiled(
        img, list(bbox_wsen), out_path, crs=crs, scale_m=scale_m,
        bands=[*S2_BAND_NAMES, "obs"],
        band_names=[*S2_BAND_NAMES, "obs"],
    )
    return {
        "window": [start, end],
        "n_scenes": stats.get("n_scenes"),
        "cloud_or_nodata_cover_pct": cover_pct,
        "clear_fraction": round(float(clear_fraction), 4),
        "cloud_mask": stats.get("cloud_mask"),
        "raster": str(out_path),
    }
