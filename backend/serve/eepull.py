"""Earth Engine access for the live app.

Authentication is **service-account only** (``earthengine authenticate`` /
interactive auth is never used). The key is resolved, in order:

  1. ``GEE_KEY_JSON``  - the full JSON key contents as a string. Written to a
     private temp file in the OS temp dir (never inside the repo) at startup and
     removed at process exit. For containers / Railway, which have no persistent
     secret filesystem.
  2. ``GEE_KEY_PATH`` / ``EE_SERVICE_ACCOUNT_KEY`` - path to the JSON key file
     (local development; behaviour unchanged).
  3. ``earth_engine.service_account_key`` in configs/region.yaml (a git-ignored
     path).

The key contents are never logged and never written anywhere under the repo.

The Sentinel-2 composite is built with the *same* recipe as
``src/preprocessing/download_data.s2_composite`` (S2_SR_HARMONIZED + Cloud
Score+ ``cs_cdf >= 0.60`` median, 4 bands, reflectance in [0,1]); this module
imports that function directly rather than re-implementing it. A 5th band
``obs`` (1 where the composite has a clear observation, 0 where every scene was
masked) is added so the tiler can build a valid-pixel mask and report the
cloud / no-data cover of each date.
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import tempfile

import ee
import yaml

from .config import EE_KEY_JSON, EE_KEY_PATH, EE_PROJECT, REGION_CFG, REPO

# reuse the exact training composite recipe
from src.preprocessing.download_data import (  # noqa: E402
    S2_BAND_NAMES, s2_composite,
)
from src.preprocessing.eeutil import download_image_tiled  # noqa: E402

_INITED = False
_TMP_KEY_FILES: list[str] = []

_NO_KEY_MSG = (
    "No Earth Engine service-account key. Set one of, in priority order:\n"
    "  GEE_KEY_JSON            - the JSON key contents as a string (containers / Railway)\n"
    "  GEE_KEY_PATH            - path to the JSON key file (local development)\n"
    "  EE_SERVICE_ACCOUNT_KEY  - alias for GEE_KEY_PATH\n"
    "or earth_engine.service_account_key in configs/region.yaml. "
    "See backend/README.md."
)


def _cfg() -> dict:
    with open(REGION_CFG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@atexit.register
def _cleanup_tmp_keys() -> None:
    for p in _TMP_KEY_FILES:
        try:
            os.unlink(p)
        except OSError:
            pass


def _key_json_to_tempfile(info: dict) -> str:
    """Write a service-account key dict to a 0600 temp file in the OS temp dir
    (never under the repo). Removed at process exit."""
    fd, path = tempfile.mkstemp(prefix="ee-sa-key-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(info, fh)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    resolved = pathlib.Path(path).resolve()
    if str(resolved).startswith(str(pathlib.Path(REPO).resolve())):
        os.unlink(path)
        raise RuntimeError("refusing to write the Earth Engine key inside the repo directory")
    _TMP_KEY_FILES.append(str(resolved))
    return str(resolved)


def init_ee() -> str:
    """Initialise Earth Engine with a service account. Returns the project id."""
    global _INITED
    if _INITED:
        return getattr(ee.data, "_cloud_api_user_project", None) or "(initialised)"

    cfg = _cfg()

    if EE_KEY_JSON:
        try:
            info = json.loads(EE_KEY_JSON)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GEE_KEY_JSON is set but is not valid JSON.") from exc
        if not isinstance(info, dict) or not info.get("client_email"):
            raise RuntimeError("GEE_KEY_JSON is not a service-account key "
                               "(no client_email).")
        key_file = _key_json_to_tempfile(info)
    else:
        key_path = EE_KEY_PATH or (cfg.get("earth_engine") or {}).get("service_account_key")
        if not key_path:
            raise RuntimeError(_NO_KEY_MSG)
        key_path = str(pathlib.Path(key_path).expanduser())
        if not pathlib.Path(key_path).is_file():
            raise RuntimeError(f"EE service-account key not found at {key_path!r}.")
        with open(key_path, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        if not info.get("client_email"):
            raise RuntimeError(f"{key_path!r} is not a service-account key "
                               "(no client_email).")
        key_file = key_path

    project = EE_PROJECT or info.get("project_id") or cfg["earth_engine"]["project"]
    creds = ee.ServiceAccountCredentials(info["client_email"], key_file)
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
