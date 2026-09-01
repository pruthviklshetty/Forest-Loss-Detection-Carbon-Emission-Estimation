"""Blocking job pipeline: Earth Engine pull -> model -> area/carbon -> mask PNG.

Called by ``JobStore.run`` inside a worker thread. ``update(status, progress,
message)`` pushes state back to the Job so ``GET /jobs/{id}`` can report it.
"""

from __future__ import annotations

import numpy as np
import rasterio
from PIL import Image

from .config import CLOUD_FLAG_PCT, JOBS_DIR, MIN_SCENES
from .eepull import fetch_composite, init_ee
from .inference import Segmenter, estimate
from .modelcard import build_model_card

_SEG: Segmenter | None = None


def _segmenter() -> Segmenter:
    global _SEG
    if _SEG is None:
        _SEG = Segmenter()
    return _SEG


def _stretch(a, lo_hi=(2, 98)):
    a = a.astype(np.float32)
    lo, hi = np.nanpercentile(a, lo_hi)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def _render_mask_png(pred: dict, out_path, max_px: int = 1400) -> None:
    img_t = pred["img_t"]              # (4,H,W): green,red,nir,swir1
    loss = pred["loss"].astype(bool)
    H, W = loss.shape
    step = max(1, int(max(H, W) / max_px))
    g = _stretch(img_t[0, ::step, ::step])
    r = _stretch(img_t[1, ::step, ::step])
    n = _stretch(img_t[2, ::step, ::step])
    fc = np.dstack([n, r, g])                       # NIR-red-green false colour
    base = (fc * 0.45 * 255).astype(np.uint8)
    lo = loss[::step, ::step]
    h2, w2 = lo.shape
    base = base[:h2, :w2]
    rgba = np.dstack([base, np.full((h2, w2), 255, np.uint8)])
    rgba[lo] = [230, 30, 30, 255]
    Image.fromarray(rgba, "RGBA").save(out_path)


def run_pipeline(job, update) -> dict:
    spec = job.spec
    bbox = spec["bbox_wsen"]
    wt, wt1 = spec["window_t"], spec["window_t1"]
    jd = JOBS_DIR / job.id
    jd.mkdir(parents=True, exist_ok=True)

    update("fetching", 0.05, "authenticating with Earth Engine (service account)")
    project = init_ee()

    update("fetching", 0.12, f"fetching {wt[0][:4]} Sentinel-2 composite")
    prov_t = fetch_composite(bbox, wt[0], wt[1], jd / "s2_T.tif")

    update("fetching", 0.38, f"fetching {wt1[0][:4]} Sentinel-2 composite")
    prov_t1 = fetch_composite(bbox, wt1[0], wt1[1], jd / "s2_T1.tif")

    update("inferring", 0.55, "running the segmentation model")
    seg = _segmenter()
    pred = seg.predict(jd / "s2_T.tif", jd / "s2_T1.tif",
                       progress=lambda f: update(progress=0.55 + 0.3 * f))

    update("estimating", 0.9, "computing cleared area and committed CO2")
    est = estimate(pred)

    _render_mask_png(pred, jd / "mask.png")

    cloud = {
        "window_t": {
            "dates": wt, "n_scenes": prov_t["n_scenes"],
            "cloud_or_nodata_cover_pct": prov_t["cloud_or_nodata_cover_pct"],
            "high_cloud": prov_t["cloud_or_nodata_cover_pct"] is not None
            and prov_t["cloud_or_nodata_cover_pct"] > CLOUD_FLAG_PCT,
            "few_scenes": (prov_t["n_scenes"] or 0) < MIN_SCENES,
        },
        "window_t1": {
            "dates": wt1, "n_scenes": prov_t1["n_scenes"],
            "cloud_or_nodata_cover_pct": prov_t1["cloud_or_nodata_cover_pct"],
            "high_cloud": prov_t1["cloud_or_nodata_cover_pct"] is not None
            and prov_t1["cloud_or_nodata_cover_pct"] > CLOUD_FLAG_PCT,
            "few_scenes": (prov_t1["n_scenes"] or 0) < MIN_SCENES,
        },
        "flag_threshold_pct": CLOUD_FLAG_PCT,
        "min_scenes": MIN_SCENES,
    }
    cloud["any_flag"] = any(cloud[w][k] for w in ("window_t", "window_t1")
                            for k in ("high_cloud", "few_scenes"))

    with rasterio.open(jd / "s2_T.tif") as s:
        px_h, px_w = s.height, s.width

    return {
        "mask_ready": True,
        "earth_engine_project": project,
        "domain": {
            "selection": spec.get("selection"),
            "region_id": spec.get("region_id"),
            "region_name": spec.get("region_name"),
            "in_training_set": spec.get("in_training_set", False),
            "bbox_wsen": bbox,
            "area_km2": spec.get("area_km2"),
            "derived": spec.get("derived"),
            "window_t": wt,
            "window_t1": wt1,
            "raster_px": [px_w, px_h],
        },
        "metric_case": spec.get("metric_case", "loro"),
        "metric_case_region": spec.get("metric_case_region"),
        "small_area": spec.get("small_area", False),
        "small_area_threshold_km2": spec.get("small_area_threshold_km2"),
        "operating_threshold": seg.threshold,
        "checkpoint_epoch": seg.checkpoint_epoch,
        "area_carbon": est,
        "cloud": cloud,
        "provenance": {"window_t": prov_t, "window_t1": prov_t1},
        "model_card": build_model_card(),
    }
