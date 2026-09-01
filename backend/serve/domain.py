"""Domain gate for the live app.

The model was trained on Western Ghats moist forest, January-April composites,
2019 vs 2021. Requests outside that envelope are **refused**, not warned:

  - a bbox must lie entirely inside `domain_extent_wsen` from configs/region.yaml
  - the requested area must not exceed the guard-rail cap
  - both date windows must fall within January-April (months 1-4) of a single
    calendar year, matching the training composites

Preset regions are the four Phase 8 training blocks plus a few more inside the
domain extent; the latter are marked `in_training_set: false` so the frontend
can say so.
"""

from __future__ import annotations

import datetime as dt
import math

import yaml

from .config import (MAX_AREA_KM2, MAX_RADIUS_KM, REGION_CFG, SMALL_AREA_KM2,
                     TILE_KM)

_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON_EQ = 111.320

# --- extra presets inside the domain extent, NOT in the training set ----------
# Bboxes are Western Ghats moist-forest blocks within domain_extent_wsen; their
# Hansen loss density was not probed, so gfc_loss_ha_2019_20 is unknown.
_EXTRA_PRESETS = [
    {
        "id": "agumbe",
        "name": "Agumbe / Someshwara, Western Ghats, Karnataka, India",
        "bbox_wsen": [75.05, 13.40, 75.33, 13.65],
        "admin_context": "Agumbe rainforest and Someshwara Wildlife Sanctuary. "
                         "Wet evergreen forest; one of the wettest parts of the "
                         "Ghats. Inside the domain extent, not a training region.",
    },
    {
        "id": "silent_valley",
        "name": "Silent Valley / New Amarambalam, Western Ghats, Kerala, India",
        "bbox_wsen": [76.35, 11.05, 76.63, 11.30],
        "admin_context": "Silent Valley National Park and the New Amarambalam "
                         "reserve. Undisturbed tropical wet evergreen forest. "
                         "Inside the domain extent, not a training region.",
    },
    {
        "id": "periyar",
        "name": "Periyar / Idukki, Western Ghats, Kerala, India",
        "bbox_wsen": [76.95, 9.40, 77.23, 9.65],
        "admin_context": "Periyar Tiger Reserve and the Idukki high ranges. "
                         "Moist-deciduous to montane wet forest with cardamom "
                         "estates. Inside the domain extent, not a training region.",
    },
]


class DomainError(ValueError):
    """Raised when a request is outside the model's training domain."""


def _cfg() -> dict:
    with open(REGION_CFG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def domain_extent() -> list[float]:
    return [float(x) for x in _cfg()["domain_extent_wsen"]]


# The training window shown to the user comes from the served checkpoint
# (config.TRAINING_WINDOW, surfaced by main._served_training_windows), not from
# configs/region.yaml - so it stays correct whichever period's model is served.


def preset_regions() -> list[dict]:
    """Preset list the frontend renders: 4 training blocks + extras, all inside
    the domain extent."""
    cfg = _cfg()
    ext = [float(x) for x in cfg["domain_extent_wsen"]]
    out = []
    for r in cfg["regions"]:
        wsen = r["bbox"]["wsen"]
        out.append({
            "id": r["id"],
            "name": r["name"],
            "bbox_wsen": [float(x) for x in wsen],
            "admin_context": " ".join((r.get("admin_context") or "").split()),
            "in_training_set": True,
            "gfc_loss_ha_2019_20": r.get("gfc_loss_ha_2019_20"),
            "area_km2": round(bbox_area_km2(wsen), 1),
        })
    for r in _EXTRA_PRESETS:
        _require_inside(r["bbox_wsen"], ext)          # sanity: keep this list honest
        out.append({
            **r,
            "in_training_set": False,
            "gfc_loss_ha_2019_20": None,
            "area_km2": round(bbox_area_km2(r["bbox_wsen"]), 1),
        })
    return out


def preset_by_id(rid: str) -> dict | None:
    for r in preset_regions():
        if r["id"] == rid:
            return r
    return None


def bbox_area_km2(wsen) -> float:
    w, s, e, n = (float(x) for x in wsen)
    mean_lat = math.radians((s + n) / 2.0)
    km_per_deg_lon = _KM_PER_DEG_LON_EQ * math.cos(mean_lat)
    return abs((e - w) * km_per_deg_lon) * abs((n - s) * _KM_PER_DEG_LAT)


def bbox_sides_km(wsen) -> tuple[float, float]:
    """(west-east span, south-north span) in km."""
    w, s, e, n = (float(x) for x in wsen)
    mean_lat = math.radians((s + n) / 2.0)
    return (abs(e - w) * _KM_PER_DEG_LON_EQ * math.cos(mean_lat),
            abs(n - s) * _KM_PER_DEG_LAT)


def training_regions() -> list[dict]:
    cfg = _cfg()
    out = []
    for r in cfg["regions"]:
        out.append({"id": r["id"], "bbox_wsen": [float(x) for x in r["bbox"]["wsen"]]})
    return out


def metric_case_for_bbox(wsen) -> tuple[str, str | None]:
    """'pooled' + region id if the whole bbox sits inside one training region,
    else 'loro' (the model is in its leave-one-region-out regime)."""
    w, s, e, n = (float(x) for x in wsen)
    for tr in training_regions():
        tw, ts, te, tn = tr["bbox_wsen"]
        if w >= tw and s >= ts and e <= te and n <= tn:
            return "pooled", tr["id"]
    return "loro", None


def derive_bbox_from_point(lat: float, lon: float, radius_km: float) -> dict:
    """Square bbox centred on (lat, lon), side = 2*radius_km snapped UP to a
    whole number of 2.56 km model tiles. Refuses a radius that would give an AOI
    smaller than one tile - no silent padding of a too-small request."""
    try:
        lat = float(lat)
        lon = float(lon)
        radius_km = float(radius_km)
    except (TypeError, ValueError) as exc:
        raise DomainError(f"center/radius not numeric: {exc}") from exc
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise DomainError(f"center [{lat}, {lon}] is not a valid lat/lon.")
    if radius_km <= 0:
        raise DomainError("radius_km must be positive.")
    if radius_km > MAX_RADIUS_KM:
        raise DomainError(
            f"radius {radius_km:g} km exceeds the {MAX_RADIUS_KM:g} km cap. "
            f"Larger areas mean long Earth Engine pulls with no benefit for a "
            f"local query; use a preset region for a whole district.")

    want_side_km = 2.0 * radius_km
    if want_side_km < TILE_KM - 1e-9:
        raise DomainError(
            f"a {radius_km:g} km radius is a {want_side_km:.2f} km square, "
            f"smaller than one model tile ({TILE_KM:.2f} km). The model "
            f"processes 2.56 km tiles from 10 m Sentinel-2 imagery and the "
            f"Hansen GFC labels are 30 m, so a smaller area cannot produce a "
            f"meaningful result. Choose a radius of at least "
            f"{TILE_KM / 2:.2f} km.")

    n_tiles = max(1, math.ceil(want_side_km / TILE_KM - 1e-9))
    side_km = n_tiles * TILE_KM
    half_lat_deg = (side_km / 2.0) / _KM_PER_DEG_LAT
    half_lon_deg = (side_km / 2.0) / (_KM_PER_DEG_LON_EQ * math.cos(math.radians(lat)))
    wsen = [lon - half_lon_deg, lat - half_lat_deg,
            lon + half_lon_deg, lat + half_lat_deg]
    return {
        "bbox_wsen": [round(x, 6) for x in wsen],
        "center": [lat, lon],
        "radius_km": radius_km,
        "requested_side_km": round(want_side_km, 3),
        "derived_side_km": round(side_km, 3),
        "n_tiles_per_side": n_tiles,
    }


def _require_inside(wsen, ext) -> None:
    w, s, e, n = (float(x) for x in wsen)
    ew, es, ee_, en = ext
    if not (w >= ew and s >= es and e <= ee_ and n <= en):
        raise DomainError(
            f"bbox {wsen} is not fully inside the Western Ghats domain extent "
            f"{ext}. The model was trained only on Western Ghats moist forest; "
            f"it is not valid outside this box.")
    if not (e > w and n > s):
        raise DomainError(f"bbox {wsen} is degenerate (need W<E and S<N).")


def validate_bbox(wsen, *, enforce_area_cap: bool = True) -> list[float]:
    ext = domain_extent()
    _require_inside(wsen, ext)
    sx, sy = bbox_sides_km(wsen)
    if min(sx, sy) < TILE_KM - 1e-6:
        raise DomainError(
            f"bbox is {sx:.2f} x {sy:.2f} km; the smaller side is below one "
            f"model tile ({TILE_KM:.2f} km). The model processes 2.56 km tiles "
            f"from 10 m Sentinel-2 imagery and the Hansen GFC labels are 30 m, "
            f"so a smaller area cannot produce a meaningful result.")
    if enforce_area_cap:
        area = bbox_area_km2(wsen)
        if area > MAX_AREA_KM2:
            raise DomainError(
                f"requested area {area:.0f} km2 exceeds the {MAX_AREA_KM2:.0f} "
                f"km2 cap. Split the area into smaller requests.")
    return [float(x) for x in wsen]


def _parse_window(win, label: str) -> tuple[str, str]:
    try:
        start_s, end_s = win
        start = dt.date.fromisoformat(str(start_s))
        end = dt.date.fromisoformat(str(end_s))
    except Exception as exc:  # noqa: BLE001
        raise DomainError(f"{label}: expected [start, end] as ISO dates "
                          f"(YYYY-MM-DD); got {win!r} ({exc}).") from exc
    if end <= start:
        raise DomainError(f"{label}: end {end} must be after start {start}.")
    if start.year != end.year:
        raise DomainError(f"{label}: start and end must be in the same calendar "
                          f"year ({start.year} vs {end.year}).")
    if not (1 <= start.month <= 4 and 1 <= end.month <= 4):
        raise DomainError(
            f"{label}: {start}..{end} is outside January-April. The model was "
            f"trained on Jan-Apr dry-season composites; other seasons are out of "
            f"domain and are not accepted.")
    if (end - start).days > 120:
        raise DomainError(f"{label}: window {(end - start).days} days is too "
                          f"long; keep it within a single Jan-Apr season "
                          f"(<= 120 days).")
    return start.isoformat(), end.isoformat()


def validate_windows(window_t, window_t1) -> dict:
    t = _parse_window(window_t, "window_t")
    t1 = _parse_window(window_t1, "window_t1")
    yt = dt.date.fromisoformat(t[0]).year
    yt1 = dt.date.fromisoformat(t1[0]).year
    if yt1 <= yt:
        raise DomainError(f"window_t1 year ({yt1}) must be after window_t year "
                          f"({yt}) - this is a change-detection model.")
    return {"window_t": list(t), "window_t1": list(t1)}


def resolve_request(region_id: str | None, bbox_wsen, window_t, window_t1,
                    center=None, radius_km=None) -> dict:
    """Return a normalised, in-domain job spec or raise DomainError.

    Selection priority: preset region_id -> point (center + radius_km) ->
    raw bbox_wsen (advanced/fallback).
    """
    derived = None
    if region_id:
        preset = preset_by_id(region_id)
        if preset is None:
            raise DomainError(f"unknown preset region '{region_id}'.")
        bbox = preset["bbox_wsen"]
        src = {"region_id": region_id, "region_name": preset["name"],
               "in_training_set": preset["in_training_set"],
               "selection": "preset"}
    elif center is not None and radius_km is not None:
        derived = derive_bbox_from_point(center[0], center[1], radius_km)
        bbox = validate_bbox(derived["bbox_wsen"], enforce_area_cap=False)
        src = {"region_id": None, "region_name": None, "in_training_set": False,
               "selection": "point_radius"}
    elif bbox_wsen is not None:
        bbox = validate_bbox(bbox_wsen)
        src = {"region_id": None, "region_name": None, "in_training_set": False,
               "selection": "bbox"}
    else:
        raise DomainError("provide region_id, or center + radius_km, or bbox_wsen.")

    windows = validate_windows(window_t, window_t1)

    if src["selection"] == "preset":
        case, case_region = ("pooled", region_id) if src["in_training_set"] \
            else ("loro", None)
    else:
        case, case_region = metric_case_for_bbox(bbox)
        src["in_training_set"] = case == "pooled"

    area_km2 = round(bbox_area_km2(bbox), 2)
    return {
        "bbox_wsen": [float(x) for x in bbox],
        "area_km2": area_km2,
        "small_area": area_km2 < SMALL_AREA_KM2,
        "small_area_threshold_km2": SMALL_AREA_KM2,
        "metric_case": case,
        "metric_case_region": case_region,
        "derived": derived,
        **windows,
        **src,
    }
