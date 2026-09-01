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

from .config import MAX_AREA_KM2, REGION_CFG

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


def training_windows() -> dict:
    tw = _cfg()["time_windows"]
    return {"T": dict(tw["T"]), "T_plus_1": dict(tw["T_plus_1"])}


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
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(mean_lat)
    return abs((e - w) * km_per_deg_lon) * abs((n - s) * km_per_deg_lat)


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


def validate_bbox(wsen) -> list[float]:
    ext = domain_extent()
    _require_inside(wsen, ext)
    area = bbox_area_km2(wsen)
    if area > MAX_AREA_KM2:
        raise DomainError(
            f"requested area {area:.0f} km2 exceeds the {MAX_AREA_KM2:.0f} km2 "
            f"cap. Split the area into smaller requests.")
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


def resolve_request(region_id: str | None, bbox_wsen, window_t, window_t1) -> dict:
    """Return a normalised, in-domain job spec or raise DomainError."""
    if region_id:
        preset = preset_by_id(region_id)
        if preset is None:
            raise DomainError(f"unknown preset region '{region_id}'.")
        bbox = preset["bbox_wsen"]
        src = {"region_id": region_id, "region_name": preset["name"],
               "in_training_set": preset["in_training_set"]}
    elif bbox_wsen is not None:
        bbox = validate_bbox(bbox_wsen)
        src = {"region_id": None, "region_name": None, "in_training_set": False}
    else:
        raise DomainError("provide either region_id or bbox_wsen.")

    windows = validate_windows(window_t, window_t1)
    return {
        "bbox_wsen": [float(x) for x in bbox],
        "area_km2": round(bbox_area_km2(bbox), 2),
        **windows,
        **src,
    }
