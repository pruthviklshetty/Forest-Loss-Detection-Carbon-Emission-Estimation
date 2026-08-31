"""Region list access for the (now multi-region) pipeline.

`load_regions(cfg)` returns a list of normalised region dicts:

    {
      "id":            str,           # short slug, unique, used in paths / index
      "name":          str,
      "bbox_wsen":     [W, S, E, N],  # EPSG:4326
      "utm_epsg":      int,
      "gsd_m":         int,           # shared target GSD
      "admin_context": str,
      "gfc_loss_ha_2019_20": float | None,
    }

Two config shapes are supported:
  - `regions:` — a list (Phase 8 onward). Preferred.
  - a bare `region:` block with `short_id` / `bbox.wsen` / `utm_epsg`
    (Phase 1-7 single-region form) — wrapped into a one-element list.
"""

from __future__ import annotations

import pathlib

from .common import load_yaml

_DEF_CFG = "configs/region.yaml"


def _norm(entry: dict, gsd_m: int) -> dict:
    box = entry.get("bbox", {})
    wsen = box.get("wsen") if isinstance(box, dict) else box
    if wsen is None and isinstance(box, dict):
        wsen = [box["lon_min"], box["lat_min"], box["lon_max"], box["lat_max"]]
    rid = entry.get("id") or entry.get("short_id") or entry.get("name", "region")
    rid = str(rid).strip().lower().replace(" ", "_")
    return {
        "id": rid,
        "name": entry.get("name", rid),
        "bbox_wsen": [float(x) for x in wsen],
        "utm_epsg": int(entry.get("utm_epsg", 32643)),
        "gsd_m": int(entry.get("target_gsd_m", gsd_m)),
        "admin_context": (entry.get("admin_context") or "").strip(),
        "gfc_loss_ha_2019_20": entry.get("gfc_loss_ha_2019_20"),
    }


def load_regions(cfg: dict | str | pathlib.Path | None = None) -> list[dict]:
    if cfg is None:
        cfg = load_yaml(_DEF_CFG)
    elif isinstance(cfg, (str, pathlib.Path)):
        cfg = load_yaml(cfg)

    gsd_m = int(cfg.get("target_gsd_m") or cfg.get("region", {}).get("target_gsd_m", 10))

    if cfg.get("regions"):
        regions = [_norm(r, gsd_m) for r in cfg["regions"]]
    elif cfg.get("region"):
        regions = [_norm(cfg["region"], gsd_m)]
    else:
        raise SystemExit("configs/region.yaml has neither `regions:` nor `region:`")

    ids = [r["id"] for r in regions]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate region ids: {ids}")
    return regions


def region_by_id(rid: str, cfg=None) -> dict:
    for r in load_regions(cfg):
        if r["id"] == rid:
            return r
    raise SystemExit(f"no region with id '{rid}' in config")


def domain_extent(cfg=None) -> list[float] | None:
    if cfg is None:
        cfg = load_yaml(_DEF_CFG)
    elif isinstance(cfg, (str, pathlib.Path)):
        cfg = load_yaml(cfg)
    return cfg.get("domain_extent_wsen")
