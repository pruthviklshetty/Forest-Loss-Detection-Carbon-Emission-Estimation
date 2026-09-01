"""Phase 2/8, step 1 - raw data acquisition, per region.

For every region in configs/region.yaml (see src/regions.py), pulls:
  1. A cloud-masked Sentinel-2 SR median composite for window T   -> data/raw/<id>/s2_T.tif
  2. The same for window T+1                                       -> data/raw/<id>/s2_T1.tif
     Each has 4 float32 bands in [0,1]: green(B3), red(B4), nir(B8), swir1(B11).
  3. Hansen GFC treecover2000 / lossyear / datamask               -> data/masks/<id>/hansen_gfc_raw.tif
     resampled (nearest) onto the same 10 m UTM grid as the S2 rasters.

Per region the three rasters share that region's UTM CRS, 10 m pixels and an
identical footprint, so they stack pixel-for-pixel. A per-region manifest is
written to data/raw/<id>/manifest.json.

Run:  python -m src.preprocessing.download_data                 # all regions
      python -m src.preprocessing.download_data --regions kodagu,nilgiris
      python -m src.preprocessing.download_data --skip-existing
"""

from __future__ import annotations

import argparse
import pathlib

import ee

from ..paths import masks_dir as _masks_dir
from ..paths import raw_dir as _raw_dir
from ..regions import load_regions
from .eeutil import download_image_tiled, init_ee, load_cfg, write_manifest

_REPO = pathlib.Path(__file__).resolve().parents[2]

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CS_BAND = "cs_cdf"          # Cloud Score+ cumulative-distribution score, 0..1
CS_KEEP_THRESH = 0.60       # keep pixels with cs_cdf >= this (Google-suggested 0.6)
S2_BANDS = ["B3", "B4", "B8", "B11"]
S2_BAND_NAMES = ["green", "red", "nir", "swir1"]
S2_REFLECTANCE_SCALE = 10000.0   # DN -> reflectance; HARMONIZED needs no baseline offset


def s2_composite(aoi: ee.Geometry, start: str, end: str) -> tuple[ee.Image, dict]:
    """Cloud-masked S2 SR median composite, 4 bands, float32 reflectance in [0,1]."""
    base = (
        ee.ImageCollection(S2_SR)
        .filterBounds(aoi)
        .filterDate(start, end)
    )
    csp = ee.ImageCollection(CLOUD_SCORE_PLUS)
    linked = base.linkCollection(csp, [CS_BAND])

    def _mask(img: ee.Image) -> ee.Image:
        keep = img.select(CS_BAND).gte(CS_KEEP_THRESH)
        # bilinear so the 20 m SWIR1 band upsamples smoothly to the 10 m grid
        return img.updateMask(keep).resample("bilinear")

    masked = linked.map(_mask)
    comp = (
        masked.select(S2_BANDS, S2_BAND_NAMES)
        .median()
        .divide(S2_REFLECTANCE_SCALE)
        .clamp(0.0, 1.0)
        .toFloat()
    )
    stats = {
        "collection": S2_SR,
        "cloud_mask": {
            "collection": CLOUD_SCORE_PLUS,
            "band": CS_BAND,
            "keep_threshold": CS_KEEP_THRESH,
        },
        "date_start": start,
        "date_end": end,
        "n_scenes": base.size().getInfo(),
        "scene_ids": base.aggregate_array("system:index").getInfo(),
        "cloudy_pixel_pct_sorted": sorted(
            round(float(x), 2)
            for x in base.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        ),
        "bands_src": S2_BANDS,
        "bands_out": S2_BAND_NAMES,
        "reflectance_scale": S2_REFLECTANCE_SCALE,
        "reducer": "median",
    }
    return comp, stats


def download_region(region: dict, cfg: dict, project: str, skip_existing: bool) -> None:
    rid = region["id"]
    bbox = region["bbox_wsen"]
    crs = f"EPSG:{region['utm_epsg']}"
    scale = int(region["gsd_m"])
    aoi = ee.Geometry.Rectangle(bbox, "EPSG:4326", geodesic=False)
    tw = cfg["time_windows"]
    raw_dir = _raw_dir(cfg, rid)
    masks_dir = _masks_dir(cfg, rid)

    want = [raw_dir / "s2_T.tif", raw_dir / "s2_T1.tif", masks_dir / "hansen_gfc_raw.tif"]
    if skip_existing and all(p.exists() for p in want):
        print(f"\n=== {rid}: all rasters present, skipping ===")
        return

    print(f"\n=== {rid}  {region['name']} ===\n  bbox {bbox}  {crs}  {scale} m")
    manifest: dict = {
        "region_id": rid, "region_name": region["name"],
        "bbox_wsen_4326": bbox, "crs": crs, "gsd_m": scale,
        "earth_engine_project": project, "outputs": {},
    }

    for key, out_name in (("T", "s2_T.tif"), ("T_plus_1", "s2_T1.tif")):
        win = tw[key]
        print(f"\n[Sentinel-2 {win['label']}]  {win['start']} .. {win['end']}")
        comp, stats = s2_composite(aoi, win["start"], win["end"])
        print(f"  {stats['n_scenes']} scenes in window")
        out = download_image_tiled(comp, bbox, raw_dir / out_name, crs=crs, scale_m=scale,
                                   band_names=S2_BAND_NAMES)
        manifest["outputs"][out_name] = {"path": str(out.relative_to(_REPO)),
                                         "window": key, **stats}

    gfc_asset = cfg["ground_truth"]["gee_asset"]
    print(f"\n[Hansen GFC]  {gfc_asset}")
    gfc = ee.Image(gfc_asset).select(["treecover2000", "lossyear", "datamask"])
    out = download_image_tiled(
        gfc, bbox, masks_dir / "hansen_gfc_raw.tif", crs=crs, scale_m=scale,
        band_names=["treecover2000", "lossyear", "datamask"],
    )
    manifest["outputs"]["hansen_gfc_raw.tif"] = {
        "path": str(out.relative_to(_REPO)),
        "asset": gfc_asset,
        "bands": ["treecover2000", "lossyear", "datamask"],
        "note": ("Native 30 m, resampled nearest to the 10 m S2 grid on download "
                 "so labels align pixel-for-pixel. lossyear: 0 = no loss, "
                 "k = loss in year 2000+k."),
    }
    write_manifest(raw_dir / "manifest.json", manifest)
    print(f"  -> data/raw/{rid}/ , data/masks/{rid}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=None, help="comma-separated region ids (default: all)")
    ap.add_argument("--config", default="configs/region.yaml",
                    help="config file (use configs/period_2021_2023.yaml for Phase 10)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a region whose three rasters already exist")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    project = init_ee(cfg)
    if cfg.get("period_id"):
        print(f"period: {cfg['period_id']}  ->  data/*/{cfg['period_id']}/")
    print(f"Earth Engine initialised on project: {project}")

    regions = load_regions(cfg)
    if args.regions:
        want = {s.strip() for s in args.regions.split(",")}
        regions = [r for r in regions if r["id"] in want]
        if not regions:
            raise SystemExit(f"no matching regions for {args.regions}")
    print(f"regions: {[r['id'] for r in regions]}")

    for region in regions:
        download_region(region, cfg, project, args.skip_existing)
    print("\nDone.")


if __name__ == "__main__":
    main()
