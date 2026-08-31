"""Phase 8 site selection: Hansen GFC 2019-2020 loss density + Sentinel-2
availability for candidate Western Ghats blocks.

Run before committing any bbox. A block with near-zero loss adds patches but no
positives, which is worse than useless at ~0.3% prevalence.

    python scripts/site_probe.py                 # probe the committed regions
    python scripts/site_probe.py --candidates    # probe the wider candidate set
"""

from __future__ import annotations

import argparse

import ee

from src.common import load_yaml
from src.regions import load_regions

GFC_DEFAULT = "UMD/hansen/global_forest_change_2024_v1_12"
S2 = "COPERNICUS/S2_SR_HARMONIZED"
WINDOWS = [("T", "2019-01-01", "2019-04-15"), ("T+1", "2021-01-01", "2021-04-15")]

# wider candidate set used during selection (kept for the record)
CANDIDATES = {
    "wayanad":            [76.00, 11.55, 76.28, 11.80],
    "kodagu_madikeri":    [75.60, 12.30, 75.88, 12.55],
    "kodagu_virajpet":    [75.70, 12.05, 75.98, 12.30],
    "kodagu_brahmagiri":  [75.75, 11.90, 76.03, 12.15],
    "nilgiris_gudalur":   [76.35, 11.45, 76.63, 11.70],
    "nilgiris_ooty":      [76.60, 11.35, 76.88, 11.60],
    "nilgiris_kotagiri":  [76.75, 11.35, 77.03, 11.60],
    "anamalai_valparai":  [76.85, 10.25, 77.13, 10.50],
    "anamalai_foothill":  [76.90, 10.35, 77.18, 10.60],
    "anamalai_topslip":   [76.75, 10.40, 77.03, 10.65],
}


def probe(name: str, box: list[float], gfc: ee.Image, canopy: int, codes: list[int]) -> None:
    aoi = ee.Geometry.Rectangle(box, "EPSG:4326", geodesic=False)
    area_km2 = round(aoi.area(1).getInfo() / 1e6, 1)
    forest = gfc.select("treecover2000").gte(canopy).And(gfc.select("datamask").eq(1))
    ly = gfc.select("lossyear")
    loss = ly.eq(codes[0])
    for c in codes[1:]:
        loss = loss.Or(ly.eq(c))
    loss = loss.And(forest)
    px = ee.Image.pixelArea()
    f_ha = ee.Number(forest.multiply(px).reduceRegion(
        ee.Reducer.sum(), aoi, 30, maxPixels=1e10).get("treecover2000")).divide(1e4).getInfo()
    l_ha = ee.Number(loss.multiply(px).reduceRegion(
        ee.Reducer.sum(), aoi, 30, maxPixels=1e10).get("lossyear")).divide(1e4).getInfo()
    prev = 100 * l_ha / f_ha if f_ha else 0.0
    s2 = []
    for lbl, a, b in WINDOWS:
        col = ee.ImageCollection(S2).filterBounds(aoi).filterDate(a, b)
        n = col.size().getInfo()
        cl = sorted(col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo())
        s2.append(f"{lbl} {n}sc/{cl[len(cl)//2]:.0f}%" if cl else f"{lbl} 0sc")
    print(f"  {name:20s} {box} ~{area_km2:6.1f} km2 | forest {f_ha:9.1f} ha | "
          f"loss {l_ha:7.2f} ha | prev {prev:.4f}% | {' '.join(s2)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true",
                    help="probe the wider candidate set instead of the committed regions")
    args = ap.parse_args()

    cfg = load_yaml("configs/region.yaml")
    ee.Initialize(project=cfg["earth_engine"]["project"])
    gfc = ee.Image(cfg["ground_truth"]["gee_asset"])
    canopy = int(cfg["ground_truth"]["canopy_threshold_pct"])
    codes = list(cfg["ground_truth"]["loss_year_codes"])
    print(f"GFC {cfg['ground_truth']['gee_asset']} | canopy>={canopy}% | lossyear {codes}\n")

    if args.candidates:
        for name, box in CANDIDATES.items():
            probe(name, box, gfc, canopy, codes)
    else:
        for r in load_regions(cfg):
            probe(r["id"], r["bbox_wsen"], gfc, canopy, codes)


if __name__ == "__main__":
    main()
