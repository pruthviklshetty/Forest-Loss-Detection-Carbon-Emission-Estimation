"""Phase 2, step 2 - convert the Hansen GFC raster into a binary forest-loss
label aligned to the Sentinel-2 grid.

This is the documented GFC-conversion sub-step. Hansen's `lossyear` is a
year-of-loss CODE (0 = no loss detected 2001-2024; k = stand-replacement loss
in calendar year 2000+k), NOT a T-vs-T+1 change mask. The conversion:

    forest2000  := treecover2000 >= CANOPY_THRESHOLD_PCT        (canopy cover at 2000)
    land        := datamask == 1                                (mapped land; drop water / no-data)
    loss_window := lossyear in LOSS_YEAR_CODES                   ({19, 20} = calendar 2019 & 2020)

    label = 1  where  forest2000 AND land AND loss_window        (positive: new forest loss T -> T+1)
    label = 0  elsewhere on land
    valid = 1  where  land, else 0                               (pixels usable for training / metrics)

Why loss codes {19, 20}: the T composite (Jan-Apr 2019) represents the canopy
state entering 2019 and the T+1 composite (Jan-Apr 2021) the state entering
2021, so stand-replacement loss dated to calendar 2019 or 2020 is exactly what
occurred between the two acquisitions. Loss dated <= 18 already happened before
T (those pixels are non-forest in the T image and are treated as negatives, not
positives). Loss dated >= 21 happened after T+1 and is likewise a negative.

Runs per region. Outputs (uint8, region UTM, 10 m, same footprint as
data/raw/<id>/s2_*.tif):
    data/masks/<id>/loss_label.tif    0 / 1   binary forest-loss target
    data/masks/<id>/valid_mask.tif    0 / 1   1 = usable pixel (datamask == 1)
    data/masks/<id>/forest2000.tif    0 / 1   canopy >= threshold at 2000
    data/masks/<id>/loss_label_summary.json

Run:  python -m src.preprocessing.build_labels                    # all regions
      python -m src.preprocessing.build_labels --regions kodagu
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import rasterio

from ..paths import masks_dir as _masks_dir
from ..regions import load_regions
from .eeutil import load_cfg

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _read_gfc(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Return {'treecover2000','lossyear','datamask'} arrays, matching by band
    description when present, else by the known download order."""
    order = ["treecover2000", "lossyear", "datamask"]
    with rasterio.open(path) as src:
        arrs = src.read()
        desc = list(src.descriptions)
        profile = src.profile
        transform = src.transform
        crs = src.crs
    if all(desc) and set(order).issubset(desc):
        idx = {d: i for i, d in enumerate(desc)}
        out = {name: arrs[idx[name]] for name in order}
    else:
        # EE GeoTIFF export drops band names; fall back to the known
        # download order (treecover2000, lossyear, datamask).
        out = {name: arrs[i] for i, name in enumerate(order)}
    out["_profile"] = profile
    out["_transform"] = transform
    out["_crs"] = crs
    return out


def build_region(region: dict, cfg: dict) -> dict:
    rid = region["id"]
    canopy_thr = float(cfg["ground_truth"]["canopy_threshold_pct"])
    loss_codes = list(cfg["ground_truth"]["loss_year_codes"])
    gsd = int(region["gsd_m"])
    mdir = _masks_dir(cfg, rid)
    gfc_path = mdir / "hansen_gfc_raw.tif"
    if not gfc_path.exists():
        raise SystemExit(f"missing {gfc_path}; run src.preprocessing.download_data first")

    g = _read_gfc(gfc_path)
    treecover = g["treecover2000"].astype(np.float32)
    lossyear = g["lossyear"].astype(np.int16)
    datamask = g["datamask"].astype(np.int16)

    forest2000 = treecover >= canopy_thr
    land = datamask == 1
    loss_window = np.isin(lossyear, loss_codes)
    label = (forest2000 & land & loss_window).astype(np.uint8)
    valid = land.astype(np.uint8)
    forest2000_u8 = forest2000.astype(np.uint8)

    prof = g["_profile"].copy()
    prof.update(count=1, dtype="uint8", nodata=None, compress="deflate")

    def _write(name: str, arr: np.ndarray, band_desc: str) -> None:
        with rasterio.open(mdir / name, "w", **prof) as dst:
            dst.write(arr, 1)
            dst.descriptions = (band_desc,)
        print(f"  -> {mdir.name}/{name}  ({arr.shape[1]}x{arr.shape[0]}, sum={int(arr.sum())})")

    _write("loss_label.tif", label, "forest_loss_T_to_T1")
    _write("valid_mask.tif", valid, "datamask_eq_1")
    _write("forest2000.tif", forest2000_u8, f"treecover2000_ge_{int(canopy_thr)}pct")

    n, n_valid, n_pos, n_forest = label.size, int(valid.sum()), int(label.sum()), int(forest2000_u8.sum())
    lossyear_hist = {int(k): int(v) for k, v in zip(*np.unique(lossyear[land], return_counts=True))}
    summary = {
        "region_id": rid, "region_name": region["name"],
        "source_raster": str(gfc_path.relative_to(_REPO)),
        "gee_asset": cfg["ground_truth"]["gee_asset"],
        "canopy_threshold_pct": canopy_thr, "loss_year_codes": loss_codes,
        "grid": {"width": int(label.shape[1]), "height": int(label.shape[0]),
                 "crs": str(g["_crs"]), "gsd_m": gsd},
        "pixels": {
            "total": n, "valid_land": n_valid,
            "valid_land_pct": round(100 * n_valid / n, 3),
            "forest2000": n_forest,
            "forest2000_pct_of_land": round(100 * n_forest / max(n_valid, 1), 3),
            "loss_positive": n_pos,
            "loss_positive_pct_of_land": round(100 * n_pos / max(n_valid, 1), 4),
            "loss_positive_pct_of_forest2000": round(100 * n_pos / max(n_forest, 1), 4),
        },
        "lossyear_histogram_over_land": lossyear_hist,
        "ha_lost_gfc_reference": round(n_pos * gsd * gsd / 1e4, 2),
    }
    (mdir / "loss_label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  {rid}: {summary['ha_lost_gfc_reference']} ha loss, "
          f"{summary['pixels']['loss_positive_pct_of_forest2000']}% of forest2000")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=None)
    ap.add_argument("--config", default="configs/region.yaml",
                    help="config file (use configs/period_2021_2023.yaml for Phase 10)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if cfg.get("period_id"):
        print(f"period: {cfg['period_id']}  |  loss_year_codes {cfg['ground_truth']['loss_year_codes']}")
    regions = load_regions(cfg)
    if args.regions:
        want = {s.strip() for s in args.regions.split(",")}
        regions = [r for r in regions if r["id"] in want]

    totals = {"ha": 0.0, "loss_px": 0, "forest_px": 0}
    for region in regions:
        s = build_region(region, cfg)
        totals["ha"] += s["ha_lost_gfc_reference"]
        totals["loss_px"] += s["pixels"]["loss_positive"]
        totals["forest_px"] += s["pixels"]["forest2000"]
    print(f"\nPOOLED across {len(regions)} region(s): "
          f"{totals['ha']:.1f} ha GFC loss, "
          f"{100 * totals['loss_px'] / max(totals['forest_px'], 1):.4f}% of forest2000")


if __name__ == "__main__":
    main()
