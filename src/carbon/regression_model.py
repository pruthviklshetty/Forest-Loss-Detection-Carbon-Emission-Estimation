"""Phase 6 contribution: an NDVI -> continuous carbon-density regression.

Replaces the piecewise-constant 3-bin baseline with a continuous mapping,
**calibrated against literature-reported regional carbon densities, not
pixel-aligned biomass rasters** (a deliberate, documented simplification).

Calibration set (see data/carbon/reference_table.csv):
  - carbon densities: field-inventory aboveground carbon (tC/ha) for Western
    Ghats forest types from Padmakumar et al. 2018, Kothandaraman et al. 2020
    (Jose et al. 2025 as an unfitted cross-check).
  - NDVI for each row: the matching forest-cover percentile of THIS study
    region's Year-T Sentinel-2 composite, so the curve is anchored to the same
    sensor/'scene it will be applied to.

Two models are fitted and reported:
  - linear:       AGC = a * NDVI + b            (the "start simple" variant)
  - exponential:  AGC = exp(a * NDVI + b)       (monotone, non-negative; primary)
The exponential fit is the primary model; both are reported side by side with
the 3-bin baseline.

    python -m src.carbon.regression_model        # fits, prints, writes coefs json
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import rasterio

from ..common import REPO

CARBON_DIR = REPO / "data" / "carbon"
RAW = REPO / "data" / "raw"
MASKS = REPO / "data" / "masks"
OUT = REPO / "results" / "carbon_validation"

# forest2000 NDVI percentile -> reference_table row key
ANCHOR_PCTILE = {
    "dry_open_forest": 5,
    "degraded_moist_deciduous": 20,
    "moist_deciduous": 40,
    "semi_evergreen": 60,
    "semi_evergreen_dense": 72,
    "evergreen": 85,
    "evergreen_dense": 97,
    "evergreen_oldgrowth": 99,
}


def _year_t_ndvi_over_forest() -> np.ndarray:
    with rasterio.open(RAW / "s2_T.tif") as s:
        a = s.read()
    red, nir = a[1], a[2]
    ndvi = (nir - red) / (nir + red + 1e-6)
    with rasterio.open(MASKS / "forest2000.tif") as s:
        forest = s.read(1).astype(bool)
    with rasterio.open(MASKS / "valid_mask.tif") as s:
        valid = s.read(1).astype(bool)
    m = forest & valid & np.isfinite(ndvi)
    return ndvi[m]


def build_calibration() -> list[dict]:
    rows = {}
    with open(CARBON_DIR / "reference_table.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["vegetation_class"]] = r
    forest_ndvi = _year_t_ndvi_over_forest()
    cal = []
    for cls, pct in ANCHOR_PCTILE.items():
        ndvi_v = float(np.percentile(forest_ndvi, pct))
        cal.append({
            "class": cls,
            "ndvi": round(ndvi_v, 4),
            "agc_MgC_ha": float(rows[cls]["agc_MgC_ha"]),
            "source": rows[cls]["source_key"],
        })
    return cal


def fit(cal: list[dict]) -> dict:
    x = np.array([c["ndvi"] for c in cal], float)
    y = np.array([c["agc_MgC_ha"] for c in cal], float)

    a_lin, b_lin = np.polyfit(x, y, 1)
    yhat_lin = a_lin * x + b_lin
    r2_lin = 1 - np.sum((y - yhat_lin) ** 2) / np.sum((y - y.mean()) ** 2)

    a_exp, b_exp = np.polyfit(x, np.log(y), 1)
    yhat_exp = np.exp(a_exp * x + b_exp)
    r2_exp = 1 - np.sum((y - yhat_exp) ** 2) / np.sum((y - y.mean()) ** 2)

    return {
        "calibration_points": cal,
        "n_points": len(cal),
        "linear": {"a": float(a_lin), "b": float(b_lin), "r2": float(r2_lin),
                   "form": "AGC = a*NDVI + b, clipped at 0"},
        "exponential": {"a": float(a_exp), "b": float(b_exp), "r2": float(r2_exp),
                        "form": "AGC = exp(a*NDVI + b)"},
        "primary": "exponential",
        "note": ("Calibrated against literature-reported regional aboveground "
                 "carbon densities, NOT pixel-aligned biomass rasters. n is "
                 "small (8 anchors); r2 describes curve fit, not held-out "
                 "accuracy."),
    }


def predict(ndvi: np.ndarray, coefs: dict, model: str | None = None) -> np.ndarray:
    model = model or coefs["primary"]
    if model == "linear":
        c = coefs["linear"]
        return np.clip(c["a"] * ndvi + c["b"], 0.0, None).astype(np.float32)
    if model == "exponential":
        c = coefs["exponential"]
        return np.exp(c["a"] * ndvi + c["b"]).astype(np.float32)
    raise ValueError(model)


def load_or_fit(path: pathlib.Path | None = None) -> dict:
    path = path or (OUT / "regression_coefs.json")
    if path.exists():
        return json.loads(path.read_text())
    coefs = fit(build_calibration())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coefs, indent=2), encoding="utf-8")
    return coefs


def main() -> None:
    coefs = fit(build_calibration())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regression_coefs.json").write_text(json.dumps(coefs, indent=2), encoding="utf-8")
    print("calibration anchors (NDVI -> AGC tC/ha):")
    for c in coefs["calibration_points"]:
        print(f"  {c['class']:26s} NDVI {c['ndvi']:.3f}  AGC {c['agc_MgC_ha']:6.1f}  [{c['source']}]")
    print(f"\nlinear      : AGC = {coefs['linear']['a']:.1f}*NDVI + "
          f"{coefs['linear']['b']:.1f}   r2={coefs['linear']['r2']:.3f}")
    print(f"exponential : AGC = exp({coefs['exponential']['a']:.3f}*NDVI + "
          f"{coefs['exponential']['b']:.3f})   r2={coefs['exponential']['r2']:.3f}   [primary]")
    print(f"\n-> {OUT / 'regression_coefs.json'}")


if __name__ == "__main__":
    main()
