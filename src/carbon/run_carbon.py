"""Phase 6 / Phase 8, apply step - turn deforested area into tonnes of CO2, per
region and pooled.

For every deforested pixel, take its Year-T (pre-clearing) NDVI, map it to an
aboveground carbon density with (a) the 3-bin baseline and (b) the calibrated
regression, multiply by the pixel area (0.01 ha) to get tonnes C lost, and
convert to CO2 with the 44/12 molecular-mass ratio.

Per region, applied to four pixel sets so the honest held-out number is
separable:
  predicted loss (test blocks)      <- headline
  predicted loss (full region)
  Hansen GFC loss (test blocks)     <- reference
  Hansen GFC loss (full region)     <- reference
A `pooled` block sums tonnes C / CO2 across regions for each set.

Assumptions (stated, not hidden): only aboveground carbon; complete committed
emission of that carbon on clearing; no belowground / deadwood / litter / soil
pools; no post-clearing regrowth; NDVI->carbon calibrated to literature regional
means, not pixel-matched biomass. The regression is fit once on 8 Western-Ghats
anchors; the 3-bin NDVI cut points are each region's own forest terciles.

    python -m src.carbon.run_carbon --experiment p8_pooled_unet_s43
    python -m src.carbon.run_carbon --experiment baseline_unet --regions wayanad
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import rasterio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common import REPO, RESULTS  # noqa: E402
from ..paths import masks_dir as _masks_dir  # noqa: E402
from ..paths import proc_dir as _proc_dir  # noqa: E402
from ..paths import raw_dir as _raw_dir  # noqa: E402
from ..regions import load_regions  # noqa: E402
from .ndvi import THREE_BIN_VALUES_tC_ha, compute_ndvi, three_bin_carbon_density  # noqa: E402
from .regression_model import fit, build_calibration, predict  # noqa: E402

# rebound in main() when --period is given
RAW = REPO / "data" / "raw"
MASKS = REPO / "data" / "masks"
PROC = REPO / "data" / "processed"
DEFOR = RESULTS / "deforestation"
OUT = RESULTS / "carbon_validation"
FIG = RESULTS / "figures"
_FIG_SUFFIX = ""

HA_PER_PX = 0.01
CO2_PER_C = 44.0 / 12.0
BIN_LO_PCTILE, BIN_HI_PCTILE = 33.3, 66.7
_SETNAMES = ["predicted_test", "gfc_test", "predicted_full_region", "gfc_full_region"]


def _first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    raise SystemExit(f"missing raster; looked for {[str(p) for p in paths]}")


def _read(path, band=1):
    with rasterio.open(path) as s:
        return s.read(band)


def _test_map(rid, H, W):
    sm = np.zeros((H, W), bool)
    with open(PROC / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("region", rid) != rid:
                continue
            if int(r["is_overlap"]):
                continue
            split = r.get("pooled_split") or r["split"]
            if split != "test":
                continue
            r0, c0, p = int(r["px_r0"]), int(r["px_c0"]), int(r["size"])
            sm[r0:r0 + p, c0:c0 + p] = True
    return sm


def _totals(ndvi_vals, carbon_density_fn):
    if ndvi_vals.size == 0:
        return {"tC": 0.0, "tCO2": 0.0, "mean_AGC_tC_ha": 0.0, "n_px": 0}
    agc = carbon_density_fn(ndvi_vals)                 # tC/ha per pixel
    tC = float(np.sum(agc) * HA_PER_PX)
    return {"tC": round(tC, 1), "tCO2": round(tC * CO2_PER_C, 1),
            "mean_AGC_tC_ha": round(float(np.mean(agc)), 1), "n_px": int(ndvi_vals.size)}


def _region_estimates(rid, exp, coefs):
    p_t = _first_existing(RAW / rid / "s2_T.tif", RAW / "s2_T.tif")
    with rasterio.open(p_t) as s:
        a = s.read()
    ndvi = compute_ndvi(a[2], a[1])
    H, W = ndvi.shape

    pred = _read(DEFOR / f"{exp}__{rid}_loss.tif").astype(bool)[:H, :W]
    gfc = _read(_first_existing(MASKS / rid / "loss_label.tif",
                                MASKS / "loss_label.tif")).astype(bool)[:H, :W]
    valid = _read(_first_existing(MASKS / rid / "valid_mask.tif",
                                  MASKS / "valid_mask.tif")).astype(bool)[:H, :W] & np.isfinite(ndvi)
    forest = _read(_first_existing(MASKS / rid / "forest2000.tif",
                                   MASKS / "forest2000.tif")).astype(bool)[:H, :W] & valid
    test = _test_map(rid, H, W)

    lo = float(np.percentile(ndvi[forest], BIN_LO_PCTILE))
    hi = float(np.percentile(ndvi[forest], BIN_HI_PCTILE))

    methods = {
        "three_bin_baseline": lambda v: three_bin_carbon_density(v, lo, hi),
        "regression_linear": lambda v: predict(v, coefs, "linear"),
        "regression_exponential_primary": lambda v: predict(v, coefs, "exponential"),
    }
    sets = {
        "predicted_test": pred & test & valid,
        "predicted_full_region": pred & valid,
        "gfc_test": gfc & test & valid,
        "gfc_full_region": gfc & valid,
    }
    est = {"ndvi_lo": round(lo, 4), "ndvi_hi": round(hi, 4)}
    for sname, mask in sets.items():
        v = ndvi[mask]
        est[sname] = {m: _totals(v, fn) for m, fn in methods.items()}
        est[sname]["area_ha"] = round(int(mask.sum()) * HA_PER_PX, 1)
    return est


def _pool_estimates(per_region):
    methods = ["three_bin_baseline", "regression_linear", "regression_exponential_primary"]
    pooled = {}
    for sname in _SETNAMES:
        pooled[sname] = {}
        area = sum(per_region[r][sname]["area_ha"] for r in per_region)
        for m in methods:
            tC = sum(per_region[r][sname][m]["tC"] for r in per_region)
            npx = sum(per_region[r][sname][m]["n_px"] for r in per_region)
            pooled[sname][m] = {
                "tC": round(tC, 1), "tCO2": round(tC * CO2_PER_C, 1),
                "mean_AGC_tC_ha": round(tC / area, 1) if area else 0.0,
                "n_px": npx}
        pooled[sname]["area_ha"] = round(area, 1)
    return pooled


def _co2_figure(estimates, exp):
    order = ["three_bin_baseline", "regression_linear", "regression_exponential_primary"]
    groups = list(estimates)                    # region ids + "pooled"
    fig, ax = plt.subplots(figsize=(2.2 * len(groups) + 4, 5))
    x = np.arange(len(groups))
    w = 0.26
    for i, m in enumerate(order):
        vals = [estimates[g]["predicted_test"][m]["tCO2"] for g in groups]
        b = ax.bar(x + (i - 1) * w, vals, w, label=m)
        ax.bar_label(b, fmt="%.0f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=10)
    ax.set_ylabel("tonnes CO2 (committed, aboveground)")
    ax.set_title(f"CO2 from predicted loss on held-out test blocks ({exp})")
    ax.legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"phase6_co2_estimates{_FIG_SUFFIX}.png", dpi=110)
    plt.close(fig)


def _calibration_figure(coefs):
    cal = coefs["calibration_points"]
    xs = np.linspace(0.30, 0.90, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([c["ndvi"] for c in cal], [c["agc_MgC_ha"] for c in cal],
               c="k", zorder=5, label="literature anchors (Western Ghats)")
    ax.plot(xs, np.clip(coefs["linear"]["a"] * xs + coefs["linear"]["b"], 0, None),
            "--", label=f"linear (r2={coefs['linear']['r2']:.2f})")
    ax.plot(xs, np.exp(coefs["exponential"]["a"] * xs + coefs["exponential"]["b"]),
            "-", lw=2, label=f"exponential, primary (r2={coefs['exponential']['r2']:.2f})")
    ax.set_xlabel("NDVI (Year-T composite)")
    ax.set_ylabel("aboveground carbon density (tC/ha)")
    ax.set_title("NDVI -> carbon density: calibrated regression on 8 anchors")
    ax.legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"phase6_carbon_calibration{_FIG_SUFFIX}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    global RAW, MASKS, PROC, OUT, _FIG_SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline_unet")
    ap.add_argument("--regions", default=None,
                    help="comma-separated region ids (default: all in configs/region.yaml)")
    ap.add_argument("--period", default=None,
                    help="read data/*/<period>/, write results/carbon_validation/<period>/ "
                         "(e.g. 2021_2023) so nothing overwrites the 2019-2021 carbon output")
    args = ap.parse_args()
    exp = args.experiment
    if args.period:
        RAW, MASKS, PROC = (_raw_dir(period=args.period), _masks_dir(period=args.period),
                            _proc_dir(period=args.period))
        OUT = RESULTS / "carbon_validation" / args.period
        _FIG_SUFFIX = f"_{args.period}"
        print(f"period: {args.period}  ->  {OUT.relative_to(REPO).as_posix()}/")
    OUT.mkdir(parents=True, exist_ok=True)

    regions = load_regions()
    if args.regions:
        want = {s.strip() for s in args.regions.split(",")}
        regions = [r for r in regions if r["id"] in want]
    rids = [r["id"] for r in regions]

    coefs = fit(build_calibration())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regression_coefs.json").write_text(json.dumps(coefs, indent=2), encoding="utf-8")

    per_region = {rid: _region_estimates(rid, exp, coefs) for rid in rids}
    pooled = _pool_estimates(per_region)

    results = {
        "experiment": exp,
        "co2_per_c": CO2_PER_C, "ha_per_pixel": HA_PER_PX,
        "regions": rids,
        "three_bin_note": "per-region NDVI terciles of that region's own Year-T "
                          "forest composite; assumed scheme, not an IPCC table",
        "three_bin_values_tC_ha": THREE_BIN_VALUES_tC_ha,
        "regression": {"primary": "exponential",
                       "linear": coefs["linear"], "exponential": coefs["exponential"],
                       "calibration_points": coefs["calibration_points"],
                       "calibration_note": coefs["note"]},
        "per_region": per_region,
        "pooled": pooled,
    }
    (OUT / "carbon_estimates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    _calibration_figure(coefs)
    _co2_figure({**per_region, "pooled": pooled}, exp)

    # ---- markdown ----
    L = ["# Phase 6/8 - Carbon Estimation (multi-region)\n",
         "NDVI is from each region's Year-T (2019) composite (pre-clearing "
         "canopy). Aboveground carbon density per pixel -> tonnes C over the "
         f"deforested area -> tonnes CO2 (x {CO2_PER_C:.3f}). The exponential "
         "regression `AGC = exp(a*NDVI + b)` is fit once on 8 Western-Ghats "
         "field-inventory anchors; the 3-bin cut points are each region's own "
         "forest terciles. Literature-calibrated, not pixel-matched biomass.\n",
         "## Predicted loss on held-out test blocks (primary = exponential regression)\n",
         "| Region | Area (ha) | 3-bin tCO2 | reg-linear tCO2 | reg-exp tCO2 (primary) | mean AGC (tC/ha) |",
         "|---|---|---|---|---|---|"]
    groups = {**per_region, "POOLED": pooled}
    for g, e in groups.items():
        pt = e["predicted_test"]
        L.append(f"| {g} | {pt['area_ha']:.1f} | "
                 f"{pt['three_bin_baseline']['tCO2']:,.0f} | "
                 f"{pt['regression_linear']['tCO2']:,.0f} | "
                 f"{pt['regression_exponential_primary']['tCO2']:,.0f} | "
                 f"{pt['regression_exponential_primary']['mean_AGC_tC_ha']:.0f} |")

    pp = pooled["predicted_test"]["regression_exponential_primary"]
    pg = pooled["gfc_test"]["regression_exponential_primary"]
    area_pred = pooled["predicted_test"]["area_ha"]
    area_gfc = pooled["gfc_test"]["area_ha"]
    co2_ratio = pp["tCO2"] / pg["tCO2"] if pg["tCO2"] else float("nan")
    area_ratio = area_pred / area_gfc if area_gfc else float("nan")
    L += ["",
          f"**Headline (pooled held-out test blocks, primary regression):** "
          f"predicted loss of {area_pred:.1f} ha -> **{pp['tCO2']:,.0f} t CO2** "
          f"(mean {pp['mean_AGC_tC_ha']:.0f} tC/ha). The Hansen-GFC reference "
          f"area for the same blocks gives {pg['tCO2']:,.0f} t CO2 "
          f"(mean {pg['mean_AGC_tC_ha']:.0f} tC/ha) - the model estimate is "
          f"{co2_ratio:.2f}x the reference.",
          "",
          f"Two biases act in opposite directions: the model under-predicts the "
          f"cleared *area* ({area_ratio:.2f}x: {area_pred:.1f} vs {area_gfc:.1f} "
          f"ha) but tends to over-predict the mean carbon density of the pixels "
          f"it does flag ({pp['mean_AGC_tC_ha']:.0f} vs {pg['mean_AGC_tC_ha']:.0f} "
          f"tC/ha), so the CO2 ratio ({co2_ratio:.2f}x) is less extreme than the "
          f"area ratio. Neither the area nor the CO2 total is independent evidence "
          f"of accuracy.",
          "",
          "Figures: `results/figures/phase6_carbon_calibration.png`, "
          "`results/figures/phase6_co2_estimates.png`.",
          ""]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")

    print("tCO2 (predicted loss, held-out test blocks; exp regression is primary):")
    for g, e in groups.items():
        pt = e["predicted_test"]
        print(f"  {g:10s} area {pt['area_ha']:7.1f} ha | "
              f"3-bin {pt['three_bin_baseline']['tCO2']:9,.0f} | "
              f"lin {pt['regression_linear']['tCO2']:9,.0f} | "
              f"exp {pt['regression_exponential_primary']['tCO2']:9,.0f}")
    print(f"\n-> {OUT / 'carbon_estimates.json'}")
    print(f"-> {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
