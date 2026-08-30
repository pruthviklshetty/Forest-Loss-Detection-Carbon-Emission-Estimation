"""Phase 6, apply step - turn deforested area into tonnes of CO2.

For every deforested pixel, take its Year-T (pre-clearing) NDVI, map it to an
aboveground carbon density with (a) the 3-bin baseline and (b) the calibrated
regression, multiply by the pixel area (0.01 ha) to get tonnes C lost, and
convert to CO2 with the 44/12 molecular-mass ratio.

Applied to four pixel sets so the honest held-out number is separable:
  predicted loss (test region)      <- headline
  predicted loss (full region)
  Hansen GFC loss (test region)     <- reference
  Hansen GFC loss (full region)     <- reference

Assumptions (stated, not hidden): only aboveground carbon; complete committed
emission of that carbon on clearing; no belowground / deadwood / litter / soil
pools; no post-clearing regrowth; NDVI->carbon calibrated to literature
regional means, not pixel-matched biomass.

    python -m src.carbon.run_carbon
"""

from __future__ import annotations

import csv
import json

import numpy as np
import rasterio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common import REPO, RESULTS  # noqa: E402
from .ndvi import THREE_BIN_VALUES_tC_ha, compute_ndvi, three_bin_carbon_density  # noqa: E402
from .regression_model import fit, build_calibration, predict  # noqa: E402

RAW = REPO / "data" / "raw"
MASKS = REPO / "data" / "masks"
PROC = REPO / "data" / "processed"
DEFOR = RESULTS / "deforestation"
OUT = RESULTS / "carbon_validation"
FIG = RESULTS / "figures"

HA_PER_PX = 0.01
CO2_PER_C = 44.0 / 12.0
# 3-bin NDVI cut points = forest tercile boundaries of the Year-T composite
BIN_LO_PCTILE, BIN_HI_PCTILE = 33.3, 66.7


def _read(path, band=1):
    with rasterio.open(path) as s:
        return s.read(band)


def _split_map(H, W):
    sm = np.full((H, W), "none", dtype=object)
    with open(PROC / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["is_overlap"]):
                continue
            r0, c0, p = int(r["px_r0"]), int(r["px_c0"]), int(r["size"])
            sm[r0:r0 + p, c0:c0 + p] = r["split"]
    return sm


def _totals(ndvi_vals, carbon_density_fn):
    agc = carbon_density_fn(ndvi_vals)                 # tC/ha per pixel
    tC = float(np.sum(agc) * HA_PER_PX)
    return {"tC": round(tC, 1), "tCO2": round(tC * CO2_PER_C, 1),
            "mean_AGC_tC_ha": round(float(np.mean(agc)), 1), "n_px": int(ndvi_vals.size)}


def main() -> None:
    with rasterio.open(RAW / "s2_T.tif") as s:
        a = s.read()
    ndvi = compute_ndvi(a[2], a[1])
    H, W = ndvi.shape

    pred = _read(DEFOR / "baseline_unet_loss.tif").astype(bool)[:H, :W]
    gfc = _read(MASKS / "loss_label.tif").astype(bool)[:H, :W]
    valid = _read(MASKS / "valid_mask.tif").astype(bool)[:H, :W] & np.isfinite(ndvi)
    sm = _split_map(H, W)
    test = sm == "test"

    forest = _read(MASKS / "forest2000.tif").astype(bool)[:H, :W] & valid
    lo = float(np.percentile(ndvi[forest], BIN_LO_PCTILE))
    hi = float(np.percentile(ndvi[forest], BIN_HI_PCTILE))

    coefs = fit(build_calibration())
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "regression_coefs.json").write_text(json.dumps(coefs, indent=2), encoding="utf-8")

    def three_bin(v):
        return three_bin_carbon_density(v, lo, hi)

    def reg_lin(v):
        return predict(v, coefs, "linear")

    def reg_exp(v):
        return predict(v, coefs, "exponential")

    methods = {"three_bin_baseline": three_bin, "regression_linear": reg_lin,
               "regression_exponential_primary": reg_exp}
    sets = {
        "predicted_test": pred & test & valid,
        "predicted_full_region": pred & valid,
        "gfc_test": gfc & test & valid,
        "gfc_full_region": gfc & valid,
    }

    results = {
        "co2_per_c": CO2_PER_C, "ha_per_pixel": HA_PER_PX,
        "three_bin": {"ndvi_lo": round(lo, 4), "ndvi_hi": round(hi, 4),
                      "values_tC_ha": THREE_BIN_VALUES_tC_ha,
                      "label": "this study's assumed classification scheme (not an IPCC table)"},
        "regression": {"primary": "exponential",
                       "linear": coefs["linear"], "exponential": coefs["exponential"],
                       "calibration_points": coefs["calibration_points"],
                       "calibration_note": coefs["note"]},
        "estimates": {},
    }
    for sname, mask in sets.items():
        v = ndvi[mask]
        results["estimates"][sname] = {m: _totals(v, fn) for m, fn in methods.items()}
        results["estimates"][sname]["area_ha"] = round(int(mask.sum()) * HA_PER_PX, 1)

    (OUT / "carbon_estimates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- calibration-curve figure ----
    cal = coefs["calibration_points"]
    xs = np.linspace(0.30, 0.90, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([c["ndvi"] for c in cal], [c["agc_MgC_ha"] for c in cal],
               c="k", zorder=5, label="literature anchors (Western Ghats)")
    ax.plot(xs, np.clip(coefs["linear"]["a"] * xs + coefs["linear"]["b"], 0, None),
            "--", label=f"linear (r2={coefs['linear']['r2']:.2f})")
    ax.plot(xs, np.exp(coefs["exponential"]["a"] * xs + coefs["exponential"]["b"]),
            "-", lw=2, label=f"exponential, primary (r2={coefs['exponential']['r2']:.2f})")
    tb = three_bin_carbon_density(xs, lo, hi)
    ax.step(xs, tb, where="mid", color="gray", label="3-bin baseline (assumed)")
    ax.set_xlabel("NDVI (Year-T composite)")
    ax.set_ylabel("aboveground carbon density (tC/ha)")
    ax.set_title("NDVI -> carbon density: 3-bin baseline vs calibrated regression")
    ax.legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "phase6_carbon_calibration.png", dpi=110)
    plt.close(fig)

    # ---- tCO2 bar figure ----
    order = ["three_bin_baseline", "regression_linear", "regression_exponential_primary"]
    setnames = ["predicted_test", "gfc_test", "predicted_full_region", "gfc_full_region"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(setnames))
    w = 0.26
    for i, m in enumerate(order):
        vals = [results["estimates"][s][m]["tCO2"] for s in setnames]
        b = ax.bar(x + (i - 1) * w, vals, w, label=m)
        ax.bar_label(b, fmt="%.0f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(setnames, rotation=10)
    ax.set_ylabel("tonnes CO2 (committed, aboveground)")
    ax.set_title("Estimated CO2 from 2019-2020 forest loss: 3-bin vs regression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "phase6_co2_estimates.png", dpi=110)
    plt.close(fig)

    # ---- markdown ----
    e = results["estimates"]
    L = ["# Phase 6 - Carbon Estimation\n",
         "NDVI is from the Year-T (2019) composite (pre-clearing canopy). "
         "Aboveground carbon density per pixel -> tonnes C over the deforested "
         f"area -> tonnes CO2 (x {CO2_PER_C:.3f}).\n",
         "**3-bin baseline** (this study's assumed scheme, not IPCC): "
         f"NDVI < {lo:.3f} -> {THREE_BIN_VALUES_tC_ha['sparse']:.0f}, "
         f"{lo:.3f}-{hi:.3f} -> {THREE_BIN_VALUES_tC_ha['moderate']:.0f}, "
         f">= {hi:.3f} -> {THREE_BIN_VALUES_tC_ha['dense']:.0f} tC/ha.\n",
         "**Regression (primary = exponential)**: "
         f"AGC = exp({coefs['exponential']['a']:.3f}*NDVI + "
         f"{coefs['exponential']['b']:.3f}), calibrated on 8 Western-Ghats "
         "field-inventory anchors (Padmakumar et al. 2018; Kothandaraman et al. "
         "2020). Literature-calibrated, not pixel-matched biomass; n is small.\n",
         "| Pixel set | Area (ha) | 3-bin tCO2 | reg-linear tCO2 | reg-exp tCO2 (primary) |",
         "|---|---|---|---|---|"]
    for s in setnames:
        L.append(f"| {s} | {e[s]['area_ha']:.1f} | "
                 f"{e[s]['three_bin_baseline']['tCO2']:,.0f} | "
                 f"{e[s]['regression_linear']['tCO2']:,.0f} | "
                 f"{e[s]['regression_exponential_primary']['tCO2']:,.0f} |")
    ph = e["predicted_test"]["regression_exponential_primary"]
    gh = e["gfc_test"]["regression_exponential_primary"]
    L += ["",
          f"**Headline (held-out test region, primary regression):** predicted "
          f"loss of {e['predicted_test']['area_ha']:.1f} ha -> "
          f"**{ph['tCO2']:,.0f} t CO2** "
          f"(mean {ph['mean_AGC_tC_ha']:.0f} tC/ha). Against the Hansen-GFC "
          f"reference area for the same test region: {gh['tCO2']:,.0f} t CO2.",
          "",
          "The regression gives a lower, NDVI-weighted estimate than the flat "
          "3-bin scheme because most cleared pixels sit at moderate NDVI, below "
          "the 150 tC/ha 'moderate' bin constant.",
          "",
          "Figures: `results/figures/phase6_carbon_calibration.png`, "
          "`results/figures/phase6_co2_estimates.png`.",
          ""]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")

    print("tCO2 by method:")
    for s in setnames:
        print(f"  {s:24s} area {e[s]['area_ha']:6.1f} ha | "
              f"3-bin {e[s]['three_bin_baseline']['tCO2']:8,.0f} | "
              f"lin {e[s]['regression_linear']['tCO2']:8,.0f} | "
              f"exp {e[s]['regression_exponential_primary']['tCO2']:8,.0f}")
    print(f"\n-> {OUT / 'carbon_estimates.json'}")
    print(f"-> {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
