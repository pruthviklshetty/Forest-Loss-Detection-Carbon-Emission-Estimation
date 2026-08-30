"""Phase 2, step 4 - data report and figures.

Reads the manifest / label summary / split summary produced by the earlier
steps and writes:
    docs/phase2_data_report.md
    results/figures/phase2_aoi_overview.png     T / T+1 false-colour + GFC loss + valid mask
    results/figures/phase2_patch_examples.png   up to 6 patches: T, T+1, label
    results/figures/phase2_split_balance.png     patches and positive-rate per split

False colour = NIR / red / green as R / G / B (no blue band in the 4-band set);
vegetation is red, bare / cleared ground is cyan-grey.

Run:  python -m src.preprocessing.data_report
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rasterio  # noqa: E402

from .eeutil import load_cfg  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
_RAW = _REPO / "data" / "raw"
_MASKS = _REPO / "data" / "masks"
_PROC = _REPO / "data" / "processed"
_FIG = _REPO / "results" / "figures"
_DOCS = _REPO / "docs"


def _stretch(a: np.ndarray, lo_pct=2, hi_pct=98) -> np.ndarray:
    lo, hi = np.nanpercentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _false_colour(path: pathlib.Path, max_side=1200) -> np.ndarray:
    """RGB uint8 image from an s2_*.tif: NIR, red, green -> R, G, B, contrast-stretched."""
    with rasterio.open(path) as src:
        step = max(1, int(max(src.width, src.height) / max_side))
        out_h = (src.height + step - 1) // step
        out_w = (src.width + step - 1) // step
        bands = src.read(
            out_shape=(src.count, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        )
        desc = list(src.descriptions)
    idx = {d: i for i, d in enumerate(desc) if d}
    if {"nir", "red", "green"} <= idx.keys():
        r, g, b = bands[idx["nir"]], bands[idx["red"]], bands[idx["green"]]
    else:
        r, g, b = bands[2], bands[1], bands[0]
    rgb = np.dstack([_stretch(r), _stretch(g), _stretch(b)])
    return (rgb * 255).astype(np.uint8)


def _read_mask(path: pathlib.Path, max_side=1200) -> np.ndarray:
    with rasterio.open(path) as src:
        step = max(1, int(max(src.width, src.height) / max_side))
        out_h = (src.height + step - 1) // step
        out_w = (src.width + step - 1) // step
        return src.read(1, out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.nearest)


def _fig_aoi_overview() -> pathlib.Path:
    fc_t = _false_colour(_RAW / "s2_T.tif")
    fc_t1 = _false_colour(_RAW / "s2_T1.tif")
    loss = _read_mask(_MASKS / "loss_label.tif")
    valid = _read_mask(_MASKS / "valid_mask.tif")
    forest = _read_mask(_MASKS / "forest2000.tif")

    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    ax[0].imshow(fc_t); ax[0].set_title("Sentinel-2 T (2019) false colour\nNIR/red/green")
    ax[1].imshow(fc_t1); ax[1].set_title("Sentinel-2 T+1 (2021) false colour")
    ov = fc_t1.copy()
    m = loss.astype(bool)
    if m.shape == ov.shape[:2]:
        ov[m] = [255, 255, 0]
    ax[2].imshow(ov); ax[2].set_title("T+1 with GFC forest loss (yellow)")
    comp = np.zeros((*forest.shape, 3), np.uint8)
    comp[forest.astype(bool)] = [30, 120, 30]
    comp[~valid.astype(bool)] = [40, 60, 140]
    comp[loss.astype(bool)] = [230, 40, 40]
    ax[3].imshow(comp)
    ax[3].set_title("GFC layers\ngreen=forest2000  blue=non-land  red=loss")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    out = _FIG / "phase2_aoi_overview.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _fig_patch_examples(n: int = 6) -> pathlib.Path | None:
    idx_csv = _PROC / "index.csv"
    if not idx_csv.exists():
        return None
    import csv
    with open(idx_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with_loss = sorted((r for r in rows if int(r["n_loss_px"]) > 0),
                       key=lambda r: -int(r["n_loss_px"]))[:n]
    if not with_loss:
        return None
    norm = json.loads((_PROC / "norm_stats.json").read_text())
    fig, ax = plt.subplots(len(with_loss), 3, figsize=(9, 3 * len(with_loss)))
    if len(with_loss) == 1:
        ax = ax[None, :]
    for i, r in enumerate(with_loss):
        d = np.load(_PROC / "patches" / f"{r['patch_id']}.npz")
        img, lab = d["img"], d["label"]
        # channels: 0..3 T g,r,nir,swir1 ; 4..7 T1
        t_fc = np.dstack([_stretch(img[2]), _stretch(img[1]), _stretch(img[0])])
        t1_fc = np.dstack([_stretch(img[6]), _stretch(img[5]), _stretch(img[4])])
        ax[i, 0].imshow(t_fc); ax[i, 0].set_ylabel(r["patch_id"], fontsize=8)
        ax[i, 1].imshow(t1_fc)
        ax[i, 2].imshow(lab, cmap="Reds", vmin=0, vmax=1)
        if i == 0:
            ax[i, 0].set_title("T false colour")
            ax[i, 1].set_title("T+1 false colour")
            ax[i, 2].set_title(f"loss label ({r['n_loss_px']} px)")
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Phase 2 patch examples (highest-loss patches)", y=1.0)
    fig.tight_layout()
    out = _FIG / "phase2_patch_examples.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _fig_split_balance() -> pathlib.Path | None:
    sp = _PROC / "split.json"
    if not sp.exists():
        return None
    s = json.loads(sp.read_text())["splits"]
    names = ["train", "val", "test"]
    npatch = [s[n]["n_patches"] for n in names]
    nloss = [s[n]["n_patches_with_loss"] for n in names]
    rate = [s[n]["loss_px_pct_of_valid"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(3)
    ax[0].bar(x - 0.2, npatch, 0.4, label="patches")
    ax[0].bar(x + 0.2, nloss, 0.4, label="patches w/ loss")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].legend()
    ax[0].set_title("patch counts per split")
    ax[1].bar(x, rate, color="firebrick")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names)
    ax[1].set_title("positive pixel rate (% of valid px)")
    fig.tight_layout()
    out = _FIG / "phase2_split_balance.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    cfg = load_cfg()
    _FIG.mkdir(parents=True, exist_ok=True)
    _DOCS.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((_RAW / "manifest.json").read_text())
    lab = json.loads((_MASKS / "loss_label_summary.json").read_text())
    split = json.loads((_PROC / "split.json").read_text())
    norm = json.loads((_PROC / "norm_stats.json").read_text())

    print("figures:")
    f1 = _fig_aoi_overview(); print(f"  {f1}")
    f2 = _fig_patch_examples(); print(f"  {f2}")
    f3 = _fig_split_balance(); print(f"  {f3}")

    s2 = {k: v for k, v in manifest["outputs"].items() if k.startswith("s2_")}
    gsd = int(cfg["region"]["target_gsd_m"])

    lines = []
    A = lines.append
    A("# Phase 2 - Data Report\n")
    A(f"**Region:** {cfg['region']['name']}  ")
    A(f"**BBox (WGS84 W,S,E,N):** `{cfg['region']['bbox']['wsen']}`  ")
    A(f"**Grid:** {lab['grid']['width']} x {lab['grid']['height']} px @ {gsd} m, "
      f"{lab['grid']['crs']}\n")

    A("## 1. Sentinel-2 acquisition\n")
    A(f"Collection `COPERNICUS/S2_SR_HARMONIZED`, cloud-masked with Cloud Score+ "
      f"(`cs_cdf >= 0.60`), per-band **median** composite, reflectance scaled by "
      f"10000 and clipped to [0,1]. Bands B3/B4/B8/B11 -> green/red/nir/swir1; "
      f"SWIR1 bilinearly upsampled 20 m -> 10 m.\n")
    A("| Window | Dates | Scenes in window | Cloudy-pixel % (min / median / max) |")
    A("|---|---|---|---|")
    for name, key in (("T", "s2_T.tif"), ("T+1", "s2_T1.tif")):
        o = manifest["outputs"][key]
        cp = o["cloudy_pixel_pct_sorted"]
        med = cp[len(cp) // 2] if cp else float("nan")
        A(f"| {name} | {o['date_start']} .. {o['date_end']} | {o['n_scenes']} | "
          f"{cp[0]:.1f} / {med:.1f} / {cp[-1]:.1f} |")
    A("")

    A("## 2. Ground truth: Hansen GFC -> binary forest-loss label\n")
    A(f"Asset `{lab['gee_asset']}`. `lossyear` is a year-of-loss code, not a "
      f"T-vs-T+1 mask; conversion (documented in `src/preprocessing/build_labels.py`):\n")
    A("```")
    A(f"forest2000  = treecover2000 >= {lab['canopy_threshold_pct']}%")
    A("land        = datamask == 1")
    A(f"loss_window = lossyear in {lab['loss_year_codes']}   # calendar 2019 & 2020")
    A("label = 1  where  forest2000 AND land AND loss_window   (new loss T -> T+1)")
    A("label = 0  elsewhere on land ;  valid = land")
    A("```")
    A(f"Loss codes `{lab['loss_year_codes']}` chosen because the T composite "
      f"(Jan-Apr 2019) is the canopy state entering 2019 and T+1 (Jan-Apr 2021) "
      f"the state entering 2021, so calendar-2019/2020 stand-replacement loss is "
      f"exactly what happened between acquisitions.\n")
    px = lab["pixels"]
    A("| Quantity | Value |")
    A("|---|---|")
    A(f"| Valid land pixels | {px['valid_land']:,} ({px['valid_land_pct']}% of grid) |")
    A(f"| Forest (canopy >= {lab['canopy_threshold_pct']}%) at 2000 | {px['forest2000']:,} "
      f"({px['forest2000_pct_of_land']}% of land) |")
    A(f"| Forest-loss positives (2019-20) | {px['loss_positive']:,} "
      f"({px['loss_positive_pct_of_land']}% of land, "
      f"{px['loss_positive_pct_of_forest2000']}% of forest2000) |")
    A(f"| GFC reference area lost | **{lab['ha_lost_gfc_reference']:,} ha** |")
    A(f"\n`lossyear` histogram over land: `{lab['lossyear_histogram_over_land']}`\n")

    A("## 3. Patches and split\n")
    A(f"{split['n_patches']} non-overlapping {split['patch_size_px']}x"
      f"{split['patch_size_px']} patches, 8 bands "
      f"`{split['band_order']}`. Split: {split['split_method']} "
      f"({split['split_block_patches']}x{split['split_block_patches']}-patch blocks, "
      f"seed {split['split_seed']}).\n")
    A("| Split | Patches | % | Patches w/ loss | Positive px (% of valid) | Area lost (ha) |")
    A("|---|---|---|---|---|---|")
    for n in ("train", "val", "test"):
        d = split["splits"][n]
        A(f"| {n} | {d['n_patches']} | {d['patch_frac']:.0%} | {d['n_patches_with_loss']} | "
          f"{d['loss_px_pct_of_valid']:.4f}% | {d['ha_lost']:.1f} |")
    A("")
    A("Severe class imbalance is expected and is handled at train time "
      "(Dice + BCE, positive weighting).\n")

    A("## 4. Normalisation\n")
    A("Bands are stored as [0,1] reflectance. Per-band mean/std over valid "
      f"train pixels ({norm['n_pixels']:,}) for optional z-scoring at train time:\n")
    A("| band | " + " | ".join(norm["band_order"]) + " |")
    A("|---|" + "---|" * len(norm["band_order"]))
    A("| mean | " + " | ".join(f"{v:.4f}" for v in norm["mean"]) + " |")
    A("| std  | " + " | ".join(f"{v:.4f}" for v in norm["std"]) + " |")
    A("")

    A("## 5. Figures\n")
    for f in (f1, f2, f3):
        if f:
            A(f"- `results/figures/{f.name}`")
    A("")

    patch_ha = sum(v["ha_lost"] for v in split["splits"].values())
    A("## 6. Caveats carried into Phase 3\n")
    A(f"- **Small dataset.** {split['splits']['train']['n_patches']} train / "
      f"{split['splits']['val']['n_patches']} val / "
      f"{split['splits']['test']['n_patches']} test non-overlapping patches. "
      f"Mitigations for Phase 3: overlapping train patches (stride 128 ~4x the "
      f"train set), strong augmentation, or widening the AOI.")
    A(f"- **Extreme class imbalance.** Positive rate ~0.27-0.44% of valid pixels. "
      f"Needs Dice/Tversky + weighted BCE and threshold tuning; pixel accuracy "
      f"will be near-trivial and is reported only for completeness.")
    A(f"- **Edge margin.** The 256-px grid covers 2816 x 2560 of the 3064 x 2778 "
      f"raster; the right/bottom margin holds "
      f"{lab['ha_lost_gfc_reference'] - patch_ha:.1f} ha of GFC loss not in any "
      f"patch ({patch_ha:.1f} of {lab['ha_lost_gfc_reference']:.1f} ha retained).")
    A(f"- **Blocked split is not perfectly balanced.** Test positive rate "
      f"({split['splits']['test']['loss_px_pct_of_valid']:.3f}%) exceeds train "
      f"({split['splits']['train']['loss_px_pct_of_valid']:.3f}%); seed 42 is "
      f"fixed in config and not re-picked to avoid gaming the split.")
    A(f"- **Bi-temporal signal is subtle.** In the highest-loss patches the T "
      f"and T+1 false-colour composites look similar at loss sites, and some "
      f"patches retain thin-haze / BRDF differences from median compositing. "
      f"This is a genuinely hard change-detection setting; Phase 4's comparison "
      f"to John & Zhang (2022) must account for it.")
    A("")

    out_md = _DOCS / "phase2_data_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {out_md}")


if __name__ == "__main__":
    main()
