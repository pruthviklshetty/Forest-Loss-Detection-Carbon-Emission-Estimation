"""Phase 5, step 2 - area computation and the deforestation-map figure.

Compares the model's region-wide predicted forest-loss raster
(results/deforestation/<exp>_loss.tif) against the Hansen GFC ground-truth
loss raster, converts pixel counts to hectares with the 10 m Sentinel-2 GSD,
and breaks the numbers down by the Phase 2 split so the held-out TEST figure
is reported separately from the train-contaminated full-region figure.

    python -m src.change_detection.area_report --experiment baseline_unet

Outputs:
    results/deforestation/<exp>_area_summary.json
    results/deforestation/summary.md
    results/figures/phase5_deforestation_map.png
    results/figures/phase5_hectares.png
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

RAW = REPO / "data" / "raw"
MASKS = REPO / "data" / "masks"
PROC = REPO / "data" / "processed"
OUT = RESULTS / "deforestation"
FIG = RESULTS / "figures"
GSD = 10
HA_PER_PX = GSD * GSD / 1e4          # 0.01 ha


def _read(path, band=1):
    with rasterio.open(path) as s:
        return s.read(band)


def _split_map(H, W):
    """Per-pixel split label from the canonical (non-overlap) patch boxes."""
    sm = np.full((H, W), "none", dtype=object)
    with open(PROC / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["is_overlap"]):
                continue
            r0, c0, p = int(r["px_r0"]), int(r["px_c0"]), int(r["size"])
            sm[r0:r0 + p, c0:c0 + p] = r["split"]
    return sm


def _confusion(pred, gt, mask):
    p = pred[mask].astype(bool)
    g = gt[mask].astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum())
    fn = int((~p & g).sum()); tn = int((~p & ~g).sum())
    eps = 1e-9
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": tp / (tp + fp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "pred_ha": (tp + fp) * HA_PER_PX,
            "gt_ha": (tp + fn) * HA_PER_PX}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline_unet")
    args = ap.parse_args()
    exp = args.experiment

    pred = _read(OUT / f"{exp}_loss.tif")
    prob = _read(OUT / f"{exp}_prob.tif")
    gt = _read(MASKS / "loss_label.tif")
    valid = _read(MASKS / "valid_mask.tif").astype(bool)
    H, W = pred.shape
    gt = gt[:H, :W]; valid = valid[:H, :W]
    finite = np.isfinite(prob)
    valid = valid & finite

    sm = _split_map(H, W)
    thr = float(json.loads((RESULTS / "metrics" / f"{exp}.json").read_text())["operating_threshold"])

    regions = {
        "test_only": (sm == "test") & valid,
        "val_only": (sm == "val") & valid,
        "train_only": (sm == "train") & valid,
        "canonical_all": (sm != "none") & valid,
        "full_region": valid,
    }
    summary = {"experiment": exp, "operating_threshold": thr,
               "gsd_m": GSD, "ha_per_pixel": HA_PER_PX, "regions": {}}
    for name, m in regions.items():
        c = _confusion(pred, gt, m)
        c["pred_minus_gt_ha"] = round(c["pred_ha"] - c["gt_ha"], 2)
        c["pred_over_gt_ratio"] = round(c["pred_ha"] / c["gt_ha"], 3) if c["gt_ha"] else None
        for k in ("iou", "dice", "precision", "recall"):
            c[k] = round(c[k], 4)
        c["pred_ha"] = round(c["pred_ha"], 2)
        c["gt_ha"] = round(c["gt_ha"], 2)
        summary["regions"][name] = c

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{exp}_area_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- figure 1: deforestation map (full region) ----
    def st(a):
        lo, hi = np.nanpercentile(a, [2, 98])
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    with rasterio.open(RAW / "s2_T1.tif") as s:
        step = max(1, int(max(s.width, s.height) / 1400))
        oh, ow = (s.height + step - 1) // step, (s.width + step - 1) // step
        b = s.read(out_shape=(s.count, oh, ow), resampling=rasterio.enums.Resampling.average)
    fc = np.dstack([st(b[2]), st(b[1]), st(b[0])])
    pr = pred[::step, ::step].astype(bool)[:oh, :ow]
    gr = gt[::step, ::step].astype(bool)[:oh, :ow]
    agree = np.zeros((*fc.shape[:2], 3), np.uint8)
    agree[gr & ~pr] = [40, 90, 220]      # missed  (FN) blue
    agree[pr & ~gr] = [240, 170, 30]     # false alarm (FP) orange
    agree[pr & gr] = [220, 30, 30]       # hit (TP) red

    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    ax[0].imshow(fc); ax[0].set_title("Sentinel-2 T+1 (2021) false colour")
    ov = fc.copy(); ov[pr] = [1, 1, 0]
    ax[1].imshow(ov); ax[1].set_title(f"Predicted new forest loss (yellow)\n{exp}, thr {thr:.2f}")
    ax[2].imshow(fc * 0.35); ax[2].imshow(agree, alpha=(agree.sum(-1) > 0).astype(float))
    ax[2].set_title("vs Hansen GFC   red=hit  blue=missed  orange=false alarm")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "phase5_deforestation_map.png", dpi=110)
    plt.close(fig)

    # ---- figure 2: hectares bar ----
    names = ["test_only", "val_only", "train_only", "full_region"]
    pred_ha = [summary["regions"][n]["pred_ha"] for n in names]
    gt_ha = [summary["regions"][n]["gt_ha"] for n in names]
    x = np.arange(len(names))
    fig, axb = plt.subplots(figsize=(9, 4.5))
    axb.bar(x - 0.2, gt_ha, 0.4, label="Hansen GFC (reference)", color="steelblue")
    axb.bar(x + 0.2, pred_ha, 0.4, label=f"predicted ({exp})", color="firebrick")
    for i, (g, p) in enumerate(zip(gt_ha, pred_ha)):
        axb.text(i - 0.2, g, f"{g:.0f}", ha="center", va="bottom", fontsize=8)
        axb.text(i + 0.2, p, f"{p:.0f}", ha="center", va="bottom", fontsize=8)
    axb.set_xticks(x); axb.set_xticklabels(names)
    axb.set_ylabel("hectares lost (2019-2020)")
    axb.set_title("Forest area lost: predicted vs Hansen GFC")
    axb.legend()
    fig.tight_layout()
    fig.savefig(FIG / "phase5_hectares.png", dpi=110)
    plt.close(fig)

    # ---- markdown ----
    R = summary["regions"]
    L = ["# Phase 5 - Change Detection & Area Computation\n",
         f"Model: **{exp}** (chosen over the Attention U-Net on test IoU/Dice). "
         f"One forward pass on the 8-band bi-temporal stack yields the "
         f"newly-deforested mask directly; overlapping 256 px tiles (stride 128) "
         f"are averaged, thresholded at the val-tuned **{thr:.2f}**, and masked "
         f"to valid land.\n",
         f"Pixel -> area: 10 m GSD, {HA_PER_PX} ha per pixel.\n",
         "| Region | GFC ref (ha) | Predicted (ha) | Pred - GFC (ha) | Pred/GFC | IoU | Dice | Precision | Recall |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in ["test_only", "val_only", "train_only", "canonical_all", "full_region"]:
        c = R[n]
        L.append(f"| {n} | {c['gt_ha']:.1f} | {c['pred_ha']:.1f} | "
                 f"{c['pred_minus_gt_ha']:+.1f} | {c['pred_over_gt_ratio']} | "
                 f"{c['iou']:.3f} | {c['dice']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} |")
    L += ["",
          f"**Headline (held-out test region):** predicted "
          f"**{R['test_only']['pred_ha']:.1f} ha** vs Hansen GFC "
          f"**{R['test_only']['gt_ha']:.1f} ha** "
          f"({R['test_only']['pred_over_gt_ratio']:.2f}x; "
          f"{R['test_only']['pred_minus_gt_ha']:+.1f} ha), pixel IoU "
          f"{R['test_only']['iou']:.3f}.",
          "",
          "`full_region` and `train_only` include pixels the model was trained "
          "on and overstate agreement; `test_only` is the honest number.",
          "",
          "Figures: `results/figures/phase5_deforestation_map.png`, "
          "`results/figures/phase5_hectares.png`.",
          ""]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")

    print("Region area summary:")
    for n in ["test_only", "val_only", "train_only", "full_region"]:
        c = R[n]
        print(f"  {n:14s}  GFC {c['gt_ha']:7.1f} ha | pred {c['pred_ha']:7.1f} ha "
              f"({c['pred_over_gt_ratio']}x)  IoU {c['iou']:.3f}")
    print(f"\n-> {OUT / (exp + '_area_summary.json')}")
    print(f"-> {OUT / 'summary.md'}")
    print("-> results/figures/phase5_deforestation_map.png , phase5_hectares.png")


if __name__ == "__main__":
    main()
