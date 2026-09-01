"""Phase 5 / Phase 8, step 2 - area computation and the deforestation-map figure,
per region and pooled.

For every region, compares the model's region-wide predicted forest-loss raster
(results/deforestation/<exp>__<rid>_loss.tif) against the Hansen GFC ground-truth
loss raster, converts pixel counts to hectares with the 10 m Sentinel-2 GSD, and
breaks the numbers down by the pooled split so the held-out TEST figure is
reported separately from the train-contaminated full-region figure. A `pooled`
block re-aggregates the confusion over every region's held-out test pixels.

    python -m src.change_detection.area_report --experiment p8_pooled_unet_s43
    python -m src.change_detection.area_report --experiment baseline_unet --regions wayanad

Outputs:
    results/deforestation/<exp>_area_summary.json
    results/deforestation/<exp>_summary.md
    results/figures/phase5_deforestation_map.png
    results/figures/phase5_hectares.png

Pixel IoU is reported strict (primary) and tolerance (GT dilated one 30 m GFC
cell = +/-3 px, strict undilated union; secondary), consistent with
`src/eval/evaluate.py`.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common import REPO, RESULTS  # noqa: E402
from ..paths import masks_dir as _masks_dir  # noqa: E402
from ..paths import proc_dir as _proc_dir  # noqa: E402
from ..paths import raw_dir as _raw_dir  # noqa: E402
from ..regions import load_regions  # noqa: E402

# rebound in main() when --period is given
RAW = REPO / "data" / "raw"
MASKS = REPO / "data" / "masks"
PROC = REPO / "data" / "processed"
OUT = RESULTS / "deforestation"
FIG = RESULTS / "figures"
_FIG_SUFFIX = ""     # e.g. "_2021_2023" so period figures do not overwrite
GSD = 10
HA_PER_PX = GSD * GSD / 1e4          # 0.01 ha
_DILATE_PX = 3                       # one 30 m Hansen GFC cell at the 10 m GSD
_DILATE_STRUCT = np.ones((2 * _DILATE_PX + 1, 2 * _DILATE_PX + 1), dtype=bool)
_SPLITS = ["test_only", "val_only", "train_only", "canonical_all", "full_region"]


def _first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    raise SystemExit(f"missing raster; looked for {[str(p) for p in paths]}")


def _read(path, band=1):
    with rasterio.open(path) as s:
        return s.read(band)


def _split_map(rid, H, W):
    """Per-pixel split label from that region's canonical (non-overlap) patch boxes."""
    sm = np.full((H, W), "none", dtype=object)
    with open(PROC / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("region", rid) != rid:
                continue
            if int(r["is_overlap"]):
                continue
            split = r.get("pooled_split") or r["split"]
            r0, c0, p = int(r["px_r0"]), int(r["px_c0"]), int(r["size"])
            sm[r0:r0 + p, c0:c0 + p] = split
    return sm


def _confusion(pred, gt, mask, gt_dil):
    p = pred[mask].astype(bool)
    g = gt[mask].astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum())
    fn = int((~p & g).sum()); tn = int((~p & ~g).sum())
    gd = gt_dil[mask].astype(bool)
    tol_inter = int((p & gd).sum())
    return _metrics_from_counts(tp, fp, fn, tn, tol_inter)


def _metrics_from_counts(tp, fp, fn, tn, tol_inter):
    eps = 1e-9
    union = tp + fp + fn
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tol_inter": tol_inter, "union": union,
           "iou": tp / (union + eps),
           "dice": 2 * tp / (2 * tp + fp + fn + eps),
           "precision": tp / (tp + fp + eps),
           "recall": tp / (tp + fn + eps),
           "tolerance_iou": tol_inter / (union + eps),
           "pred_ha": (tp + fp) * HA_PER_PX,
           "gt_ha": (tp + fn) * HA_PER_PX}
    out["strict_iou"] = out["iou"]
    return out


def _round_block(c):
    c = dict(c)
    c["pred_minus_gt_ha"] = round(c["pred_ha"] - c["gt_ha"], 2)
    c["pred_over_gt_ratio"] = round(c["pred_ha"] / c["gt_ha"], 3) if c["gt_ha"] else None
    for k in ("iou", "strict_iou", "tolerance_iou", "dice", "precision", "recall"):
        c[k] = round(c[k], 4)
    c["pred_ha"] = round(c["pred_ha"], 2)
    c["gt_ha"] = round(c["gt_ha"], 2)
    return c


def _region_report(rid, exp, gt_dil_cache):
    pred = _read(OUT / f"{exp}__{rid}_loss.tif").astype(bool)
    prob = _read(OUT / f"{exp}__{rid}_prob.tif")
    gt = _read(_first_existing(MASKS / rid / "loss_label.tif", MASKS / "loss_label.tif")).astype(bool)
    valid = _read(_first_existing(MASKS / rid / "valid_mask.tif",
                                  MASKS / "valid_mask.tif")).astype(bool)
    H, W = pred.shape
    gt = gt[:H, :W]; valid = valid[:H, :W] & np.isfinite(prob)
    sm = _split_map(rid, H, W)
    gt_dil = binary_dilation(gt, structure=_DILATE_STRUCT)

    masks = {
        "test_only": (sm == "test") & valid,
        "val_only": (sm == "val") & valid,
        "train_only": (sm == "train") & valid,
        "canonical_all": (sm != "none") & valid,
        "full_region": valid,
    }
    blocks = {name: _confusion(pred, gt, m, gt_dil) for name, m in masks.items()}
    return blocks, pred, prob, gt, valid, sm


def _pool(blocks_by_region, split):
    keys = ("tp", "fp", "fn", "tn", "tol_inter")
    agg = {k: sum(b[split][k] for b in blocks_by_region.values()) for k in keys}
    return _metrics_from_counts(agg["tp"], agg["fp"], agg["fn"], agg["tn"], agg["tol_inter"])


def _map_figure(per_region, exp, thr):
    rids = list(per_region)
    n = len(rids)
    fig, ax = plt.subplots(n, 3, figsize=(15, 4.4 * n))
    if n == 1:
        ax = ax[None, :]

    def st(a):
        lo, hi = np.nanpercentile(a, [2, 98])
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    for i, rid in enumerate(rids):
        _, pred, _prob, gt, _valid, _sm = per_region[rid]
        p_t1 = _first_existing(RAW / rid / "s2_T1.tif", RAW / "s2_T1.tif")
        with rasterio.open(p_t1) as s:
            step = max(1, int(max(s.width, s.height) / 1200))
            oh, ow = (s.height + step - 1) // step, (s.width + step - 1) // step
            b = s.read(out_shape=(s.count, oh, ow),
                       resampling=rasterio.enums.Resampling.average)
        fc = np.dstack([st(b[2]), st(b[1]), st(b[0])])
        pr = pred[::step, ::step].astype(bool)[:oh, :ow]
        gr = gt[::step, ::step].astype(bool)[:oh, :ow]
        agree = np.zeros((*fc.shape[:2], 3), np.uint8)
        agree[gr & ~pr] = [40, 90, 220]      # missed  (FN) blue
        agree[pr & ~gr] = [240, 170, 30]     # false alarm (FP) orange
        agree[pr & gr] = [220, 30, 30]       # hit (TP) red
        ov = fc.copy(); ov[pr] = [1, 1, 0]
        ax[i, 0].imshow(fc); ax[i, 0].set_ylabel(rid, fontsize=11)
        ax[i, 1].imshow(ov)
        ax[i, 2].imshow(fc * 0.35)
        ax[i, 2].imshow(agree, alpha=(agree.sum(-1) > 0).astype(float))
        if i == 0:
            ax[i, 0].set_title("Sentinel-2 T+1 (2021) false colour")
            ax[i, 1].set_title(f"predicted new forest loss (yellow)\n{exp}, thr {thr:.2f}")
            ax[i, 2].set_title("vs Hansen GFC   red=hit  blue=missed  orange=false alarm")
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"phase5_deforestation_map{_FIG_SUFFIX}.png", dpi=110)
    plt.close(fig)


def _hectares_figure(summary, exp):
    rids = list(summary["per_region"])
    pred_ha = [summary["per_region"][r]["test_only"]["pred_ha"] for r in rids]
    gt_ha = [summary["per_region"][r]["test_only"]["gt_ha"] for r in rids]
    rids2 = rids + ["POOLED"]
    pred_ha.append(summary["pooled"]["test_only"]["pred_ha"])
    gt_ha.append(summary["pooled"]["test_only"]["gt_ha"])
    x = np.arange(len(rids2))
    fig, axb = plt.subplots(figsize=(1.7 * len(rids2) + 3, 4.5))
    axb.bar(x - 0.2, gt_ha, 0.4, label="Hansen GFC (reference)", color="steelblue")
    axb.bar(x + 0.2, pred_ha, 0.4, label=f"predicted ({exp})", color="firebrick")
    for i, (g, p) in enumerate(zip(gt_ha, pred_ha)):
        axb.text(i - 0.2, g, f"{g:.0f}", ha="center", va="bottom", fontsize=8)
        axb.text(i + 0.2, p, f"{p:.0f}", ha="center", va="bottom", fontsize=8)
    axb.set_xticks(x); axb.set_xticklabels(rids2)
    axb.set_ylabel("hectares lost (2019-2020), held-out test blocks")
    axb.set_title("Forest area lost on held-out test blocks: predicted vs Hansen GFC")
    axb.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"phase5_hectares{_FIG_SUFFIX}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    global RAW, MASKS, PROC, _FIG_SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline_unet")
    ap.add_argument("--regions", default=None,
                    help="comma-separated region ids (default: all in configs/region.yaml)")
    ap.add_argument("--period", default=None,
                    help="read data/*/<period>/ and suffix figures (e.g. 2021_2023)")
    args = ap.parse_args()
    exp = args.experiment
    if args.period:
        RAW, MASKS, PROC = (_raw_dir(period=args.period), _masks_dir(period=args.period),
                            _proc_dir(period=args.period))
        _FIG_SUFFIX = f"_{args.period}"
        print(f"period: {args.period}")

    regions = load_regions()
    if args.regions:
        want = {s.strip() for s in args.regions.split(",")}
        regions = [r for r in regions if r["id"] in want]
    rids = [r["id"] for r in regions]

    thr_path = RESULTS / "metrics" / f"{exp}.json"
    thr = float(json.loads(thr_path.read_text())["operating_threshold"]) if thr_path.exists() \
        else float(json.loads((RESULTS / "deforestation" / f"{exp}_infer_summary.json")
                              .read_text())["operating_threshold"])

    per_region_raw = {}
    per_region_blocks = {}
    for rid in rids:
        blocks, pred, prob, gt, valid, sm = _region_report(rid, exp, {})
        per_region_blocks[rid] = blocks
        per_region_raw[rid] = (blocks, pred, prob, gt, valid, sm)

    summary = {
        "experiment": exp, "operating_threshold": thr,
        "gsd_m": GSD, "ha_per_pixel": HA_PER_PX,
        "tolerance_iou_dilation_px": _DILATE_PX,
        "tolerance_iou_note": "intersection vs GFC GT dilated one 30 m cell "
                              "(7x7, +/-3 px); strict undilated union; secondary, "
                              "strict_iou is primary",
        "regions": rids,
        "per_region": {rid: {s: _round_block(per_region_blocks[rid][s]) for s in _SPLITS}
                       for rid in rids},
        "pooled": {s: _round_block(_pool(per_region_blocks, s)) for s in _SPLITS},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{exp}_area_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _map_figure(per_region_raw, exp, thr)
    _hectares_figure(summary, exp)

    # ---- markdown ----
    P = summary["pooled"]["test_only"]
    L = ["# Phase 5/8 - Change Detection & Area Computation (multi-region)\n",
         f"Model: **{exp}**. One forward pass on the 8-band bi-temporal stack "
         f"yields the newly-deforested mask directly; overlapping 256 px tiles "
         f"(stride 128) are averaged, thresholded at the val-tuned **{thr:.2f}**, "
         f"and masked to valid land. Pixel -> area: 10 m GSD, {HA_PER_PX} ha per "
         f"pixel. Strict pixel IoU is primary; tolerance IoU (+/-3 px GFC-cell GT "
         f"dilation, strict union) is secondary, consistent with "
         f"`src/eval/evaluate.py`.\n",
         "## Held-out test blocks, per region\n",
         "| Region | GFC ref (ha) | Predicted (ha) | Pred - GFC (ha) | Pred/GFC | strict IoU | tol IoU | Dice | Precision | Recall |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for rid in rids:
        c = summary["per_region"][rid]["test_only"]
        L.append(f"| {rid} | {c['gt_ha']:.1f} | {c['pred_ha']:.1f} | "
                 f"{c['pred_minus_gt_ha']:+.1f} | {c['pred_over_gt_ratio']} | "
                 f"{c['strict_iou']:.3f} | {c['tolerance_iou']:.3f} | "
                 f"{c['dice']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} |")
    L.append(f"| **POOLED** | {P['gt_ha']:.1f} | {P['pred_ha']:.1f} | "
             f"{P['pred_minus_gt_ha']:+.1f} | {P['pred_over_gt_ratio']} | "
             f"{P['strict_iou']:.3f} | {P['tolerance_iou']:.3f} | "
             f"{P['dice']:.3f} | {P['precision']:.3f} | {P['recall']:.3f} |")
    L += ["",
          f"**Headline (pooled held-out test blocks):** predicted "
          f"**{P['pred_ha']:.1f} ha** vs Hansen GFC **{P['gt_ha']:.1f} ha** "
          f"({P['pred_over_gt_ratio']}x; {P['pred_minus_gt_ha']:+.1f} ha), strict "
          f"pixel IoU {P['strict_iou']:.3f} (tolerance {P['tolerance_iou']:.3f}).",
          "",
          "`full_region` and `train_only` include pixels the model was trained on "
          "and overstate agreement; the test blocks are the honest number.",
          "",
          "Figures: `results/figures/phase5_deforestation_map.png`, "
          "`results/figures/phase5_hectares.png`.",
          ""]
    (OUT / f"{exp}_summary.md").write_text("\n".join(L), encoding="utf-8")

    print("Per-region held-out test blocks:")
    for rid in rids:
        c = summary["per_region"][rid]["test_only"]
        print(f"  {rid:10s}  GFC {c['gt_ha']:7.1f} ha | pred {c['pred_ha']:7.1f} ha "
              f"({c['pred_over_gt_ratio']}x)  IoU {c['strict_iou']:.3f} strict / "
              f"{c['tolerance_iou']:.3f} tol")
    print(f"  {'POOLED':10s}  GFC {P['gt_ha']:7.1f} ha | pred {P['pred_ha']:7.1f} ha "
          f"({P['pred_over_gt_ratio']}x)  IoU {P['strict_iou']:.3f} strict / "
          f"{P['tolerance_iou']:.3f} tol")
    print(f"\n-> {OUT / (exp + '_area_summary.json')}")
    print(f"-> {OUT / (exp + '_summary.md')}")
    print("-> results/figures/phase5_deforestation_map.png , phase5_hectares.png")


if __name__ == "__main__":
    main()
