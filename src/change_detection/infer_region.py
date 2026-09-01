"""Phase 5 / Phase 8, step 1 - run a trained model over the whole study area,
per region.

Because the model consumes the 8-band bi-temporal stack and predicts the
forest-loss (T -> T+1 change) mask directly, "comparing the T and T+1 masks
pixel-by-pixel" is exactly what one forward pass produces. For each region this
script tiles the full footprint with overlap, averages the per-pixel
probabilities, applies the checkpoint's val-tuned threshold, and writes
georeferenced rasters.

    python -m src.change_detection.infer_region --experiment p8_pooled_unet_s43
    python -m src.change_detection.infer_region --experiment baseline_unet --regions wayanad

Outputs (each region's UTM CRS, 10 m, full region footprint):
    results/deforestation/<exp>__<rid>_prob.tif   float32  mean predicted P(loss)
    results/deforestation/<exp>__<rid>_loss.tif   uint8    P >= threshold AND valid

A single-region (legacy) config still works: `load_regions` yields one region and
the raw/mask rasters are looked up at both data/raw/<rid>/... and the flat
data/raw/... paths.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
import torch

from ..common import CKPT_DIR, REPO, RESULTS, get_device
from ..models.unet import build_model
from ..regions import load_regions

RAW = REPO / "data" / "raw"
MASKS = REPO / "data" / "masks"
OUT = RESULTS / "deforestation"


def _first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    raise SystemExit(f"missing raster; looked for {[str(p) for p in paths]}")


def _region_paths(rid: str):
    s2t = _first_existing(RAW / rid / "s2_T.tif", RAW / "s2_T.tif")
    s2t1 = _first_existing(RAW / rid / "s2_T1.tif", RAW / "s2_T1.tif")
    valid = _first_existing(MASKS / rid / "valid_mask.tif", MASKS / "valid_mask.tif")
    return s2t, s2t1, valid


def _load_stack(rid: str):
    p_t, p_t1, p_v = _region_paths(rid)
    with rasterio.open(p_t) as a, rasterio.open(p_t1) as b, rasterio.open(p_v) as v:
        h = min(a.height, b.height, v.height)
        w = min(a.width, b.width, v.width)
        img = np.concatenate([a.read()[:, :h, :w], b.read()[:, :h, :w]], 0).astype(np.float32)
        valid_land = v.read(1)[:h, :w].astype(bool)
        profile = a.profile
    finite = np.isfinite(img).all(axis=0)
    img = np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    valid = valid_land & finite
    return img, valid, profile, h, w


@torch.no_grad()
def infer_region(rid: str, model, mean, std, thr: float, device: str, exp: str,
                 stride: int = 128) -> dict:
    img, valid, profile, H, W = _load_stack(rid)
    x = (img - mean) / std
    x *= valid[None]

    P = 256                                          # patch size fixed for this project
    prob_sum = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)

    r0s = sorted(set(list(range(0, H - P + 1, stride)) + [H - P]))
    c0s = sorted(set(list(range(0, W - P + 1, stride)) + [W - P]))
    total = len(r0s) * len(c0s)
    done = 0
    for r0 in r0s:
        for c0 in c0s:
            tile = torch.from_numpy(x[:, r0:r0 + P, c0:c0 + P][None]).to(device)
            p = torch.sigmoid(model(tile).float())[0, 0].cpu().numpy()
            prob_sum[r0:r0 + P, c0:c0 + P] += p
            cnt[r0:r0 + P, c0:c0 + P] += 1.0
            done += 1
        print(f"  [{rid}] tiles {done}/{total}", end="\r")
    print()

    prob = np.where(cnt > 0, prob_sum / np.maximum(cnt, 1e-6), 0.0).astype(np.float32)
    prob[~valid] = 0.0
    loss = ((prob >= thr) & valid).astype(np.uint8)

    OUT.mkdir(parents=True, exist_ok=True)
    pf = profile.copy()
    pf.update(count=1, dtype="float32", compress="deflate", nodata=None)
    with rasterio.open(OUT / f"{exp}__{rid}_prob.tif", "w", **pf) as dst:
        dst.write(prob, 1)
        dst.descriptions = ("mean_pred_prob_loss",)
    pf.update(dtype="uint8")
    with rasterio.open(OUT / f"{exp}__{rid}_loss.tif", "w", **pf) as dst:
        dst.write(loss, 1)
        dst.descriptions = (f"pred_loss_ge_{thr:.2f}",)

    gsd = 10
    n_px = int(loss.sum())
    ha = n_px * gsd * gsd / 1e4
    print(f"  [{rid}] threshold {thr:.2f} | predicted loss {n_px:,} px = {ha:.1f} ha")
    return {"region": rid, "pred_loss_px": n_px, "pred_loss_ha": round(ha, 2),
            "valid_px": int(valid.sum())}


@torch.no_grad()
def run(experiment: str, region_ids: list[str] | None, stride: int) -> None:
    ckpt = torch.load(CKPT_DIR / f"{experiment}_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    # authoritative operating threshold = the one src.eval.evaluate tuned on val
    ev_path = RESULTS / "metrics" / f"{experiment}.json"
    if ev_path.exists():
        thr = float(json.loads(ev_path.read_text())["operating_threshold"])
    else:
        thr = float(ckpt.get("val_threshold", 0.5))
    device = get_device()
    kw = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(cfg["model"]["name"], **kw).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm = json.loads((REPO / cfg["data"]["proc_dir"] / "norm_stats.json").read_text())
    mean = np.asarray(norm["mean"], np.float32)[:, None, None]
    std = np.asarray(norm["std"], np.float32)[:, None, None]

    regions = load_regions()
    if region_ids:
        want = set(region_ids)
        regions = [r for r in regions if r["id"] in want]
        if not regions:
            raise SystemExit(f"no regions match {sorted(want)}")

    summary = {"experiment": experiment, "operating_threshold": thr,
               "stride": stride, "regions": {}}
    for r in regions:
        summary["regions"][r["id"]] = infer_region(
            r["id"], model, mean, std, thr, device, experiment, stride)
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / f"{experiment}_infer_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / (experiment + '_infer_summary.json')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline_unet")
    ap.add_argument("--regions", default=None,
                    help="comma-separated region ids (default: all in configs/region.yaml)")
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()
    ids = [s.strip() for s in args.regions.split(",")] if args.regions else None
    run(args.experiment, ids, args.stride)


if __name__ == "__main__":
    main()
