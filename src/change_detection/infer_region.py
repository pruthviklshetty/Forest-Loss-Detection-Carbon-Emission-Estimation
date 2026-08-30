"""Phase 5, step 1 - run a trained model over the whole study region.

Because the model consumes the 8-band bi-temporal stack and predicts the
forest-loss (T -> T+1 change) mask directly, "comparing the T and T+1 masks
pixel-by-pixel" is exactly what one forward pass produces. This script tiles
the full region with overlap, averages the per-pixel probabilities, applies
the checkpoint's val-tuned threshold, and writes georeferenced rasters.

    python -m src.change_detection.infer_region --experiment baseline_unet

Outputs (EPSG:32643, 10 m, full region footprint):
    results/deforestation/<exp>_prob.tif    float32  mean predicted P(loss)
    results/deforestation/<exp>_loss.tif     uint8    P >= threshold AND valid
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
import torch

from ..common import CKPT_DIR, REPO, RESULTS, get_device, load_yaml
from ..models.unet import build_model

RAW = REPO / "data" / "raw"
MASKS = REPO / "data" / "masks"
OUT = RESULTS / "deforestation"


def _load_stack():
    with rasterio.open(RAW / "s2_T.tif") as a, rasterio.open(RAW / "s2_T1.tif") as b, \
         rasterio.open(MASKS / "valid_mask.tif") as v:
        h = min(a.height, b.height, v.height)
        w = min(a.width, b.width, v.width)
        img = np.concatenate([a.read()[:, :h, :w], b.read()[:, :h, :w]], 0).astype(np.float32)
        valid_land = v.read(1)[:h, :w].astype(bool)
        profile = a.profile
        transform = a.transform
        crs = a.crs
    finite = np.isfinite(img).all(axis=0)
    img = np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    valid = valid_land & finite
    return img, valid, profile, transform, crs, h, w


@torch.no_grad()
def infer(experiment: str, stride: int = 128) -> None:
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

    img, valid, profile, transform, crs, H, W = _load_stack()
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
        print(f"  tiles {done}/{total}", end="\r")
    print()

    prob = np.where(cnt > 0, prob_sum / np.maximum(cnt, 1e-6), 0.0).astype(np.float32)
    prob[~valid] = 0.0
    loss = ((prob >= thr) & valid).astype(np.uint8)

    OUT.mkdir(parents=True, exist_ok=True)
    pf = profile.copy()
    pf.update(count=1, dtype="float32", compress="deflate", nodata=None)
    with rasterio.open(OUT / f"{experiment}_prob.tif", "w", **pf) as dst:
        dst.write(prob, 1)
        dst.descriptions = ("mean_pred_prob_loss",)
    pf.update(dtype="uint8")
    with rasterio.open(OUT / f"{experiment}_loss.tif", "w", **pf) as dst:
        dst.write(loss, 1)
        dst.descriptions = (f"pred_loss_ge_{thr:.2f}",)

    gsd = 10
    ha = int(loss.sum()) * gsd * gsd / 1e4
    print(f"  threshold {thr:.2f} | predicted loss pixels {int(loss.sum()):,} "
          f"= {ha:.1f} ha over the full region")
    print(f"  -> {OUT / (experiment + '_prob.tif')}")
    print(f"  -> {OUT / (experiment + '_loss.tif')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline_unet")
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()
    infer(args.experiment, args.stride)


if __name__ == "__main__":
    main()
