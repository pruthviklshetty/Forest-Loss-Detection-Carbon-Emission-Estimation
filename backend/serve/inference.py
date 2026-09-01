"""Model inference + carbon for one job.

Tiling is byte-identical to ``src/change_detection/infer_region.py``: 256 px
tiles, stride 128, per-pixel probabilities averaged over overlaps, z-scored with
the checkpoint's ``norm_stats.json``, masked pixels set to 0, then thresholded
at the val-tuned operating threshold. Carbon reuses ``src.carbon`` (exponential
regression primary, 3-bin baseline alongside).
"""

from __future__ import annotations

import json

import numpy as np
import rasterio
import torch

from .config import CHECKPOINT, EVAL_JSON, REPO, TILE_PX, TILE_STRIDE

from src.models.unet import build_model  # noqa: E402
from src.carbon.ndvi import compute_ndvi, three_bin_carbon_density  # noqa: E402
from src.carbon.regression_model import build_calibration, fit, predict  # noqa: E402

HA_PER_PX = 0.01
CO2_PER_C = 44.0 / 12.0
_BAND_T = ["green", "red", "nir", "swir1"]     # order in each composite raster


def _read_composite(path):
    """Return (img[4,H,W] float32 clipped [0,1], obs[H,W] bool)."""
    with rasterio.open(path) as s:
        arr = s.read().astype(np.float32)
        transform, crs = s.transform, s.crs
    bands, obs = arr[:4], (arr[4] if arr.shape[0] >= 5 else np.ones_like(arr[0]))
    finite = np.isfinite(bands).all(axis=0)
    bands = np.clip(np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    valid = finite & np.isfinite(obs) & (obs > 0.5)
    return bands, valid, transform, crs


class Segmenter:
    def __init__(self) -> None:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        kw = {k: v for k, v in self.cfg["model"].items() if k != "name"}
        self.model = build_model(self.cfg["model"]["name"], **kw)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        norm = json.loads((REPO / self.cfg["data"]["proc_dir"] / "norm_stats.json")
                          .read_text(encoding="utf-8"))
        self.mean = np.asarray(norm["mean"], np.float32)[:, None, None]
        self.std = np.asarray(norm["std"], np.float32)[:, None, None]

        if EVAL_JSON.exists():
            self.threshold = float(json.loads(EVAL_JSON.read_text())["operating_threshold"])
        else:
            self.threshold = float(ckpt.get("val_threshold", 0.5))
        self.checkpoint_epoch = ckpt.get("epoch")

    @torch.no_grad()
    def predict(self, path_t, path_t1, progress=None):
        b_t, v_t, transform, crs = _read_composite(path_t)
        b_t1, v_t1, _, _ = _read_composite(path_t1)
        h = min(b_t.shape[1], b_t1.shape[1])
        w = min(b_t.shape[2], b_t1.shape[2])
        img = np.concatenate([b_t[:, :h, :w], b_t1[:, :h, :w]], axis=0)     # (8,H,W)
        valid = (v_t[:h, :w] & v_t1[:h, :w])

        x = (img - self.mean) / self.std
        x *= valid[None]
        P, stride = TILE_PX, TILE_STRIDE
        H, W = h, w
        if H < P or W < P:
            raise ValueError(f"AOI {W}x{H} px is smaller than one {P} px tile; "
                             "request a larger bbox.")
        prob_sum = np.zeros((H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)
        r0s = sorted(set(list(range(0, H - P + 1, stride)) + [H - P]))
        c0s = sorted(set(list(range(0, W - P + 1, stride)) + [W - P]))
        total = len(r0s) * len(c0s)
        done = 0
        for r0 in r0s:
            for c0 in c0s:
                tile = torch.from_numpy(x[:, r0:r0 + P, c0:c0 + P][None]).to(self.device)
                p = torch.sigmoid(self.model(tile).float())[0, 0].cpu().numpy()
                prob_sum[r0:r0 + P, c0:c0 + P] += p
                cnt[r0:r0 + P, c0:c0 + P] += 1.0
                done += 1
                if progress:
                    progress(done / total)
        prob = np.where(cnt > 0, prob_sum / np.maximum(cnt, 1e-6), 0.0).astype(np.float32)
        prob[~valid] = 0.0
        loss = ((prob >= self.threshold) & valid).astype(np.uint8)
        return {
            "prob": prob, "loss": loss, "valid": valid,
            "img_t": b_t[:, :h, :w], "transform": transform, "crs": str(crs),
            "gsd_m": _gsd_from_transform(transform),
        }


def _gsd_from_transform(transform) -> float:
    return float(round(abs(transform.a), 3))


_CARBON_COEFS = None


def _carbon_coefs():
    global _CARBON_COEFS
    if _CARBON_COEFS is None:
        _CARBON_COEFS = fit(build_calibration())
    return _CARBON_COEFS


def estimate(pred: dict) -> dict:
    """Hectares + committed aboveground CO2 for the predicted-loss pixels."""
    loss = pred["loss"].astype(bool)
    valid = pred["valid"].astype(bool)
    gsd = pred["gsd_m"] or 10.0
    ha_per_px = gsd * gsd / 1e4

    green, red, nir = pred["img_t"][0], pred["img_t"][1], pred["img_t"][2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = compute_ndvi(nir, red)
    veg = valid & np.isfinite(ndvi) & (ndvi > 0.3)
    if veg.sum() >= 100:
        lo = float(np.percentile(ndvi[veg], 33.3))
        hi = float(np.percentile(ndvi[veg], 66.7))
    else:
        lo, hi = 0.5, 0.7

    m = loss & valid & np.isfinite(ndvi)
    n_px = int(m.sum())
    area_ha = round(n_px * ha_per_px, 2)
    coefs = _carbon_coefs()
    out = {
        "predicted_loss_pixels": n_px,
        "predicted_loss_ha": area_ha,
        "gsd_m": gsd,
        "co2_scope": "aboveground carbon only; committed CO2 only (x 44/12). "
                     "No belowground / deadwood / litter / soil pools, no "
                     "non-CO2 gases, no regrowth. NDVI->carbon calibrated to "
                     "published Western Ghats field means, not pixel-matched "
                     "biomass.",
        "carbon_primary": "regression_exponential",
    }
    if n_px == 0:
        out.update({"co2_tonnes_exponential": 0.0, "co2_tonnes_three_bin": 0.0,
                    "mean_agc_tC_ha": 0.0})
        return out
    v = ndvi[m]
    agc_exp = predict(v, coefs, "exponential")
    agc_bin = three_bin_carbon_density(v, lo, hi)
    tC_exp = float(np.sum(agc_exp) * ha_per_px)
    tC_bin = float(np.sum(agc_bin) * ha_per_px)
    out.update({
        "co2_tonnes_exponential": round(tC_exp * CO2_PER_C, 1),
        "co2_tonnes_three_bin": round(tC_bin * CO2_PER_C, 1),
        "mean_agc_tC_ha": round(float(np.mean(agc_exp)), 1),
        "ndvi_terciles": [round(lo, 4), round(hi, 4)],
    })
    return out
