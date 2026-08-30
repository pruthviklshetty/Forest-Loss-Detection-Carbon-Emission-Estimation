"""Phase 2, step 3 - build the 8-band patch dataset and the train/val/test split.

Inputs (all EPSG:32643, 10 m, identical footprint):
    data/raw/s2_T.tif        4 bands  green, red, nir, swir1   (reflectance, [0,1])
    data/raw/s2_T1.tif       4 bands  green, red, nir, swir1
    data/masks/loss_label.tif   1 band  0 / 1  forest-loss target
    data/masks/valid_mask.tif   1 band  0 / 1  usable pixel

Output:
    data/processed/patches/<id>.npz   img (8,256,256) float32 | label (256,256) uint8 | valid (256,256) uint8
    data/processed/index.csv          per-patch metadata + split assignment
    data/processed/split.json         {train:[ids], val:[...], test:[...]} + parameters
    data/processed/norm_stats.json    per-band mean/std over TRAIN patches (optional z-score at train time)

8-band channel order (configs/region.yaml stack_layout):
    [T_green, T_red, T_nir, T_swir1, T1_green, T1_red, T1_nir, T1_swir1]

Split: spatially blocked. Patches are grouped into BLOCK x BLOCK super-blocks;
whole blocks are assigned to train/val/test by a seeded shuffle so no test
patch is edge-adjacent to a train patch (limits spatial leakage). Actual patch
fractions and per-split positive rates are printed and stored.

Run:  python -m src.preprocessing.build_dataset
"""

from __future__ import annotations

import csv
import json
import pathlib
import random

import numpy as np
import rasterio

from .eeutil import load_cfg

_REPO = pathlib.Path(__file__).resolve().parents[2]
_RAW = _REPO / "data" / "raw"
_MASKS = _REPO / "data" / "masks"
_PROC = _REPO / "data" / "processed"

BAND_ORDER = ["T_green", "T_red", "T_nir", "T_swir1",
              "T1_green", "T1_red", "T1_nir", "T1_swir1"]


def _open_checked(path: pathlib.Path):
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return rasterio.open(path)


def _load_stack() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with _open_checked(_RAW / "s2_T.tif") as a, \
         _open_checked(_RAW / "s2_T1.tif") as b, \
         _open_checked(_MASKS / "loss_label.tif") as lab, \
         _open_checked(_MASKS / "valid_mask.tif") as val:
        shapes = {a.shape, b.shape, lab.shape, val.shape}
        h = min(s[0] for s in shapes)
        w = min(s[1] for s in shapes)
        if len(shapes) > 1:
            print(f"  note: source shapes differ {shapes}; cropping to common ({h},{w})")
        s2t = a.read()[:, :h, :w].astype(np.float32)
        s2t1 = b.read()[:, :h, :w].astype(np.float32)
        label = lab.read(1)[:h, :w].astype(np.uint8)
        valid = val.read(1)[:h, :w].astype(np.uint8)
        meta = {"crs": str(a.crs), "transform": a.transform, "height": h, "width": w,
                "bands_T": list(a.descriptions), "bands_T1": list(b.descriptions)}
    img = np.concatenate([s2t, s2t1], axis=0)          # (8, h, w)
    if img.shape[0] != 8:
        raise SystemExit(f"expected 8 bands, got {img.shape[0]}")

    # Earth Engine exports fully-cloud-masked (no clear observation in the
    # window) pixels as non-finite. Track them as a data-coverage mask and
    # zero-fill so nothing downstream sees inf/nan; such pixels are then
    # excluded from `valid` (and therefore from training loss and metrics).
    finite = np.isfinite(img).all(axis=0)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        print(f"  note: {n_nonfinite} px ({100 * n_nonfinite / finite.size:.3f}%) "
              f"non-finite in S2 (persistent cloud) -> zero-filled, marked invalid")
    img = np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    valid = (valid.astype(bool) & finite).astype(np.uint8)
    return img, label, valid, meta


def _blocked_split(rows: int, cols: int, block: int, fracs: dict, seed: int) -> dict:
    blocks = [(br, bc) for br in range(0, rows, block) for bc in range(0, cols, block)]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    n = len(blocks)
    n_train = round(fracs["train"] * n)
    n_val = round(fracs["val"] * n)
    assign = {}
    for i, blk in enumerate(blocks):
        split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        assign[blk] = split
    return assign


def main() -> None:
    cfg = load_cfg()
    P = int(cfg["patching"]["patch_size_px"])
    stride = int(cfg["patching"]["stride_px"])
    fracs = cfg["patching"]["split_fractions"]
    seed = int(cfg["patching"]["split_seed"])
    block = int(cfg["patching"].get("split_block_patches", 2))

    img, label, valid, meta = _load_stack()
    _, H, W = img.shape
    print(f"  stack: 8 x {H} x {W}  ({BAND_ORDER})")

    starts_r = list(range(0, H - P + 1, stride))
    starts_c = list(range(0, W - P + 1, stride))
    print(f"  patch grid: {len(starts_r)} rows x {len(starts_c)} cols "
          f"= {len(starts_r) * len(starts_c)} patches of {P}x{P}")

    block_assign = _blocked_split(len(starts_r), len(starts_c), block, fracs, seed)

    (_PROC / "patches").mkdir(parents=True, exist_ok=True)
    for old in (_PROC / "patches").glob("*.npz"):
        old.unlink()

    rows = []
    train_pixel_sums = np.zeros(8, np.float64)
    train_pixel_sqsums = np.zeros(8, np.float64)
    train_pixel_count = 0

    for ri, r0 in enumerate(starts_r):
        for ci, c0 in enumerate(starts_c):
            pid = f"p_{ri:02d}_{ci:02d}"
            sl = (slice(r0, r0 + P), slice(c0, c0 + P))
            pimg = img[:, sl[0], sl[1]].copy()
            plab = label[sl].copy()
            pval = valid[sl].copy()
            split = block_assign[(ri - ri % block, ci - ci % block)]

            n_valid = int(pval.sum())
            n_loss = int(plab.sum())
            rows.append({
                "patch_id": pid, "row_idx": ri, "col_idx": ci,
                "px_r0": r0, "px_c0": c0, "size": P,
                "split": split,
                "n_valid_px": n_valid,
                "valid_frac": round(n_valid / (P * P), 4),
                "n_loss_px": n_loss,
                "loss_frac": round(n_loss / (P * P), 6),
                "loss_frac_of_valid": round(n_loss / max(n_valid, 1), 6),
                "has_loss": int(n_loss > 0),
            })
            np.savez_compressed(_PROC / "patches" / f"{pid}.npz",
                                img=pimg.astype(np.float32),
                                label=plab.astype(np.uint8),
                                valid=pval.astype(np.uint8))
            if split == "train":
                m = pval.astype(bool)
                if m.any():
                    v = pimg[:, m]                       # (8, k)
                    train_pixel_sums += v.sum(axis=1)
                    train_pixel_sqsums += (v ** 2).sum(axis=1)
                    train_pixel_count += m.sum()

    mean = train_pixel_sums / max(train_pixel_count, 1)
    var = train_pixel_sqsums / max(train_pixel_count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    norm = {"band_order": BAND_ORDER,
            "computed_over": "valid pixels of train patches",
            "n_pixels": int(train_pixel_count),
            "mean": [round(float(x), 6) for x in mean],
            "std": [round(float(x), 6) for x in std]}
    (_PROC / "norm_stats.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")

    with open(_PROC / "index.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    split_lists = {s: [r["patch_id"] for r in rows if r["split"] == s]
                   for s in ("train", "val", "test")}
    gsd = int(cfg["region"]["target_gsd_m"])
    summary = {
        "patch_size_px": P, "stride_px": stride,
        "split_fractions_target": fracs, "split_seed": seed,
        "split_block_patches": block,
        "split_method": "spatially blocked (whole super-blocks per split)",
        "band_order": BAND_ORDER,
        "n_patches": len(rows),
        "splits": {s: {
            "n_patches": len(ids),
            "patch_frac": round(len(ids) / len(rows), 3),
            "n_patches_with_loss": sum(1 for r in rows if r["split"] == s and r["has_loss"]),
            "loss_px": sum(r["n_loss_px"] for r in rows if r["split"] == s),
            "valid_px": sum(r["n_valid_px"] for r in rows if r["split"] == s),
            "loss_px_pct_of_valid": round(
                100 * sum(r["n_loss_px"] for r in rows if r["split"] == s)
                / max(sum(r["n_valid_px"] for r in rows if r["split"] == s), 1), 4),
            "ha_lost": round(sum(r["n_loss_px"] for r in rows if r["split"] == s) * gsd * gsd / 1e4, 2),
        } for s, ids in split_lists.items()},
        "ids": split_lists,
    }
    (_PROC / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  wrote {len(rows)} patches to data/processed/patches/")
    for s in ("train", "val", "test"):
        d = summary["splits"][s]
        print(f"    {s:5s}: {d['n_patches']:3d} patches ({d['patch_frac']:.0%}), "
              f"{d['n_patches_with_loss']:3d} with loss, "
              f"positive rate {d['loss_px_pct_of_valid']:.3f}% of valid px, "
              f"{d['ha_lost']:.1f} ha")
    print("  index.csv / split.json / norm_stats.json written")


if __name__ == "__main__":
    main()
