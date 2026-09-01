"""Phase 2/8, step 3 - build the 8-band patch dataset, per region, plus two
split schemes.

Per region, inputs (that region's UTM CRS, 10 m, identical footprint):
    data/raw/<id>/s2_T.tif    4 bands  green, red, nir, swir1  (reflectance [0,1])
    data/raw/<id>/s2_T1.tif   4 bands
    data/masks/<id>/loss_label.tif   1 band 0/1
    data/masks/<id>/valid_mask.tif   1 band 0/1

Output:
    data/processed/patches/<id>__<pid>.npz   img (8,256,256) f32 | label u8 | valid u8
    data/processed/index.csv     per-patch rows: patch_id, region, px_r0, px_c0,
                                 size, is_overlap, pooled_split, ... loss/valid stats
    data/processed/norm_stats.json   per-band mean/std over POOLED canonical train patches
    data/processed/split.json    pooled split summary (per region + pooled totals)
    data/processed/loro.json     leave-one-region-out folds (canonical patches only)

8-band channel order: [T_green, T_red, T_nir, T_swir1, T1_green, T1_red, T1_nir, T1_swir1].

Two split schemes:
  - POOLED (primary): each region is spatially blocked into 512 px super-blocks;
    whole blocks -> train/val/test (seed 42, 70/15/15). Regions are then pooled.
    Stride-128 overlap crops are added to a region's train, but only where the
    ENTIRE 256x256 footprint lies inside that region's train blocks.
  - LORO: fold k tests on ALL of region k's canonical patches; trains on the
    other three regions' canonical patches (pooled train+test) plus their
    overlap crops; validates on the other three regions' pooled-val canonical
    patches. Overlap crops never touch a pooled-val block, so LORO-val is clean.

Run:  python -m src.preprocessing.build_dataset               # all regions
      python -m src.preprocessing.build_dataset --regions wayanad
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random

import numpy as np
import rasterio

from ..paths import masks_dir as _masks_dir
from ..paths import proc_dir as _proc_dir
from ..paths import raw_dir as _raw_dir
from ..regions import load_regions
from .eeutil import load_cfg

_REPO = pathlib.Path(__file__).resolve().parents[2]
# defaults (2019->2021 layout); main() rebinds these from the loaded config so a
# `period_id` routes everything under data/*/<period_id>/
_RAW = _REPO / "data" / "raw"
_MASKS = _REPO / "data" / "masks"
_PROC = _REPO / "data" / "processed"

BAND_ORDER = ["T_green", "T_red", "T_nir", "T_swir1",
              "T1_green", "T1_red", "T1_nir", "T1_swir1"]


def _open_checked(path: pathlib.Path):
    if not path.exists():
        raise SystemExit(f"missing {path}; run download_data / build_labels first")
    return rasterio.open(path)


def _load_stack(rid: str):
    rraw, rmask = _RAW / rid, _MASKS / rid
    with _open_checked(rraw / "s2_T.tif") as a, _open_checked(rraw / "s2_T1.tif") as b, \
         _open_checked(rmask / "loss_label.tif") as lab, _open_checked(rmask / "valid_mask.tif") as val:
        shapes = {a.shape, b.shape, lab.shape, val.shape}
        h = min(s[0] for s in shapes)
        w = min(s[1] for s in shapes)
        if len(shapes) > 1:
            print(f"  [{rid}] source shapes differ {shapes}; cropping to ({h},{w})")
        s2t = a.read()[:, :h, :w].astype(np.float32)
        s2t1 = b.read()[:, :h, :w].astype(np.float32)
        label = lab.read(1)[:h, :w].astype(np.uint8)
        valid = val.read(1)[:h, :w].astype(np.uint8)
    img = np.concatenate([s2t, s2t1], axis=0)
    if img.shape[0] != 8:
        raise SystemExit(f"[{rid}] expected 8 bands, got {img.shape[0]}")
    finite = np.isfinite(img).all(axis=0)
    n_nf = int((~finite).sum())
    if n_nf:
        print(f"  [{rid}] {n_nf} px ({100 * n_nf / finite.size:.3f}%) non-finite in S2 "
              f"-> zero-filled, marked invalid")
    img = np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    valid = (valid.astype(bool) & finite).astype(np.uint8)
    return img, label, valid


def _blocked_split(rows: int, cols: int, block: int, fracs: dict, seed: int) -> dict:
    blocks = [(br, bc) for br in range(0, rows, block) for bc in range(0, cols, block)]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    n = len(blocks)
    n_train = round(fracs["train"] * n)
    n_val = round(fracs["val"] * n)
    assign = {}
    for i, blk in enumerate(blocks):
        assign[blk] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assign


def process_region(region: dict, params: dict, rows: list, stats: dict) -> None:
    rid = region["id"]
    P, stride, block = params["P"], params["stride"], params["block"]
    tr_overlap = params["tr_overlap"]
    img, label, valid = _load_stack(rid)
    _, H, W = img.shape
    starts_r = list(range(0, H - P + 1, stride))
    starts_c = list(range(0, W - P + 1, stride))
    block_assign = _blocked_split(len(starts_r), len(starts_c), block,
                                  params["fracs"], params["seed"])
    print(f"  [{rid}] 8 x {H} x {W}  ->  {len(starts_r)}x{len(starts_c)} = "
          f"{len(starts_r) * len(starts_c)} canonical patches")

    def canonical_split(r0, c0):
        ri = min(len(starts_r) - 1, r0 // stride)
        ci = min(len(starts_c) - 1, c0 // stride)
        return block_assign[(ri - ri % block, ci - ci % block)]

    def overlap_crop_all_train(r0, c0):
        ri0, ri1 = r0 // stride, (r0 + P - 1) // stride
        ci0, ci1 = c0 // stride, (c0 + P - 1) // stride
        for ri in range(ri0, ri1 + 1):
            for ci in range(ci0, ci1 + 1):
                ri_c = min(ri, len(starts_r) - 1)
                ci_c = min(ci, len(starts_c) - 1)
                if block_assign[(ri_c - ri_c % block, ci_c - ci_c % block)] != "train":
                    return False
        return True

    def emit(pid, r0, c0, split, is_overlap):
        sl = (slice(r0, r0 + P), slice(c0, c0 + P))
        pimg = img[:, sl[0], sl[1]].astype(np.float32).copy()
        plab = label[sl].astype(np.uint8).copy()
        pval = valid[sl].astype(np.uint8).copy()
        n_valid, n_loss = int(pval.sum()), int(plab.sum())
        rows.append({
            "patch_id": pid, "region": rid, "px_r0": r0, "px_c0": c0, "size": P,
            "is_overlap": is_overlap, "pooled_split": split,
            "n_valid_px": n_valid, "valid_frac": round(n_valid / (P * P), 4),
            "n_loss_px": n_loss, "loss_frac": round(n_loss / (P * P), 6),
            "loss_frac_of_valid": round(n_loss / max(n_valid, 1), 6),
            "has_loss": int(n_loss > 0),
        })
        np.savez_compressed(_PROC / "patches" / f"{pid}.npz", img=pimg, label=plab, valid=pval)
        if split == "train" and not is_overlap:
            m = pval.astype(bool)
            if m.any():
                v = pimg[:, m]
                stats["sums"] += v.sum(axis=1)
                stats["sq"] += (v ** 2).sum(axis=1)
                stats["n"] += int(m.sum())

    for ri, r0 in enumerate(starts_r):
        for ci, c0 in enumerate(starts_c):
            emit(f"{rid}__p_{ri:02d}_{ci:02d}", r0, c0, canonical_split(r0, c0), 0)

    n_ov = n_drop = 0
    if tr_overlap and tr_overlap < P:
        canon = {(r0, c0) for r0 in starts_r for c0 in starts_c}
        for r0 in range(0, H - P + 1, tr_overlap):
            for c0 in range(0, W - P + 1, tr_overlap):
                if (r0, c0) in canon:
                    continue
                if not overlap_crop_all_train(r0, c0):
                    n_drop += 1
                    continue
                emit(f"{rid}__p_ov_{r0:05d}_{c0:05d}", r0, c0, "train", 1)
                n_ov += 1
        print(f"  [{rid}] + {n_ov} overlap train crops ({n_drop} dropped for touching val/test)")


def _pooled_summary(rows, gsd):
    def stat(subset):
        canr = [r for r in subset if not r["is_overlap"]]
        loss_px = sum(r["n_loss_px"] for r in canr)
        valid_px = sum(r["n_valid_px"] for r in canr)
        return {
            "n_patches": len(subset),
            "n_patches_canonical": len(canr),
            "n_patches_overlap": len(subset) - len(canr),
            "n_patches_with_loss": sum(1 for r in subset if r["has_loss"]),
            "loss_px_pct_of_valid": round(100 * loss_px / max(valid_px, 1), 4),
            "ha_lost_canonical": round(loss_px * gsd * gsd / 1e4, 2),
        }

    out = {"pooled": {s: stat([r for r in rows if r["pooled_split"] == s])
                      for s in ("train", "val", "test")},
           "per_region": {}}
    for rid in sorted({r["region"] for r in rows}):
        rr = [r for r in rows if r["region"] == rid]
        out["per_region"][rid] = {s: stat([r for r in rr if r["pooled_split"] == s])
                                  for s in ("train", "val", "test")}
    return out


def _loro_folds(rows):
    """Leave-one-region-out. Canonical patches only for train/val/test; plus the
    other regions' overlap crops for train (they never touch a pooled-val block)."""
    regions = sorted({r["region"] for r in rows})
    by = {r["patch_id"]: r for r in rows}
    folds = []
    for test_rid in regions:
        train_regions = [x for x in regions if x != test_rid]
        test_ids = [pid for pid, r in by.items()
                    if r["region"] == test_rid and not r["is_overlap"]]
        val_ids = [pid for pid, r in by.items()
                   if r["region"] in train_regions and not r["is_overlap"]
                   and r["pooled_split"] == "val"]
        train_ids = [pid for pid, r in by.items()
                     if r["region"] in train_regions
                     and ((not r["is_overlap"] and r["pooled_split"] in ("train", "test"))
                          or r["is_overlap"])]
        folds.append({
            "test_region": test_rid,
            "train_regions": train_regions,
            "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
            "n_train_canonical": sum(1 for i in train_ids if not by[i]["is_overlap"]),
            "test_pos_px_pct_of_valid": round(
                100 * sum(by[i]["n_loss_px"] for i in test_ids)
                / max(sum(by[i]["n_valid_px"] for i in test_ids), 1), 4),
            "ids": {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)},
        })
    return {"folds": folds}


def main() -> None:
    global _RAW, _MASKS, _PROC
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=None)
    ap.add_argument("--config", default="configs/region.yaml",
                    help="config file (use configs/period_2021_2023.yaml for Phase 10)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    _RAW, _MASKS, _PROC = _raw_dir(cfg), _masks_dir(cfg), _proc_dir(cfg)
    if cfg.get("period_id"):
        print(f"period: {cfg['period_id']}  ->  {_PROC.relative_to(_REPO).as_posix()}/")
    regions = load_regions(cfg)
    if args.regions:
        want = {s.strip() for s in args.regions.split(",")}
        regions = [r for r in regions if r["id"] in want]
    gsd = int(regions[0]["gsd_m"])

    pc = cfg["patching"]
    params = {
        "P": int(pc["patch_size_px"]), "stride": int(pc["stride_px"]),
        "block": int(pc.get("split_block_patches", 2)),
        "fracs": pc["split_fractions"], "seed": int(pc["split_seed"]),
        "tr_overlap": int(pc["train_overlap_stride_px"]) if pc.get("train_overlap_stride_px") else None,
    }

    (_PROC / "patches").mkdir(parents=True, exist_ok=True)
    for old in (_PROC / "patches").glob("*.npz"):
        old.unlink()

    rows: list[dict] = []
    stats = {"sums": np.zeros(8, np.float64), "sq": np.zeros(8, np.float64), "n": 0}
    for region in regions:
        process_region(region, params, rows, stats)

    mean = stats["sums"] / max(stats["n"], 1)
    std = np.sqrt(np.maximum(stats["sq"] / max(stats["n"], 1) - mean ** 2, 1e-12))
    (_PROC / "norm_stats.json").write_text(json.dumps({
        "band_order": BAND_ORDER,
        "computed_over": "valid pixels of POOLED canonical (non-overlap) train patches",
        "regions": [r["id"] for r in regions],
        "n_pixels": int(stats["n"]),
        "mean": [round(float(x), 6) for x in mean],
        "std": [round(float(x), 6) for x in std],
    }, indent=2), encoding="utf-8")

    with open(_PROC / "index.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    pooled = _pooled_summary(rows, gsd)
    pooled.update({"patch_size_px": params["P"], "stride_px": params["stride"],
                   "train_overlap_stride_px": params["tr_overlap"],
                   "split_seed": params["seed"], "split_block_patches": params["block"],
                   "band_order": BAND_ORDER,
                   "regions": [r["id"] for r in regions],
                   "n_patches": len(rows),
                   "ids": {s: sorted(r["patch_id"] for r in rows if r["pooled_split"] == s)
                           for s in ("train", "val", "test")}})
    (_PROC / "split.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    (_PROC / "loro.json").write_text(json.dumps(_loro_folds(rows), indent=2), encoding="utf-8")

    print(f"\n  wrote {len(rows)} patches across {len(regions)} regions")
    print("  POOLED split:")
    for s in ("train", "val", "test"):
        d = pooled["pooled"][s]
        print(f"    {s:5s}: {d['n_patches']:4d} ({d['n_patches_canonical']} canon + "
              f"{d['n_patches_overlap']} ov)  {d['n_patches_with_loss']:3d} w/loss  "
              f"pos {d['loss_px_pct_of_valid']:.3f}%  {d['ha_lost_canonical']:.1f} ha")
    print("  LORO folds (test region -> n_train / n_val / n_test canon):")
    for f in _loro_folds(rows)["folds"]:
        print(f"    test={f['test_region']:9s}  train {f['n_train']:4d} / val {f['n_val']:3d} "
              f"/ test {f['n_test']:3d}  (test pos {f['test_pos_px_pct_of_valid']:.3f}%)")
    print("  index.csv / split.json / loro.json / norm_stats.json written")


if __name__ == "__main__":
    main()
