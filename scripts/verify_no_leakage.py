"""Data-leakage check for the Phase 2 patch split.

The train split adds stride-128 overlapping 256x256 crops inside train
super-blocks; val/test are canonical non-overlapping patches assigned by
spatial block. This script verifies that no train crop's pixel extent
intersects any val or test patch's pixel extent in the full-region raster
grid.

It is deliberately independent of build_dataset.py's split logic: it only
reads the recorded pixel origin (px_r0, px_c0) and size of every patch from
data/processed/index.csv and does rectangle intersection.

Exit 0 if clean, 1 if any val/test patch overlaps any train patch.

    python scripts/verify_no_leakage.py
"""

from __future__ import annotations

import csv
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
INDEX = REPO / "data" / "processed" / "index.csv"


def _extent(row: dict) -> tuple[int, int, int, int]:
    r0 = int(row["px_r0"])
    c0 = int(row["px_c0"])
    s = int(row["size"])
    return r0, r0 + s, c0, c0 + s          # row_min, row_max, col_min, col_max


def _overlap_px(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0a, r1a, c0a, c1a = a
    r0b, r1b, c0b, c1b = b
    rov = max(0, min(r1a, r1b) - max(r0a, r0b))
    cov = max(0, min(c1a, c1b) - max(c0a, c0b))
    return rov * cov


def main() -> int:
    if not INDEX.exists():
        print(f"ERROR: {INDEX} not found", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(open(INDEX, encoding="utf-8")))
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)

    print("Patches per split:")
    for s in ("train", "val", "test"):
        n = len(by_split.get(s, []))
        n_ov = sum(1 for r in by_split.get(s, []) if int(r["is_overlap"]))
        print(f"  {s:5s}: {n:4d}  ({n - n_ov} canonical + {n_ov} overlap)")
    print()

    train_ext = [(r["patch_id"], int(r["is_overlap"]), _extent(r))
                 for r in by_split["train"]]

    offenders: list[dict] = []
    for s in ("val", "test"):
        for r in by_split[s]:
            ext = _extent(r)
            hits = []
            for tid, t_is_ov, t_ext in train_ext:
                px = _overlap_px(ext, t_ext)
                if px > 0:
                    hits.append((tid, t_is_ov, px))
            if hits:
                total_px = sum(h[2] for h in hits)
                # unique leaked area of THIS patch = union of intersection
                # rectangles, rasterised into the patch's own 0..size grid
                r0, r1, c0, c1 = ext
                size = r1 - r0
                seen = [[False] * size for _ in range(size)]
                for _, _, _ in hits:
                    pass
                for tid, t_is_ov, _px in hits:
                    t = next(te for te in train_ext if te[0] == tid)[2]
                    ir0 = max(r0, t[0]) - r0
                    ir1 = min(r1, t[1]) - r0
                    ic0 = max(c0, t[2]) - c0
                    ic1 = min(c1, t[3]) - c0
                    for rr in range(ir0, ir1):
                        row = seen[rr]
                        for cc in range(ic0, ic1):
                            row[cc] = True
                unique_px = sum(v for row in seen for v in row)
                offenders.append({"split": s, "patch_id": r["patch_id"],
                                  "extent": ext, "hits": hits,
                                  "total_px": total_px, "unique_px": unique_px,
                                  "unique_frac": unique_px / (size * size)})

    n_val_off = sum(1 for o in offenders if o["split"] == "val")
    n_test_off = sum(1 for o in offenders if o["split"] == "test")
    print(f"val patches with any train overlap : {n_val_off} / {len(by_split['val'])}")
    print(f"test patches with any train overlap: {n_test_off} / {len(by_split['test'])}")
    print()

    if offenders:
        print("OFFENDERS:")
        for o in offenders:
            r0, r1, c0, c1 = o["extent"]
            print(f"  [{o['split']}] {o['patch_id']}  extent rows {r0}:{r1} cols {c0}:{c1}"
                  f"  unique_leaked_px={o['unique_px']:,} ({o['unique_frac']:.0%} of patch)"
                  f"  [sum-over-crops={o['total_px']:,}]")
            for tid, t_is_ov, px in sorted(o["hits"], key=lambda h: -h[2]):
                kind = "overlap-crop" if t_is_ov else "canonical"
                print(f"      <- train {tid} ({kind})  overlap_px={px:,}")
        canon_hits = sum(1 for o in offenders for _, iv, _ in o["hits"] if not iv)
        print(f"\n  (canonical train patches involved in any overlap: {canon_hits})")
        print()
        print("RESULT: LEAKAGE FOUND")
        return 1

    print("RESULT: CLEAN - no val/test patch intersects any train patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
