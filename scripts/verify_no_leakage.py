"""Data-leakage check for the patch splits (multi-region aware).

POOLED split: for every region, no train patch's pixel extent may intersect any
val or test patch's extent of the SAME region. (Different regions have disjoint
pixel grids, so cross-region rectangle intersection is meaningless and is not
checked - instead patch ids must carry their region and not collide.)

LORO folds (data/processed/loro.json): for every fold, (a) no train or val
patch may belong to the held-out test region, (b) train / val / test id sets
must be pairwise disjoint, (c) every test id must belong to the test region.

Reads data/processed/index.csv (columns patch_id, region, px_r0, px_c0, size,
is_overlap, pooled_split; falls back to `split` for a legacy single-region
index). Exit 0 if clean, 1 if any leak is found.

    python scripts/verify_no_leakage.py
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
_PROC = REPO / "data" / "processed"
INDEX = _PROC / "index.csv"
LORO = _PROC / "loro.json"


def _extent(row: dict):
    r0, c0, s = int(row["px_r0"]), int(row["px_c0"]), int(row["size"])
    return r0, r0 + s, c0, c0 + s


def _overlap_px(a, b):
    r0a, r1a, c0a, c1a = a
    r0b, r1b, c0b, c1b = b
    rov = max(0, min(r1a, r1b) - max(r0a, r0b))
    cov = max(0, min(c1a, c1b) - max(c0a, c0b))
    return rov * cov


def _split_of(row: dict) -> str:
    return row.get("pooled_split") or row["split"]


def check_pooled(rows: list[dict]) -> list[dict]:
    regions = sorted({r.get("region", "_single") for r in rows})
    print(f"POOLED split — regions: {regions}")
    offenders = []
    for rid in regions:
        rr = [r for r in rows if r.get("region", "_single") == rid]
        by = {"train": [], "val": [], "test": []}
        for r in rr:
            by[_split_of(r)].append(r)
        n_ov = sum(1 for r in by["train"] if int(r["is_overlap"]))
        print(f"  [{rid}] train {len(by['train'])} ({len(by['train']) - n_ov}+{n_ov}ov) "
              f"| val {len(by['val'])} | test {len(by['test'])}")
        train_ext = [(r["patch_id"], _extent(r)) for r in by["train"]]
        for s in ("val", "test"):
            for r in by[s]:
                ext = _extent(r)
                hits = [(tid, _overlap_px(ext, te)) for tid, te in train_ext
                        if _overlap_px(ext, te) > 0]
                if hits:
                    r0, r1, c0, c1 = ext
                    size = r1 - r0
                    seen = [[False] * size for _ in range(size)]
                    tmap = dict(train_ext)
                    for tid, _px in hits:
                        t = tmap[tid]
                        for rr_ in range(max(r0, t[0]) - r0, min(r1, t[1]) - r0):
                            for cc in range(max(c0, t[2]) - c0, min(c1, t[3]) - c0):
                                seen[rr_][cc] = True
                    uniq = sum(v for row in seen for v in row)
                    offenders.append({"region": rid, "split": s, "patch_id": r["patch_id"],
                                      "unique_px": uniq, "unique_frac": uniq / (size * size),
                                      "hits": hits})
    if offenders:
        print("\n  POOLED OFFENDERS:")
        for o in offenders:
            print(f"    [{o['region']}/{o['split']}] {o['patch_id']}  "
                  f"leaked {o['unique_px']:,} px ({o['unique_frac']:.0%}) from "
                  f"{len(o['hits'])} train crop(s)")
    else:
        print("  POOLED: clean — no same-region train/val or train/test overlap")
    return offenders


def check_loro(rows: list[dict]) -> list[str]:
    if not LORO.exists():
        print("\nLORO: data/processed/loro.json not found — skipped")
        return []
    region_of = {r["patch_id"]: r.get("region", "_single") for r in rows}
    folds = json.loads(LORO.read_text())["folds"]
    print(f"\nLORO — {len(folds)} folds")
    problems = []
    for f in folds:
        tr, va, te = (set(f["ids"]["train"]), set(f["ids"]["val"]), set(f["ids"]["test"]))
        trn = f["test_region"]
        bad_tr = [i for i in tr | va if region_of.get(i) == trn]
        bad_te = [i for i in te if region_of.get(i) != trn]
        ov_tv, ov_tt, ov_vt = tr & va, tr & te, va & te
        ok = not (bad_tr or bad_te or ov_tv or ov_tt or ov_vt)
        print(f"  test={trn:9s} train {len(tr)} / val {len(va)} / test {len(te)}  "
              f"{'OK' if ok else 'LEAK'}")
        if bad_tr:
            problems.append(f"{trn}: {len(bad_tr)} train/val ids belong to the test region")
        if bad_te:
            problems.append(f"{trn}: {len(bad_te)} test ids are not from the test region")
        if ov_tv or ov_tt or ov_vt:
            problems.append(f"{trn}: id-set overlap train/val={len(ov_tv)} "
                            f"train/test={len(ov_tt)} val/test={len(ov_vt)}")
    if not problems:
        print("  LORO: clean — every fold's train/val excludes the test region and "
              "id sets are disjoint")
    return problems


def main() -> int:
    global INDEX, LORO
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default=None,
                    help="check data/processed/<period>/ instead of data/processed/ "
                         "(e.g. 2021_2023 for Phase 10)")
    args = ap.parse_args()
    proc = _PROC / args.period if args.period else _PROC
    INDEX, LORO = proc / "index.csv", proc / "loro.json"
    print(f"checking {proc.relative_to(REPO).as_posix()}/")

    if not INDEX.exists():
        print(f"ERROR: {INDEX} not found", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(open(INDEX, encoding="utf-8")))

    ids = [r["patch_id"] for r in rows]
    if len(ids) != len(set(ids)):
        print("RESULT: LEAKAGE FOUND — duplicate patch ids in index.csv")
        return 1

    pooled_off = check_pooled(rows)
    loro_problems = check_loro(rows)

    print()
    if pooled_off or loro_problems:
        for p in loro_problems:
            print(f"  LORO problem: {p}")
        print("RESULT: LEAKAGE FOUND")
        return 1
    print("RESULT: CLEAN — pooled and LORO splits pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
