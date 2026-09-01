"""Aggregate the Phase 8 runs into results/metrics/phase8_seed_runs.json and
print the tables.

Pooled: plain U-Net on the pooled multi-region split, seeds 42/43/44
    results/metrics/p8_pooled_unet_s<seed>.json          (from src.eval.evaluate)
    results/metrics/p8_pooled_unet_s<seed>_history.json  (from src.train)

LORO: one fold per held-out region, 1 seed (42)
    results/metrics/p8_loro_<region>.json
    results/metrics/p8_loro_<region>_history.json

Strict "iou" is primary; "tolerance_iou" is the secondary +/-3 px metric.
Prints pooled mean +/- sd, and per-fold LORO with the transfer gap vs the
pooled test IoU.

    python scripts/aggregate_phase8.py
"""

from __future__ import annotations

import json
import pathlib
import statistics as st

REPO = pathlib.Path(__file__).resolve().parents[1]
MET = REPO / "results" / "metrics"

POOLED_SEEDS = [42, 43, 44]
LORO_REGIONS = ["wayanad", "kodagu", "nilgiris", "anamalai"]
WAYANAD_ONLY_POOLED_IOU = (0.158, 0.016)   # Phase 4 single-region baseline, 3 seeds


def _load(stem: str) -> dict:
    ev = json.loads((MET / f"{stem}.json").read_text())
    hi = json.loads((MET / f"{stem}_history.json").read_text())
    t = ev["test_at_operating_threshold"]
    return {
        "stem": stem,
        "stop_epoch": hi["epochs_run"],
        "best_epoch": hi["best_epoch"],
        "best_val_dice": hi["best_val_dice"],
        "early_stopped": hi["early_stopped"],
        "operating_threshold": ev["operating_threshold"],
        "n_test_patches": ev["n_patches"]["test"],
        "test_iou": t["iou"],
        "test_strict_iou": t.get("strict_iou", t["iou"]),
        "test_tolerance_iou": t.get("tolerance_iou"),
        "test_dice": t["dice"],
        "test_precision": t["precision"],
        "test_recall": t["recall"],
    }


def _ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    pooled_runs = [_load(f"p8_pooled_unet_s{s}") for s in POOLED_SEEDS]
    for r, s in zip(pooled_runs, POOLED_SEEDS):
        r["seed"] = s

    fields = ["test_strict_iou", "test_tolerance_iou", "test_dice",
              "test_precision", "test_recall", "best_val_dice"]
    pooled_summary = {}
    for f in fields:
        m, sd = _ms([r[f] for r in pooled_runs])
        pooled_summary[f] = {"mean": round(m, 4), "sd": round(sd, 4),
                             "values": [round(r[f], 4) for r in pooled_runs]}
    pooled_summary["stop_epoch"] = [r["stop_epoch"] for r in pooled_runs]
    pooled_summary["best_epoch"] = [r["best_epoch"] for r in pooled_runs]
    pooled_summary["operating_threshold"] = [r["operating_threshold"] for r in pooled_runs]

    pooled_iou_mean = pooled_summary["test_strict_iou"]["mean"]
    pooled_iou_sd = pooled_summary["test_strict_iou"]["sd"]

    loro_runs = []
    for reg in LORO_REGIONS:
        stem = f"p8_loro_{reg}"
        if not (MET / f"{stem}.json").exists():
            print(f"  (skip {stem}: not found)")
            continue
        r = _load(stem)
        r["test_region"] = reg
        r["transfer_gap_vs_pooled"] = round(pooled_iou_mean - r["test_strict_iou"], 4)
        loro_runs.append(r)

    out = {
        "pooled": {
            "seeds": POOLED_SEEDS,
            "early_stop_patience": 15,
            "runs": pooled_runs,
            "summary": pooled_summary,
        },
        "loro": {
            "seed": 42,
            "runs": loro_runs,
            "mean_strict_iou": round(_ms([r["test_strict_iou"] for r in loro_runs])[0], 4)
            if loro_runs else None,
            "sd_strict_iou": round(_ms([r["test_strict_iou"] for r in loro_runs])[1], 4)
            if len(loro_runs) > 1 else None,
        },
        "reference_wayanad_only_pooled_iou": {
            "mean": WAYANAD_ONLY_POOLED_IOU[0], "sd": WAYANAD_ONLY_POOLED_IOU[1],
            "note": "Phase 4 single-region (Wayanad) pooled-split test IoU, 3 seeds",
        },
        "did_more_data_help": {
            "pooled_iou_mean": pooled_iou_mean,
            "pooled_iou_sd": pooled_iou_sd,
            "delta_vs_wayanad_only": round(pooled_iou_mean - WAYANAD_ONLY_POOLED_IOU[0], 4),
            "within_seed_variance": abs(pooled_iou_mean - WAYANAD_ONLY_POOLED_IOU[0])
            <= (pooled_iou_sd + WAYANAD_ONLY_POOLED_IOU[1]),
        },
    }
    (MET / "phase8_seed_runs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nPOOLED multi-region, plain U-Net, per seed (test, operating threshold on val):\n")
    hdr = f"{'seed':>4s} {'stopE':>5s} {'bestE':>5s} {'thr':>5s} {'valDice':>8s} " \
          f"{'strictIoU':>9s} {'tolIoU':>7s} {'Dice':>7s} {'P':>6s} {'R':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in pooled_runs:
        print(f"{r['seed']:>4d} {r['stop_epoch']:>5d} {r['best_epoch']:>5d} "
              f"{r['operating_threshold']:>5.2f} {r['best_val_dice']:>8.4f} "
              f"{r['test_strict_iou']:>9.4f} {(r['test_tolerance_iou'] or 0):>7.4f} "
              f"{r['test_dice']:>7.4f} {r['test_precision']:>6.3f} {r['test_recall']:>6.3f}")

    def cell(k):
        a = pooled_summary[k]
        return f"{a['mean']:.3f} +/- {a['sd']:.3f}"

    print("\nPOOLED mean +/- sd (strict IoU primary):")
    print(f"  strict IoU   {cell('test_strict_iou')}")
    print(f"  tol IoU      {cell('test_tolerance_iou')}")
    print(f"  Dice         {cell('test_dice')}")
    print(f"  precision    {cell('test_precision')}")
    print(f"  recall       {cell('test_recall')}")
    print(f"  val Dice     {cell('best_val_dice')}")
    print(f"\n  vs Wayanad-only pooled IoU {WAYANAD_ONLY_POOLED_IOU[0]:.3f} "
          f"+/- {WAYANAD_ONLY_POOLED_IOU[1]:.3f}  ->  "
          f"delta {out['did_more_data_help']['delta_vs_wayanad_only']:+.3f} "
          f"({'within' if out['did_more_data_help']['within_seed_variance'] else 'outside'} "
          f"seed variance)")

    if loro_runs:
        print("\nLORO (leave-one-region-out), 1 seed each:\n")
        h2 = f"{'test region':12s} {'stopE':>5s} {'thr':>5s} {'strictIoU':>9s} " \
             f"{'tolIoU':>7s} {'Dice':>7s} {'P':>6s} {'R':>6s} {'gap vs pooled':>13s}"
        print(h2); print("-" * len(h2))
        for r in loro_runs:
            print(f"{r['test_region']:12s} {r['stop_epoch']:>5d} "
                  f"{r['operating_threshold']:>5.2f} {r['test_strict_iou']:>9.4f} "
                  f"{(r['test_tolerance_iou'] or 0):>7.4f} {r['test_dice']:>7.4f} "
                  f"{r['test_precision']:>6.3f} {r['test_recall']:>6.3f} "
                  f"{r['transfer_gap_vs_pooled']:>+13.4f}")
        print(f"\n  LORO mean strict IoU {out['loro']['mean_strict_iou']} "
              f"(pooled in-domain {pooled_iou_mean:.3f})")

    print(f"\n-> {MET / 'phase8_seed_runs.json'}")


if __name__ == "__main__":
    main()
