"""Aggregate the Phase 10 (2021->2023) runs into
results/metrics/phase10_seed_runs.json and print the tables, side by side with
the committed 2019->2021 (Phase 8) pooled numbers.

Pooled: plain U-Net on the pooled 2021->2023 split, seeds 42/43/44
    results/metrics/p10_pooled_unet_s<seed>.json          (src.eval.evaluate)
    results/metrics/p10_pooled_unet_s<seed>_history.json  (src.train)
LORO (optional): results/metrics/p10_loro_<region>{,_history}.json

Strict "iou" is primary; "tolerance_iou" is the secondary +/-3 px metric.
Model selection is on validation Dice only.

    python scripts/aggregate_p10.py
"""

from __future__ import annotations

import json
import pathlib
import statistics as st

REPO = pathlib.Path(__file__).resolve().parents[1]
MET = REPO / "results" / "metrics"

POOLED_SEEDS = [42, 43, 44]
LORO_REGIONS = ["wayanad", "kodagu", "nilgiris", "anamalai"]
PHASE8_SEED_RUNS = MET / "phase8_seed_runs.json"


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
        "test_strict_iou": t.get("strict_iou", t["iou"]),
        "test_tolerance_iou": t.get("tolerance_iou"),
        "test_dice": t["dice"],
        "test_precision": t["precision"],
        "test_recall": t["recall"],
    }


def _ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)


def _summ(runs, keys):
    out = {}
    for k in keys:
        m, sd = _ms([r[k] for r in runs])
        out[k] = {"mean": round(m, 4), "sd": round(sd, 4),
                  "values": [round(r[k], 4) for r in runs]}
    return out


def _phase8_pooled():
    if not PHASE8_SEED_RUNS.exists():
        return None
    d = json.loads(PHASE8_SEED_RUNS.read_text())
    s = d.get("pooled", {}).get("summary", {})
    return {
        "strict_iou": s.get("test_strict_iou"),
        "tolerance_iou": s.get("test_tolerance_iou"),
        "dice": s.get("test_dice"),
        "precision": s.get("test_precision"),
        "recall": s.get("test_recall"),
        "best_val_dice": s.get("best_val_dice"),
        "operating_threshold": s.get("operating_threshold"),
    }


FIELDS = ["test_strict_iou", "test_tolerance_iou", "test_dice",
          "test_precision", "test_recall", "best_val_dice"]


def main() -> None:
    pooled_runs = [_load(f"p10_pooled_unet_s{s}") for s in POOLED_SEEDS]
    for r, s in zip(pooled_runs, POOLED_SEEDS):
        r["seed"] = s
    summary = _summ(pooled_runs, FIELDS)
    summary["stop_epoch"] = [r["stop_epoch"] for r in pooled_runs]
    summary["best_epoch"] = [r["best_epoch"] for r in pooled_runs]
    summary["operating_threshold"] = [r["operating_threshold"] for r in pooled_runs]

    loro_runs = []
    for reg in LORO_REGIONS:
        stem = f"p10_loro_{reg}"
        if (MET / f"{stem}.json").exists():
            r = _load(stem)
            r["test_region"] = reg
            loro_runs.append(r)

    p8 = _phase8_pooled()
    p10_iou = summary["test_strict_iou"]["mean"]
    p10_sd = summary["test_strict_iou"]["sd"]
    comparable = None
    if p8 and p8["strict_iou"]:
        d = p10_iou - p8["strict_iou"]["mean"]
        comparable = abs(d) <= (p10_sd + p8["strict_iou"]["sd"])

    out = {
        "period": "2021_2023",
        "pooled": {"seeds": POOLED_SEEDS, "early_stop_patience": 15,
                   "runs": pooled_runs, "summary": summary},
        "loro": {"runs": loro_runs,
                 "mean_strict_iou": round(_ms([r["test_strict_iou"] for r in loro_runs])[0], 4)
                 if loro_runs else None,
                 "sd_strict_iou": round(_ms([r["test_strict_iou"] for r in loro_runs])[1], 4)
                 if len(loro_runs) > 1 else None},
        "vs_2019_2021_pooled": p8,
        "comparable_within_seed_variance": comparable,
    }
    (MET / "phase10_seed_runs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nPOOLED 2021->2023, plain U-Net, per seed (test, operating threshold on val):\n")
    hdr = f"{'seed':>4s} {'stopE':>5s} {'bestE':>5s} {'thr':>5s} {'valDice':>8s} " \
          f"{'strictIoU':>9s} {'tolIoU':>7s} {'Dice':>7s} {'P':>6s} {'R':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in pooled_runs:
        print(f"{r['seed']:>4d} {r['stop_epoch']:>5d} {r['best_epoch']:>5d} "
              f"{r['operating_threshold']:>5.2f} {r['best_val_dice']:>8.4f} "
              f"{r['test_strict_iou']:>9.4f} {(r['test_tolerance_iou'] or 0):>7.4f} "
              f"{r['test_dice']:>7.4f} {r['test_precision']:>6.3f} {r['test_recall']:>6.3f}")

    def cell(k):
        a = summary[k]
        return f"{a['mean']:.3f} +/- {a['sd']:.3f}"

    print("\n                       2021->2023            2019->2021 (Phase 8)")
    rows = [("strict IoU", "test_strict_iou", "strict_iou"),
            ("tolerance IoU", "test_tolerance_iou", "tolerance_iou"),
            ("Dice", "test_dice", "dice"),
            ("precision", "test_precision", "precision"),
            ("recall", "test_recall", "recall"),
            ("best val Dice", "best_val_dice", "best_val_dice")]
    for label, k10, k8 in rows:
        p8cell = "-"
        if p8 and p8.get(k8):
            p8cell = f"{p8[k8]['mean']:.3f} +/- {p8[k8]['sd']:.3f}"
        print(f"  {label:18s}  {cell(k10):>18s}   {p8cell:>18s}")

    if comparable is not None:
        print(f"\n  strict IoU delta 2021->2023 minus 2019->2021: "
              f"{p10_iou - p8['strict_iou']['mean']:+.3f}  "
              f"({'within' if comparable else 'outside'} combined seed variance)")

    if loro_runs:
        print("\nLORO (2021->2023), 1 seed each:")
        for r in loro_runs:
            print(f"  test={r['test_region']:10s} strict IoU {r['test_strict_iou']:.4f} "
                  f"tol {r['test_tolerance_iou']:.4f} Dice {r['test_dice']:.4f}")
        print(f"  mean strict IoU {out['loro']['mean_strict_iou']}")
    else:
        print("\nLORO (2021->2023): not run. LORO for the 2019->2021 model was "
              "0.092 mean; it may not transfer to this period.")

    print(f"\n-> {MET / 'phase10_seed_runs.json'}")


if __name__ == "__main__":
    main()
