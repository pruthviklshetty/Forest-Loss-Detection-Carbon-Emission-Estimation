"""Aggregate the multi-seed training runs into results/metrics/seed_runs.json
and print a mean +/- sd table.

Expects, for each model stem in {baseline_unet, attention_unet} and each seed
in SEEDS:
    results/metrics/<stem>_s<seed>.json          (from src.eval.evaluate)
    results/metrics/<stem>_s<seed>_history.json  (from src.train)

    python scripts/aggregate_seeds.py
"""

from __future__ import annotations

import json
import pathlib
import statistics as st

REPO = pathlib.Path(__file__).resolve().parents[1]
MET = REPO / "results" / "metrics"

SEEDS = [42, 43, 44]
MODELS = {"baseline_unet": "U-Net (baseline)",
          "attention_unet": "Attn U-Net + MNv2"}
FIELDS = ["iou", "dice", "precision", "recall"]


def _load(stem: str, seed: int) -> dict:
    ev = json.loads((MET / f"{stem}_s{seed}.json").read_text())
    hi = json.loads((MET / f"{stem}_s{seed}_history.json").read_text())
    t = ev["test_at_operating_threshold"]
    return {
        "seed": seed,
        "stop_epoch": hi["epochs_run"],
        "best_epoch": hi["best_epoch"],
        "best_val_dice": hi["best_val_dice"],
        "early_stopped": hi["early_stopped"],
        "operating_threshold": ev["operating_threshold"],
        "test_iou": t["iou"], "test_dice": t["dice"],
        "test_precision": t["precision"], "test_recall": t["recall"],
        "test_pixel_acc": t["pixel_acc"],
    }


def _ms(xs: list[float]) -> tuple[float, float]:
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    out = {"seeds": SEEDS, "early_stop_patience": 15, "runs": {}, "summary": {}}
    per_model_iou: dict[str, list[float]] = {}

    for stem in MODELS:
        runs = [_load(stem, s) for s in SEEDS]
        out["runs"][stem] = runs
        agg = {}
        for f in FIELDS + ["best_val_dice"]:
            key = f if f == "best_val_dice" else f"test_{f}"
            vals = [r[key] for r in runs]
            m, sd = _ms(vals)
            agg[key] = {"mean": round(m, 4), "sd": round(sd, 4),
                        "values": [round(v, 4) for v in vals]}
        agg["stop_epoch"] = [r["stop_epoch"] for r in runs]
        agg["best_epoch"] = [r["best_epoch"] for r in runs]
        agg["operating_threshold"] = [r["operating_threshold"] for r in runs]
        out["summary"][stem] = agg
        per_model_iou[stem] = [r["test_iou"] for r in runs]

    (MET / "seed_runs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Per-seed test metrics (operating threshold tuned on val):\n")
    hdr = f"{'model':22s} {'seed':>4s} {'stopE':>5s} {'bestE':>5s} {'thr':>5s} " \
          f"{'valDice':>8s} {'IoU':>7s} {'Dice':>7s} {'P':>6s} {'R':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for stem, label in MODELS.items():
        for r in out["runs"][stem]:
            print(f"{label:22s} {r['seed']:>4d} {r['stop_epoch']:>5d} "
                  f"{r['best_epoch']:>5d} {r['operating_threshold']:>5.2f} "
                  f"{r['best_val_dice']:>8.4f} {r['test_iou']:>7.4f} "
                  f"{r['test_dice']:>7.4f} {r['test_precision']:>6.3f} "
                  f"{r['test_recall']:>6.3f}")
        print()

    print("Mean +/- sd across seeds:\n")
    print(f"{'model':22s} {'IoU':>16s} {'Dice':>16s} {'P':>16s} {'R':>16s} {'valDice':>16s}")
    for stem, label in MODELS.items():
        a = out["summary"][stem]
        def cell(k):
            return f"{a[k]['mean']:.3f} +/- {a[k]['sd']:.3f}"
        print(f"{label:22s} {cell('test_iou'):>16s} {cell('test_dice'):>16s} "
              f"{cell('test_precision'):>16s} {cell('test_recall'):>16s} "
              f"{cell('best_val_dice'):>16s}")

    print("\nTest-IoU interval overlap (mean +/- 1 sd):")
    lo, hi = {}, {}
    for stem in MODELS:
        m, sd = _ms(per_model_iou[stem])
        lo[stem], hi[stem] = m - sd, m + sd
        print(f"  {MODELS[stem]:22s} [{lo[stem]:.4f}, {hi[stem]:.4f}]  "
              f"(values {[round(v,4) for v in per_model_iou[stem]]})")
    a, b = list(MODELS)
    overlap = not (hi[a] < lo[b] or hi[b] < lo[a])
    verdict = ("OVERLAP -> difference NOT supported at +/-1 sd"
               if overlap else
               "NO OVERLAP -> difference supported at +/-1 sd")
    out["summary"]["test_iou_intervals_overlap"] = overlap
    (MET / "seed_runs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  => {verdict}")
    print(f"\n-> {MET / 'seed_runs.json'}")


if __name__ == "__main__":
    main()
