"""Assemble the model card the results page shows.

Every number is read live from the served checkpoint's result JSON at call
time. Analyses that were only ever measured on the 2019->2021 period
(leave-one-region-out, the more-data finding) are carried from that period's
aggregate and **labelled with the period they were measured on** - never
returned as a null with prose that implies a measurement for the served
checkpoint. A genuinely missing field is returned as ``None`` and the frontend
renders a pending state.
"""

from __future__ import annotations

import json

from .config import (AREA_SUMMARY, CARBON_ESTIMATES, CHECKPOINT_STEM, EVAL_JSON,
                     REFERENCE_PERIOD, REFERENCE_SEED_RUNS, SEED_RUNS,
                     SERVED_PERIOD, TRAINING_WINDOW)


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _interval(block, key):
    """Return {'mean':..,'sd':..} from a phase8_seed_runs summary block, or None."""
    if not isinstance(block, dict):
        return None
    v = block.get(key)
    if not isinstance(v, dict) or "mean" not in v:
        return None
    return {"mean": v.get("mean"), "sd": v.get("sd"), "values": v.get("values")}


def build_model_card() -> dict:
    seed = _load(SEED_RUNS)
    ref = _load(REFERENCE_SEED_RUNS) or {}
    ev = _load(EVAL_JSON)
    area = _load(AREA_SUMMARY)
    carbon = _load(CARBON_ESTIMATES)

    pooled = (seed or {}).get("pooled", {})
    pooled_sum = pooled.get("summary", {})
    loro = (seed or {}).get("loro", {})

    # in-domain (pooled multi-region test split), mean +/- sd over 3 seeds
    in_domain = {
        "strict_iou": _interval(pooled_sum, "test_strict_iou"),
        "tolerance_iou": _interval(pooled_sum, "test_tolerance_iou"),
        "dice": _interval(pooled_sum, "test_dice"),
        "precision": _interval(pooled_sum, "test_precision"),
        "recall": _interval(pooled_sum, "test_recall"),
        "seeds": pooled.get("seeds"),
        "note": "pooled 4-region held-out test split; mean +/- sd over the seeds "
                "listed. Strict IoU is primary; the +/-3 px tolerance IoU is "
                "secondary.",
    }

    # --- out-of-training-set (leave-one-region-out) ---
    # Prefer LORO measured for the SERVED checkpoint's period; if it was not
    # measured (e.g. 2021->2023), carry the 2019->2021 figure explicitly
    # labelled, never a null with prose that implies a measurement.
    def _folds(block):
        runs = (block or {}).get("runs") or []
        return [
            {"test_region": r.get("test_region"),
             "strict_iou": r.get("test_strict_iou"),
             "tolerance_iou": r.get("test_tolerance_iou"),
             "dice": r.get("test_dice")}
            for r in runs
        ] or None

    served_loro_runs = loro.get("runs") or []
    if served_loro_runs:
        transfer = {
            "measured": True,
            "loro_period": SERVED_PERIOD,
            "loro_mean_strict_iou": loro.get("mean_strict_iou"),
            "loro_sd_strict_iou": loro.get("sd_strict_iou"),
            "loro_in_domain_strict_iou_mean": (in_domain["strict_iou"] or {}).get("mean"),
            "folds": _folds(loro),
            "note": "Leave-one-region-out: performance on a Western Ghats region "
                    "not in the training set is materially lower than the "
                    "in-domain figure. Treat any result for a custom bbox or a "
                    "non-training preset as an upper-bound-limited estimate.",
        }
    else:
        ref_loro = ref.get("loro", {})
        ref_in = (((ref.get("pooled") or {}).get("summary") or {})
                  .get("test_strict_iou") or {})
        transfer = {
            "measured": False,
            "served_period": SERVED_PERIOD,
            "loro_period": REFERENCE_PERIOD,
            "loro_mean_strict_iou": ref_loro.get("mean_strict_iou"),
            "loro_sd_strict_iou": ref_loro.get("sd_strict_iou"),
            "loro_in_domain_strict_iou_mean": ref_in.get("mean"),
            "folds": _folds(ref_loro),
            "note": (f"Transfer to an unseen Western Ghats region was NOT "
                     f"re-measured for this {SERVED_PERIOD.replace('_', '-')} "
                     f"checkpoint. For the {REFERENCE_PERIOD.replace('_', '-')} "
                     f"model, leave-one-region-out mean strict IoU was "
                     f"{ref_loro.get('mean_strict_iou')} +/- "
                     f"{ref_loro.get('sd_strict_iou')} (about half its in-domain "
                     f"{ref_in.get('mean')}). Treat out-of-training-set results "
                     f"here as at least as limited."),
        }

    # --- "more data did not raise the ceiling" - a 2019->2021 analysis only ---
    ref_did = ref.get("did_more_data_help") or {}
    if ref_did:
        more_data = {
            "period": REFERENCE_PERIOD,
            "pooled_iou_mean": ref_did.get("pooled_iou_mean"),
            "wayanad_only_iou": ref.get("reference_wayanad_only_pooled_iou") or {},
            "delta_vs_wayanad_only": ref_did.get("delta_vs_wayanad_only"),
            "within_seed_variance": ref_did.get("within_seed_variance"),
            "note": (f"Measured on the {REFERENCE_PERIOD.replace('_', '-')} "
                     f"period: expanding from one region to four did not move the "
                     f"pooled test IoU beyond seed variance."),
        }
    else:
        more_data = None

    area_pooled = ((area or {}).get("pooled") or {}).get("test_only") or {}
    area_summary = {
        "pooled_test_pred_ha": area_pooled.get("pred_ha"),
        "pooled_test_gfc_ha": area_pooled.get("gt_ha"),
        "pooled_test_pred_over_gfc": area_pooled.get("pred_over_gt_ratio"),
        "strict_iou": area_pooled.get("strict_iou"),
        "tolerance_iou": area_pooled.get("tolerance_iou"),
        "note": "Region-wide predicted vs Hansen-GFC cleared area on the pooled "
                "held-out test blocks (pixel-level).",
    }

    carbon_reg = (carbon or {}).get("regression", {})
    carbon_summary = {
        "primary": carbon_reg.get("primary"),
        "exponential": carbon_reg.get("exponential"),
        "calibration_points": carbon_reg.get("calibration_points"),
        "co2_per_c": (carbon or {}).get("co2_per_c"),
        "scope": "aboveground carbon only, committed CO2 only; no belowground / "
                 "deadwood / litter / soil pools, no non-CO2 gases, no regrowth. "
                 "Calibrated to published Western Ghats field means, not "
                 "pixel-matched biomass.",
    }

    tw = TRAINING_WINDOW
    training_window = {
        "period": tw.get("period"),
        "t": tw.get("t"),
        "t1": tw.get("t1"),
        "gfc_lossyear": tw.get("gfc_lossyear"),
        "label": (f"{tw['t'][0][:7]} – {tw['t'][1][:7]}  vs  "
                  f"{tw['t1'][0][:7]} – {tw['t1'][1][:7]}"
                  if tw.get("t") and tw.get("t1") else None),
    }

    return {
        "checkpoint": str(EVAL_JSON.name).replace(".json", ""),
        "checkpoint_stem": CHECKPOINT_STEM,
        "training_window": training_window,
        "operating_threshold": (ev or {}).get("operating_threshold"),
        "in_domain": in_domain,
        "transfer_out_of_training_set": transfer,
        "more_data_finding": more_data,
        "area": area_summary,
        "carbon": carbon_summary,
        "label_resolution_note": "Ground truth is Hansen GFC at 30 m; predictions "
                                 "are 10 m. Strict IoU structurally under-credits "
                                 "sub-cell boundary offsets; the tolerance IoU is "
                                 "reported alongside for that reason.",
        "sources": {
            "seed_runs": SEED_RUNS.exists(),
            "reference_seed_runs": REFERENCE_SEED_RUNS.exists(),
            "eval_json": EVAL_JSON.exists(),
            "area_summary": AREA_SUMMARY.exists(),
            "carbon_estimates": CARBON_ESTIMATES.exists(),
        },
    }
