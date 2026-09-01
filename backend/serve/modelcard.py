"""Assemble the model card the results page shows.

Every number here is read live from the Phase 8 result JSON at call time. If a
file or a field is missing, that entry is returned as ``None`` and the frontend
renders it as a pending/error state - it is never replaced with a placeholder
value.
"""

from __future__ import annotations

import json

from .config import (AREA_SUMMARY, CARBON_ESTIMATES, CHECKPOINT_STEM, EVAL_JSON,
                     PHASE8_SEED_RUNS, TRAINING_WINDOW)


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
    seed = _load(PHASE8_SEED_RUNS)
    ev = _load(EVAL_JSON)
    area = _load(AREA_SUMMARY)
    carbon = _load(CARBON_ESTIMATES)

    pooled = (seed or {}).get("pooled", {})
    pooled_sum = pooled.get("summary", {})
    loro = (seed or {}).get("loro", {})
    did_help = (seed or {}).get("did_more_data_help", {})

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

    # out-of-training-set (leave-one-region-out) - surfaced because it is lower
    loro_runs = loro.get("runs") or []
    loro_folds = [
        {
            "test_region": r.get("test_region"),
            "strict_iou": r.get("test_strict_iou"),
            "tolerance_iou": r.get("test_tolerance_iou"),
            "dice": r.get("test_dice"),
        }
        for r in loro_runs
    ] or None
    transfer = {
        "loro_mean_strict_iou": loro.get("mean_strict_iou"),
        "loro_sd_strict_iou": loro.get("sd_strict_iou"),
        "folds": loro_folds,
        "in_domain_strict_iou_mean": (in_domain["strict_iou"] or {}).get("mean"),
        "warning": "Measured performance on a Western Ghats region NOT in the "
                   "training set is materially lower than the in-domain figure "
                   "(leave-one-region-out). Treat any result for a custom bbox "
                   "or a non-training preset as an upper-bound-limited estimate.",
    }

    more_data = {
        "pooled_iou_mean": did_help.get("pooled_iou_mean"),
        "wayanad_only_iou": ((seed or {}).get("reference_wayanad_only_pooled_iou") or {}),
        "delta_vs_wayanad_only": did_help.get("delta_vs_wayanad_only"),
        "within_seed_variance": did_help.get("within_seed_variance"),
        "note": "Expanding from one region to four did not move the pooled test "
                "IoU beyond seed variance.",
    }

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
            "phase8_seed_runs": PHASE8_SEED_RUNS.exists(),
            "eval_json": EVAL_JSON.exists(),
            "area_summary": AREA_SUMMARY.exists(),
            "carbon_estimates": CARBON_ESTIMATES.exists(),
        },
    }
