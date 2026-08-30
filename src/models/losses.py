"""Losses and pixel metrics for binary forest-loss segmentation.

All operations honour a per-pixel `valid` mask (0 where the S2 stack had no
clear observation or the pixel is outside mapped land) so masked pixels never
contribute to the loss or the metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_valid(logits, target, valid):
    if valid is None:
        valid = torch.ones_like(target)
    m = valid.bool()
    return logits[m], target[m]


class DiceBCELoss(nn.Module):
    """weight_bce * BCEWithLogits(pos_weight) + weight_dice * softDice, masked."""

    def __init__(self, pos_weight: float = 50.0, weight_bce: float = 1.0,
                 weight_dice: float = 1.0, dice_smooth: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.w_bce = weight_bce
        self.w_dice = weight_dice
        self.smooth = dice_smooth

    def forward(self, logits, target, valid=None):
        lo, ta = _flatten_valid(logits, target, valid)
        if lo.numel() == 0:
            return logits.sum() * 0.0
        pw = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(lo, ta.float(), pos_weight=pw)
        prob = torch.sigmoid(lo)
        inter = (prob * ta).sum()
        denom = prob.sum() + ta.sum()
        dice = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        return self.w_bce * bce + self.w_dice * dice


@torch.no_grad()
def binary_metrics(logits, target, valid=None, threshold: float = 0.5) -> dict:
    lo, ta = _flatten_valid(logits, target, valid)
    if lo.numel() == 0:
        return {k: float("nan") for k in
                ("iou", "dice", "pixel_acc", "precision", "recall", "f1", "tp", "fp", "fn", "tn")}
    pred = (torch.sigmoid(lo) >= threshold)
    ta = ta.bool()
    tp = (pred & ta).sum().item()
    fp = (pred & ~ta).sum().item()
    fn = (~pred & ta).sum().item()
    tn = (~pred & ~ta).sum().item()
    eps = 1e-9
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return {"iou": iou, "dice": dice, "pixel_acc": acc, "precision": prec,
            "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def accumulate_confusion(logits, target, valid, threshold: float):
    """Return (tp, fp, fn, tn) ints for streaming aggregation over a loader."""
    m = binary_metrics(logits, target, valid, threshold)
    return m["tp"], m["fp"], m["fn"], m["tn"]


def metrics_from_confusion(tp, fp, fn, tn) -> dict:
    eps = 1e-9
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return {"iou": iou, "dice": dice, "pixel_acc": acc, "precision": prec,
            "recall": rec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
