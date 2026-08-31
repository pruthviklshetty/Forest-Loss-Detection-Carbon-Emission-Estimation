# Phase 4 - plain U-Net vs Attention U-Net + MobileNetV2 (recorded)

Single-model pipeline paper: the plain U-Net is the pipeline segmenter. The Attention U-Net + MobileNetV2 was trained under an identical schedule and **did not improve on the plain U-Net**; it is kept as a recorded negative result. No statistical architecture comparison is made.

**Mean +/- sd across seeds [42, 43, 44]** (early stopping, patience 15; configs otherwise byte-identical). Per-seed values in `results/metrics/seed_runs.json`.

| Metric | plain U-Net (pipeline) | Attn U-Net + MNv2 (recorded) |
|---|---|---|
| **test IoU (strict, primary)** | 0.158 +/- 0.016 | 0.113 +/- 0.023 |
| test IoU (+/-3 px tolerance, secondary) | 0.248 +/- 0.018 | 0.199 +/- 0.037 |
| test Dice | 0.273 +/- 0.025 | 0.203 +/- 0.038 |
| test precision | 0.332 +/- 0.018 | 0.206 +/- 0.031 |
| test recall | 0.231 +/- 0.026 | 0.202 +/- 0.051 |
| best val Dice | 0.250 +/- 0.006 | 0.237 +/- 0.009 |

Per-seed strict test IoU: U-Net [0.1649, 0.1695, 0.1393], Attn [0.1281, 0.0865, 0.1249]. The seed sd (0.016-0.023) is large relative to the metric; every number is a 3-seed mean +/- sd, not a single run.

---

Single representative run below (U-Net results/checkpoints/baseline_unet_s43_best.pt, Attn results/checkpoints/attention_unet_s42_best.pt - the median-best-val-Dice seed of each). Identical splits and schedule; operating threshold tuned on val by max Dice; metrics on the held-out 18-patch test set.

| | Baseline U-Net | Attn U-Net + MNv2 | Delta |
|---|---|---|---|
| Params | 7,764,481 | 6,703,809 | -1,060,672 |
| Op. threshold | 0.92 | 0.88 | |
| iou | 0.1695 | 0.1281 | -0.0415 |
| tolerance_iou | 0.2617 | 0.2217 | -0.0400 |
| dice | 0.2899 | 0.2271 | -0.0629 |
| pixel_acc | 0.9946 | 0.9926 | -0.0020 |
| precision | 0.3426 | 0.2092 | -0.1334 |
| recall | 0.2513 | 0.2482 | -0.0031 |
| f1 | 0.2899 | 0.2271 | -0.0629 |

## vs. John & Zhang (2022), reported test numbers

| Their dataset | Attn U-Net IoU / F1 | U-Net IoU / F1 |
|---|---|---|
| RGB Amazon | 0.9516 / 0.9753 | 0.9473 / 0.9731 |
| 4-band Amazon | 0.9199 / 0.9581 | 0.8883 / 0.9399 |
| 4-band Atlantic | 0.9028 / 0.9550 | 0.8888 / 0.9522 |

This study (Wayanad, test): Attn U-Net IoU 0.1281 / F1 0.2271; U-Net IoU 0.1695 / F1 0.2899.

The order-of-magnitude gap is expected and is explained in `docs/refs/john_zhang_2022.md` and `docs/phase4_notes.md`: different task (bi-temporal change vs single-image segmentation), ~0.3% positive prevalence vs an abundant positive class, 30 m Hansen GFC labels vs hand-digitised polygons, and fragmented smallholder loss vs Amazon clear-cutting.
