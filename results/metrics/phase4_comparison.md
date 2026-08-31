# Phase 4 - Standard U-Net vs Attention U-Net + MobileNetV2

**Headline = mean +/- sd across seeds [42, 43, 44]** (early stopping, patience 15; configs otherwise byte-identical). Per-seed values in `results/metrics/seed_runs.json`.

| Metric | U-Net (baseline) | Attn U-Net + MNv2 |
|---|---|---|
| test IoU | 0.158 +/- 0.016 | 0.113 +/- 0.023 |
| test Dice | 0.273 +/- 0.024 | 0.203 +/- 0.038 |
| test precision | 0.332 +/- 0.018 | 0.206 +/- 0.031 |
| test recall | 0.231 +/- 0.026 | 0.202 +/- 0.051 |
| best val Dice | 0.250 +/- 0.006 | 0.237 +/- 0.009 |

Test-IoU values: U-Net [0.1649, 0.1696, 0.1396], Attn [0.1281, 0.0864, 0.1249]. Mean +/- 1 sd intervals **do NOT overlap** -> the U-Net > Attn difference is **supported** by this criterion (n=3 per group; a lenient bar - see docs/phase7_notes.md).

---

Single representative run below (U-Net results/checkpoints/baseline_unet_s43_best.pt, Attn results/checkpoints/attention_unet_s42_best.pt - the median-best-val-Dice seed of each). Identical splits and schedule; operating threshold tuned on val by max Dice; metrics on the held-out 18-patch test set.

| | Baseline U-Net | Attn U-Net + MNv2 | Delta |
|---|---|---|---|
| Params | 7,764,481 | 6,703,809 | -1,060,672 |
| Op. threshold | 0.92 | 0.88 | |
| iou | 0.1695 | 0.1280 | -0.0415 |
| dice | 0.2899 | 0.2270 | -0.0629 |
| pixel_acc | 0.9946 | 0.9926 | -0.0020 |
| precision | 0.3426 | 0.2093 | -0.1333 |
| recall | 0.2513 | 0.2480 | -0.0033 |
| f1 | 0.2899 | 0.2270 | -0.0629 |

## vs. John & Zhang (2022), reported test numbers

| Their dataset | Attn U-Net IoU / F1 | U-Net IoU / F1 |
|---|---|---|
| RGB Amazon | 0.9516 / 0.9753 | 0.9473 / 0.9731 |
| 4-band Amazon | 0.9199 / 0.9581 | 0.8883 / 0.9399 |
| 4-band Atlantic | 0.9028 / 0.9550 | 0.8888 / 0.9522 |

This study (Wayanad, test): Attn U-Net IoU 0.1280 / F1 0.2270; U-Net IoU 0.1695 / F1 0.2899.

The order-of-magnitude gap is expected and is explained in `docs/refs/john_zhang_2022.md` and `docs/phase4_notes.md`: different task (bi-temporal change vs single-image segmentation), ~0.3% positive prevalence vs an abundant positive class, 30 m Hansen GFC labels vs hand-digitised polygons, and fragmented smallholder loss vs Amazon clear-cutting.
