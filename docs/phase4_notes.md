# Phase 4 - Attention U-Net + MobileNetV2

> **Numbers are the leak-free split with early stopping, mean +/- sd over 3
> seeds (42/43/44).** See `docs/phase7_notes.md` for the full pre-audit ->
> post-audit -> early-stopping -> multi-seed chain and the before/after tables.

## What was built

- [`src/models/attention_unet.py`](../src/models/attention_unet.py) -
  `AttentionUNetMobileNetV2`: ImageNet-pretrained **MobileNetV2 encoder** from
  `segmentation_models.pytorch` + a hand-built decoder with an **Oktay additive
  attention gate on every skip** (4 gates). **6,703,809 parameters**.
- [`configs/train_attention.yaml`](../configs/train_attention.yaml) - differs
  from `train_baseline.yaml` only in the `model` block.
- [`src/eval/compare.py`](../src/eval/compare.py) - now leads with the 3-seed
  mean +/- sd table and the test-IoU interval-overlap verdict from
  `results/metrics/seed_runs.json`; a single representative run (median
  val-Dice seed of each model) follows.

## Training run (identical schedule to Phase 3, 3 seeds, early stopping)

| | Baseline U-Net | Attn U-Net + MNv2 |
|---|---|---|
| Params | 7,764,481 | 6,703,809 |
| best val Dice (mean +/- sd) | 0.250 +/- 0.006 | 0.237 +/- 0.009 |
| best epoch (per seed) | 8, 7, 1 | 15, 35, 17 |
| stop epoch (per seed) | 23, 22, 16 | 30, 50, 32 |
| train time (per seed) | 8-12 min | 14-20 min |

## Test-set comparison (held-out 18 patches), mean +/- sd over 3 seeds

| Metric | Baseline U-Net | Attn U-Net + MNv2 | Delta (mean) |
|---|---|---|---|
| **test IoU (strict, primary)** | **0.158 +/- 0.016** | 0.113 +/- 0.023 | **-0.045** |
| test IoU (+/-3 px tolerance, secondary) | 0.248 +/- 0.018 | 0.199 +/- 0.037 | -0.049 |
| **test Dice / F1** | **0.273 +/- 0.024** | 0.203 +/- 0.038 | -0.070 |
| test precision | 0.332 +/- 0.018 | 0.206 +/- 0.031 | -0.126 |
| test recall | 0.231 +/- 0.026 | 0.202 +/- 0.052 | -0.029 |

Per-seed strict test IoU: U-Net **0.165 / 0.170 / 0.139**, Attn **0.128 / 0.087
/ 0.125**. Tolerance IoU (GT dilated by one 30 m GFC cell, +/-3 px; strict
union; secondary, never replaces strict): U-Net 0.255 / 0.262 / 0.227, Attn
0.222 / 0.156 / 0.218.

**Interval-overlap test (strict IoU).** Mean +/- 1 sd: U-Net [0.142, 0.174],
Attn [0.090, 0.136]. **The intervals do not overlap**, so the plain U-Net's
advantage is **supported** at that criterion (it also holds on tolerance IoU).
Caveat: n = 3 per group and +/-1 sd is a lenient bar; a Welch t-test on the six
strict test-IoU values gives t = 2.75, p ~ 0.06 - probably real, not
established at p < 0.05.

Files: `results/metrics/seed_runs.json`, `results/metrics/attention_unet_s*.json`,
`results/metrics/phase4_comparison.{json,md}`,
`results/figures/phase4_compare_examples.png`.

## Honest read - the core result of this phase

**The Attention U-Net + MobileNetV2 scores below the plain U-Net on the
held-out test set** (mean IoU 0.113 vs 0.158; Dice 0.203 vs 0.273), and the
+/-1 sd intervals separate them. It loses most on precision (0.206 vs 0.332):
on test it produces far more false positives. Contributing factors, none a bug:

1. **Pretrained RGB encoder vs an 8-band reflectance stack.** MobileNetV2's
   ImageNet features are natural-RGB priors; the input is an 8-band z-scored
   reflectance stack, and 261 augmented patches is not enough to re-fit the
   encoder. The attention model overfits harder and its test recall is also the
   least stable across seeds (+/- 0.052).
2. **Fair-comparison constraint.** John & Zhang tuned learning rate and epochs
   per model (Attn U-Net LR 5e-4 / 50-60 ep; U-Net LR 1e-4 / 20-30 ep). We hold
   the schedule identical for a clean ablation.
3. **Task hardness dominates.** ~0.3% prevalence, coarse 30 m labels, a
   16-patch val set on which best val Dice lands within a few epochs - a
   skip-attention refinement plus a mismatched pretrained encoder has no room
   to help.

Consistent with the project's positioning: the architecture is **not** the
contribution; the integrated pipeline and the carbon regression upgrade are.

## Model carried into Phase 5 - selection rationale

The carried-forward checkpoint is chosen **on validation only**: the U-Net seed
with **median best validation Dice** = **seed 43** (0.252; seeds 42/44 give
0.244/0.255). Test metrics are never used to choose. Seed 43's test scores (IoU
0.170, Dice 0.290) are recorded for traceability, not as a selection criterion.
Reasons the plain U-Net rather than the attention model carries forward, all
**independent of test performance**: simpler architecture, no pretrained-RGB-
encoder mismatch against the 8-band stack, fewer moving parts for region-wide
inference. Both models' full per-seed results are in Section 5.1 of `report.md`
and `results/metrics/seed_runs.json`.

## Comparison vs. John & Zhang (2022)

Their reported **test** numbers (`docs/refs/john_zhang_2022.md`):

| Dataset | Attn U-Net IoU / F1 | U-Net IoU / F1 |
|---|---|---|
| RGB Amazon | 0.9516 / 0.9753 | 0.9473 / 0.9731 |
| 4-band Amazon | 0.9199 / 0.9581 | 0.8883 / 0.9399 |
| 4-band Atlantic Forest | 0.9028 / 0.9550 | 0.8888 / 0.9522 |

This study (Wayanad, test, 3-seed mean): Attn U-Net IoU 0.113 / F1 0.203;
U-Net IoU 0.158 / F1 0.273 - far lower. Expected, not a like-for-like failure:
different task (bi-temporal change vs single-image segmentation), ~0.3% vs
abundant positive prevalence, 30 m GFC labels vs hand-digitised polygons, and
fragmented smallholder loss vs Amazon clear-cutting. Their Attn-vs-U-Net delta
is also small (F1 +0.002 to +0.018).

## Needed before Phase 5

Nothing external. Phase 5 = run the carry-forward U-Net (seed 43) over the
study region, isolate newly-deforested pixels, convert to hectares, produce the
deforestation map + hectares figure.
