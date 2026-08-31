# Phase 4 - Attention U-Net + MobileNetV2 (recorded negative result)

> **Scope note.** The paper is a single-model pipeline paper (plain U-Net).
> The Attention U-Net + MobileNetV2 was trained under an identical schedule,
> **did not improve on the plain U-Net**, and is kept in the repository as a
> recorded negative result - code, checkpoints and `seed_runs.json` entries
> are all retained. It is not a headline and there is no statistical
> architecture comparison. Numbers are leak-free split, early stopping,
> mean +/- sd over 3 seeds (42/43/44). Full history in `docs/phase7_notes.md`.

## What was built

- [`src/models/attention_unet.py`](../src/models/attention_unet.py) -
  `AttentionUNetMobileNetV2`: ImageNet-pretrained **MobileNetV2 encoder** from
  `segmentation_models.pytorch` + a hand-built decoder with an **Oktay additive
  attention gate on every skip** (4 gates). **6,703,809 parameters**.
- [`configs/train_attention.yaml`](../configs/train_attention.yaml) - differs
  from `train_baseline.yaml` only in the `model` block.
- [`src/eval/compare.py`](../src/eval/compare.py) - side-by-side table of the
  two models' per-seed metrics from `results/metrics/seed_runs.json`, plus a
  qualitative figure. Retained for the record; not a significance test.

## Result

| Metric (mean +/- sd, 3 seeds) | Plain U-Net | Attn U-Net + MNv2 |
|---|---|---|
| strict test IoU (primary) | **0.158 +/- 0.016** | 0.113 +/- 0.023 |
| tolerance test IoU (+/-3 px, secondary) | 0.248 +/- 0.018 | 0.199 +/- 0.037 |
| test Dice / F1 | 0.273 +/- 0.024 | 0.203 +/- 0.038 |
| test precision | 0.332 +/- 0.018 | 0.206 +/- 0.031 |
| test recall | 0.231 +/- 0.026 | 0.202 +/- 0.052 |
| best val Dice | 0.250 +/- 0.006 | 0.237 +/- 0.009 |
| best epoch (per seed) | 8, 7, 1 | 15, 35, 17 |

Per-seed strict test IoU: U-Net 0.165 / 0.170 / 0.139; Attn 0.128 / 0.087 /
0.125. Full per-seed values in `results/metrics/seed_runs.json` and
`results/metrics/attention_unet_s*.json`.

**The Attention U-Net + MobileNetV2 did not improve on the plain U-Net under
the identical schedule**, so the pipeline uses the plain U-Net. Likely reasons,
recorded for future work: the pretrained encoder's ImageNet-RGB features do not
transfer well to an 8-band z-scored reflectance stack, and 261 training patches
is too few to re-fit it; the attention model also overfits harder (higher
false-positive count on test). No claim is made that either architecture is
significantly better - the seed sd (Section 5.7 of `report.md`) is large
relative to the difference.

## Model carried into Phase 5 - selection rule

The carried-forward checkpoint is chosen **on validation only**: the U-Net seed
with **median best validation Dice** = **seed 43** (0.252; seeds 42/44 give
0.244/0.255). Test metrics are never used to choose. Seed 43's test scores
(IoU 0.170, Dice 0.290) are recorded for traceability, not as a selection
criterion. The plain U-Net (not the attention model) is the pipeline segmenter
for reasons independent of any test score: simpler architecture, no
pretrained-RGB-encoder mismatch against the 8-band stack, fewer moving parts
for region-wide inference.

## Comparison vs. John & Zhang (2022)

Their reported **test** numbers (`docs/refs/john_zhang_2022.md`): Attn U-Net
IoU 0.90-0.95, F1 0.955-0.977 across three datasets. This study (Wayanad, test,
3-seed mean): U-Net IoU 0.158 / F1 0.273; Attn U-Net IoU 0.113 / F1 0.203 - far
lower for both. Expected, not a like-for-like failure: different task
(bi-temporal change vs single-image segmentation), ~0.3% vs abundant positive
prevalence, 30 m GFC labels vs hand-digitised polygons, fragmented smallholder
loss vs Amazon clear-cutting.

## Needed before Phase 5

Nothing external. Phase 5 = run the carry-forward U-Net (seed 43) over the
study region, isolate newly-deforested pixels, convert to hectares, produce the
deforestation map + hectares figure.
