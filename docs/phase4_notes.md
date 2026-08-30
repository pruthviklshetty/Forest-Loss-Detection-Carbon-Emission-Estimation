# Phase 4 - Attention U-Net + MobileNetV2

## What was built

- [`src/models/attention_unet.py`](../src/models/attention_unet.py) -
  `AttentionUNetMobileNetV2`: an ImageNet-pretrained **MobileNetV2 encoder**
  from `segmentation_models.pytorch` (not hand-rolled) + a hand-built U-Net
  decoder in which **every skip connection passes through an Oktay-style
  additive attention gate** (gating signal = the upsampled coarser decoder
  feature). 4 gates, one per real skip. **6,703,809 parameters** (~1.06 M
  fewer than the baseline U-Net).
- [`configs/train_attention.yaml`](../configs/train_attention.yaml) - data,
  schedule, loss, seed **byte-identical** to `train_baseline.yaml`; only the
  architecture differs.
- [`src/eval/compare.py`](../src/eval/compare.py) - baseline-vs-proposed table,
  delta, John & Zhang (2022) reference block, side-by-side qualitative figure.
- Threshold sweep widened to `[0.10, 0.98, 0.02]` and **both** models
  re-evaluated on it, so neither operating point is clipped.

## Training run (real, identical schedule to Phase 3)

| | Baseline U-Net | Attn U-Net + MNv2 |
|---|---|---|
| Params | 7,764,481 | 6,703,809 |
| Epochs / schedule | 80, Adam 3e-4 cosine, batch 8, AMP, Dice+BCE pos_weight 40 | same |
| Train time (RTX 3050) | 23.9 min | 24.8 min |
| Final train loss | 0.714 | 0.966 |
| **Best val Dice** | 0.3165 @ e54 | **0.3232 @ e63** |

## Test-set comparison (held-out 18 patches, val-tuned threshold)

| Metric | Baseline U-Net | Attn U-Net + MNv2 | Delta (proposed - baseline) |
|---|---|---|---|
| Operating threshold | 0.92 | 0.94 | |
| **IoU** | **0.1955** | 0.1677 | **-0.0278** |
| **Dice / F1** | **0.3270** | 0.2872 | **-0.0398** |
| Pixel accuracy | 0.9940 | 0.9946 | +0.0006 |
| Precision | 0.3232 | 0.3400 | +0.0168 |
| Recall | 0.3309 | 0.2486 | -0.0823 |

Files: `results/metrics/attention_unet.json`,
`results/metrics/phase4_comparison.{json,md}`,
`results/figures/attention_unet_training_curves.png`,
`results/figures/phase4_compare_examples.png`.

## Honest read - the core result of this phase

**Under an identical training schedule, the Attention U-Net + MobileNetV2 did
not improve on the plain U-Net baseline.** It is marginally ahead on
validation (Dice +0.007) but behind on test (IoU -0.028, Dice -0.040), with a
precision/recall trade: it is slightly more precise (+0.017) and notably less
sensitive (recall -0.082), predicting fewer, higher-confidence blobs.

The val and test verdicts disagree, and both splits have only 16 / 18 patches,
so the honest conclusion is **no measurable architecture benefit in this
setup** rather than a real regression. Contributing factors, none of them a bug:

1. **Tiny dataset vs a pretrained RGB encoder.** MobileNetV2's ImageNet
   features are natural-RGB priors; the input here is an 8-band z-scored
   reflectance stack, far from that distribution, and 304 augmented 256x256
   patches is not enough to re-fit the encoder.
2. **Fair-comparison constraint.** John & Zhang tuned learning rate and epochs
   per model (Attn U-Net LR 5e-4 / 50-60 ep; U-Net LR 1e-4 / 20-30 ep). We
   deliberately hold the schedule identical, which gives a clean architecture
   ablation but does not let the attention model use its own optimum.
3. **Task hardness dominates.** With ~0.3% positive prevalence and coarse 30 m
   labels, both models sit near an IoU ceiling around 0.2 (see Phase 3 notes),
   where a skip-attention refinement has little room to help.

This is consistent with the project's stated positioning: the architecture is
**not** the contribution; the integrated raw-imagery -> hectares -> tCO2
pipeline and the carbon regression upgrade are.

## Comparison vs. John & Zhang (2022)

Their reported **test** numbers (`docs/refs/john_zhang_2022.md`):

| Dataset | Attn U-Net IoU / F1 | U-Net IoU / F1 |
|---|---|---|
| RGB Amazon | 0.9516 / 0.9753 | 0.9473 / 0.9731 |
| 4-band Amazon | 0.9199 / 0.9581 | 0.8883 / 0.9399 |
| 4-band Atlantic Forest | 0.9028 / 0.9550 | 0.8888 / 0.9522 |

This study (Wayanad, test): Attn U-Net IoU 0.168 / F1 0.287; U-Net IoU 0.196 /
F1 0.327 - roughly a factor of 3-5 lower on IoU/F1. That gap is expected and is
**not** a like-for-like failure, because the two studies solve different
problems:

- **Task.** They segment a deforestation mask from a *single* image where the
  clearing is already visible; we detect *new* loss between two dated
  composites (the model must localise change, not cleared-looking land).
- **Positive prevalence.** Their deforestation class is an abundant fraction of
  every scene (F1 ~ 0.95 is reachable); ours is ~0.3% of valid pixels.
- **Labels.** Hand-digitised polygons on the exact pixels vs. Hansen GFC 30 m
  annual loss rasterised onto a 10 m grid (coarser, temporally quantised,
  imperfectly aligned).
- **Landscape.** Amazon / Atlantic Forest large clear-cuts vs. Western Ghats
  fragmented smallholder and plantation loss (smaller, fainter objects).
- **Their Attn-vs-U-Net delta is also small** (F1 +0.002 / +0.018 / +0.003),
  so even in their favourable setting the attention gate is a minor refinement,
  not a step change - consistent with what we see here.

## Needed before Phase 5

Nothing external. Phase 5 = run the better model (the plain U-Net baseline, by
test IoU/Dice) over the study region, isolate newly-deforested pixels, convert
to hectares with the 10 m GSD, and produce the deforestation map + hectares
figure.
