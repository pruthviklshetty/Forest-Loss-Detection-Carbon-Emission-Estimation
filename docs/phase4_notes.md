# Phase 4 - Attention U-Net + MobileNetV2

> **Numbers below are the post-leakage-audit re-run** (see the audit section in
> `docs/phase7_notes.md`). Pre-audit figures (best val Dice: baseline 0.317 /
> attention 0.323; test IoU: baseline 0.196 / attention 0.168; delta -0.028)
> are preserved in the before/after table there and in git history (`c9947eb`).

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
- Threshold sweep `[0.10, 0.98, 0.02]`; both models evaluated on it.

## Training run (real, identical schedule to Phase 3)

| | Baseline U-Net | Attn U-Net + MNv2 |
|---|---|---|
| Params | 7,764,481 | 6,703,809 |
| Epochs / schedule | 80, Adam 3e-4 cosine, batch 8, AMP, Dice+BCE pos_weight 40 | same |
| Train time (RTX 3050) | 23.7 min | 22.4 min |
| Final train loss | 0.838 | 1.019 |
| **Best val Dice** | 0.2453 @ e8 | 0.2458 @ e36 |

Validation Dice is a **statistical dead heat** (0.0005 apart) and neither
model shows a sustained learning curve on the leak-free split.

## Test-set comparison (held-out 18 patches, val-tuned threshold)

| Metric | Baseline U-Net | Attn U-Net + MNv2 | Delta (proposed - baseline) |
|---|---|---|---|
| Operating threshold | 0.88 | 0.78 | |
| **IoU** | **0.1611** | 0.0807 | **-0.0804** |
| **Dice / F1** | **0.2775** | 0.1493 | **-0.1282** |
| Pixel accuracy | 0.9944 | 0.9914 | -0.0029 |
| Precision | 0.3178 | 0.1325 | -0.1853 |
| Recall | 0.2463 | 0.1711 | -0.0752 |
| Confusion (px) | tp 1268 / fp 2722 / fn 3881 | tp 881 / fp 5769 / fn 4268 | |

Files: `results/metrics/attention_unet.json`,
`results/metrics/phase4_comparison.{json,md}`,
`results/figures/attention_unet_training_curves.png`,
`results/figures/phase4_compare_examples.png`.

## Honest read - the core result of this phase

**Under an identical training schedule, the Attention U-Net + MobileNetV2 is
clearly worse than the plain U-Net on the held-out test set** (IoU 0.081 vs
0.161, Dice 0.149 vs 0.278), losing on every metric except pixel accuracy. On
the leak-contaminated first run the two models looked close (delta IoU -0.028);
removing the leak roughly tripled the gap (delta IoU -0.080), i.e. the leak had
masked the attention model's poorer generalisation. Contributing factors, none
of them a bug:

1. **Pretrained RGB encoder vs an 8-band reflectance stack.** MobileNetV2's
   ImageNet features are natural-RGB priors; the input here is an 8-band
   z-scored reflectance stack, far from that distribution, and 261 augmented
   256x256 patches is not enough to re-fit the encoder. The attention model
   overfits harder (final train loss 1.02 with far more false positives on
   test: fp 5769 vs the baseline's 2722).
2. **Fair-comparison constraint.** John & Zhang tuned learning rate and epochs
   per model (Attn U-Net LR 5e-4 / 50-60 ep; U-Net LR 1e-4 / 20-30 ep). We
   deliberately hold the schedule identical, which gives a clean architecture
   ablation but does not let the attention model use its own optimum.
3. **Task hardness dominates.** With ~0.3% positive prevalence, coarse 30 m
   labels and a 16-patch val set, both models overfit; a skip-attention
   refinement plus a mismatched pretrained encoder has no room to help.

This is consistent with the project's stated positioning: the architecture is
**not** the contribution; the integrated raw-imagery -> hectares -> tCO2
pipeline and the carbon regression upgrade are.

## Model carried into Phase 5 - selection rationale

Validation marginally preferred the attention model (Dice 0.2458 vs 0.2453);
the held-out test set preferred the baseline decisively (IoU 0.161 vs 0.081).
**Both differences are within the noise of a 16-validation / 18-test-patch
split**, and choosing on the test set would be selection on the held-out data.
The plain U-Net is carried forward for reasons **independent of test
performance**:

- simpler architecture (no pretrained encoder, no attention gates);
- no pretrained-RGB-encoder distribution mismatch against the 8-band stack;
- fewer moving parts for the region-wide inference in Phase 5.

Both models' test numbers are reported side by side above and in
`results/metrics/phase4_comparison.md`.

## Comparison vs. John & Zhang (2022)

Their reported **test** numbers (`docs/refs/john_zhang_2022.md`):

| Dataset | Attn U-Net IoU / F1 | U-Net IoU / F1 |
|---|---|---|
| RGB Amazon | 0.9516 / 0.9753 | 0.9473 / 0.9731 |
| 4-band Amazon | 0.9199 / 0.9581 | 0.8883 / 0.9399 |
| 4-band Atlantic Forest | 0.9028 / 0.9550 | 0.8888 / 0.9522 |

This study (Wayanad, test): Attn U-Net IoU 0.081 / F1 0.149; U-Net IoU 0.161 /
F1 0.278 - far lower. That gap is expected and is **not** a like-for-like
failure, because the two studies solve different problems:

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
- **Their Attn-vs-U-Net delta is also small** (F1 +0.002 / +0.018 / +0.003), so
  even in their favourable setting the attention gate is a minor refinement,
  not a step change.

## Needed before Phase 5

Nothing external. Phase 5 = run the plain U-Net baseline over the study region,
isolate newly-deforested pixels, convert to hectares with the 10 m GSD, and
produce the deforestation map + hectares figure.
