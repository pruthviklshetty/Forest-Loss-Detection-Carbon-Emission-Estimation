# Phase 3 - Baseline: Standard U-Net

> **Numbers below are the post-leakage-audit re-run** (see
> `docs/phase7_notes.md` section on the audit). The pre-audit figures
> (train 304 patches, best val Dice 0.317, test IoU 0.196 / Dice 0.327) are
> preserved in the before/after table there and in git history (commit
> `c9947eb`).

## What was built

- [`src/models/unet.py`](../src/models/unet.py) - plain U-Net (double-conv
  blocks, max-pool down, transpose-conv up, skip connections). No attention,
  no pretrained / MobileNetV2 encoder. `base_channels=32, depth=4` ->
  **7,764,481 parameters**. Input 8-band bi-temporal stack, output 1 logit map.
- [`src/models/losses.py`](../src/models/losses.py) - `DiceBCELoss`
  (soft-Dice + `pos_weight` BCE), plus masked pixel metrics. Every op honours
  the per-pixel `valid` mask.
- [`src/preprocessing/dataset.py`](../src/preprocessing/dataset.py) -
  `PatchDataset`: z-scores bands with the train mean/std, zeros masked pixels,
  8 dihedral flip/rotation augmentations for train (no photometric jitter).
- [`src/train.py`](../src/train.py) - training loop, AMP, cosine LR, per-epoch
  val threshold sweep, checkpoint by best val Dice.
- [`src/eval/evaluate.py`](../src/eval/evaluate.py) - tunes the operating
  threshold on val, reports test metrics, writes qualitative triptychs.
- [`configs/train_baseline.yaml`](../configs/train_baseline.yaml) - the schedule
  Phase 4 reuses verbatim for a fair comparison.

## Training run (real, post-audit)

| | |
|---|---|
| Data | 261 train (76 canonical + 185 footprint-constrained stride-128 overlap), 16 val, 18 test; `min_valid_frac >= 0.10` |
| Schedule | 80 epochs, Adam lr 3e-4 cosine->0, batch 8, AMP, grad-clip 1.0 |
| Loss | Dice + BCE, `pos_weight=40` |
| Hardware | RTX 3050 6 GB, ~3.0 GB used, **23.7 min** |
| Selection | best **val Dice 0.245 @ epoch 8** |
| Train loss | 1.74 (e1) -> 0.84 (e80) |
| Val Dice over run | bounces in ~[0.07, 0.25]; no sustained rise -> overfitting against a 16-patch val set |

## Test-set results (held-out, 18 patches)

Operating threshold **0.88**, selected by max val Dice over the `[0.10, 0.98]`
sweep. Unlike the pre-audit run, the result is **threshold-sensitive**: at 0.50
the model collapses to near-trivial precision.

| Metric | Test @ thr 0.88 | Test @ thr 0.50 |
|---|---|---|
| **IoU** | **0.161** | 0.077 |
| **Dice / F1** | **0.277** | 0.142 |
| **Pixel accuracy** | 0.9944 | 0.9637 |
| Precision | 0.318 | 0.079 |
| Recall | 0.246 | 0.685 |
| Confusion (px) | tp 1268 / fp 2722 / fn 3881 / tn 1,164,622 | - |

Files: `results/metrics/baseline_unet.json` (full sweep) and a copy at
`results/metrics/baseline.json`.
Figures: `results/figures/baseline_unet_training_curves.png`,
`results/figures/phase3_baseline_unet_examples.png`.

## Honest read

A **weak baseline**, and on the leak-free split it is weaker and less stable
than the pre-audit run suggested (test IoU 0.161 vs 0.196; collapses at
threshold 0.5).

- **Pixel accuracy ~0.99 is trivial** - 99.6% of valid pixels are negative.
  IoU / Dice are the metrics that matter.
- The model recovers only ~25% of GFC loss pixels at its operating threshold
  (recall 0.25) with ~32% precision, predicting smooth blobs against fragmented
  30 m GFC specks - see the qualitative figure.
- Validation Dice never sustains a rise over 80 epochs while training loss
  keeps falling: the model overfits 261 patches against a 16-patch val set.
- Drivers of the ceiling, all pre-registered in the Phase 2 report: tiny AOI /
  dataset, ~0.3% positive prevalence, a subtle bi-temporal signal (T and T+1
  composites look near-identical at loss sites), 30 m GFC labels on a 10 m
  grid, and a 16-patch val set that makes threshold choice noisy.

## Needed before Phase 4

Nothing external. Phase 4 = build the Attention U-Net + MobileNetV2 encoder,
train with `configs/train_baseline.yaml`'s schedule (new `experiment` name),
evaluate identically, and report the delta vs. these numbers and vs. John &
Zhang (2022).
