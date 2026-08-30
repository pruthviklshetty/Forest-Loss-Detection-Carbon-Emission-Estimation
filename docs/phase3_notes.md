# Phase 3 - Baseline: Standard U-Net

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
  Phase 4 will reuse verbatim for a fair comparison.

## Training run (real)

| | |
|---|---|
| Data | 304 train (76 canonical + 228 stride-128 overlap), 16 val, 18 test; `min_valid_frac >= 0.10` |
| Schedule | 80 epochs, Adam lr 3e-4 cosine->0, batch 8, AMP, grad-clip 1.0 |
| Loss | Dice + BCE, `pos_weight=40` |
| Hardware | RTX 3050 6 GB, ~3.0 GB used, **23.9 min** |
| Selection | best **val Dice 0.3165 @ epoch 54** |
| Train loss | 1.73 (e1) -> 0.71 (e80); val Dice plateaus ~0.30 from ~e50 |

## Test-set results (held-out, 18 patches)

Operating threshold **0.90**, selected by max val Dice. Results are stable to
threshold: test Dice is 0.324 at 0.5 and 0.331 at 0.90.

| Metric | Test @ thr 0.90 | Test @ thr 0.50 |
|---|---|---|
| **IoU** | **0.198** | 0.193 |
| **Dice / F1** | **0.331** | 0.324 |
| **Pixel accuracy** | **0.9938** | 0.988 |
| Precision | 0.316 | 0.253 |
| Recall | 0.348 | 0.449 |
| Confusion (px) | tp 1792 / fp 3887 / fn 3357 / tn 1,163,457 | - |

Files: `results/metrics/baseline_unet.json` (full sweep) and a copy at
`results/metrics/baseline.json` (the name used in the build brief).
Figures: `results/figures/baseline_unet_training_curves.png`,
`results/figures/phase3_baseline_unet_examples.png`.

## Honest read

This is a **weak but real** baseline, which is exactly its job - the Phase 4
Attention U-Net + MobileNetV2 has to beat these numbers under the identical
schedule.

- **Pixel accuracy 0.994 is trivial** - 99.6% of valid pixels are negative.
  IoU / Dice are the metrics that matter.
- The model finds *roughly where* change is (recall ~0.35) but not the exact
  GFC pixels (precision ~0.32), predicting smooth blobs against fragmented
  30 m GFC specks - see the qualitative figure. This caps IoU around 0.20.
- Drivers of the ceiling, all pre-registered in the Phase 2 report: tiny AOI
  / dataset, ~0.3% positive prevalence, a genuinely subtle bi-temporal signal
  (T and T+1 composites look near-identical at loss sites), 30 m GFC labels on
  a 10 m grid, and a 16-patch val set that makes threshold choice noisy.

## Needed before Phase 4

Nothing external. Phase 4 = build the Attention U-Net + MobileNetV2 encoder in
`src/models/`, train with `configs/train_baseline.yaml`'s schedule (new
`experiment` name), evaluate identically, and report the delta vs. these
numbers and vs. John & Zhang (2022).
