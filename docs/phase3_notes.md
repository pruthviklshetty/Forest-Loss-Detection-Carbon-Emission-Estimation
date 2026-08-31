# Phase 3 - Baseline: Standard U-Net

> **Numbers are the leak-free split with early stopping, reported as mean +/-
> sd over 3 seeds (42/43/44).** History: pre-leakage-audit (`c9947eb`), then
> post-audit single 80-epoch run (`82f6948`), then early stopping (`defa73f`),
> then this 3-seed protocol. `docs/phase7_notes.md` carries the full
> before/after chain.

## What was built

- [`src/models/unet.py`](../src/models/unet.py) - plain U-Net (double-conv,
  max-pool down, transpose-conv up, 4 skips). No attention, no pretrained
  encoder. `base_channels=32, depth=4` -> **7,764,481 parameters**.
- [`src/models/losses.py`](../src/models/losses.py) - masked `DiceBCELoss` +
  masked pixel metrics.
- [`src/preprocessing/dataset.py`](../src/preprocessing/dataset.py) -
  `PatchDataset`: z-score with train mean/std, zero masked pixels, 8 dihedral
  flip/rotation augmentations for train.
- [`src/train.py`](../src/train.py) - training loop, AMP, cosine LR,
  **early stopping** (`optim.early_stop_patience`, monitors val Dice, restores
  best checkpoint), `--seed` / `--experiment` overrides for multi-seed sweeps.
- [`src/eval/evaluate.py`](../src/eval/evaluate.py) - tunes operating threshold
  on val, reports test metrics, qualitative triptychs.
- [`scripts/aggregate_seeds.py`](../scripts/aggregate_seeds.py) - collates the
  per-seed runs into `results/metrics/seed_runs.json` + a mean/sd table.
- [`configs/train_baseline.yaml`](../configs/train_baseline.yaml) - schedule
  Phase 4 reuses verbatim; `seed` and `early_stop_patience` are config fields.

## Training protocol

80-epoch cosine `T_max`, Adam lr 3e-4, batch 8, AMP, grad-clip 1.0,
Dice + BCE `pos_weight` 40, early stopping patience 15. 3 seeds. RTX 3050;
each run 8-24 min (baseline) depending on stop epoch.

## Results - held-out test split (18 patches), mean +/- sd over 3 seeds

| Metric | value |
|---|---|
| **test IoU (strict, primary)** | **0.158 +/- 0.016** |
| test IoU (+/-3 px tolerance, secondary) | 0.248 +/- 0.018 |
| test Dice / F1 | **0.273 +/- 0.024** |
| test precision | 0.332 +/- 0.018 |
| test recall | 0.231 +/- 0.026 |
| best val Dice | 0.250 +/- 0.006 |
| operating threshold (per seed) | 0.92 / 0.92 / 0.92 |
| stop epoch / best epoch (per seed) | 23/8, 22/7, 16/1 |

Per-seed strict test IoU: **0.165 (s42), 0.170 (s43), 0.139 (s44)**;
tolerance IoU 0.255 / 0.262 / 0.227. Full values in
`results/metrics/seed_runs.json` and the per-seed `baseline_unet_s*.json`.

**Tolerance IoU** (secondary, never replaces strict): intersection counted
against the GFC ground truth dilated by one 30 m cell (7x7, +/-3 px at 10 m
GSD), strict undilated union - GFC's 30 m label boundary otherwise penalises a
prediction that is correct but offset by less than a GFC cell. Definition and
rationale in `src/eval/evaluate.py`.

**Carry-forward checkpoint** (used by Phases 5-7): the **median best-val-Dice
seed = 43** (val Dice 0.252). Selection is on validation only. Seed 43 test:
IoU 0.170, Dice 0.290, P 0.343, R 0.251, op threshold 0.92; test@0.5 still
collapses (IoU 0.065).

## Honest read

- **Pixel accuracy ~0.99 is trivial** - 99.6% of valid pixels are negative.
- The model recovers ~25% of GFC loss pixels at its operating threshold with
  ~33% precision - smooth blobs against fragmented 30 m GFC specks.
- **Best validation Dice lands early on every seed** (epochs 1, 7, 8) and never
  improves - early stopping confirms rather than fixes the overfitting. With
  261 train patches at ~0.3% positive prevalence there is little signal to fit
  before the model memorises.
- **Run-to-run seed sd (~0.016 on test IoU) is the size of the U-Net vs
  Attention gap** - see `docs/phase4_notes.md`; single-run numbers are
  unfalsifiable, hence mean +/- sd.

## Needed before Phase 4

Nothing external. Phase 4 = Attention U-Net + MobileNetV2, same schedule + 3
seeds, mean +/- sd comparison + interval-overlap test vs the baseline and vs
John & Zhang (2022).
