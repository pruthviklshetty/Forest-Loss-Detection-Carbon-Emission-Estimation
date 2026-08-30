# Phase 4 - Standard U-Net vs Attention U-Net + MobileNetV2

Both models: identical splits, identical schedule (`configs/train_baseline.yaml` == `configs/train_attention.yaml` for data/optim/loss/seed), operating threshold tuned on val by max Dice, metrics on the held-out 18-patch test set.

| | Baseline U-Net | Attn U-Net + MNv2 | Delta |
|---|---|---|---|
| Params | 7,764,481 | 6,703,809 | -1,060,672 |
| Op. threshold | 0.92 | 0.94 | |
| iou | 0.1955 | 0.1677 | -0.0278 |
| dice | 0.3270 | 0.2872 | -0.0398 |
| pixel_acc | 0.9940 | 0.9946 | +0.0006 |
| precision | 0.3232 | 0.3400 | +0.0168 |
| recall | 0.3309 | 0.2486 | -0.0823 |
| f1 | 0.3270 | 0.2872 | -0.0398 |

## vs. John & Zhang (2022), reported test numbers

| Their dataset | Attn U-Net IoU / F1 | U-Net IoU / F1 |
|---|---|---|
| RGB Amazon | 0.9516 / 0.9753 | 0.9473 / 0.9731 |
| 4-band Amazon | 0.9199 / 0.9581 | 0.8883 / 0.9399 |
| 4-band Atlantic | 0.9028 / 0.9550 | 0.8888 / 0.9522 |

This study (Wayanad, test): Attn U-Net IoU 0.1677 / F1 0.2872; U-Net IoU 0.1955 / F1 0.3270.

The order-of-magnitude gap is expected and is explained in `docs/refs/john_zhang_2022.md` and `docs/phase4_notes.md`: different task (bi-temporal change vs single-image segmentation), ~0.3% positive prevalence vs an abundant positive class, 30 m Hansen GFC labels vs hand-digitised polygons, and fragmented smallholder loss vs Amazon clear-cutting.
