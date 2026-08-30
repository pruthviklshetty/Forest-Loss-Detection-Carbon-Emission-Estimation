# John, D. & Zhang, C. (2022) - reference notes

**An attention-based U-Net for detecting deforestation within satellite sensor
imagery.** *International Journal of Applied Earth Observation and
Geoinformation*, 107, 102685.
Open-access accepted manuscript: https://eprints.lancs.ac.uk/id/eprint/164622/1/JAG_accepted.pdf

## Setup (as reported in the paper)

- **Task:** single-image semantic segmentation of a deforestation mask. Each
  input is one (512, 512, 3) RGB or (512, 512, 4) image with a matching
  (512, 512, 1) deforestation mask. It is **not** bi-temporal T-vs-T+1 change
  differencing - the paper calls it a "cover change detection problem" but the
  model segments visible deforestation from a single image.
- **Imagery:** Sentinel-2. Three datasets: RGB Amazon, 4-band Amazon, 4-band
  Atlantic Forest (South America).
- **Data volume:** ~250 training images per dataset ("selected 250 training
  images due to memory limitations"), 512x512 tiles.
- **Labels:** digitised deforestation polygons on the same imagery; the
  positive (deforestation) class is an abundant fraction of each scene -
  consistent with achievable F1 ~ 0.95.
- **Protocol:** learning rate and epoch count tuned **per model** for max
  validation accuracy (their Table 1: Attention U-Net LR 5e-4, 50/60 epochs;
  plain U-Net LR 1e-4, 30/20 epochs). Not a fixed shared schedule.
- Metric: weighted Precision / Recall / F1 and Jaccard (IoU).

## Reported test-set numbers

| Dataset | Model | IoU (test) | F1 (test) |
|---|---|---|---|
| RGB Amazon (Table 2)      | Attention U-Net | 0.9516 | 0.9753 |
| RGB Amazon                | U-Net           | 0.9473 | 0.9731 |
| 4-band Amazon (Table 3)   | Attention U-Net | 0.9199 | 0.9581 |
| 4-band Amazon             | U-Net           | 0.8883 | 0.9399 |
| 4-band Atlantic (Table 4) | Attention U-Net | 0.9028 | 0.9550 |
| 4-band Atlantic           | U-Net           | 0.8888 | 0.9522 |

Attention U-Net vs plain U-Net (test): **F1 +0.0022 / +0.0182 / +0.0028**,
**IoU +0.0043 / +0.0316 / +0.0140** across the three datasets - small but
consistent gains.

## Why our numbers are far lower (honest comparison points for Phase 4/7)

1. **Different task.** Ours detects *new* loss between two dated composites;
   the model must find what changed, not what looks cleared. Theirs segments
   already-visible deforestation from one image.
2. **Label prevalence.** Their positive class is a large share of pixels;
   ours is ~0.3% (Hansen GFC 30 m annual loss rasterised onto a 10 m grid).
   Rare-class F1/IoU is bounded far below 0.9 before any modelling.
3. **Label source.** Theirs: hand-digitised polygons on the same pixels.
   Ours: Hansen GFC 30 m annual product - spatially coarse, temporally
   quantised, imperfectly aligned to 10 m Sentinel-2.
4. **Region.** Amazon / Atlantic Forest large clear-cut clearing vs. Western
   Ghats fragmented smallholder / plantation loss (fainter, smaller objects).
5. **Comparison protocol.** They tuned hyper-parameters per model; we hold the
   schedule byte-identical between U-Net and Attention U-Net, so our delta is a
   cleaner architecture ablation but our absolute numbers are not per-model
   tuned.

The takeaway to state plainly: this project's contribution is the integrated,
validated raw-imagery -> hectares -> tCO2 pipeline plus the binned->regression
carbon upgrade, not beating John & Zhang's segmentation F1.
