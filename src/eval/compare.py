"""Phase 4 - baseline vs proposed comparison.

Reads results/metrics/baseline_unet.json and results/metrics/attention_unet.json
(both written by src.eval.evaluate) and emits:
    results/metrics/phase4_comparison.json
    results/metrics/phase4_comparison.md
    results/figures/phase4_compare_examples.png   (test triptychs: GT | U-Net | Attn U-Net)

    python -m src.eval.compare
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common import CKPT_DIR, RESULTS, get_device, load_yaml  # noqa: E402
from ..models.unet import build_model  # noqa: E402
from ..preprocessing.dataset import PatchDataset  # noqa: E402

# John & Zhang (2022) reported TEST numbers - see docs/refs/john_zhang_2022.md
JZ2022 = {
    "RGB Amazon": {"attn": {"iou": 0.9516, "f1": 0.9753}, "unet": {"iou": 0.9473, "f1": 0.9731}},
    "4-band Amazon": {"attn": {"iou": 0.9199, "f1": 0.9581}, "unet": {"iou": 0.8883, "f1": 0.9399}},
    "4-band Atlantic": {"attn": {"iou": 0.9028, "f1": 0.9550}, "unet": {"iou": 0.8888, "f1": 0.9522}},
}


def _load(exp: str) -> dict:
    p = RESULTS / "metrics" / f"{exp}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}; run `python -m src.eval.evaluate --experiment {exp}` first")
    return json.loads(p.read_text())


def _load_model(exp: str, device: str):
    ckpt = torch.load(CKPT_DIR / f"{exp}_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    kw = {k: v for k, v in cfg["model"].items() if k != "name"}
    m = build_model(cfg["model"]["name"], **kw).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, ckpt.get("val_threshold", 0.5)


@torch.no_grad()
def _fig(base_exp: str, attn_exp: str, device: str, out, n: int = 6) -> None:
    mb, tb = _load_model(base_exp, device)
    ma, ta = _load_model(attn_exp, device)
    ds = PatchDataset("test", augment=False)
    order = sorted(range(len(ds)),
                   key=lambda i: -int(np.load(ds.dir / f"{ds.ids[i]}.npz")["label"].sum()))[:n]
    fig, ax = plt.subplots(len(order), 4, figsize=(12, 3 * len(order)))
    for r, i in enumerate(order):
        img, label, valid, pid = ds[i]
        x = img[None].to(device)
        pb = torch.sigmoid(mb(x))[0, 0].cpu().numpy()
        pa = torch.sigmoid(ma(x))[0, 0].cpu().numpy()
        raw = np.load(ds.dir / f"{ds.ids[i]}.npz")["img"]
        st = lambda a: np.clip((a - np.nanpercentile(a, 2)) /
                               (np.nanpercentile(a, 98) - np.nanpercentile(a, 2) + 1e-6), 0, 1)
        ax[r, 0].imshow(np.dstack([st(raw[6]), st(raw[5]), st(raw[4])]))
        ax[r, 0].set_ylabel(pid, fontsize=8)
        ax[r, 1].imshow(label[0], cmap="Reds", vmin=0, vmax=1)
        ax[r, 2].imshow(pb, cmap="Reds", vmin=0, vmax=1)
        ax[r, 3].imshow(pa, cmap="Reds", vmin=0, vmax=1)
        if r == 0:
            for c, t in enumerate(["T+1 false colour", "ground truth",
                                   "U-Net prob", "Attn U-Net prob"]):
                ax[r, c].set_title(t)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    base = _load("baseline_unet")
    attn = _load("attention_unet")
    keys = ("iou", "tolerance_iou", "dice", "pixel_acc", "precision", "recall", "f1")

    bt = base["test_at_operating_threshold"]
    at = attn["test_at_operating_threshold"]
    delta = {k: round(at.get(k, 0) - bt.get(k, 0), 4) for k in keys}

    comp = {
        "baseline": {"experiment": "baseline_unet",
                     "n_params": base["n_params"],
                     "operating_threshold": base["operating_threshold"],
                     "test": {k: round(bt[k], 4) for k in keys}},
        "proposed": {"experiment": "attention_unet",
                     "n_params": attn["n_params"],
                     "operating_threshold": attn["operating_threshold"],
                     "test": {k: round(at[k], 4) for k in keys}},
        "delta_proposed_minus_baseline": delta,
        "john_zhang_2022_reference": JZ2022,
    }
    (RESULTS / "metrics" / "phase4_comparison.json").write_text(
        json.dumps(comp, indent=2), encoding="utf-8")

    L = []
    A = L.append
    A("# Phase 4 - plain U-Net vs Attention U-Net + MobileNetV2 (recorded)\n")
    A("Single-model pipeline paper: the plain U-Net is the pipeline segmenter. "
      "The Attention U-Net + MobileNetV2 was trained under an identical schedule "
      "and **did not improve on the plain U-Net**; it is kept as a recorded "
      "negative result. No statistical architecture comparison is made.\n")

    # --- per-seed mean +/- sd, if the sweep has been run --------------------
    sr_path = RESULTS / "metrics" / "seed_runs.json"
    if sr_path.exists():
        sr = json.loads(sr_path.read_text())
        A(f"**Mean +/- sd across seeds {sr['seeds']}** (early stopping, patience "
          f"{sr['early_stop_patience']}; configs otherwise byte-identical). "
          f"Per-seed values in `results/metrics/seed_runs.json`.\n")
        A("| Metric | plain U-Net (pipeline) | Attn U-Net + MNv2 (recorded) |")
        A("|---|---|---|")
        sb, sa = sr["summary"]["baseline_unet"], sr["summary"]["attention_unet"]
        for k, lbl in (("test_iou", "**test IoU (strict, primary)**"),
                       ("test_tolerance_iou", "test IoU (+/-3 px tolerance, secondary)"),
                       ("test_dice", "test Dice"),
                       ("test_precision", "test precision"),
                       ("test_recall", "test recall"),
                       ("best_val_dice", "best val Dice")):
            if k not in sb:
                continue
            A(f"| {lbl} | {sb[k]['mean']:.3f} +/- {sb[k]['sd']:.3f} | "
              f"{sa[k]['mean']:.3f} +/- {sa[k]['sd']:.3f} |")
        vb, va = sb["test_iou"]["values"], sa["test_iou"]["values"]
        A(f"\nPer-seed strict test IoU: U-Net {vb}, Attn {va}. The seed sd "
          f"(0.016-0.023) is large relative to the metric; every number is a "
          f"3-seed mean +/- sd, not a single run.\n")

    A("---\n")
    A(f"Single representative run below (U-Net {base.get('checkpoint','')}, "
      f"Attn {attn.get('checkpoint','')} - the median-best-val-Dice seed of "
      f"each). Identical splits and schedule; operating threshold tuned on val "
      f"by max Dice; metrics on the held-out 18-patch test set.\n")
    A("| | Baseline U-Net | Attn U-Net + MNv2 | Delta |")
    A("|---|---|---|---|")
    A(f"| Params | {base['n_params']:,} | {attn['n_params']:,} | "
      f"{attn['n_params'] - base['n_params']:+,} |")
    A(f"| Op. threshold | {base['operating_threshold']} | {attn['operating_threshold']} | |")
    for k in keys:
        A(f"| {k} | {bt[k]:.4f} | {at[k]:.4f} | {delta[k]:+.4f} |")
    A("")
    A("## vs. John & Zhang (2022), reported test numbers\n")
    A("| Their dataset | Attn U-Net IoU / F1 | U-Net IoU / F1 |")
    A("|---|---|---|")
    for d, v in JZ2022.items():
        A(f"| {d} | {v['attn']['iou']:.4f} / {v['attn']['f1']:.4f} | "
          f"{v['unet']['iou']:.4f} / {v['unet']['f1']:.4f} |")
    A(f"\nThis study (Wayanad, test): Attn U-Net IoU {at['iou']:.4f} / F1 "
      f"{at['f1']:.4f}; U-Net IoU {bt['iou']:.4f} / F1 {bt['f1']:.4f}.\n")
    A("The order-of-magnitude gap is expected and is explained in "
      "`docs/refs/john_zhang_2022.md` and `docs/phase4_notes.md`: different task "
      "(bi-temporal change vs single-image segmentation), ~0.3% positive "
      "prevalence vs an abundant positive class, 30 m Hansen GFC labels vs "
      "hand-digitised polygons, and fragmented smallholder loss vs Amazon "
      "clear-cutting.\n")
    (RESULTS / "metrics" / "phase4_comparison.md").write_text("\n".join(L), encoding="utf-8")

    _fig("baseline_unet", "attention_unet", get_device(),
         RESULTS / "figures" / "phase4_compare_examples.png")

    print("baseline test :", {k: round(bt[k], 4) for k in keys})
    print("proposed test :", {k: round(at[k], 4) for k in keys})
    print("delta         :", delta)
    print("\n-> results/metrics/phase4_comparison.{json,md}")
    print("-> results/figures/phase4_compare_examples.png")


if __name__ == "__main__":
    main()
