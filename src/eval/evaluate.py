"""Phase 3/4 evaluation.

    python -m src.eval.evaluate --experiment baseline_unet

Loads results/checkpoints/<experiment>_best.pt, picks the operating threshold
by max Dice on the validation split, then reports IoU / Dice / pixel accuracy /
precision / recall / F1 on the held-out TEST split (at the tuned threshold and
at 0.5). Writes results/metrics/<experiment>.json and a qualitative figure.

**Tolerance IoU (secondary).** Ground truth is Hansen GFC at 30 m; predictions
are 10 m, so every GFC loss pixel is a 3x3 block whose boundary is quantised to
30 m and strict pixel IoU structurally under-credits a prediction that is
correct but offset by up to a GFC cell. The tolerance IoU therefore counts the
INTERSECTION against the ground truth dilated by one 30 m cell (a 7x7 square,
i.e. +/-3 px at the 10 m GSD) while keeping the STRICT (undilated) UNION:

    tolerance_IoU = |pred AND dilate(gt, 7x7)| / |pred OR gt|      (both masked by `valid`)

It is <= 1 (numerator <= |pred| <= |union|), is always reported ALONGSIDE the
strict IoU, and never replaces it. Strict IoU stays the primary metric.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common import CKPT_DIR, RESULTS, get_device, load_yaml, seed_everything  # noqa: E402
from ..models.losses import metrics_from_confusion  # noqa: E402
from ..models.unet import build_model  # noqa: E402
from ..preprocessing.dataset import PatchDataset  # noqa: E402


@torch.no_grad()
def _confusions(model, loader, device, thresholds):
    conf = {t: np.zeros(4, dtype=np.int64) for t in thresholds}
    for img, label, valid, _ in loader:
        img = img.to(device, non_blocking=True)
        prob = torch.sigmoid(model(img).float()).cpu()
        m = valid.bool()
        p = prob[m]
        ta = label[m].bool()
        for t in thresholds:
            pred = p >= t
            conf[t] += ((pred & ta).sum().item(), (pred & ~ta).sum().item(),
                        (~pred & ta).sum().item(), (~pred & ~ta).sum().item())
    return {round(float(t), 3): metrics_from_confusion(*c.tolist()) for t, c in conf.items()}


_DILATE_PX = 3          # one 30 m Hansen GFC cell at the 10 m GSD
_DILATE_STRUCT = np.ones((2 * _DILATE_PX + 1, 2 * _DILATE_PX + 1), dtype=bool)


@torch.no_grad()
def _tolerance_iou(model, loader, device, thresholds):
    """Return {threshold: {strict_iou, tolerance_iou, tol_inter, strict_inter, union}}.

    Computed patch-by-patch in 2D so the ground truth can be spatially dilated
    before the intersection term; the union stays strict (undilated).
    """
    acc = {t: {"si": 0, "ti": 0, "u": 0} for t in thresholds}
    for img, label, valid, _ in loader:
        img = img.to(device, non_blocking=True)
        prob = torch.sigmoid(model(img).float()).cpu().numpy()      # (B,1,H,W)
        lab = label.numpy().astype(bool)                            # (B,1,H,W)
        val = valid.numpy().astype(bool)
        for b in range(prob.shape[0]):
            g = lab[b, 0] & val[b, 0]
            v = val[b, 0]
            g_dil = binary_dilation(g, structure=_DILATE_STRUCT) & v
            for t in thresholds:
                p = (prob[b, 0] >= t) & v
                acc[t]["si"] += int((p & g).sum())
                acc[t]["ti"] += int((p & g_dil).sum())
                acc[t]["u"] += int((p | g).sum())
    out = {}
    for t, a in acc.items():
        u = max(a["u"], 1)
        out[round(float(t), 3)] = {
            "strict_iou": a["si"] / u,
            "tolerance_iou": a["ti"] / u,
            "strict_inter": a["si"], "tol_inter": a["ti"], "union": a["u"],
        }
    return out


@torch.no_grad()
def _qualitative(model, ds, device, out_path, n=6):
    order = sorted(range(len(ds)), key=lambda i: -int(np.load(ds.dir / f"{ds.ids[i]}.npz")["label"].sum()))
    pick = order[:n]
    fig, ax = plt.subplots(len(pick), 3, figsize=(9, 3 * len(pick)))
    if len(pick) == 1:
        ax = ax[None, :]
    for r, i in enumerate(pick):
        img, label, valid, pid = ds[i]
        prob = torch.sigmoid(model(img[None].to(device)).float())[0, 0].cpu().numpy()
        raw = np.load(ds.dir / f"{ds.ids[i]}.npz")["img"]
        def st(a):
            lo, hi = np.nanpercentile(a, [2, 98]); return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        t1_fc = np.dstack([st(raw[6]), st(raw[5]), st(raw[4])])
        ax[r, 0].imshow(t1_fc); ax[r, 0].set_ylabel(pid, fontsize=8)
        ax[r, 1].imshow(label[0], cmap="Reds", vmin=0, vmax=1)
        ax[r, 2].imshow(prob, cmap="Reds", vmin=0, vmax=1)
        if r == 0:
            ax[r, 0].set_title("T+1 false colour")
            ax[r, 1].set_title("ground truth")
            ax[r, 2].set_title("predicted prob")
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    args = ap.parse_args()

    ckpt_path = CKPT_DIR / f"{args.experiment}_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing {ckpt_path}; train first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_yaml(args.config) if args.config else ckpt["config"]
    seed_everything(cfg["seed"])
    device = get_device()

    model_kw = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(cfg["model"]["name"], **model_kw).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dc = cfg["data"]
    va = PatchDataset("val", augment=False, proc_dir=dc["proc_dir"], min_valid_frac=dc["min_valid_frac"])
    te = PatchDataset("test", augment=False, proc_dir=dc["proc_dir"], min_valid_frac=dc["min_valid_frac"])
    vl = DataLoader(va, batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"])
    tl = DataLoader(te, batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"])

    s0, s1, ss = cfg["eval"]["threshold_sweep"]
    sweep = [round(x, 3) for x in np.arange(s0, s1 + 1e-9, ss)]

    val_by_t = _confusions(model, vl, device, sweep)
    op_t = max(val_by_t, key=lambda t: val_by_t[t]["dice"])
    test_by_t = _confusions(model, tl, device, sorted(set(sweep) | {0.5}))

    # tolerance IoU (secondary) at the operating threshold and at 0.5
    tol = _tolerance_iou(model, tl, device, sorted({op_t, 0.5}))
    for thr, block_key in ((op_t, "test_at_operating_threshold"), (0.5, "test_at_0.5")):
        tb = test_by_t[thr]
        tt = tol[round(float(thr), 3)]
        tb["strict_iou"] = round(tb["iou"], 6)
        tb["tolerance_iou"] = round(tt["tolerance_iou"], 6)
        tb["tolerance_iou_note"] = (
            "intersection vs GT dilated by one 30 m GFC cell (7x7, +/-3 px); "
            "strict undilated union; secondary metric, strict_iou is primary")

    result = {
        "experiment": args.experiment,
        "checkpoint": ckpt_path.relative_to(RESULTS.parent).as_posix(),
        "checkpoint_epoch": ckpt["epoch"],
        "n_params": ckpt.get("n_params"),
        "device": device,
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"model": cfg["model"], "optim": cfg["optim"],
                   "loss": cfg["loss"], "data": cfg["data"], "seed": cfg["seed"]},
        "n_patches": {"val": len(va), "test": len(te)},
        "operating_threshold": op_t,
        "operating_threshold_selected_on": "max val Dice",
        "tolerance_iou_dilation_px": _DILATE_PX,
        "val_at_operating_threshold": val_by_t[op_t],
        "test_at_operating_threshold": test_by_t[op_t],
        "test_at_0.5": test_by_t[0.5],
        "val_threshold_sweep": val_by_t,
    }
    out = RESULTS / "metrics" / f"{args.experiment}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig_path = RESULTS / "figures" / f"phase3_{args.experiment}_examples.png"
    _qualitative(model, te, device, fig_path)

    tm = test_by_t[op_t]
    print(f"\n=== {args.experiment}  (test, threshold {op_t}) ===")
    for k in ("iou", "dice", "pixel_acc", "precision", "recall", "f1"):
        print(f"  {k:14s} {tm[k]:.4f}")
    print(f"  strict_iou     {tm['strict_iou']:.4f}   (primary)")
    print(f"  tolerance_iou  {tm['tolerance_iou']:.4f}   (secondary, +/-3 px GT dilation)")
    print(f"  tp={tm['tp']}  fp={tm['fp']}  fn={tm['fn']}  tn={tm['tn']}")
    print(f"\nmetrics -> {out}")
    print(f"figure  -> {fig_path}")


if __name__ == "__main__":
    main()
