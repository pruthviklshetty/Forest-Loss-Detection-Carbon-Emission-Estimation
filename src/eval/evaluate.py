"""Phase 3/4 evaluation.

    python -m src.eval.evaluate --experiment baseline_unet

Loads results/checkpoints/<experiment>_best.pt, picks the operating threshold
by max Dice on the validation split, then reports IoU / Dice / pixel accuracy /
precision / recall / F1 on the held-out TEST split (at the tuned threshold and
at 0.5). Writes results/metrics/<experiment>.json and a qualitative figure.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
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

    model = build_model(cfg["model"]["name"],
                        in_channels=cfg["model"]["in_channels"],
                        classes=cfg["model"]["classes"],
                        base_channels=cfg["model"]["base_channels"],
                        depth=cfg["model"]["depth"]).to(device)
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

    result = {
        "experiment": args.experiment,
        "checkpoint": str(ckpt_path.relative_to(RESULTS.parent)),
        "checkpoint_epoch": ckpt["epoch"],
        "n_params": ckpt.get("n_params"),
        "device": device,
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"model": cfg["model"], "optim": cfg["optim"],
                   "loss": cfg["loss"], "data": cfg["data"], "seed": cfg["seed"]},
        "n_patches": {"val": len(va), "test": len(te)},
        "operating_threshold": op_t,
        "operating_threshold_selected_on": "max val Dice",
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
        print(f"  {k:10s} {tm[k]:.4f}")
    print(f"  tp={tm['tp']}  fp={tm['fp']}  fn={tm['fn']}  tn={tm['tn']}")
    print(f"\nmetrics -> {out}")
    print(f"figure  -> {fig_path}")


if __name__ == "__main__":
    main()
