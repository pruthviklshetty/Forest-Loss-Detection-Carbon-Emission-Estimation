"""Phase 3/4 training entry point.

    python -m src.train --config configs/train_baseline.yaml

Trains the model named in the config on the Phase 2 patches, selecting the
checkpoint by best validation Dice. Writes:
    results/checkpoints/<experiment>_best.pt
    results/metrics/<experiment>_history.json
    results/figures/<experiment>_training_curves.png
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

from .common import CKPT_DIR, RESULTS, get_device, load_yaml, seed_everything  # noqa: E402
from .models.losses import DiceBCELoss, metrics_from_confusion  # noqa: E402
from .models.unet import build_model  # noqa: E402
from .preprocessing.dataset import PatchDataset  # noqa: E402


def _loaders(dc: dict):
    tr = PatchDataset("train", augment=dc["augment"], proc_dir=dc["proc_dir"],
                      min_valid_frac=dc["min_valid_frac"])
    va = PatchDataset("val", augment=False, proc_dir=dc["proc_dir"],
                      min_valid_frac=dc["min_valid_frac"])
    tl = DataLoader(tr, batch_size=dc["batch_size"], shuffle=True,
                    num_workers=dc["num_workers"], drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=dc["batch_size"], shuffle=False,
                    num_workers=dc["num_workers"], pin_memory=True)
    return tr, va, tl, vl


@torch.no_grad()
def _validate(model, loader, device, thresholds) -> dict:
    model.eval()
    conf = {t: np.zeros(4, dtype=np.int64) for t in thresholds}   # tp,fp,fn,tn
    for img, label, valid, _ in loader:
        img = img.to(device, non_blocking=True)
        logits = model(img).float().cpu()
        m = valid.bool()
        lo = logits[m]
        ta = label[m].bool()
        prob = torch.sigmoid(lo)
        for t in thresholds:
            pred = prob >= t
            tp = (pred & ta).sum().item()
            fp = (pred & ~ta).sum().item()
            fn = (~pred & ta).sum().item()
            tn = (~pred & ~ta).sum().item()
            conf[t] += (tp, fp, fn, tn)
    out = {}
    for t, c in conf.items():
        out[round(float(t), 3)] = metrics_from_confusion(*c.tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    device = get_device()
    exp = cfg["experiment"]
    print(f"experiment: {exp} | device: {device}")

    dc = cfg["data"]
    tr, va, tl, vl = _loaders(dc)
    print(f"train patches: {len(tr)} | val patches: {len(va)} | steps/epoch: {len(tl)}")

    model = build_model(cfg["model"]["name"],
                        in_channels=cfg["model"]["in_channels"],
                        classes=cfg["model"]["classes"],
                        base_channels=cfg["model"]["base_channels"],
                        depth=cfg["model"]["depth"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    oc = cfg["optim"]
    opt = torch.optim.Adam(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=oc["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=oc["amp"] and device == "cuda")
    lossfn = DiceBCELoss(pos_weight=cfg["loss"]["pos_weight"],
                         weight_bce=cfg["loss"]["weight_bce"],
                         weight_dice=cfg["loss"]["weight_dice"])

    s0, s1, ss = cfg["eval"]["threshold_sweep"]
    sweep = [round(x, 3) for x in np.arange(s0, s1 + 1e-9, ss)]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    history = []
    best_dice, best_epoch = -1.0, -1
    t_start = time.time()

    for epoch in range(1, oc["epochs"] + 1):
        model.train()
        run = 0.0
        for img, label, valid, _ in tl:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=oc["amp"] and device == "cuda"):
                logits = model(img)
                loss = lossfn(logits, label, valid)
            scaler.scale(loss).backward()
            if oc.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            run += loss.item() * img.size(0)
        sched.step()
        train_loss = run / (len(tl) * dc["batch_size"])

        val = _validate(model, vl, device, sweep)
        best_t = max(val, key=lambda t: val[t]["dice"])
        v05 = val.get(0.5, val[best_t])
        rec = {"epoch": epoch, "train_loss": round(train_loss, 5),
               "lr": round(sched.get_last_lr()[0], 7),
               "val_dice_best": round(val[best_t]["dice"], 5),
               "val_dice_thr": best_t,
               "val_iou_at_best": round(val[best_t]["iou"], 5),
               "val_dice_0p5": round(v05["dice"], 5),
               "val_recall_at_best": round(val[best_t]["recall"], 5),
               "val_precision_at_best": round(val[best_t]["precision"], 5)}
        history.append(rec)
        print(f"  e{epoch:03d}  loss {train_loss:.4f}  "
              f"val Dice {rec['val_dice_best']:.4f}@{best_t}  "
              f"IoU {rec['val_iou_at_best']:.4f}  "
              f"P {rec['val_precision_at_best']:.3f} R {rec['val_recall_at_best']:.3f}")

        if val[best_t]["dice"] > best_dice:
            best_dice, best_epoch = val[best_t]["dice"], epoch
            torch.save({"model_state": model.state_dict(),
                        "config": cfg, "epoch": epoch,
                        "val_threshold": best_t,
                        "val_metrics": val[best_t],
                        "n_params": n_params},
                       CKPT_DIR / f"{exp}_best.pt")

    mins = (time.time() - t_start) / 60
    summary = {"experiment": exp, "best_epoch": best_epoch,
               "best_val_dice": round(best_dice, 5),
               "epochs": oc["epochs"], "train_minutes": round(mins, 1),
               "n_params": n_params, "device": device,
               "history": history}
    (RESULTS / "metrics" / f"{exp}_history.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history])
    ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["val_dice_best"] for h in history], label="val Dice (best thr)")
    ax[1].plot(ep, [h["val_iou_at_best"] for h in history], label="val IoU")
    ax[1].axvline(best_epoch, color="k", ls=":", lw=1, label=f"best e{best_epoch}")
    ax[1].set_title("validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / f"{exp}_training_curves.png", dpi=110)

    print(f"\nbest val Dice {best_dice:.4f} @ epoch {best_epoch}  ({mins:.1f} min)")
    print(f"checkpoint -> results/checkpoints/{exp}_best.pt")


if __name__ == "__main__":
    main()
