"""PatchDataset - serves the Phase 2 patch tensors to the models.

Each item: img (8, P, P) float32 standardised with the train mean/std from
data/processed/norm_stats.json (masked pixels set to 0 after standardising),
label (1, P, P) float32 in {0,1}, valid (1, P, P) float32 in {0,1}.

Train augmentation is limited to the 8 dihedral flips/rotations (all
label-preserving for this task); no photometric jitter, to keep the bi-temporal
radiometry intact.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import PROC


class PatchDataset(Dataset):
    """split: 'train' | 'val' | 'test'.
    scheme: 'pooled' (default) reads the `pooled_split` column (or legacy
    `split`); 'loro' reads data/processed/loro.json and requires
    `loro_test_region`.
    """

    def __init__(self, split: str, augment: bool = False,
                 proc_dir: pathlib.Path = PROC, min_valid_frac: float = 0.0,
                 scheme: str = "pooled", loro_test_region: str | None = None) -> None:
        self.split = split
        self.augment = augment
        self.dir = pathlib.Path(proc_dir) / "patches"
        proc_dir = pathlib.Path(proc_dir)
        norm = json.loads((proc_dir / "norm_stats.json").read_text())
        self.mean = np.asarray(norm["mean"], np.float32)[:, None, None]
        self.std = np.asarray(norm["std"], np.float32)[:, None, None]

        rows = list(csv.DictReader(open(proc_dir / "index.csv", encoding="utf-8")))
        by_id = {r["patch_id"]: r for r in rows}

        if scheme == "loro":
            if not loro_test_region:
                raise SystemExit("scheme='loro' requires loro_test_region")
            folds = json.loads((proc_dir / "loro.json").read_text())["folds"]
            fold = next((f for f in folds if f["test_region"] == loro_test_region), None)
            if fold is None:
                raise SystemExit(f"no LORO fold for test region '{loro_test_region}'")
            wanted = set(fold["ids"][split])
            sel = [by_id[i] for i in wanted if i in by_id]
        else:
            def _sp(r):
                return r.get("pooled_split") or r["split"]
            sel = [r for r in rows if _sp(r) == split]

        if min_valid_frac > 0:
            sel = [r for r in sel if float(r["valid_frac"]) >= min_valid_frac]
        self.ids = [r["patch_id"] for r in sel]
        if not self.ids:
            raise SystemExit(
                f"no patches for split '{split}' (scheme={scheme}"
                f"{', region=' + loro_test_region if loro_test_region else ''}) in {proc_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def _augment(self, img, label, valid):
        if np.random.rand() < 0.5:
            img, label, valid = img[:, :, ::-1], label[:, ::-1], valid[:, ::-1]
        if np.random.rand() < 0.5:
            img, label, valid = img[:, ::-1, :], label[::-1, :], valid[::-1, :]
        k = np.random.randint(4)
        if k:
            img = np.rot90(img, k, axes=(1, 2))
            label = np.rot90(label, k)
            valid = np.rot90(valid, k)
        return np.ascontiguousarray(img), np.ascontiguousarray(label), np.ascontiguousarray(valid)

    def __getitem__(self, i: int):
        d = np.load(self.dir / f"{self.ids[i]}.npz")
        img = d["img"].astype(np.float32)
        label = d["label"].astype(np.float32)
        valid = d["valid"].astype(np.float32)

        img = (img - self.mean) / self.std
        img *= valid[None, :, :]                      # keep masked pixels at 0

        if self.augment:
            img, label, valid = self._augment(img, label, valid)

        return (
            torch.from_numpy(img),
            torch.from_numpy(label)[None, :, :],
            torch.from_numpy(valid)[None, :, :],
            self.ids[i],
        )
