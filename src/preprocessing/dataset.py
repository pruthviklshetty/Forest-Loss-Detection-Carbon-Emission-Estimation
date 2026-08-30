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
    def __init__(self, split: str, augment: bool = False,
                 proc_dir: pathlib.Path = PROC, min_valid_frac: float = 0.0) -> None:
        self.split = split
        self.augment = augment
        self.dir = pathlib.Path(proc_dir) / "patches"
        norm = json.loads((pathlib.Path(proc_dir) / "norm_stats.json").read_text())
        self.mean = np.asarray(norm["mean"], np.float32)[:, None, None]
        self.std = np.asarray(norm["std"], np.float32)[:, None, None]

        with open(pathlib.Path(proc_dir) / "index.csv", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["split"] == split]
        if min_valid_frac > 0:
            rows = [r for r in rows if float(r["valid_frac"]) >= min_valid_frac]
        self.ids = [r["patch_id"] for r in rows]
        if not self.ids:
            raise SystemExit(f"no patches for split '{split}' in {proc_dir}")

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
