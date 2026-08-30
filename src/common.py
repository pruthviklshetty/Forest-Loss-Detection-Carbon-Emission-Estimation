"""Shared helpers: repo paths, config loading, seeding, device."""

from __future__ import annotations

import os
import pathlib
import random

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RESULTS = REPO / "results"
CKPT_DIR = RESULTS / "checkpoints"


def load_yaml(path: str | pathlib.Path) -> dict:
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = REPO / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
