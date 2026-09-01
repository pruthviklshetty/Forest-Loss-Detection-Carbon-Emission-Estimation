"""Period-scoped data paths.

Phase 1-8 wrote to `data/raw/<region>/`, `data/masks/<region>/`,
`data/processed/`. Phase 10 adds a second temporal period (2021->2023) whose
artifacts must sit beside, not overwrite, the first. A config carrying
`period_id: "<id>"` routes every data path under `data/<kind>/<id>/`; a config
without it keeps the legacy (2019->2021) layout untouched.

    from .paths import raw_dir, masks_dir, proc_dir
    raw_dir(cfg, "wayanad")     # data/raw/2021_2023/wayanad   (or data/raw/wayanad)
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def period_id(cfg: dict | None) -> str | None:
    return (cfg or {}).get("period_id")


def _root(kind: str, period: str | None) -> pathlib.Path:
    base = REPO / "data" / kind
    return base / period if period else base


def raw_dir(cfg: dict | None = None, rid: str | None = None, *,
            period: str | None = None) -> pathlib.Path:
    d = _root("raw", period or period_id(cfg))
    return d / rid if rid else d


def masks_dir(cfg: dict | None = None, rid: str | None = None, *,
              period: str | None = None) -> pathlib.Path:
    d = _root("masks", period or period_id(cfg))
    return d / rid if rid else d


def proc_dir(cfg: dict | None = None, *, period: str | None = None) -> pathlib.Path:
    return _root("processed", period or period_id(cfg))


def period_from_proc_dir(proc: str | pathlib.Path) -> str | None:
    """`data/processed/2021_2023` -> `2021_2023`; `data/processed` -> None.

    Lets inference read the period straight from a checkpoint's stored
    `data.proc_dir` without a separate flag.
    """
    parts = pathlib.PurePosixPath(str(proc).replace("\\", "/")).parts
    if len(parts) >= 2 and parts[-2] == "processed" and parts[-1] != "processed":
        return parts[-1]
    return None
