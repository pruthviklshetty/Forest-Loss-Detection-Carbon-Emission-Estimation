"""Earth Engine helpers: config loading, init, and a robust tiled GeoTIFF
downloader.

The AOI at 10 m is ~3100 x 2800 px; a single `getDownloadURL` call exceeds the
~48 MB request cap, so images are downloaded in lat/lon tiles and mosaicked
locally with rasterio. Because every tile is fetched with a projected CRS
(EPSG:32643) and an integer 10 m scale, Earth Engine snaps each tile to the
same global UTM pixel grid, so the tiles are mutually pixel-aligned and
`rasterio.merge` stitches them without seams.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import time
from typing import Iterable, Sequence

import ee
import rasterio
import requests
import yaml
from rasterio.merge import merge as rio_merge

_REPO = pathlib.Path(__file__).resolve().parents[2]


def load_cfg(path: str | pathlib.Path = "configs/region.yaml") -> dict:
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = _REPO / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def init_ee(cfg: dict | None = None) -> str:
    """ee.Initialize with the project from config; returns the project id."""
    cfg = cfg or load_cfg()
    project = cfg["earth_engine"]["project"]
    ee.Initialize(project=project)
    return project


def _frange(start: float, stop: float, step: float) -> list[float]:
    out, x = [], start
    while x < stop - 1e-9:
        out.append(x)
        x += step
    return out


def _tiles(
    bbox: Sequence[float], step_deg: float, overlap_deg: float
) -> Iterable[tuple[float, float, float, float]]:
    w, s, e, n = bbox
    for x0 in _frange(w, e, step_deg):
        for y0 in _frange(s, n, step_deg):
            yield (
                max(w, x0 - overlap_deg),
                max(s, y0 - overlap_deg),
                min(e, x0 + step_deg + overlap_deg),
                min(n, y0 + step_deg + overlap_deg),
            )


def _get_url_with_retry(image: ee.Image, params: dict, tries: int = 5) -> str:
    for attempt in range(1, tries + 1):
        try:
            return image.getDownloadURL(params)
        except ee.ee_exception.EEException as exc:  # transient 429/500
            if attempt == tries:
                raise
            wait = 5 * attempt
            print(f"    getDownloadURL retry {attempt}/{tries} in {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _download_with_retry(url: str, dst: pathlib.Path, tries: int = 5) -> None:
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return
        except requests.RequestException as exc:
            if attempt == tries:
                raise
            wait = 5 * attempt
            print(f"    tile GET retry {attempt}/{tries} in {wait}s ({exc})")
            time.sleep(wait)


def download_image_tiled(
    image: ee.Image,
    bbox_wsen: Sequence[float],
    out_path: str | pathlib.Path,
    crs: str = "EPSG:32643",
    scale_m: int = 10,
    step_deg: float = 0.07,
    overlap_deg: float = 5.0e-4,
    bands: Sequence[str] | None = None,
    band_names: Sequence[str] | None = None,
) -> pathlib.Path:
    """Download `image` clipped to `bbox_wsen` (a [W,S,E,N] lat/lon box) as one
    mosaicked GeoTIFF at `out_path`, fetched in tiles. `band_names`, if given,
    is written as the output band descriptions (EE GeoTIFF export drops them)."""
    if bands is not None:
        image = image.select(list(bands))
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tiles = list(_tiles(bbox_wsen, step_deg, overlap_deg))
    print(f"  {out_path.name}: {len(tiles)} tiles @ {scale_m} m / {crs}")

    with tempfile.TemporaryDirectory() as td:
        tdir = pathlib.Path(td)
        parts: list[pathlib.Path] = []
        for i, (tw, ts, te, tn) in enumerate(tiles):
            region = ee.Geometry.Rectangle([tw, ts, te, tn], "EPSG:4326", geodesic=False)
            url = _get_url_with_retry(
                image,
                {
                    "region": region,
                    "scale": scale_m,
                    "crs": crs,
                    "format": "GEO_TIFF",
                    "filePerBand": False,
                },
            )
            part = tdir / f"tile_{i:03d}.tif"
            _download_with_retry(url, part)
            parts.append(part)
            print(f"    tile {i + 1}/{len(tiles)}  {part.stat().st_size / 1e6:.1f} MB")

        srcs = [rasterio.open(p) for p in parts]
        try:
            mosaic, transform = rio_merge(srcs)
            meta = srcs[0].meta.copy()
            meta.update(
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                count=mosaic.shape[0],
                transform=transform,
                compress="deflate",
            )
            tile_desc = list(srcs[0].descriptions) if any(srcs[0].descriptions) else None
        finally:
            for s in srcs:
                s.close()

        out_desc = band_names or tile_desc
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
            if out_desc and len(out_desc) == meta["count"]:
                dst.descriptions = tuple(out_desc)

    with rasterio.open(out_path) as chk:
        print(
            f"  -> {out_path}  {chk.width}x{chk.height} px, {chk.count} bands, "
            f"{chk.dtypes[0]}, CRS {chk.crs}"
        )
    return out_path


def write_manifest(path: str | pathlib.Path, payload: dict) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  manifest -> {path}")
