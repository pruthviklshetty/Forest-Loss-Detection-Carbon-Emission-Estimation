"""Phase 1 install checkpoint.

Imports every load-bearing dependency, prints its version, and does a couple of
tiny functional smoke tests (build an smp model, run a forward pass, open a
rasterio dataset in memory). Exits non-zero if anything fails so it can gate CI
or a fresh-Colab check.

Run:  python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys

EXPECTED = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "segmentation_models_pytorch": "0.3.4",
    "timm": "0.9.7",
    "numpy": "1.26.4",
    "scipy": "1.14.1",
    "sklearn": "1.5.2",
    "pandas": "2.2.3",
    "rasterio": "1.3.11",
    "shapely": "2.0.6",
    "pyproj": "3.7.0",
    "cv2": "4.10.0",
    "PIL": "10.4.0",
    "ee": None,          # earthengine-api version string is not always exposed
    "requests": "2.32.3",
    "yaml": "6.0.2",
    "matplotlib": "3.9.2",
    "tqdm": "4.66.5",
}

IMPORT_NAME_TO_DIST = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "ee": "earthengine-api",
    "yaml": "PyYAML",
    "segmentation_models_pytorch": "segmentation-models-pytorch",
}


def _version(mod) -> str:
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    try:
        from importlib.metadata import version

        name = IMPORT_NAME_TO_DIST.get(mod.__name__, mod.__name__)
        return version(name)
    except Exception:
        return "unknown"


def main() -> int:
    print(f"Python  {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
    if sys.version_info[:2] != (3, 11):
        print(f"  WARNING: expected CPython 3.11, got {sys.version_info.major}.{sys.version_info.minor}")

    failures: list[str] = []
    mismatches: list[str] = []

    for import_name, expected in EXPECTED.items():
        try:
            mod = importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{import_name}: import failed -> {exc!r}")
            print(f"  [FAIL] {import_name:32s} import error")
            continue
        ver = _version(mod)
        tag = ""
        if expected and not ver.startswith(expected):
            tag = f"  (expected {expected})"
            mismatches.append(f"{import_name}: got {ver}, expected {expected}")
        print(f"  [ ok ] {import_name:32s} {ver}{tag}")

    print("\nFunctional smoke tests:")

    # torch forward pass
    try:
        import torch

        x = torch.randn(1, 8, 64, 64)
        y = torch.nn.Conv2d(8, 4, 3, padding=1)(x)
        assert y.shape == (1, 4, 64, 64)
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU only"
        print(f"  [ ok ] torch conv2d forward pass; CUDA available: {cuda} ({dev})")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"torch smoke test -> {exc!r}")
        print(f"  [FAIL] torch smoke test -> {exc!r}")

    # segmentation_models_pytorch: build a U-Net w/ MobileNetV2, 8-band input
    try:
        import segmentation_models_pytorch as smp
        import torch

        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,      # no download during the check
            in_channels=8,
            classes=1,
        )
        out = model(torch.randn(1, 8, 256, 256))
        assert out.shape == (1, 1, 256, 256), out.shape
        print("  [ ok ] smp.Unet(mobilenet_v2, in_channels=8) forward pass -> (1,1,256,256)")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"smp smoke test -> {exc!r}")
        print(f"  [FAIL] smp smoke test -> {exc!r}")

    # rasterio: write + read a tiny in-memory GeoTIFF
    try:
        import numpy as np
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        data = (np.random.rand(2, 32, 32) * 255).astype("uint8")
        transform = from_origin(76.0, 11.8, 1e-4, 1e-4)
        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff", height=32, width=32, count=2,
                dtype="uint8", crs="EPSG:4326", transform=transform,
            ) as dst:
                dst.write(data)
            with mem.open() as src:
                back = src.read()
        assert np.array_equal(back, data)
        print("  [ ok ] rasterio GeoTIFF round-trip (MemoryFile)")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"rasterio smoke test -> {exc!r}")
        print(f"  [FAIL] rasterio smoke test -> {exc!r}")

    # pyproj: UTM 43N transform used for area computation
    try:
        from pyproj import Transformer

        t = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
        e, n = t.transform(76.14, 11.675)
        assert 600000 < e < 800000 and 1_200_000 < n < 1_400_000, (e, n)
        print(f"  [ ok ] pyproj 4326 -> 32643 transform ({e:.0f}, {n:.0f})")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"pyproj smoke test -> {exc!r}")
        print(f"  [FAIL] pyproj smoke test -> {exc!r}")

    print()
    if mismatches:
        print("Version mismatches (not fatal, but pins drifted):")
        for m in mismatches:
            print(f"  - {m}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("\nRESULT: environment is NOT clean.")
        return 1

    print("RESULT: environment OK — all imports and smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
