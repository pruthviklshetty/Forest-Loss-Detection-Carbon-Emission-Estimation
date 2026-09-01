"""NDVI and the 3-bin carbon-density baseline.

The 3-bin scheme (dense / moderate / sparse -> 200 / 150 / 100 tC/ha) is
**this study's assumed classification scheme**. It is NOT taken from an IPCC
table; the round numbers are a deliberately coarse stand-in that the Phase 6
regression is meant to replace. NDVI cut points are the forest tercile
boundaries of the study region's own Year-T composite unless overridden.
"""

from __future__ import annotations

import numpy as np

# assumed (not IPCC): carbon density per NDVI bin, tonnes C per hectare
THREE_BIN_VALUES_tC_ha = {"sparse": 100.0, "moderate": 150.0, "dense": 200.0}


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    # nodata / masked pixels can carry NaN; callers mask on np.isfinite, so the
    # transient invalid-op warning here is noise.
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - red) / (nir + red + eps)


def three_bin_carbon_density(
    ndvi: np.ndarray,
    lo: float,
    hi: float,
    values: dict[str, float] = THREE_BIN_VALUES_tC_ha,
) -> np.ndarray:
    """Map NDVI to a piecewise-constant carbon density (tC/ha).

    ndvi < lo            -> sparse
    lo <= ndvi < hi      -> moderate
    ndvi >= hi           -> dense
    """
    out = np.full(ndvi.shape, values["moderate"], dtype=np.float32)
    out[ndvi < lo] = values["sparse"]
    out[ndvi >= hi] = values["dense"]
    return out
