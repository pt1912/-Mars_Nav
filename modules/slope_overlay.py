import numpy as np

from functools import lru_cache
from matplotlib import colormaps

from modules.terrain_analysis import (
    load_terrain,
    calculate_slope
)


# ==========================================
# SLOPE OVERLAY CONFIGURATION
# ==========================================

SLOPE_MAX_DISPLAY = 5.0
SLOPE_MIN_DISPLAY = 0.3


# ==========================================
# GENERATE SLOPE OVERLAY
# ==========================================

@lru_cache(maxsize=1)
def get_slope_overlay():

    # Load MOLA elevation data
    elevation, latitudes, longitudes = load_terrain()

    # Calculate terrain slope
    slope = calculate_slope(
        elevation,
        latitudes,
        longitudes
    )

    # Normalize slope for visualization
    normalized = np.clip(
        slope / SLOPE_MAX_DISPLAY,
        0,
        1
    )

    # Apply color map
    cmap = colormaps["magma"]

    rgba = (
        cmap(normalized) * 255
    ).astype(np.uint8)

    # Make low slopes transparent
    valid = (
        np.isfinite(slope)
        & (elevation > -32000)
    )

    rgba[:, :, 3] = np.where(
        valid & (slope >= SLOPE_MIN_DISPLAY),
        220,
        0
    ).astype(np.uint8)

    # ======================================
    # GEOGRAPHIC BOUNDS
    # ======================================

    lat_step = abs(
        latitudes[1] - latitudes[0]
    )

    lon_step = abs(
        longitudes[1] - longitudes[0]
    )

    lat_min = latitudes[-1] - lat_step / 2
    lat_max = latitudes[0] + lat_step / 2

    lon_min = longitudes[0] - lon_step / 2
    lon_max = longitudes[-1] + lon_step / 2

    # Convert longitude from 0-360 to -180/180
    bounds = [
        [lat_min, lon_min - 360],
        [lat_max, lon_max - 360]
    ]

    # ======================================
    # TERRAIN STATISTICS
    # ======================================

    statistics = {
        "min_slope": float(np.min(slope[valid])),
        "max_slope": float(np.max(slope[valid])),
        "mean_slope": float(np.mean(slope[valid])),
        "min_elevation": float(np.min(elevation[valid])),
        "max_elevation": float(np.max(elevation[valid]))
    }

    return rgba, bounds, statistics