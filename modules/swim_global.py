"""RGBA overlays for the two SWIM4MIM ice-consistency depth products.

The global layer shows only nonnegative Ci; negative values are transparent,
NOT interpreted as confirmed absence of water ice. NoData remains transparent.
"""

from functools import lru_cache
import numpy as np
from matplotlib import colormaps

from modules.swim_data import DEPTHS, load_grid


@lru_cache(maxsize=2)
def load_swim_global(depth='Ci_0_1'):
    """Return RGBA image, Leaflet bounds, and descriptive metadata."""
    grid = load_grid(depth)
    positive = grid.valid & (grid.values >= 0)
    if not positive.any():
        raise ValueError('No nonnegative SWIM Ci cells available for display')

    norm = np.clip(grid.values / 0.85, 0, 1)
    rgba = (colormaps['YlGnBu'](norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(positive, 240, 0).astype(np.uint8)
    rgba.flags.writeable = False
    valid_scores = grid.values[grid.valid]
    info = {
        'depth': DEPTHS[depth],
        'valid_cells': int(grid.valid.sum()),
        'displayed_cells': int(positive.sum()),
        'min_ci': float(valid_scores.min()),
        'max_ci': float(valid_scores.max()),
        'resolution_deg': grid.resolution_deg,
        'display_range': [0.0, 0.85],
    }
    return rgba, grid.bounds, info
