"""SWIM4MIM ice-consistency raster overlays for the Mars global map.

The values represent a consistency index (Ci), not an ice probability.
Negative values are scientifically meaningful and must not be masked.
Only the integer NoData sentinel (-128) is rendered transparent.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import folium
import numpy as np
from PIL import Image


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "swim"
DEPTH_LABELS = {"Ci_0_1": "0–1 m", "Ci_1_5": "1–5 m"}

# Fixed diverging scale -1..+1; identical for both depths, so they are comparable.
# Negative Ci: warm; zero: neutral; positive Ci: cyan/blue.
_PALETTE_STOPS = np.array([-1.0, -0.4, 0.0, 0.3, 0.65, 1.0])
_PALETTE_RGB = np.array(
    [
        [127, 29, 29],   # -1.0 dark rust
        [203, 106, 71],  # -0.4 terracotta
        [230, 233, 233], #  0.0 light neutral
        [136, 209, 208], # +0.3 turquoise
        [42, 148, 190],  # +0.65 blue
        [23, 59, 141],   # +1.0 deep blue
    ],
    dtype=np.float32,
)


@lru_cache(maxsize=2)
def load_swim_overlay(depth: str):
    """Read/validate one i8 grid, encode PNG once, and cache the resulting URL.

    Returns (data_url, leaflet_bounds, metadata).
    The geographic grid is equirectangular, with row 0 at ymax (north).
    """
    if depth not in DEPTH_LABELS:
        raise ValueError(f"Unsupported SWIM depth: {depth}")

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SWIM manifest missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grid = manifest["swim"][depth]["grid"]

    width = int(grid["width"])
    height = int(grid["height"])
    xmin = float(grid["xmin"])
    ymax = float(grid["ymax"])
    resolution = float(grid["res"])
    scale = float(grid["scale"])
    nodata = int(grid["nodata"])
    filename = str(grid["file"])

    if not (0 < width <= 10000 and 0 < height <= 10000):
        raise ValueError("Invalid SWIM raster dimensions")
    if not (0 < resolution <= 2 and 0 < scale <= 1):
        raise ValueError("Invalid SWIM resolution or scale")
    if not (-128 <= nodata <= 127):
        raise ValueError("Invalid SWIM NoData sentinel")
    if filename != Path(filename).name or not filename.endswith(".i8"):
        raise ValueError("Unexpected SWIM raster filename")

    ymin = ymax - height * resolution
    xmax = xmin + width * resolution
    if not (-90 <= ymin < ymax <= 90 and -180 <= xmin < xmax <= 180.0001):
        raise ValueError("SWIM geographic bounds exceed planetary coordinates")

    raster_path = DATA_DIR / filename
    if not raster_path.is_file():
        raise FileNotFoundError(f"SWIM raster missing: {raster_path}")
    if raster_path.stat().st_size != width * height:
        raise ValueError(f"Invalid size for {filename}: expected {width * height} bytes")

    raw = np.fromfile(raster_path, dtype=np.int8).reshape(height, width)
    valid = raw != nodata
    if not valid.any():
        raise ValueError(f"No valid cells in {filename}")

    ci = raw.astype(np.float32) * scale
    clipped = np.clip(ci, -1, 1)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    for channel in range(3):
        rgba[..., channel] = np.interp(
            clipped, _PALETTE_STOPS, _PALETTE_RGB[:, channel]
        ).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG", optimize=True)
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    valid_ci = ci[valid]
    info = {
        "depth": DEPTH_LABELS[depth],
        "valid_cells": int(valid.sum()),
        "min_ci": float(valid_ci.min()),
        "max_ci": float(valid_ci.max()),
        "resolution_deg": resolution,
        "coverage": f"{ymin:.0f}° to {ymax:.0f}° latitude",
    }
    bounds = [[ymin, xmin], [ymax, xmax]]
    return image_url, bounds, info


def create_swim_overlay_group(*, show_0_1=False, show_1_5=False, opacity=0.55):
    """Build a replaceable Folium group containing only enabled ice layers."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("SWIM opacity must be between 0 and 1")

    group = folium.FeatureGroup(name="SWIM Ice Overlays", control=False)
    for enabled, depth in (
        (show_0_1, "Ci_0_1"),
        (show_1_5, "Ci_1_5"),
    ):
        if not enabled:
            continue
        image_url, bounds, _ = load_swim_overlay(depth)
        folium.raster_layers.ImageOverlay(
            image=image_url,
            bounds=bounds,
            name=f"SWIM4MIM | Ci {DEPTH_LABELS[depth]}",
            opacity=opacity,
            pixelated=True,
            interactive=False,
            control=False,
        ).add_to(group)
    return group
