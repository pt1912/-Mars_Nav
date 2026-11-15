"""SWIM4MIM derived global ice-consistency layer for ARCADIA ICE EXPLORER.

Input is the signed int8 0.2-degree grid and manifest from Twin Worlds:
https://github.com/faresnagyeleissawy/twin-worlds
Original scientific product: SWIM4MIM, Planetary Science Institute.
The numbers are derived, averaged and quantized; they are NOT probabilities.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWIM_DIR = PROJECT_ROOT / "data" / "raw" / "swim"


@lru_cache(maxsize=2)
def load_swim_global(depth="Ci_0_1"):
    """Return (RGBA uint8 image, geographic bounds, metadata) for Folium."""
    if depth not in ("Ci_0_1", "Ci_1_5"):
        raise ValueError(f"Unsupported SWIM depth: {depth}")

    manifest_file = SWIM_DIR / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            "Missing data/raw/swim/manifest.json. Copy the manifest and .i8 "
            "files from the Twin Worlds public/data directory."
        )

    metadata = json.loads(manifest_file.read_text(encoding="utf-8"))
    grid = metadata["swim"][depth]["grid"]
    width = int(grid["width"])
    height = int(grid["height"])
    xmin = float(grid["xmin"])
    ymax = float(grid["ymax"])
    resolution = float(grid["res"])
    scale = float(grid["scale"])
    nodata = int(grid["nodata"])
    if not (width > 0 and height > 0 and resolution > 0 and scale > 0):
        raise ValueError("Invalid SWIM grid geometry")
    if not -128 <= nodata <= 127:
        raise ValueError("SWIM NoData outside Int8 range")

    # Never trust a path contained in a third-party manifest.
    binary = SWIM_DIR / Path(str(grid["file"])).name
    if not binary.is_file():
        raise FileNotFoundError(f"Missing SWIM grid: {binary}")
    if binary.stat().st_size != width * height:
        raise ValueError(
            f"SWIM grid byte length {binary.stat().st_size} does not match "
            f"{width} x {height} Int8 cells."
        )

    raw = np.fromfile(binary, dtype=np.int8).reshape(height, width)
    scores = raw.astype(np.float32) * scale
    valid = (raw != nodata) & np.isfinite(scores) & (scores >= -1) & (scores <= 1)

    # Visualize only nonnegative Ci values. NoData/negative values are transparent,
    # not classified as places where water ice is absent.
    shown = valid & (scores >= 0)
    if not np.any(shown):
        raise ValueError("SWIM layer contains no valid nonnegative consistency cells")

    norm = np.clip(scores / 0.85, 0, 1)
    rgba = (colormaps["YlGnBu"](norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(shown, 240, 0).astype(np.uint8)

    # Grid is north-up; Leaflet's ImageOverlay expects SW and NE corners.
    bounds = [
        [ymax - height * resolution, xmin],
        [ymax, xmin + width * resolution],
    ]
    info = {
        "depth": "0–1 m" if depth == "Ci_0_1" else "1–5 m",
        "valid_cells": int(np.count_nonzero(valid)),
        "displayed_cells": int(np.count_nonzero(shown)),
        "min_ci": float(np.min(scores[valid])),
        "max_ci": float(np.max(scores[valid])),
        "resolution_deg": resolution,
        "display_range": [0.0, 0.85],
    }
    return rgba, bounds, info


if __name__ == "__main__":
    image, box, info = load_swim_global()
    print("SWIM:", image.shape, image.dtype, "bounds:", box)
    print(info)
