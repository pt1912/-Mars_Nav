"""Regional SWIM4MIM ice-consistency candidate search for Mars_Nav.

Data: Twin Worlds derived 0.2-degree signed-int8 grids from PSI SWIM4MIM.
Only the ice-consistency index Ci is used for candidate ranking. Results
are coarse grid cells for follow-up study, not detected deposits, verified
landing/EVA targets, or probabilities of water ice.
"""

from functools import lru_cache
from pathlib import Path
import json
import math

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "swim"
MARS_RADIUS_KM = 3396.0
DEPTHS = {"Ci_0_1": "0–1 m", "Ci_1_5": "1–5 m"}


@lru_cache(maxsize=2)
def _load_grid(depth):
    """Return (Ci float matrix, valid mask, 1D cell-center lat/lon arrays)."""
    if depth not in DEPTHS:
        raise ValueError(f"Unsupported SWIM depth: {depth}")

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing SWIM metadata: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grid = manifest["swim"][depth]["grid"]

    h, w = int(grid["height"]), int(grid["width"])
    xmin, ymax = float(grid["xmin"]), float(grid["ymax"])
    step, scale, nodata = float(grid["res"]), float(grid["scale"]), int(grid["nodata"])

    if h <= 0 or w <= 0 or step <= 0 or scale <= 0 or not (-128 <= nodata <= 127):
        raise ValueError("Invalid SWIM grid metadata")

    filename = Path(str(grid["file"])).name
    if filename != grid["file"] or not filename.endswith(".i8"):
        raise ValueError("Unexpected SWIM grid file path")

    binary = DATA_DIR / filename
    if not binary.is_file():
        raise FileNotFoundError(f"Missing SWIM grid: {binary}")
    if binary.stat().st_size != w * h:
        raise ValueError(f"SWIM grid size does not match {w}×{h} Int8 cells")

    raw = np.fromfile(binary, dtype=np.int8).reshape(h, w)
    ci = raw.astype(np.float32) * scale
    valid = (raw != nodata) & np.isfinite(ci) & (ci >= -1) & (ci <= 1)
    lat = ymax - (np.arange(h) + 0.5) * step
    lon = xmin + (np.arange(w) + 0.5) * step
    return ci, valid, lat, lon, step


def find_ice_candidates(
    center_lat, center_lon, *, depth="Ci_0_1", radius_km=200.0,
    min_ci=0.50, max_points=10, min_separation_km=30.0,
):
    """Rank positive Ci cells within a Mars-surface-radius of a site.

    Geographic search uses Martian great-circle distance, and handles
    longitudes that cross the ±180° meridian. Results are sorted by Ci,
    then proximity to the region center, and spaced to avoid duplicates.
    Returns both candidates and statistics, including zero-result cases.
    """
    center_lat = float(center_lat)
    center_lon = ((float(center_lon) + 180) % 360) - 180
    radius_km = float(radius_km)
    min_ci = float(min_ci)
    min_separation_km = float(min_separation_km)
    if not (-90 <= center_lat <= 90 and 0 < radius_km <= 3000):
        raise ValueError("Invalid search center or radius")
    if not (-1 <= min_ci <= 1) or not (1 <= max_points <= 100) or min_separation_km <= 0:
        raise ValueError("Invalid search filters")

    ci, valid, latitudes, longitudes, resolution = _load_grid(depth)

    # Broadcast to the full 600×1800 grid. This is small and avoids an
    # incorrect rectangular cutoff near the dateline or search boundary.
    lat0 = math.radians(center_lat)
    lat_rad = np.deg2rad(latitudes)[:, None]
    dlat = lat_rad - lat0
    dlon_deg = ((longitudes - center_lon + 180) % 360) - 180
    dlon = np.deg2rad(dlon_deg)[None, :]
    hav = (np.sin(dlat / 2) ** 2
           + np.cos(lat0) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2)
    dist = 2 * MARS_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))

    inside = dist <= radius_km
    available = inside & valid
    eligible = available & (ci >= min_ci)
    rows, cols = np.where(eligible)

    # A higher Ci comes first; tie-break with distance from selected site.
    order = sorted(
        zip(rows.tolist(), cols.tolist()),
        key=lambda rc: (-float(ci[rc]), float(dist[rc]), rc[0], rc[1]),
    )

    result = []
    for row, col in order:
        la, lo = float(latitudes[row]), float(longitudes[col])
        if result:
            p_lat = np.deg2rad([p["lat"] for p in result])
            p_lon = np.deg2rad([p["lon"] for p in result])
            dla = p_lat - math.radians(la)
            dlo = (p_lon - math.radians(lo) + math.pi) % (2 * math.pi) - math.pi
            hh = np.sin(dla / 2) ** 2 + np.cos(math.radians(la)) * np.cos(p_lat) * np.sin(dlo / 2) ** 2
            gaps = 2 * MARS_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hh, 0, 1)))
            if np.any(gaps < min_separation_km):
                continue
        result.append({
            "id": f"SWIM-{len(result)+1:02d}",
            "lat": round(la, 6),
            "lon": round(lo, 6),
            "ci": round(float(ci[row, col]), 3),
            "distance_km": round(float(dist[row, col]), 1),
            "grid_row": row,
            "grid_col": col,
        })
        if len(result) == max_points:
            break

    valid_ci = ci[available]
    return {
        "depth": depth,
        "depth_label": DEPTHS[depth],
        "center": [center_lat, center_lon],
        "radius_km": radius_km,
        "min_ci": min_ci,
        "resolution_deg": resolution,
        "cells_in_radius": int(inside.sum()),
        "valid_cells": int(available.sum()),
        "eligible_cells": int(eligible.sum()),
        "max_ci_in_radius": float(np.max(valid_ci)) if valid_ci.size else None,
        "min_ci_in_radius": float(np.min(valid_ci)) if valid_ci.size else None,
        "points": result,
        "source": "SWIM4MIM / PSI; 0.2° Int8 derivative by Twin Worlds",
    }
