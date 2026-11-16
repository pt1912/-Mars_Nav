"""MOLA DEM availability footprints for the Mars Ice Explorer map.

Amber dashed: raw MOLA mosaic geographic footprint (reference only).
Green filled: committed regional NPZ geographic footprint (route data available).
No raster is loaded and no large source DEM is read.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import folium
import numpy as np

from modules.mola_dem import RAW_DIR, PROCESSED_DIR

_MEGT_RE = re.compile(r"^megt(\d{2})([ns])(\d{3})hb\.img$", flags=re.IGNORECASE)


def _raw_bounds(name: str):
    """Get 44x90-degree MEGDR tile bounds from its standard filename."""
    match = _MEGT_RE.fullmatch(Path(name).name)
    if not match:
        return None
    lat_number, hemisphere, lon = match.groups()
    n = int(lat_number)
    if hemisphere.lower() == "n":
        north, south = float(n), float(n - 44)
    else:
        south, north = float(-n), float(-n + 44)
    west_east = float(int(lon))
    return south, north, west_east, west_east + 90.0


def _as_lon(lon_east: float):
    """Map east-positive longitude to Leaflet -180..180. Keep 180 as +180 at edge."""
    value = (float(lon_east) + 180.0) % 360.0 - 180.0
    return 180.0 if abs(value + 180.0) < 1e-9 and float(lon_east) % 360.0 == 180 else value


def _rectangle_parts(south, north, west_east, east_east):
    """Return non-dateline-crossing rectangles in Leaflet longitude convention."""
    width = east_east - west_east
    if width <= 0 or width > 360:
        return []
    if width >= 360 - 1e-8:
        return [[[south, -180], [north, 180]]]
    # Split when the source footprint crosses the anti-meridian.
    boundaries = [west_east]
    k1 = int((west_east - 180) // 360) + 1
    k2 = int((east_east - 180) // 360)
    boundaries += [180 + 360 * k for k in range(k1, k2 + 1)
                   if west_east < 180 + 360 * k < east_east]
    boundaries += [east_east]
    result = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        west = _as_lon(start)
        east = _as_lon(stop)
        # Segment ending at +180 must end at 180 (not -180).
        if abs((stop - 180) % 360) < 1e-8:
            east = 180.0
        # Segment starting at a post-dateline -180 boundary.
        if abs((start - 180) % 360) < 1e-8:
            west = -180.0
        if east <= west:
            east = west + (stop - start)
        result.append([[south, west], [north, east]])
    return result


def available_dem_coverage():
    """Collect available clipped regions, without loading their elevation matrices.

    Returns (raw_mosaics, regional_cutouts, error_messages).
    """
    source_names = {
        p.name for p in RAW_DIR.iterdir()
        if p.is_file() and _MEGT_RE.fullmatch(p.name)
    } if RAW_DIR.is_dir() else set()
    clips, errors = [], []
    if PROCESSED_DIR.is_dir():
        for path in sorted(PROCESSED_DIR.glob("*_mola.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    meta = json.loads(str(data["metadata_json"].item()))
                clips.append({
                    "site_id": meta["site_id"],
                    "site_name": meta.get("site_name", meta["site_id"]),
                    "file": path.name,
                    "bounds": (
                        float(meta["south_edge"]), float(meta["north_edge"]),
                        float(meta["west_edge_east"]), float(meta["east_edge_east"]),
                    ),
                    "source_img": meta.get("source_img", ""),
                    "partial": bool(meta.get("partial_source_coverage", False)),
                })
                if meta.get("source_img"):
                    source_names.add(meta["source_img"])
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"{path.name}: {exc}")
    sources = [dict(name=name, bounds=_raw_bounds(name), is_local=(RAW_DIR / name).is_file())
               for name in sorted(source_names) if _raw_bounds(name) is not None]
    return sources, clips, errors


def create_dem_coverage_overlay(sources=None, clips=None):
    """Draw the original MOLA footprint and available processed DEM extents."""
    if sources is None or clips is None:
        sources, clips, _ = available_dem_coverage()
    group = folium.FeatureGroup(name="MOLA DEM coverage", control=False, show=True)
    for src in sources:
        south, north, west, east = src["bounds"]
        label = ("Local source" if src["is_local"] else "Source footprint (not installed)")
        for bounds in _rectangle_parts(south, north, west, east):
            folium.Rectangle(
                bounds=bounds,
                color="#F59E0B", weight=2, dash_array="8, 6", fill=False,
                opacity=0.95, interactive=False,
                tooltip=f"MOLA original · {src['name']} · {label}",
            ).add_to(group)
    for clip in clips:
        south, north, west, east = clip["bounds"]
        for bounds in _rectangle_parts(south, north, west, east):
            folium.Rectangle(
                bounds=bounds, color="#22C55E", weight=2.5, fill=True,
                fill_color="#22C55E", fill_opacity=0.12,
                interactive=False,
                tooltip=(f"DEM ready · {clip['site_name']} · {clip['file']}"
                         + (" · partial source coverage" if clip['partial'] else "")),
            ).add_to(group)
    return group
