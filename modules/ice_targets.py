"""Regional candidate finder using SWIM4MIM-derived Ci cells.

Candidates are coarse scientific search targets, NOT confirmed deposits,
landing sites, EVA safety assessments, or a multicriteria optimization.
"""

from functools import lru_cache
import math

import numpy as np

from modules.swim_data import DEPTHS, load_grid

MARS_RADIUS_KM = 3396.0  # Preserve the current regional-search model.


@lru_cache(maxsize=128)
def find_ice_candidates(
    center_lat, center_lon, *, depth='Ci_0_1', radius_km=200.0,
    min_ci=0.50, max_points=10, min_separation_km=30.0,
):
    """Rank eligible cells by Ci, then distance; use Mars great circles.

    Accelerated by restricting work to latitude rows within search radius.
    Properly wraps longitude across ±180°. Outputs a fresh immutable-by-
    convention dict; callers should not modify its fields.
    """
    lat0_deg = float(center_lat)
    lon0_deg = ((float(center_lon) + 180) % 360) - 180
    radius_km = float(radius_km)
    min_ci = float(min_ci)
    min_separation_km = float(min_separation_km)
    max_points = int(max_points)
    if not (np.isfinite([lat0_deg, lon0_deg, radius_km, min_ci, min_separation_km]).all()):
        raise ValueError('Non-finite ice-search parameter')
    if not (-90 <= lat0_deg <= 90 and 0 < radius_km <= 3000):
        raise ValueError('Invalid center or search radius')
    if not (-1 <= min_ci <= 1 and 1 <= max_points <= 100 and min_separation_km > 0):
        raise ValueError('Invalid search filters')

    grid = load_grid(depth)
    lat_limit = math.degrees(radius_km / MARS_RADIUS_KM) + grid.resolution_deg
    row_indices = np.flatnonzero(np.abs(grid.latitudes - lat0_deg) <= lat_limit)

    if len(row_indices) == 0:
        return _no_data_result(depth, lat0_deg, lon0_deg, radius_km, min_ci, grid.resolution_deg)

    lat_rad = np.deg2rad(grid.latitudes[row_indices])[:, None]
    lon_rad = np.deg2rad(((grid.longitudes - lon0_deg + 180) % 360) - 180)[None, :]
    lat0 = math.radians(lat0_deg)
    hav = (np.sin((lat_rad - lat0) / 2) ** 2
           + np.cos(lat0) * np.cos(lat_rad) * np.sin(lon_rad / 2) ** 2)
    dist = 2 * MARS_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))

    inside = dist <= radius_km
    valid_slice = grid.valid[row_indices, :]
    ci_slice = grid.values[row_indices, :]
    available = inside & valid_slice
    eligible = available & (ci_slice >= min_ci)
    local_rows, cols = np.where(eligible)

    # Sort primarily by scientific consistency, secondarily by proximity.
    ranked = sorted(
        zip(local_rows.tolist(), cols.tolist()),
        key=lambda rc: (-float(ci_slice[rc]), float(dist[rc]), rc[0], rc[1]),
    )
    candidates = []
    for row_idx, col in ranked:
        row = int(row_indices[row_idx])
        la = float(grid.latitudes[row])
        lo = float(grid.longitudes[col])
        if candidates:
            p_lat = np.deg2rad([p['lat'] for p in candidates])
            p_lon = np.deg2rad([p['lon'] for p in candidates])
            dlat = p_lat - math.radians(la)
            dlon = (p_lon - math.radians(lo) + math.pi) % (2 * math.pi) - math.pi
            hh = np.sin(dlat / 2) ** 2 + np.cos(math.radians(la)) * np.cos(p_lat) * np.sin(dlon / 2) ** 2
            separation = 2 * MARS_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hh, 0, 1)))
            if np.any(separation < min_separation_km):
                continue
        candidates.append({
            'id': f'SWIM-{len(candidates)+1:02d}',
            'lat': round(la, 6), 'lon': round(lo, 6),
            'ci': round(float(grid.values[row, col]), 3),
            'distance_km': round(float(dist[row_idx, col]), 1),
            'grid_row': row, 'grid_col': int(col),
        })
        if len(candidates) >= max_points:
            break

    available_ci = ci_slice[available]
    return {
        'depth': depth, 'depth_label': DEPTHS[depth],
        'center': [lat0_deg, lon0_deg], 'radius_km': radius_km,
        'min_ci': min_ci, 'resolution_deg': grid.resolution_deg,
        'cells_in_radius': int(inside.sum()),
        'valid_cells': int(available.sum()),
        'eligible_cells': int(eligible.sum()),
        'max_ci_in_radius': float(available_ci.max()) if available_ci.size else None,
        'min_ci_in_radius': float(available_ci.min()) if available_ci.size else None,
        'points': candidates,
        'source': 'SWIM4MIM / PSI; 0.2° Int8 derivative by Twin Worlds',
    }


def _no_data_result(depth, lat, lon, radius, min_ci, resolution):
    return {
        'depth': depth, 'depth_label': DEPTHS[depth], 'center': [lat, lon],
        'radius_km': radius, 'min_ci': min_ci, 'resolution_deg': resolution,
        'cells_in_radius': 0, 'valid_cells': 0, 'eligible_cells': 0,
        'max_ci_in_radius': None, 'min_ci_in_radius': None,
        'points': [],
        'source': 'SWIM4MIM / PSI; 0.2° Int8 derivative by Twin Worlds',
    }
