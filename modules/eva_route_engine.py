"""Single MOLA-backed EVA route engine: Shortest / Safest / Balanced A*.

This is a *regional corridor planner*. The 128-pixel/degree MOLA grid cannot
resolve meter-scale EVA hazards. Environmental values are user scenarios, NOT
NASA weather observations. No online request is used during routing.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from modules.mola_dem import MARS_RADIUS_M, grid_lat_lon, load_region_dem, point_rc

ROUTE_STYLE = {
    'Shortest Route': {'color': '#34D399', 'dash': None},
    'Safest Route': {'color': '#60A5FA', 'dash': '10, 5'},
    'Balanced Route': {'color': '#FDBA74', 'dash': '4, 4'},
}


@dataclass(frozen=True)
class RouteSettings:
    max_slope_deg: float = 15.0       # experimental DEM-scale screening
    roughness_limit_m: float = 20.0   # experimental; local elevation stddev
    map_margin_km: float = 30.0
    speed_kmh: float = 2.0             # assumed EVA horizontal nominal speed
    max_eva_hours: float = 8.0        # user-configured mission constraint
    dust: str = 'Clear'               # assumed scenario
    temperature_c: float = -50.0      # assumed ambient scenario
    illumination: str = 'Daylight'   # assumed scenario, not ephemeris
    balanced_risk_weight: float = 1.0
    max_grid_cells: int = 150_000


def _local_box(a, b, meta, shape, margin_km):
    ar, ac = point_rc(a['lat'], a['lon'], meta, shape)
    br, bc = point_rc(b['lat'], b['lon'], meta, shape)
    mean_lat = (float(a['lat']) + float(b['lat'])) / 2
    ppd = float(meta['resolution_ppd'])
    dy = math.pi / 180 * MARS_RADIUS_M / ppd
    dx = dy * max(0.03, math.cos(math.radians(mean_lat)))
    margin_r = int(math.ceil(margin_km * 1000 / dy))
    margin_c = int(math.ceil(margin_km * 1000 / dx))
    r0 = max(0, min(ar, br) - margin_r)
    r1 = min(shape[0], max(ar, br) + margin_r + 1)
    c0 = max(0, min(ac, bc) - margin_c)
    c1 = min(shape[1], max(ac, bc) + margin_c + 1)
    return r0, r1, c0, c1, dx, dy, (ar, ac), (br, bc)


def _coarsen(elev, cell_limit):
    """Avoid costly large grids using mean pooling, never pixel interpolation."""
    h, w = elev.shape
    factor = max(1, int(math.ceil(math.sqrt(h * w / cell_limit))))
    if factor == 1:
        return elev.copy(), 1
    H, W = h // factor, w // factor
    if H < 3 or W < 3:
        raise ValueError('Grid too small after downsampling')
    with np.errstate(invalid='ignore'):
        blocks = elev[:H * factor, :W * factor].reshape(H, factor, W, factor)
        finite = np.isfinite(blocks)
        sums = np.where(finite, blocks, 0).sum(axis=(1, 3))
        counts = finite.sum(axis=(1, 3))
        coarse = sums / np.maximum(counts, 1)
        coarse[counts != factor * factor] = np.nan  # conservative missing-data mask
    return coarse.astype(np.float32), factor


def _neighborhood_std(elevation, dx_m=1.0, dy_m=1.0):
    """3x3 detrended roughness in meters, excluding the locally planar slope.

    The old standard deviation of absolute elevation overestimated roughness on
    perfectly smooth sloping surfaces, double-counting slope in the hazard map.
    A missing neighbor invalidates the local roughness estimate.
    """
    valid = np.isfinite(elevation)
    with np.errstate(invalid='ignore'):
        gy, gx = np.gradient(elevation, dy_m, dx_m)
    h, w = elevation.shape
    filled = np.where(valid, elevation, 0.0).astype(np.float64)
    padded = np.pad(filled, 1, mode='edge')
    padded_valid = np.pad(valid.astype(np.float64), 1, mode='edge')
    sum_resid = np.zeros((h, w), dtype=np.float64)
    sum_resid2 = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            neig = padded[dr + 1:dr + 1 + h, dc + 1:dc + 1 + w]
            ok = padded_valid[dr + 1:dr + 1 + h, dc + 1:dc + 1 + w]
            resid = np.where(ok > 0, neig - (gy * dr * dy_m + gx * dc * dx_m), 0.0)
            sum_resid += np.nan_to_num(resid, nan=0)
            sum_resid2 += np.nan_to_num(resid * resid, nan=0)
            count += ok
    mean = sum_resid / np.maximum(count, 1)
    std = np.sqrt(np.maximum(sum_resid2 / np.maximum(count, 1) - mean * mean, 0))
    std[~valid | (count < 9) | ~np.isfinite(gy) | ~np.isfinite(gx)] = np.nan
    return std.astype(np.float32)


def terrain_maps(elevation, dx_m, dy_m, max_slope_deg, roughness_limit_m):
    finite = np.isfinite(elevation)
    # NaNs spread through the derivatives; they're masked below.
    with np.errstate(invalid='ignore', divide='ignore'):
        row_grad, col_grad = np.gradient(elevation, dy_m, dx_m)
        slope = np.degrees(np.arctan(np.hypot(row_grad, col_grad)))
        rough = _neighborhood_std(elevation, dx_m, dy_m)
    traversable = finite & np.isfinite(slope) & np.isfinite(rough)
    traversable &= slope <= max_slope_deg
    terrain_risk = (
        0.68 * np.clip(slope / max(max_slope_deg, 0.1), 0, 1)
        + 0.32 * np.clip(rough / max(roughness_limit_m, 0.1), 0, 1)
    ).astype(np.float32)
    terrain_risk[~traversable] = np.nan
    return slope.astype(np.float32), rough, traversable, terrain_risk


def _ambient_scenario(config):
    dust = {'Clear': 0, 'Dusty': 1, 'Severe dust': 2}[config.dust]
    light = {'Daylight': 0, 'Low Sun': 1, 'Dark': 2}[config.illumination]
    # This is a dimensionless scenario exposure factor, not physical NASA data.
    cold = min(1.0, max(0.0, (-config.temperature_c - 50.0) / 70.0))
    return min(1.0, 0.10 + 0.22 * dust + 0.15 * light + 0.18 * cold), dust, light


def _great_circle_m(a, b):
    """Martian spherical surface distance for clicked point / cell center links."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dl = lat2 - lat1
    dn = (lon2 - lon1 + math.pi) % (2 * math.pi) - math.pi
    h = math.sin(dl / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dn / 2) ** 2
    return 2 * MARS_RADIUS_M * math.asin(math.sqrt(max(0, min(1, h))))


def cost_surface(slope, rough, risk, valid, config: RouteSettings, strategy: str):
    """Visual *per-meter cost proxy* for explaining route choices.

    Unlike A*, this map cannot represent direction-specific uphill cost;
    it is labeled as a proxy, not an exact per-edge cost.
    """
    env, dust, light = _ambient_scenario(config)
    local_rough = np.clip(np.nan_to_num(rough, nan=0) / max(config.roughness_limit_m, 0.1), 0, 2)
    model_time = 1 + .35 * local_rough + .12 * light + .08 * dust
    grade = np.clip(np.tan(np.radians(np.nan_to_num(slope, nan=0))), 0, 2)
    effort = np.clip(.7 * grade + .3 * local_rough, 0, 3)
    if strategy == 'Shortest Route':
        out = np.ones_like(slope, dtype=np.float32)
    elif strategy == 'Safest Route':
        out = (.20 + 3.6 * np.nan_to_num(risk, nan=0) + .85 * env * model_time).astype(np.float32)
    elif strategy == 'Balanced Route':
        out = (.72 + config.balanced_risk_weight * 1.4 * np.nan_to_num(risk, nan=0)
               + .55 * effort + .70 * env * model_time).astype(np.float32)
    else:
        raise ValueError(f'Unknown route strategy: {strategy}')
    out[~valid] = np.nan
    return out


def _a_star(elev, hazard, rough, valid, sr, sc, tr, tc, dx, dy, config, strategy):
    height, width = elev.shape
    src = sr * width + sc
    dst = tr * width + tc
    if not valid[sr, sc] or not valid[tr, tc]:
        raise ValueError('Start or destination falls on invalid / over-limit DEM cell')
    g = np.full(height * width, np.inf, dtype=np.float64)
    parents = np.full(height * width, -1, dtype=np.int32)
    closed = np.zeros(height * width, dtype=np.bool_)
    g[src] = 0.0
    env, dust, light = _ambient_scenario(config)
    direction = [(-1, 0, dy), (1, 0, dy), (0, -1, dx), (0, 1, dx),
                 (-1, -1, math.hypot(dx, dy)), (-1, 1, math.hypot(dx, dy)),
                 (1, -1, math.hypot(dx, dy)), (1, 1, math.hypot(dx, dy))]
    base = {'Shortest Route': 1.0, 'Safest Route': 0.20,
            'Balanced Route': 0.72}[strategy]

    def heuristic(r, c):
        return base * math.hypot((r - tr) * dy, (c - tc) * dx)

    frontier = [(heuristic(sr, sc), src)]
    expanded = 0
    while frontier:
        _, idx = heapq.heappop(frontier)
        if closed[idx]:
            continue
        if idx == dst:
            break
        closed[idx] = True
        expanded += 1
        if expanded > config.max_grid_cells:
            raise RuntimeError('Search exceeded allowed cell expansions')
        r, c = divmod(idx, width)
        for dr, dc, step_m in direction:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or not valid[nr, nc]:
                continue
            if dr and dc and not (valid[r, nc] and valid[nr, c]):
                continue  # never cut corners around blocked terrain
            nidx = nr * width + nc
            if closed[nidx]:
                continue
            dz = float(elev[nr, nc] - elev[r, c])
            local_grade = min(abs(dz) / step_m, 2.0)
            local_rough = min(2.0, 0.5 * (float(rough[r, c] + rough[nr, nc])) /
                              max(config.roughness_limit_m, 0.1))
            uphill = max(0.0, dz) / step_m
            effort = min(3.0, 0.7 * local_grade + 0.3 * local_rough + 2.0 * uphill)
            exposure_time_factor = 1.0 + 1.5 * uphill + 0.35 * local_rough + 0.12 * light + 0.08 * dust
            hval = 0.5 * (float(hazard[r, c]) + float(hazard[nr, nc]))
            if strategy == 'Shortest Route':
                cost = step_m
            elif strategy == 'Safest Route':
                cost = step_m * (0.20 + 3.6 * hval + 0.85 * env * exposure_time_factor)
            else:
                cost = step_m * (0.72 + config.balanced_risk_weight * 1.4 * hval
                                 + 0.55 * effort + 0.70 * env * exposure_time_factor)
            ng = g[idx] + cost
            if ng < g[nidx]:
                g[nidx] = ng
                parents[nidx] = idx
                heapq.heappush(frontier, (ng + heuristic(nr, nc), nidx))
    if not np.isfinite(g[dst]):
        raise ValueError('No traversable path found between points under selected constraints')
    path = []
    index = dst
    while index >= 0:
        path.append(divmod(int(index), width))
        if index == src:
            break
        index = int(parents[index])
    if not path or path[-1] != (sr, sc):
        raise RuntimeError('Could not reconstruct A* route')
    path.reverse()
    return np.asarray(path, dtype=np.int32), float(g[dst]), expanded


def _route_stats(path, elevation, slope, rough, hazard, dx, dy, config):
    rows, cols = path[:, 0], path[:, 1]
    drow, dcol = np.diff(rows), np.diff(cols)
    steps = np.hypot(drow * dy, dcol * dx)
    heights = elevation[rows, cols]
    dz = np.diff(heights)
    distance_km = float(np.sum(steps)) / 1000
    ascent = float(np.maximum(0, dz).sum())
    descent = float(np.maximum(0, -dz).sum())
    avg_risk = float(np.average(hazard[rows[1:], cols[1:]], weights=steps)) if len(steps) else 0.0
    peak_slope = float(np.max(slope[rows, cols]))
    rough_mean = float(np.mean(rough[rows, cols]))
    env, dust, light = _ambient_scenario(config)
    up_rel = np.maximum(0, dz) / np.maximum(steps, 1)
    rough_rel = np.clip(rough[rows[1:], cols[1:]] / max(config.roughness_limit_m, 0.1), 0, 2)
    # Scenario travel-time approximation; not calibrated EVA physiology.
    travel_time_h = float(np.sum(steps / 1000 / config.speed_kmh *
                         (1 + 1.5 * up_rel + 0.35 * rough_rel + 0.12 * light + 0.08 * dust)))
    env_exposure = env * travel_time_h
    # Only report an effort proxy; no claim of energy consumption in watts.
    effort_proxy = distance_km + ascent / 1000 * 6.0
    return dict(distance_km=round(distance_km, 3), ascent_m=round(ascent, 1),
                descent_m=round(descent, 1), maximum_slope_deg=round(peak_slope, 2),
                mean_roughness_m=round(rough_mean, 2), mean_terrain_risk=round(avg_risk, 3),
                duration_hours=round(travel_time_h, 2), environmental_exposure_index_hours=round(env_exposure, 2),
                effort_proxy=round(effort_proxy, 2),
                within_eva_time=bool(travel_time_h <= config.max_eva_hours),
                elevation_profile_m=heights[::max(1, len(heights)//300)].astype(float).tolist())


def _add_endpoint_links(stats, extra_m, config):
    """Report travel all the way to the clicked coordinates, not grid centers.

    Connector segments are sub-grid approximations. Their slope cannot be
    determined at cell resolution; their ground distance and nominal time
    are included consistently for all three strategies.
    """
    if extra_m <= 0:
        return stats
    extra_km = extra_m / 1000.0
    env, dust, light = _ambient_scenario(config)
    extra_hours = extra_km / config.speed_kmh * (1 + .12 * light + .08 * dust)
    stats['distance_km'] = round(stats['distance_km'] + extra_km, 3)
    stats['duration_hours'] = round(stats['duration_hours'] + extra_hours, 2)
    stats['environmental_exposure_index_hours'] = round(
        stats['environmental_exposure_index_hours'] + env * extra_hours, 2)
    stats['effort_proxy'] = round(stats['effort_proxy'] + extra_km, 2)
    stats['within_eva_time'] = bool(stats['duration_hours'] <= config.max_eva_hours)
    return stats


def plan_eva_routes(site_id, start, destination, config: RouteSettings, output_dir=None):
    """Return three paths and hazard map based on only a small local DEM window."""
    if config.max_slope_deg <= 0 or config.speed_kmh <= 0:
        raise ValueError('Slope limit and walking speed must be positive')
    elev, source = load_region_dem(site_id, output_dir)
    r0, r1, c0, c1, dx, dy, a_rc, b_rc = _local_box(start, destination, source, elev.shape,
                                                       config.map_margin_km)
    src = elev[r0:r1, c0:c1]
    grid, factor = _coarsen(src, config.max_grid_cells)
    dx *= factor
    dy *= factor
    start_rc = ((a_rc[0] - r0) // factor, (a_rc[1] - c0) // factor)
    end_rc = ((b_rc[0] - r0) // factor, (b_rc[1] - c0) // factor)
    if (not 0 <= start_rc[0] < grid.shape[0] or not 0 <= start_rc[1] < grid.shape[1]
            or not 0 <= end_rc[0] < grid.shape[0] or not 0 <= end_rc[1] < grid.shape[1]):
        raise ValueError('Waypoint too close to an incomplete DEM boundary after grid reduction')
    if start_rc == end_rc:
        raise ValueError('Waypoints fall within the same MOLA analysis cell; choose more distant points')
    slope, rough, traversable, risk = terrain_maps(grid, dx, dy, config.max_slope_deg,
                                                    config.roughness_limit_m)
    if min(grid.shape) < 3 or not (traversable[start_rc] and traversable[end_rc]):
        raise ValueError('Start or destination has invalid terrain or exceeds the slope threshold')
    meta = dict(source)
    # Account for the region crop offset and the pool factor in geographic centers.
    meta['north_edge'] = float(source['north_edge']) - r0 / float(source['resolution_ppd'])
    meta['west_edge_east'] = float(source['west_edge_east']) + c0 / float(source['resolution_ppd'])
    meta['resolution_ppd'] = float(source['resolution_ppd']) / factor
    results = {}
    errors = {}
    exact_start = (float(start['lat']), float(start['lon']))
    exact_end = (float(destination['lat']), float(destination['lon']))
    start_center = grid_lat_lon(*start_rc, meta)
    end_center = grid_lat_lon(*end_rc, meta)
    start_offset_m = _great_circle_m(exact_start, start_center)
    end_offset_m = _great_circle_m(exact_end, end_center)
    connector_m = start_offset_m + end_offset_m
    for name in ROUTE_STYLE:
        try:
            path, cost, expanded = _a_star(grid, risk, rough, traversable,
                                            *start_rc, *end_rc, dx, dy, config, name)
            stats = _route_stats(path, grid, slope, rough, risk, dx, dy, config)
            # The raster search starts at a cell center, but the user DID NOT
            # select that center. Always draw exactly the clicked endpoints.
            positions = [grid_lat_lon(int(r), int(c), meta) for r, c in path]
            positions = [exact_start, *positions, exact_end]
            _add_endpoint_links(stats, connector_m, config)
            results[name] = {
                'name': name, 'stats': stats, 'cost': round(cost, 2),
                'expanded_cells': expanded, 'coordinates': positions,
                'start_cell_center': start_center,
                'destination_cell_center': end_center,
                'endpoint_grid_offset_m': [round(start_offset_m, 1), round(end_offset_m, 1)],
            }
        except (ValueError, RuntimeError) as exc:
            errors[name] = str(exc)
    return {
        'site_id': site_id, 'routes': results, 'errors': errors,
        'resolution_m': round(math.sqrt(dx * dy), 1),
        'grid_shape': list(grid.shape), 'stride': factor,
        'valid_fraction': float(np.mean(traversable)),
        'scenario': {'dust': config.dust, 'temperature_c': config.temperature_c,
                     'illumination': config.illumination, 'speed_kmh': config.speed_kmh,
                     'max_eva_hours': config.max_eva_hours,
                     'max_slope_deg': config.max_slope_deg,
                     'roughness_limit_m': config.roughness_limit_m,
                     'map_margin_km': config.map_margin_km,
                     'balanced_risk_weight': config.balanced_risk_weight},
        'hazard': risk, 'slope_deg': slope, 'roughness_m': rough,
        'traversable': traversable,
        'blocked': np.isfinite(grid) & ~traversable,
        'cost_safest': cost_surface(slope, rough, risk, traversable, config, 'Safest Route'),
        'cost_balanced': cost_surface(slope, rough, risk, traversable, config, 'Balanced Route'),
        'endpoint_grid_offset_m': [round(start_offset_m, 1), round(end_offset_m, 1)],
        'map_metadata': meta,
        'summary': {
            'mean_slope_deg': round(float(np.nanmean(slope)), 2),
            'max_slope_deg': round(float(np.nanmax(slope)), 2),
            'valid_fraction': round(float(np.mean(traversable)), 3),
        },
    }
