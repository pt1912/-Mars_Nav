"""Streamlit EVA route comparison panel and optional Folium map overlays."""
from __future__ import annotations

import io
import base64
import json
import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from modules.mola_dem import PROCESSED_DIR, grid_lat_lon, load_region_dem, point_rc
from modules.eva_route_engine import ROUTE_STYLE, RouteSettings, plan_eva_routes


def _same_endpoints(plan, site_id, a, b):
    return bool(plan and a and b and plan.get('site_id') == site_id and
                plan.get('start') == (a['lat'], a['lon']) and
                plan.get('destination') == (b['lat'], b['lon']))


def _hazard_overlay(plan):
    h = plan['hazard']
    good = np.isfinite(h)
    t = np.clip(np.nan_to_num(h, nan=0), 0, 1)
    # Warm hazardous / green low-hazard palette. Invalid pixels fully transparent.
    rgb = np.empty((*h.shape, 4), dtype=np.uint8)
    rgb[..., 0] = np.where(t < 0.5, 55 + 310 * t, 210 + 90 * (t - 0.5))
    rgb[..., 1] = np.where(t < 0.5, 190 - 55 * t, 165 - 320 * (t - 0.5))
    rgb[..., 2] = np.where(t < 0.5, 85 - 80 * t, 15)
    rgb[..., 3] = np.where(good, 210, 0)
    img = Image.fromarray(rgb, mode='RGBA')
    if max(img.size) > 700:
        ratio = 700 / max(img.size)
        img = img.resize((max(1, round(img.width * ratio)),
                          max(1, round(img.height * ratio))), resample=Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    m = plan['map_metadata']
    ppd = float(m['resolution_ppd'])
    north = float(m['north_edge'])
    west = float(m['west_edge_east'])
    south = north - h.shape[0] / ppd
    east = west + h.shape[1] / ppd
    # Planning window is strictly inside a single MOLA tile; geographic wrapping
    # only for the east-longitude to -180...180 Leaflet convention.
    lon_w = ((west + 180) % 360) - 180
    lon_e = ((east + 180) % 360) - 180
    if lon_w > lon_e:
        return None  # avoid drawing a bogus dateline-spanning overlay
    return folium.raster_layers.ImageOverlay(
        image='data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii'), bounds=[[south, lon_w], [north, lon_e]],
        opacity=0.58, name='MOLA terrain hazard (experimental)',
        interactive=False, cross_origin=False, zindex=6,
    )


def create_eva_route_overlay(plan, show_hazard=False, visible=None):
    group = folium.FeatureGroup(name='EVA Route Alternatives', control=False)
    if not plan:
        return group
    if show_hazard:
        hazard = _hazard_overlay(plan)
        if hazard is not None:
            hazard.add_to(group)
    for name, route in plan.get('routes', {}).items():
        if visible is not None and name not in visible:
            continue
        style = ROUTE_STYLE[name]
        coords = route['coordinates']
        if len(coords) < 2:
            continue
        folium.PolyLine(
            locations=coords, color=style['color'], weight=5, opacity=0.94,
            dash_array=style['dash'], interactive=False,
            tooltip=f'{name} · {route["stats"]["distance_km"]:.2f} km',
        ).add_to(group)
    return group


def _rank_routes(routes):
    eligible = [(name, r['stats']) for name, r in routes.items()
                if r['stats']['within_eva_time']]
    if not eligible:
        return None, None
    keys = [('distance_km', 0.32), ('mean_terrain_risk', 0.38),
            ('duration_hours', 0.20), ('effort_proxy', 0.10)]
    metrics = {}
    for field, weight in keys:
        numbers = [stats[field] for _, stats in eligible]
        low, high = min(numbers), max(numbers)
        metrics[field] = [(val - low) / (high - low) if high > low else 0
                          for val in numbers]
    scores = {}
    for index, (name, _) in enumerate(eligible):
        scores[name] = sum(weight * metrics[field][index] for field, weight in keys)
    best = min(scores, key=scores.get)
    return best, scores


def _route_result_table(routes):
    rows = []
    for name, r in routes.items():
        x = r['stats']
        rows.append({
            'Route': name.replace(' Route', ''),
            'Distance (km)': x['distance_km'],
            'Maximum slope (°)': x['maximum_slope_deg'],
            'Ascent (m)': x['ascent_m'],
            'Terrain risk (0–1)': x['mean_terrain_risk'],
            'EVA duration (h)': x['duration_hours'],
            'Environmental exposure (idx·h)': x['environmental_exposure_index_hours'],
            'Effort proxy': x['effort_proxy'],
            'Within EVA time': x['within_eva_time'],
        })
    return pd.DataFrame(rows)


def render_eva_route_panel(site_id, start, destination):
    st.divider()
    st.subheader('EVA Route Analysis')
    st.caption('MOLA regional corridor planning · Shortest / Safest / Balanced. '
               'Not validated for astronaut navigation or local obstacles.')
    if site_id is None:
        return
    dem_path = PROCESSED_DIR / f'{site_id}_mola.npz'
    if not dem_path.exists():
        st.info(f'No MOLA regional cutout for this region ({site_id}). '
                'Generate it locally with: `python scripts/build_mola_regions.py`. '
                'Additional source IMG files are required outside the available MOLA tile.')
        return
    try:
        dem, meta = load_region_dem(site_id)
    except (OSError, KeyError, ValueError) as e:
        st.error(f'Could not load region DEM: {e}')
        return
    north, south = meta['north_edge'], meta['south_edge']
    west = ((meta['west_edge_east'] + 180) % 360) - 180
    east = ((meta['east_edge_east'] + 180) % 360) - 180
    st.caption(f'DEM: {dem.shape[1]} × {dem.shape[0]} cells · 128 px/degree · '
               f'lat {south:.2f}°–{north:.2f}° · lon {west:.2f}°–{east:.2f}°')
    if meta.get('partial_source_coverage'):
        st.warning('Partial DEM coverage: the available source tile stops at its geographic boundary. '
                   'Routes cannot extend into missing areas.')
    if start is None or destination is None:
        st.info('Choose **Set start** and **Set destination** above to activate routing.')
        return
    try:
        point_rc(start['lat'], start['lon'], meta, dem.shape)
        point_rc(destination['lat'], destination['lon'], meta, dem.shape)
    except ValueError as exc:
        st.warning(f'Routing unavailable: {exc}. Select waypoints inside the DEM cutout.')
        return

    st.markdown('**Terrain constraints**')
    p1, p2, p3 = st.columns(3)
    with p1:
        max_slope = st.slider('Max modeled slope (°)', 3, 35, 15, 1,
                              key='route_max_slope')
    with p2:
        rough_limit = st.slider('Roughness reference (m)', 3, 80, 20, 1,
                                key='route_rough_limit')
    with p3:
        margin = st.slider('Detour margin (km)', 5, 80, 30, 5,
                           key='route_margin_km')

    st.markdown('**Mission / environment scenario**')
    st.caption('Scenario values are manually chosen assumptions, not live Mars weather.')
    q1, q2, q3 = st.columns(3)
    with q1:
        dust = st.selectbox('Dust conditions', ['Clear', 'Dusty', 'Severe dust'],
                            key='route_dust')
        temperature = st.slider('Ambient temperature (°C)', -120, 10, -50, 5,
                                 key='route_temp')
    with q2:
        lighting = st.selectbox('Illumination', ['Daylight', 'Low Sun', 'Dark'],
                                key='route_light')
        speed = st.slider('Assumed EVA speed (km/h)', 0.5, 5.0, 2.0, 0.5,
                          key='route_speed')
    with q3:
        max_time = st.slider('Max EVA travel time (h)', 1.0, 12.0, 8.0, 0.5,
                             key='route_max_time')
        balanced_weight = st.slider('Balanced risk preference', 0.2, 3.0, 1.0, 0.2,
                                    key='route_balanced_weight')

    if st.button('Calculate 3 EVA routes', type='primary', use_container_width=True,
                 key='calculate_eva_routes'):
        settings = RouteSettings(
            max_slope_deg=float(max_slope), roughness_limit_m=float(rough_limit),
            map_margin_km=float(margin), speed_kmh=float(speed),
            max_eva_hours=float(max_time), dust=dust, temperature_c=float(temperature),
            illumination=lighting, balanced_risk_weight=float(balanced_weight),
        )
        with st.spinner('Analyzing terrain and optimizing three routes…'):
            try:
                result = plan_eva_routes(site_id, start, destination, settings)
                result['start'] = (start['lat'], start['lon'])
                result['destination'] = (destination['lat'], destination['lon'])
                st.session_state.eva_route_plan = result
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
                st.session_state.eva_route_plan = None
                st.error(str(exc))
            else:
                st.rerun()

    plan = st.session_state.get('eva_route_plan')
    if not _same_endpoints(plan, site_id, start, destination):
        return
    st.markdown('**Calculated routes**')
    r1, r2, r3 = st.columns(3)
    for container, route_name in zip((r1, r2, r3), ROUTE_STYLE):
        with container:
            st.checkbox('Show ' + route_name, value=True, key='eva_show_' + route_name.replace(' ', '_'))
    st.checkbox('Terrain hazard overlay', value=False, key='eva_show_hazard')
    st.caption('Map paths use different styles. Activate the terrain hazard layer to examine the DEM-derived cost surface.')
    if st.button('Focus on route area', key='eva_focus_area'):
        st.session_state.map_center = (
            (float(start['lat']) + float(destination['lat'])) / 2,
            (float(start['lon']) + float(destination['lon'])) / 2,
        )
        st.session_state.map_zoom = 7
        st.rerun()
    st.metric('Planning grid resolution', f"~{plan['resolution_m']:.0f} m")
    st.caption(f"Search grid: {plan['grid_shape'][0]} × {plan['grid_shape'][1]} cells · "
               f"Valid terrain: {plan['valid_fraction']:.1%} · "
               f"Resampling factor: {plan['stride']}×")

    if plan.get('routes'):
        table = _route_result_table(plan['routes'])
        st.dataframe(table, hide_index=True, use_container_width=True)
        profiles = {}
        for name, route in plan['routes'].items():
            heights = route['stats']['elevation_profile_m']
            if len(heights) >= 2:
                profiles[name] = np.interp(np.linspace(0, 1, 100),
                                            np.linspace(0, 1, len(heights)), heights)
        if profiles:
            st.markdown('**Elevation profiles**')
            st.line_chart(pd.DataFrame(profiles).rename_axis('Route progress (normalized)'),
                          height=230)
        chosen, scores = _rank_routes(plan['routes'])
        st.markdown('**Route Assessment**')
        if chosen is None:
            st.warning('No route satisfies the configured EVA travel-time limit. '
                       'Adjust mission constraints; no route is recommended.')
        else:
            distances = [r['stats']['distance_km'] for r in plan['routes'].values()]
            st.success(f'Recommended under the illustrative multicriteria scoring: **{chosen}**')
            st.caption('Comparison weights: terrain risk 38%, distance 32%, '
                       'estimated duration 20%, effort proxy 10%. '
                       'Routes failing the EVA time limit are not eligible. '
                       'These weights are research assumptions.')
            if max(distances) - min(distances) < 0.01:
                st.caption('The alternatives nearly coincide at MOLA resolution. '
                           'This is a limitation of coarse topography, not proof of equal EVA safety.')
        st.caption('Hazards modeled here: slope, local relief variability, uphill effort, '
                   'travel-time exposure under user-specified dust / temperature / illumination. '
                   'CTX imagery is visual context only; rock-scale hazards, actual storms, '
                   'radiation and communications have NOT been measured or modeled.')
        # Only small route coordinates and metrics are exported, not DEM arrays.
        serializable = {
            'site': site_id, 'scenario': plan['scenario'], 'source': meta.get('source_img'),
            'resolution_m': plan['resolution_m'],
            'routes': {name: {k:v for k,v in item.items() if k != 'expanded_cells'}
                       for name, item in plan['routes'].items()},
            'errors': plan.get('errors', {}),
        }
        st.download_button('Export route assessment (JSON)',
                           json.dumps(serializable, indent=2),
                           file_name=f'mars_eva_{site_id}_routes.json',
                           mime='application/json', key='eva_export_json')
    for name, msg in plan.get('errors', {}).items():
        st.warning(f'{name}: {msg}')
