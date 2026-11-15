"""Waypoint selection for a future Mars EVA route planner.

This module prepares start/end coordinates and map markers only.
It deliberately does NOT calculate traversability, risk or optimal routes.
"""

from __future__ import annotations

import html
import math

import folium


# Keep the same spherical approximation used by the current SWIM search.
# We can replace it with a common geographic model when planning routes.
MARS_RADIUS_KM = 3396.0


def _coordinates(lat, lon):
    """Validate and normalize a point reported by Leaflet."""
    lat = float(lat)
    lon = float(lon)
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90):
        raise ValueError("Invalid waypoint latitude or longitude")
    return round(lat, 6), round((lon + 180) % 360 - 180, 6)


def build_waypoint_catalog(site, search_results):
    """Catalog of clickable region and SWIM markers, for exact coordinates."""
    catalog = {
        "region": {
            "lat": float(site["lat"]),
            "lon": float(site["lon"]),
            "label": f'{site["name"]} · reference point',
            "source": "Region marker",
        }
    }
    for result in search_results:
        depth = result["depth"]
        depth_label = result["depth_label"]
        for point in result["points"]:
            # Grid indices provide an ID stable across candidate-list reordering.
            code = f'swim:{depth}:{point["grid_row"]}:{point["grid_col"]}'
            catalog[code] = {
                "lat": float(point["lat"]),
                "lon": float(point["lon"]),
                "label": f'SWIM {depth_label} · {point["id"]} · Ci {point["ci"]:.2f}',
                "source": "SWIM research candidate",
                "ci": float(point["ci"]),
                "depth": depth,
                "point_id": str(point["id"]),
            }
    return catalog


def decode_map_click(event, previous_marker_count, previous_map_click):
    """Return (point, is_marker, counter, last_map_click).

    The component remembers both last clicked coordinates and last marker.
    A changed marker click counter takes priority; otherwise use a new map
    click, so old marker positions never override a background click.
    """
    if not isinstance(event, dict):
        return None, False, previous_marker_count, previous_map_click

    count = event.get("last_object_clicked_count")
    marker = event.get("last_object_clicked")
    background = event.get("last_clicked")
    marker_new = (
        isinstance(marker, dict)
        and marker.get("lat") is not None
        and marker.get("lng") is not None
        and count is not None
        and count != previous_marker_count
    )
    background_new = (
        isinstance(background, dict)
        and background.get("lat") is not None
        and background.get("lng") is not None
        and background != previous_map_click
    )
    # Store newest observed signals even when we are in Browse mode.
    last_map_click = dict(background) if isinstance(background, dict) else previous_map_click
    last_count = count if count is not None else previous_marker_count

    if marker_new:
        try:
            coords = _coordinates(marker["lat"], marker["lng"])
        except (ValueError, TypeError):
            return None, False, last_count, last_map_click
        return coords, True, last_count, last_map_click

    if background_new:
        try:
            coords = _coordinates(background["lat"], background["lng"])
        except (ValueError, TypeError):
            return None, False, last_count, last_map_click
        return coords, False, last_count, last_map_click

    return None, False, last_count, last_map_click


def waypoint_from_click(lat, lon, is_marker, catalog, event, sites):
    """Resolve the exact marker coordinates, not the cursor position.

    Leaflet may report the cursor's position inside a CircleMarker rather
    than its center, so the tooltip identity is more reliable than comparing
    floating-point coordinates alone.
    """
    lat, lon = _coordinates(lat, lon)
    if is_marker:
        raw_tooltip = str(event.get("last_object_clicked_tooltip") or "")
        tooltip = html.unescape(raw_tooltip)

        # Prefer a matching SWIM ID AND depth. IDs restart at SWIM-01 per depth.
        for entry in catalog.values():
            depth = entry.get("depth")
            if not depth:
                continue
            depth_label = "0–1 m" if depth == "Ci_0_1" else "1–5 m"
            if entry["point_id"] in tooltip and depth_label in tooltip:
                return dict(entry)

        # A region marker also has a descriptive scientific hover.
        for site in sites.values():
            if str(site["name"]) in tooltip:
                return {
                    "lat": float(site["lat"]),
                    "lon": float(site["lon"]),
                    "label": str(site["name"]),
                    "source": "Region marker",
                }

        # Fallback for clients that do not provide tooltip contents.
        for entry in catalog.values():
            if abs(entry["lat"] - lat) < 0.005 and abs(entry["lon"] - lon) < 0.005:
                return dict(entry)

    return {
        "lat": lat,
        "lon": lon,
        "label": "Map-selected coordinates",
        "source": "Map click",
    }


def straight_line_distance_km(start, end):
    """Great-circle distance; NOT a terrain-following route length."""
    lat1, lon1 = map(math.radians, _coordinates(start["lat"], start["lon"]))
    lat2, lon2 = map(math.radians, _coordinates(end["lat"], end["lon"]))
    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + math.pi) % (2 * math.pi) - math.pi
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * MARS_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, hav))))


def create_waypoint_markers(start, destination):
    """Add Start and Destination on top of research candidates."""
    group = folium.FeatureGroup(name="EVA Start and Destination", control=False)

    # Show a dashed *visual reference*, NOT a computed or safe EVA route.
    # Add the line before waypoints so the markers remain clearly visible.
    if start is not None and destination is not None:
        a_lat, a_lon = _coordinates(start["lat"], start["lon"])
        b_lat, b_lon = _coordinates(destination["lat"], destination["lon"])
        if (a_lat, a_lon) != (b_lat, b_lon):
            folium.PolyLine(
                locations=[[a_lat, a_lon], [b_lat, b_lon]],
                color="#E8EDF2",
                weight=3,
                opacity=0.95,
                dash_array="2, 8",
                line_cap="round",
                interactive=False,
            ).add_to(group)

    for role, point, color in (
        ("START", start, "#34D399"),
        ("DESTINATION", destination, "#FB7185"),
    ):
        if point is None:
            continue
        lat, lon = _coordinates(point["lat"], point["lon"])
        label = html.escape(str(point.get("label", role)))
        tooltip = folium.Tooltip(
            '<div style="font-family:Arial,sans-serif;color:#162331;'
            'line-height:1.45;min-width:170px">'
            f'<b>{role}</b><br>{label}<br>'
            f'{lat:.5f}°, {lon:.5f}°'
            '</div>',
            sticky=True,
        )
        folium.CircleMarker(
            location=[lat, lon], radius=11, color="#FFFFFF",
            weight=3, fill=True, fill_color=color, fill_opacity=1.0,
            bubbling_mouse_events=False, tooltip=tooltip,
        ).add_to(group)
    return group
