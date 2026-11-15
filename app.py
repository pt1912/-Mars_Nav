import html
import json
import re
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

from modules.swim_global import load_swim_global
from modules.ice_targets import find_ice_candidates
from modules.usgs_overlay import USGSGeologyOverlay
from modules.nasa_tiles import VIKING_WMTS, MOLA_WMTS, get_ctx_tiles
from modules.mars_theme import apply_theme

BASE_DIR = Path(__file__).resolve().parent
SITES_FILE = BASE_DIR / "data" / "locations" / "mars_sites.json"

GLOBAL_ID = "global"
GLOBAL_CENTER = (0.0, 0.0)
GLOBAL_ZOOM = 1
REGION_ZOOM = 5

CATEGORY_COLORS = {
    "Subsurface ice": "#38bdf8",
    "Robotic exploration": "#22C55E",
    "Terrain and geology": "#fb923c",
}
POI_COLORS = {"Ci_0_1": "#E879F9", "Ci_1_5": "#FDE047"}

st.set_page_config(
    page_title="ARCADIA ICE EXPLORER",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


@st.cache_data(show_spinner=False)
def load_sites():
    if not SITES_FILE.is_file():
        raise FileNotFoundError(f"Site catalog not found: {SITES_FILE}")
    entries = json.loads(SITES_FILE.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("The site catalog must be a non-empty JSON array.")

    sites = {}
    required = ("id", "name", "lat", "lon", "category", "summary", "focus")
    for item in entries:
        if not all(k in item for k in required):
            raise ValueError(f"Invalid catalog entry: {item!r}")
        site = dict(item)
        site_id = str(site["id"])
        if site_id in sites or site_id == GLOBAL_ID:
            raise ValueError(f"Duplicate or reserved site ID: {site_id}")
        site["lat"] = float(site["lat"])
        site["lon"] = float(site["lon"])
        if not (-90 <= site["lat"] <= 90 and -180 <= site["lon"] <= 180):
            raise ValueError(f"Invalid coordinates for {site_id}")
        sites[site_id] = site
    return sites


SITES = load_sites()


def select_site(site_id):
    if site_id != GLOBAL_ID and site_id not in SITES:
        raise ValueError(f"Unknown site: {site_id}")
    if st.session_state.selected_site != site_id:
        # Ice controls belong to the selected region, not the planetary view.
        st.session_state.show_swim_0_1 = False
        st.session_state.show_swim_1_5 = False
    st.session_state.selected_site = site_id
    if site_id == GLOBAL_ID:
        st.session_state.nav_center = GLOBAL_CENTER
        st.session_state.nav_zoom = GLOBAL_ZOOM
    else:
        site = SITES[site_id]
        st.session_state.nav_center = (site["lat"], site["lon"])
        st.session_state.nav_zoom = REGION_ZOOM
    st.session_state.map_revision += 1


def on_region_change():
    if st.session_state.site_selector is not None:
        select_site(st.session_state.site_selector)


def on_global_view():
    st.session_state.site_selector = None
    select_site(GLOBAL_ID)


def tooltip_for(site):
    lat, lon = site["lat"], site["lon"]
    name, category = html.escape(site["name"]), html.escape(site["category"])
    summary, focus = html.escape(site["summary"]), html.escape(site["focus"])
    return folium.Tooltip(
        '<div style="width:300px;white-space:normal;line-height:1.45;'
        'font-family:Arial,sans-serif;font-size:12px">'
        f'<b style="font-size:14px">{name}</b><br>'
        f'<span style="color:#526378">{category} · '
        f'{abs(lat):.2f}°{"N" if lat >= 0 else "S"}, '
        f'{abs(lon):.2f}°{"E" if lon >= 0 else "W"}</span><br>'
        f'{summary}<br><b>Analysis:</b> {focus}</div>',
        sticky=True,
    )


def resolve_clicked_site(map_data):
    if not map_data:
        return None
    point = map_data.get("last_object_clicked")
    if isinstance(point, dict) and "lat" in point and "lng" in point:
        try:
            lat, lon = float(point["lat"]), float(point["lng"])
            for sid, site in SITES.items():
                if abs(lat - site["lat"]) < 0.0001 and abs(lon - site["lon"]) < 0.0001:
                    return sid
        except (TypeError, ValueError):
            pass
    raw = map_data.get("last_object_clicked_tooltip")
    if isinstance(raw, str):
        plain = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        for sid, site in SITES.items():
            if site["name"] in plain:
                return sid
    return None


# Session state
for key, initial in {
    "map_revision": 0,
    "selected_site": GLOBAL_ID,
    "nav_center": GLOBAL_CENTER,
    "nav_zoom": GLOBAL_ZOOM,
}.items():
    if key not in st.session_state:
        st.session_state[key] = initial

if st.session_state.get("site_selector") == GLOBAL_ID:
    st.session_state.site_selector = None
if "pending_site" in st.session_state:
    pending = st.session_state.pop("pending_site")
    if pending in SITES:
        st.session_state.site_selector = pending
        select_site(pending)

st.title("ARCADIA ICE EXPLORER")

# Sidebar
with st.sidebar:
    st.header("Mission Control")
    st.subheader("Explore Mars")
    st.selectbox(
        "Region", options=list(SITES), index=None,
        placeholder="Select a region", format_func=lambda sid: SITES[sid]["name"],
        key="site_selector", on_change=on_region_change,
    )
    st.button("🌐 Global Map", use_container_width=True, on_click=on_global_view)

    st.divider()
    st.subheader("Map Layers")
    show_ctx = st.checkbox("MRO CTX imagery", value=False, key="show_ctx")
    show_mola = st.checkbox("MOLA shaded relief", value=False, key="show_mola")
    show_geology = st.checkbox("USGS SIM 3292 geology", value=False, key="show_geology")
    geology_opacity = (
        st.slider("Geology opacity", 0.10, 0.60, 0.25, 0.05, key="geology_opacity")
        if show_geology else 0.25
    )

    # Not shown or rendered until a region is selected.
    region = st.session_state.selected_site
    show_swim_0_1 = False
    show_swim_1_5 = False
    swim_opacity = 0.65
    if region != GLOBAL_ID:
        st.divider()
        st.subheader("Ice Overlays")
        show_swim_0_1 = st.checkbox(
            "SWIM ice consistency (0–1 m)", value=False, key="show_swim_0_1"
        )
        show_swim_1_5 = st.checkbox(
            "SWIM ice consistency (1–5 m)", value=False, key="show_swim_1_5"
        )
        if show_swim_0_1 or show_swim_1_5:
            swim_opacity = st.slider(
                "SWIM opacity", 0.20, 1.00, 0.65, 0.05, key="swim_opacity"
            )
            st.markdown("**Ice Target Search**")
            ice_search_radius = st.slider(
                "Search radius (km)", 50, 500, 200, 25, key="ice_search_radius"
            )
            ice_min_ci = st.slider(
                "Minimum ice consistency (Ci)", 0.0, 0.85, 0.50, 0.05,
                key="ice_min_ci",
            )
            ice_max_points = st.slider(
                "Maximum targets", 1, 15, 10, 1, key="ice_max_points"
            )

map_col, info_col = st.columns([3.6, 1], gap="medium")
active_swim = []
poi_stats = []
render_pois = []

with map_col:
    st.subheader("Mars Surface Map")
    mars_map = folium.Map(
        location=list(st.session_state.nav_center),
        zoom_start=st.session_state.nav_zoom,
        tiles=None, crs="EPSG4326", zoom_snap=0.5,
        min_zoom=0, max_zoom=12, control_scale=False,
        prefer_canvas=True, zoom_control=True,
    )

    folium.TileLayer(
        tiles=VIKING_WMTS, name="NASA Viking | global mosaic",
        attr="NASA Mars Trek / Viking Orbiter",
        overlay=False, control=False, show=True,
        min_zoom=0, max_zoom=12, max_native_zoom=7, no_wrap=True,
    ).add_to(mars_map)

    if show_mola:
        folium.TileLayer(
            tiles=MOLA_WMTS, name="MGS MOLA | shaded relief",
            attr="NASA Mars Trek / MGS MOLA", opacity=0.72,
            overlay=True, control=False, show=True,
            min_zoom=0, max_zoom=12, max_native_zoom=8, no_wrap=True,
        ).add_to(mars_map)

    if show_ctx:
        ctx_tiles, ctx_id, ctx_error = get_ctx_tiles(region)
        if ctx_tiles:
            folium.TileLayer(
                tiles=ctx_tiles, name="MRO CTX | imagery",
                attr=f"NASA Mars Trek / MRO CTX ({ctx_id})",
                opacity=0.90, overlay=True, control=False, show=True,
                min_zoom=0, max_zoom=12, max_native_zoom=12, no_wrap=True,
            ).add_to(mars_map)
        if ctx_error:
            st.sidebar.warning(ctx_error)

    if show_geology:
        USGSGeologyOverlay(opacity=geology_opacity).add_to(mars_map)

    for enabled, depth, label in (
        (show_swim_0_1, "Ci_0_1", "0–1 m"),
        (show_swim_1_5, "Ci_1_5", "1–5 m"),
    ):
        if not enabled:
            continue
        try:
            rgba, bounds, info = load_swim_global(depth)
            folium.raster_layers.ImageOverlay(
                image=rgba, bounds=bounds,
                name=f"SWIM4MIM | Ci, {label}",
                opacity=swim_opacity, interactive=False,
                pixelated=True, control=False,
            ).add_to(mars_map)
            active_swim.append(label)
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            st.sidebar.warning(f"SWIM {label} unavailable: {exc}")

        # Search the selected region, using the GLOBAL SWIM raster.
        # No Arcadia-specific JSON or precomputed list is required.
        try:
            site = SITES[region]
            search = find_ice_candidates(
                site["lat"], site["lon"],
                depth=depth,
                radius_km=ice_search_radius,
                min_ci=ice_min_ci,
                max_points=ice_max_points,
                min_separation_km=30.0,
            )
            poi_stats.append((label, search))
            render_pois.extend(
                (poi, depth, label) for poi in search["points"]
            )
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            st.sidebar.warning(f"SWIM {label} target search unavailable: {exc}")

    # Add scientific markers AFTER all raster overlays, so neither depth hides them.
    for poi, depth, label in render_pois:
        folium.CircleMarker(
            location=[poi["lat"], poi["lon"]],
            radius=5, color="#111827", weight=1.5,
            fill=True, fill_color=POI_COLORS[depth], fill_opacity=0.95,
            tooltip=folium.Tooltip(
                f'<b>{html.escape(poi["id"])}</b><br>'
                f'Depth: {label}<br>'
                f'Ice consistency Ci: {poi["ci"]:.2f}<br>'
                f'Distance from region center: {poi["distance_km"]:.1f} km<br>'
                'Coarse SWIM grid cell; research candidate, not confirmed ice.',
                sticky=True,
            ),
        ).add_to(mars_map)

    # Permanent map landmarks remain visible above every scientific overlay.
    for site in SITES.values():
        folium.CircleMarker(
            location=[site["lat"], site["lon"]], radius=7,
            color="#172033", weight=2, fill=True,
            fill_color=CATEGORY_COLORS.get(site["category"].strip(), "#94a3b8"),
            fill_opacity=0.96, tooltip=tooltip_for(site),
        ).add_to(mars_map)

    map_data = st_folium(
        mars_map, key=f"mars_global_v2_{st.session_state.map_revision}",
        height=610, use_container_width=True,
        returned_objects=[
            "last_object_clicked", "last_object_clicked_tooltip", "center", "zoom"
        ],
    )
    if map_data:
        center = map_data.get("center")
        zoom = map_data.get("zoom")
        if isinstance(center, dict) and "lat" in center and "lng" in center:
            st.session_state.nav_center = (float(center["lat"]), float(center["lng"]))
        if isinstance(zoom, (int, float)):
            st.session_state.nav_zoom = zoom

    clicked_site = resolve_clicked_site(map_data)
    if clicked_site is not None and clicked_site != st.session_state.selected_site:
        st.session_state.pending_site = clicked_site
        st.rerun()

with info_col:
    st.subheader("Region")
    selected = st.session_state.selected_site
    if selected == GLOBAL_ID:
        st.write("Select a marker or choose a region.")
    else:
        site = SITES[selected]
        st.markdown(f"**{site['name']}**")
        st.write(site["category"])
        st.write(f"Lat {site['lat']:.2f}° · Lon {site['lon']:.2f}°")

    st.divider()
    st.markdown("**Map key**")
    for category, color in CATEGORY_COLORS.items():
        st.markdown(
            f'<span style="color:{color};font-size:16px">●</span> '
            f'{html.escape(category)}', unsafe_allow_html=True,
        )
    if active_swim:
        st.divider()
        st.markdown("**SWIM4MIM · Ice consistency (Ci)**")
        st.markdown(
            '<div style="height:10px;border-radius:3px;'
            'background:linear-gradient(90deg,'
            '#ffffd9,#c7e9b4,#7fcdbb,#41b6c4,#2c7fb8,#253494)"></div>',
            unsafe_allow_html=True,
        )
        st.caption("0 → 0.85 · Higher Ci means stronger ice consistency; not probability.")
        for depth_label, stats in poi_stats:
            if stats["valid_cells"] == 0:
                st.caption(f"{depth_label}: No SWIM data within search radius.")
            elif stats["eligible_cells"] == 0:
                peak = stats["max_ci_in_radius"]
                st.caption(
                    f"{depth_label}: No cells meet Ci ≥ {stats['min_ci']:.2f} "
                    f"(regional maximum: {peak:.2f})."
                )
            else:
                st.caption(
                    f"{depth_label}: {len(stats['points'])} research candidates "
                    f"· {stats['eligible_cells']} eligible cells "
                    f"· regional max Ci {stats['max_ci_in_radius']:.2f}"
                )
        st.caption(
            "Site-centered regional search · coarse 0.2° grid · "
            "Ci is not a probability or a verified ice detection."
        )
