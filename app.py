import html

import streamlit as st
from streamlit_folium import st_folium

from modules.mars_map import (
    CATEGORY_COLORS,
    ICE_TARGET_COLORS,
    GLOBAL_CENTER,
    GLOBAL_ZOOM,
    REGION_ZOOM,
    create_mars_map,
    create_ice_target_markers,
    create_site_markers,
    load_sites,
    site_from_map_event,
)
from modules.mars_theme import apply_theme
from modules.ice_targets import find_ice_candidates
from modules.eva_waypoints import (
    build_waypoint_catalog,
    create_waypoint_markers,
    decode_map_click,
    straight_line_distance_km,
    waypoint_from_click,
)
from modules.swim_overlays import create_swim_overlay_group, load_swim_overlay
from modules.mars_layers import GLOBAL_LAYERS, create_selected_overlays
from modules.eva_route_ui import render_eva_route_panel, create_eva_route_overlay
from modules.eva_route_engine import ROUTE_STYLE


st.set_page_config(
    page_title="MARS ICE EXPLORER",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

SITES = load_sites()

if "region_selector" not in st.session_state:
    st.session_state.region_selector = None
if "map_center" not in st.session_state:
    st.session_state.map_center = GLOBAL_CENTER
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = GLOBAL_ZOOM

# EVA waypoint state: kept separate from the regional research candidates.
for key, default in (
    ("eva_start", None),
    ("eva_destination", None),
    ("eva_click_mode", "Browse"),
    ("eva_catalog", {}),
    ("eva_last_marker_count", None),
    ("eva_last_map_click", None),
    ("eva_route_plan", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


def reset_eva_waypoints():
    """Do not carry start/end coordinates into another study region."""
    st.session_state.eva_start = None
    st.session_state.eva_destination = None
    st.session_state.eva_click_mode = "Browse"
    st.session_state.eva_catalog = {}
    st.session_state.eva_route_plan = None


def navigate_to(site_id):
    if site_id is None:
        st.session_state.map_center = GLOBAL_CENTER
        st.session_state.map_zoom = GLOBAL_ZOOM
    else:
        site = SITES[site_id]
        st.session_state.map_center = (site["lat"], site["lon"])
        st.session_state.map_zoom = REGION_ZOOM


def on_region_change():
    reset_eva_waypoints()
    navigate_to(st.session_state.region_selector)


def on_global_map():
    reset_eva_waypoints()
    st.session_state.region_selector = None
    navigate_to(None)


def clear_eva_point(role):
    st.session_state[f"eva_{role}"] = None
    st.session_state.eva_route_plan = None


def on_marker_click():
    """Handle map clicks without listening to zoom or map movement."""
    event = st.session_state.get("mars_surface_map")
    position, is_marker, counter, last_map_click = decode_map_click(
        event,
        st.session_state.eva_last_marker_count,
        st.session_state.eva_last_map_click,
    )
    st.session_state.eva_last_marker_count = counter
    st.session_state.eva_last_map_click = last_map_click
    if position is None:
        return

    selected_region = st.session_state.region_selector
    mode = st.session_state.eva_click_mode
    if selected_region is not None and mode in ("Set start", "Set destination"):
        role = "start" if mode == "Set start" else "destination"
        waypoint = waypoint_from_click(
            position[0], position[1], is_marker,
            st.session_state.eva_catalog, event, SITES,
        )
        st.session_state[f"eva_{role}"] = waypoint
        st.session_state.eva_route_plan = None
        # Return to normal navigation after selecting one point.
        st.session_state.eva_click_mode = "Browse"
        return

    # Normal mode: clicking a region marker changes the selected region.
    if is_marker:
        site_id = site_from_map_event(event, SITES)
        if site_id is not None and site_id != selected_region:
            reset_eva_waypoints()
            st.session_state.region_selector = site_id
            navigate_to(site_id)


with st.sidebar:
    st.markdown("### MISSION CONTROL")
    st.caption("MARS ICE EXPLORER")
    st.divider()
    st.subheader("Explore Mars")
    st.selectbox(
        "Region",
        options=list(SITES),
        index=None,
        format_func=lambda site_id: SITES[site_id]["name"],
        placeholder="Select a region",
        key="region_selector",
        on_change=on_region_change,
    )
    st.button("🌐 Global Map", use_container_width=True, on_click=on_global_map)

    st.divider()
    st.subheader("Map Layers")

    selected_layers = []
    layer_categories = (
        ("Topography", "Terrain"),
        ("Thermal", "Thermal"),
        ("Surface", "Surface"),
        ("Atmosphere / Dust (historical)", "Dust (historic)"),
        ("Composition", "Composition"),
    )
    for group_title, prefix in layer_categories:
        with st.expander(group_title, expanded=(prefix == "Terrain")):
            for name, layer_id, _opacity, _native_zoom in GLOBAL_LAYERS:
                category, _, label = name.partition(" | ")
                if category == prefix and st.checkbox(
                    label, key=f"layer_{layer_id}"
                ):
                    selected_layers.append(layer_id)

    # Global CTX now belongs to the Surface category above. No per-region
    # CTX controls or dynamically loaded regional mosaics are needed.
    show_geology = st.checkbox(
        "USGS SIM 3292 geology", key="show_usgs_geology"
    )
    geology_opacity = 0.35
    if show_geology:
        geology_opacity = st.slider(
            "USGS opacity", 0.0, 1.0, 0.35, 0.05,
            key="usgs_opacity",
        )

    # -------------------------------------------------
    # ICE OVERLAYS — SWIM4MIM composite Ci
    # -------------------------------------------------
    st.divider()
    st.subheader("Ice Overlays")
    show_swim_0_1 = st.checkbox(
        "SWIM ice consistency (0–1 m)", key="show_swim_0_1"
    )
    show_swim_1_5 = st.checkbox(
        "SWIM ice consistency (1–5 m)", key="show_swim_1_5"
    )
    swim_opacity = 0.55
    swim_enabled = show_swim_0_1 or show_swim_1_5
    if swim_enabled:
        swim_opacity = st.slider(
            "SWIM opacity", 0.20, 1.00, 0.55, 0.05,
            key="swim_opacity",
        )

    # Search controls are conditional on BOTH a region and a SWIM layer.
    search_enabled = swim_enabled and st.session_state.region_selector is not None
    if search_enabled:
        st.markdown("**Ice Target Search**")
        ice_search_radius = st.slider(
            "Search radius (km)", 50, 500, 200, 25,
            key="ice_search_radius",
        )
        ice_min_ci = st.slider(
            "Minimum ice consistency (Ci)", 0.0, 1.0, 0.50, 0.05,
            key="ice_min_ci",
        )
        ice_max_points = st.slider(
            "Maximum targets per depth", 1, 20, 10, 1,
            key="ice_max_points",
        )
        st.caption("Target spacing: 30 km · Grid: 0.2°")
    elif swim_enabled:
        st.caption("Select a region to search for scientific targets.")

with st.container(key="mars_intro"):
    st.html('<div class="mars-label">PLANETARY EXPLORATION SYSTEM</div>')
    st.title("MARS ICE EXPLORER")
    st.markdown(
        "Geospatial Platform designed for path planning and "
        "scientific data integration for Mars Ice exploration."
    )
    st.html('<div class="mars-separator"></div>')

# Perform numeric SWIM searches only when their controls are visible.
# The search module caches both raster grids and query results.
search_results = []
if search_enabled:
    selected_site = SITES[st.session_state.region_selector]
    for active, depth in (
        (show_swim_0_1, "Ci_0_1"),
        (show_swim_1_5, "Ci_1_5"),
    ):
        if not active:
            continue
        try:
            search_results.append(
                find_ice_candidates(
                    selected_site["lat"],
                    selected_site["lon"],
                    depth=depth,
                    radius_km=ice_search_radius,
                    min_ci=ice_min_ci,
                    max_points=ice_max_points,
                    min_separation_km=30.0,
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.sidebar.warning(f"SWIM {depth} target search unavailable: {exc}")

# The catalog resolves scientific marker clicks to their exact coordinates.
# SWIM entries appear only when their corresponding ice overlay is enabled.
if st.session_state.region_selector is not None:
    st.session_state.eva_catalog = build_waypoint_catalog(
        SITES[st.session_state.region_selector], search_results
    )
else:
    st.session_state.eva_catalog = {}

map_col, info_col = st.columns([3.6, 1], gap="medium")

with map_col:
    st.subheader("Mars Surface Map")
    groups = []
    groups.append(
        create_selected_overlays(
            selected_layers=selected_layers,
            geology=show_geology,
            geology_opacity=geology_opacity,
        )
    )
    # SWIM is kept separate from the static Viking map. PNG encoding is
    # cached in modules/swim_overlays.py; toggling does not reprocess i8.
    if show_swim_0_1 or show_swim_1_5:
        try:
            groups.append(
                create_swim_overlay_group(
                    show_0_1=show_swim_0_1,
                    show_1_5=show_swim_1_5,
                    opacity=swim_opacity,
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.sidebar.error(f"Unable to load SWIM overlay: {exc}")

    if search_results:
        groups.append(create_ice_target_markers(search_results))
    # Keep region markers above SWIM candidates in the drawing order.
    groups.append(create_site_markers(SITES, st.session_state.region_selector))
    # Computed route alternatives are independent overlays, beneath the endpoints.
    active_plan = st.session_state.get("eva_route_plan")
    if (active_plan and st.session_state.region_selector is not None
            and st.session_state.eva_start and st.session_state.eva_destination
            and active_plan.get("site_id") == st.session_state.region_selector
            and active_plan.get("start") == (st.session_state.eva_start["lat"],
                                             st.session_state.eva_start["lon"])
            and active_plan.get("destination") == (st.session_state.eva_destination["lat"],
                                                   st.session_state.eva_destination["lon"])):
        shown_routes = {
            name for name in ROUTE_STYLE
            if st.session_state.get("eva_show_" + name.replace(" ", "_"), True)
        }
        groups.append(
            create_eva_route_overlay(
                active_plan,
                show_hazard=bool(st.session_state.get("eva_show_hazard", False)),
                visible=shown_routes,
            )
        )
    # EVA waypoints are drawn last so they remain visible above SWIM markers.
    if st.session_state.region_selector is not None:
        groups.append(
            create_waypoint_markers(
                st.session_state.eva_start,
                st.session_state.eva_destination,
            )
        )

    st_folium(
        create_mars_map(),
        key="mars_surface_map",
        height=610,
        use_container_width=True,
        center=st.session_state.map_center,
        zoom=st.session_state.map_zoom,
        feature_group_to_add=groups,
        returned_objects=[
            "last_clicked",
            "last_object_clicked",
            "last_object_clicked_tooltip",
            "last_object_clicked_count",
        ],
        on_change=on_marker_click,
    )

    # -------------------------------------------------
    # EVA WAYPOINT SELECTION — ONLY FOR A SELECTED REGION
    # -------------------------------------------------
    if st.session_state.region_selector is not None:
        st.divider()
        st.subheader("EVA Route Planning")
        st.caption(
            "Define start and destination points on the map, then calculate "
            "Shortest, Safest and Balanced routes below."
        )

        st.radio(
            "Map click mode",
            options=["Browse", "Set start", "Set destination"],
            horizontal=True,
            key="eva_click_mode",
            help="Select Set start or Set destination, then click a SWIM marker, "
                 "region marker, or an empty location on the map. "
                 "The map returns to Browse after one selection.",
        )

        # Coordinates are chosen exclusively by clicking the map while
        # Set start or Set destination is active. No dropdowns are needed.
        start_col, end_col = st.columns(2, gap="medium")
        start = st.session_state.eva_start
        destination = st.session_state.eva_destination

        with start_col:
            st.markdown("**Start point**")
            if start is None:
                st.caption("Latitude: —  ·  Longitude: —")
                st.caption("Select Set start, then click the map or a marker.")
            else:
                st.info(
                    f'Latitude: {start["lat"]:.5f}°\n\n'
                    f'Longitude: {start["lon"]:.5f}°'
                )
                st.button(
                    "Clear start",
                    key="eva_clear_start",
                    on_click=clear_eva_point,
                    args=("start",),
                )

        with end_col:
            st.markdown("**Destination**")
            if destination is None:
                st.caption("Latitude: —  ·  Longitude: —")
                st.caption("Select Set destination, then click the map or a marker.")
            else:
                st.info(
                    f'Latitude: {destination["lat"]:.5f}°\n\n'
                    f'Longitude: {destination["lon"]:.5f}°'
                )
                st.button(
                    "Clear destination",
                    key="eva_clear_destination",
                    on_click=clear_eva_point,
                    args=("destination",),
                )

        if start is not None and destination is not None:
            distance = straight_line_distance_km(start, destination)
            if distance < 0.005:
                st.warning("Start and destination must be different points.")
            else:
                st.metric("Straight-line distance (reference)", f"{distance:.2f} km")
                st.caption(
                    "The dashed line is a visual connection, not an optimized "
                    "or terrain-safe EVA route."
                )

# Route scenarios, three A* alternatives, comparison and elevation profiles.
# Display in the SAME left column directly beneath the map and waypoints.
with map_col:
    if st.session_state.region_selector is not None:
        render_eva_route_panel(
            st.session_state.region_selector,
            st.session_state.eva_start,
            st.session_state.eva_destination,
        )

with info_col:
    st.subheader("Region")
    site_id = st.session_state.region_selector
    if site_id is None:
        st.write("Select a marker or choose a region.")
    else:
        site = SITES[site_id]
        st.markdown(f"**{site['name']}**")
        st.caption(f"{site['category']} · {site['priority']}")
        st.write(f"Lat {site['lat']:.2f}° · Lon {site['lon']:.2f}°")
        st.write(site["summary"])
        st.markdown("**Research focus**")
        st.write(site["focus"])
        st.caption(site["reference_note"])

    st.divider()
    st.markdown("**Map key**")
    for category, color in CATEGORY_COLORS.items():
        st.markdown(
            f'<span style="color:{color};font-size:17px;">●</span> '
            f'{html.escape(category)}',
            unsafe_allow_html=True,
        )


    # The two SWIM datasets use the SAME color scale: -1 to +1.
    if show_swim_0_1 or show_swim_1_5:
        st.divider()
        st.markdown("**SWIM4MIM · Ice consistency (Ci)**")
        st.markdown(
            '<div style="height:12px;border-radius:4px;'
            'background:linear-gradient(to right,'
            '#7f1d1d 0%,#cb6a47 30%,#e6e9e9 50%,'
            '#88d1d0 65%,#2a94be 82.5%,#173b8d 100%);">'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("−1  ·  0  ·  +1")
        st.caption(
            "Negative Ci: lower ice consistency. Positive Ci: higher ice "
            "consistency. Not an ice probability or confirmed deposit."
        )
        st.caption("SWIM4MIM derived grid · 0.2° · 60°S–60°N")
        if show_swim_0_1 and show_swim_1_5:
            st.caption("Both depths enabled: the 1–5 m layer is drawn above 0–1 m.")


    # Candidate statistics are shown only when a region and SWIM depth are active.
    if search_enabled:
        st.divider()
        st.markdown("**Ice Target Analysis**")
        for result in search_results:
            label = result["depth_label"]
            points = result["points"]
            peak = result["max_ci_in_radius"]
            marker_color = ICE_TARGET_COLORS[result["depth"]]
            st.markdown(
                f'<span style="color:{marker_color};font-size:16px;">●</span> '
                f'<b>SWIM {html.escape(label)}</b>',
                unsafe_allow_html=True,
            )
            if result["valid_cells"] == 0:
                st.caption("No valid SWIM grid cells in the search area.")
            elif not points:
                st.caption(
                    f"No candidates reach Ci ≥ {result['min_ci']:.2f}. "
                    f"Maximum regional Ci: {peak:.2f}."
                )
            else:
                st.caption(
                    f"{len(points)} candidates · "
                    f"{result['eligible_cells']} eligible cells · "
                    f"Peak Ci {peak:.2f}"
                )
                with st.expander(f"SWIM {label} targets ({len(points)})"):
                    for point in points:
                        st.markdown(
                            f"`{point['id']}` · **Ci {point['ci']:.2f}** "
                            f"· {point['distance_km']:.0f} km"
                        )
        st.caption(
            "Targets are ranked by SWIM consistency, not EVA safety or "
            "verified ice. Values describe coarse 0.2° raster cells."
        )
