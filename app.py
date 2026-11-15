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
from modules.swim_overlays import create_swim_overlay_group, load_swim_overlay
from modules.mars_layers import (
    GLOBAL_LAYERS,
    create_regional_ctx_group,
    create_selected_overlays,
    regional_ctx_info,
)


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


def navigate_to(site_id):
    if site_id is None:
        st.session_state.map_center = GLOBAL_CENTER
        st.session_state.map_zoom = GLOBAL_ZOOM
    else:
        site = SITES[site_id]
        st.session_state.map_center = (site["lat"], site["lon"])
        st.session_state.map_zoom = REGION_ZOOM


def on_region_change():
    navigate_to(st.session_state.region_selector)


def on_global_map():
    st.session_state.region_selector = None
    navigate_to(None)


def on_marker_click():
    event = st.session_state.get("mars_surface_map")
    site_id = site_from_map_event(event, SITES)
    if site_id is not None and site_id != st.session_state.region_selector:
        # Callback executes before widgets are created in the new run.
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

    show_geology = st.checkbox(
        "USGS SIM 3292 geology", key="show_usgs_geology"
    )
    geology_opacity = 0.35
    if show_geology:
        geology_opacity = st.slider(
            "USGS opacity", 0.0, 1.0, 0.35, 0.05,
            key="usgs_opacity",
        )

    ctx_info = regional_ctx_info(st.session_state.region_selector, SITES)
    show_ctx = False
    if ctx_info is not None:
        st.subheader("Regional Imagery")
        show_ctx = st.checkbox("MRO CTX mosaic", key="show_region_ctx")
        st.caption(ctx_info[0] + " · limited local coverage")

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

map_col, info_col = st.columns([3.6, 1], gap="medium")

with map_col:
    st.subheader("Mars Surface Map")
    groups = []
    if show_ctx:
        group = create_regional_ctx_group(st.session_state.region_selector, SITES)
        if group is not None:
            groups.append(group)
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

    st_folium(
        create_mars_map(),
        key="mars_surface_map",
        height=610,
        use_container_width=True,
        center=st.session_state.map_center,
        zoom=st.session_state.map_zoom,
        feature_group_to_add=groups,
        returned_objects=[
            "last_object_clicked",
            "last_object_clicked_tooltip",
            "last_object_clicked_count",
        ],
        on_change=on_marker_click,
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
