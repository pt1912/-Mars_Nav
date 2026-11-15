import html

import streamlit as st
from streamlit_folium import st_folium

from modules.mars_map import (
    CATEGORY_COLORS,
    GLOBAL_CENTER,
    GLOBAL_ZOOM,
    REGION_ZOOM,
    create_mars_map,
    create_site_markers,
    load_sites,
    site_from_map_event,
)
from modules.mars_theme import apply_theme
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

with st.container(key="mars_intro"):
    st.html('<div class="mars-label">PLANETARY EXPLORATION SYSTEM</div>')
    st.title("MARS ICE EXPLORER")
    st.markdown(
        "Geospatial Platform designed for path planning and "
        "scientific data integration for Mars Ice exploration."
    )
    st.html('<div class="mars-separator"></div>')

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
