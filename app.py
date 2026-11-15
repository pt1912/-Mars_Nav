import streamlit as st
import folium

from streamlit_folium import st_folium
from modules.nasa_maps import get_arcadia_wmts
from modules.mars_geometry import mars_distance
from modules.slope_overlay import get_slope_overlay
from modules.path_planning import plan_route_from_points


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Mars_Nav",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# NASA MAP CONFIGURATION
# ==========================================

MARS_GLOBAL_URL = (
    "https://trek.nasa.gov/tiles/Mars/EQ/"
    "Mars_Viking_MDIM21_ClrMosaic_global_232m/"
    "1.0.0/default/default028mm/"
    "{z}/{y}/{x}.jpg"
)

# NASA MOLA Global Color Hillshade

MOLA_URL = (
    "https://trek.nasa.gov/tiles/Mars/EQ/"
    "Mars_MGS_MOLA_ClrShade_merge_global_463m/"
    "1.0.0/default/default028mm/"
    "{z}/{y}/{x}.jpg"
)

ARCADIA_BOUNDS = [
    [44.5079794, -173.645143],
    [47.8921384, -168.7548608]
]

ARCADIA_CENTER = [46.2, -171.2]


# ==========================================
# SESSION STATE
# ==========================================

if "start_point" not in st.session_state:
    st.session_state.start_point = None

if "destination" not in st.session_state:
    st.session_state.destination = None

if "last_processed_click" not in st.session_state:
    st.session_state.last_processed_click = None

# Keep computed routes across Streamlit reruns.
if "planned_route" not in st.session_state:
    st.session_state.planned_route = None
if "route_signature" not in st.session_state:
    st.session_state.route_signature = None
if "route_error" not in st.session_state:
    st.session_state.route_error = None


# ==========================================
# HEADER
# ==========================================

st.title("🚀 Mars_Nav")

st.subheader(
    "Martian Surface Exploration & Path Planning System"
)

st.markdown(
    "**NASA Space Apps Challenge 2026 | Arcadia Planitia**"
)

st.divider()


# ==========================================
# SIDEBAR - MISSION CONTROL
# ==========================================

with st.sidebar:

    st.header("Mission Control")

    st.subheader("Exploration Region")

    st.selectbox(
        "Region",
        ["Arcadia Planitia"]
    )

    st.divider()

    # Scientific layers
    st.subheader("Scientific Layers")

    optical = st.checkbox(
        "CTX Optical Imagery",
        value=True
    )

    show_bounds = st.checkbox(
        "Show CTX Coverage",
        value=False
    )

    mola = st.checkbox(
    "MOLA Topographic Relief",
    value=False
    )

    show_slope = st.checkbox(
    "Terrain Slope Analysis",
    value=False
    )

    st.divider()

    # Waypoint selection
    st.subheader("Waypoint Selection")

    selection_mode = st.radio(
        "Select Point",
        ["Start Point", "Destination"],
        horizontal=False
    )

    st.caption(
        "Select a point type and click "
        "on the Martian surface."
    )

    if st.button(
        "Clear Points",
        use_container_width=True
    ):
        st.session_state.start_point = None
        st.session_state.destination = None
        st.session_state.planned_route = None
        st.session_state.route_signature = None
        st.session_state.route_error = None
        st.rerun()

    st.divider()

    # Path planning
    st.subheader("Path Planning")

    algorithm = st.selectbox(
        "Algorithm",
        ["A*", "Dijkstra"]
    )

    exploration_mode = st.selectbox(
        "Exploration Mode",
        ["Astronaut", "Rover"]
    )

    max_slope_deg = st.slider(
        "Max slope (experimental, degrees)",
        min_value=0.5,
        max_value=20.0,
        value=15.0,
        step=0.5,
    )

    slope_weight = st.slider(
        "Slope cost penalty",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    st.caption(
        "Slope thresholds are experimental, not EVA safety limits. "
        "Astronaut/Rover does not yet change mobility parameters."
    )

    calculate_route_clicked = st.button(
        "Calculate Route",
        disabled=(
            st.session_state.start_point is None
            or st.session_state.destination is None
        ),
        type="primary",
        use_container_width=True,
    )


# ==========================================
# RUN PATH PLANNING
# ==========================================

def point_signature(point):
    if point is None:
        return None
    return (point["lat"], point["lon"])


current_route_signature = (
    point_signature(st.session_state.start_point),
    point_signature(st.session_state.destination),
    algorithm,
    exploration_mode,
    max_slope_deg,
    slope_weight,
)

# Any changed waypoint or planning parameter invalidates the previous route.
if st.session_state.route_signature != current_route_signature:
    st.session_state.planned_route = None
    st.session_state.route_error = None

if calculate_route_clicked:
    st.session_state.route_signature = current_route_signature
    st.session_state.planned_route = None
    st.session_state.route_error = None
    try:
        start = st.session_state.start_point
        destination = st.session_state.destination
        direct_km = mars_distance(
            start["lat"], start["lon"],
            destination["lat"], destination["lon"],
        )
        # Large regional routes can take longer; the planner itself
        # validates NoData, coordinates and traversability.
        with st.spinner("Calculating route on the MOLA terrain grid..."):
            result = plan_route_from_points(
                start_point=start,
                destination_point=destination,
                algorithm=algorithm,
                max_slope_deg=max_slope_deg,
                slope_weight=slope_weight,
            )

        st.session_state.planned_route = {
            "coordinates": result["coordinates"],
            "metrics": result["metrics"],
            "cells": len(result["path"]),
            "expanded_nodes": result["expanded_nodes"],
            "algorithm": result["algorithm"],
            "start_cell": result["start_cell"],
            "goal_cell": result["goal_cell"],
        }
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        st.session_state.route_error = str(error)


# ==========================================
# MAIN LAYOUT
# ==========================================

col_map, col_info = st.columns([3, 1])


# ==========================================
# MARTIAN SURFACE MAP
# ==========================================

with col_map:

    st.header("Martian Surface Map")

    try:

        nasa = get_arcadia_wmts()

        ctx_url = nasa["url"]

        ctx_url = ctx_url.replace(
            "{Style}", "default"
        )

        ctx_url = ctx_url.replace(
            "/1.0.0//", "/1.0.0/"
        )

        # ==================================
        # CREATE MAP
        # ==================================

        mars_map = folium.Map(
            location=ARCADIA_CENTER,
            zoom_start=8,
            tiles=None,
            crs="EPSG4326",
            control_scale=False,
            prefer_canvas=True
        )

        # ==================================
        # GLOBAL MARS MAP
        # ==================================

        folium.TileLayer(
            tiles=MARS_GLOBAL_URL,
            name="NASA Mars Global Map",
            attr="NASA Mars Trek / Viking Orbiter",
            overlay=False,
            control=True,
            show=True,
            min_zoom=0,
            max_zoom=12,
            max_native_zoom=7,
            no_wrap=True
        ).add_to(mars_map)

        # ==================================
        # ARCADIA CTX MAP
        # ==================================

        if optical:

            folium.TileLayer(
                tiles=ctx_url,
                name="NASA MRO CTX - Arcadia Planitia",
                attr="NASA Mars Trek / MRO CTX",
                min_zoom=nasa["min_zoom"],
                max_zoom=nasa["max_zoom"],
                max_native_zoom=nasa["max_zoom"],
                overlay=True,
                control=True,
                opacity=1.0,
                show=True,
                no_wrap=True
            ).add_to(mars_map)
        # ==================================
        # NASA MOLA TOPOGRAPHIC RELIEF
        # ==================================

        if mola:

            folium.TileLayer(
                tiles=MOLA_URL,
                name="NASA MOLA Topographic Relief",
                attr="NASA MGS / MOLA",
                min_zoom=0,
                max_zoom=12,
                max_native_zoom=7,
                overlay=True,
                control=True,
                show=True,
                opacity=0.65,
                no_wrap=True
            ).add_to(mars_map)
        # ==========================================
        # TERRAIN SLOPE ANALYSIS
        # ==========================================

        if show_slope:

            slope_image, slope_bounds, slope_stats = (
                get_slope_overlay()
            )

            folium.raster_layers.ImageOverlay(
                image=slope_image,
                bounds=slope_bounds,
                name="Mars_Nav Terrain Slope",
                opacity=0.75,
                overlay=True,
                control=True,
                show=True,
                origin="upper",
                interactive=False,
                zindex=2
            ).add_to(mars_map)

        # ==================================
        # CTX COVERAGE
        # ==================================

        if show_bounds:

            folium.Rectangle(
                bounds=ARCADIA_BOUNDS,
                color="orange",
                weight=2,
                fill=False,
                tooltip="Arcadia Planitia CTX Coverage"
            ).add_to(mars_map)

        # ==================================
        # DIRECT ROUTE REFERENCE
        # ==================================

        if (
            st.session_state.start_point is not None
            and st.session_state.destination is not None
        ):

            start = st.session_state.start_point
            destination = st.session_state.destination

            folium.PolyLine(
                locations=[
                    [start["lat"], start["lon"]],
                    [destination["lat"], destination["lon"]]
                ],
                color="cyan",
                weight=4,
                opacity=0.9,
                dash_array="8, 8",
                tooltip="Direct reference - Not a planned route"
            ).add_to(mars_map)

        # ==================================
        # PLANNED ROUTE (MOLA CELL CENTERS)
        # ==================================

        if st.session_state.planned_route is not None:
            planned = st.session_state.planned_route
            route_coordinates = planned["coordinates"]
            if len(route_coordinates) > 1:
                folium.PolyLine(
                    locations=route_coordinates,
                    color="#FFDE59",
                    weight=5,
                    opacity=1.0,
                    tooltip=f"{planned['algorithm']} - MOLA planned route",
                ).add_to(mars_map)
            elif route_coordinates:
                # Two selected positions may fall in the same MOLA cell.
                folium.CircleMarker(
                    location=route_coordinates[0],
                    radius=7,
                    color="#FFDE59",
                    fill=True,
                    fill_opacity=1.0,
                    tooltip="Start and destination share one MOLA cell",
                ).add_to(mars_map)

        # ==================================
        # START POINT MARKER
        # ==================================

        if st.session_state.start_point is not None:

            start = st.session_state.start_point

            folium.Marker(
                location=[start["lat"], start["lon"]],
                popup="Start Point",
                tooltip="Start Point",
                icon=folium.Icon(
                    color="green",
                    icon="play"
                )
            ).add_to(mars_map)

        # ==================================
        # DESTINATION MARKER
        # ==================================

        if st.session_state.destination is not None:

            destination = st.session_state.destination

            folium.Marker(
                location=[
                    destination["lat"],
                    destination["lon"]
                ],
                popup="Destination",
                tooltip="Destination",
                icon=folium.Icon(
                    color="red",
                    icon="flag"
                )
            ).add_to(mars_map)

        # ==================================
        # MAP CONFIGURATION
        # ==================================

        if (
            st.session_state.planned_route is not None
            and len(st.session_state.planned_route["coordinates"]) > 1
        ):
            mars_map.fit_bounds(
                st.session_state.planned_route["coordinates"],
                padding=(50, 50),
                max_zoom=11,
            )
        else:
            mars_map.fit_bounds(
                ARCADIA_BOUNDS,
                max_zoom=8
            )

        folium.LayerControl(
            position="topright",
            collapsed=True
        ).add_to(mars_map)

        # ==================================
        # DISPLAY MAP
        # ==================================

        map_data = st_folium(
            mars_map,
            height=650,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="mars_nav_map"
        )

        # ==================================
        # HANDLE MAP CLICKS
        # ==================================

        if map_data and map_data.get("last_clicked"):

            clicked = map_data["last_clicked"]

            lat = clicked["lat"]
            lon = clicked["lng"]

            click_id = (
                round(lat, 7),
                round(lon, 7)
            )

            # Process only new clicks.
            # Prevent old clicks from being
            # processed again after reruns.

            if click_id != st.session_state.last_processed_click:

                st.session_state.last_processed_click = click_id

                point = {
                    "lat": lat,
                    "lon": lon
                }

                # Only allow points inside
                # the selected CTX coverage

                inside_arcadia = (
                    ARCADIA_BOUNDS[0][0] <= lat <= ARCADIA_BOUNDS[1][0]
                    and
                    ARCADIA_BOUNDS[0][1] <= lon <= ARCADIA_BOUNDS[1][1]
                )

                if inside_arcadia:

                    if selection_mode == "Start Point":

                        st.session_state.start_point = point

                    elif selection_mode == "Destination":

                        st.session_state.destination = point

                    # A new point invalidates any previously calculated route.
                    st.session_state.planned_route = None
                    st.session_state.route_signature = None
                    st.session_state.route_error = None
                    st.rerun()

                else:

                    st.warning(
                        "Please select a point "
                        "inside the Arcadia CTX coverage."
                    )

        st.success("NASA Mars Trek WMTS connected.")

    except Exception as error:

        st.error(
            f"NASA map error: {error}"
        )


# ==========================================
# MISSION ANALYSIS
# ==========================================

with col_info:

    st.header("Mission Analysis")

    st.subheader("Selected Waypoints")

    # ======================================
    # START POINT
    # ======================================

    st.markdown("🟢 **Start Point**")

    if st.session_state.start_point:

        start = st.session_state.start_point

        st.write(
            f"Latitude: {start['lat']:.6f}°"
        )

        st.write(
            f"Longitude: {start['lon'] % 360:.6f}° E"
        )

    else:

        st.info("Start point not selected.")

    st.divider()

    # ======================================
    # DESTINATION
    # ======================================

    st.markdown("🔴 **Destination**")

    if st.session_state.destination:

        destination = st.session_state.destination

        st.write(
            f"Latitude: {destination['lat']:.6f}°"
        )

        st.write(
            f"Longitude: {destination['lon'] % 360:.6f}° E"
        )

    else:

        st.info("Destination not selected.")

    st.divider()

    # ======================================
    # PLANNING STATUS
    # ======================================

    st.subheader("Path Planning")

    st.write(
        "**Algorithm:**",
        algorithm
    )

    st.write(
        "**Exploration Mode:**",
        exploration_mode
    )

    if st.session_state.route_error:
        st.error(st.session_state.route_error)

    if st.session_state.planned_route is not None:
        if st.session_state.planned_route["cells"] == 1:
            st.info(
                "Both waypoints fall inside the same MOLA cell. "
                "Select points farther apart to display a planned line."
            )
        else:
            st.success("A terrain-aware route has been calculated.")
    elif (
        st.session_state.start_point is not None
        and st.session_state.destination is not None
    ):
        st.info("Points selected. Press Calculate Route in the sidebar.")

    else:

        st.warning(
            "Select both waypoints to continue."
        )

    st.divider()

    st.subheader("Route Metrics")

    if (
        st.session_state.start_point is not None
        and st.session_state.destination is not None
    ):

        start = st.session_state.start_point
        destination = st.session_state.destination

        distance = mars_distance(
            start["lat"],
            start["lon"],
            destination["lat"],
            destination["lon"]
        )

        st.metric(
            "Direct Distance",
            f"{distance:.2f} km"
        )

        # Advisory for demonstrations, not an EVA operational limit.
        if distance > 30:
            st.warning(
                "Long regional route: planning may be slower. "
                "For a quick demonstration, choose points within 30 km."
            )

    else:

        st.metric(
            "Direct Distance",
            "N/A"
        )

    planned = st.session_state.planned_route
    if planned is not None:
        metrics = planned["metrics"]
        st.metric("Planned Distance", f"{metrics['distance_km']:.2f} km")
        st.metric("Maximum Slope", f"{metrics['max_cell_slope_deg']:.2f}°")
        st.metric(
            "Elevation Change",
            f"{metrics['net_elevation_change_m']:+.1f} m"
        )
        st.metric("Cumulative Ascent", f"{metrics['cumulative_ascent_m']:.1f} m")
        with st.expander("Route calculation details"):
            st.write(f"Algorithm: {planned['algorithm']}")
            st.write(f"Traversed MOLA cells: {planned['cells']}")
            st.write(f"Expanded nodes: {planned['expanded_nodes']}")
            st.write(f"Cumulative descent: {metrics['cumulative_descent_m']:.1f} m")
            st.write(f"Cost: {metrics['route_cost_m']:.1f} equivalent meters")
    else:
        st.metric("Planned Distance", "N/A")
        st.metric("Maximum Slope", "N/A")
        st.metric("Elevation Change", "N/A")

    st.caption(
        "Direct distance: spherical reference between clicked coordinates. "
        "Planned distance: MOLA cell centers. Grid spacing is hundreds of "
        "meters; small obstacles and actual EVA safety are not resolved."
    )

    # ==========================================
    # REGIONAL TERRAIN ANALYSIS
    # ==========================================

    st.divider()

    st.subheader("Regional Terrain Analysis")

    if show_slope:

        _, _, stats = get_slope_overlay()

        st.metric(
            "Maximum Regional Slope",
            f"{stats['max_slope']:.2f}°"
        )

        st.metric(
            "Average Regional Slope",
            f"{stats['mean_slope']:.2f}°"
        )

        st.metric(
            "Elevation Range",
            f"{stats['min_elevation']:.0f} "
            f"to {stats['max_elevation']:.0f} m"
        )

        st.caption(
            "Source: NASA MGS MOLA. "
            "Regional terrain analysis."
        )

    else:

        st.info(
            "Enable Terrain Slope Analysis "
            "to display regional terrain statistics."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Mars_Nav | NASA Space Apps Challenge 2026"
)