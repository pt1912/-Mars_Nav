import streamlit as st
import folium

from streamlit_folium import st_folium
from modules.nasa_maps import get_arcadia_wmts
from modules.mars_geometry import mars_distance
from modules.slope_overlay import get_slope_overlay
from modules.path_planning import plan_route_from_points
from modules.scientific_targets import load_swim_pois
from modules.swim_heatmap import get_swim_heatmap


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

# Persist the Leaflet viewport and allow reusing a clicked location.
if "map_center" not in st.session_state:
    st.session_state.map_center = list(ARCADIA_CENTER)
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 8
if "last_map_point" not in st.session_state:
    st.session_state.last_map_point = None
if "last_swim_marker_click_count" not in st.session_state:
    st.session_state.last_swim_marker_click_count = 0

# A clicked SWIM marker becomes the selected sidebar target on the NEXT rerun.
# This avoids modifying an existing Streamlit widget after it was instantiated.
pending_swim = st.session_state.pop("pending_swim_choice", None)
if pending_swim:
    st.session_state["swim_target_selector"] = pending_swim


def set_waypoint(which, point):
    lat, lon = float(point["lat"]), float(point["lon"])
    if not (ARCADIA_BOUNDS[0][0] <= lat <= ARCADIA_BOUNDS[1][0]
            and ARCADIA_BOUNDS[0][1] <= lon <= ARCADIA_BOUNDS[1][1]):
        raise ValueError("El punto está fuera de Arcadia CTX.")
    if which not in ("start_point", "destination"):
        raise ValueError("Tipo de marcador no reconocido.")
    st.session_state[which] = {"lat": lat, "lon": lon}
    st.session_state.planned_route = None
    st.session_state.route_signature = None
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

    show_swim = st.checkbox(
        "SWIM — Ice Consistency Heatmap (0–1 m)",
        value=False
    )
    swim_opacity = 0.70
    if show_swim:
        swim_opacity = st.slider(
            "SWIM layer opacity", min_value=0.20, max_value=1.00,
            value=0.70, step=0.05
        )
    show_ice_pois = st.checkbox(
        "Show SWIM research points", value=True,
        disabled=not show_swim
    ) and show_swim

    st.divider()

    # Waypoint selection
    st.subheader("Waypoint Selection")

    selection_mode = st.radio(
        "Select Point",
        ["Start Point", "Destination"],
        horizontal=False
    )

    st.caption(
        "Click the map to set the selected waypoint. "
        "You can also reassign the most recently clicked location:"
    )
    if st.session_state.last_map_point is not None:
        p = st.session_state.last_map_point
        st.caption(f"Selected: {p['lat']:.5f}°, {p['lon']:.5f}°")
        last_col_start, last_col_end = st.columns(2)
        with last_col_start:
            if st.button("Use as Start", key="clicked_as_start", use_container_width=True):
                set_waypoint("start_point", p)
                st.rerun()
        with last_col_end:
            if st.button("Use as Destination", key="clicked_as_dest", use_container_width=True):
                set_waypoint("destination", p)
                st.rerun()

    if show_swim:
        st.markdown("**Scientific target selection**")
        try:
            swim_poi_catalog = load_swim_pois()
        except (FileNotFoundError, ValueError) as error:
            st.warning(f"SWIM POIs: {error}")
            swim_poi_catalog = None
        if swim_poi_catalog and swim_poi_catalog["points"]:
            poi_lookup = {p["id"]: p for p in swim_poi_catalog["points"]}
            existing = st.session_state.get("swim_target_selector")
            if existing not in poi_lookup:
                st.session_state["swim_target_selector"] = next(iter(poi_lookup))
            target_id = st.selectbox(
                "Choose a SWIM point or click its marker on the map",
                options=list(poi_lookup),
                format_func=lambda v: f"{v} · Ci={poi_lookup[v]['consistency']:.2f}",
                key="swim_target_selector",
            )
            target = poi_lookup[target_id]
            st.caption(f"{target['lat']:.4f}° N, {target['lon']:.4f}° · Ci={target['consistency']:.2f}")
            poi_col_start, poi_col_end = st.columns(2)
            with poi_col_start:
                if st.button("POI → Start", key="poi_to_start", use_container_width=True):
                    set_waypoint("start_point", target)
                    st.rerun()
            with poi_col_end:
                if st.button("POI → Destination", key="poi_to_dest", use_container_width=True):
                    set_waypoint("destination", target)
                    st.rerun()
            st.caption("Ci es consistencia de hielo, no una probabilidad.")

    if st.button(
        "Clear Points",
        use_container_width=True
    ):
        st.session_state.start_point = None
        st.session_state.destination = None
        st.session_state.planned_route = None
        st.session_state.route_signature = None
        st.session_state.route_error = None
        st.session_state.last_map_point = None
        st.session_state.last_processed_click = None
        st.rerun()

    if st.button("Reset Map View", use_container_width=True):
        st.session_state.map_center = list(ARCADIA_CENTER)
        st.session_state.map_zoom = 8
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
            location=st.session_state.map_center,
            zoom_start=st.session_state.map_zoom,
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
        # SWIM4MIM ICE CONSISTENCY — REGIONAL RASTER
        # ==================================

        if show_swim:
            try:
                swim_rgba, swim_bounds, swim_stats = get_swim_heatmap()
            except (FileNotFoundError, ValueError) as error:
                st.warning(f"SWIM heatmap: {error}")
            else:
                folium.raster_layers.ImageOverlay(
                    image=swim_rgba,
                    bounds=swim_bounds,
                    name="SWIM4MIM — Ice Consistency (0–1 m)",
                    opacity=swim_opacity,
                    origin="upper",
                    interactive=False,
                    control=True,
                    overlay=True,
                    show=True,
                    zindex=3,
                ).add_to(mars_map)
                # Visual legend: fixed stretch, not an EVA safety classification.
                from branca.element import Element
                legend = """<div style="position:fixed;bottom:35px;left:30px;z-index:9998;
                    width:220px;background:rgba(255,255,255,0.94);padding:9px 12px;
                    border:1px solid #777;border-radius:6px;font:12px Arial;">
                  <b>SWIM Ci · 0–1 m (not temperature)</b><br>
                  <div style="height:13px;margin:7px 0 4px;
                  background:linear-gradient(90deg,#ffffd9,#7fcdbb,#41b6c4,#225ea8,#081d58)"></div>
                  <span style="float:left">0.45</span><span style="float:right">0.80</span>
                  <div style="clear:both"></div>
                  <small>Higher Ci = darker blue; regional 0.2° grid.</small>
                </div>"""
                mars_map.get_root().html.add_child(Element(legend))
                st.caption(
                    "SWIM4MIM ice-consistency raster (0–1 m), "
                    f"{swim_stats['valid_cells']} valid source cells. "
                    "Derived from Twin Worlds; not temperature or ice probability."
                )

        # ==================================
        # SCIENTIFIC TARGETS — SWIM 0–1 m
        # ==================================

        if show_ice_pois:
            try:
                swim_data = load_swim_pois()
            except (FileNotFoundError, ValueError) as error:
                st.warning(f"SWIM POIs: {error}")
            else:
                swim_layer = folium.FeatureGroup(
                    name="SWIM — Candidate Ice Investigation Sites",
                    overlay=True,
                    control=True,
                    show=True,
                )
                for poi in swim_data["points"]:
                    folium.CircleMarker(
                        location=[poi["lat"], poi["lon"]],
                        radius=7,
                        color="#40C9FF",
                        weight=2,
                        fill=True,
                        fill_color="#40C9FF",
                        fill_opacity=0.85,
                        tooltip=f"{poi['id']} — SWIM ice research candidate",
                        popup=folium.Popup(
                            f"<b>{poi['id']} — SWIM candidate</b><br>"
                            f"Consistency index: {poi['consistency']:.3f}<br>"
                            f"Depth range: 0–1 m<br>"
                            f"Coordinates: {poi['lat']:.5f}° N, "
                            f"{poi['lon'] % 360:.5f}° E<br>"
                            "<small>Derived sampling location, not "
                            "a confirmed ice detection.</small>",
                            max_width=280,
                        ),
                    ).add_to(swim_layer)
                swim_layer.add_to(mars_map)
                st.caption(
                    f"SWIM4MIM (0–1 m): {len(swim_data['points'])} "
                    "candidate investigation waypoints. "
                    "Ice consistency is not a probability."
                )

        # ==================================
        # CTX COVERAGE
        # ==================================

        if show_bounds:

            ctx_coverage = folium.Rectangle(
                bounds=ARCADIA_BOUNDS,
                color="orange",
                weight=2,
                fill=False,
            )
            # Folium omits interactive=False in its constructor options;
            # explicitly serialize the Leaflet option to allow map clicks.
            ctx_coverage.options["interactive"] = False
            ctx_coverage.add_to(mars_map)

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

        # Keep the user's current center and zoom; never call fit_bounds() on reruns.

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
            returned_objects=[
                "last_clicked", "last_object_clicked_tooltip",
                "last_object_clicked_count", "center", "zoom"
            ],
            center=tuple(st.session_state.map_center),
            zoom=st.session_state.map_zoom,
            key="mars_nav_map"
        )

        # ==================================
        # PRESERVE CURRENT MAP VIEW + MARKER SELECTION
        # ==================================

        if map_data:
            center = map_data.get("center")
            if (isinstance(center, dict) and
                "lat" in center and "lng" in center):
                st.session_state.map_center = [float(center["lat"]), float(center["lng"])]
            zoom = map_data.get("zoom")
            if zoom is not None:
                st.session_state.map_zoom = int(zoom)

            # Folium sends the marker tooltip on click. Select the matching
            # POI on the next rerun, before the sidebar widget is rendered.
            tooltip = map_data.get("last_object_clicked_tooltip")
            click_count = map_data.get("last_object_clicked_count")
            if (show_ice_pois and click_count is not None and
                click_count != st.session_state.last_swim_marker_click_count):
                st.session_state.last_swim_marker_click_count = click_count
                if isinstance(tooltip, str) and tooltip.startswith("SWIM-"):
                    poi_id = tooltip.split(" — ", 1)[0]
                    try:
                        known = {p["id"] for p in load_swim_pois()["points"]}
                    except (FileNotFoundError, ValueError):
                        known = set()
                    if poi_id in known and poi_id != st.session_state.get("swim_target_selector"):
                        st.session_state.pending_swim_choice = poi_id
                        st.rerun()

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
                    st.session_state.last_map_point = point
                    if selection_mode == "Start Point":
                        set_waypoint("start_point", point)
                    elif selection_mode == "Destination":
                        set_waypoint("destination", point)
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