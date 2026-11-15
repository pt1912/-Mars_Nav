import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen

import folium
import streamlit as st
from streamlit_folium import st_folium

from modules.swim_global import load_swim_global
from modules.usgs_overlay import USGSGeologyOverlay


BASE_DIR = Path(__file__).resolve().parent
SITES_FILE = BASE_DIR / "data" / "locations" / "mars_sites.json"
ARCADIA_POI_DIR = BASE_DIR / "data" / "regions" / "arcadia_planitia"

NASA_TREK_ROOT = "https://trek.nasa.gov/tiles/Mars/EQ"

VIKING_WMTS = (
    f"{NASA_TREK_ROOT}/Mars_Viking_MDIM21_ClrMosaic_global_232m/"
    "1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
)

MOLA_WMTS = (
    f"{NASA_TREK_ROOT}/Mars_MGS_MOLA_ClrShade_merge_global_463m/"
    "1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
)

CTX_GLOBAL_ID = "CTX_beta01_uncontrolled_5m_Caltech"

CTX_REGIONAL_IDS = {
    "arcadia": "Arcadia_Planitia_CTX_BlockAdj_dd",
    "jezero": "JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0",
    "gale": "Gale_CTX_BlockAdj_dd",
    "olympus": "olympus_mons.eq",
}

GLOBAL_ID = "global"
GLOBAL_CENTER = (0.0, 0.0)
GLOBAL_ZOOM = 1
REGION_ZOOM = 5

CATEGORY_COLORS = {
    "Subsurface ice": "#38bdf8",
    "Robotic exploration": "#22C55E",
    "Terrain and geology": "#fb923c",
}

POI_COLORS = {
    "Ci_0_1": "#E879F9",
    "Ci_1_5": "#FDE047",
}


st.set_page_config(
    page_title="ARCADIA ICE EXPLORER",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SITE CATALOG
# ============================================================

@st.cache_data(show_spinner=False)
def load_sites():

    if not SITES_FILE.is_file():
        raise FileNotFoundError(
            f"Site catalog not found: {SITES_FILE}"
        )

    entries = json.loads(
        SITES_FILE.read_text(encoding="utf-8")
    )

    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "The site catalog must be a non-empty JSON array."
        )

    sites = {}

    required = (
        "id", "name", "lat", "lon",
        "category", "summary", "focus"
    )

    for item in entries:

        if not all(k in item for k in required):
            raise ValueError(f"Invalid catalog entry: {item!r}")

        site = dict(item)
        site_id = str(site["id"])

        if site_id in sites or site_id == GLOBAL_ID:
            raise ValueError(
                f"Duplicate or reserved site ID: {site_id}"
            )

        site["lat"] = float(site["lat"])
        site["lon"] = float(site["lon"])

        if not (
            -90 <= site["lat"] <= 90
            and -180 <= site["lon"] <= 180
        ):
            raise ValueError(
                f"Invalid coordinates for {site_id}"
            )

        sites[site_id] = site

    return sites


# ============================================================
# NASA CTX WMTS
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def resolve_ctx_tiles(layer_id):

    capabilities = (
        f"{NASA_TREK_ROOT}/{layer_id}/"
        "1.0.0/WMTSCapabilities.xml"
    )

    request = Request(
        capabilities,
        headers={"User-Agent": "MarsNav/1.0"}
    )

    try:

        with urlopen(request, timeout=8) as response:
            xml = ET.fromstring(response.read())

        local = lambda tag: tag.rsplit("}", 1)[-1]

        layer = next(
            node for node in xml.iter()
            if local(node.tag) == "Layer"
        )

        resource = next(
            node for node in layer.iter()
            if (
                local(node.tag) == "ResourceURL"
                and node.get("resourceType") == "tile"
            )
        )

        template = resource.get("template")

        if not template:
            raise ValueError(
                "Missing WMTS ResourceURL template"
            )

        style = "default"
        matrix_set = "default028mm"

        for node in layer.iter():

            if local(node.tag) == "Style":

                ident = next(
                    (
                        n.text for n in node.iter()
                        if local(n.tag) == "Identifier"
                    ),
                    None,
                )

                if ident:
                    style = ident

                break

        for node in layer.iter():

            if local(node.tag) == "TileMatrixSetLink":

                ident = next(
                    (
                        n.text for n in node.iter()
                        if local(n.tag) == "TileMatrixSet"
                    ),
                    None,
                )

                if ident:
                    matrix_set = ident

                break

        tile_url = template

        replacements = {
            "{Style}": style,
            "{TileMatrixSet}": matrix_set,
            "{TileMatrix}": "{z}",
            "{TileRow}": "{y}",
            "{TileCol}": "{x}",
        }

        for placeholder, value in replacements.items():
            tile_url = tile_url.replace(
                placeholder, value
            )

        if not tile_url.startswith("https://"):
            raise ValueError(
                "Unexpected WMTS URL template"
            )

        if not all(
            value in tile_url
            for value in ("{z}", "{x}", "{y}")
        ):
            raise ValueError(
                "Incomplete WMTS tile template"
            )

        return tile_url, None

    except (
        OSError,
        ValueError,
        ET.ParseError,
        StopIteration,
    ) as exc:

        fallback = (
            f"{NASA_TREK_ROOT}/{layer_id}/"
            "1.0.0/default/default028mm/"
            "{z}/{y}/{x}.png"
        )

        return fallback, str(exc)


# ============================================================
# SCIENTIFIC POINTS OF INTEREST
# ============================================================

def load_arcadia_pois(depth):

    filenames = {
        "Ci_0_1": "swim_poi.json",
        "Ci_1_5": "swim_poi_1_5.json",
    }

    file = ARCADIA_POI_DIR / filenames[depth]

    if file.is_file():

        record = json.loads(
            file.read_text(encoding="utf-8")
        )

    else:

        # Reuse the existing SWIM candidate extractor.

        try:
            from scripts.prepare_swim_i8_poi import prepare_pois

        except ImportError:

            return [], (
                f"Missing {file.name}. Generate it using "
                f"scripts/prepare_swim_i8_poi.py"
            )

        record = prepare_pois(
            BASE_DIR / "data" / "raw" / "swim",
            depth=depth,
        )

    points = record.get("points", [])

    if not isinstance(points, list):
        raise ValueError(
            f"Invalid 'points' field in {file}"
        )

    cleaned = []

    for poi in points:

        try:

            lat = float(poi["lat"])
            lon = float(poi["lon"])
            ci = float(poi["consistency"])

            if (
                -90 <= lat <= 90
                and -180 <= lon <= 180
                and -1 <= ci <= 1
            ):

                cleaned.append({
                    "id": str(
                        poi.get(
                            "id",
                            f"SWIM-{len(cleaned) + 1:02d}"
                        )
                    ),
                    "lat": lat,
                    "lon": lon,
                    "ci": ci,
                })

        except (KeyError, TypeError, ValueError):
            continue

    return cleaned, None


SITES = load_sites()


# ============================================================
# NAVIGATION
# ============================================================

def select_site(site_id):

    if site_id != GLOBAL_ID and site_id not in SITES:
        raise ValueError(f"Unknown site: {site_id}")

    if st.session_state.selected_site != site_id:

        st.session_state.show_swim_0_1 = False
        st.session_state.show_swim_1_5 = False

    st.session_state.selected_site = site_id

    if site_id == GLOBAL_ID:

        st.session_state.nav_center = GLOBAL_CENTER
        st.session_state.nav_zoom = GLOBAL_ZOOM

    else:

        site = SITES[site_id]

        st.session_state.nav_center = (
            site["lat"],
            site["lon"],
        )

        st.session_state.nav_zoom = REGION_ZOOM

    st.session_state.map_revision += 1


def on_region_change():

    selected = st.session_state.site_selector

    if selected is not None:
        select_site(selected)


def on_global_view():

    st.session_state.site_selector = None
    select_site(GLOBAL_ID)


# ============================================================
# TOOLTIPS
# ============================================================

def tooltip_for(site):

    lat = site["lat"]
    lon = site["lon"]

    name = html.escape(site["name"])
    category = html.escape(site["category"])
    summary = html.escape(site["summary"])
    focus = html.escape(site["focus"])

    return folium.Tooltip(
        '<div style="'
        'width:300px;'
        'white-space:normal;'
        'line-height:1.45;'
        'font-family:Arial,sans-serif;'
        'font-size:12px">'

        f'<b style="font-size:14px">{name}</b><br>'

        f'<span style="color:#526378">'
        f'{category} · '
        f'{abs(lat):.2f}°{"N" if lat >= 0 else "S"}, '
        f'{abs(lon):.2f}°{"E" if lon >= 0 else "W"}'
        '</span><br>'

        f'{summary}<br>'
        f'<b>Analysis:</b> {focus}'
        '</div>',

        sticky=True,
    )


def resolve_clicked_site(map_data):

    if not map_data:
        return None

    point = map_data.get("last_object_clicked")

    if isinstance(point, dict):

        if "lat" in point and "lng" in point:

            try:

                lat = float(point["lat"])
                lon = float(point["lng"])

                for sid, site in SITES.items():

                    if (
                        abs(lat - site["lat"]) < 0.0001
                        and abs(lon - site["lon"]) < 0.0001
                    ):
                        return sid

            except (TypeError, ValueError):
                pass

    raw = map_data.get(
        "last_object_clicked_tooltip"
    )

    if isinstance(raw, str):

        plain = html.unescape(
            re.sub(r"<[^>]+>", " ", raw)
        )

        for sid, site in SITES.items():

            if site["name"] in plain:
                return sid

    return None


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "map_revision": 0,
    "selected_site": GLOBAL_ID,
    "nav_center": GLOBAL_CENTER,
    "nav_zoom": GLOBAL_ZOOM,
}

for key, initial in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = initial


if st.session_state.get("site_selector") == GLOBAL_ID:
    st.session_state.site_selector = None


if "pending_site" in st.session_state:

    pending = st.session_state.pop("pending_site")

    if pending in SITES:

        st.session_state.site_selector = pending
        select_site(pending)


# ============================================================
# HEADER
# ============================================================

st.title("ARCADIA ICE EXPLORER")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Mission Control")

    st.subheader("Explore Mars")

    st.selectbox(
        "Region",
        options=list(SITES),
        index=None,
        placeholder="Select a region",
        format_func=lambda sid: SITES[sid]["name"],
        key="site_selector",
        on_change=on_region_change,
    )

    st.button(
        "🌐 Global Map",
        use_container_width=True,
        on_click=on_global_view,
    )

    st.divider()

    # MAP LAYERS

    st.subheader("Map Layers")

    show_ctx = st.checkbox(
        "MRO CTX imagery",
        value=False,
        key="show_ctx",
    )

    show_mola = st.checkbox(
        "MOLA shaded relief",
        value=False,
        key="show_mola",
    )

    show_geology = st.checkbox(
        "USGS SIM 3292 geology",
        value=False,
        key="show_geology",
    )

    if show_geology:

        geology_opacity = st.slider(
            "Geology opacity",
            0.10,
            0.60,
            0.25,
            0.05,
            key="geology_opacity",
        )

    else:

        geology_opacity = 0.25

    # ICE OVERLAYS

    region = st.session_state.selected_site

    show_swim_0_1 = False
    show_swim_1_5 = False
    swim_opacity = 0.65

    if region != GLOBAL_ID:

        st.divider()

        st.subheader("Ice Overlays")

        show_swim_0_1 = st.checkbox(
            "SWIM ice consistency (0–1 m)",
            value=False,
            key="show_swim_0_1",
        )

        show_swim_1_5 = st.checkbox(
            "SWIM ice consistency (1–5 m)",
            value=False,
            key="show_swim_1_5",
        )

        if show_swim_0_1 or show_swim_1_5:

            swim_opacity = st.slider(
                "SWIM opacity",
                0.20,
                1.00,
                0.65,
                0.05,
                key="swim_opacity",
            )


# ============================================================
# MAP LAYOUT
# ============================================================

map_col, info_col = st.columns(
    [3.6, 1],
    gap="medium",
)

active_swim = []
poi_counts = []
render_pois = []


with map_col:

    st.subheader("Mars Surface Map")

    mars_map = folium.Map(
        location=list(st.session_state.nav_center),
        zoom_start=st.session_state.nav_zoom,
        tiles=None,
        crs="EPSG4326",
        zoom_snap=0.5,
        min_zoom=0,
        max_zoom=12,
        control_scale=False,
        prefer_canvas=True,
        zoom_control=True,
    )

    # VIKING BASE MAP

    folium.TileLayer(
        tiles=VIKING_WMTS,
        name="NASA Viking | global mosaic",
        attr="NASA Mars Trek / Viking Orbiter",
        overlay=False,
        control=False,
        show=True,
        min_zoom=0,
        max_zoom=12,
        max_native_zoom=7,
        no_wrap=True,
    ).add_to(mars_map)

    # MOLA

    if show_mola:

        folium.TileLayer(
            tiles=MOLA_WMTS,
            name="MGS MOLA | shaded relief",
            attr="NASA Mars Trek / MGS MOLA",
            opacity=0.72,
            overlay=True,
            control=False,
            show=True,
            min_zoom=0,
            max_zoom=12,
            max_native_zoom=8,
            no_wrap=True,
        ).add_to(mars_map)

    # CTX

    if show_ctx:

        ctx_id = CTX_REGIONAL_IDS.get(
            region,
            CTX_GLOBAL_ID,
        )

        ctx_tiles, ctx_error = resolve_ctx_tiles(
            ctx_id
        )

        folium.TileLayer(
            tiles=ctx_tiles,
            name="MRO CTX | imagery",
            attr=f"NASA Mars Trek / MRO CTX ({ctx_id})",
            opacity=0.90,
            overlay=True,
            control=False,
            show=True,
            min_zoom=0,
            max_zoom=12,
            max_native_zoom=12,
            no_wrap=True,
        ).add_to(mars_map)

        if ctx_error:

            st.sidebar.warning(
                "CTX WMTS metadata unavailable; "
                "using a fallback tile URL."
            )

    # USGS GEOLOGY

    if show_geology:

        USGSGeologyOverlay(
            opacity=geology_opacity
        ).add_to(mars_map)

    # SWIM OVERLAYS

    swim_layers = (
        (show_swim_0_1, "Ci_0_1", "0–1 m"),
        (show_swim_1_5, "Ci_1_5", "1–5 m"),
    )

    for enabled, depth, label in swim_layers:

        if not enabled:
            continue

        try:

            rgba, bounds, info = load_swim_global(
                depth
            )

            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=bounds,
                name=f"SWIM4MIM | Ci, {label}",
                opacity=swim_opacity,
                interactive=False,
                pixelated=True,
                control=False,
            ).add_to(mars_map)

            active_swim.append(label)

        except (
            FileNotFoundError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:

            st.sidebar.warning(
                f"SWIM {label} unavailable: {exc}"
            )

        # SCIENTIFIC POIs: ARCADIA

        if region == "arcadia":

            try:

                pois, issue = load_arcadia_pois(
                    depth
                )

            except (
                OSError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:

                pois = []
                issue = str(exc)

            if issue:

                st.sidebar.warning(
                    f"SWIM {label} POIs: {issue}"
                )

            else:

                render_pois.extend(
                    (poi, depth, label)
                    for poi in pois
                )

                poi_counts.append(
                    (label, len(pois))
                )

    # SCIENTIFIC MARKERS

    for poi, depth, label in render_pois:

        folium.CircleMarker(
            location=[
                poi["lat"],
                poi["lon"],
            ],
            radius=5,
            color="#111827",
            weight=1.5,
            fill=True,
            fill_color=POI_COLORS[depth],
            fill_opacity=0.95,
            tooltip=folium.Tooltip(
                f'<b>{html.escape(poi["id"])}</b><br>'
                f'Depth: {label}<br>'
                f'Ice consistency Ci: {poi["ci"]:.2f}<br>'
                'Candidate SWIM raster cell; '
                'not a confirmed ice deposit.',
                sticky=True,
            ),
        ).add_to(mars_map)

    # PERMANENT LOCATIONS

    for site in SITES.values():

        site_color = CATEGORY_COLORS.get(
            site["category"].strip(),
            "#94a3b8",
        )

        folium.CircleMarker(
            location=[
                site["lat"],
                site["lon"],
            ],
            radius=7,
            color="#172033",
            weight=2,
            fill=True,
            fill_color=site_color,
            fill_opacity=0.96,
            tooltip=tooltip_for(site),
        ).add_to(mars_map)

    # STREAMLIT MAP

    map_data = st_folium(
        mars_map,
        key=(
            f"mars_global_v2_"
            f"{st.session_state.map_revision}"
        ),
        height=610,
        use_container_width=True,
        returned_objects=[
            "last_object_clicked",
            "last_object_clicked_tooltip",
            "center",
            "zoom",
        ],
    )

    # MAP POSITION

    if map_data:

        center = map_data.get("center")
        zoom = map_data.get("zoom")

        if (
            isinstance(center, dict)
            and "lat" in center
            and "lng" in center
        ):

            st.session_state.nav_center = (
                float(center["lat"]),
                float(center["lng"]),
            )

        if isinstance(zoom, (int, float)):
            st.session_state.nav_zoom = zoom

    # REGION SELECTION

    clicked_site = resolve_clicked_site(
        map_data
    )

    if (
        clicked_site is not None
        and clicked_site != st.session_state.selected_site
    ):

        st.session_state.pending_site = clicked_site
        st.rerun()


# ============================================================
# REGION INFORMATION
# ============================================================

with info_col:

    st.subheader("Region")

    selected = st.session_state.selected_site

    if selected == GLOBAL_ID:

        st.write(
            "Select a marker or choose a region."
        )

    else:

        site = SITES[selected]

        st.markdown(
            f"**{site['name']}**"
        )

        st.write(
            site["category"]
        )

        st.write(
            f"Lat {site['lat']:.2f}° · "
            f"Lon {site['lon']:.2f}°"
        )

    st.divider()

    st.markdown("**Map key**")

    for category, color in CATEGORY_COLORS.items():

        st.markdown(
            f'<span style="color:{color};'
            f'font-size:16px">●</span> '
            f'{html.escape(category)}',
            unsafe_allow_html=True,
        )

    # SWIM LEGEND

    if active_swim:

        st.divider()

        st.markdown(
            "**SWIM4MIM · Ice consistency (Ci)**"
        )

        st.markdown(
            '<div style="'
            'height:10px;'
            'border-radius:3px;'
            'background:linear-gradient('
            '90deg,'
            '#ffffd9,'
            '#c7e9b4,'
            '#7fcdbb,'
            '#41b6c4,'
            '#2c7fb8,'
            '#253494'
            ')"></div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "0 → 0.85 · Higher Ci means stronger "
            "ice consistency; not probability."
        )

        for depth, count in poi_counts:

            st.caption(
                f"{depth} | {count} research "
                "candidates in Arcadia"
            )

        if selected != "arcadia":

            st.caption(
                "Research candidate markers are "
                "currently available for Arcadia only."
            )