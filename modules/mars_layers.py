"""Mars Trek global scientific overlays and USGS geology.

The CTX global image mosaic is selectable under Surface in the sidebar.
No region-specific CTX imagery or DOM controls are created here.
"""

from functools import lru_cache
from urllib.request import Request, urlopen
import re
import xml.etree.ElementTree as ET

import folium

from modules.usgs_overlay import USGSGeologyOverlay


NASA_EQ = "https://trek.nasa.gov/tiles/Mars/EQ"

# (name shown in Leaflet, NASA Mars Trek layer ID, initial opacity, native zoom)
GLOBAL_LAYERS = (
    ("Terrain | MOLA shaded relief", "Mars_MGS_MOLA_ClrShade_merge_global_463m", 0.80, 7),
    ("Terrain | MOLA + HRSC relief", "Mars_MOLA_blend200ppx_HRSC_ClrShade_clon0dd_200mpp_lzw", 0.75, 8),
    ("Terrain | MOLA roughness", "mola_roughness", 0.75, 7),
    ("Thermal | THEMIS daytime IR", "THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018", 0.80, 9),
    ("Thermal | THEMIS nighttime IR", "THEMIS_NightIR_ControlledMosaics_100m_v2_oct2018", 0.80, 9),
    ("Thermal | TES thermal inertia", "TES_Thermal_Inertia", 0.75, 7),
    # CTX is a photographic mosaic (5 m source imagery), NOT an elevation DEM.
    ("Surface | CTX global mosaic (5 m/px)", "CTX_beta01_uncontrolled_5m_Caltech", 0.96, 13),
    ("Surface | TES albedo", "Mars_MGS_TES_Albedo_mosaic_global_7410m", 0.75, 6),
    ("Dust (historic) | TES dust", "TES_Dust", 0.70, 7),
    ("Dust (historic) | TES dust index", "tes_ruffdust", 0.70, 7),
    ("Composition | TES pyroxene", "TES_Clinopyroxene", 0.65, 7),
    ("Composition | TES plagioclase", "TES_Plagioclase", 0.65, 7),
    ("Composition | TES glass / clay", "TES_Glass_Clay", 0.65, 7),
)

def _name(tag):
    return tag.rsplit("}", 1)[-1]


def _child_identifier(element):
    for child in element:
        if _name(child.tag) == "Identifier":
            return child.text
    return None


@lru_cache(maxsize=40)
def mars_wmts_url(layer_id):
    """Get the actual png/jpg tile URL from NASA's WMTS capabilities.

    On an unavailable metadata endpoint, fall back to the standard NASA
    template so that the browser can still try to render the layer.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", layer_id):
        raise ValueError("Invalid NASA Mars Trek layer ID")

    capabilities = f"{NASA_EQ}/{layer_id}/1.0.0/WMTSCapabilities.xml"
    try:
        req = Request(capabilities, headers={"User-Agent": "MarsIceExplorer/1.0"})
        with urlopen(req, timeout=5) as response:
            root = ET.fromstring(response.read())

        for layer in root.iter():
            if _name(layer.tag) != "Layer":
                continue
            if _child_identifier(layer) not in (None, layer_id):
                continue

            style = "default"
            matrix_set = "default028mm"
            templates = []

            for child in layer:
                if _name(child.tag) == "Style":
                    style = _child_identifier(child) or style
                elif _name(child.tag) == "TileMatrixSetLink":
                    for link in child:
                        if _name(link.tag) == "TileMatrixSet" and link.text:
                            matrix_set = link.text
                elif _name(child.tag) == "ResourceURL":
                    if child.get("resourceType") == "tile" and child.get("template"):
                        templates.append(child.get("template"))

            if matrix_set != "default028mm":
                # Need a matching equirectangular tile matrix.
                continue

            for template in templates:
                url = template
                for key, value in {
                    "{Style}": style,
                    "{TileMatrixSet}": matrix_set,
                    "{TileMatrix}": "{z}",
                    "{TileRow}": "{y}",
                    "{TileCol}": "{x}",
                }.items():
                    url = url.replace(key, value)
                url = url.replace("http://trek.nasa.gov/", "https://trek.nasa.gov/")
                url = url.replace("/1.0.0//", "/1.0.0/")
                if url.startswith("https://trek.nasa.gov/") and all(
                    part in url for part in ("{z}", "{x}", "{y}")
                ):
                    return url
    except (OSError, ValueError, ET.ParseError):
        pass

    # Fallback when server-side GetCapabilities is unavailable.
    # Extension is a guess: check the service in the browser if blank.
    extension = ".jpg" if layer_id.startswith(("Mars_MGS_MOLA", "Mars_MOLA_")) else ".png"
    return f"{NASA_EQ}/{layer_id}/1.0.0/default/default028mm/{{z}}/{{y}}/{{x}}{extension}"


def create_selected_overlays(*, selected_layers=(), geology=False, geology_opacity=0.35):
    """Return a replaceable FeatureGroup; the Viking base map never changes."""
    group = folium.FeatureGroup(name="Scientific Overlays", control=False)
    selected = set(selected_layers)

    for name, layer_id, opacity, native_zoom in GLOBAL_LAYERS:
        if layer_id not in selected:
            continue
        folium.TileLayer(
            tiles=mars_wmts_url(layer_id),
            name=name,
            attr="NASA Mars Trek | MGS / MRO / Mars Odyssey",
            overlay=True,
            control=False,
            show=True,
            opacity=opacity,
            min_zoom=0,
            max_zoom=14,
            max_native_zoom=native_zoom,
            no_wrap=True,
        ).add_to(group)

    if geology:
        USGSGeologyOverlay(opacity=geology_opacity).add_to(group)

    return group



