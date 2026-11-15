"""Mars Trek scientific overlays and region-specific CTX mosaics.

Scientific overlays are updated dynamically from the left sidebar.
Changing the overlays does not replace the Viking base map.
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
    ("Surface | TES albedo", "Mars_MGS_TES_Albedo_mosaic_global_7410m", 0.75, 6),
    ("Dust (historic) | TES dust", "TES_Dust", 0.70, 7),
    ("Dust (historic) | TES dust index", "tes_ruffdust", 0.70, 7),
    ("Composition | TES pyroxene", "TES_Clinopyroxene", 0.65, 7),
    ("Composition | TES plagioclase", "TES_Plagioclase", 0.65, 7),
    ("Composition | TES glass / clay", "TES_Glass_Clay", 0.65, 7),
)

# Bounding boxes (west, south, east, north): only the local mosaics
# known to cover the representative point in mars_sites.json.
CTX_REGIONS = {
    "arcadia": (
        "Arcadia Planitia CTX", "Arcadia_Planitia_CTX_BlockAdj_dd",
        (-173.65, 44.5, -168.75, 47.9),
    ),
    "jezero": (
        "Jezero Controlled CTX", "JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0",
        (77.1604655, 18.2110274, 77.6991776, 18.7211945),
    ),
    "gale": (
        "Gale CTX", "Gale_CTX_BlockAdj_dd",
        (135.1842593, -6.5732988, 139.2413012, -2.8077809),
    ),
    "olympus": (
        "Olympus Mons CTX", "olympus_mons.eq",
        (-136, 12, -124, 24),
    ),
}


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
            max_zoom=12,
            max_native_zoom=native_zoom,
            no_wrap=True,
        ).add_to(group)

    if geology:
        USGSGeologyOverlay(opacity=geology_opacity).add_to(group)

    return group



def regional_ctx_info(site_id, sites):
    """Return a regional mosaic only when it covers the selected marker."""
    item = CTX_REGIONS.get(site_id)
    site = sites.get(site_id) if site_id else None
    if not item or not site:
        return None
    label, layer_id, (west, south, east, north) = item
    if west <= site["lon"] <= east and south <= site["lat"] <= north:
        return label, layer_id
    return None


def create_regional_ctx_group(site_id, sites):
    """CTX is added separately, never as a permanent global overlay."""
    info = regional_ctx_info(site_id, sites)
    if info is None:
        return None
    label, layer_id = info
    feature_group = folium.FeatureGroup(name=label, control=False)
    folium.TileLayer(
        tiles=mars_wmts_url(layer_id),
        name=label,
        attr="NASA Mars Trek / MRO CTX",
        overlay=True,
        control=False,
        opacity=0.95,
        min_zoom=0,
        max_zoom=12,
        max_native_zoom=12,
        no_wrap=True,
    ).add_to(feature_group)
    return feature_group
