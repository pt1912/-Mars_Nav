import requests
import xml.etree.ElementTree as ET
import streamlit as st


WMTS_URL = (
    "https://trek.nasa.gov/tiles/Mars/EQ/"
    "Arcadia_Planitia_CTX_BlockAdj_dd/"
    "1.0.0/WMTSCapabilities.xml"
)


@st.cache_data(ttl=3600)
def get_arcadia_wmts():

    response = requests.get(WMTS_URL, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    ns = {
        "wmts": "http://www.opengis.net/wmts/1.0",
        "ows": "http://www.opengis.net/ows/1.1"
    }

    layer = root.find(
        "./wmts:Contents/wmts:Layer",
        ns
    )

    if layer is None:
        raise ValueError("No se encontró la capa WMTS.")

    resource = layer.find("wmts:ResourceURL", ns)

    if resource is None:
        raise ValueError("No se encontró la URL de teselas.")

    tile_url = resource.attrib["template"]

    tile_matrix_set = layer.findtext(
        "wmts:TileMatrixSetLink/wmts:TileMatrixSet",
        namespaces=ns
    )

    matrix_sets = root.findall(
        "./wmts:Contents/wmts:TileMatrixSet",
        ns
    )

    zoom_levels = []

    for matrix_set in matrix_sets:

        identifier = matrix_set.findtext(
            "ows:Identifier",
            namespaces=ns
        )

        if identifier != tile_matrix_set:
            continue

        for matrix in matrix_set.findall(
            "wmts:TileMatrix",
            ns
        ):

            zoom = matrix.findtext(
                "ows:Identifier",
                namespaces=ns
            )

            if zoom and zoom.isdigit():
                zoom_levels.append(int(zoom))

    if not zoom_levels:
        raise ValueError(
            "No se encontraron niveles de zoom válidos."
        )

        # Adaptar plantilla WMTS para Folium/Leaflet

    style = layer.findtext(
        "wmts:Style/ows:Identifier",
        namespaces=ns
    )

    if not style:
        raise ValueError("No se encontró el estilo WMTS.")

    tile_url = tile_url.replace(
        "{Style}", style
    )

    tile_url = tile_url.replace(
        "{TileMatrixSet}", tile_matrix_set
    )

    tile_url = tile_url.replace(
        "{TileMatrix}", "{z}"
    )

    tile_url = tile_url.replace(
        "{TileRow}", "{y}"
    )

    tile_url = tile_url.replace(
        "{TileCol}", "{x}"
    )

    # Corregir la doble barra del template
    tile_url = tile_url.replace(
        "/1.0.0//", "/1.0.0/"
    )

    tile_url = tile_url.replace(
        "http://", "https://"
    )

    return {
        "url": tile_url,
        "min_zoom": min(zoom_levels),
        "max_zoom": max(zoom_levels)
    }