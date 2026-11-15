"""NASA Mars Trek WMTS access, with explicit fallbacks and caching."""

from functools import lru_cache
import time
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen

NASA_ROOT = 'https://trek.nasa.gov/tiles/Mars/EQ'
VIKING_WMTS = (
    f'{NASA_ROOT}/Mars_Viking_MDIM21_ClrMosaic_global_232m/'
    '1.0.0/default/default028mm/{z}/{y}/{x}.jpg'
)
MOLA_WMTS = (
    f'{NASA_ROOT}/Mars_MGS_MOLA_ClrShade_merge_global_463m/'
    '1.0.0/default/default028mm/{z}/{y}/{x}.jpg'
)
CTX_GLOBAL = 'CTX_beta01_uncontrolled_5m_Caltech'
CTX_REGIONAL = {
    # Publicly documented Mars Trek Arcadia Planitia mosaic.
    'arcadia': 'Arcadia_Planitia_CTX_BlockAdj_dd',
}


def _local_name(tag):
    return tag.rsplit('}', 1)[-1]


def parse_wmts_template(contents):
    """Extract a Leaflet XYZ-like tile URL from WMTS ResourceURL metadata."""
    root = ET.fromstring(contents)
    layers = [n for n in root.iter() if _local_name(n.tag) == 'Layer']
    for layer in layers:
        resources = [n for n in layer.iter()
                     if _local_name(n.tag) == 'ResourceURL'
                     and n.get('resourceType') == 'tile']
        if not resources:
            continue
        template = resources[0].get('template')
        if not template:
            continue
        style = 'default'
        matrix_set = 'default028mm'
        for node in layer.iter():
            if _local_name(node.tag) == 'Style':
                styles = [n.text for n in node.iter()
                          if _local_name(n.tag) == 'Identifier' and n.text]
                if styles:
                    style = styles[0]
                break
        for node in layer.iter():
            if _local_name(node.tag) == 'TileMatrixSetLink':
                identifiers = [n.text for n in node.iter()
                               if _local_name(n.tag) == 'TileMatrixSet' and n.text]
                if identifiers:
                    matrix_set = identifiers[0]
                break
        for old, new in {
            '{Style}': style,
            '{TileMatrixSet}': matrix_set,
            '{TileMatrix}': '{z}',
            '{TileRow}': '{y}',
            '{TileCol}': '{x}',
        }.items():
            template = template.replace(old, new)
        if (template.startswith('https://trek.nasa.gov/')
                and all(p in template for p in ('{z}', '{x}', '{y}'))):
            return template
    raise ValueError('Could not determine a Mars Trek WMTS tile URL')


@lru_cache(maxsize=32)
def _get_template(layer_id):
    if not layer_id or any(x not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.' for x in layer_id):
        raise ValueError('Invalid NASA layer identifier')
    endpoint = f'{NASA_ROOT}/{layer_id}/1.0.0/WMTSCapabilities.xml'
    request = Request(endpoint, headers={'User-Agent': 'ArcadiaIceExplorer/1.0'})
    with urlopen(request, timeout=8) as response:
        return parse_wmts_template(response.read())


def get_ctx_tiles(region):
    """Return tile URL, layer ID and optional warning (10-minute cache).

    Cache failed lookups too: a failed NASA service must not block every
    Streamlit rerun. Restart the app or wait up to ten minutes to retry.
    """
    return _ctx_tiles_cached(region, int(time.monotonic() // 600))


@lru_cache(maxsize=32)
def _ctx_tiles_cached(region, time_bucket):
    """Resolve regional CTX first, then a global fallback."""
    requested = CTX_REGIONAL.get(region, CTX_GLOBAL)
    try:
        return _get_template(requested), requested, None
    except (OSError, ValueError, ET.ParseError) as exc:
        if requested == CTX_GLOBAL:
            return None, CTX_GLOBAL, f'CTX WMTS unavailable: {exc}'
        try:
            return (_get_template(CTX_GLOBAL), CTX_GLOBAL,
                    f'Regional CTX unavailable; displaying global CTX mosaic.')
        except (OSError, ValueError, ET.ParseError) as global_exc:
            return None, CTX_GLOBAL, f'CTX WMTS unavailable: {global_exc}'
