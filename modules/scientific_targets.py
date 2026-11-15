"""Lightweight SWIM scientific waypoint reader for Mars_Nav Streamlit.

Reads a small local JSON; runtime never downloads the full GeoTIFF.
The markers are planning candidates sampled from SWIM ice-consistency
raster, not detections or confirmed ice-exposing craters.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POI_FILE = ROOT / 'data/regions/arcadia_planitia/swim_poi.json'


@lru_cache(maxsize=1)
def load_swim_pois():
    if not POI_FILE.exists():
        raise FileNotFoundError(
            'Falta data/regions/arcadia_planitia/swim_poi.json. '
            'Ejecuta primero scripts/prepare_swim_poi.py con el GeoTIFF oficial.'
        )
    data = json.loads(POI_FILE.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('points'), list):
        raise ValueError('Archivo SWIM POI JSON inválido.')
    for p in data['points']:
        if not isinstance(p, dict) or not all(k in p for k in ('id', 'lat', 'lon', 'consistency')):
            raise ValueError('Marcador SWIM incompleto.')
        if not (-90 <= float(p['lat']) <= 90 and -180 <= float(p['lon']) <= 180
                and -1 <= float(p['consistency']) <= 1):
            raise ValueError('Marcador SWIM con coordenadas o consistencia inválidas.')
    return data
