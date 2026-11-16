"""MOLA MEGDR regional extraction and bounded loading.

The original binary is never loaded wholesale. PDS3 detached .LBL supplies shape,
16-bit sample type, dimensions and map extents. Region outputs are compressed NPZ.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "mola"
PROCESSED_DIR = ROOT / "data" / "processed" / "mola"
MARS_RADIUS_M = 3_396_000.0


def label_value(label: str, name: str, default=None):
    m = re.search(r"(?im)^\s*" + re.escape(name) + r"\s*=\s*([^\r\n]+)", label)
    if m is None:
        return default
    return m.group(1).split("/*", 1)[0].strip().strip('"')


def label_num(label, name, default=None):
    raw = label_value(label, name, default)
    if raw is None:
        return None
    if isinstance(raw, (float, int)):
        return float(raw)
    m = re.match(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", raw)
    if not m:
        raise ValueError(f"Invalid {name} in PDS label: {raw}")
    return float(m.group())


def inspect_megdr(img_path: Path):
    """Return metadata for nonpolar 128 pix/degree MOLA MEGT tiles."""
    img_path = Path(img_path)
    lbl = img_path.with_suffix('.lbl')
    if not lbl.is_file():
        matches = list(img_path.parent.glob(img_path.stem + '.LBL'))
        if matches:
            lbl = matches[0]
    if not (img_path.is_file() and lbl.is_file()):
        raise FileNotFoundError(f"Missing .img/.lbl pair: {img_path}")
    name_match = re.fullmatch(r"megt(\d{2})([ns])(\d{3})(h)b", img_path.stem.lower())
    if not name_match:
        raise ValueError(f"Unsupported MOLA filename: {img_path.name} (expected megt??n???hb.img)")
    lat_start, ns, lon_start, resolution = name_match.groups()
    if resolution != 'h':
        raise ValueError('Only 128 pixels/degree H tiles are supported in this release')
    north = float(lat_start) * (1 if ns == 'n' else -1)
    west = float(lon_start)
    body = lbl.read_text(encoding='latin-1')
    height = int(label_num(body, 'LINES'))
    width = int(label_num(body, 'LINE_SAMPLES'))
    sample_bits = int(label_num(body, 'SAMPLE_BITS'))
    sample_type = str(label_value(body, 'SAMPLE_TYPE', '')).upper()
    if sample_bits != 16 or sample_type not in ('MSB_INTEGER', 'SUN_INTEGER'):
        raise ValueError(f'Unexpected PDS sample encoding: {sample_bits} bits, {sample_type}')
    # MOLA is MSB-first signed 16-bit. Dimensions 5632 x 11520 for H tiles.
    if height != 44 * 128 or width != 90 * 128:
        raise ValueError(f'Unexpected 128-ppd tile dimensions: {height} x {width}')
    expected_size = height * width * 2
    if img_path.stat().st_size != expected_size:
        raise ValueError(f'IMG size mismatch: {img_path.stat().st_size} bytes, expected {expected_size}')
    scale = label_num(body, 'SCALING_FACTOR', 1.0)
    offset = label_num(body, 'OFFSET', 0.0)
    # This MUST be topography, not planetary radius or counts.
    return dict(img=img_path, lbl=lbl, north=north, south=north - 44,
                west=west, east=west + 90, ppd=128.0, height=height, width=width,
                dtype='>i2', scale=scale, offset=offset,
                sample_type=sample_type)


def build_region_cutout(site: dict, tiles: list, radius_km=350.0, output_dir=PROCESSED_DIR):
    """Crop a rectangular search area around a site from one tile, never extrapolating.

    When a region spans multiple source tiles, use the tile containing its center.
    Missing coverage is explicit and the route engine refuses points outside.
    """
    lat = float(site['lat'])
    lon_e = float(site['lon']) % 360
    tile = next((x for x in tiles if x['south'] <= lat < x['north']
                 and x['west'] <= lon_e < x['east']), None)
    if tile is None:
        return None
    pad_lat = math.degrees(radius_km * 1000 / MARS_RADIUS_M)
    pad_lon = pad_lat / max(math.cos(math.radians(lat)), 0.05)
    south = max(tile['south'], lat - pad_lat)
    north = min(tile['north'], lat + pad_lat)
    west = max(tile['west'], lon_e - pad_lon)
    east = min(tile['east'], lon_e + pad_lon)
    ppd = tile['ppd']
    r0 = max(0, int(math.floor((tile['north'] - north) * ppd)))
    r1 = min(tile['height'], int(math.ceil((tile['north'] - south) * ppd)))
    c0 = max(0, int(math.floor((west - tile['west']) * ppd)))
    c1 = min(tile['width'], int(math.ceil((east - tile['west']) * ppd)))
    if r1 <= r0 or c1 <= c0:
        return None
    original = np.memmap(tile['img'], mode='r', dtype='>i2', shape=(tile['height'], tile['width']))
    raw = np.asarray(original[r0:r1, c0:c1], dtype=np.int16)
    elevations = raw.astype(np.float32) * tile['scale'] + tile['offset']
    # MEGDR's missing-value convention; don't confuse low elevations with nulls.
    elevations[raw == -32768] = np.nan
    if not np.isfinite(elevations).any():
        raise ValueError(f'All elevations are missing for {site["id"]}')
    plausible = elevations[np.isfinite(elevations)]
    if abs(float(np.nanmedian(plausible))) > 25000:
        raise ValueError('Unrealistic MOLA elevations. Check label SCALING_FACTOR/OFFSET.')
    meta = {
        'site_id': site['id'], 'site_name': site['name'],
        'source_img': tile['img'].name, 'source_lbl': tile['lbl'].name,
        'north_edge': tile['north'] - r0 / ppd,
        'west_edge_east': tile['west'] + c0 / ppd,
        'resolution_ppd': ppd,
        'south_edge': tile['north'] - r1 / ppd,
        'east_edge_east': tile['west'] + c1 / ppd,
        'radius_requested_km': float(radius_km),
        'partial_source_coverage': bool(
            south > lat - pad_lat + 1e-6 or north < lat + pad_lat - 1e-6 or
            west > lon_e - pad_lon + 1e-6 or east < lon_e + pad_lon - 1e-6
        ),
        'description': 'MOLA MEGDR 128 pixel/degree topography in meters; regional analysis only',
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{site['id']}_mola.npz"
    np.savez_compressed(out, elevation_m=elevations, metadata_json=json.dumps(meta))
    del original
    return out, meta


@lru_cache(maxsize=6)
def load_region_dem(site_id: str, output_dir=None):
    if not re.fullmatch('[a-z0-9_-]+', site_id):
        raise ValueError('Invalid site ID')
    out = Path(output_dir) if output_dir is not None else PROCESSED_DIR
    path = out / f'{site_id}_mola.npz'
    if not path.is_file():
        raise FileNotFoundError(f'No regional DEM available: {path}. Run scripts/build_mola_regions.py')
    with np.load(path, allow_pickle=False) as npz:
        elev = np.asarray(npz['elevation_m'], dtype=np.float32)
        meta = json.loads(str(npz['metadata_json']))
    if elev.ndim != 2 or min(elev.shape) < 3:
        raise ValueError('Invalid DEM window')
    return elev, meta


def normalize_lon_east(lon):
    return float(lon) % 360.0


def point_rc(lat, lon, meta, shape):
    """Nearest pixel center, without wrapping longitude to another image."""
    ppd = float(meta['resolution_ppd'])
    e = normalize_lon_east(lon)
    r = int(round((float(meta['north_edge']) - float(lat)) * ppd - 0.5))
    c = int(round((e - float(meta['west_edge_east'])) * ppd - 0.5))
    if not (0 <= r < shape[0] and 0 <= c < shape[1]):
        raise ValueError('Waypoint is outside the available MOLA regional cutout')
    return r, c


def grid_lat_lon(row, col, meta):
    ppd = float(meta['resolution_ppd'])
    lat = float(meta['north_edge']) - (row + 0.5) / ppd
    lon_e = float(meta['west_edge_east']) + (col + 0.5) / ppd
    lon = ((lon_e + 180) % 360) - 180
    return float(lat), float(lon)
