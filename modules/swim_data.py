"""Validated, cached SWIM4MIM-derived Int8 raster reader.

Data format: Twin Worlds 0.2-degree global derivative of PSI SWIM4MIM.
Values are signed integers × scale, NOT probabilities of buried ice.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'swim'
DEPTHS = {'Ci_0_1': '0–1 m', 'Ci_1_5': '1–5 m'}


@dataclass(frozen=True)
class SwimGrid:
    values: np.ndarray
    valid: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    bounds: list
    resolution_deg: float
    depth: str


def load_grid(depth: str = 'Ci_0_1', *, data_dir=None) -> SwimGrid:
    """Read the SWIM grid with a single cache shared by overlays and search.

    Restart Streamlit (or call _load_grid_cached.cache_clear()) after
    replacing the binary files on disk.
    """
    if depth not in DEPTHS:
        raise ValueError(f'Unsupported SWIM depth: {depth}')
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return _load_grid_cached(depth, str(root.resolve()))


@lru_cache(maxsize=8)
def _load_grid_cached(depth: str, root_path: str) -> SwimGrid:
    root = Path(root_path)
    manifest = root / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing SWIM manifest: {manifest}')
    try:
        details = json.loads(manifest.read_text(encoding='utf-8'))['swim'][depth]['grid']
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('Invalid SWIM manifest') from exc

    width, height = int(details['width']), int(details['height'])
    res = float(details['res'])
    scale = float(details['scale'])
    xmin, ymax = float(details['xmin']), float(details['ymax'])
    nodata = int(details['nodata'])
    filename = str(details['file'])
    if not (1 <= width <= 10000 and 1 <= height <= 10000):
        raise ValueError('Unsupported SWIM raster dimensions')
    if not (0 < res <= 2 and 0 < scale <= 1):
        raise ValueError('Invalid SWIM scale or resolution')
    if not (-128 <= nodata <= 127 and np.isfinite([xmin, ymax, res, scale]).all()):
        raise ValueError('Invalid SWIM metadata values')
    if not (filename == Path(filename).name and filename.endswith('.i8')):
        raise ValueError('Invalid SWIM binary filename')
    if not (-180 <= xmin <= 180 and -90 <= ymax <= 90):
        raise ValueError('Invalid SWIM geographic origin')
    if (xmin + width * res > 180.0001 or ymax - height * res < -90.0001):
        raise ValueError('SWIM geographic extent exceeds planetary bounds')

    binary = root / filename
    if not binary.is_file():
        raise FileNotFoundError(f'Missing SWIM raster: {binary}')
    if binary.stat().st_size != width * height:
        raise ValueError(f'SWIM binary size != {width}×{height} bytes')

    raw = np.fromfile(binary, dtype=np.int8).reshape(height, width)
    values = raw.astype(np.float32) * scale
    valid = (raw != nodata) & np.isfinite(values) & (values >= -1) & (values <= 1)
    latitudes = ymax - (np.arange(height) + 0.5) * res
    longitudes = xmin + (np.arange(width) + 0.5) * res
    bounds = [[ymax - height * res, xmin], [ymax, xmin + width * res]]

    for matrix in (values, valid, latitudes, longitudes):
        matrix.flags.writeable = False
    return SwimGrid(values, valid, latitudes, longitudes, bounds, res, depth)
