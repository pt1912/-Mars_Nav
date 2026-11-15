"""Visualize georeferenced SWIM4MIM ice-consistency index as an RGBA raster.

NOT temperature, NOT probability, NOT precision EVA obstacle data.
Input: preprocessed Arcadia grid produced by scripts/prepare_swim_heatmap.py.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[1]
RASTER = ROOT / "data/regions/arcadia_planitia/swim_heatmap_0_1.npz"

@lru_cache(maxsize=1)
def get_swim_heatmap():
    if not RASTER.is_file():
        raise FileNotFoundError(
            "Falta el raster SWIM procesado. Ejecuta: "
            "python scripts/prepare_swim_heatmap.py "
            "--data-dir /tmp/twin-worlds/public/data"
        )
    with np.load(RASTER, allow_pickle=False) as data:
        scores = np.asarray(data["scores"], dtype=np.float64)
        bounds = np.asarray(data["bounds"], dtype=np.float64)

    if scores.ndim != 2 or scores.size == 0 or bounds.shape != (2, 2):
        raise ValueError("Archivo SWIM incompatible: formas inválidas.")
    if not np.all(np.isfinite(bounds)) or bounds[0,0] >= bounds[1,0] or bounds[0,1] >= bounds[1,1]:
        raise ValueError("Límites geográficos SWIM inválidos.")
    valid = np.isfinite(scores) & (scores >= -1.0) & (scores <= 1.0)
    if not np.any(valid):
        raise ValueError("La región SWIM no contiene datos válidos.")

    # Escala visual explícita. Aumentar la densidad de píxeles NO añade resolución.
    color_min, color_max = 0.45, 0.80
    normalized = np.clip((np.nan_to_num(scores, nan=color_min) - color_min) /
                         (color_max - color_min), 0.0, 1.0)
    rgba = (colormaps["YlGnBu"](normalized) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 245, 0).astype(np.uint8)
    return rgba, bounds.tolist(), {
        "min": float(np.min(scores[valid])),
        "max": float(np.max(scores[valid])),
        "valid_cells": int(np.count_nonzero(valid)),
        "total_cells": int(scores.size),
        "visual_min": color_min,
        "visual_max": color_max,
    }

if __name__ == "__main__":
    img, box, stats = get_swim_heatmap()
    print("SWIM RGBA:", img.shape, img.dtype)
    print("Bounds:", box)
    print("Stats:", stats)
