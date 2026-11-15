"""Análisis topográfico regional de Mars_Nav con MOLA y Horn 3x3.

Entrada: data/regions/arcadia_planitia/elevation.npz (elevación y
coordenadas 1D en grados). Las longitudes del recorte se expresan
0..360° E. NoData finito solo se interpreta si metadata.json lo declara.

El mapa de costos es un *multiplicador adimensional*: path_planning.py
lo combinará con la longitud física de cada movimiento en metros.
"""

from functools import lru_cache
from pathlib import Path
import json

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data/regions/arcadia_planitia"
ELEVATION_FILE = DATA_DIR / "elevation.npz"
METADATA_FILE = DATA_DIR / "metadata.json"
OUTPUT_DIR = BASE_DIR / "data/processed"

# Radio de referencia esférico utilizado para el análisis MOLA (metros).
# Unificar después este valor con mars_geometry.py para comparar rutas.
MARS_RADIUS_M = 3_396_000.0


def _validate_axis(values, name):
    """Comprueba eje angular 1D, monótono y de espaciamiento uniforme."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError(f"{name}: deben existir >=3 coordenadas 1D finitas.")
    increments = np.diff(values)
    if not (np.all(increments > 0) or np.all(increments < 0)):
        raise ValueError(f"{name}: coordenadas repetidas o no monótonas.")
    if not np.allclose(increments, increments[0], rtol=1e-5, atol=1e-10):
        raise ValueError(f"{name}: espaciamiento angular no uniforme.")
    if abs(increments[0]) >= 180:
        raise ValueError(f"{name}: discontinuidad angular sospechosa.")
    return values


@lru_cache(maxsize=1)
def load_terrain():
    """Lee NPZ y devuelve (elevation, latitudes, longitudes).

    Normaliza la orientación a: primera fila norte, primera columna oeste.
    Conserva NaN y convierte NoData *declarado en metadatos* a NaN.
    """
    if not ELEVATION_FILE.is_file():
        raise FileNotFoundError(f"No existe el DEM recortado: {ELEVATION_FILE}")

    with np.load(ELEVATION_FILE, allow_pickle=False) as data:
        elevation = np.array(data["elevation"], dtype=np.float64)
        lat = _validate_axis(data["latitudes"], "latitudes")
        lon = _validate_axis(data["longitudes"], "longitudes")

    if elevation.ndim != 2 or elevation.shape != (lat.size, lon.size):
        raise ValueError("Dimensiones del DEM incompatibles con sus coordenadas.")
    if np.any(np.abs(lat) >= 90):
        raise ValueError("No se admiten latitudes fuera de (-90°, 90°).")

    # No asumir que -32768 o -32000 son NoData si el producto no lo declara.
    if METADATA_FILE.is_file():
        with METADATA_FILE.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        nodata = metadata.get("no_data_value")
        if nodata is not None:
            nodata = float(nodata)
            if not np.isfinite(nodata):
                raise ValueError("El valor NoData del metadata debe ser finito.")
            elevation[elevation == nodata] = np.nan

    elevation[~np.isfinite(elevation)] = np.nan

    if lat[0] < lat[-1]:
        lat = lat[::-1].copy()
        elevation = elevation[::-1, :].copy()
    if lon[0] > lon[-1]:
        lon = lon[::-1].copy()
        elevation = elevation[:, ::-1].copy()

    return elevation, lat, lon


def calculate_gradients_horn(elevation, lat, lon):
    """Devuelve dz/dx (este) y dz/dy (norte), adimensionales.

    Horn 3x3 sobre retícula lat/lon uniforme, orientada norte-arriba.
    Las ventanas con alguna celda NaN y todos los bordes quedan NaN.
    """
    elevation = np.asarray(elevation, dtype=np.float64)
    lat = _validate_axis(lat, "latitudes")
    lon = _validate_axis(lon, "longitudes")

    if elevation.shape != (lat.size, lon.size):
        raise ValueError("Dimensiones de la elevación y coordenadas incompatibles.")
    if np.any(np.abs(lat) >= 90):
        raise ValueError("Latitudes fuera del rango permitido.")
    if not np.all(np.diff(lat) < 0) or not np.all(np.diff(lon) > 0):
        raise ValueError("Se requiere norte-arriba y oeste-a-este: usar load_terrain().")

    dy = MARS_RADIUS_M * np.deg2rad(lat[0] - lat[1])  # metros, positivo
    dx_rows = (
        MARS_RADIUS_M * np.cos(np.deg2rad(lat))
        * np.deg2rad(lon[1] - lon[0])
    )
    if dy <= 0 or np.any(dx_rows <= 0):
        raise ValueError("Espaciamiento métrico inválido.")

    z1, z2, z3 = elevation[:-2, :-2], elevation[:-2, 1:-1], elevation[:-2, 2:]
    z4, z5, z6 = elevation[1:-1, :-2], elevation[1:-1, 1:-1], elevation[1:-1, 2:]
    z7, z8, z9 = elevation[2:, :-2], elevation[2:, 1:-1], elevation[2:, 2:]

    valid_window = (
        np.isfinite(z1) & np.isfinite(z2) & np.isfinite(z3)
        & np.isfinite(z4) & np.isfinite(z5) & np.isfinite(z6)
        & np.isfinite(z7) & np.isfinite(z8) & np.isfinite(z9)
    )

    gx_inner = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (
        8.0 * dx_rows[1:-1, None]
    )
    # Las filas aumentan hacia el SUR; norte menos sur conserva el signo norte.
    gy_inner = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8.0 * dy)

    gx = np.full(elevation.shape, np.nan, dtype=np.float64)
    gy = np.full(elevation.shape, np.nan, dtype=np.float64)
    gx[1:-1, 1:-1] = np.where(valid_window, gx_inner, np.nan)
    gy[1:-1, 1:-1] = np.where(valid_window, gy_inner, np.nan)
    return gx, gy


def calculate_slope(elevation, lat, lon):
    """Devuelve inclinación Horn 3x3 en grados; NaN si no es evaluable."""
    gx, gy = calculate_gradients_horn(elevation, lat, lon)
    return np.rad2deg(np.arctan(np.hypot(gx, gy)))


def analyze_terrain(max_slope_deg=15.0, slope_weight=2.0,
                    slope_reference_deg=5.0):
    """Genera pendientes, transitabilidad y factores de costo para A*.

    Umbrales experimentales, NO certificados para astronautas o rovers.
    cost_grid es un multiplicador (>0), no una distancia ni un tiempo.
    """
    if not np.isfinite(max_slope_deg) or not (0 < max_slope_deg < 90):
        raise ValueError("max_slope_deg debe estar entre 0° y 90°.")
    if not np.isfinite(slope_weight) or slope_weight < 0:
        raise ValueError("slope_weight debe ser finito y >= 0.")
    if not np.isfinite(slope_reference_deg) or slope_reference_deg <= 0:
        raise ValueError("slope_reference_deg debe ser finito y > 0.")

    elevation, lat, lon = load_terrain()
    gx, gy = calculate_gradients_horn(elevation, lat, lon)
    slope = np.rad2deg(np.arctan(np.hypot(gx, gy)))

    valid = np.isfinite(elevation) & np.isfinite(slope)
    evaluated = int(np.count_nonzero(valid))
    if evaluated == 0:
        raise ValueError("No existen celdas con pendiente evaluable.")

    traversable = valid & (slope <= max_slope_deg)
    cost_grid = np.full(elevation.shape, np.inf, dtype=np.float64)
    cost_grid[traversable] = (
        1.0 + slope_weight
        * (slope[traversable] / slope_reference_deg) ** 2
    )
    finite_elevation = elevation[np.isfinite(elevation)]
    stats = {
        "elevation_min_m": float(np.min(finite_elevation)),
        "elevation_max_m": float(np.max(finite_elevation)),
        "elevation_mean_m": float(np.mean(finite_elevation)),
        "slope_min_deg": float(np.min(slope[valid])),
        "slope_max_deg": float(np.max(slope[valid])),
        "slope_mean_deg": float(np.mean(slope[valid])),
        "evaluated_cells": evaluated,
        "traversable_cells": int(np.count_nonzero(traversable)),
        "unevaluated_cells": int(elevation.size - evaluated),
        "coverage_percent": float(100 * evaluated / elevation.size),
        "traversable_percent": float(100 * np.count_nonzero(traversable) / evaluated),
    }
    return {
        "elevation": elevation,
        "latitudes": lat,
        "longitudes": lon,
        "dzdx_east": gx,
        "dzdy_north": gy,
        "slope": slope,
        "valid": valid,
        "traversable": traversable,
        "cost_grid": cost_grid,
        "stats": stats,
    }


def generate_maps(elevation, slope, latitudes, longitudes):
    """Conserva exportación PNG que utilizabas para revisar los resultados."""
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dlat = abs(latitudes[1] - latitudes[0]); dlon = abs(longitudes[1] - longitudes[0])
    extent = [
        longitudes[0] - dlon/2, longitudes[-1] + dlon/2,
        latitudes[-1] - dlat/2, latitudes[0] + dlat/2,
    ]
    aspect = 1.0 / np.cos(np.deg2rad(np.mean(latitudes)))
    for values, cmap, unit, filename in [
        (elevation, "terrain", "Elevation (m)", "arcadia_elevation.png"),
        (slope, "magma", "Slope (degrees)", "arcadia_slope.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                       origin="upper", extent=extent, aspect=aspect)
        fig.colorbar(im, ax=ax, label=unit)
        ax.set_title("Mars_Nav — Arcadia Planitia — " + unit)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        fig.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    result = analyze_terrain()
    print("\n========== MARS_NAV — HORN 3x3 ==========")
    print(f"Región: Arcadia Planitia | DEM: {result['elevation'].shape}")
    for key, value in result["stats"].items():
        print(f"{key}: {value:.4f}" if isinstance(value, float)
              else f"{key}: {value}")
    generate_maps(result["elevation"], result["slope"],
                  result["latitudes"], result["longitudes"])
    print("Mapas: data/processed/arcadia_elevation.png y arcadia_slope.png")
    print("Umbral: 15° experimental, no certificado para EVA.")
