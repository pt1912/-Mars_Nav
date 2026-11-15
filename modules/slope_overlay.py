"""Capa georreferenciada RGBA de pendientes Horn para Mars_Nav.

Módulo de visualización: no determina seguridad EVA ni recalcula pendientes.
`get_slope_overlay()` conserva la interfaz (imagen, límites, estadísticas)
esperada por app.py.
"""

from functools import lru_cache

import numpy as np
from matplotlib import colormaps

from modules.terrain_analysis import analyze_terrain

# Estos umbrales son exclusivamente VISUALES, no límites de seguridad EVA.
SLOPE_MIN_DISPLAY_DEG = 0.3
SLOPE_MAX_DISPLAY_DEG = 5.0


def _to_leaflet_longitudes(longitudes):
    """Convierte 0-360 E a [-180, 180) y rechaza una discontinuidad.

    Arcadia Planitia (186-191° E) corresponde a ~-174 a -169°.
    Para una futura región que cruce el meridiano 180° se necesitará
    un tratamiento especial (dividir capa o cambiar origen del mapa).
    """
    longitude = np.asarray(longitudes, dtype=np.float64)
    converted = ((longitude + 180.0) % 360.0) - 180.0
    if converted.size < 2 or not np.all(np.diff(converted) > 0):
        raise ValueError(
            "Las longitudes cruzan una discontinuidad o no están ordenadas; "
            "el overlay debe prepararse por sectores."
        )
    return converted


@lru_cache(maxsize=1)
def get_slope_overlay():
    """Devuelve (rgba_uint8, leaflet_bounds, stats) para Folium ImageOverlay.

    Las celdas sin pendiente evaluable (incluidos los bordes Horn)
    tienen transparencia 0. No convierte los NaN en datos científicos.
    """
    result = analyze_terrain()
    slope = result["slope"]
    valid = result["valid"]
    lat = result["latitudes"]
    lon = _to_leaflet_longitudes(result["longitudes"])

    if not np.any(valid):
        raise ValueError("No existen pendientes válidas para representar.")

    # Convertir NaN solo para aplicar colores, sin modificar la matriz original.
    display_slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(display_slope / SLOPE_MAX_DISPLAY_DEG, 0.0, 1.0)
    rgba = np.rint(colormaps["magma"](normalized) * 255).astype(np.uint8)

    # Los umbrales cromáticos son para visibilidad, NO transitabilidad.
    shown = valid & (slope >= SLOPE_MIN_DISPLAY_DEG)
    rgba[..., 3] = np.where(shown, 220, 0).astype(np.uint8)

    # npz guarda coordenadas en los centros de píxeles.
    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    bounds = [
        [float(lat[-1] - dlat / 2), float(lon[0] - dlon / 2)],
        [float(lat[0] + dlat / 2), float(lon[-1] + dlon / 2)],
    ]

    metrics = result["stats"]
    stats = {
        "min_slope": metrics["slope_min_deg"],
        "max_slope": metrics["slope_max_deg"],
        "mean_slope": metrics["slope_mean_deg"],
        "min_elevation": metrics["elevation_min_m"],
        "max_elevation": metrics["elevation_max_m"],
    }
    return rgba, bounds, stats


if __name__ == "__main__":
    overlay, limits, stats = get_slope_overlay()
    print("MARS_NAV — SLOPE OVERLAY")
    print("RGBA:", overlay.shape, overlay.dtype)
    print("Límites Leaflet:", limits)
    print("Píxeles visibles:", int(np.count_nonzero(overlay[..., 3])))
    print("Pendiente máxima Horn:", f"{stats['max_slope']:.4f}°")
