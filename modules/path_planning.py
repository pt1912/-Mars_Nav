"""Planificación regional de rutas sobre MOLA para Mars_Nav.

Los multiplicadores de `cost_grid` son adimensionales; el coste de cada
movimiento es distancia física (m) x promedio de multiplicadores extremos.
Las rutas son aproximaciones de escala MOLA, NO rutas EVA certificadas.

Dependencias: NumPy y la biblioteca estándar de Python.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any, Mapping

import numpy as np

from modules.terrain_analysis import MARS_RADIUS_M, analyze_terrain

NEIGHBORS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


def _great_circle_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia de círculo máximo en una esfera marciana (coordenadas en radianes)."""
    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + math.pi) % (2 * math.pi) - math.pi
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    a = min(1.0, max(0.0, a))
    return 2 * MARS_RADIUS_M * math.asin(math.sqrt(a))


def coordinate_to_cell(lat_deg: float, lon_deg: float,
                       latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[int, int]:
    """Convierte coordenadas 0..360 E o -180..180 E a la celda DEM más próxima.

    Rechaza coordenadas fuera de los bordes de píxel de la cuadrícula.
    No desplaza automáticamente el punto a una celda transitable.
    """
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError("Las coordenadas del DEM deben ser vectores 1D.")
    if latitudes.size < 2 or longitudes.size < 2:
        raise ValueError("El DEM requiere al menos dos filas y columnas.")
    if not (np.isfinite(lat_deg) and np.isfinite(lon_deg)):
        raise ValueError("Las coordenadas seleccionadas deben ser finitas.")
    if not (np.all(np.diff(latitudes) < 0) and np.all(np.diff(longitudes) > 0)):
        raise ValueError("La cuadrícula debe estar orientada norte-arriba y oeste-este.")

    # Elegir la representación equivalente más cercana al centro longitudinal.
    lon_mid = (longitudes[0] + longitudes[-1]) / 2
    lon_adjusted = lon_deg + 360 * round((lon_mid - lon_deg) / 360)
    half_lat = abs(latitudes[1] - latitudes[0]) / 2
    half_lon = abs(longitudes[1] - longitudes[0]) / 2
    if not (latitudes[-1] - half_lat <= lat_deg <= latitudes[0] + half_lat
            and longitudes[0] - half_lon <= lon_adjusted <= longitudes[-1] + half_lon):
        raise ValueError("El punto seleccionado está fuera del DEM MOLA recortado.")
    row = int(np.argmin(np.abs(latitudes - lat_deg)))
    col = int(np.argmin(np.abs(longitudes - lon_adjusted)))
    return row, col


def cells_to_leaflet(path: list[tuple[int, int]], latitudes: np.ndarray,
                     longitudes: np.ndarray) -> list[list[float]]:
    """Transforma celdas de una ruta en posiciones [lat, lon] de Folium."""
    return [[float(latitudes[r]), float((longitudes[c] + 180) % 360 - 180)]
            for r, c in path]


def _search(cost_grid: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray,
            start: tuple[int, int], goal: tuple[int, int],
            *, heuristic_enabled: bool, max_expansions: int = 300_000) -> dict[str, Any]:
    costs = np.asarray(cost_grid, dtype=np.float64)
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)

    if costs.ndim != 2 or costs.shape != (len(latitudes), len(longitudes)):
        raise ValueError("Dimensiones de cost_grid y coordenadas incompatibles.")
    if len(latitudes) < 2 or len(longitudes) < 2:
        raise ValueError("El DEM debe tener al menos dos filas y columnas.")
    if not np.all(np.isfinite(latitudes)) or not np.all(np.isfinite(longitudes)):
        raise ValueError("Los vectores geográficos deben ser finitos.")
    if not (np.all(np.diff(latitudes) < 0) and np.all(np.diff(longitudes) > 0)):
        raise ValueError("Se necesita norte-arriba y oeste-este.")
    if not isinstance(max_expansions, int) or max_expansions <= 0:
        raise ValueError("max_expansions debe ser entero positivo.")
    if np.any(np.isnan(costs)) or np.any(np.isneginf(costs)) or np.any(costs <= 0):
        raise ValueError("Los costos deben ser positivos o +inf para celdas bloqueadas.")

    height, width = costs.shape
    try:
        sr, sc = (int(start[0]), int(start[1]))
        gr, gc = (int(goal[0]), int(goal[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("start y goal deben ser (fila, columna).") from exc
    if not (0 <= sr < height and 0 <= sc < width and 0 <= gr < height and 0 <= gc < width):
        raise ValueError("Inicio o destino fuera del DEM.")
    if not np.isfinite(costs[sr, sc]) or not np.isfinite(costs[gr, gc]):
        raise ValueError("Inicio o destino está en una celda no transitable.")

    # El factor de coste mínimo garantiza que la heurística es admisible.
    min_factor = float(np.min(costs[np.isfinite(costs)]))
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)

    def h(r: int, c: int) -> float:
        if not heuristic_enabled:
            return 0.0
        return min_factor * _great_circle_m(
            float(lat_rad[r]), float(lon_rad[c]),
            float(lat_rad[gr]), float(lon_rad[gc]),
        )

    start_idx = sr * width + sc
    goal_idx = gr * width + gc
    g_score = np.full(height * width, np.inf)
    parent = np.full(height * width, -1, dtype=np.int64)
    g_score[start_idx] = 0.0
    count = itertools.count()
    queue = [(h(sr, sc), next(count), 0.0, start_idx)]
    expanded = 0

    while queue:
        _f, _order, g, idx = heapq.heappop(queue)
        if g > g_score[idx] + 1e-8:
            continue  # Entrada obsoleta de la cola
        if idx == goal_idx:
            break
        expanded += 1
        if expanded > max_expansions:
            raise RuntimeError(
                "Límite de exploración alcanzado. Selecciona una ruta menor "
                "o aumenta max_expansions."
            )
        r, c = divmod(idx, width)
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not np.isfinite(costs[nr, nc]):
                continue
            if dr and dc:
                # No cruzar en diagonal entre celdas bloqueadas.
                if not (np.isfinite(costs[r + dr, c])
                        and np.isfinite(costs[r, c + dc])):
                    continue
            length_m = _great_circle_m(
                float(lat_rad[r]), float(lon_rad[c]),
                float(lat_rad[nr]), float(lon_rad[nc]),
            )
            multiplier = 0.5 * (costs[r, c] + costs[nr, nc])
            candidate = g + length_m * multiplier
            next_idx = nr * width + nc
            if candidate + 1e-9 < g_score[next_idx]:
                g_score[next_idx] = candidate
                parent[next_idx] = idx
                heapq.heappush(queue, (candidate + h(nr, nc), next(count), candidate, next_idx))
    else:
        raise ValueError("No existe una ruta transitable entre esos dos puntos.")

    path = []
    idx = goal_idx
    while idx != -1:
        path.append(divmod(int(idx), width))
        idx = int(parent[idx])
    path.reverse()
    if path[0] != (sr, sc):
        raise RuntimeError("Error interno al reconstruir la ruta.")

    distance_m = 0.0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        distance_m += _great_circle_m(
            float(lat_rad[r1]), float(lon_rad[c1]),
            float(lat_rad[r2]), float(lon_rad[c2]),
        )
    return {
        "path": path,
        "distance_m": float(distance_m),
        "cost_m": float(g_score[goal_idx]),
        "expanded_nodes": int(expanded),
        "start_cell": (sr, sc),
        "goal_cell": (gr, gc),
    }


def astar(cost_grid: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray,
          start: tuple[int, int], goal: tuple[int, int],
          *, max_expansions: int = 300_000) -> dict[str, Any]:
    """A* con distancia marciana real y penalizaciones de terreno."""
    return _search(cost_grid, latitudes, longitudes, start, goal,
                   heuristic_enabled=True, max_expansions=max_expansions)


def dijkstra(cost_grid: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray,
             start: tuple[int, int], goal: tuple[int, int],
             *, max_expansions: int = 300_000) -> dict[str, Any]:
    """Dijkstra usa exactamente los mismos costos que A* para poder compararlos."""
    return _search(cost_grid, latitudes, longitudes, start, goal,
                   heuristic_enabled=False, max_expansions=max_expansions)


def route_metrics(result: dict[str, Any], terrain: dict[str, Any]) -> dict[str, float]:
    """Métricas aproximadas sobre los centros de celdas del DEM."""
    path = result["path"]
    elevations = terrain["elevation"]
    slopes = terrain["slope"]
    heights = np.array([elevations[r, c] for r, c in path], dtype=float)
    dz = np.diff(heights)
    return {
        "distance_km": result["distance_m"] / 1000,
        "cumulative_ascent_m": float(np.sum(np.clip(dz, 0, None))),
        "cumulative_descent_m": float(np.sum(np.clip(-dz, 0, None))),
        "net_elevation_change_m": float(heights[-1] - heights[0]),
        "max_cell_slope_deg": float(np.max([slopes[r, c] for r, c in path])),
        "route_cost_m": float(result["cost_m"]),
    }



def plan_route_from_points(
    start_point: Mapping[str, float],
    destination_point: Mapping[str, float],
    *,
    algorithm: str = "A*",
    max_slope_deg: float = 15.0,
    slope_weight: float = 2.0,
    max_expansions: int = 300_000,
) -> dict[str, Any]:
    """Planifica entre dos marcadores de Streamlit/Folium.

    Ejemplo de entrada::

        start_point = {"lat": 46.20, "lon": -171.20}
        destination_point = {"lat": 46.25, "lon": -171.15}

    Las longitudes se aceptan tanto en [-180, 180] como en [0, 360].
    No utiliza puntos predefinidos: los recibe en cada llamada.
    La ruta devuelta conecta *centros* de celdas MOLA, no posiciones
    de precisión peatonal. Umbrales y costos son experimentales.
    """
    if algorithm not in ("A*", "Dijkstra"):
        raise ValueError("Algoritmo desconocido. Usa 'A*' o 'Dijkstra'.")

    def parse_marker(point: Mapping[str, float], name: str) -> tuple[float, float]:
        if not isinstance(point, Mapping):
            raise ValueError(f"{name}: se espera un punto con 'lat' y 'lon'.")
        try:
            lat = float(point["lat"])
            lon = float(point["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}: coordenadas 'lat'/'lon' inválidas.") from exc
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise ValueError(f"{name}: coordenadas no finitas.")
        return lat, lon

    start_lat, start_lon = parse_marker(start_point, "Start Point")
    dest_lat, dest_lon = parse_marker(destination_point, "Destination")

    terrain = analyze_terrain(
        max_slope_deg=max_slope_deg,
        slope_weight=slope_weight,
    )
    latitudes = terrain["latitudes"]
    longitudes = terrain["longitudes"]

    start_cell = coordinate_to_cell(start_lat, start_lon, latitudes, longitudes)
    goal_cell = coordinate_to_cell(dest_lat, dest_lon, latitudes, longitudes)

    planner = astar if algorithm == "A*" else dijkstra
    result = planner(
        terrain["cost_grid"],
        latitudes,
        longitudes,
        start_cell,
        goal_cell,
        max_expansions=max_expansions,
    )
    return {
        "algorithm": algorithm,
        "coordinates": cells_to_leaflet(result["path"], latitudes, longitudes),
        "path": result["path"],
        "start_cell": start_cell,
        "goal_cell": goal_cell,
        "metrics": route_metrics(result, terrain),
        "expanded_nodes": result["expanded_nodes"],
    }

def _smoke_test() -> None:
    """Pruebas sintéticas sin depender de datasets NASA."""
    lat = np.linspace(46.01, 45.99, 9)
    lon = np.linspace(188.79, 188.81, 9)
    cost = np.ones((9, 9))
    cost[4, 1:8] = np.inf
    cost[4, 5] = 1  # Única abertura del obstáculo
    start, goal = (1, 1), (7, 7)
    a = astar(cost, lat, lon, start, goal)
    d = dijkstra(cost, lat, lon, start, goal)
    assert math.isclose(a["cost_m"], d["cost_m"], abs_tol=1e-5)
    assert a["path"][0] == start and a["path"][-1] == goal
    assert (4, 5) in a["path"]
    # Prueba de corte en diagonal: inicio aislado por celdas ortogonales.
    blocked = np.ones((3, 3))
    blocked[0, 1] = np.inf
    blocked[1, 0] = np.inf
    try:
        astar(blocked, lat[:3], lon[:3], (0, 0), (1, 1))
    except ValueError as exc:
        assert "No existe" in str(exc)
    else:
        raise AssertionError("Se permitió cortar una esquina bloqueada.")
    print("Pruebas sintéticas: OK (A* = Dijkstra y sin corte de esquinas)")
    print(f"Ruta sintética: {len(a['path'])} celdas, {a['distance_m']/1000:.2f} km")


def _test_arcadia() -> None:
    terrain = analyze_terrain()
    lat, lon = terrain["latitudes"], terrain["longitudes"]
    # Puntos de prueba en el sector central de Arcadia, en formato 0..360 E.
    start = coordinate_to_cell(46.20, 188.80, lat, lon)
    goal = coordinate_to_cell(46.25, 188.85, lat, lon)
    route = astar(terrain["cost_grid"], lat, lon, start, goal)
    metrics = route_metrics(route, terrain)
    print(f"Arcadia: inicio {start}, destino {goal}, celdas {len(route['path'])}")
    print(f"Distancia sobre centros de celdas: {metrics['distance_km']:.2f} km")
    print(f"Ascenso: {metrics['cumulative_ascent_m']:.1f} m; pendiente máxima: {metrics['max_cell_slope_deg']:.2f}°")
    print(f"Costo acumulado: {metrics['route_cost_m']:.1f} m equivalentes de costo")


if __name__ == "__main__":
    _smoke_test()
    try:
        _test_arcadia()
    except FileNotFoundError:
        print("No se encontró elevation.npz. Prueba Arcadia omitida; pruebas sintéticas completas.")
