"""Mars_Nav: extract research-candidate POIs from Twin Worlds' SWIM4MIM Int8 grid.

Source data: processed SWIM4MIM Ci grid (NOT original PSI GeoTIFF).
Grid metadata comes from public/data/manifest.json; north-up, row-major Int8.
Coordinates are pixel centers; output longitude is [-180, 180).

Usage, from Mars_Nav root:
    python scripts/prepare_swim_i8_poi.py
    python scripts/prepare_swim_i8_poi.py --min-score 0.3

Original SWIM4MIM: Planetary Science Institute / SWIM project.
Processed data: https://github.com/faresnagyeleissawy/twin-worlds
Markers are scientific research candidates, NOT confirmed ice sites or precise EVA targets.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/raw/swim"
DEFAULT_OUTPUT = ROOT / "data/regions/arcadia_planitia/swim_poi.json"

# Confirmed NASA Mars Trek CTX rectangle: south, west, north, east.
ARCADIA_CTX_BOUNDS = (44.5079794, -173.645143, 47.8921384, -168.7548608)
MARS_RADIUS_KM = 3396.0  # Same regional approximation as current terrain analysis.


def mars_distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    h = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * MARS_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def prepare_pois(data_dir, *, depth="Ci_0_1", min_score=0.5,
                 min_separation_km=18.0, max_points=10):
    data_dir = Path(data_dir)
    if not (-1.0 <= min_score <= 1.0):
        raise ValueError("--min-score debe estar entre -1 y 1.")
    if min_separation_km <= 0 or max_points < 1:
        raise ValueError("La separación debe ser positiva y max-points >= 1.")

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        grid = manifest["swim"][depth]["grid"]
        width, height = int(grid["width"]), int(grid["height"])
        xmin, ymax = float(grid["xmin"]), float(grid["ymax"])
        res, scale = float(grid["res"]), float(grid["scale"])
        nodata = int(grid["nodata"])
        filename = str(grid["file"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Metadatos incompletos para {depth} en manifest.json") from exc

    if width <= 0 or height <= 0 or res <= 0 or scale <= 0 or not (-128 <= nodata <= 127):
        raise ValueError("Dimensiones, escala o NoData inválidos en manifest.json")
    if "/" in filename or "\\" in filename or not filename.endswith(".i8"):
        raise ValueError("Nombre de archivo .i8 inesperado en manifest.json")
    binary_path = data_dir / filename
    if not binary_path.is_file():
        raise FileNotFoundError(f"No existe {binary_path}")
    if binary_path.stat().st_size != width * height:
        raise ValueError(
            f"Tamaño inválido de {filename}: se esperaban {width*height:,} bytes, "
            f"pero hay {binary_path.stat().st_size:,}."
        )

    # Source JS uses new Int8Array(buffer), row 0 = ymax; indexed row * width + col.
    raw = np.fromfile(binary_path, dtype=np.int8).reshape((height, width))
    valid = (raw != nodata)

    # Cell-center coordinates, not pixel boundaries. Latitude decreases by row.
    lats = ymax - (np.arange(height) + 0.5) * res
    lons = xmin + (np.arange(width) + 0.5) * res
    south, west, north, east = ARCADIA_CTX_BOUNDS
    roi_rows = np.flatnonzero((lats >= south) & (lats <= north))
    roi_cols = np.flatnonzero((lons >= west) & (lons <= east))
    if not len(roi_rows) or not len(roi_cols):
        raise ValueError("La cuadrícula SWIM no intersecta la cobertura CTX de Arcadia.")

    values = raw[np.ix_(roi_rows, roi_cols)].astype(np.float64) * scale
    good = valid[np.ix_(roi_rows, roi_cols)] & np.isfinite(values) & (values >= -1.0) & (values <= 1.0)
    qualifying = good & (values >= min_score)
    rr, cc = np.where(qualifying)
    candidates = [(float(values[r, c]), int(roi_rows[r]), int(roi_cols[c]))
                  for r, c in zip(rr, cc)]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    points = []
    for score, row, col in candidates:
        lat = float(lats[row]); lon = float(lons[col])
        if any(mars_distance_km(lat, lon, p["lat"], p["lon"]) < min_separation_km
               for p in points):
            continue
        points.append({
            "id": f"SWIM-{len(points) + 1:02d}",
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "consistency": round(score, 3),
            "depth_range": "0–1 m" if depth == "Ci_0_1" else "1–5 m",
            "type": "Candidate investigation waypoint (coarse SWIM raster cell)",
            "grid_row": row,
            "grid_col": col,
        })
        if len(points) >= max_points:
            break

    valid_scores = values[good]
    record = {
        "dataset": "SWIM4MIM",
        "product": "Combined Ice Consistency Map (0–1 m)" if depth == "Ci_0_1"
                   else "Combined Ice Consistency Map (1–5 m)",
        "source_url": "https://github.com/faresnagyeleissawy/twin-worlds",
        "original_source": "Planetary Science Institute (PSI) / SWIM4MIM, swim.psi.edu",
        "data_lineage": "Twin Worlds derivative: average 4x4 source samples, round(value*100), store signed Int8; scale=0.01",
        "interpretation": "Ice-consistency index, NOT a probability or a verified discovery; pixels are coarse research regions.",
        "selection": {
            "method": "Highest-scoring raster cell centers within Arcadia CTX bounds with minimum Mars-surface separation",
            "min_consistency": min_score,
            "min_separation_km": min_separation_km,
            "max_points": max_points,
        },
        "raster_info": {
            "file": filename,
            "shape": [height, width],
            "resolution_degrees": res,
            "scale": scale,
            "nodata": nodata,
            "arcadia_cells": int(values.size),
            "arcadia_valid_cells": int(good.sum()),
            "arcadia_above_threshold": int(qualifying.sum()),
            "arcadia_min_consistency": float(valid_scores.min()) if valid_scores.size else None,
            "arcadia_max_consistency": float(valid_scores.max()) if valid_scores.size else None,
            "bounds": {"south": south, "west": west, "north": north, "east": east},
        },
        "points": points,
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--depth", choices=("Ci_0_1", "Ci_1_5"), default="Ci_0_1")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--min-separation-km", type=float, default=18.0)
    parser.add_argument("--max-points", type=int, default=10)
    args = parser.parse_args()

    result = prepare_pois(args.data_dir, depth=args.depth, min_score=args.min_score,
                          min_separation_km=args.min_separation_km, max_points=args.max_points)
    stats = result["raster_info"]
    print("MARS_NAV — SWIM4MIM INT8")
    print("Producto:", result["product"])
    print("Celdas SWIM dentro de Arcadia:", stats["arcadia_cells"])
    print("Celdas con datos válidos:", stats["arcadia_valid_cells"])
    print("Valores superiores al umbral:", stats["arcadia_above_threshold"])
    print("Rango del índice válido:", stats["arcadia_min_consistency"], "a", stats["arcadia_max_consistency"])
    print("Puntos seleccionados:", len(result["points"]))
    for p in result["points"]:
        print(f"  {p['id']} | {p['lat']:.3f}°N, {p['lon']:.3f}° | Ci={p['consistency']:.2f}")

    output = args.output
    if output.exists():
        backup = output.with_name("swim_poi.before_i8.json")
        shutil.copy2(output, backup)
        print("Archivo anterior respaldado:", backup)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("JSON guardado:", output)
    if not result["points"]:
        print("Sin puntos con este umbral. Puedes probar --min-score 0.3; no implica hielo confirmado.")


if __name__ == "__main__":
    main()
