"""Recorta la cuadrícula SWIM4MIM derivada de Twin Worlds para el visor Mars_Nav.

Uso desde la raíz Mars_Nav:
  python scripts/prepare_swim_heatmap.py --data-dir /tmp/twin-worlds/public/data

Fuente original: SWIM4MIM / Planetary Science Institute.
Derivado: https://github.com/faresnagyeleissawy/twin-worlds
Los píxeles son promedios cuantizados; NO son probabilidades ni temperaturas.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/regions/arcadia_planitia/swim_heatmap_0_1.npz"
CTX = (44.5079794, -173.645143, 47.8921384, -168.7548608)  # S W N E


def prepare(data_dir: Path, output: Path = OUT, depth: str = "Ci_0_1"):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf8"))
    g = manifest["swim"][depth]["grid"]
    w, h = int(g["width"]), int(g["height"])
    xmin, ymax, res = float(g["xmin"]), float(g["ymax"]), float(g["res"])
    nodata, scale = int(g["nodata"]), float(g["scale"])
    if not (w > 0 and h > 0 and res > 0 and scale > 0 and -128 <= nodata <= 127):
        raise ValueError("Metadatos SWIM no válidos")
    f = data_dir / Path(str(g["file"])).name
    if f.stat().st_size != w*h:
        raise ValueError("Tamaño del archivo .i8 distinto a las dimensiones esperadas")
    data = np.fromfile(f, dtype=np.int8).reshape(h, w)
    s, west, n, east = CTX
    # Recortar CÉLULAS QUE INTERSECTAN la cobertura CTX; conservar sus bordes reales.
    r0 = max(0, int(np.floor((ymax - n) / res)))
    r1 = min(h, int(np.ceil((ymax - s) / res)))
    c0 = max(0, int(np.floor((west - xmin) / res)))
    c1 = min(w, int(np.ceil((east - xmin) / res)))
    if r1 <= r0 or c1 <= c0:
        raise ValueError("SWIM no cubre el sector CTX de Arcadia")
    raw = data[r0:r1, c0:c1]
    scores = raw.astype(np.float32) * scale
    scores[raw == nodata] = np.nan
    bounds = np.array([
        [ymax - r1*res, xmin + c0*res],
        [ymax - r0*res, xmin + c1*res],
    ], dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, scores=scores, bounds=bounds)
    valid = np.isfinite(scores)
    print("MARS_NAV — SWIM HEATMAP")
    print("Archivo:", output)
    print("Shape:", scores.shape, "dtype:", scores.dtype)
    print("Celdas con datos:", int(valid.sum()), "/", scores.size)
    print("Límites reales de las celdas:", bounds.tolist())
    if valid.any():
        print("Índice Ci:", float(np.min(scores[valid])), "a", float(np.max(scores[valid])))
    print("Interpretación: índice SWIM, no temperatura ni probabilidad de hielo")
    return scores, bounds


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/swim")
    p.add_argument("--depth", choices=["Ci_0_1", "Ci_1_5"], default="Ci_0_1")
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    prepare(a.data_dir, a.output, a.depth)
