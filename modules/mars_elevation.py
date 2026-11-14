import numpy as np
from pathlib import Path

# Archivo oficial NASA MOLA
DEM_PATH = Path("data/raw/mola/megt88n180hb.img")

# Dimensiones publicadas por NASA
ROWS = 5632
COLS = 11520

# Resolución cartográfica
PIXELS_PER_DEGREE = 128


def get_elevation(lat, lon):
    """
    Obtiene la elevación MOLA correspondiente
    a unas coordenadas marcianas.

    lat: latitud en grados
    lon: longitud en grados (-180 a 180 o 0 a 360)

    Retorna elevación en metros.
    """

    lon = lon % 360

    if not (44 <= lat <= 88 and 180 <= lon < 270):
        raise ValueError("Coordinates outside DEM coverage")

    # Conversión de coordenadas a índices
    row = int((88 - lat) * PIXELS_PER_DEGREE)
    col = int((lon - 180) * PIXELS_PER_DEGREE)

    row = min(max(row, 0), ROWS - 1)
    col = min(max(col, 0), COLS - 1)

    # Datos PDS: enteros de 16 bits, big-endian
    dem = np.memmap(
        DEM_PATH,
        dtype=">i2",
        mode="r",
        shape=(ROWS, COLS)
    )

    elevation = int(dem[row, col])

    return elevation


if __name__ == "__main__":

    latitude = 46.2
    longitude = 188.8

    elevation = get_elevation(latitude, longitude)

    print("====================================")
    print("     MARS_NAV - MOLA ELEVATION")
    print("====================================")

    print(f"Latitude: {latitude}°")
    print(f"Longitude: {longitude}° E")
    print(f"Elevation: {elevation} m")

    print("====================================")