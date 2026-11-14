import json
import numpy as np

from pathlib import Path

from modules.mars_elevation import (
    DEM_PATH,
    ROWS,
    COLS,
    PIXELS_PER_DEGREE
)


# ==========================================
# ARCADIA PLANITIA CONFIGURATION
# ==========================================

REGION_NAME = "Arcadia Planitia"

LAT_MIN = 44.5079794
LAT_MAX = 47.8921384

# Longitudes marcianas: 0 - 360 E
LON_MIN = 186.354857
LON_MAX = 191.2451392

OUTPUT_DIR = Path("data/regions/arcadia_planitia")


# ==========================================
# EXTRACT MOLA REGION
# ==========================================

def prepare_arcadia():

    print("\n====================================")
    print("   MARS_NAV - MOLA DATA PREPARATION")
    print("====================================")

    if not DEM_PATH.exists():
        raise FileNotFoundError(
            f"MOLA file not found: {DEM_PATH}"
        )

    # Convert geographic coordinates
    # into MOLA matrix indices

    row_start = int(
        np.floor((88 - LAT_MAX) * PIXELS_PER_DEGREE)
    )

    row_end = int(
        np.ceil((88 - LAT_MIN) * PIXELS_PER_DEGREE)
    )

    col_start = int(
        np.floor((LON_MIN - 180) * PIXELS_PER_DEGREE)
    )

    col_end = int(
        np.ceil((LON_MAX - 180) * PIXELS_PER_DEGREE)
    )

    # Validate indices

    if not (
        0 <= row_start < row_end <= ROWS
        and 0 <= col_start < col_end <= COLS
    ):
        raise ValueError(
            "Selected region exceeds MOLA coverage."
        )

    # Read NASA MOLA data

    mola = np.memmap(
        DEM_PATH,
        dtype=">i2",
        mode="r",
        shape=(ROWS, COLS)
    )

    # Extract only Arcadia Planitia

    elevation = np.array(
        mola[
            row_start:row_end,
            col_start:col_end
        ],
        dtype=np.int16
    )

    # Generate geographic coordinates
    # corresponding to pixel centers

    latitudes = 88 - (
        np.arange(row_start, row_end) + 0.5
    ) / PIXELS_PER_DEGREE

    longitudes = 180 + (
        np.arange(col_start, col_end) + 0.5
    ) / PIXELS_PER_DEGREE

    # ======================================
    # SAVE NUMERICAL DATA
    # ======================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_DIR / "elevation.npz",
        elevation=elevation,
        latitudes=latitudes,
        longitudes=longitudes
    )

    # ======================================
    # SAVE METADATA
    # ======================================

    metadata = {
        "region": REGION_NAME,
        "planet": "Mars",
        "source": "NASA MGS MOLA MEGDR",
        "source_file": DEM_PATH.name,
        "instrument": "Mars Orbiter Laser Altimeter",
        "resolution_pixels_per_degree": PIXELS_PER_DEGREE,
        "elevation_unit": "meters",
        "longitude_convention": "0-360 east",
        "latitude_orientation": "north_to_south",
        "bounds": {
            "latitude_min": LAT_MIN,
            "latitude_max": LAT_MAX,
            "longitude_min": LON_MIN,
            "longitude_max": LON_MAX
        },
        "raster_bounds": {
            "latitude_min": 88 - row_end / PIXELS_PER_DEGREE,
            "latitude_max": 88 - row_start / PIXELS_PER_DEGREE,
            "longitude_min": 180 + col_start / PIXELS_PER_DEGREE,
            "longitude_max": 180 + col_end / PIXELS_PER_DEGREE
        },
        "shape": {
            "rows": int(elevation.shape[0]),
            "columns": int(elevation.shape[1])
        }
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # ======================================
    # RESULTS
    # ======================================

    print(f"\nRegion: {REGION_NAME}")
    print(f"Matrix shape: {elevation.shape}")

    print(
        f"Minimum elevation: "
        f"{elevation.min()} m"
    )

    print(
        f"Maximum elevation: "
        f"{elevation.max()} m"
    )

    print("\nFiles generated:")

    for file in OUTPUT_DIR.iterdir():

        size_kb = file.stat().st_size / 1024

        print(
            f"{file.name}: {size_kb:.2f} KB"
        )

    print("\nMOLA REGION READY")
    print("====================================")


if __name__ == "__main__":
    prepare_arcadia()