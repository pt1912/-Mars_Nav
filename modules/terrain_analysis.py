import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from functools import lru_cache


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data/regions/arcadia_planitia"

ELEVATION_FILE = DATA_DIR / "elevation.npz"

OUTPUT_DIR = BASE_DIR / "data/processed"

# Mars radius in meters
MARS_RADIUS_M = 3396000


# ==========================================
# LOAD ELEVATION DATA
# ==========================================

@lru_cache(maxsize=1)
def load_terrain():

    if not ELEVATION_FILE.exists():
        raise FileNotFoundError(
            f"Elevation file not found: {ELEVATION_FILE}"
        )

    with np.load(ELEVATION_FILE) as data:

        elevation = data["elevation"].astype(np.float64)

        latitudes = data["latitudes"]

        longitudes = data["longitudes"]

    if elevation.shape != (
        len(latitudes),
        len(longitudes)
    ):
        raise ValueError(
            "Elevation matrix dimensions do not match coordinates."
        )

    return elevation, latitudes, longitudes


# ==========================================
# CALCULATE TERRAIN SLOPE
# ==========================================

def calculate_slope(elevation, latitudes, longitudes):

    # Angular resolution in radians
    lat_step = np.radians(
        abs(latitudes[1] - latitudes[0])
    )

    lon_step = np.radians(
        abs(longitudes[1] - longitudes[0])
    )

    # North-South pixel spacing
    dy = MARS_RADIUS_M * lat_step

    # East-West pixel spacing
    # Changes with latitude
    dx = (
        MARS_RADIUS_M
        * np.cos(np.radians(latitudes))
        * lon_step
    )

    # Elevation gradients
    dz_dy = np.gradient(
        elevation,
        axis=0
    ) / dy

    dz_dx = np.gradient(
        elevation,
        axis=1
    ) / dx[:, np.newaxis]

    # Slope magnitude
    gradient = np.sqrt(
        dz_dx**2 + dz_dy**2
    )

    # Convert to degrees
    slope = np.degrees(
        np.arctan(gradient)
    )

    return slope


# ==========================================
# GENERATE MAPS
# ==========================================

def generate_maps(
    elevation,
    slope,
    latitudes,
    longitudes
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    lat_step = abs(latitudes[1] - latitudes[0])
    lon_step = abs(longitudes[1] - longitudes[0])

    # Geographic boundaries of raster cells
    extent = [
        longitudes[0] - lon_step / 2,
        longitudes[-1] + lon_step / 2,
        latitudes[-1] - lat_step / 2,
        latitudes[0] + lat_step / 2
    ]

    # Approximate correction of geographic aspect ratio
    center_lat = np.mean(latitudes)

    map_aspect = 1 / np.cos(
        np.radians(center_lat)
    )

    # ======================================
    # ELEVATION MAP
    # ======================================

    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.imshow(
        elevation,
        cmap="terrain",
        origin="upper",
        extent=extent,
        aspect=map_aspect
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Elevation (m)"
    )

    ax.set_title(
        "Mars_Nav - Arcadia Planitia Elevation"
    )

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    fig.savefig(
        OUTPUT_DIR / "arcadia_elevation.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # SLOPE MAP
    # ======================================

    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.imshow(
        slope,
        cmap="magma",
        origin="upper",
        extent=extent,
        aspect=map_aspect
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Slope (degrees)"
    )

    ax.set_title(
        "Mars_Nav - Arcadia Planitia Slope"
    )

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    fig.savefig(
        OUTPUT_DIR / "arcadia_slope.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# MAIN TEST
# ==========================================

if __name__ == "__main__":

    print("\n========================================")
    print("    MARS_NAV - TERRAIN ANALYSIS")
    print("========================================")

    # Load optimized MOLA data
    elevation, latitudes, longitudes = load_terrain()

    # Calculate slope
    slope = calculate_slope(
        elevation,
        latitudes,
        longitudes
    )

    # Statistics
    print("\nRegion: Arcadia Planitia")
    print("Source: NASA MGS MOLA")

    print(f"\nMatrix shape: {elevation.shape}")

    print("\nELEVATION DATA")

    print(
        f"Minimum elevation: {np.min(elevation):.2f} m"
    )

    print(
        f"Maximum elevation: {np.max(elevation):.2f} m"
    )

    print(
        f"Mean elevation: {np.mean(elevation):.2f} m"
    )

    print("\nSLOPE DATA")

    print(
        f"Minimum slope: {np.min(slope):.4f}°"
    )

    print(
        f"Maximum slope: {np.max(slope):.4f}°"
    )

    print(
        f"Mean slope: {np.mean(slope):.4f}°"
    )

    # Generate PNG maps
    generate_maps(
        elevation,
        slope,
        latitudes,
        longitudes
    )

    print("\nMAPS GENERATED")

    print("data/processed/arcadia_elevation.png")
    print("data/processed/arcadia_slope.png")

    print("\n========================================")