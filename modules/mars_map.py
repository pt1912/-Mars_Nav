"""Mars base imagery, region catalog, and clickable scientific site markers."""

import html
import json
from pathlib import Path

import folium



PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITES_FILE = PROJECT_ROOT / "data" / "locations" / "mars_sites.json"

VIKING_WMTS = (
    "https://trek.nasa.gov/tiles/Mars/EQ/"
    "Mars_Viking_MDIM21_ClrMosaic_global_232m/"
    "1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
)

GLOBAL_CENTER = (0.0, 0.0)
GLOBAL_ZOOM = 1
REGION_ZOOM = 5

CATEGORY_COLORS = {
    "Subsurface ice": "#38BDF8",
    "Robotic exploration": "#22C55E",
    "Terrain and geology": "#FB923C",
}


def load_sites():
    """Read the six region records and validate coordinates and categories."""
    if not SITES_FILE.is_file():
        raise FileNotFoundError(f"Missing region catalog: {SITES_FILE}")

    data = json.loads(SITES_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("mars_sites.json must contain a JSON list")

    required = {"id", "name", "lat", "lon", "category", "priority", "summary", "reference_note", "focus"}
    sites = {}

    for site in data:
        if not isinstance(site, dict) or not required.issubset(site):
            raise ValueError(f"Incomplete region entry: {site!r}")
        site_id = str(site["id"])
        lat, lon = float(site["lat"]), float(site["lon"])
        if site_id in sites or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"Duplicate ID or invalid coordinates: {site_id}")
        if site["category"] not in CATEGORY_COLORS:
            raise ValueError(f"Unknown category for {site_id}: {site['category']}")
        sites[site_id] = {**site, "lat": lat, "lon": lon}

    return sites


def create_mars_map():
    """Create only the static Viking basemap; region markers are added separately."""
    m = folium.Map(
        location=list(GLOBAL_CENTER),
        zoom_start=GLOBAL_ZOOM,
        tiles=None,
        crs="EPSG4326",
        min_zoom=0,
        max_zoom=12,
        zoom_snap=1,
        zoom_control=True,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles=VIKING_WMTS,
        name="NASA Viking | Mars Global Mosaic",
        attr="NASA Mars Trek / Viking Orbiter / USGS",
        max_native_zoom=7,
        max_zoom=12,
        no_wrap=True,
        control=False,
    ).add_to(m)
    return m


def _tooltip(site):
    def clean(key):
        return html.escape(str(site[key]))

    return folium.Tooltip(
        '<div style="width:310px;white-space:normal;line-height:1.5;'
        'font-family:Arial,sans-serif;font-size:12px;color:#1B2633;">'
        f'<strong style="font-size:14px;">{clean("name")}</strong><br>'
        f'<span style="color:#536478;">{clean("category")} · {clean("priority")}</span><br>'
        f'{site["lat"]:.2f}° latitude · {site["lon"]:.2f}° longitude<br><br>'
        f'{clean("summary")}<br><br>'
        f'<b>Research focus:</b> {clean("focus")}<br>'
        f'<span style="color:#677687;"><b>Location note:</b> {clean("reference_note")}</span>'
        '</div>',
        sticky=True,
    )


def create_site_markers(sites, selected_id=None):
    """Return a replaceable FeatureGroup, independent of the Viking map."""
    group = folium.FeatureGroup(name="Regions of Interest", control=False)
    for site_id, site in sites.items():
        selected = site_id == selected_id
        folium.CircleMarker(
            location=[site["lat"], site["lon"]],
            radius=8 if selected else 6,
            color="#F8FAFC" if selected else "#14212E",
            weight=2 if selected else 1.7,
            fill=True,
            fill_color=CATEGORY_COLORS[site["category"]],
            fill_opacity=1.0,
            tooltip=_tooltip(site),
        ).add_to(group)
    return group


def site_from_map_event(event, sites):
    """Resolve a site click without subscribing to zoom/pan events."""
    if not isinstance(event, dict) or not event.get("last_object_clicked"):
        return None

    tooltip = event.get("last_object_clicked_tooltip") or ""
    for site_id, site in sites.items():
        if site["name"] in tooltip:
            return site_id

    # Fallback for clients that report position but not tooltip text.
    point = event.get("last_object_clicked")
    if isinstance(point, dict) and "lat" in point and "lng" in point:
        try:
            lat, lon = float(point["lat"]), float(point["lng"])
        except (TypeError, ValueError):
            return None
        for site_id, site in sites.items():
            if abs(lat - site["lat"]) < 0.0001 and abs(lon - site["lon"]) < 0.0001:
                return site_id
    return None
