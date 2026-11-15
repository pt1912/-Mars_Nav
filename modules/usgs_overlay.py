"""Overlay for USGS SIM 3292 geology on Leaflet EPSG:4326.

ArcGIS Mars 2000 tiles start at latitude +180°, while Leaflet EPSG:4326
covers +90°..-90° in the vertical view. The row shift aligns both grids.
The layer is optional; Viking stays the basemap.
"""

from branca.element import MacroElement, Template


class USGSGeologyOverlay(MacroElement):
    """Geologic map tiles on top of Viking with adjustable transparency."""

    def __init__(self, opacity=0.25):
        super().__init__()
        self._name = "USGSGeologyOverlay"
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self._template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            const map = {{ this._parent.get_name() }};
            const base =
                'https://tiles.arcgis.com/tiles/nzS0F0zdNLvs7nc8/' +
                'arcgis/rest/services/Mars_geology_All_layers_100pct/MapServer/tile/';
            const layer = L.GridLayer.extend({
                createTile: function(coords, done) {
                    const z = coords.z;
                    if (z === 0) {
                        const canvas = document.createElement('canvas');
                        canvas.width = canvas.height = 256;
                        canvas.style.pointerEvents = 'none';
                        const ctx = canvas.getContext('2d');
                        let remaining = 2;
                        let hasError = false;
                        function half(row, sourceY, targetY) {
                            const image = new Image();
                            image.onload = function() {
                                if (hasError) return;
                                ctx.drawImage(image, 0, sourceY, 256, 128,
                                    0, targetY, 256, 128);
                                if (--remaining === 0) done(null, canvas);
                            };
                            image.onerror = function(error) {
                                if (!hasError) {
                                    hasError = true;
                                    done(error, canvas);
                                }
                            };
                            image.src = base + '1/' + row + '/' + coords.x;
                        }
                        half(0, 128, 0);
                        half(1, 0, 128);
                        return canvas;
                    }
                    const tile = document.createElement('img');
                    tile.alt = '';
                    tile.style.pointerEvents = 'none';
                    tile.onload = () => done(null, tile);
                    tile.onerror = (error) => done(error, tile);
                    const level = z + 1;
                    const row = coords.y + Math.pow(2, z - 1);
                    tile.src = base + level + '/' + row + '/' + coords.x;
                    return tile;
                }
            });
            new layer({
                tileSize: 256,
                minZoom: 0,
                maxZoom: 12,
                maxNativeZoom: 5,
                noWrap: true,
                opacity: {{ this.opacity }},
                attribution: 'USGS SIM 3292 | Tanaka et al. (2014)'
            }).addTo(map);
        })();
        {% endmacro %}
        """)
