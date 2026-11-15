"""USGS SIM 3292 ArcGIS tiles controlled by the Streamlit sidebar.

The ArcGIS Mars tiled service uses a northward row offset relative to
Leaflet EPSG:4326. This adapter reindexes rows to align them correctly.
"""

from branca.element import MacroElement, Template


class USGSGeologyOverlay(MacroElement):
    def __init__(self, opacity=0.35):
        super().__init__()
        self._name = "USGSGeologyOverlay"
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self._template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            const map = {{ this._parent.get_name() }};
            const tileRoot =
                'https://tiles.arcgis.com/tiles/nzS0F0zdNLvs7nc8/' +
                'arcgis/rest/services/Mars_geology_All_layers_100pct/MapServer/tile/';

            const GeoGrid = L.GridLayer.extend({
                onAdd: function(map) {
                    L.GridLayer.prototype.onAdd.call(this, map);
                    this.setOpacity(this.options.opacity);
                },
                setOpacity: function(opacity) {
                    this.options.opacity = opacity;
                    const container = this.getContainer();
                    if (container) container.style.opacity = String(opacity);
                    return this;
                },
                createTile: function(coords, done) {
                    const z = coords.z;
                    if (z === 0) {
                        const canvas = document.createElement('canvas');
                        canvas.width = canvas.height = 256;
                        const ctx = canvas.getContext('2d');
                        let remaining = 2;
                        let failed = false;
                        const half = function(row, cropY, drawY) {
                            const img = new Image();
                            img.onload = function() {
                                if (failed) return;
                                ctx.drawImage(img, 0, cropY, 256, 128, 0, drawY, 256, 128);
                                if (--remaining === 0) done(null, canvas);
                            };
                            img.onerror = function(e) {
                                if (!failed) { failed = true; done(e, canvas); }
                            };
                            img.src = tileRoot + '1/' + row + '/' + coords.x;
                        };
                        half(0, 128, 0);
                        half(1, 0, 128);
                        return canvas;
                    }
                    const tile = document.createElement('img');
                    tile.alt = '';
                    tile.onload = function() { done(null, tile); };
                    tile.onerror = function(e) { done(e, tile); };
                    const sourceZoom = z + 1;
                    const sourceRow = coords.y + Math.pow(2, z - 1);
                    tile.src = tileRoot + sourceZoom + '/' + sourceRow + '/' + coords.x;
                    return tile;
                }
            });

            const geologyLayer = new GeoGrid({
                tileSize: 256,
                opacity: {{ this.opacity }},
                minZoom: 0,
                maxZoom: 12,
                maxNativeZoom: 5,
                noWrap: true,
                attribution: 'USGS SIM 3292 | Tanaka et al. (2014)'
            });
            // The parent is the dynamic Leaflet FeatureGroup.
            // This overlay exists only while the USGS checkbox is selected.
            geologyLayer.addTo(map);
        })();
        {% endmacro %}
        """)
