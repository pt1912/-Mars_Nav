import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.mola_dem import load_region_dem, point_rc, grid_lat_lon
from modules.eva_route_engine import RouteSettings, plan_eva_routes, terrain_maps


class TestEVAPlanner(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        # 128 ppd MOLA-like regional topography with a smooth hill near center.
        y, x = np.mgrid[0:110, 0:110]
        elev = 180 * np.exp(-((x-55)**2 + (y-55)**2)/(2*9**2))
        elev = elev.astype(np.float32) + 0.4*y
        meta = dict(site_id='arcadia', north_edge=47.0, west_edge_east=188.0,
                    resolution_ppd=128, south_edge=47.0 - 110 / 128,
                    east_edge_east=188.0 + 110 / 128, partial_source_coverage=False)
        np.savez_compressed(root/'arcadia_mola.npz', elevation_m=elev, metadata_json=json.dumps(meta))
        self.output_dir = root
        self.meta = meta

    def tearDown(self):
        load_region_dem.cache_clear()
        self.folder.cleanup()

    def test_georeference_roundtrip(self):
        lat, lon = grid_lat_lon(25, 30, self.meta)
        self.assertEqual(point_rc(lat, lon, self.meta, (110, 110)), (25, 30))

    def test_slope_risk_values(self):
        elev, _ = load_region_dem('arcadia', self.output_dir)
        slope, rough, valid, risk = terrain_maps(elev, 320., 460., 15., 20.)
        self.assertTrue(valid[25,25])
        self.assertTrue(np.isfinite(risk[25,25]))
        self.assertGreaterEqual(float(np.nanmax(slope)), 0.)
        self.assertGreaterEqual(float(np.nanmin(risk)), 0.)
        self.assertLessEqual(float(np.nanmax(risk)), 1.)

    def test_three_routes(self):
        a = dict(zip(('lat','lon'), grid_lat_lon(20, 20, self.meta)))
        b = dict(zip(('lat','lon'), grid_lat_lon(85, 85, self.meta)))
        config = RouteSettings(max_slope_deg=15., map_margin_km=5., max_grid_cells=14000)
        output = plan_eva_routes('arcadia', a, b, config, self.output_dir)
        self.assertEqual(set(output['routes']), {'Shortest Route', 'Safest Route','Balanced Route'})
        self.assertGreater(output['routes']['Shortest Route']['stats']['distance_km'], 0)
        self.assertLessEqual(output['routes']['Shortest Route']['stats']['distance_km'],
                             output['routes']['Safest Route']['stats']['distance_km'] + 0.01)
        for route in output['routes'].values():
            self.assertGreaterEqual(route['stats']['mean_terrain_risk'], 0)
            self.assertGreater(len(route['coordinates']), 2)

    def test_missing_coverage_rejected(self):
        a = {'lat': 46.5, 'lon': -171.5}
        b = {'lat': 50, 'lon': 20}
        with self.assertRaises(ValueError):
            plan_eva_routes('arcadia', a, b, RouteSettings(), self.output_dir)


if __name__ == '__main__':
    unittest.main()
