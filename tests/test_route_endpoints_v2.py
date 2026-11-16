import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules.eva_route_engine import RouteSettings, plan_eva_routes, terrain_maps, _neighborhood_std
from modules.mola_dem import load_region_dem


class TestRouteFixV2(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.location = Path(self.folder.name)
        yy, xx = np.mgrid[0:240, 0:250]
        # Broad ridge to encourage terrain-aware routing.
        z = 0.1*yy + 0.1*xx + 1000 * np.exp(-((yy-120)**2/1600 + (xx-125)**2/1600))
        metadata = {'site_id':'arcadia','north_edge':47.0,'west_edge_east':188.0,
                    'resolution_ppd':128,'south_edge':47.0 - 240/128,
                    'east_edge_east':188.0 + 250/128,'partial_source_coverage':False}
        np.savez_compressed(self.location/'arcadia_mola.npz', elevation_m=z.astype('float32'), metadata_json=json.dumps(metadata))

    def tearDown(self):
        load_region_dem.cache_clear()
        self.folder.cleanup()

    def test_exact_chosen_endpoints_even_with_coarse_grid(self):
        start = {'lat':46.735432, 'lon':-171.786543}
        end = {'lat':45.727654, 'lon':-170.388765}
        config = RouteSettings(max_slope_deg=25, map_margin_km=10, max_grid_cells=3800)
        plan = plan_eva_routes('arcadia', start, end, config, self.location)
        self.assertGreater(plan['stride'], 1)
        self.assertEqual(len(plan['routes']), 3)
        self.assertGreater(plan['endpoint_grid_offset_m'][0], 0)
        self.assertGreater(plan['endpoint_grid_offset_m'][1], 0)
        for name, route in plan['routes'].items():
            self.assertEqual(tuple(route['coordinates'][0]), (start['lat'], start['lon']), name)
            self.assertEqual(tuple(route['coordinates'][-1]), (end['lat'], end['lon']), name)
            self.assertGreater(route['stats']['distance_km'], 0)
        self.assertEqual(plan['hazard'].shape, plan['slope_deg'].shape)
        self.assertEqual(plan['hazard'].shape, plan['cost_safest'].shape)
        self.assertEqual(plan['hazard'].shape, plan['blocked'].shape)
        self.assertEqual(plan['scenario']['max_slope_deg'], 25)

    def test_detrending_flat_inclined_surface(self):
        y, x = np.mgrid[:28, :28]
        z = (15 + y * 3.0 + x * .8).astype('float32')
        rough = _neighborhood_std(z, 400, 400)
        self.assertLess(float(np.nanmax(rough[3:-3,3:-3])), .01)


if __name__ == '__main__':
    unittest.main()
