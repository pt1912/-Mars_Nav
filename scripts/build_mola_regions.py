#!/usr/bin/env python3
"""Generate small, versionable regional MOLA cutouts from local raw IMG/LBL pairs.

Usage: python scripts/build_mola_regions.py --radius-km 350
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from modules.mola_dem import RAW_DIR, PROCESSED_DIR, inspect_megdr, build_region_cutout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius-km', type=float, default=350.0)
    args = parser.parse_args()
    if not 25 <= args.radius_km <= 2000:
        parser.error('radius-km must be between 25 and 2000')
    sources = sorted(RAW_DIR.glob('megt*.img')) + sorted(RAW_DIR.glob('MEGT*.IMG'))
    if not sources:
        sys.exit(f'No MOLA IMG in {RAW_DIR}; place corresponding .img and .lbl there.')
    tiles = [inspect_megdr(p) for p in sources]
    sites_file = PROJECT / 'data/locations/mars_sites.json'
    if not sites_file.exists():
        sys.exit(f'Missing {sites_file}')
    sites = json.loads(sites_file.read_text(encoding='utf-8'))
    report = []
    for site in sites:
        out = build_region_cutout(site, tiles, args.radius_km, PROCESSED_DIR)
        if out is None:
            print(f"SKIP {site['name']}: not covered by installed source IMG files.")
            report.append({'site': site['id'], 'status': 'missing source DEM'})
            continue
        filename, meta = out
        print(f"OK   {site['name']}: {filename} ({filename.stat().st_size/1e6:.2f} MB)"
              f"  partial={meta['partial_source_coverage']}")
        report.append({'site': site['id'], 'status': 'ready', 'file': filename.name,
                       'partial': meta['partial_source_coverage']})
    (PROCESSED_DIR / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nManifest: {PROCESSED_DIR / "manifest.json"}')
    print('Commit only data/processed/mola/*.npz and manifest.json; NEVER raw IMG/LBL.')


if __name__ == '__main__':
    main()
