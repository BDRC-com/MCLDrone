#!/usr/bin/env python3
"""Pre-stage a deployment map from the offline tile store (step 6).

Operator flow: give a centre and a desired edge length, get a ready-to-fly
map + georeference sidecar + margin report — no mid-mission downloads.

    python3 prestage_map.py --center 22.8445297,114.5242310 --size 5

What it does:
  1. window  = the z<zoom> tiles covering `size` km centred on `--center`
     (window is rounded out to whole tiles, so the georeference stays on the
     tile grid: local px = global z<zoom> mercator px + geo_offset, with
     geo_offset = -tile_origin*256 — the convention mcl_node expects);
  2. download any MISSING tiles of that window (resumable; the local tile
     store is shared, so reruns only fetch what is new);
  3. refresh the MapDB sqlite index (only when new tiles arrived, or
     --rebuild-index) so the runtime MapDB sees them;
  4. stitch the window into `<out>` (PNG) + `<out>.json` sidecar (actual
     centre, gsd, tile origin, geo offsets, map_half_m, missing fraction);
  5. print the runtime parameters and a map-edge margin report (default
     margin 1 km, as required by the mission-planning constraint), optionally
     checking planned waypoints (map ENU metres: "x,y; x,y").

Run with a python that has numpy + opencv (system python3 works).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TILE = 256
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TILES = os.path.join(ROOT, 'huizhou_map', 'tiles')
DEFAULT_MAPDB = os.path.join(ROOT, 'huizhou_map', 'mapdb',
                             'huizhou_google.sqlite')
TILE_EXTS = ('jpg', 'png', 'jpeg')
# Satellite imagery is what the similarity model was trained on; the offline
# store is Google satellite (lyrs=s) as jpg.
PROVIDERS = {
    'google_sat': 'https://mt{server}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    'google_map': 'https://mt{server}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
    'osm': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
}
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'


# --------------------------------------------------------------- WEB MERCATOR
def lonlat_to_px(lon, lat, zoom):
    """Fractional global pixel coords of (lon, lat) at `zoom` (256 px tiles)."""
    n = 2.0 ** zoom * TILE
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def px_to_lonlat(x, y, zoom):
    n = 2.0 ** zoom * TILE
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def gsd_m_per_px(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


# --------------------------------------------------------------------- TILES
def tile_path(root, zoom, x, y):
    """First existing tile file for (x, y), or None."""
    for ext in TILE_EXTS:
        p = os.path.join(root, str(zoom), str(x), '%d.%s' % (y, ext))
        if os.path.exists(p):
            return p
    return None


def download_tile(root, zoom, x, y, provider, retries=3):
    """Fetch one satellite tile; saved as jpg like the existing store."""
    url = PROVIDERS[provider].format(server=(x + y) % 4, x=x, y=y, z=zoom)
    out = os.path.join(root, str(zoom), str(x), '%d.jpg' % y)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f:
                f.write(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 403):
                time.sleep(2 ** attempt * 2)
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(2 ** attempt)
    return None


def stitch(root, zoom, tx0, ty0, tiles_x, tiles_y):
    """Stitch the tile window; returns (canvas BGR, missing count)."""
    import cv2
    import numpy as np
    canvas = np.zeros((tiles_y * TILE, tiles_x * TILE, 3), np.uint8)
    missing = 0
    for iy in range(tiles_y):
        for ix in range(tiles_x):
            p = tile_path(root, zoom, tx0 + ix, ty0 + iy)
            if p is None:
                missing += 1
                continue
            t = cv2.imread(p, cv2.IMREAD_COLOR)
            if t is None or t.shape[0] != TILE or t.shape[1] != TILE:
                missing += 1
                continue
            canvas[iy * TILE:(iy + 1) * TILE, ix * TILE:(ix + 1) * TILE] = t
    return canvas, missing


def rebuild_index(tiles_root, mapdb_path):
    script = os.path.join(ROOT, 'gnss_free', 'gnss_free_core', 'mapdb',
                          'build_tile_index.py')
    cmd = [sys.executable, script, '--tiles-root', tiles_root,
           '--ext', ','.join(TILE_EXTS), '--out-sqlite', mapdb_path]
    print('refreshing MapDB index: %s' % ' '.join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------- MAIN
def parse_waypoints(text):
    wps = []
    for chunk in str(text).split(';'):
        chunk = chunk.strip()
        if chunk:
            x, y = chunk.split(',')
            wps.append((float(x), float(y)))
    return wps


def main():
    ap = argparse.ArgumentParser(
        description='Pre-stage a deployment map (download + stitch + '
                    'georeference + margin report)')
    ap.add_argument('--center', required=True,
                    help='map centre "LAT,LON" (deg, WGS-84)')
    ap.add_argument('--size', type=float, required=True,
                    help='desired map edge length [km]')
    ap.add_argument('--zoom', type=int, default=17)
    ap.add_argument('--tiles-root', default=DEFAULT_TILES)
    ap.add_argument('--mapdb', default=DEFAULT_MAPDB)
    ap.add_argument('--out', default=None,
                    help='output PNG (default: z<zoom>_<size>km_<lat>_<lon>.png)')
    ap.add_argument('--margin-m', type=float, default=1000.0,
                    help='map-edge margin kept clear for recovery [m]')
    ap.add_argument('--waypoints', default='',
                    help='planned waypoints "x,y; x,y" (map ENU m) to check')
    ap.add_argument('--provider', default='google_sat',
                    choices=sorted(PROVIDERS))
    ap.add_argument('--no-download', action='store_true',
                    help='fail instead of downloading missing tiles')
    ap.add_argument('--rebuild-index', action='store_true',
                    help='refresh the MapDB sqlite even without downloads')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    lat, lon = (float(v) for v in args.center.split(','))
    size_m = args.size * 1000.0
    gsd = gsd_m_per_px(lat, args.zoom)

    # tile window covering the requested square, rounded out to whole tiles
    cx, cy = lonlat_to_px(lon, lat, args.zoom)
    half = size_m / gsd / 2.0
    tx0 = int(math.floor((cx - half) / TILE))
    tx1 = int(math.floor((cx + half) / TILE))
    ty0 = int(math.floor((cy - half) / TILE))
    ty1 = int(math.floor((cy + half) / TILE))
    tiles_x, tiles_y = tx1 - tx0 + 1, ty1 - ty0 + 1
    size_px_x, size_px_y = tiles_x * TILE, tiles_y * TILE

    # actual centre = centre of the tile-aligned window
    clon, clat = px_to_lonlat(tx0 * TILE + size_px_x / 2.0,
                              ty0 * TILE + size_px_y / 2.0, args.zoom)
    size_m_x, size_m_y = size_px_x * gsd, size_px_y * gsd
    geo_off_x, geo_off_y = -tx0 * TILE, -ty0 * TILE

    print('requested centre  : %.7f, %.7f  size %.2f km @ z%d (%.4f m/px)'
          % (lat, lon, args.size, args.zoom, gsd))
    print('stitched window   : %d x %d tiles = %d x %d px = %.2f x %.2f km'
          % (tiles_x, tiles_y, size_px_x, size_px_y,
             size_m_x / 1000.0, size_m_y / 1000.0))
    print('actual centre     : %.7f, %.7f  (shift %.0f m)'
          % (clat, clon, math.hypot((clat - lat) * 111320.0,
                                    (clon - lon) * 111320.0 * math.cos(
                                        math.radians(lat)))))
    print('tile origin       : (%d, %d)  geo offsets (%d, %d)'
          % (tx0, ty0, geo_off_x, geo_off_y))

    # ---- 1) find / fetch missing tiles -------------------------------------
    todo = [(tx0 + ix, ty0 + iy)
            for iy in range(tiles_y) for ix in range(tiles_x)
            if tile_path(args.tiles_root, args.zoom, tx0 + ix, ty0 + iy) is None]
    print('tiles: %d present, %d missing'
          % (tiles_x * tiles_y - len(todo), len(todo)))
    downloaded = 0
    if todo and args.dry_run:
        print('dry-run: would download %d tiles from %s'
              % (len(todo), args.provider))
    elif todo:
        if args.no_download:
            raise SystemExit('%d tiles missing and --no-download set' % len(todo))
        print('downloading %d tiles from %s ...' % (len(todo), args.provider))
        with ThreadPoolExecutor(max_workers=4) as ex:
            for data in ex.map(lambda t: download_tile(args.tiles_root,
                                                       args.zoom, *t,
                                                       args.provider), todo):
                downloaded += data is not None
        print('downloaded %d/%d' % (downloaded, len(todo)))
        if downloaded < len(todo):
            raise SystemExit('some tiles failed; rerun to resume')

    if not args.dry_run and (downloaded or args.rebuild_index):
        rebuild_index(args.tiles_root, args.mapdb)

    # ---- 2) stitch ---------------------------------------------------------
    out = args.out or os.path.join(
        ROOT, 'z%d_%gkm_%.4f_%.4f.png'
        % (args.zoom, args.size, clat, clon))
    if args.dry_run:
        print('dry-run: would write %s + sidecar' % out)
        return 0
    canvas, missing = stitch(args.tiles_root, args.zoom, tx0, ty0,
                             tiles_x, tiles_y)
    import cv2
    cv2.imwrite(out, canvas)
    missing_frac = missing / float(tiles_x * tiles_y)
    print('wrote %s  (%.1f MB, %d tiles missing = %.1f%%)'
          % (out, os.path.getsize(out) / 1e6, missing, 100 * missing_frac))

    # ---- 3) sidecar --------------------------------------------------------
    sidecar = {
        'image': os.path.basename(out),
        'zoom': args.zoom,
        'gsd_m_per_px': gsd,
        'center_lat': clat, 'center_lon': clon,
        'requested_center_lat': lat, 'requested_center_lon': lon,
        'width': size_px_x, 'height': size_px_y,
        'size_m_x': size_m_x, 'size_m_y': size_m_y,
        'map_half_m': size_m_x / 2.0,
        'tile_x0': tx0, 'tile_y0': ty0,
        'tiles_x': tiles_x, 'tiles_y': tiles_y,
        'geo_offset_x': geo_off_x, 'geo_offset_y': geo_off_y,
        'missing_tile_frac': round(missing_frac, 4),
        'provider': args.provider,
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(out + '.json', 'w') as f:
        json.dump(sidecar, f, indent=1)
    print('wrote %s.json' % out)

    # ---- 4) margin report --------------------------------------------------
    usable = size_m_x / 2.0 - args.margin_m
    print('\nmap-edge margin report (margin %.0f m):' % args.margin_m)
    print('  usable half-extent: +-%.0f m from the map centre (%.1f km box)'
          % (usable, 2 * usable / 1000.0))
    ok = True
    for i, (wx, wy) in enumerate(parse_waypoints(args.waypoints)):
        ok_wp = abs(wx) <= usable and abs(wy) <= usable
        ok = ok and ok_wp
        print('  waypoint %d (%8.1f, %8.1f): %s (edge margin %.0f m)'
              % (i, wx, wy, 'OK' if ok_wp else 'OUTSIDE USABLE BOX',
                 min(size_m_x / 2.0 - abs(wx), size_m_y / 2.0 - abs(wy))))
    if missing_frac > 0.02:
        print('  WARNING: %.1f%% of tiles missing (black holes in the map)'
              % (100 * missing_frac))
    if parse_waypoints(args.waypoints) and not ok:
        print('  FAILED: waypoints violate the margin — widen --size or '
              'shorten the route')
        return 1

    print('\nruntime parameters:')
    print('  map_path:=%s' % out)
    print('  map_geo_offset_x:=%d map_geo_offset_y:=%d'
          % (geo_off_x, geo_off_y))
    print('  map_half_m:=%.0f' % (size_m_x / 2.0))
    print('  map_center_lat:=%.7f map_center_lon:=%.7f  (EKF2 origin)'
          % (clat, clon))
    print('  gsd: %.4f m/px' % gsd)
    return 0


if __name__ == '__main__':
    sys.exit(main())