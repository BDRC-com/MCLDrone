#!/usr/bin/env python3
"""Build a MapDB index from an offline XYZ/TMS tile tree.

This scans a directory layout like:
  <tiles_root>/<z>/<x>/<y>.<ext>

and writes a SQLite DB containing per-tile geographic bounds in EPSG:4326.

No third-party Python dependencies.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TileRow:
    z: int
    x: int
    y: int
    rel_path: str
    top_left_lat: float
    top_left_lon: float
    bottom_right_lat: float
    bottom_right_lon: float

    @property
    def tile_id(self) -> str:
        return f"{self.z}/{self.x}/{self.y}"


def _clamp_lat(lat_deg: float) -> float:
    return max(min(lat_deg, 85.05112878), -85.05112878)


def _xyz_tile_bounds4326(
    z: int, x: int, y_xyz: int
) -> Tuple[float, float, float, float]:
    """Return (top_lat, left_lon, bottom_lat, right_lon) in degrees for XYZ tiles."""
    n = 2.0**z

    left_lon = x / n * 360.0 - 180.0
    right_lon = (x + 1) / n * 360.0 - 180.0

    def lat_for_y(y: int) -> float:
        lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
        return _clamp_lat(math.degrees(lat_rad))

    top_lat = lat_for_y(y_xyz)
    bottom_lat = lat_for_y(y_xyz + 1)

    return top_lat, left_lon, bottom_lat, right_lon


def _tile_bounds4326(
    z: int, x: int, y: int, scheme: str
) -> Tuple[float, float, float, float]:
    scheme = scheme.lower().strip()
    if scheme not in {"xyz", "tms"}:
        raise ValueError(f"Unsupported scheme: {scheme!r} (expected 'xyz' or 'tms')")

    if scheme == "xyz":
        y_xyz = y
    else:
        # TMS uses origin at bottom-left.
        # Convert TMS y to XYZ y.
        n = 2**z
        y_xyz = (n - 1) - y

    return _xyz_tile_bounds4326(z, x, y_xyz)


def _parse_bbox4326(values: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError(
            "bbox4326 must have 4 numbers: min_lat min_lon max_lat max_lon"
        )
    min_lat, min_lon, max_lat, max_lon = (float(v) for v in values)
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon
    return min_lat, min_lon, max_lat, max_lon


def _intersects_bbox(
    *,
    tile_top_lat: float,
    tile_left_lon: float,
    tile_bottom_lat: float,
    tile_right_lon: float,
    bbox: Tuple[float, float, float, float],
) -> bool:
    min_lat, min_lon, max_lat, max_lon = bbox

    # Normalize tile bounds (top might be < bottom in southern hemisphere)
    t_min_lat = min(tile_top_lat, tile_bottom_lat)
    t_max_lat = max(tile_top_lat, tile_bottom_lat)
    t_min_lon = min(tile_left_lon, tile_right_lon)
    t_max_lon = max(tile_left_lon, tile_right_lon)

    # AABB intersection
    if t_max_lat < min_lat or t_min_lat > max_lat:
        return False
    if t_max_lon < min_lon or t_min_lon > max_lon:
        return False
    return True


def _iter_tiles(
    tiles_root: Path, image_exts: Sequence[str]
) -> Iterator[Tuple[Path, int, int, int]]:
    # Layout: root/z/x/y.ext
    # We avoid recursive globbing on the whole tree by iterating z/x dirs.
    for z_dir in sorted([p for p in tiles_root.iterdir() if p.is_dir()]):
        if not z_dir.name.isdigit():
            continue
        z = int(z_dir.name)
        for x_dir in sorted([p for p in z_dir.iterdir() if p.is_dir()]):
            if not x_dir.name.isdigit():
                continue
            x = int(x_dir.name)
            for ext in image_exts:
                for y_path in sorted(x_dir.glob(f"*.{ext}")):
                    stem = y_path.stem
                    if not stem.isdigit():
                        continue
                    y = int(stem)
                    yield y_path, z, x, y


def _tile_pk(z: int, x: int, y: int) -> int:
    # Deterministic 64-bit key for (z,x,y), safe for z well beyond 20.
    return (int(z) << 52) | (int(x) << 26) | int(y)


def _init_sqlite_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(db_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
          k TEXT PRIMARY KEY,
          v TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tiles (
          id INTEGER PRIMARY KEY,
          tile_id TEXT NOT NULL UNIQUE,
          z INTEGER NOT NULL,
          x INTEGER NOT NULL,
          y INTEGER NOT NULL,
          scheme TEXT NOT NULL,
          rel_path TEXT NOT NULL,
          top_left_lat REAL NOT NULL,
          top_left_lon REAL NOT NULL,
          bottom_right_lat REAL NOT NULL,
          bottom_right_lon REAL NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS tiles_rtree USING rtree(
          id,
          min_lat, max_lat,
          min_lon, max_lon
        );

        CREATE INDEX IF NOT EXISTS idx_tiles_z ON tiles(z);
        CREATE INDEX IF NOT EXISTS idx_tiles_zxy ON tiles(z, x, y);
        """
    )

    conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('schema_version','1');")
    conn.commit()
    return conn


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build MapDB tile_meta.sqlite from an offline tile tree"
    )
    p.add_argument(
        "--tiles-root", type=Path, required=True, help="Tile root directory (z/x/y.ext)"
    )
    p.add_argument(
        "--scheme",
        type=str,
        default="xyz",
        choices=["xyz", "tms"],
        help="Tile scheme for y axis. xyz=origin top-left, tms=origin bottom-left.",
    )
    p.add_argument(
        "--ext",
        type=str,
        default="png,jpg,jpeg",
        help="Comma-separated image extensions to include (default: png,jpg,jpeg)",
    )
    p.add_argument(
        "--zoom-min",
        type=int,
        default=-1,
        help="Minimum zoom to include (default: no min)",
    )
    p.add_argument(
        "--zoom-max",
        type=int,
        default=-1,
        help="Maximum zoom to include (default: no max)",
    )
    p.add_argument(
        "--bbox4326",
        type=float,
        nargs=4,
        default=None,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        help="Optional EPSG:4326 bbox filter; only tiles intersecting bbox are indexed",
    )
    p.add_argument(
        "--out-sqlite",
        type=Path,
        default=Path("results/mapdb/tile_meta.sqlite"),
        help="Output tile_meta.sqlite path (default: results/mapdb/tile_meta.sqlite)",
    )

    args = p.parse_args()

    tiles_root: Path = args.tiles_root
    if not tiles_root.exists():
        raise SystemExit(f"tiles-root not found: {tiles_root}")

    image_exts = [e.strip().lstrip(".") for e in str(args.ext).split(",") if e.strip()]

    bbox = _parse_bbox4326(args.bbox4326) if args.bbox4326 is not None else None

    counts_by_zoom: Dict[int, int] = {}

    # Track union bbox for sanity
    union_min_lat: Optional[float] = None
    union_min_lon: Optional[float] = None
    union_max_lat: Optional[float] = None
    union_max_lon: Optional[float] = None

    out_db: Path = args.out_sqlite
    conn = _init_sqlite_db(out_db)
    cur = conn.cursor()

    tile_rows_to_insert: List[Tuple] = []
    rtree_rows_to_insert: List[Tuple] = []
    batch_size = 50_000

    def flush() -> None:
        if not tile_rows_to_insert:
            return
        cur.executemany(
            """
            INSERT OR REPLACE INTO tiles(
              id, tile_id, z, x, y, scheme, rel_path,
              top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            tile_rows_to_insert,
        )
        cur.executemany(
            "INSERT OR REPLACE INTO tiles_rtree(id,min_lat,max_lat,min_lon,max_lon) VALUES(?,?,?,?,?)",
            rtree_rows_to_insert,
        )
        conn.commit()
        tile_rows_to_insert.clear()
        rtree_rows_to_insert.clear()

    total = 0
    for path, z, x, y in _iter_tiles(tiles_root, image_exts):
        if args.zoom_min >= 0 and z < args.zoom_min:
            continue
        if args.zoom_max >= 0 and z > args.zoom_max:
            continue

        top_lat, left_lon, bottom_lat, right_lon = _tile_bounds4326(
            z, x, y, args.scheme
        )

        if bbox is not None:
            if not _intersects_bbox(
                tile_top_lat=top_lat,
                tile_left_lon=left_lon,
                tile_bottom_lat=bottom_lat,
                tile_right_lon=right_lon,
                bbox=bbox,
            ):
                continue

        rel = path.relative_to(tiles_root).as_posix()
        tid = f"{z}/{x}/{y}"
        pk = _tile_pk(z, x, y)

        r_min_lat = min(top_lat, bottom_lat)
        r_max_lat = max(top_lat, bottom_lat)
        r_min_lon = min(left_lon, right_lon)
        r_max_lon = max(left_lon, right_lon)

        tile_rows_to_insert.append(
            (
                pk,
                tid,
                int(z),
                int(x),
                int(y),
                str(args.scheme),
                rel,
                float(top_lat),
                float(left_lon),
                float(bottom_lat),
                float(right_lon),
            )
        )
        rtree_rows_to_insert.append((pk, r_min_lat, r_max_lat, r_min_lon, r_max_lon))
        total += 1
        counts_by_zoom[z] = counts_by_zoom.get(z, 0) + 1

        union_min_lat = (
            r_min_lat if union_min_lat is None else min(union_min_lat, r_min_lat)
        )
        union_min_lon = (
            r_min_lon if union_min_lon is None else min(union_min_lon, r_min_lon)
        )
        union_max_lat = (
            r_max_lat if union_max_lat is None else max(union_max_lat, r_max_lat)
        )
        union_max_lon = (
            r_max_lon if union_max_lon is None else max(union_max_lon, r_max_lon)
        )

        if len(tile_rows_to_insert) >= batch_size:
            flush()

    flush()
    conn.execute(
        "INSERT OR REPLACE INTO meta(k,v) VALUES('tiles_indexed', ?)",
        (str(total),),
    )
    if union_min_lat is not None:
        conn.execute(
            "INSERT OR REPLACE INTO meta(k,v) VALUES('union_bbox4326', ?)",
            (
                f"[{union_min_lat:.8f}, {union_min_lon:.8f}, {union_max_lat:.8f}, {union_max_lon:.8f}]",
            ),
        )
    conn.commit()
    conn.close()

    tmp = out_db.with_suffix(out_db.suffix + ".tmp")
    if tmp.exists():
        tmp.replace(out_db)

    print(f"tiles_root: {tiles_root}")
    print(f"scheme: {args.scheme}")
    print(f"tiles_indexed: {total}")
    if counts_by_zoom:
        for z in sorted(counts_by_zoom):
            print(f"  z={z}: {counts_by_zoom[z]}")

    if union_min_lat is not None:
        print(
            "union_bbox4326: "
            f"[{union_min_lat:.8f}, {union_min_lon:.8f}, {union_max_lat:.8f}, {union_max_lon:.8f}]"
        )

    print(f"out_sqlite: {out_db}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
