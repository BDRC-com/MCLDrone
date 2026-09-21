"""Ground-truth evaluation of bag MCL runs against PX4 ulog state estimation.

The drone has NO GPS (GNSS-denied testbed), so truth comes from the FC EKF's
local position (optical-flow + IMU dead-reckoning) in 03_10_02.ulg (verified
as bag_0002's flight by yaw-rate correlation). This ulog comes from a custom
logger whose topic ids don't match the format table, so streams are keyed by
empirical ids:

  id 30 = vehicle_attitude      @20 Hz  (quaternion,   verified |q|=1)
  id 39 = vehicle_local_position@10 Hz  (verified: ref lat/lon, sane x/y/z/v)

The ulog clock has no valid UTC, so the bag↔flight time offset is pinned by
correlating quaternion-derived yaw-rate with the bag IMU gyro-z. Truth x/y
(NED, ENU-swapped for the map frame) is interpolated at each camera frame and
compared with MCL displacement from the first frame (start-anchored, so no
map georeference is needed).

Usage:
    python gt_eval.py RUN_DIR [RUN_DIR ...]      # each holds replay_log.npz
"""

import struct
import sqlite3
import sys
import numpy as np

ULOG = "/home/one/GNSS-denied-Localization/datasets/03_10_02.ulg"
DB3 = ("/home/one/GNSS-denied-Localization/datasets/"
       "bag_0002_20260824_190926/bag_0002_20260824_190926_0.db3")
CAMERA_STRIDE = 5
ID_ATT = 30
# vehicle_global_position @5 Hz, verified against 03_10_02.kml (user-extracted
# trace): payload = ts_sample(8) lat(double@8) lon(double@16) alt(float@24)
ID_GPOS = 36


def ulog_streams(path):
    """Return {msg_id: [(t_sec, payload_bytes), ...]}."""
    data = open(path, 'rb').read()
    pos, n = 16, len(data)
    out = {}
    while pos + 3 <= n:
        mlen = data[pos] | (data[pos + 1] << 8)
        mtype = data[pos + 2]
        body = data[pos + 3:pos + 3 + mlen]
        if mtype == ord('D'):
            mid = body[0] | (body[1] << 8)
            out.setdefault(mid, []).append(
                (struct.unpack('<Q', body[2:10])[0] / 1e6, body[10:]))
        pos += 3 + mlen
    return out


def bag_gyro_z(db3):
    db = sqlite3.connect(f'file:{db3}?mode=ro', uri=True)
    tid = dict(db.execute('SELECT name,id FROM topics'))['/imu0']
    from bag_reader import parse_imu
    rows = db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? '
                      'ORDER BY timestamp', (tid,)).fetchall()
    db.close()
    t = np.array([ts for ts, _ in rows]) / 1e9
    gz = np.array([parse_imu(b)[1][2] for _, b in rows])
    return t, gz


def attitude_yaw_rate(streams):
    """Yaw [rad] and yaw-rate [rad/s] from the id-30 quaternion stream."""
    s = streams[ID_ATT]
    t = np.array([x[0] for x in s])
    q = np.array([struct.unpack_from('<4f', x[1], 8) for x in s])
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                     1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    yaw = np.unwrap(yaw)
    return t, yaw, np.gradient(yaw, t)


def align_clocks(streams, bt, bgz):
    """Offset (flight_t = bag_utc + off) via yaw-rate cross-correlation."""
    ft, _, fr = attitude_yaw_rate(streams)
    step = 0.1
    bg = np.arange(bt[0], bt[-1], step)
    bz = np.interp(bg, bt, bgz)
    bz = (bz - bz.mean()) / bz.std()
    fg = np.arange(ft[0], ft[-1], step)
    fz = np.interp(fg, ft, fr)
    fz = (fz - fz.mean()) / fz.std()
    corr = np.correlate(fz, bz, mode='valid') / len(bz)
    best = int(np.argmax(corr))
    # parabolic sub-bin refinement
    if 0 < best < len(corr) - 1:
        y0, y1, y2 = corr[best - 1], corr[best], corr[best + 1]
        den = y0 - 2 * y1 + y2
        shift = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-12 else 0.0
    else:
        shift = 0.0
    t0 = fg[best] + shift * step
    # refine on the highest-energy (spin) 60-s window of the bag gyro —
    # hover segments are near-zero noise and blur the global peak
    energy = np.convolve(bz ** 2, np.ones(600), mode='valid')  # 60 s @0.1 Hz grid
    i0 = int(np.argmax(energy))
    win = bg[i0:i0 + 600]
    bwin = np.interp(win, bt, bgz)
    best_c, best_lag = -2, 0.0
    for lag in np.arange(-3, 3.001, 0.05):
        fwin = np.interp(win - bt[0] + t0 + lag, ft, fr)
        c = np.corrcoef(bwin, fwin)[0, 1]
        if c > best_c:
            best_c, best_lag = c, lag
    t0 += best_lag
    return t0 - bt[0], best_c


def truth_enu(streams, frame_utc, off):
    """ENU displacement [m] rel. bag start + altitude at the frame times.

    vehicle_global_position (id 36, matches the 03_10_02.kml trace): lat/lon
    doubles → ENU meters via local equirectangular approximation.
    """
    s = streams[ID_GPOS]
    t = np.array([x[0] for x in s]) - off
    lat = np.radians(np.array([struct.unpack_from('<d', b, 8)[0] for _, b in s]))
    lon = np.radians(np.array([struct.unpack_from('<d', b, 16)[0] for _, b in s]))
    alt = np.array([struct.unpack_from('<f', b, 24)[0] for _, b in s])
    R = 6378137.0
    clat = np.cos(lat[0])
    e = (lon - lon[0]) * R * clat
    n = (lat - lat[0]) * R
    return (np.interp(frame_utc, t, e),
            np.interp(frame_utc, t, n),
            np.interp(frame_utc, t, alt))


def bag_camera_times(db3, stride):
    """UTC seconds of every stride-th camera frame (matches BagReader.frames)."""
    db = sqlite3.connect(f'file:{db3}?mode=ro', uri=True)
    tid = dict(db.execute('SELECT name,id FROM topics'))['/cam0/image_raw']
    ts = [ts / 1e9 for (ts,) in db.execute(
        'SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp',
        (tid,))]
    db.close()
    return np.array(ts[::stride])


def main():
    streams = ulog_streams(ULOG)
    bt, bgz = bag_gyro_z(DB3)
    off, peak = align_clocks(streams, bt, bgz)
    print(f'clock offset: flight_t = bag_utc + {off:+.3f} s '
          f'(yaw-rate corr peak {peak:.3f})')

    frame_utc = bag_camera_times(DB3, CAMERA_STRIDE)
    ge, gn, galt = truth_enu(streams, frame_utc, off)
    span = np.hypot(ge.max() - ge.min(), gn.max() - gn.min())
    print(f'truth: {len(frame_utc)} frames, ENU box '
          f'E[{ge.min():.0f},{ge.max():.0f}] N[{gn.min():.0f},{gn.max():.0f}] m, '
          f'diagonal {span:.0f} m, alt median {np.median(galt):.0f} m')

    runs = sys.argv[1:] or [
        '/home/one/GNSS-denied-Localization/sample_images/mcl_test/bag_0002_yawfix',
        '/home/one/GNSS-denied-Localization/sample_images/mcl_test/bag_bag_0002_20260824_190926',
        '/tmp/var_a', '/tmp/var_b',
    ]
    hdr = (f'{"run":34s} {"n":>5s} {"err med":>8s} {"err mean":>9s} '
           f'{"err p90":>8s} {"err max":>8s} {">30m":>5s}')
    print(hdr)
    for run in runs:
        try:
            d = np.load(run.rstrip('/') + '/replay_log.npz')
        except FileNotFoundError:
            print(f'{run:34s}  (no replay_log.npz, skipped)')
            continue
        utc = frame_utc[d['k']]
        e, n, _ = truth_enu(streams, utc, off)
        dx = d['x'].astype(float) - float(d['x'][0])
        dy = d['y'].astype(float) - float(d['y'][0])
        err = np.hypot(dx - e, dy - n)
        print(f'{run.split("/")[-1]:34s} {len(d["k"]):5d} {np.median(err):8.1f} '
              f'{err.mean():9.1f} {np.percentile(err, 90):8.1f} {err.max():8.1f} '
              f'{100 * (err > 30).mean():4.0f}%')


if __name__ == '__main__':
    main()
