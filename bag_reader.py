"""ROS2 bag reader for the GNSS-denied localization replay pipeline.

Reads rosbag2 SQLite (.db3) files directly (no ROS2 install needed) and
decodes the CDR-serialized messages we care about:
  - sensor_msgs/msg/Image  (/cam0/image_raw, mono8 752x480)
  - sensor_msgs/msg/Imu   (/imu0, 200 Hz — gyro at offset 128, accel at 224)

CDR notes (little-endian, encapsulation header 0x00 0x01 0x00 0x00):
  - doubles are aligned to 8 bytes relative to the CDR body start
  - strings are uint32 length + bytes (the bridge appends a trailing '\0')
"""

import sqlite3
import struct
import numpy as np


def _parse_header(d):
    """CDR std_msgs/Header: sec, nsec, frame_id. Returns (t_sec, offset_after)."""
    sec, nsec = struct.unpack_from('<iI', d, 0)
    off = 8
    (flen,) = struct.unpack_from('<I', d, off)
    off += 4 + flen
    return sec + nsec * 1e-9, off


def parse_image(blob):
    """Decode a sensor_msgs/msg/Image CDR blob. Returns (t, gray HxW uint8,
    encoding). Assumes mono8 (the bridge's format)."""
    d = blob[4:]
    t, off = _parse_header(d)
    # uint32 height, uint32 width (4-byte aligned after the header)
    off = (off + 3) & ~3
    height, width = struct.unpack_from('<II', d, off)
    off += 8
    (elen,) = struct.unpack_from('<I', d, off)
    off += 4
    encoding = d[off:off + elen].rstrip(b'\0').decode()
    off += elen
    # uint8 is_bigendian (no alignment), pad to 4, uint32 step, uint32 data_len
    off += 1
    off = (off + 3) & ~3
    (step,) = struct.unpack_from('<I', d, off)
    off += 4
    (dlen,) = struct.unpack_from('<I', d, off)
    off += 4
    data = np.frombuffer(d, dtype=np.uint8, count=dlen, offset=off)
    if encoding != 'mono8':
        raise NotImplementedError(f'encoding {encoding!r} not handled (mono8 only)')
    img = data.reshape(height, step)[:, :width].copy()
    return t, img, encoding


def parse_imu(blob):
    """Decode a sensor_msgs/msg/Imu CDR blob.

    Returns (t, gyro xyz rad/s, accel xyz m/s^2). Orientation is left out —
    the bridge never fills it (verified: quaternion is all zeros).

    Layout after the header (8-byte double alignment):
      quaternion (32) + orientation_cov (72) + gyro (24) +
      gyro_cov (72) + accel (24) + accel_cov (72)
    """
    d = blob[4:]
    t, off = _parse_header(d)
    off = (off + 7) & ~7            # align to 8 for the quaternion doubles
    off += 32                        # quaternion (all zeros from this bridge)
    off += 72                        # orientation_covariance
    w = struct.unpack_from('<ddd', d, off)
    off += 24                        # angular_velocity
    off += 72                        # angular_velocity_covariance
    a = struct.unpack_from('<ddd', d, off)
    return t, np.array(w), np.array(a)


def gravity_tilt(accel, alpha=0.98, state=None):
    """Roll/pitch from the accelerometer gravity direction, low-pass filtered.

    The camera/IMU frame is the optical convention (x right, y down in
    image, z toward the ground), so a level hover has specific force
    (0, 0, -g). Gravity direction in camera coords = -accel/|accel|.

    Returns (roll, pitch) in radians, matching orthoprojection.ground_frame
    conventions: gravity_cam = (-sin(pitch), sin(roll)cos(pitch), cos(roll)cos(pitch)).
    """
    a = np.asarray(accel, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-6:
        return (0.0, 0.0) if state is None else tuple(state[:2])
    g = -a / n                       # gravity direction in camera frame
    roll = np.arctan2(g[1], g[2])
    pitch = -np.arcsin(np.clip(g[0], -1.0, 1.0))
    if state is not None and len(state) >= 2:
        # exponential low-pass on the angle (careful with wraparound)
        for i, ang in enumerate((roll, pitch)):
            prev = state[i]
            d = (ang - prev + np.pi) % (2 * np.pi) - np.pi
            state[i] = prev + (1 - alpha) * d
        return state[0], state[1]
    return roll, pitch


class BagReader:
    """Read camera frames and IMU samples from a rosbag2 .db3 file."""

    def __init__(self, db3_path, camera_topic='/cam0/image_raw',
                 imu_topic='/imu0'):
        self.db = sqlite3.connect(f'file:{db3_path}?mode=ro', uri=True)
        self.cur = self.db.cursor()
        self.camera_topic = camera_topic
        self.imu_topic = imu_topic

        self.cur.execute('SELECT name, id FROM topics')
        self.topics = dict(self.cur.fetchall())

    def _topic_id(self, name):
        if name not in self.topics:
            raise KeyError(f'topic {name!r} not in bag (have {list(self.topics)})')
        return self.topics[name]

    def frames(self, stride=1):
        """Yield (t, gray image). `stride` subsamples the camera stream."""
        tid = self._topic_id(self.camera_topic)
        self.cur.execute(
            'SELECT timestamp, data FROM messages WHERE topic_id=? '
            'ORDER BY timestamp', (tid,))
        i = 0
        for ts, blob in self.cur:          # stream rows (no fetchall: 2.7 GB otherwise)
            if i % stride == 0:
                t, img, _ = parse_image(blob)
                yield t, img
            i += 1

    def imu_samples(self):
        """Yield (t, gyro, accel)."""
        tid = self._topic_id(self.imu_topic)
        self.cur.execute(
            'SELECT timestamp, data FROM messages WHERE topic_id=? '
            'ORDER BY timestamp', (tid,))
        for ts, blob in self.cur.fetchall():
            yield parse_imu(blob)

    def frame_times(self, stride=1):
        return [t for t, _ in self.frames(stride)]

    def close(self):
        self.db.close()


def imu_at(imu_list, times, t_query):
    """Nearest IMU sample index to t_query (imu_list from imu_samples())."""
    import bisect
    i = bisect.bisect_left(times, t_query)
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if (times[i] - t_query) < (t_query - times[i - 1]) else i - 1


if __name__ == '__main__':
    import sys
    import cv2
    bag = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/one/GNSS-denied-Localization/datasets/bag_0002_20260824_190926/bag_0002_20260824_190926_0.db3'
    r = BagReader(bag)
    print(f'topics: {r.topics}')

    n = 0
    for t, img in r.frames():
        if n < 3:
            print(f'frame {n}: t={t:.3f} {img.shape} mean={img.mean():.1f}')
            cv2.imwrite(f'/tmp/bag_frame_{n}.png', img)
        n += 1
    print(f'total frames: {n}')

    m = 0
    T, W, A = [], [], []
    for t, w, a in r.imu_samples():
        T.append(t); W.append(w); A.append(a)
        m += 1
    print(f'total imu: {m}, dt={np.mean(np.diff(T))*1000:.2f} ms')

    # tilt over the bag (low-pass filtered)
    state = [0.0, 0.0]
    rolls, pitches = [], []
    for a in A:
        roll, pitch = gravity_tilt(a, alpha=0.98, state=state)
        rolls.append(np.degrees(roll))
        pitches.append(np.degrees(pitch))
    rolls, pitches = np.array(rolls), np.array(pitches)
    print(f'roll  over bag: min={rolls.min():.1f}° max={rolls.max():.1f}° mean={rolls.mean():.1f}°')
    print(f'pitch over bag: min={pitches.min():.1f}° max={pitches.max():.1f}° mean={pitches.mean():.1f}°')
    r.close()
