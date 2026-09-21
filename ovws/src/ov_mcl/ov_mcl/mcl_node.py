#!/usr/bin/env python3
"""Online MCL localization node (GNSS-denied UAV).

Subscriptions:
  /cam0/image_raw   (sensor_msgs/Image, mono8)  — downward camera
  /ov_msckf/odomimu (nav_msgs/Odometry)         — OpenVINS pose (IMU in the
                                                   VIO global frame)
  /mavros/local_position/pose (geometry_msgs/PoseStamped, live only) —
                                                   FCU baro height (ENU z)

Publications:
  mcl/odom  (nav_msgs/Odometry) — GPS-substitute External-Vision odometry for
                                  the PX4 EKF2 (step 4): twist = VIO velocity
                                  (base_link FLU), pose = raw MCL estimate in
                                  the ENU map frame — honesty rules in
                                  publish_ev. Remap to /mavros/odometry/out
                                  when MAVROS runs (launch arg ev_topic).
  mcl/health (std_msgs/String)  — 1 Hz JSON health snapshot (step 3)

  - tilt (roll/pitch)      <- FCU EKF attitude: LIVE = orientation of the
                              MAVROS local_position/pose topic, REPLAY = ULog
                              vehicle_attitude beside the bag (clock-aligned);
                              VIO attitude, then a raw-IMU complementary
                              filter, as fallbacks
  - altitude               <- FCU baro height: LIVE = MAVROS local_position
                              topic, REPLAY = ULog beside the bag (clock-
                              aligned); VIO z + alt_anchor only as fallback
  - inter-frame motion     <- VIO position delta + yaw delta (was: phase
                              correlation + gyro-z integration + consensus)

The particle filter itself (mcl.py in the GNSS-denied-Localization
workspace) is unchanged.

Coordinate conventions (verified against OpenVINS source):
  - odomimu quaternion is Hamilton (x,y,z,w) = R_ItoG (IMU -> VIO global)
  - VIO global frame has z UP (propagator: v_dot = R_GtoI^T a - g, g=(0,0,9.81))
  - camera rotation: R_CtoG = R_ItoG @ R_CI^T, R_CI from Kalibr T_cam_imu
  - VIO global x/y is arbitrary (init heading): only RELATIVE quantities are
    used; the constant rotation to the map frame is absorbed by the MCL yaw.

Environment note: torch/cv2 live in the workspace venv (python3.12);
rclpy comes from /opt/ros/jazzy via PYTHONPATH. We append the venv
site-packages so a plain `ros2 run/launch` works after sourcing ROS2
and the workspace overlay.

Run:
  ros2 launch ov_mcl mcl_localization.launch.py [bag:=<bag_dir>]
"""

import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

# --- portable roots: no absolute dev paths ----------------------------------
# GNSS_DIR holds mcl.py/bag_mcl.py. Candidates in order: $MCL_ROOT, walking up
# from this file (the MCLDrone bundle), the dev checkout, then ~/MCLDrone
# (deployed layout where this package is colcon-installed from ~/ovws).
def _find_root():
    here = os.path.dirname(os.path.abspath(__file__))
    up = here
    while up != '/':
        if os.path.isfile(os.path.join(up, 'mcl.py')):
            break
        up = os.path.dirname(up)
    for cand in (os.environ.get('MCL_ROOT'), up,
                 '/home/one/GNSS-denied-Localization',
                 os.path.expanduser('~/MCLDrone')):
        if cand and os.path.isfile(os.path.join(cand, 'mcl.py')):
            return cand
    return here


def _default_calib():
    """Rig calibration shipped in this folder, else the dev copy."""
    rel = os.path.join('ovws', 'src', 'ov_SchurVINS', 'config',
                       'cyperstereo_012_752x480_equi',
                       'kalibr_imucam_chain.yaml')
    cand = os.path.join(GNSS_DIR, rel)
    if os.path.isfile(cand):
        return cand
    return ('/home/one/workspace/ovws/src/ov_SchurVINS/config/'
            'cyperstereo_012_752x480_equi/kalibr_imucam_chain.yaml')


GNSS_DIR = _find_root()
# optional venv holding torch/cv2: beside the folder (deployed) or the dev
# venv (local dev machine). Whichever exists is used; absent = system python.
_VENV_CANDIDATES = (
    os.path.join(GNSS_DIR, 'venv_mcl', 'lib',
                 'python3.%d' % sys.version_info.minor, 'site-packages'),
    '/home/one/workspace/ovws/venv_mcl/lib/python3.12/site-packages',
)
VENV_SITE = next((p for p in _VENV_CANDIDATES if os.path.isdir(p)),
                 _VENV_CANDIDATES[0])
if os.path.isdir(VENV_SITE) and VENV_SITE not in sys.path:
    sys.path.append(VENV_SITE)

import cv2  # noqa: E402
import torch  # noqa: E402

if GNSS_DIR not in sys.path:
    sys.path.insert(0, GNSS_DIR)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import Image, Imu  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from mcl import (MCL, load_rgb, MAP_PATH, PATCH_PX, COVERAGE_M,  # noqa: E402
                 CHECKPOINT, OUTPUT_DIR, set_seed)
from utils.utils import cropRectangularSectionFromImage  # noqa: E402
from bag_mcl import (load_kalibr, valid_coverage_m, coarse_scan,  # noqa: E402
                     local_yaw_scan, particles_from_peaks)
from orthoprojection import orthoproject, extract_patch  # noqa: E402

# --- sign conventions of the VIO-derived motion (MCL body frame) -------------
# The MCL body frame is defined by the observation model (see mcl.py
# score_particles / cropRectangularSectionFromImage): x = camera optical x,
# y = -camera optical y (image up), yaw = azimuth of the optical x-axis.
# The state yaw must follow the azimuth DELTA of the optical x-axis measured
# in a z-up world frame — the VIO azimuth (atan2 of R_CtoG's first column)
# IS that quantity directly, regardless of rig mounting:
#     dyaw = +1 * (VIO azimuth delta of optical x)          (YAW_SIGN = +1)
# The old YAW_SIGN = -1 was inherited from bag_mcl.py, whose yaw state was
# the camera-frame gyro-z integral (z-down rigs: gyro-z = -azimuth rate) —
# a DIFFERENT yaw convention. Feeding it here MIRRORED every turn: on
# bag_0001_184004 frames 2000-8200 the drone turned +364 deg CCW (ulog)
# while the filter was fed -352.7 deg, so the map patch turned right when
# the drone turned left and the estimated trajectory mirrored the true one.
YAW_SIGN = +1.0        # dyaw = YAW_SIGN * (VIO azimuth delta of optical x)
BODY_Y_FLIP = True     # dy = -(displacement along optical y)

# --- filter constants (same as bag_mcl.py offline replay) --------------------
T_LIKELIHOOD = 0.02
ROUGHEN_STD = (5.0, 5.0, 0.03, 0.01)
NOISE_STD = np.array([1.5, 1.5, 0.02, 0.005])   # VIO motion is accurate
HOVER_NOISE = 0.6
LOST_SCORE = 0.6
LOST_FRAMES = 5
V_MAX = 15.0
RESAMPLE_EFF = 0.3
# --- recovery / kidnapped-scan constants (2026-09-16 step 2) -------------------
RECOVER_FRAMES = 10    # consecutive good frames before a rescan is CONFIRMED
MAX_LOCAL_RESCANS = 3  # failed local rescans before escalating to a global scan
RESCAN_WINDOW_S = 90.0   # rescan-frequency evidence window (see below)
MAX_WINDOW_RESCANS = 4   # locals within RESCAN_WINDOW_S -> global scan
# False sites score 0.98-1.0 nearly every frame, so a 10-frame streak CONFIRMS
# a false lock and resets the fails ladder before it can climb (kidnapped
# replay v2, 2026-09-16: 13 locals, 0 globals, never recovered). The escape is
# FREQUENCY evidence: a genuine reseed holds for minutes, a false lock needs a
# new rescan every few seconds — 4 rescans within 90 s survive the confirms.
RELOC_GROW = 1.5       # local rescan radius growth per consecutive fail
RELOC_RADIUS_MAX = 2000.0   # m, cap of the escalating local rescan radius
SCAN_STEP_M = 96.0     # global nomination grid pitch (m)
SCAN_TOPK = 10         # candidate regions seeded by the global scan
SCAN_WINDOW_S = 25.0   # patch-buffer window used by the global scan
SCAN_FRAMES = 9        # max buffered frames scored per global scan
SCAN_MIN_COV = 88.0    # coverage gate: only patches inside the model's
                       # RELIABLE envelope are measured/scanned. The nominal
                       # model tolerance is 32-96 m (synthetic
                       # coverage_sweep_test), but REAL patches below ~88 m
                       # coverage (below ~44 m altitude — coverage ~= 2*alt
                       # for this camera) are mushy: the realistic-init
                       # replays v6/v7 (2026-09-16) init-scanned at 15-20 m
                       # altitude (cov 32-40) and false-locked ~300-590 m off
                       # within 45 frames; every VALIDATED scan success
                       # (kidnap v4, climb-run init, nomination tests) used
                       # cov >= 88 frames. Gates the scan buffers, the
                       # per-frame measurement update (motion-only predict
                       # below it) and the point-init scan timing.
SCAN_WAIT_CAP_S = 240.0  # absolute scan-init fallback: if usable frames never
                         # appear (no VIO / coverage never in range) scan anyway
                         # after this long from the first processed frame
STEER_SPAN = np.radians(24.0)  # yaw window half-width around the compass
STEER_YAWS = 7         # yaws inside the steered window (~8 deg steps)
# --- health monitor / plausibility gates (2026-09-16 step 3) -------------------
ALT_MIN = 3.0          # m: HONEST altitude floor for frame processing. The old
                       # max(20, alt) clamp FAKED a 40 m coverage on the ground
                       # (true alt ~0), letting below-envelope mushy patches
                       # into the init scan and measurement updates — the
                       # realistic-init replays false-tracked 60-230 s on the
                       # ground before climbing into the envelope.
INNOV_ALPHA = 0.1      # EMA factor of the position-innovation health metric
                       # (|estimate delta − VIO-predicted delta|): ~0 while
                       # tracking, ≈|VIO motion| when a false lookalike keeps
                       # re-capturing the estimate as the cloud moves away
# compass -> MCL-map yaw calibration 'a,b' (b in deg):
#     yaw_mcl = a * heading_ned + radians(b)
# Fitted on the verified s2 replay of bag 190020 (residual std 4.3 deg,
# p95 6.5 deg); a = -1 (NED heading is CW-positive vs CCW math azimuth),
# b = magnetic declination + camera-mounting offset — a rig+map constant.
YAW_CAL_DEFAULT = '-1,1.3'
# --- GPS-substitute EV odometry constants (2026-09-17 step 4) -----------------
EV_HUGE = 9999.0        # covariance of a suppressed axis: EKF2 gain -> ~0
                        # (gated/lost frames publish VELOCITY-ONLY messages)
EV_VEL_VAR = 0.04       # m^2 twist variance (VIO finite-difference ~0.2 m/s)
EV_POS_VAR_MIN = 4.0    # m^2 pose variance floor (~2 m/axis std — the model's
                        # best-case discriminative power)
EV_POSE_STALE_S = 1.0   # s without an mcl.step -> suppress the pose (the
                        # estimate is dead-reckoned further than the spread)
EV_VEL_MAX = 20.0       # m/s horizontal plausibility gate on the VIO velocity
EV_VZ_MAX = 10.0        # m/s vertical plausibility gate
EV_VEL_STALE_S = 0.5    # s after which a frozen velocity is marked stale

POSE_MAX_AGE = 0.15   # s: max |odomimu stamp - image stamp| for a match
FCU_ALT_MAX_AGE = 1.0  # s: max |FCU altitude stamp - image stamp| (baro is
                       # slow-moving; interpolation bridges the gap)
FCU_ATT_MAX_AGE = 0.5  # s: max |FCU attitude stamp - image stamp| (tilt moves
                       # faster than baro — tighter gate, no fresh-fallback:
                       # on stamp mismatch the VIO attitude takes over)
# Constant mounting rotation FCU-body (FRD) -> /imu0 VIO-IMU frame, row-major.
# Wahba-fitted from bag 190020 (2026-09-16, 412 clean 1-s acc-mean windows,
# residual 1.45 deg): the VIO IMU is the FCU body yawed ~89 deg about z, with
# ~3 deg cross-axis mounting offsets. Recalibrate if the rig is re-mounted
# (fit windowed /imu0 acc means against ULog-attitude gravity).
FCU_IMU_ROT_ROWMAJOR = [0.02305096, -0.99967990, -0.01042873,
                        0.99823785,  0.02358575, -0.05445105,
                        0.05467959, -0.00915521,  0.99846198]


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def latlon_to_map_px(lat, lon, off_x, off_y, z=17):
    """WGS-84 lat/lon -> deployment-map pixel.

    Web-mercator projection at zoom z, then shifted by the map's geo offset:
    local map px = global z17 mercator px + (off_x, off_y) — the same
    convention as geo_offset in train_config.yml (for a map stitched from
    z17 tiles with top-left tile origin (tx,ty): offset = (-tx*256, -ty*256)).
    """
    n = 2 ** z * 256.0
    gx = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, 85.05112878), -85.05112878)   # mercator domain
    gy = ((1.0 - math.log(math.tan(math.radians(lat)) +
                          1.0 / math.cos(math.radians(lat))) / math.pi)
          / 2.0 * n)
    return gx + off_x, gy + off_y


def bag_gps_truth(bag_dir, frame_t, off_x, off_y):
    """GPS truth trace in map pixels for a replayed bag (PLOTTING ONLY —
    never fed to the filter).

    Reads the ULog recorded next to the bag (<bag_dir>/*.ulg) and aligns its
    boot clock to the bag/bridge clock by correlating the bag's /imu0 gyro-z
    with the ulog attitude yaw-rate (same method as train_similarity.py
    align_clocks; the PX4 time_utc_usec field is wrong on this rig — off by
    ~21 h). Returns (px, py, off, corr): map-pixel coordinates of the GPS
    track at the frame times in frame_t, the clock offset, and the peak
    correlation. Returns None when the bag dir has no .ulg/.db3, no
    /imu0, or the correlation is too weak (< 0.3) to trust the alignment.
    """
    import glob
    import sqlite3
    from pyulog import ULog
    from bag_reader import parse_imu

    ulgs = sorted(glob.glob(os.path.join(bag_dir, '*.ulg')))
    db3s = sorted(glob.glob(os.path.join(bag_dir, '*.db3')))
    if not ulgs or not db3s or len(frame_t) == 0:
        return None
    u = ULog(ulgs[0])
    att = gps = None
    for d in u.data_list:
        if d.name == 'vehicle_attitude' and att is None:
            t = d.data['timestamp'] / 1e6
            q = np.stack([d.data[f'q[{i}]'] for i in range(4)], 1)
            # PX4 q = [w, x, y, z] (w-first!) — yaw is the NED heading
            yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                             1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            att = (t, np.unwrap(yaw))
        elif d.name == 'vehicle_gps_position' and gps is None:
            ok = d.data['fix_type'] >= 3
            gps = (d.data['timestamp'][ok] / 1e6,
                   d.data['latitude_deg'][ok], d.data['longitude_deg'][ok])
    if att is None or gps is None:
        return None

    # bag /imu0 gyro-z on HEADER stamps — same bridge clock as the images
    db = sqlite3.connect(f'file:{db3s[0]}?mode=ro', uri=True)
    try:
        tid = dict(db.execute('SELECT name,id FROM topics')).get('/imu0')
        if tid is None:
            return None
        rows = db.execute(
            'SELECT timestamp,data FROM messages WHERE topic_id=? '
            'ORDER BY timestamp', (tid,)).fetchall()
    finally:
        db.close()
    t_imu = np.array([parse_imu(b)[0] for _, b in rows])
    gz = np.array([parse_imu(b)[1][2] for _, b in rows])

    # clock offset (ulog_t = bag_t + off) via gyro-z / yaw-rate correlation
    fr = np.gradient(att[1], att[0])

    def corr_at(off):
        x = t_imu + off
        m = (x >= att[0][0]) & (x <= att[0][-1])
        if m.sum() < 1000:
            return 0.0
        return abs(np.corrcoef(gz[m], np.interp(x[m], att[0], fr))[0, 1])

    best_c, best_off = 0.0, 0.0
    for off in np.arange(att[0][0] - t_imu[-1], att[0][-1] - t_imu[0], 1.0):
        c = corr_at(off)
        if c > best_c:
            best_c, best_off = c, off
    for off in best_off + np.arange(-1.5, 1.51, 0.05):
        c = corr_at(off)
        if c > best_c:
            best_c, best_off = c, off
    if best_c < 0.3:
        return None

    # project GPS samples to map px, then interpolate at the frame times
    # (project-then-interpolate — same as train_similarity.flight_truth, so
    # the overlay matches the training/eval truth bit-for-bit)
    t_g, lat, lon = gps
    gpx = np.empty(len(t_g))
    gpy = np.empty(len(t_g))
    for i in range(len(t_g)):
        gpx[i], gpy[i] = latlon_to_map_px(lat[i], lon[i], off_x, off_y)
    px = np.interp(frame_t, t_g - best_off, gpx)
    py = np.interp(frame_t, t_g - best_off, gpy)
    return px, py, best_off, best_c


def quat_xyzw_to_R(qx, qy, qz, qw):
    """Hamilton quaternion (x,y,z,w) -> rotation matrix (body -> world)."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
         2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
         1 - 2 * (qx * qx + qy * qy)],
    ])


def camera_tilt(R_CtoG):
    """Roll/pitch of the down-facing camera from its world rotation.

    Gravity-down direction in camera coords: g = R_CtoG.T @ [0,0,-1]
    (level nadir camera -> g = (0,0,1), matching
    bag_reader.gravity_tilt / orthoprojection.ground_frame)."""
    g = R_CtoG.T @ np.array([0.0, 0.0, -1.0])
    roll = math.atan2(g[1], g[2])
    pitch = -math.asin(float(np.clip(g[0], -1.0, 1.0)))
    return roll, pitch


def gravity_to_tilt(g_cam):
    """Gravity unit vector in the CAMERA optical frame -> (roll, pitch),
    the camera_tilt convention (level nadir -> g = (0,0,1))."""
    g = np.asarray(g_cam, dtype=np.float64)
    g = g / np.linalg.norm(g)
    roll = math.atan2(g[1], g[2])
    pitch = -math.asin(float(np.clip(g[0], -1.0, 1.0)))
    return roll, pitch


def fcu_frd_tilt(g_frd, R_fcu2cam):
    """Gravity in the FCU BODY frame (FRD, z down: level -> (0,0,1)) ->
    camera (roll, pitch) via the calibrated mounting rotation."""
    g = np.asarray(g_frd, dtype=np.float64)
    return gravity_to_tilt(R_fcu2cam @ g)


def mavros_enu_tilt(qx, qy, qz, qw, R_fcu2cam):
    """Camera (roll, pitch) from a MAVROS ENU orientation quaternion.

    MAVROS local_position/imu orientations rotate base_link (FLU) -> the
    ENU world. Chain: gravity ENU (0,0,-1) -> base_link FLU (R^T) -> FCU
    body FRD (y,z flip) -> /imu0 (calibrated mounting rotation) -> camera.
    Level hover yields ~0/0 — a convention error anywhere in this chain
    shows up as a large phantom tilt and is caught by the VIO cross-check
    in process_one."""
    R = quat_xyzw_to_R(qx, qy, qz, qw)      # base_link FLU -> world ENU
    g = R.T @ np.array([0.0, 0.0, -1.0])    # gravity in FLU base_link
    g = g * np.array([1.0, -1.0, -1.0])     # FLU -> FRD (y left->right, z up->down)
    return fcu_frd_tilt(g, R_fcu2cam)


def enu_quat_to_ned_heading(qx, qy, qz, qw):
    """MAVROS ENU orientation quaternion (base_link FLU -> ENU world) ->
    FCU NED heading (CW-positive from north).

    The FLU x-axis (body front) in ENU is R[:, 0] = (E, N, U); the NED
    heading is atan2(E, N). Unit-tested against the replay path's w-first
    vehicle_attitude quaternion yaw (test_step2.py)."""
    R = quat_xyzw_to_R(qx, qy, qz, qw)
    return math.atan2(R[0, 0], R[1, 0])


def vio_to_map_delta(d_vio, theta):
    """2D VIO-global displacement -> map-frame displacement.

    theta = azimuth_map - azimuth_vio of the same world-fixed axis (the
    compass-observed VIO->map rotation): v_map = R(theta) v_vio."""
    c, s = math.cos(theta), math.sin(theta)
    return (c * d_vio[0] - s * d_vio[1], s * d_vio[0] + c * d_vio[1])


def bilinear_grid_sample(G, cols, rows):
    """Bilinear sample of grid G (rows-major array) at fractional (col, row)
    coordinates (used for the motion-compensated grid accumulation)."""
    c0 = np.floor(cols).astype(int)
    r0 = np.floor(rows).astype(int)
    fc, fr = cols - c0, rows - r0
    c1 = np.clip(c0 + 1, 0, G.shape[1] - 1)
    r1 = np.clip(r0 + 1, 0, G.shape[0] - 1)
    c0 = np.clip(c0, 0, G.shape[1] - 1)
    r0 = np.clip(r0, 0, G.shape[0] - 1)
    return (G[r0, c0] * (1 - fc) * (1 - fr) + G[r0, c1] * fc * (1 - fr)
            + G[r1, c0] * (1 - fc) * fr + G[r1, c1] * fc * fr)


def parse_yaw_cal(s):
    """'a,b' (b in deg) -> (a, b) floats; falls back to the default on any
    parse error (the STRING param is passed through the launch file)."""
    try:
        vals = [float(v) for v in str(s).replace(',', ' ').split()]
        if len(vals) != 2:
            raise ValueError
        return vals[0], vals[1]
    except (ValueError, TypeError):
        d = YAW_CAL_DEFAULT.replace(',', ' ').split()
        return float(d[0]), float(d[1])


def leveled_axes(R_CtoG):
    """Camera optical x/y axes with roll/pitch removed (world coords).

    x_h: optical x projected to horizontal; y_h = z_down x x_h keeps the
    optical frame's handedness (x right, y down-in-image, z toward ground).
    """
    z_up = np.array([0.0, 0.0, 1.0])
    z_down = np.array([0.0, 0.0, -1.0])
    x = R_CtoG[:, 0].astype(np.float64)
    x_h = x - (x @ z_up) * z_up
    n = np.linalg.norm(x_h)
    if n < 1e-6:                      # optical x near vertical (extreme tilt)
        y = R_CtoG[:, 1].astype(np.float64)
        y_h = y - (y @ z_up) * z_up
        x_h = np.cross(y_h, z_down)
        x_h /= max(1e-9, np.linalg.norm(x_h))
    else:
        x_h = x_h / n
    y_h = np.cross(z_down, x_h)
    return x_h, y_h


def vio_motion(p0, R0, p1, R1):
    """Inter-frame motion (dx, dy, dyaw) in the MCL body frame from VIO.

    dx: displacement along the (leveled) optical x-axis;
    dy: along -optical y (BODY_Y_FLIP);
    dyaw: azimuth change of the optical x-axis (YAW_SIGN).
    The VIO-global -> map-frame rotation cancels in these relative quantities.
    """
    dpos = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    x_h, y_h = leveled_axes(R1)
    dx = float(dpos @ x_h)
    dy = float(dpos @ y_h)
    if BODY_Y_FLIP:
        dy = -dy
    psi0 = math.atan2(R0[1, 0], R0[0, 0])   # azimuth of optical x in VIO frame
    psi1 = math.atan2(R1[1, 0], R1[0, 0])
    dyaw = YAW_SIGN * wrap_angle(psi1 - psi0)
    return dx, dy, dyaw


def vio_yaw_proxy(R_CtoG):
    """VIO-frame yaw proxy (azimuth of optical x). Only deltas are used."""
    return math.atan2(R_CtoG[1, 0], R_CtoG[0, 0])


class ImuTilt:
    """Roll/pitch (tilt relative to gravity) from acc + gyro ONLY — no VIO.

    A vector complementary filter (Mahony-lite): tracks the gravity unit
    vector in the IMU body frame. The gyro PROPAGATES it between samples
    (smooth/short-term; correct during the aggressive maneuvers that corrupt
    a single accelerometer reading) and the accelerometer CORRECTS it
    (drift-free long-term, via the gravity direction -a/|a|). This stays
    valid when a VIO estimator diverges, because it uses the raw IMU.

    Yaw is intentionally NOT estimated (unobservable from acc+gyro without a
    magnetometer) — the orthoprojection does not need yaw (ground_frame uses
    roll/pitch only), and MCL recovers heading via the rotation sweep/rescans.

    Camera-frame roll/pitch use the same optical-frame formulas as
    bag_reader.gravity_tilt: g = gravity direction, roll=atan2(g1,g2),
    pitch=-asin(g0).
    """

    def __init__(self, k_acc=0.02):
        self.g = None              # gravity unit vector in IMU frame
        self.t_last = None
        self.k_acc = k_acc         # accelerometer correction gain (per sample)

    def ready(self):
        return self.g is not None

    def update(self, t, gyr, acc):
        gyr = np.asarray(gyr, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)
        if self.g is None:
            n = np.linalg.norm(acc)
            if n < 1e-6:
                return
            self.g = -acc / n      # gravity direction (opposes specific force)
            self.t_last = t
            return
        dt = float(np.clip(t - self.t_last, 0.0, 0.05))
        self.t_last = t
        # --- gyro propagation: a world-fixed vector expressed in the body
        # frame evolves as g_body' = exp(-[w dt]x) g_body (strapdown) -----
        w = -gyr * dt
        ang = np.linalg.norm(w)
        if ang > 1e-12:
            u = w / ang
            ca, sa = math.cos(ang), math.sin(ang)
            R = (np.eye(3) * ca
                 + (1 - ca) * np.outer(u, u)
                 + sa * np.array([[0, -u[2], u[1]],
                                  [u[2], 0, -u[0]],
                                  [-u[1], u[0], 0]]))
            self.g = R @ self.g
        # --- accelerometer correction (drift-free gravity direction) ------
        n = np.linalg.norm(acc)
        if n > 1e-6:
            g_meas = -acc / n
            self.g = (1 - self.k_acc) * self.g + self.k_acc * g_meas
            self.g = self.g / np.linalg.norm(self.g)

    def camera_tilt(self, R_CI):
        """(roll, pitch) radians in the camera OPTICAL frame. R_CI rotates
        IMU-frame vectors into the camera frame (Kalibr T_cam_imu)."""
        gc = R_CI @ self.g
        roll = math.atan2(gc[1], gc[2])
        pitch = -math.asin(float(np.clip(gc[0], -1.0, 1.0)))
        return roll, pitch


def bag_fcu_streams(bag_dir, R_fcu2cam):
    """FCU baro height, EKF attitude tilt AND NED heading vs the BAG clock,
    from the ULog beside the bag, clock-aligned by the gyro-z / attitude
    yaw-rate correlation (ulog_t = bag_t + off, same method as
    bag_gps_truth). The flight-controller EKF handles this airframe's heavy
    prop vibration (the raw /imu0 accelerometer does NOT: |a-g| p50
    7.4 m/s^2), so its attitude is the tilt source for the orthoprojection.
    Returns (alt_at, tilt_at, hdg_at): callables bag_t -> height m /
    (roll, pitch) camera-frame / NED heading rad (unwrapped); all None
    when the bag has no ULog or the correlation is too weak."""
    import glob
    import sqlite3
    from pyulog import ULog
    from bag_reader import parse_imu

    ulgs = sorted(glob.glob(os.path.join(bag_dir, '*.ulg')))
    db3s = sorted(glob.glob(os.path.join(bag_dir, '*.db3')))
    if not ulgs or not db3s:
        return None, None, None
    u = ULog(ulgs[0])
    att = alt = None
    for d in u.data_list:
        if d.name == 'vehicle_attitude' and att is None:
            t = d.data['timestamp'] / 1e6
            q = np.stack([d.data[f'q[{i}]'] for i in range(4)], 1)
            # PX4 q = [w, x, y, z] (w-first!) — yaw is the NED heading
            yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                             1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            # gravity in the FCU body (FRD): R_wb^T @ (0,0,1) with R from the
            # reordered (x,y,z,w) quaternion
            qq = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], 1)
            qx, qy, qz, qw = qq.T
            R_wb = np.stack([
                np.stack([1 - 2 * (qy * qy + qz * qz),
                          2 * (qx * qy - qz * qw),
                          2 * (qx * qz + qy * qw)], -1),
                np.stack([2 * (qx * qy + qz * qw),
                          1 - 2 * (qx * qx + qz * qz),
                          2 * (qy * qz - qx * qw)], -1),
                np.stack([2 * (qx * qz - qy * qw),
                          2 * (qy * qz + qx * qw),
                          1 - 2 * (qx * qx + qy * qy)], -1)], -2)
            g_frd = np.einsum('nji,j->ni', R_wb, np.array([0., 0., 1.]))
            att = (t, np.unwrap(yaw), g_frd)
        elif d.name == 'vehicle_local_position' and alt is None:
            ok = np.isfinite(d.data['z'])
            alt = (d.data['timestamp'][ok] / 1e6, -d.data['z'][ok])
    if att is None or alt is None:
        return None, None, None

    db = sqlite3.connect(f'file:{db3s[0]}?mode=ro', uri=True)
    try:
        tid = dict(db.execute('SELECT name,id FROM topics')).get('/imu0')
        if tid is None:
            return None, None, None
        rows = db.execute(
            'SELECT timestamp,data FROM messages WHERE topic_id=? '
            'ORDER BY timestamp', (tid,)).fetchall()
    finally:
        db.close()
    t_imu = np.array([parse_imu(b)[0] for _, b in rows])
    gz = np.array([parse_imu(b)[1][2] for _, b in rows])
    fr = np.gradient(att[1], att[0])

    def corr_at(off):
        x = t_imu + off
        m = (x >= att[0][0]) & (x <= att[0][-1])
        if m.sum() < 1000:
            return 0.0
        return abs(np.corrcoef(gz[m], np.interp(x[m], att[0], fr))[0, 1])

    bc, bo = 0.0, 0.0
    for off in np.arange(att[0][0] - t_imu[-1], att[0][-1] - t_imu[0], 1.0):
        c = corr_at(off)
        if c > bc:
            bc, bo = c, off
    for off in bo + np.arange(-1.5, 1.51, 0.05):
        c = corr_at(off)
        if c > bc:
            bc, bo = c, off
    if bc < 0.3:
        return None, None, None
    ta, za = alt
    tg, g_frd = att[0], att[2]

    def alt_at(t_bag):
        return float(np.interp(t_bag, ta - bo, za))

    def tilt_at(t_bag):
        tu = t_bag + bo
        g = np.array([np.interp(tu, tg, g_frd[:, i]) for i in range(3)])
        return fcu_frd_tilt(g, R_fcu2cam)

    def hdg_at(t_bag):
        return float(np.interp(t_bag + bo, tg, att[1]))

    return alt_at, tilt_at, hdg_at


class MclNode(Node):

    def __init__(self):
        super().__init__('mcl_node')

        def param(name, default):
            return self.get_parameter(name).value

        self.declare_parameter('map_path', MAP_PATH)
        self.declare_parameter('map_gsd', 1.10)
        self.declare_parameter('checkpoint', CHECKPOINT)
        self.declare_parameter('stride', 5)
        self.declare_parameter('n_particles', 1000)
        self.declare_parameter('seed', -1)
        self.declare_parameter('init_mode', 'scan')       # 'scan' | 'point'
        self.declare_parameter('init_px_x', 0)
        self.declare_parameter('init_px_y', 0)
        # init_mode=point: the click prior bounds a scan REGION of this
        # radius around the point (a user click is an approximate position),
        # not a precise point seed
        self.declare_parameter('init_radius_m', 300.0)
        # WGS-84 init position (init_mode=point): overrides init_px_x/y
        self.declare_parameter('init_lat', -999.0)
        self.declare_parameter('init_lon', -999.0)
        # map georeference: local map px = global z17 mercator px + offset
        # (defaults match the default map z17_5120.png; for a map stitched
        # from z17 tiles with top-left tile origin (tx,ty): (-tx*256, -ty*256))
        self.declare_parameter('map_geo_offset_x', -27449088.0)
        self.declare_parameter('map_geo_offset_y', -14586624.0)
        self.declare_parameter('alt_anchor', 0.0)
        self.declare_parameter('out_dir',
                               os.path.join(OUTPUT_DIR, 'online'))
        self.declare_parameter('frame_topic', '/cam0/image_raw')
        self.declare_parameter('imu_topic', '/imu0')
        self.declare_parameter('odom_topic', '/ov_msckf/odomimu')
        # live FCU baro height + EKF attitude source (MAVROS ENU local
        # position pose: position.z = height, orientation = EKF attitude;
        # replay reads the ULog beside the bag instead — see fcu_alt_at /
        # fcu_tilt_at)
        self.declare_parameter('fcu_alt_topic', '/mavros/local_position/pose')
        # FCU-body(FRD) -> /imu0 mounting rotation, comma-separated row-major
        # 3x3 ('' = built-in Wahba-fitted default from bag 190020 — see
        # FCU_IMU_ROT_ROWMAJOR). STRING type so the launch file can pass it
        # through like every other launch arg.
        self.declare_parameter('fcu_imu_rot', '')
        # compass -> MCL-map yaw calibration 'a,b' (b in deg):
        # yaw_mcl = a * NED-heading + b — steers every scan to a +-24 deg
        # yaw window instead of the full circle (see YAW_CAL_DEFAULT)
        self.declare_parameter('yaw_cal', YAW_CAL_DEFAULT)
        self.declare_parameter('calib_path', _default_calib())
        self.declare_parameter('save_debug', False)
        self.declare_parameter('debug_stride', 1)
        # camera-frame range [frame_start, frame_end) to process (0 = open):
        # skip takeoff/landing stages — VIO still sees the whole bag (it must,
        # for initialization), only the MCL/processing queue is gated
        self.declare_parameter('frame_start', 0)
        self.declare_parameter('frame_end', 0)
        # replayed bag directory (the launch 'bag' arg): used ONLY to overlay
        # the GPS truth from the ULog beside the bag on replay.png — the
        # filter itself never sees it (GNSS-denied!)
        self.declare_parameter('bag', '')

        self.stride = int(param('stride', 5))
        self.n_particles = int(param('n_particles', 1000))
        self.init_mode = str(param('init_mode', 'scan'))
        self.init_radius_m = float(param('init_radius_m', 300.0))
        cal = parse_yaw_cal(param('yaw_cal', YAW_CAL_DEFAULT))
        if str(param('yaw_cal', YAW_CAL_DEFAULT)).strip() != YAW_CAL_DEFAULT \
                and cal == parse_yaw_cal(YAW_CAL_DEFAULT):
            self.get_logger().warn(
                f"bad yaw_cal {param('yaw_cal', '')!r} -> {YAW_CAL_DEFAULT}")
        self.yaw_a, self.yaw_b = cal
        self.alt_anchor = float(param('alt_anchor', 0.0))
        self.out_dir = str(param('out_dir', ''))
        os.makedirs(self.out_dir, exist_ok=True)
        sd = param('save_debug', False)
        self.save_debug = (str(sd).lower() in ('1', 'true', 'yes')
                           if isinstance(sd, str) else bool(sd))
        self.debug_stride = max(1, int(param('debug_stride', 1)))
        self.frame_start = max(0, int(param('frame_start', 0)))
        self.frame_end = int(param('frame_end', 0))   # 0 = no upper bound
        self.bag_dir = str(param('bag', ''))
        self.geo_off = (float(param('map_geo_offset_x', -27449088.0)),
                        float(param('map_geo_offset_y', -14586624.0)))
        if self.save_debug:
            os.makedirs(os.path.join(self.out_dir, 'debug'), exist_ok=True)
        seed = int(param('seed', -1))
        if seed >= 0:
            set_seed(seed)

        # --- map + similarity model + camera calibration -------------------
        map_path = str(param('map_path', MAP_PATH))
        self.gsd_map = float(param('map_gsd', 1.10))
        map_rgb = load_rgb(map_path)
        h, w = map_rgb.shape[:2]
        self.cx_px, self.cy_px = w // 2, h // 2
        self.get_logger().info(
            f'map {w}x{h}px gsd={self.gsd_map} m/px | out {self.out_dir}')

        self.mcl = MCL(map_rgb, self.gsd_map, (self.cx_px, self.cy_px),
                       str(param('checkpoint', CHECKPOINT)))
        self.map_rgb = map_rgb

        calib_path = str(param('calib_path', ''))
        K, dist, R_CI, dist_model = load_kalibr(calib_path)
        self.R_CI = R_CI
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx_cam, self.cy_cam = K[0, 2], K[1, 2]
        self.mapx = None                  # undistortion maps (lazy: need size)
        self.dist = dist
        self.dist_model = dist_model      # 'radtan' | 'equidistant' (fisheye)
        self.K = K
        # FCU attitude -> camera: FCU-body(FRD) -> /imu0 (calibrated mounting)
        # -> camera (Kalibr). Comma-separated string param -> 3x3.
        rot_s = str(param('fcu_imu_rot', '')).strip()
        rot = ([float(v) for v in rot_s.replace(',', ' ').split()]
               if rot_s else list(FCU_IMU_ROT_ROWMAJOR))
        self.R_fcu2cam = R_CI @ np.asarray(rot, dtype=np.float64).reshape(3, 3)
        self.R_fcu2imu = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        self.get_logger().info(
            f'calib {calib_path} ({dist_model}, '
            f'f={self.fx:.1f} cx={self.cx_cam:.1f} cy={self.cy_cam:.1f})')

        # --- state ---------------------------------------------------------
        self.pose_t = deque()             # odomimu stamps
        self.pose_p = deque()             # p_IinG (3,)
        self.pose_R = deque()             # R_ItoG (3,3)
        self.imu_tilt = ImuTilt()         # acc+gyro roll/pitch (LAST fallback)
        self.imu_last_t = -1.0            # stamp of last IMU sample
        self.fcu_alt = None               # cached alt_at(bag_t) callable
        self.fcu_tilt = None              # cached tilt_at(bag_t) callable
        self.fcu_hdg = None               # cached hdg_at(bag_t) callable
        self.fcu_tried = False            # ULog parse attempted (replay)
        self.fcu_alt_t = deque(maxlen=3000)  # live FCU altitude stamps (~60 s)
        self.fcu_alt_z = deque(maxlen=3000)  # live FCU height, m (ENU, up)
        self.fcu_roll = deque(maxlen=3000)   # live FCU camera roll (rad)
        self.fcu_pitch = deque(maxlen=3000)  # live FCU camera pitch (rad)
        self.fcu_hdg_vals = deque(maxlen=3000)  # live FCU NED heading, unwrapped
        self.fcu_alt_recv = -1.0          # wall-clock time of last live sample
        self.fcu_att_seen = False         # live orientation valid, logged once
        self.fcu_alt_warned = False       # stamped-clock mismatch logged once
        self.psi_last = 0.0               # last good yaw proxy (VIO fallback)
        self.frame_q = deque(maxlen=4)    # (k, t, gray) every stride-th frame
        self.frame_idx = 0
        self.particles = None
        self.prev = None                  # (t, p, R_CtoG) of last processed
        self.last_good = None             # (x, y, yaw, t, psi) — CONFIRMED only
        self.lost_count = 0
        self.reloc_fails = 0
        self.recovering = False           # a rescan hypothesis awaits CONFIRMATION
        self.recover_count = 0            # consecutive good frames since the rescan
        self.reseed_t = None              # time of the last (re)seeding
        self.vio_map_rot = None           # EMA of the compass-observed VIO->map yaw
        self.scan_buf = deque()           # (t, patch, coverage, p_vio 2-tuple/None)
        self.local_rescan_t = deque()     # times of recent LOCAL rescans (evidence)
        self.scan_wait_start = None       # t of the first valid frame (scan init)
        self.scan_first_usable_t = None   # t of the first USABLE buffered frame
        self.global_pending_since = None  # t when the global scan began waiting
        self.log = []

        # --- health monitor state (step 3) --------------------------------
        self.last_gated = False           # last processed frame was coverage-gated
        self.cov_gated_run = 0            # consecutive gated frames (motion-only)
        self.vio_jumps = 0                # VIO 'resume' events (motion-jump gates)
        self.innov = None                 # EMA position innovation [m]
        self.prev_meas_est = None         # estimate at the previous MEASUREMENT frame
        self.prev_frame_gated = False     # for innovation across consecutive meas frames
        self.preinit_motion = np.zeros(2)  # map-frame displacement while point-init waits
        self.health = {'k': -1, 't': 0.0, 'wall_s': 0.0, 'note': 'ok',
                       'x': None, 'y': None, 'yaw_deg': None, 'score': None,
                       'spread': None, 'neff': None, 'coverage': None,
                       'alt': None, 'vio': 'none', 'innov_m': None,
                       'reloc_fails': 0, 'locals_recent': 0, 'recover_n': 0,
                       'cov_gated_run': 0, 'vio_jumps': 0}

        # --- EV odometry state (step 4) ------------------------------------
        self.ev_anchor = None           # (t, p) wide-baseline velocity anchor
        self.ev_v = None                # last good VIO velocity, base_link FLU
        self.ev_v_t = -1.0              # stamp of that velocity sample
        self.ev_est = None              # (x, y, yaw, spread, neff) latest MCL
        self.ev_pose_t = -1.0           # stamp of the last mcl.step

        self.create_subscription(Image, str(param('frame_topic',
                                                  '/cam0/image_raw')),
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(Imu, str(param('imu_topic', '/imu0')),
                                 self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, str(param('odom_topic',
                                                     '/ov_msckf/odomimu')),
                                 self.on_odom, 10)
        fcu_topic = str(param('fcu_alt_topic', ''))
        if fcu_topic:
            self.create_subscription(PoseStamped, fcu_topic, self.on_fcu_alt,
                                     qos_profile_sensor_data)
        # process at most one queued frame per tick; ticks faster than the
        # expected frame rate (stride/25 Hz) so the CNN sets the pace
        self.create_timer(0.05, self.process_one)
        # health monitor: 1 Hz JSON snapshot on /mcl/health (std_msgs/String
        # keeps it lightweight — no custom msg packages; the step-5 guard and
        # `ros2 topic echo /mcl/health` both consume it directly)
        self.health_pub = self.create_publisher(String, 'mcl/health', 10)
        self.create_timer(1.0, self.publish_health)
        # GPS-substitute EV odometry (step 4): remapped to
        # /mavros/odometry/out by the launch file when MAVROS runs
        self.ev_pub = self.create_publisher(Odometry, 'mcl/odom', 10)
        self.get_logger().info('MCL node ready (waiting for VIO + camera)')

    # ------------------------------------------------------------------ ROS
    def on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        R = quat_xyzw_to_R(q.x, q.y, q.z, q.w)
        self.pose_t.append(t)
        self.pose_p.append(pos)
        self.pose_R.append(R)
        if len(self.pose_t) > 2000:       # ~80 s at 25 Hz
            self.pose_t.popleft()
            self.pose_p.popleft()
            self.pose_R.popleft()
        self.publish_ev(msg, t, pos, R)

    def publish_ev(self, msg, t, p, R_ItoG):
        """GPS-substitute EV odometry (step 4) for the PX4 EKF2 External-Vision
        input: nav_msgs/Odometry at the odomimu rate (~200 Hz) on 'mcl/odom'.

        Channel split (configure EKF2_EV_CTRL = horizontal position + velocity):
          twist = VIO velocity by wide-baseline (~80 ms) finite difference,
                  expressed in base_link FLU (VIO global -> VIO IMU -> FCU
                  body FRD -> FLU). Pure VIO: fresh every message, survives
                  MCL loss; PX4 rotates it into its local frame with its own
                  attitude. NOT rotated by the MCL yaw — that would make the
                  velocity channel depend on the very estimator it is
                  supposed to keep alive through an MCL loss.
          pose  = RAW MCL estimate in the ENU map frame (x east / y north of
                  the MAP CENTER — set the EKF2 global origin there), z=0
                  NOT fused (baro owns altitude), published ONLY while the
                  filter state is 'tracking'. Repeating it between the ~5 Hz
                  filter steps is fine: the covariance (particle spread)
                  dwarfs the per-step motion, and the EKF2 interpolates with
                  its own prediction.

        Honesty rules (why not pre-fuse VIO+MCL position in the node: the
        EKF2 must see INGREDIENTS, not conclusions — pre-fused position would
        double-count VIO (velocity is already published), and time-correlated
        MCL errors would make the EKF2 overconfident):
          - gated frames (coverage < SCAN_MIN_COV), stale filter (> 1 s
            without an mcl.step), lost / recovering / pre-init -> position
            covariance EV_HUGE: VELOCITY-ONLY messages; the EKF2 dead-reckons
            position instead of trusting a dead or unconfirmed pose. This
            also prevents a LOST->recovery reseed from TELEPORTING the EKF2
            to an unconfirmed rescan hypothesis.
          - VIO glitch (non-finite / |v| / tilt gates) -> velocity frozen at
            the last good sample; stale beyond EV_VEL_STALE_S -> EV_HUGE.
        """
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'odom'       # ENU world (MAVROS: LOCAL_FRD)
        out.child_frame_id = 'base_link'   # twist frame (MAVROS: FLU -> FRD)

        # --- twist: VIO velocity, WIDE-BASELINE finite difference ------------
        # odomimu streams at ~200 Hz (5 ms): a consecutive-sample difference
        # amplifies the ~4 mm position noise to ~1 m/s (measured on bag
        # 190020). An ~80 ms sliding anchor gives ~0.07 m/s at the cost of
        # ~40 ms lag — and stays free of any assumption about what SchurVINS
        # fills into the twist field.
        if self.ev_anchor is None:
            self.ev_anchor = (t, p)
        else:
            dt_a = t - self.ev_anchor[0]
            if 0.06 <= dt_a <= 0.5:
                v_vio = (p - self.ev_anchor[1]) / dt_a
                # tilt gate uses the same camera_tilt convention as
                # process_one's VIO classification: a diverged VIO attitude
                # freezes the velocity instead of mis-rotating it
                vr, vpi = camera_tilt(R_ItoG @ self.R_CI.T)
                if (np.isfinite(v_vio).all()
                        and math.hypot(v_vio[0], v_vio[1]) < EV_VEL_MAX
                        and abs(v_vio[2]) < EV_VZ_MAX
                        and abs(vr) < np.radians(55)
                        and abs(vpi) < np.radians(55)):
                    v_frd = self.R_fcu2imu.T @ (R_ItoG.T @ v_vio)
                    self.ev_v = np.array([v_frd[0], -v_frd[1], -v_frd[2]])
                    self.ev_v_t = t
                    self.ev_anchor = (t, p)   # slide only on a CLEAN sample
            elif dt_a > 0.5:
                self.ev_anchor = (t, p)       # re-anchor after a gap
        if self.ev_v is not None and t - self.ev_v_t <= EV_VEL_STALE_S:
            out.twist.twist.linear.x = float(self.ev_v[0])
            out.twist.twist.linear.y = float(self.ev_v[1])
            out.twist.twist.linear.z = float(self.ev_v[2])
            out.twist.covariance[0] = out.twist.covariance[7] = \
                out.twist.covariance[14] = EV_VEL_VAR
        else:
            out.twist.covariance[0] = out.twist.covariance[7] = \
                out.twist.covariance[14] = EV_HUGE
            # stale velocity = no data: NaN, NOT zeros + huge variance (EKF2
            # still fuses the zero with a small but non-zero Kalman gain)
            out.twist.twist.linear.x = float('nan')
            out.twist.twist.linear.y = float('nan')
            out.twist.twist.linear.z = float('nan')
        out.twist.covariance[21] = out.twist.covariance[28] = \
            out.twist.covariance[35] = EV_HUGE   # angular velocity not fused

        # --- pose: raw MCL estimate, only while tracking -------------------
        tracking = (self.particles is not None and self.ev_est is not None
                    and not self.last_gated and not self.recovering
                    and self.lost_count == 0
                    and t - self.ev_pose_t <= EV_POSE_STALE_S)
        pc = out.pose.covariance
        if tracking:
            x, y, yaw, spread, neff = self.ev_est
            out.pose.pose.position.x = float(x)   # ENU map frame: x east,
            out.pose.pose.position.y = float(y)   # y north of the map center
            out.pose.pose.position.z = 0.0        # z not fused (baro owns it)
            # MCL yaw (optical-x azimuth) -> body ENU heading:
            # yaw_mcl = a*h_ned + b  =>  h_enu = pi/2 - (yaw - b)/a
            th = (yaw if abs(self.yaw_a) < 1e-6 else
                  math.pi / 2.0
                  - (yaw - math.radians(self.yaw_b)) / self.yaw_a)
            out.pose.pose.orientation.z = math.sin(th / 2.0)
            out.pose.pose.orientation.w = math.cos(th / 2.0)
            # per-axis variance: spread is the 2D RADIAL RMS (=> /2 per
            # axis); neff multiplier: a near-degenerate cloud (about to be
            # resampled) understates its uncertainty
            neff_mult = max(1.0, self.n_particles / max(1.0, float(neff)))
            var_xy = max(EV_POS_VAR_MIN, 0.5 * spread * spread * neff_mult)
            pc[0] = pc[7] = var_xy
        else:
            pc[0] = pc[7] = EV_HUGE
            # gated pose = no data: NaN position/orientation, NOT zeros +
            # huge covariance — EKF2 fuses the zeros and the estimate
            # collapses toward the origin between honest fixes
            out.pose.pose.position.x = float('nan')
            out.pose.pose.position.y = float('nan')
            out.pose.pose.position.z = float('nan')
            out.pose.pose.orientation.x = float('nan')
            out.pose.pose.orientation.y = float('nan')
            out.pose.pose.orientation.z = float('nan')
            out.pose.pose.orientation.w = float('nan')
        pc[14] = pc[21] = pc[28] = pc[35] = EV_HUGE  # z + attitude not fused
        self.ev_pub.publish(out)

    def on_imu(self, msg):
        """Feed raw acc+gyro to the VIO-free tilt filter."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_tilt.update(t,
                             [msg.angular_velocity.x, msg.angular_velocity.y,
                              msg.angular_velocity.z],
                             [msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z])
        self.imu_last_t = t

    def ensure_fcu(self):
        """Lazily parse the FCU baro height + EKF attitude/heading from the
        ULog beside the bag (replay). Returns (alt_at, tilt_at, hdg_at);
        any may be None (then VIO is the fallback / no steering). Independent
        of SchurVINS — keeps the orthoprojection geometry sane if the VIO
        diverges."""
        if self.fcu_tried:
            return self.fcu_alt, self.fcu_tilt, self.fcu_hdg
        self.fcu_tried = True
        if not self.bag_dir:
            return None, None, None
        try:
            alt_at, tilt_at, hdg_at = bag_fcu_streams(self.bag_dir,
                                                      self.R_fcu2cam)
        except Exception as e:       # parsing/alignment failed
            self.get_logger().warn(f'FCU streams unavailable: {e}')
            alt_at = tilt_at = hdg_at = None
        self.fcu_alt, self.fcu_tilt, self.fcu_hdg = alt_at, tilt_at, hdg_at
        if alt_at is not None or tilt_at is not None:
            self.get_logger().info(
                'FCU baro altitude + EKF attitude active (ULog, '
                'orthoprojection decoupled from VIO)')
        return alt_at, tilt_at, hdg_at

    # ------------------------------------------- FCU altitude + attitude
    def on_fcu_alt(self, msg):
        """Live FCU height + EKF attitude from MAVROS local_position (ENU z
        = height above the local origin, i.e. takeoff point — same quantity
        as the replay path's -vehicle_local_position.z; orientation = EKF
        attitude of base_link in the ENU world)."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not self.fcu_alt_t:
            self.get_logger().info('MAVROS FCU altitude active (live baro)')
        self.fcu_alt_t.append(t)
        self.fcu_alt_z.append(float(msg.pose.position.z))
        q = msg.pose.orientation
        n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if n2 > 0.25:                # orientation filled (not all-zero)
            roll, pitch = mavros_enu_tilt(q.x, q.y, q.z, q.w, self.R_fcu2cam)
            self.fcu_roll.append(roll)
            self.fcu_pitch.append(pitch)
            hdg = enu_quat_to_ned_heading(q.x, q.y, q.z, q.w)
            for prev in reversed(self.fcu_hdg_vals):  # unwrap vs last finite
                if np.isfinite(prev):
                    hdg = prev + wrap_angle(hdg - prev)
                    break
            self.fcu_hdg_vals.append(hdg)
            if not self.fcu_att_seen:
                self.fcu_att_seen = True
                self.get_logger().info(
                    'MAVROS FCU attitude active (live EKF tilt + heading)')
        else:
            self.fcu_roll.append(float('nan'))
            self.fcu_pitch.append(float('nan'))
            self.fcu_hdg_vals.append(float('nan'))
        self.fcu_alt_recv = time.time()

    def live_fcu_alt(self, t):
        """Height at image time t from the live MAVROS deque, or None.

        Matches on header stamps first (correct when MAVROS timesync is
        configured). If the stamps never line up with the camera clock but
        fresh data IS arriving, falls back to the newest sample — the baro
        changes slowly, so ~1 s staleness is harmless for the ortho scale
        (and strictly better than the VIO-z fallback)."""
        if not self.fcu_alt_t:
            return None
        ts = list(self.fcu_alt_t)
        zs = list(self.fcu_alt_z)
        import bisect
        i = bisect.bisect_left(ts, t)
        lo, hi = max(0, i - 1), min(len(ts) - 1, i)
        near = min(abs(ts[lo] - t), abs(ts[hi] - t))
        if near <= FCU_ALT_MAX_AGE:
            if ts[hi] == ts[lo]:
                return zs[lo]
            f = min(max((t - ts[lo]) / (ts[hi] - ts[lo]), 0.0), 1.0)
            return zs[lo] * (1.0 - f) + zs[hi] * f
        if time.time() - self.fcu_alt_recv < 2.0:
            if not self.fcu_alt_warned:
                self.fcu_alt_warned = True
                self.get_logger().warn(
                    'FCU altitude stamps do not match the camera clock '
                    '(MAVROS timesync off?) — using newest sample')
            return zs[-1]
        return None

    def fcu_alt_at(self, t):
        """FCU baro height (m) at image time t — ONE source abstraction for
        replay and live: live mode reads the MAVROS topic, replay reads the
        clock-aligned ULog beside the bag. None -> the caller falls back to
        VIO z (+ alt_anchor). Independent of SchurVINS: keeps the
        orthoprojection scale sane if the VIO diverges."""
        if self.bag_dir:
            alt_at = self.ensure_fcu()[0]
            return alt_at(t) if alt_at is not None else None
        return self.live_fcu_alt(t)

    def live_fcu_tilt(self, t):
        """(roll, pitch) at image time t from the live MAVROS deque, or None.

        Stamp-matched bisect + linear interpolation within FCU_ATT_MAX_AGE.
        No fresh-sample fallback (unlike the altitude): tilt changes fast
        during turns, and a wrong tilt is worse than falling back to the
        VIO attitude — on stamp mismatch the caller uses VIO instead."""
        if not self.fcu_att_seen or not self.fcu_alt_t:
            return None
        ts = list(self.fcu_alt_t)
        rs = list(self.fcu_roll)
        ps = list(self.fcu_pitch)
        import bisect
        i = bisect.bisect_left(ts, t)
        lo, hi = max(0, i - 1), min(len(ts) - 1, i)
        near = min(abs(ts[lo] - t), abs(ts[hi] - t))
        if near > FCU_ATT_MAX_AGE or not (np.isfinite(rs[lo]) and
                                          np.isfinite(rs[hi])):
            return None
        if ts[hi] == ts[lo]:
            return rs[lo], ps[lo]
        f = min(max((t - ts[lo]) / (ts[hi] - ts[lo]), 0.0), 1.0)
        return (rs[lo] * (1.0 - f) + rs[hi] * f,
                ps[lo] * (1.0 - f) + ps[hi] * f)

    def fcu_tilt_at(self, t):
        """FCU EKF attitude tilt (roll, pitch) at image time t — replay/live
        abstraction like fcu_alt_at. None -> caller falls back to VIO
        attitude, then the raw-IMU complementary filter."""
        if self.bag_dir:
            tilt_at = self.ensure_fcu()[1]
            return tilt_at(t) if tilt_at is not None else None
        return self.live_fcu_tilt(t)

    def live_fcu_hdg(self, t):
        """NED heading (rad, unwrapped) at image time t from the live MAVROS
        deque, or None. Same stamp-matched interpolation as the tilt; no
        fresh-sample fallback (a wrong heading mis-steers every scan)."""
        if not self.fcu_att_seen or not self.fcu_alt_t:
            return None
        ts = list(self.fcu_alt_t)
        hs = list(self.fcu_hdg_vals)
        import bisect
        i = bisect.bisect_left(ts, t)
        lo, hi = max(0, i - 1), min(len(ts) - 1, i)
        near = min(abs(ts[lo] - t), abs(ts[hi] - t))
        if near > FCU_ATT_MAX_AGE or not (np.isfinite(hs[lo])
                                          and np.isfinite(hs[hi])):
            return None
        if ts[hi] == ts[lo]:
            return hs[lo]
        f = min(max((t - ts[lo]) / (ts[hi] - ts[lo]), 0.0), 1.0)
        return hs[lo] * (1.0 - f) + hs[hi] * f

    def hdg_at(self, t):
        """FCU EKF NED heading (rad, unwrapped) at image time t — replay/live
        abstraction like fcu_tilt_at. None -> no compass steering (the scans
        fall back to full-circle yaw sweeps)."""
        if self.bag_dir:
            hdg_at = self.ensure_fcu()[2]
            return hdg_at(t) if hdg_at is not None else None
        return self.live_fcu_hdg(t)

    def yaw_pred_at(self, t):
        """Compass-predicted MCL map yaw of the camera optical x-axis, or
        None when no heading is available: yaw = a*hdg + radians(b)."""
        hdg = self.hdg_at(t)
        if hdg is None:
            return None
        return self.yaw_a * hdg + math.radians(self.yaw_b)

    def on_image(self, msg):
        if msg.encoding != 'mono8':
            self.get_logger().error(
                f'unsupported encoding {msg.encoding!r} (need mono8)')
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = np.frombuffer(msg.data, dtype=np.uint8)
        img = img.reshape(msg.height, msg.step)[:, :msg.width].copy()
        in_range = (self.frame_start <= 0 or self.frame_idx >= self.frame_start) \
            and (self.frame_end <= 0 or self.frame_idx < self.frame_end)
        if in_range and self.frame_idx % self.stride == 0:
            # k = CAMERA-frame index (0, stride, 2*stride, ...) — debug files
            # and replay_log use it, so the stride is visible in the numbering
            self.frame_q.append((self.frame_idx, t, img))
        self.frame_idx += 1

    def pose_at(self, t):
        """Index of the nearest odomimu pose within POSE_MAX_AGE, else None."""
        if not self.pose_t:
            return None
        ts = list(self.pose_t)            # bisect needs random access list
        import bisect
        i = bisect.bisect_left(ts, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts):
                if best is None or abs(ts[j] - t) < abs(ts[best] - t):
                    best = j
        if best is None or abs(ts[best] - t) > POSE_MAX_AGE:
            return None
        return best

    # ------------------------------------------------------------ scans
    def steered_global_scan(self):
        """Compass-steered multi-frame global nomination (kidnapped scan).

        Scores the buffered patches over a full-map grid within a
        +-STEER_SPAN yaw window around each frame's compass-predicted map
        yaw, motion-compensates the per-frame grids onto the newest frame
        (VIO track rotated into the map frame by the compass-observed
        VIO->map rotation), and takes the per-position MEDIAN across
        frames: persistent truth survives while frame-specific false peaks
        wash out (validated on bag 190020: truth within the top-10 regions
        in 10/11 test windows vs 6/11 for the best single frame).

        Returns (regions, yaw_ref): regions = [(score, x_m, y_m), ...] top
        SCAN_TOPK deduplicated candidates in MCL map meters, yaw_ref = the
        predicted yaw of the newest scored frame (seeding yaw). None when
        steering is impossible (no heading / no VIO track / no usable
        frames) — the caller falls back to the unsteered coarse scan.

        NOTE: takes tens of seconds (SCAN_FRAMES full-map grid passes) — in
        live operation the drone must HOVER while it runs (guard state
        machine, step 5)."""
        t0 = time.time()
        mcl = self.mcl
        h, w = self.map_rgb.shape[:2]
        step_px = SCAN_STEP_M / self.gsd_map
        margin = int(step_px)
        xs = np.arange(margin, w - margin, step_px)
        ys = np.arange(margin, h - margin, step_px)
        nx, ny = len(xs), len(ys)
        pos = [(px, py) for py in ys for px in xs]      # flat, row-major

        # --- frames to score: window, VIO-plausible, steerable, in coverage
        if not self.scan_buf or self.vio_map_rot is None:
            return None
        t_new = self.scan_buf[-1][0]
        cands = [f for f in self.scan_buf
                 if f[3] is not None and f[0] >= t_new - SCAN_WINDOW_S
                 and f[2] >= SCAN_MIN_COV]
        if not cands:
            return None
        ref_p = cands[-1][3]            # newest VIO position (gate anchor)
        frames = []
        for ft, fpatch, fcov, fp in cands:
            yaw0 = self.yaw_pred_at(ft)
            if yaw0 is None:
                continue
            # a VIO jump inside the window would corrupt the compensation
            dt = max(0.05, t_new - ft)
            if np.hypot(fp[0] - ref_p[0], fp[1] - ref_p[1]) > 3.0 * V_MAX * dt + 60.0:
                continue
            frames.append((ft, fpatch, fcov, fp, yaw0))
        if not frames:
            return None
        if len(frames) > SCAN_FRAMES:
            idx = np.unique(np.linspace(0, len(frames) - 1,
                                        SCAN_FRAMES).round().astype(int))
            frames = [frames[i] for i in idx]
        ref = frames[-1]

        grids = []
        for ft, fpatch, fcov, fp, yaw0 in frames:
            mcl.set_coverage(fcov)
            mcl.set_drone_patch(fpatch)
            df = mcl.drone_feat
            best = np.full(nx * ny, -1.0)
            for yaw in np.linspace(yaw0 - STEER_SPAN, yaw0 + STEER_SPAN,
                                   STEER_YAWS):
                for b0 in range(0, len(pos), 512):
                    chunk = pos[b0:b0 + 512]
                    patches = np.empty((len(chunk), PATCH_PX, PATCH_PX, 3),
                                       dtype=np.uint8)
                    for i, (px, py) in enumerate(chunk):
                        patches[i] = cropRectangularSectionFromImage(
                            self.map_rgb, px, py, -yaw, mcl.src_dim,
                            PATCH_PX, 1.0)
                    means = patches.reshape(len(chunk), -1).mean(axis=1)
                    tt = torch.from_numpy(
                        patches.astype(np.float32)).permute(0, 3, 1, 2)
                    tt = tt.to(mcl.device)
                    with torch.no_grad():
                        mf = mcl.branch(tt)
                        s = mcl.decision(df.expand(tt.shape[0], -1),
                                         mf).squeeze(-1).cpu().numpy()
                    s[means < 5.0] = -1.0        # map-edge black crops
                    sl = slice(b0, b0 + len(chunk))
                    best[sl] = np.maximum(best[sl], s)
            best = best.reshape(ny, nx)
            # motion-compensate onto the newest frame's grid
            dm = vio_to_map_delta((fp[0] - ref[3][0], fp[1] - ref[3][1]),
                                  self.vio_map_rot)
            dpx = dm[0] / self.gsd_map          # map px: x right
            dpy = -dm[1] / self.gsd_map         # map py: y down (y_m up)
            cols = (xs[None, :] + dpx - xs[0]) / step_px * np.ones((ny, 1))
            rows = ((ys[:, None] + dpy - ys[0]) / step_px) * np.ones((1, nx))
            grids.append(bilinear_grid_sample(best, cols, rows))
        S = np.median(np.stack(grids), 0) if len(grids) > 1 else grids[0]

        # top-K regions, deduplicated by 100 m (as validated)
        flat = sorted(((float(S[j, i]), i, j)
                       for j in range(ny) for i in range(nx)), reverse=True)
        regions, taken = [], []
        for s, i, j in flat:
            px, py = xs[i], ys[j]
            if all(np.hypot(px - qx, py - qy) > 100.0 / self.gsd_map
                   for qx, qy in taken):
                taken.append((px, py))
                regions.append((s,
                                (px - self.cx_px) * self.gsd_map,
                                (self.cy_px - py) * self.gsd_map))
            if len(regions) >= SCAN_TOPK:
                break
        self.get_logger().info(
            f'steered global scan: {len(frames)} frames / '
            f'{t_new - frames[0][0]:.0f} s window in {time.time() - t0:.1f} s'
            f' — top: ' + ', '.join(f'{s:.2f}@({x:.0f},{y:.0f})'
                                     for s, x, y in regions[:3]))
        return regions, ref[4]

    # --------------------------------------------------------------- filter
    def process_one(self):
        # drop the oldest frame if we are backing up (processing too slow)
        while len(self.frame_q) >= self.frame_q.maxlen:
            k_drop = self.frame_q.popleft()[0]
            self.get_logger().warn(f'dropping frame k={k_drop} (backlog)')
        if not self.frame_q:
            return
        k, t, gray = self.frame_q.popleft()

        # --- geometry for the drone PATCH: FCU EKF attitude + baro height.
        # These come from the flight controller, NOT from SchurVINS, so a
        # valid patch is made for EVERY frame even if the VIO diverges or
        # stops publishing. The VIO pose only drives the motion model below,
        # and only while it is trustworthy. -------------------------------
        # Tilt priority: FCU EKF attitude -> VIO attitude -> raw-IMU
        # complementary filter. The raw-IMU filter is LAST because this
        # airframe's prop vibration corrupts the accelerometer (|a-g| p50
        # 7.4 m/s^2, 96% of samples): it tracked 6-10 deg of phantom tilt
        # and caused the 2026-09-16 transit wander/false-lock (see project
        # memory). The FCU EKF fuses the same vibration cleanly.
        roll = pitch = alt = None
        fcu_rp = self.fcu_tilt_at(t)
        if fcu_rp is not None:
            roll, pitch = fcu_rp
        fcu_alt = self.fcu_alt_at(t)
        if fcu_alt is not None and np.isfinite(fcu_alt):
            alt = float(fcu_alt)          # HONEST height — no clamp: the old
                                          # max(20, z) faked a 40 m coverage on
                                          # the ground and defeated the
                                          # SCAN_MIN_COV gate (step 3)

        # VIO pose -> motion model only. Classify it:
        #   'ok'     sane tilt/alt AND plausible per-frame step
        #   'resume' sane tilt/alt but huge step (gap/glitch) -> anchor, but
        #            do NOT jump the cloud this frame
        #   'bad'    missing or catastrophic (observed: z=8 km, roll=160 deg
        #            when SchurVINS diverges) -> ignore; patch still saved
        p = R_CtoG = None
        vio = 'bad'
        i = self.pose_at(t)
        if i is not None:
            pv, R_ItoG = self.pose_p[i], self.pose_R[i]
            Rv = R_ItoG @ self.R_CI.T
            vr, vpi = camera_tilt(Rv)
            va = float(pv[2]) + self.alt_anchor
            if (np.isfinite(va) and ALT_MIN <= va <= 400.0
                    and abs(vr) < np.radians(55)
                    and abs(vpi) < np.radians(55)):
                p, R_CtoG = pv, Rv
                if self.prev is None:
                    vio = 'ok'
                else:
                    mdx, mdy, mdyaw = vio_motion(
                        self.prev[1], self.prev[2], pv, Rv)
                    # glitch gates SCALE with the frame gap: at the normal
                    # 0.2 s spacing they stay tight (15 m / 45 deg), but a
                    # long gap — frames dropped while a global scan blocked
                    # the processing thread for ~30 s — legitimately
                    # accumulates V_MAX*dt of VIO travel; injecting it is
                    # correct, freezing the cloud behind the drone is not.
                    dt_gap = max(0.0, t - self.prev[0] - 0.5)
                    lim = 15.0 + V_MAX * dt_gap
                    yaw_lim = np.radians(45.0) + dt_gap * np.radians(90.0)
                    vio = 'ok' if (abs(mdx) < lim and abs(mdy) < lim
                                   and abs(mdyaw) < yaw_lim) else 'resume'
        if vio == 'resume':
            self.vio_jumps += 1   # health counter: motion-jump gate fired

        # tilt fallbacks: VIO attitude (classified ok/resume), then raw-IMU
        # complementary filter (vibration-corrupted on this airframe — last
        # resort, e.g. bench tests with no FCU and no VIO)
        if roll is None and vio in ('ok', 'resume'):
            roll, pitch = camera_tilt(R_CtoG)

        # cross-check the FCU tilt chain against the (independent) VIO
        # attitude when both are available — a MAVROS/ULog convention error
        # would show up as a persistent large disagreement
        if (fcu_rp is not None and vio in ('ok', 'resume')
                and k % 200 == 0):
            dr = abs(np.degrees(fcu_rp[0] - vr))
            dp = abs(np.degrees(fcu_rp[1] - vpi))
            if max(dr, dp) > 10.0:
                self.get_logger().warn(
                    f'k={k}: FCU vs VIO tilt disagree '
                    f'(droll={dr:.0f} dpitch={dp:.0f} deg) — check the '
                    f'fcu_imu_rot mounting rotation / MAVROS conventions')
        if roll is None and (self.imu_tilt.ready() and self.imu_last_t >= 0.0
                             and abs(t - self.imu_last_t) < 0.5):
            roll, pitch = self.imu_tilt.camera_tilt(self.R_CI)
        if alt is None and vio in ('ok', 'resume'):
            alt = float(p[2]) + self.alt_anchor   # honest VIO z (no clamp)

        if roll is None or alt is None:
            if k % 100 == 0:
                self.get_logger().warn(
                    f'k={k}: no FCU/VIO tilt / height available yet — skipped')
            self.health.update(note='no_geom', k=int(k), t=float(t), vio=vio)
            return
        # geometry envelope: honest altitude floor (ALT_MIN) — near-ground
        # frames have degenerate ortho geometry; extreme tilt = VIO divergence.
        # With FCU attitude + baro height these stay valid during a VIO
        # failure; only real out-of-envelope frames are rejected.
        if not (np.isfinite(alt) and np.isfinite(roll) and np.isfinite(pitch)
                and ALT_MIN <= alt <= 400.0
                and abs(roll) < np.radians(50) and abs(pitch) < np.radians(50)):
            if k % 100 == 0:
                self.get_logger().warn(
                    f'k={k}: out-of-envelope (alt={alt:.0f} m '
                    f'roll={np.degrees(roll):.0f} pitch={np.degrees(pitch):.0f} '
                    f'deg) — frame skipped')
            self.health.update(note='envelope', k=int(k), t=float(t), vio=vio,
                               alt=(round(float(alt), 1)
                                    if np.isfinite(alt) else None))
            return

        # --- undistort + orthoproject + patch (same as bag_mcl.py) ---------
        try:
            if self.mapx is None:
                if self.dist_model == 'equidistant':
                    self.mapx, self.mapy = cv2.fisheye.initUndistortRectifyMap(
                        self.K, self.dist, np.eye(3), self.K,
                        (gray.shape[1], gray.shape[0]), cv2.CV_32FC1)
                else:
                    self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                        self.K, self.dist, None, self.K,
                        (gray.shape[1], gray.shape[0]), cv2.CV_32FC1)
            raw = gray.copy() if self.save_debug else None
            gray = cv2.remap(gray, self.mapx, self.mapy, cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            ortho, mask_o, extent = orthoproject(
                rgb, alt, roll, pitch, self.fx, self.fy,
                self.cx_cam, self.cy_cam)
            coverage = valid_coverage_m(mask_o, extent)
            # LOW-COVERAGE GATE (step 3): below SCAN_MIN_COV the patch is
            # outside the model's trained envelope — measurement scores are
            # mushy (~0.99 everywhere) and a measurement update false-locks
            # the cloud. Gated frames stay MOTION-ONLY (predict, no update,
            # no lost/recovery state change); the patch is still buffered
            # for scans (which apply the same gate themselves).
            gated = coverage < SCAN_MIN_COV
            if not gated:
                self.mcl.set_coverage(coverage)
            patch, _, _ = extract_patch(ortho, mask_o, extent, size_m=coverage)
            if patch.shape[0] != PATCH_PX:
                patch = cv2.resize(patch, (PATCH_PX, PATCH_PX),
                                   interpolation=cv2.INTER_LINEAR)
            if not gated:
                self.mcl.set_drone_patch(patch)
        except Exception as e:
            # one bad frame must never kill the node
            self.get_logger().warn(f'k={k}: frame processing failed ({e}) '
                                   f'— frame skipped')
            self.health.update(note='proc_fail', k=int(k), t=float(t), vio=vio)
            return

        # --- scan buffer + compass steering (kidnapped-scan input) ----------
        # Every valid frame feeds the 25 s buffer used by the (slow) steered
        # global scan, and updates the EMA of theta = VIO->map rotation
        # (compass-predicted map yaw minus the VIO yaw proxy). theta lets the
        # scan motion-compensate buffered frames using ABSOLUTE VIO positions
        # (no drift accumulation from integrating body deltas).
        yaw_pred = self.yaw_pred_at(t)
        self.scan_buf.append((t, patch, coverage,
                              (float(p[0]), float(p[1]))
                              if vio in ('ok', 'resume') else None))
        while (self.scan_buf
               and self.scan_buf[0][0] < t - SCAN_WINDOW_S - 5.0):
            self.scan_buf.popleft()
        if yaw_pred is not None and vio in ('ok', 'resume'):
            d = wrap_angle(yaw_pred - vio_yaw_proxy(R_CtoG))
            self.vio_map_rot = (d if self.vio_map_rot is None
                                else self.vio_map_rot + 0.1 * wrap_angle(
                                    d - self.vio_map_rot))

        # --- initialization ------------------------------------------------
        if self.particles is None:
            if self.init_mode == 'point':
                # LOW-COVERAGE WAIT (step 3): the init scan below sweeps a
                # mushy below-envelope patch on the ground/low hover and
                # false-locks at 0.99 (the realistic-init replays burned
                # 60-230 s tracking ~336 m off before climbing into the
                # envelope). Wait for coverage; keep the VIO anchor fresh
                # and accumulate the map-frame displacement so the click
                # prior stays anchored to the takeoff point. Absolute cap
                # for the coverage-never-opens degenerate case.
                if coverage < SCAN_MIN_COV:
                    if self.scan_wait_start is None:
                        self.scan_wait_start = t
                    if vio in ('ok', 'resume'):
                        if self.prev is not None and yaw_pred is not None:
                            dx, dy, _ = vio_motion(
                                self.prev[1], self.prev[2], p, R_CtoG)
                            c, s = math.cos(yaw_pred), math.sin(yaw_pred)
                            self.preinit_motion[0] += c * dx - s * dy
                            self.preinit_motion[1] += s * dx + c * dy
                        self.prev = (t, p, R_CtoG)
                        self.psi_last = vio_yaw_proxy(R_CtoG)
                    if t - self.scan_wait_start < SCAN_WAIT_CAP_S:
                        if k % 100 == 0:
                            self.get_logger().info(
                                f'k={k}: point init waiting for coverage '
                                f'({coverage:.0f} < {SCAN_MIN_COV:.0f} m, '
                                f'alt={alt:.0f} m)')
                        self.health.update(note='low_cov', k=int(k), t=float(t),
                                           vio=vio, coverage=round(float(coverage), 1),
                                           alt=round(float(alt), 1))
                        return
                    self.get_logger().warn(
                        f'k={k}: coverage never reached {SCAN_MIN_COV:.0f} m '
                        f'in {SCAN_WAIT_CAP_S:.0f} s — init scan on the '
                        f'current below-envelope patch (expect false peaks)')
                cv2.imwrite(os.path.join(self.out_dir, 'frame0_patch.png'),
                            cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                lat = float(self.get_parameter('init_lat').value)
                lon = float(self.get_parameter('init_lon').value)
                if lat > -900.0 and lon > -900.0:
                    off_x = float(
                        self.get_parameter('map_geo_offset_x').value)
                    off_y = float(
                        self.get_parameter('map_geo_offset_y').value)
                    px, py = latlon_to_map_px(lat, lon, off_x, off_y)
                    self.get_logger().info(
                        f'init lat/lon ({lat:.6f},{lon:.6f}) -> '
                        f'map px ({px:.1f},{py:.1f})')
                else:
                    px = int(self.get_parameter('init_px_x').value)
                    py = int(self.get_parameter('init_px_y').value)
                # shift the click prior by the displacement accumulated while
                # waiting (so "click on the ground, init in the air" works)
                x0_m = ((px - self.cx_px) * self.gsd_map
                        + self.preinit_motion[0])
                y0_m = ((self.cy_px - py) * self.gsd_map
                        + self.preinit_motion[1])
                self.get_logger().info(
                    f'init point map px ({px:.1f},{py:.1f}) -> '
                    f'({x0_m:.1f},{y0_m:.1f}) m'
                    + (f' (+{np.hypot(*self.preinit_motion):.0f} m preinit '
                       f'motion)' if np.hypot(*self.preinit_motion) > 1.0
                       else '')
                    + f', region radius {self.init_radius_m:.0f} m')
                r = self.init_radius_m
                step = max(12.0, r / 10.0)
                # compass-steered yaw window when available, else full circle
                if yaw_pred is not None:
                    yaw0, span = yaw_pred, 2.0 * STEER_SPAN
                else:
                    yaw0, span = None, 2.0 * np.pi
                peaks = local_yaw_scan(
                    self.mcl, patch, x0_m, y0_m, radius_m=r, step_m=step,
                    yaw_steps=12, topk=5, yaw0=yaw0, yaw_span=span)
                self.particles = particles_from_peaks(
                    self.mcl, peaks, self.n_particles,
                    std=(max(15.0, step), max(15.0, step), 0.20, 0.03))
                self.recovering = True
                self.recover_count = 0
                self.reseed_t = t
            else:
                # scan init: WAIT until the buffer spans the validated
                # SCAN_WINDOW_S (multi-frame nomination; the drone climbs
                # while buffering) — capped so degenerate cases (no VIO, no
                # heading, coverage never in range) still init via fallback
                if self.scan_wait_start is None:
                    self.scan_wait_start = t
                usable = [f for f in self.scan_buf
                          if f[3] is not None and f[2] >= SCAN_MIN_COV
                          and self.yaw_pred_at(f[0]) is not None]
                if usable and self.scan_first_usable_t is None:
                    self.scan_first_usable_t = usable[0][0]
                span_s = (usable[-1][0] - usable[0][0]
                          if len(usable) >= 2 else 0.0)
                # readiness clock starts at the FIRST USABLE frame — the
                # ground/hover phase must not eat the budget (scan-during-
                # climb replay 2026-09-16: the 45 s cap expired ~9 s after
                # the coverage envelope opened, the scan ran on a 9 s span,
                # truth was NOT nominated and the lookalike (1216,608) won
                # at 0.93; a 25-30 s span at the same near-stationary site
                # DOES nominate truth — kidnap v4's global recovered to
                # 6.4 m median). Absolute fallback SCAN_WAIT_CAP_S for the
                # degenerate no-usable-frames-ever case.
                span_clock = (self.scan_first_usable_t
                              if self.scan_first_usable_t is not None
                              else self.scan_wait_start)
                if (span_s < SCAN_WINDOW_S
                        and t - span_clock < SCAN_WINDOW_S + 20.0
                        and t - self.scan_wait_start < SCAN_WAIT_CAP_S):
                    if k % 50 == 0:
                        self.get_logger().info(
                            f'k={k}: scan init buffering '
                            f'({len(usable)} frames, {span_s:.0f} s span)')
                    self.health.update(note='scan_buffer', k=int(k), t=float(t),
                                       vio=vio,
                                       coverage=round(float(coverage), 1),
                                       alt=round(float(alt), 1))
                    # keep the VIO anchor fresh so the post-scan gap (the
                    # ~30 s scan blocks the thread) is measured from the
                    # true last frame — same handoff the LOST path needed
                    if vio in ('ok', 'resume'):
                        self.prev = (t, p, R_CtoG)
                        self.psi_last = vio_yaw_proxy(R_CtoG)
                    return
                cv2.imwrite(os.path.join(self.out_dir, 'frame0_patch.png'),
                            cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                out = self.steered_global_scan()
                # the steered scan left mcl at a buffered frame's
                # coverage/patch — restore the current frame BEFORE the
                # stage-2 refinement (local_yaw_scan crops map patches at
                # mcl.src_dim = the current coverage) and the seeding
                self.mcl.set_coverage(coverage)
                self.mcl.set_drone_patch(patch)
                if out is not None:
                    regions, yaw_ref = out
                    # stage 2 — refine the top nominees with the CURRENT
                    # patch (same rationale as the LOST path: the
                    # nomination grid is coarse and the multi-frame median
                    # peak can sit 100-150 m off the true match, letting a
                    # well-centered lookalike win the post-seed contest)
                    peaks = []
                    for s, x, y in regions[:3]:
                        pk = local_yaw_scan(
                            self.mcl, patch, x, y, radius_m=192.0,
                            step_m=32.0, yaw_steps=STEER_YAWS, topk=1,
                            yaw0=yaw_ref, yaw_span=2.0 * STEER_SPAN)
                        peaks.append(pk[0] if pk else (s, x, y, yaw_ref))
                    peaks += [(s, x, y, yaw_ref)
                              for s, x, y in regions[3:]]
                    std = (48.0, 48.0, 0.15, 0.02)
                    self.get_logger().warn(
                        'init nominees -> refined: '
                        + '; '.join(
                            f'{s:.3f},{x:.0f},{y:.0f}->{rs:.3f},{rx:.0f},'
                            f'{ry:.0f}'
                            for (s, x, y), (rs, rx, ry, _r) in
                            zip(regions[:3], peaks[:3])))
                else:
                    self.get_logger().info(
                        'init coarse scan (full map, 8 yaws, unsteered)')
                    peaks = coarse_scan(self.mcl, patch, topk=5)
                    std = (30.0, 30.0, 0.30, 0.03)
                self.particles = particles_from_peaks(
                    self.mcl, peaks, self.n_particles, std=std)
                self.recovering = True
                self.recover_count = 0
                self.reseed_t = t

        # --- motion model: VIO while trustworthy, otherwise hover-diffuse.
        # On VIO failure/gap we propagate with zero translation + process
        # noise and let the CNN measurement + rescans relocalize — the motion
        # model only spreads the cloud, it does not localize. ----------------
        hover_ns = np.array([HOVER_NOISE, HOVER_NOISE,
                             NOISE_STD[2], NOISE_STD[3]])
        if vio == 'ok' and self.prev is not None:
            dx, dy, dyaw = vio_motion(self.prev[1], self.prev[2], p, R_CtoG)
            motion = (dx, dy, dyaw)
            ns = NOISE_STD.copy()
            psi = vio_yaw_proxy(R_CtoG)
            self.psi_last = psi
            self.prev = (t, p, R_CtoG)
        elif vio in ('ok', 'resume'):
            motion = (0.0, 0.0, 0.0)
            ns = hover_ns
            psi = vio_yaw_proxy(R_CtoG)
            self.psi_last = psi
            self.prev = (t, p, R_CtoG)      # re-anchor (drop accumulated delta)
            if vio == 'resume':
                self.get_logger().warn(
                    f'k={k}: VIO re-anchored after gap — zero-motion step')
        else:
            # VIO bad/missing: patch already built above; hover-diffuse and
            # keep the last good pose anchor (don't touch self.prev).
            motion = (0.0, 0.0, 0.0)
            ns = hover_ns
            psi = self.psi_last
            if k % 100 == 0:
                self.get_logger().warn(
                    f'k={k}: VIO bad/missing — patch from IMU+baro, '
                    f'hover/relocalize (no motion injected)')

        # --- filter step ----------------------------------------------------
        # LOW-COVERAGE GATE (step 3): below SCAN_MIN_COV the similarity
        # scores are mushy (~0.99 everywhere) and a measurement update
        # false-locks the cloud — skip the update, particles follow the VIO
        # motion only (weights / lost / recovery state stay frozen).
        self.last_gated = gated
        self.cov_gated_run = self.cov_gated_run + 1 if gated else 0
        decay = max(0.2, 1.0 - 0.4 * k / 50.0)
        rs = ROUGHEN_STD if self.lost_count == 0 else tuple(
            v * (1.0 + 2.0 * min(self.lost_count, 4)) for v in ROUGHEN_STD)
        particles, est, scores = self.mcl.step(
            self.particles, T_LIKELIHOOD, rs, decay,
            motion=motion, noise_std=tuple(ns), resample_eff=RESAMPLE_EFF,
            skip_update=gated)
        self.particles = particles
        # cache for the EV odometry publisher (publish_ev, ~25 Hz in on_odom):
        # the raw estimate + the honest per-axis covariance inputs. Validity
        # flags (last_gated / recovering / lost_count) are read live.
        self.ev_est = (est['x'], est['y'], est['yaw'],
                       est['spread'], self.mcl.last_neff)
        self.ev_pose_t = t

        if self.save_debug and k % self.debug_stride == 0:
            self.save_debug_frame(k, raw, gray, ortho, mask_o, patch, scores)

        # innovation (health metric): estimate displacement vs the
        # VIO-predicted displacement in the map frame — only between
        # CONSECUTIVE measurement frames (a gated gap spans many motion
        # steps and would fake a spike). ~0 while tracking; persistently
        # large = a false lookalike keeps re-capturing the estimate as the
        # cloud moves away (persistent-aliasing false lock).
        if not gated:
            if self.prev_meas_est is not None and not self.prev_frame_gated:
                c, s = math.cos(est['yaw']), math.sin(est['yaw'])
                d = math.hypot(
                    (est['x'] - self.prev_meas_est[0]) - (c * motion[0] - s * motion[1]),
                    (est['y'] - self.prev_meas_est[1]) - (s * motion[0] + c * motion[1]))
                self.innov = d if self.innov is None else (
                    (1.0 - INNOV_ALPHA) * self.innov + INNOV_ALPHA * d)
            self.prev_meas_est = (est['x'], est['y'])
        self.prev_frame_gated = gated

        self.log.append({'k': k, 't': t,
                         'x': est['x'], 'y': est['y'], 'yaw': est['yaw'],
                         'scale': est['scale'], 'spread': est['spread'],
                         'neff': self.mcl.last_neff,
                         'score': (float(scores.max())
                                   if scores is not None else float('nan')),
                         'coverage': coverage, 'alt': alt,
                         'motion': motion, 'reloc': 0,
                         'gated': int(gated),
                         'innov': (round(self.innov, 2)
                                   if self.innov is not None else float('nan'))})
        # health cache (published at 1 Hz by publish_health)
        self.health.update(
            note='ok', k=int(k), t=float(t), vio=vio,
            coverage=round(float(coverage), 1), alt=round(float(alt), 1),
            x=round(float(est['x']), 1), y=round(float(est['y']), 1),
            yaw_deg=round(float(np.degrees(est['yaw'])), 1),
            spread=round(float(est['spread']), 1),
            neff=round(float(self.mcl.last_neff), 1),
            cov_gated_run=self.cov_gated_run,
            innov_m=(None if self.innov is None else round(self.innov, 2)))
        if not gated:
            self.health['score'] = round(float(scores.max()), 4)
        if k % 10 == 0:
            sc_s = (f'{scores.max():.4f}' if scores is not None else ' gated')
            self.get_logger().info(
                f"k={k:4d} t={t:8.1f}s x={est['x']:7.1f} y={est['y']:7.1f} "
                f"yaw={np.degrees(est['yaw']):7.1f} deg "
                f"spread={est['spread']:5.1f}m neff={self.mcl.last_neff:6.1f} "
                f"score={sc_s} "
                f"mov=({motion[0]:+5.1f},{motion[1]:+5.1f},"
                f"{np.degrees(motion[2]):+5.1f} deg)")

        # --- lost detection & re-localization -------------------------------
        # Anti-ratchet (see project memory): a rescan seed is only CONFIRMED
        # after RECOVER_FRAMES consecutive good frames, and last_good is NOT
        # re-anchored until then — a false lock can never hijack the anchor.
        # Escalation to the multi-frame steered global scan (kidnapped) has
        # TWO triggers: MAX_LOCAL_RESCANS consecutive failed locals, OR
        # MAX_WINDOW_RESCANS locals within RESCAN_WINDOW_S (frequency
        # evidence — a false lock CONFIRMS and resets the ladder but keeps
        # demanding rescans; a genuine reseed holds).
        # GATED frames carry no measurement information: the whole
        # lost/recovery state machine stays FROZEN (no streak counting, no
        # rescans — a descent through the coverage floor must not fire a
        # global scan on mushy scores).
        if not gated:
            if scores.max() >= LOST_SCORE:
                self.lost_count = 0
                if self.recovering:
                    self.recover_count += 1
                    if self.recover_count >= RECOVER_FRAMES:
                        self.recovering = False
                        self.recover_count = 0
                        self.reloc_fails = 0
                        self.global_pending_since = None
                        self.last_good = (est['x'], est['y'], est['yaw'], t, psi)
                        self.get_logger().info(
                            f'k={k}: recovery CONFIRMED at '
                            f"({est['x']:.0f},{est['y']:.0f}) m — tracking")
                else:
                    self.reloc_fails = 0
                    self.last_good = (est['x'], est['y'], est['yaw'], t, psi)
            else:
                self.lost_count += 1
                if self.recovering:
                    self.recover_count = 0   # strict consecutive streak
            if self.lost_count >= LOST_FRAMES:
                # uncertainty radius: V_MAX * time since the most recent anchor
                # (reseed or confirmed good fix), floored at 10 s of flight
                anchor_t = t - 10.0
                if self.reseed_t is not None:
                    anchor_t = max(anchor_t, self.reseed_t)
                if self.last_good is not None:
                    anchor_t = max(anchor_t, self.last_good[3])
                base = max(40.0, V_MAX * (t - anchor_t))
                # rescan-frequency evidence: false locks CONFIRM (resetting the
                # fails ladder) but keep needing fresh rescans; a genuine reseed
                # holds. 4 locals within RESCAN_WINDOW_S -> go global regardless
                # of the ladder (survives false confirms; cleared by the global).
                while (self.local_rescan_t
                       and self.local_rescan_t[0] < t - RESCAN_WINDOW_S):
                    self.local_rescan_t.popleft()
                if (self.reloc_fails < MAX_LOCAL_RESCANS
                        and len(self.local_rescan_t) < MAX_WINDOW_RESCANS):
                    radius = min(RELOC_RADIUS_MAX,
                                 base * RELOC_GROW ** self.reloc_fails)
                    step = max(12.0, radius / 8.0)
                    # steer with the compass when available; else the VIO yaw
                    # change since the last good fix; else a full circle
                    if yaw_pred is not None:
                        yaw0, span = yaw_pred, 2.0 * STEER_SPAN
                    elif self.last_good is not None:
                        yaw0 = self.last_good[2] + YAW_SIGN * wrap_angle(
                            psi - self.last_good[4])
                        span = np.radians(90.0)
                    else:
                        yaw0, span = None, 2.0 * np.pi
                    self.get_logger().warn(
                        f'k={k}: LOST -> local rescan +-{radius:.0f} m '
                        f'(fails={self.reloc_fails})')
                    peaks = local_yaw_scan(
                        self.mcl, patch, est['x'], est['y'],
                        radius_m=radius, step_m=step, yaw_steps=12, topk=5,
                        yaw0=yaw0, yaw_span=span)
                    std = (step, step, 0.15, 0.02)
                    self.reloc_fails += 1
                    self.local_rescan_t.append(t)
                    self.log[-1]['reloc'] = 1
                else:
                    # the multi-frame nomination needs the validated window: an
                    # 8 s window missed the truth on this very bag (kidnapped
                    # replay 2026-09-16), 25 s hit top-10 in 10/11 windows. If
                    # the usable buffer span is still short (early climb — the
                    # coverage gate only recently started passing), WAIT for it
                    # to fill instead of scanning thin; cap the wait so a
                    # buffer that can NEVER fill still gets a (fallback) scan.
                    usable = [f for f in self.scan_buf
                              if f[3] is not None and f[2] >= SCAN_MIN_COV
                              and self.yaw_pred_at(f[0]) is not None]
                    span_s = (usable[-1][0] - usable[0][0]
                              if len(usable) >= 2 else 0.0)
                    if self.global_pending_since is None:
                        self.global_pending_since = t
                    if (span_s < SCAN_WINDOW_S
                            and t - self.global_pending_since
                            < SCAN_WINDOW_S + 20.0):
                        if k % 50 == 0:
                            self.get_logger().warn(
                                f'k={k}: global scan pending — usable buffer '
                                f'{span_s:.0f} s / {SCAN_WINDOW_S:.0f} s')
                        self.health.update(note='global_wait')
                        self.lost_count = 0      # re-check on the next frame
                        return
                    self.global_pending_since = None
                    self.get_logger().warn(
                        f'k={k}: LOST -> steered GLOBAL scan '
                        f'(local fails={self.reloc_fails}, recent locals '
                        f'{len(self.local_rescan_t)}/{MAX_WINDOW_RESCANS} in '
                        f'{RESCAN_WINDOW_S:.0f} s, buffer {span_s:.0f} s)')
                    self.local_rescan_t.clear()
                    out = self.steered_global_scan()
                    # the steered scan left mcl at a buffered frame's
                    # coverage/patch — restore the current frame BEFORE the
                    # stage-2 refinement (it crops map patches at mcl.src_dim =
                    # the current coverage) and the seeding
                    self.mcl.set_coverage(coverage)
                    self.mcl.set_drone_patch(patch)
                    if out is not None:
                        regions, yaw_ref = out
                        # stage 2 — refine the top nominees with the CURRENT
                        # patch: the nomination grid is coarse (SCAN_STEP_M) and
                        # the multi-frame median peak can sit 100-150 m off the
                        # true match (kidnapped replay v5 2026-09-16: the offset
                        # truth seed lost the post-seed contest to a
                        # well-centered lookalike and false-locked to the end);
                        # a local scan re-centers each strong candidate on its
                        # sharp peak using the newest (most discriminative)
                        # patch, making the contest fair.
                        peaks = []
                        for s, x, y in regions[:3]:
                            pk = local_yaw_scan(
                                self.mcl, patch, x, y, radius_m=192.0,
                                step_m=32.0, yaw_steps=STEER_YAWS, topk=1,
                                yaw0=yaw_ref, yaw_span=2.0 * STEER_SPAN)
                            peaks.append(pk[0] if pk else (s, x, y, yaw_ref))
                        peaks += [(s, x, y, yaw_ref) for s, x, y in regions[3:]]
                        std = (48.0, 48.0, 0.15, 0.02)
                        # diagnostic: nominees -> refined + steering state
                        self.get_logger().warn(
                            'global nominees -> refined (score,x,y | '
                            'theta,yaw_ref): '
                            + '; '.join(
                                f'{s:.3f},{x:.0f},{y:.0f}->{rs:.3f},{rx:.0f},'
                                f'{ry:.0f}'
                                for (s, x, y), (rs, rx, ry, _r) in
                                zip(regions[:3], peaks[:3]))
                            + f' | {np.degrees(self.vio_map_rot):.1f} deg, '
                              f'{np.degrees(yaw_ref):.1f} deg')
                    else:
                        self.get_logger().warn(
                            'steering unavailable -> unsteered coarse scan')
                        peaks = coarse_scan(self.mcl, patch, topk=5)
                        std = (48.0, 48.0, 0.30, 0.03)
                    self.reloc_fails = 0
                    self.log[-1]['reloc'] = 2
                self.particles = particles_from_peaks(
                    self.mcl, peaks, self.n_particles, std=std)
                self.recovering = True
                self.recover_count = 0
                self.reseed_t = t
                self.lost_count = 0

    # ---------------------------------------------------------------- debug
    def save_debug_frame(self, k, raw, undist, ortho, mask, patch, scores):
        """Save the full per-frame pipeline: raw camera frame, undistorted,
        orthoprojection (invalid pixels blacked out), drone patch, and the map
        patch at this frame's best-scoring particle, plus a side-by-side
        combo for quick browsing."""
        b = os.path.join(self.out_dir, 'debug', f'k{int(k):05d}')
        cv2.imwrite(b + '_1_raw.png', raw)
        cv2.imwrite(b + '_2_undist.png', undist)
        o = ortho.copy()
        o[mask == 0] = 0
        cv2.imwrite(b + '_3_ortho.png', cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
        cv2.imwrite(b + '_4_patch.png', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        # map patch at the best-scoring particle (the filter's best match);
        # unavailable on coverage-gated frames (no scoring ran)
        sd = self.mcl.last_scored
        if sd is not None and scores is not None:
            i = int(np.argmax(scores))
            px, py = self.mcl.state_to_pixel(sd['x'][i], sd['y'][i])
            mp = cropRectangularSectionFromImage(
                self.map_rgb, px, py, -sd['yaw'][i],
                self.mcl.src_dim, PATCH_PX, sd['scale'][i])
            cv2.imwrite(b + '_5_mapatch.png',
                        cv2.cvtColor(mp, cv2.COLOR_RGB2BGR))
            combo = np.hstack([patch, mp])
            combo = cv2.resize(combo, (PATCH_PX * 6, PATCH_PX * 3),
                               interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(b + '_6_combo.png',
                        cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))

    # ---------------------------------------------------------------- health
    def publish_health(self):
        """1 Hz JSON health snapshot on /mcl/health (step 3).

        Raw facts + one state string; the step-5 guard interprets. States:
          no_frames   nothing processed yet
          init_wait   no particles yet (coverage/buffer/cap gates)
          motion_only last frame coverage-gated — predict only, no update
          recovering  a rescan hypothesis awaits CONFIRMATION
          lost        last measurement frame scored below LOST_SCORE
          tracking    last measurement frame held
        note = why the CURRENT/last frame deviated: ok | low_cov |
        scan_buffer | global_wait | no_geom | envelope | proc_fail.
        """
        h = dict(self.health)
        if h['k'] < 0:
            h['state'] = 'no_frames'
        elif self.particles is None:
            h['state'] = 'init_wait'
        elif self.last_gated:
            h['state'] = 'motion_only'
        elif self.recovering:
            h['state'] = 'recovering'
        elif self.lost_count > 0:
            h['state'] = 'lost'
        else:
            h['state'] = 'tracking'
        h['wall_s'] = round(time.time(), 3)
        # prune the rescan-evidence window against the last frame time
        while (self.local_rescan_t
               and self.local_rescan_t[0] < h['t'] - RESCAN_WINDOW_S):
            self.local_rescan_t.popleft()
        h['locals_recent'] = len(self.local_rescan_t)
        h['reloc_fails'] = self.reloc_fails
        h['recover_n'] = self.recover_count
        h['vio_jumps'] = self.vio_jumps
        self.health_pub.publish(String(data=json.dumps(h)))

    # ---------------------------------------------------------------- output
    def save_results(self):
        if not self.log:
            self.get_logger().warn('no frames processed, nothing to save')
            return
        np.savez(os.path.join(self.out_dir, 'replay_log.npz'),
                 **{key: np.array([r[key] for r in self.log])
                    for key in self.log[0]})
        self.get_logger().info(
            f"saved {self.out_dir}/replay_log.npz ({len(self.log)} frames)")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # GPS truth from the ULog beside the replayed bag (plotting only)
        gx = gy = None
        if self.bag_dir:
            try:
                truth = bag_gps_truth(
                    self.bag_dir, np.array([r['t'] for r in self.log]),
                    *self.geo_off)
                if truth is None:
                    self.get_logger().warn(
                        'GPS truth unavailable (no .ulg/.db3 beside the bag '
                        'or clock alignment failed) — plotting without truth')
                else:
                    gx, gy, off, corr = truth
                    self.get_logger().info(
                        f'GPS truth overlay: clock offset {off:+.3f} s '
                        f'(gyro/yaw-rate corr {corr:.3f})')
            except Exception as e:
                self.get_logger().warn(f'GPS truth overlay failed: {e}')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax = axes[0]
        tx = [self.cx_px + r['x'] / self.gsd_map for r in self.log]
        ty = [self.cy_px - r['y'] / self.gsd_map for r in self.log]
        mh, mw = self.map_rgb.shape[:2]
        pad_px = 150
        bx = tx + (list(gx) if gx is not None else [])
        by = ty + (list(gy) if gy is not None else [])
        x0 = int(max(0, min(bx) - pad_px)); x1 = int(min(mw, max(bx) + pad_px))
        y0 = int(max(0, min(by) - pad_px)); y1 = int(min(mh, max(by) + pad_px))
        if (x1 - x0) * (y1 - y0) > 0.25 * mw * mh:
            x0, y0, x1, y1 = 0, 0, mw, mh
        ax.imshow(self.map_rgb[y0:y1, x0:x1])
        if gx is not None:
            # dark under-stroke so the dashed truth line stays visible on
            # bright satellite imagery
            ax.plot([p - x0 for p in gx], [p - y0 for p in gy],
                    'k-', lw=3.2, alpha=0.75)
            ax.plot([p - x0 for p in gx], [p - y0 for p in gy],
                    'y--', lw=1.6, label='GPS truth')
        ax.plot([t - x0 for t in tx], [t - y0 for t in ty], 'c-', lw=1,
                label='estimate')
        ax.plot(tx[0] - x0, ty[0] - y0, 'g+', ms=15, mew=2, label='start')
        ax.plot(tx[-1] - x0, ty[-1] - y0, 'r+', ms=15, mew=2, label='end')
        if gx is not None:
            err_m = np.hypot(np.array(tx) - gx, np.array(ty) - gy) \
                * self.gsd_map
            ax.set_title('Estimated vs GPS-truth trajectory '
                         f'(err: med {np.median(err_m):.1f} m, '
                         f'p90 {np.percentile(err_m, 90):.1f} m)')
        else:
            ax.set_title('Estimated trajectory on map')
        ax.legend(loc='upper right')
        ax.axis('off')
        ax = axes[1]
        ax.plot([r['spread'] for r in self.log], label='spread (m)')
        ax.plot([r['neff'] for r in self.log], label='N_eff')
        ax.set_xlabel('frame'); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title('Spread / N_eff')
        ax = axes[2]
        ax.plot([r['score'] for r in self.log], label='max score')
        ax.set_xlabel('frame'); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title('Best-particle score')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'replay.png'), dpi=120)
        self.get_logger().info(f'saved {self.out_dir}/replay.png')


def main():
    rclpy.init()
    node = MclNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_results()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
