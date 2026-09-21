#!/usr/bin/env python3
"""Tests for the FCU-attitude tilt chain in ov_mcl/mcl_node.py.

Run (needs ROS2 for rclpy; torch/cv2 from venv_mcl via mcl_node itself):
  source /opt/ros/jazzy/setup.bash
  python3 test_fcu_tilt.py [bag_dir]     # default: bag 190020

  1. conventions  — gravity_to_tilt / fcu_frd_tilt signs and identities
  2. mounting rot — FCU_IMU_ROT_ROWMAJOR orthonormal (and == Wahba fit if
                    /tmp/R_fcu2imu.npy is still around)
  3. mavros path  — MAVROS ENU quaternion chain == ULog FRD chain for random
                    attitudes (both must describe the same physical tilt);
                    level hover with the real mounting -> small tilt
  4. live gates   — live_fcu_tilt interpolation / NaN / age (no ROS init:
                    instance via __new__, only the deques are set)
  5. bag replay   — bag_fcu_streams tilt vs the independent offline
                    validation that motivated the fix (fixed clock offset
                    10.248 s, bag 190020)
"""
import glob
import math
import os
import sys
from collections import deque

import numpy as np

SRC = '/home/one/workspace/ovws/src/ov_mcl'
GNSS = '/home/one/GNSS-denied-Localization'
for p in (SRC, GNSS):
    if p not in sys.path:
        sys.path.insert(0, p)

from ov_mcl.mcl_node import (  # noqa: E402
    FCU_IMU_ROT_ROWMAJOR, MclNode, bag_fcu_streams, fcu_frd_tilt,
    gravity_to_tilt, mavros_enu_tilt, quat_xyzw_to_R)
from bag_mcl import load_kalibr  # noqa: E402

DEFAULT_BAG = '/home/one/workspace/datasets/bag_0001_20260831_190020'
CALIB = ('/home/one/workspace/ovws/install/ov_msckf/share/ov_msckf/config/'
         'cyperstereo_012_752x480_equi/kalibr_imucam_chain.yaml')
RAD2DEG = 180.0 / math.pi


def R_to_quat_xyzw(R):
    """Rotation matrix -> Hamilton quaternion (x,y,z,w). Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
                (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
                (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
            (R[1, 0] - R[0, 1]) / s)


def test_conventions():
    # level nadir: gravity along camera z -> zero tilt
    r, p = gravity_to_tilt([0.0, 0.0, 1.0])
    assert abs(r) < 1e-12 and abs(p) < 1e-12
    # gravity in camera +y -> positive roll; camera +x -> negative pitch
    r, p = gravity_to_tilt([0.0, math.sin(0.2), math.cos(0.2)])
    assert abs(r - 0.2) < 1e-12 and abs(p) < 1e-12
    r, p = gravity_to_tilt([-math.sin(0.15), 0.0, math.cos(0.15)])
    assert abs(r) < 1e-12 and abs(p - 0.15) < 1e-12
    # level gravity in the FCU body (FRD, z down) -> zero tilt (identity rot)
    r, p = fcu_frd_tilt([0.0, 0.0, 1.0], np.eye(3))
    assert abs(r) < 1e-12 and abs(p) < 1e-12
    print('PASS conventions')


def test_mounting_rotation():
    R = np.asarray(FCU_IMU_ROT_ROWMAJOR).reshape(3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), 'not orthonormal'
    assert abs(np.linalg.det(R) - 1.0) < 1e-7, 'not a proper rotation'
    fit = '/tmp/R_fcu2imu.npy'
    if os.path.isfile(fit):
        Rf = np.load(fit)
        err = np.degrees(np.arccos(np.clip((np.trace(R.T @ Rf) - 1) / 2,
                                           -1.0, 1.0)))
        assert err < 0.01, f'constant drifted from the Wahba fit ({err} deg)'
    print('PASS mounting rotation (orthonormal'
          + (', == /tmp/R_fcu2imu.npy)' if os.path.isfile(fit) else ')'))


def test_mavros_path(R_fcu2cam):
    # level hover: identity ENU orientation -> only the small (~3 deg)
    # mounting cross-axis offsets show up. A convention error anywhere in
    # the chain gives ~90/180 deg phantom tilt.
    r, p = mavros_enu_tilt(0.0, 0.0, 0.0, 1.0, R_fcu2cam)
    assert abs(r) < np.radians(5.0) and abs(p) < np.radians(5.0), \
        f'level hover tilt {RAD2DEG * r:.1f}/{RAD2DEG * p:.1f} deg'
    # PX4 attitude q rotates body FRD -> local NED world; MAVROS orientation
    # rotates base_link FLU -> ENU. Same physical attitude, related by the
    # fixed frame flips D = diag(1,-1,-1):  R_enu_flu = D @ R_wb @ D.
    # The two tilt paths must agree for ANY attitude.
    D = np.diag([1.0, -1.0, -1.0])
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(500):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q                     # pretend PX4 w-first [w,x,y,z]
        R_wb = quat_xyzw_to_R(x, y, z, w)  # FRD -> world
        tilt_ulog = fcu_frd_tilt(R_wb.T @ np.array([0.0, 0.0, 1.0]),
                                 R_fcu2cam)
        tilt_mavros = mavros_enu_tilt(*R_to_quat_xyzw(D @ R_wb @ D),
                                      R_fcu2cam)
        worst = max(worst, abs(tilt_mavros[0] - tilt_ulog[0]),
                    abs(tilt_mavros[1] - tilt_ulog[1]))
    assert worst < 1e-9, f'paths disagree by {worst} rad'
    print(f'PASS mavros path (level {RAD2DEG * r:.2f}/{RAD2DEG * p:.2f} deg, '
          f'random-attitude agreement {worst:.1e} rad)')


def test_live_gates():
    node = MclNode.__new__(MclNode)        # skip Node/__init__ (no ROS)
    node.fcu_att_seen = True
    node.fcu_alt_t = deque([10.0, 10.1, 10.2, 10.3])
    node.fcu_roll = deque([0.01, 0.02, 0.03, 0.04])
    node.fcu_pitch = deque([-0.01, -0.02, -0.03, -0.04])
    r, p = node.live_fcu_tilt(10.15)       # linear interpolation
    assert abs(r - 0.025) < 1e-12 and abs(p + 0.025) < 1e-12
    assert node.live_fcu_tilt(10.3) == (0.04, -0.04)   # exact edge hit
    assert node.live_fcu_tilt(10.0) == (0.01, -0.01)
    assert node.live_fcu_tilt(10.9) is None            # age gate (0.5 s)
    assert node.live_fcu_tilt(9.4) is None
    node.fcu_roll = deque([0.01, float('nan'), 0.03, 0.04])
    node.fcu_pitch = deque([-0.01, -0.02, -0.03, -0.04])
    assert node.live_fcu_tilt(10.15) is None           # NaN bracket
    assert node.live_fcu_tilt(10.3) == (0.04, -0.04)   # finite bracket ok
    node.fcu_att_seen = False
    assert node.live_fcu_tilt(10.15) is None           # attitude never seen
    print('PASS live gates (interp / NaN / age)')


def test_bag_streams(bag_dir, R_fcu2cam):
    from pyulog import ULog
    alt_at, tilt_at, _ = bag_fcu_streams(bag_dir, R_fcu2cam)
    assert alt_at is not None and tilt_at is not None, \
        'ULog streams unavailable (no .ulg / weak correlation?)'

    # independent reference: the offline-validation math (2026-09-16) with
    # the fitted clock offset 10.248 s
    ulg = sorted(glob.glob(os.path.join(bag_dir, '*.ulg')))[0]
    t_att = q = None
    for d in ULog(ulg).data_list:
        if d.name == 'vehicle_attitude':
            t_att = d.data['timestamp'] / 1e6
            q = np.stack([d.data[f'q[{i}]'] for i in range(4)], 1)
            break
    assert t_att is not None
    x, y, z, w = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], 1).T
    R_wb = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], -1)], -2)
    g_frd = np.einsum('nji,j->ni', R_wb, np.array([0.0, 0.0, 1.0]))
    OFF = 10.248

    idx = np.linspace(0, len(t_att) - 1, 400).astype(int)
    dr, dp, alts = [], [], []
    for i in idx:
        r_ref, p_ref = gravity_to_tilt(R_fcu2cam @ g_frd[i])
        r, p = tilt_at(t_att[i] - OFF)
        dr.append(abs(r - r_ref))
        dp.append(abs(p - p_ref))
        alts.append(alt_at(t_att[i] - OFF))
    dr, dp = np.degrees(np.array(dr)), np.degrees(np.array(dp))
    both = np.maximum(dr, dp)
    assert np.median(both) < 0.3, \
        f'tilt med error {np.median(both):.2f} deg vs reference'
    assert np.percentile(both, 95) < 2.5, \
        f'tilt p95 error {np.percentile(both, 95):.2f} deg vs reference'

    # whole-flight sanity: sensible tilt trace, sane baro heights
    rr, pp = [], []
    for i in idx:
        r, p = tilt_at(t_att[i] - OFF)
        rr.append(r)
        pp.append(p)
    rr, pp = np.degrees(rr), np.degrees(pp)
    assert np.median(np.abs(rr)) < 15.0 and np.median(np.abs(pp)) < 15.0
    assert np.percentile(np.abs(rr), 95) < 35.0 \
        and np.percentile(np.abs(pp), 95) < 35.0
    alts = np.array(alts)
    assert np.all(np.isfinite(alts)) and 0.0 <= alts.min() <= alts.max() <= 250.0
    print(f'PASS bag streams (tilt vs ref med {np.median(both):.3f} deg '
          f'p95 {np.percentile(both, 95):.2f} deg | |roll| med '
          f'{np.median(np.abs(rr)):.1f} p95 {np.percentile(np.abs(rr), 95):.1f} '
          f'deg | alt {alts.min():.0f}..{alts.max():.0f} m)')


def main():
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BAG
    _, _, R_CI, _ = load_kalibr(CALIB)
    R_fcu2cam = R_CI @ np.asarray(FCU_IMU_ROT_ROWMAJOR).reshape(3, 3)

    test_conventions()
    test_mounting_rotation()
    test_mavros_path(R_fcu2cam)
    test_live_gates()
    if os.path.isdir(bag_dir):
        test_bag_streams(bag_dir, R_fcu2cam)
    else:
        print(f'SKIP bag streams ({bag_dir} not found)')
    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
