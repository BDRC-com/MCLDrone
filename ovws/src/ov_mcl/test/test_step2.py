#!/usr/bin/env python3
"""Tests for the step-2 kidnapped-scan infrastructure in ov_mcl/mcl_node.py.

Run (needs ROS2 for rclpy; torch/cv2 from venv_mcl via mcl_node itself):
  source /opt/ros/jazzy/setup.bash && source ~/workspace/ovws/venv_mcl/bin/activate
  python3 test_step2.py [bag_dir]     # default: bag 190020

  1. heading identity — enu_quat_to_ned_heading(MAVROS ENU quaternion) ==
     PX4 NED heading for random attitudes AND on the real ULog stream.
     NOTE the MAVROS rotation is R_enu_flu = P @ R_wb @ D — for tilt the
     left P is invisible (gravity only), for heading it is essential.
  2. vio_to_map_delta — rotation algebra (identity, 90 deg, composition,
     norm preservation).
  3. bilinear_grid_sample — constants, linear ramps, edge clamping.
  4. parse_yaw_cal — parsing + default fallback.
  5. live heading gates — live_fcu_hdg interp / NaN / age (no ROS init:
     instance via __new__, only the deques are set).
  6. compass fit — yaw_pred = -1*hdg + 1.3 deg vs the VERIFIED s2 replay
     yaw (the fit the deployment uses): residual p95 must sit well inside
     the +-STEER_SPAN steering window, and a=+1 must be rejected.
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
    FCU_IMU_ROT_ROWMAJOR, STEER_SPAN, MclNode, bag_fcu_streams,
    bilinear_grid_sample, enu_quat_to_ned_heading, parse_yaw_cal,
    vio_to_map_delta, wrap_angle)
from bag_mcl import load_kalibr  # noqa: E402

DEFAULT_BAG = '/home/one/workspace/datasets/bag_0001_20260831_190020'
S2_REPLAY = (f'{GNSS}/sample_images/mcl_test/ftmt1_190020_s2/'
             'replay_log.npz')
CALIB = ('/home/one/workspace/ovws/install/ov_msckf/share/ov_msckf/config/'
         'cyperstereo_012_752x480_equi/kalibr_imucam_chain.yaml')
RAD2DEG = 180.0 / math.pi
# MAVROS frame conversions: NED->ENU world (P), FRD->FLU body (D)
P = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
D = np.diag([1.0, -1.0, -1.0])


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


def rand_R(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.stack([
        np.array([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)]),
        np.array([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)]),
        np.array([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)])])


def test_heading_identity():
    # random attitudes: MAVROS quaternion (FLU->ENU) is P @ R_wb @ D; its
    # first column is the body front in ENU = (E, N, -U) -> atan2(E, N)
    # must equal the PX4 NED heading of R_wb exactly (any tilt).
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(1000):
        R_wb = rand_R(rng)
        px4_yaw = math.atan2(R_wb[1, 0], R_wb[0, 0])
        got = enu_quat_to_ned_heading(*R_to_quat_xyzw(P @ R_wb @ D))
        worst = max(worst, abs(wrap_angle(got - px4_yaw)))
    assert worst < 1e-9, f'random-attitude heading mismatch {worst} rad'
    # the naive D @ R_wb @ D (correct for GRAVITY/tilt) must FAIL here —
    # guards against regressing to the tilt test's convention
    rng = np.random.default_rng(11)
    bad = 0.0
    for _ in range(1000):
        R_wb = rand_R(rng)
        px4_yaw = math.atan2(R_wb[1, 0], R_wb[0, 0])
        got = enu_quat_to_ned_heading(*R_to_quat_xyzw(D @ R_wb @ D))
        bad = max(bad, abs(wrap_angle(got - px4_yaw)))
    assert bad > 0.5, 'D @ R_wb @ D unexpectedly gives the heading too?!'
    print(f'PASS heading identity (random attitudes {worst:.1e} rad; '
          f'P-less variant rejected, err {bad:.2f} rad)')


def test_vio_to_map_delta():
    assert vio_to_map_delta((3.0, -4.0), 0.0) == (3.0, -4.0)
    # CCW rotation by 90 deg: (1,0) -> (0,1), (0,1) -> (-1,0)
    assert np.allclose(vio_to_map_delta((1.0, 0.0), math.pi / 2), (0.0, 1.0))
    assert np.allclose(vio_to_map_delta((0.0, 1.0), math.pi / 2), (-1.0, 0.0))
    # composition: R(b) @ R(a) == R(a+b)
    rng = np.random.default_rng(3)
    for _ in range(200):
        d = rng.normal(size=2)
        a, b = rng.uniform(-np.pi, np.pi, 2)
        one = vio_to_map_delta(vio_to_map_delta(d, a), b)
        two = vio_to_map_delta(d, wrap_angle(a + b))
        assert np.allclose(one, two, atol=1e-12)
        assert abs(np.hypot(*one) - np.hypot(*d)) < 1e-12  # norm kept
    print('PASS vio_to_map_delta (identity / 90 deg / composition / norm)')


def test_bilinear():
    G = np.arange(20, dtype=float).reshape(4, 5)
    # constant grid -> constant
    C = np.full((4, 5), 7.5)
    cols = np.array([0.0, 1.3, 3.5, 2.75])
    rows = np.array([2.1, 0.0, 1.9, 2.2])
    assert np.allclose(bilinear_grid_sample(C, cols, rows), 7.5)
    # linear ramp 3c + 5r is reproduced EXACTLY by bilinear interpolation
    # (coords strictly inside the grid; at/outside the last index the edge
    # clamp degrades to the edge value by design)
    lin = 3.0 * np.arange(5)[None, :] + 5.0 * np.arange(4)[:, None]
    got = bilinear_grid_sample(lin, cols, rows)
    assert np.allclose(got, 3.0 * cols + 5.0 * rows), got
    # out-of-bounds clamps to the edge samples
    assert bilinear_grid_sample(G, np.array(-3.0), np.array(7.0)) == G[3, 0]
    assert bilinear_grid_sample(G, np.array(9.0), np.array(-1.0)) == G[0, 4]
    # 2x2 hand check: center = mean of 4
    H = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert abs(bilinear_grid_sample(H, 0.5, 0.5) - 2.5) < 1e-12
    # grid edge (fractional part at the last index still exact)
    assert bilinear_grid_sample(G, np.array(4.0), np.array(3.0)) == G[3, 4]
    print('PASS bilinear_grid_sample (constant / linear / clamp / hand)')


def test_parse_yaw_cal():
    assert parse_yaw_cal('-1,1.3') == (-1.0, 1.3)
    assert parse_yaw_cal(' 0.5 , 2.5 ') == (0.5, 2.5)
    assert parse_yaw_cal('garbage') == parse_yaw_cal('-1,1.3')  # default
    assert parse_yaw_cal('1') == parse_yaw_cal('-1,1.3')        # wrong count
    assert parse_yaw_cal('-1,1.3,7') == parse_yaw_cal('-1,1.3')
    assert parse_yaw_cal(None) == parse_yaw_cal('-1,1.3')
    assert parse_yaw_cal('-1,1.3') == (-1.0, 1.3)               # idempotent
    print('PASS parse_yaw_cal (parse + fallbacks)')


def test_live_hdg_gates():
    node = MclNode.__new__(MclNode)        # skip Node/__init__ (no ROS)
    node.fcu_att_seen = True
    node.fcu_alt_t = deque([10.0, 10.1, 10.2, 10.3])
    node.fcu_hdg_vals = deque([6.2, 6.3, 6.4, 6.5])   # unwrapped past 2*pi
    assert abs(node.live_fcu_hdg(10.15) - 6.35) < 1e-12   # interp, no wrap
    assert node.live_fcu_hdg(10.3) == 6.5                 # exact edge hit
    assert node.live_fcu_hdg(10.0) == 6.2
    assert node.live_fcu_hdg(10.9) is None                # age gate (0.5 s)
    assert node.live_fcu_hdg(9.4) is None
    node.fcu_hdg_vals = deque([6.2, float('nan'), 6.4, 6.5])
    assert node.live_fcu_hdg(10.15) is None               # NaN bracket
    assert node.live_fcu_hdg(10.3) == 6.5                 # finite bracket ok
    node.fcu_att_seen = False
    assert node.live_fcu_hdg(10.15) is None               # attitude never seen
    print('PASS live heading gates (interp / unwrapped / NaN / age)')


def test_heading_vs_ulog(bag_dir):
    """Real-stream check: MAVROS-form quaternion of the ULog attitude ->
    enu_quat_to_ned_heading == PX4's own yaw, on the actual flight data."""
    from pyulog import ULog
    ulg = sorted(glob.glob(os.path.join(bag_dir, '*.ulg')))[0]
    t = q = None
    for d in ULog(ulg).data_list:
        if d.name == 'vehicle_attitude':
            t = d.data['timestamp'] / 1e6
            q = np.stack([d.data[f'q[{i}]'] for i in range(4)], 1)
            break
    assert t is not None, 'no vehicle_attitude in the ULog'
    idx = np.linspace(0, len(t) - 1, 500).astype(int)
    worst = 0.0
    for i in idx:
        w, x, y, z = q[i]                     # PX4 w-first [w,x,y,z]
        R_wb = quat_to_R_wb(x, y, z, w)
        px4_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        got = enu_quat_to_ned_heading(*R_to_quat_xyzw(P @ R_wb @ D))
        worst = max(worst, abs(wrap_angle(got - px4_yaw)))
    assert worst < 1e-5, f'ULog heading mismatch {worst} rad'
    print(f'PASS heading vs ULog (500 real attitudes, max err {worst:.1e} '
          f'rad — float32 ULog precision)')


def quat_to_R_wb(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)]])


def test_compass_fit(bag_dir, R_fcu2cam):
    """The deployment calibration yaw = a*hdg + radians(b), a=-1, b=1.3:
    residual vs the VERIFIED s2 replay yaw must match the offline fit
    (std 4.3 deg, p95 6.5 deg) and sit well inside the steering window."""
    if not os.path.isfile(S2_REPLAY):
        print(f'SKIP compass fit ({S2_REPLAY} not found)')
        return
    d = np.load(S2_REPLAY)
    t_s2, yaw_s2 = d['t'], d['yaw']
    alt_at, tilt_at, hdg_at = bag_fcu_streams(bag_dir, R_fcu2cam)
    assert hdg_at is not None, 'ULog heading stream unavailable'
    hdg = np.array([hdg_at(tt) for tt in t_s2])
    ok = np.isfinite(hdg)
    assert ok.mean() > 0.9, f'heading coverage only {ok.mean():.0%}'

    a, b = parse_yaw_cal('-1,1.3')
    resid = np.array([wrap_angle(a * h + math.radians(b) - yy)
                      for h, yy in zip(hdg[ok], yaw_s2[ok])])
    med, p95 = np.median(np.abs(resid)), np.percentile(np.abs(resid), 95)
    assert med < math.radians(3.0), f'residual median {RAD2DEG * med:.2f} deg'
    # offline fit reported p95 6.5 deg with the LS-fitted b and a fixed
    # clock offset; the deployed constants (rounded b, correlation offset)
    # run slightly looser — 10 deg still leaves >2x margin to the window
    assert p95 < math.radians(10.0), f'residual p95 {RAD2DEG * p95:.2f} deg'
    # the steering window must cover the residual with margin
    assert p95 < STEER_SPAN, \
        f'p95 {RAD2DEG * p95:.1f} deg not inside +-STEER_SPAN ' \
        f'{RAD2DEG * STEER_SPAN:.0f} deg'
    # sign check: a=+1 must be far worse (rms 89 deg vs 5 deg on the fit
    # flight; the median alone is a weak statistic because the wrong-sign
    # error ~ -2*yaw is small whenever the drone happens to fly near the
    # degenerate heading)
    wrong = np.array([wrap_angle(h + math.radians(b) - yy)
                      for h, yy in zip(hdg[ok], yaw_s2[ok])])
    rms_r = np.sqrt((resid ** 2).mean())
    rms_w = np.sqrt((wrong ** 2).mean())
    assert rms_w > 10.0 * rms_r and rms_w > math.radians(45.0), \
        f'a=+1 not rejected: rms {RAD2DEG * rms_w:.1f} vs ' \
        f'{RAD2DEG * rms_r:.1f} deg'
    print(f'PASS compass fit (n={ok.sum()}, |resid| med '
          f'{RAD2DEG * med:.2f} deg p95 {RAD2DEG * p95:.2f} deg rms '
          f'{RAD2DEG * rms_r:.2f} deg, window +-{RAD2DEG * STEER_SPAN:.0f} '
          f'deg; a=+1 rms {RAD2DEG * rms_w:.0f} deg rejected)')


def main():
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BAG
    _, _, R_CI, _ = load_kalibr(CALIB)
    R_fcu2cam = R_CI @ np.asarray(FCU_IMU_ROT_ROWMAJOR).reshape(3, 3)

    test_heading_identity()
    test_vio_to_map_delta()
    test_bilinear()
    test_parse_yaw_cal()
    test_live_hdg_gates()
    if os.path.isdir(bag_dir):
        test_heading_vs_ulog(bag_dir)
        test_compass_fit(bag_dir, R_fcu2cam)
    else:
        print(f'SKIP bag tests ({bag_dir} not found)')
    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
