#!/usr/bin/env python3
"""Plot the DELIVERED MCL odometry (/mcl/odom) against GPS truth.

Three trajectories on one map-frame axes (east/north, metres):
  - black  : GPS truth from the ULog recorded next to the original flight bag
  - blue   : fused odometry /mcl/odom extracted from the replay recording
             (gaps = moments the node published a NaN pose, e.g. gated
             frames or an anchor older than EV_POSE_PROP_S; the EV velocity
             is still valid there)
  - red  o : raw MCL fixes (tracked frames only, from replay_log.npz)

Inputs:
  RUN_DIR        mcl_runs/<name>/ directory containing replay_log.npz
  --bag DIR      replay recording dir (rosbag2, contains *.db3) with the
                 /mcl/odom stream; default: newest mcl_replay_* dir next to
                 RUN_DIR (override when that guess is wrong)
  --fly DIR      ORIGINAL flight bag dir with the *.ulg (GPS truth) and its
                 *.db3 (/imu0 for clock alignment); default: $MCL_FLY_BAG.
                 Without it the plot is drawn odometry+fixes only.

Examples (Jetson, after: source ~/ovws/install/setup.bash):
  python3 plot_odom_vs_gps.py \
      ~/MCLDrone/test/mcl_runs/replay_184004_point_fixed2 \
      --bag ~/MCLDrone/test/mcl_replay_184004_log_fixed2 \
      --fly ~/datasets/bag_0001_20260831_184004

Dev (no install needed, src tree is used directly):
  source /opt/ros/jazzy/setup.bash
  python3 plot_odom_vs_gps.py \
      test/mcl_runs/replay_184004_point_fixed2 \
      --bag test/mcl_replay_184004_log_fixed2 \
      --fly /home/one/workspace/datasets/bag_0001_20260831_184004
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # bag_reader

# map frame: local map px = global zoom-17 mercator px + offset (launch args)
OFF_X, OFF_Y = -27449088.0, -14586624.0
GSD, C = 1.1, 2560.0     # m/px and half-width of the cropped map (z17_5120)


def import_bag_gps_truth(script_dir):
    """bag_gps_truth lives in ov_mcl.mcl_node; find the package in the
    sourced colcon install, the drone's ~/ovws install, or the src tree."""
    cands = []
    home = os.path.expanduser('~')
    cands += sorted(glob.glob(os.path.join(
        home, 'ovws', 'install', 'ov_mcl', 'lib', 'python*',
        'site-packages')), reverse=True)
    # src layout: the importable package dir is ovws/src/ov_mcl (it contains
    # the ov_mcl/ subpackage), one level deeper than for an install tree
    cands.append(os.path.join(script_dir, 'ovws', 'src', 'ov_mcl'))
    cands.append('/home/one/workspace/ovws/src/ov_mcl')     # dev machine
    for c in cands:
        if os.path.isdir(c):
            sys.path.insert(0, c)
    from ov_mcl.mcl_node import bag_gps_truth
    return bag_gps_truth


def read_ev_bag(bag_dir, ev_topic):
    try:
        import rosbag2_py
    except ImportError:
        raise SystemExit(
            "rosbag2_py not found for %s. ROS Jazzy bindings are built for "
            "the SYSTEM python3.12 — a conda/venv interpreter (e.g. sivl, "
            "3.11) cannot load them. Run:\n"
            "  source /opt/ros/jazzy/setup.bash && /usr/bin/python3 %s ...\n"
            "(or 'conda deactivate' first)."
            % (sys.executable, os.path.abspath(__file__)))
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    # Jazzy ros2 bag record defaults to mcap; older recordings are sqlite3
    sid = 'mcap' if glob.glob(os.path.join(bag_dir, '*.mcap')) \
        else 'sqlite3'
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=sid),
           rosbag2_py.ConverterOptions('', ''))
    cls = {t.name: get_message(t.type)
           for t in r.get_all_topics_and_types()}
    if ev_topic not in cls:
        raise SystemExit("topic %s not in %s (have: %s)"
                         % (ev_topic, bag_dir, ', '.join(sorted(cls))))
    et, ex, ey = [], [], []
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic != ev_topic:
            continue
        m = deserialize_message(data, cls[topic])
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        ok = np.isfinite(m.pose.pose.position.x)
        et.append(t)
        ex.append(m.pose.pose.position.x if ok else np.nan)
        ey.append(m.pose.pose.position.y if ok else np.nan)
    return np.array(et), np.array(ex), np.array(ey)


def load_fixes(run_dir):
    p = os.path.join(run_dir, 'replay_log.npz')
    d = np.load(p, allow_pickle=True)
    trk = np.asarray(d['gated']).astype(int) == 0
    return (np.asarray(d['t'], float)[trk],
            np.asarray(d['x'], float)[trk],
            np.asarray(d['y'], float)[trk])


def guess_bag_dir(run_dir):
    root = os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))
    cands = [d for d in glob.glob(os.path.join(root, 'mcl_replay_*'))
             if os.path.isdir(d)
             and (glob.glob(os.path.join(d, '*.db3'))
                  or glob.glob(os.path.join(d, '*.mcap')))]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', help='dir containing replay_log.npz')
    ap.add_argument('--bag', help='replay rosbag2 dir with /mcl/odom '
                    '(default: newest mcl_replay_* next to run_dir)')
    ap.add_argument('--fly', default=os.environ.get('MCL_FLY_BAG', ''),
                    help='original flight bag dir with *.ulg (GPS truth)')
    ap.add_argument('--ev-topic', default='/mcl/odom')
    ap.add_argument('--out', help='output png (default: <run_dir>/'
                    'fused_vs_gps.png)')
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    bag_dir = os.path.abspath(args.bag) if args.bag else guess_bag_dir(run_dir)
    if not bag_dir:
        raise SystemExit('no --bag given and no mcl_replay_* with *.db3 '
                         'found next to %s' % run_dir)
    out = args.out or os.path.join(run_dir, 'fused_vs_gps.png')
    print('run dir : %s' % run_dir)
    print('ev bag  : %s' % bag_dir)

    ev_t, ev_x, ev_y = read_ev_bag(bag_dir, args.ev_topic)
    fin = np.isfinite(ev_x)
    npz_p = os.path.join(run_dir, 'replay_log.npz')
    if os.path.exists(npz_p):
        ct, xt, yt = load_fixes(run_dir)
    else:
        print('WARNING: %s missing (killed before flush?) — plotting EV + '
              'GPS truth without MCL fix markers' % npz_p)
        ct, xt, yt = None, np.array([]), np.array([])
    print('/mcl/odom: %d msgs, %d finite poses (%.1f%% pose duty)'
          % (len(ev_t), fin.sum(), 100.0 * fin.mean()))

    title = None
    if args.fly:
        bag_gps_truth = import_bag_gps_truth(
            os.path.dirname(os.path.abspath(__file__)))
        tr = bag_gps_truth(args.fly, ev_t[fin], OFF_X, OFF_Y)
        if tr is None:
            print('WARNING: GPS truth unavailable for %s (no .ulg, no /imu0 '
                  'in .db3/.mcap, or corr < 0.3) - plotting without truth'
                  % args.fly)
            gps_x = gps_y = None
        else:
            gpx, gpy, off, corr = tr
            gx = (np.asarray(gpx, float) - C) * GSD
            gy = (C - np.asarray(gpy, float)) * GSD
            seg_t = ev_t[fin]
            # continuous truth: interpolate across sub-second pose gaps,
            # break only at >3 s gaps between tracked fixes
            gps_x = np.interp(ev_t, seg_t, gx)
            gps_y = np.interp(ev_t, seg_t, gy)
            gps_x[(ev_t < seg_t[0]) | (ev_t > seg_t[-1])] = np.nan
            gps_y[(ev_t < seg_t[0]) | (ev_t > seg_t[-1])] = np.nan
            if ct is not None:
                for k in np.where(np.diff(ct) > 3.0)[0]:
                    m = (ev_t > ct[k]) & (ev_t < ct[k + 1])
                    gps_x[m] = gps_y[m] = np.nan
            err = np.hypot(ev_x[fin] - gx, ev_y[fin] - gy)
            print('GPS truth: clock offset %+.3f s, corr %.3f' % (off, corr))
            print('fused err vs GPS: median %.2f  p90 %.2f  max %.2f m'
                  % (np.median(err), np.percentile(err, 90), err.max()))
            title = ('fused odometry vs GPS truth\n'
                     'median %.1f m   p90 %.1f m   max %.1f m   '
                     '(clock %+.2f s, corr %.2f)'
                     % (np.median(err), np.percentile(err, 90), err.max(),
                        off, corr))
    else:
        print('no --fly / $MCL_FLY_BAG: plotting odometry + fixes only')
        gps_x = gps_y = None

    fig, ax = plt.subplots(figsize=(9.5, 9))
    if gps_x is not None:
        ax.plot(gps_x, gps_y, '-', color='black', lw=2.0, zorder=1,
                label='GPS truth (ULog)')
    ax.plot(ev_x, ev_y, '-', color='#0057E7', lw=1.5, zorder=2,
            label='fused odometry %s (gaps = pose invalid)' % args.ev_topic)
    if xt.size:
        ax.plot(xt, yt, 'o', color='#E63946', ms=4, mfc='none', mew=0.9,
                zorder=3, label='MCL fixes')
    ax.set_xlabel('east [m]')
    ax.set_ylabel('north [m]')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_title(title or 'fused odometry and MCL fixes (no GPS truth)')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print('saved %s' % out)


if __name__ == '__main__':
    main()
