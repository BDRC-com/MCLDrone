"""Bag replay pipeline: rosbag2 -> orthoprojection -> MCL localization.

Streams camera frames + IMU from a rosbag2 .db3, computes tilt from the
accelerometer (low-pass gravity direction), orthoprojects each frame to a
96 m top-down patch, estimates inter-frame motion (phase correlation for
translation, gyro z for yaw), and runs the MCL particle filter with
multi-frame temporal weight accumulation.

Initialization (--init):
  scan   (default) coarse grid scan of the whole map (8 yaws) on frame 0,
         particles start as a mixture over the top peaks.
  click  interactive: you click the drone's initial position on the map
         (two-stage: rough on the full map, precise on a zoomed crop; the
         drone's frame-0 patch is shown alongside for matching). A local
         yaw scan (±48 m, 12 yaws) then resolves the unknown initial yaw.
         Needs a display — run outside the sandbox.

Usage:
    ~/anaconda3/envs/sivl/bin/python bag_mcl.py <bag.db3> \
        [--stride 5] [--alt 100] [--frames N] [--init click] [--out DIR]
"""

import argparse
import os
import sys
import bisect
import numpy as np
import cv2
import torch
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bag_reader import BagReader, gravity_tilt                     # noqa: E402
from mcl import (MCL, load_rgb, make_drone_patch, MAP_PATH, MAP_GSD,  # noqa: E402
                 PATCH_PX, COVERAGE_M, CHECKPOINT, OUTPUT_DIR, set_seed)
from orthoprojection import orthoproject, extract_patch           # noqa: E402
from utils.utils import cropRectangularSectionFromImage            # noqa: E402

# --- Camera model: Kalibr calibration (configs/kalibr_imucam_chain.yaml) ---
KALIB_IMUCAM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'configs', 'kalibr_imucam_chain.yaml')


def load_kalibr(path=KALIB_IMUCAM, cam='cam0'):
    """Load camera intrinsics, distortion, and IMU->camera rotation from a
    Kalibr imucam chain yaml (strips the %YAML:1.0 header if present).

    Returns (K 3x3, dist (4,), R_imu2cam 3x3, model str) where model is
    'radtan' or 'equidistant' (fisheye)."""
    txt = open(path).read()
    if txt.startswith('%'):
        txt = txt.split('\n', 1)[1]
    d = yaml.safe_load(txt)[cam]
    fx, fy, cx, cy = d['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(d['distortion_coeffs'], dtype=np.float64)
    R_imu2cam = np.array(d['T_cam_imu'], dtype=np.float64)[:3, :3]
    model = str(d.get('distortion_model', 'radtan')).lower()
    return K, dist, R_imu2cam, model


def valid_coverage_m(mask, extent, resolution=1.0, max_m=COVERAGE_M,
                     floor_m=24.0):
    """Largest square [m], centered on the nadir, fully inside the valid
    footprint of the orthoprojection. Falls back to max_m if the nadir is
    invalid (degenerate frame)."""
    x_min, _, y_min, _ = extent
    h, w = mask.shape
    nx = int(round(-x_min / resolution))
    ny = int(round(-y_min / resolution))
    if not (0 <= nx < w and 0 <= ny < h) or mask[ny, nx] == 0:
        return max_m

    def run(arr, i):
        lo = hi = i
        while lo > 0 and arr[lo - 1]:
            lo -= 1
        while hi < len(arr) - 1 and arr[hi + 1]:
            hi += 1
        return hi - lo + 1

    col = mask[:, nx] > 127
    row = mask[ny, :] > 127
    side_px = min(run(col, ny), run(row, nx))
    cov = float(np.clip(side_px * resolution, floor_m, max_m))
    return float(np.round(cov / 4.0) * 4.0)   # quantize: stable across frames


class BagStream:
    """Lazy camera frames + preloaded IMU (tilt + gyro) from a rosbag.

    IMU samples are rotated into the camera optical frame with R_imu2cam
    (from Kalibr T_cam_imu) at load time, so gravity_tilt() sees the
    camera-frame gravity direction and gyro-z integrates camera yaw."""

    def __init__(self, db3_path, stride=1, max_frames=None, R_imu2cam=None):
        self.reader = BagReader(db3_path)
        self.db3_path = db3_path
        self.stride = stride
        self.max_frames = max_frames
        self._frame_iter = None

        t, w, a = [], [], []
        for ti, wi, ai in self.reader.imu_samples():
            t.append(ti); w.append(wi); a.append(ai)
        self.imu_t = np.array(t)
        gyro = np.array(w)
        accel = np.array(a)
        if R_imu2cam is not None:
            # v_cam = R * v_imu  (Kalibr T_cam_imu maps IMU frame -> cam frame)
            gyro = gyro @ R_imu2cam.T
            accel = accel @ R_imu2cam.T
        self.gyro = gyro
        self.accel = accel

        # Low-pass filtered gravity tilt over the whole IMU stream.
        state = [0.0, 0.0]
        rolls, pitches = [], []
        for ai in self.accel:
            r_, p_ = gravity_tilt(ai, alpha=0.98, state=state)
            rolls.append(r_); pitches.append(p_)
        self.roll = np.array(rolls)
        self.pitch = np.array(pitches)

    def frames(self):
        """Yield (t, gray image), honoring stride/max_frames."""
        for k, (t, img) in enumerate(self.reader.frames(self.stride)):
            if self.max_frames and k >= self.max_frames:
                break
            yield k, t, img

    def _idx_at(self, t):
        i = bisect.bisect_left(self.imu_t, t)
        return int(np.clip(i, 0, len(self.imu_t) - 1))

    def tilt_at(self, t):
        i = self._idx_at(t)
        return self.roll[i], self.pitch[i]

    def gyro_dyaw(self, t0, t1):
        """Integrate gyro z between two frame timestamps (camera-frame z is
        the vertical/down axis; positive wz = yaw increasing, matching the
        MCL convention derived from the map-alignment check)."""
        i0, i1 = self._idx_at(t0), self._idx_at(t1)
        if i1 <= i0:
            return 0.0
        wz = self.gyro[i0:i1 + 1, 2]
        tt = self.imu_t[i0:i1 + 1]
        return float(np.trapz(wz, tt))


def coarse_scan(mcl, drone_patch, step_m=96.0, yaw_steps=8, topk=8, batch=512):
    """Full-map grid scan with the first drone patch. Returns the top-K list
    of (score, x_m, y_m, yaw) in MCL state coordinates (meters)."""
    map_rgb = mcl.map_rgb
    h, w = map_rgb.shape[:2]
    step_px = step_m / mcl.gsd_map
    margin = int(step_px)
    xs_px = np.arange(margin, w - margin, step_px)
    ys_px = np.arange(margin, h - margin, step_px)
    drone_feat = mcl.branch(
        torch.from_numpy(drone_patch[None].astype(np.float32))
        .permute(0, 3, 1, 2).to(mcl.device))

    peaks = []
    for yi in range(yaw_steps):
        yaw = yi * 2 * np.pi / yaw_steps
        score_grid = np.zeros((len(ys_px), len(xs_px)))
        for j, py in enumerate(ys_px):
            patches = np.empty((len(xs_px), PATCH_PX, PATCH_PX, 3), dtype=np.uint8)
            for i, px in enumerate(xs_px):
                patches[i] = cropRectangularSectionFromImage(
                    map_rgb, px, py, -yaw, mcl.src_dim, PATCH_PX, 1.0)
            for b0 in range(0, len(xs_px), batch):
                t = torch.from_numpy(
                    patches[b0:b0 + batch].astype(np.float32)).permute(0, 3, 1, 2).to(mcl.device)
                with torch.no_grad():
                    mf = mcl.branch(t)
                    s = mcl.decision(drone_feat.expand(t.shape[0], -1), mf).squeeze(-1)
                score_grid[j, b0:b0 + batch] = s.cpu().numpy()
        # local maxima of this yaw slice
        for j in range(len(ys_px)):
            for i in range(len(xs_px)):
                s = score_grid[j, i]
                nbr = score_grid[max(0, j - 1):j + 2, max(0, i - 1):i + 2]
                if s == nbr.max():
                    peaks.append((float(s),
                                  (xs_px[i] - mcl.cx_px) * mcl.gsd_map,
                                  (mcl.cy_px - ys_px[j]) * mcl.gsd_map,
                                  float(yaw)))
        print(f"  scan yaw {np.degrees(yaw):5.1f}° done (max={score_grid.max():.4f})")
    peaks.sort(reverse=True)
    # suppress near-duplicate peaks (< 100 m apart, any yaw)
    kept = []
    for p in peaks:
        if all(np.hypot(p[1] - q[1], p[2] - q[2]) > 100.0 for q in kept):
            kept.append(p)
        if len(kept) >= topk:
            break
    return kept


def select_initial_point(map_rgb, drone_patch, gsd_map, cx_px, cy_px,
                         disp_max_px=1600, zoom_px=1200):
    """Interactively click the drone's initial position on the map.

    Single-stage zoomable viewer (replacing the old two-stage flow):
      - Scroll wheel = zoom in/out centered on the cursor (x0.7 / x1.4 per
        step, clamped 0.25x…8x full-map view)
      - Left-mouse drag = pan the view when zoomed in
      - Single left click = CONFIRM the point under the cursor as the
        initial position (the crosshair / cursor is exactly where the
        point is selected; use zoom + pan to pixel-accurate placement)
    The drone's frame-0 ortho patch is shown alongside for matching, and
    the HUD shows current view zoom, cursor (map px / map meters).

    Returns (x_m, y_m) in MCL map-frame meters.
    """
    import matplotlib
    # NOTE: mcl.py forces `matplotlib.use("Agg")` at module import, which
    # happens before this function is reached.  `matplotlib.use()` after
    # pyplot import is a no-op, so we must explicitly SWITCH the backend
    # here.  `force=True` (mpl 3.1+) tears down the existing Agg canvas so
    # TkAgg creates a real window instead of failing silently.
    try:
        matplotlib.use("TkAgg", force=True)
    except TypeError:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton

    h, w = map_rgb.shape[:2]
    patch_disp = cv2.resize(drone_patch, (384, 384),
                           interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    ax_map, ax_patch = axes
    im = ax_map.imshow(map_rgb)
    ax_map.set_title("Click to confirm the initial position\n"
                     "SCROLL = zoom (x1.4 / x0.7)  |  DRAG (left-mouse) = pan")
    ax_map.axis("off")

    ax_patch.imshow(patch_disp)
    ax_patch.set_title("Drone view (frame-0 ortho patch, x4)")
    ax_patch.axis("off")

    fig.canvas.manager.set_window_title("Initial point selector")

    HUD = ax_map.text(0.01, 0.99, "", va="top", ha="left",
                      transform=ax_map.transAxes,
                      color="white", fontsize=10,
                      bbox=dict(facecolor="black", alpha=0.6, pad=4))
    cross_v = ax_map.axvline(0, color="lime", ls="--", lw=1, alpha=0.8)
    cross_h = ax_map.axhline(0, color="lime", ls="--", lw=1, alpha=0.8)

    MIN_SCALE = 0.25            # zoomed way out: 4x the whole-map view
    MAX_SCALE = 8.0             # zoomed in far: each map px = 8 screen px
    STEP = 1.4                  # zoom factor per wheel step

    def get_view():
        xl = ax_map.get_xlim(); yl = ax_map.get_ylim()
        return (xl[1] - xl[0]), (yl[0] - yl[1]), xl, yl

    def current_scale():
        vw, vh, _, _ = get_view()
        # scale in screen-px per map-px (using figure DPI approximation via
        # the displayed image bbox — more robust: ratio to full-image width)
        return float(w / vw)

    def apply_view(x_ctr, y_ctr, scale):
        vw = w / scale
        vh = h / scale
        # keep inside the image bounds
        x0 = max(0, min(w - vw, x_ctr - vw / 2))
        y1 = max(vh, min(h, y_ctr + vh / 2))
        ax_map.set_xlim(x0, x0 + vw)
        ax_map.set_ylim(y1, y1 - vh)

    # start: fit whole image (scale ~1, with some slack)
    apply_view(w / 2, h / 2, 0.99 * min(disp_max_px / max(h, w), 1.0) or 0.5)

    def update_hud(x_map, y_map):
        s = current_scale()
        xm = (x_map - cx_px) * gsd_map
        ym = (cy_px - y_map) * gsd_map
        HUD.set_text(f"zoom {s:5.2f}x | map px: ({x_map:.0f}, {y_map:.0f})"
                     f"\nmap m : ({xm:+.1f}, {ym:+.1f})")

    _drag = [None]          # (btn, start_xlim, start_ylim, start_x, start_y)

    def on_scroll(event):
        if event.inaxes is not ax_map:
            return
        mx, my = event.xdata, event.ydata
        if mx is None: return
        cur = current_scale()
        new = cur * (STEP if event.button == "up" else 1.0 / STEP)
        new = np.clip(new, MIN_SCALE, MAX_SCALE)
        vw = w / new; vh = h / new
        # re-center around the cursor position (so cursor stays on the
        # same map pixel — classic "zoom to cursor" behaviour)
        fx = (mx - ax_map.get_xlim()[0]) / (ax_map.get_xlim()[1] - ax_map.get_xlim()[0])
        fy = (my - ax_map.get_ylim()[1]) / (ax_map.get_ylim()[0] - ax_map.get_ylim()[1])
        x0 = max(0, min(w - vw, mx - fx * vw))
        y1 = max(vh, min(h, my + fy * vh))
        ax_map.set_xlim(x0, x0 + vw)
        ax_map.set_ylim(y1, y1 - vh)
        update_hud(mx, my)
        fig.canvas.draw_idle()

    _last_draw = [0.0]

    def throttled_draw(min_interval=1.0 / 30.0):
        # redrawing the 5120x5120 map is expensive; cap hover updates at
        # ~30 fps so mouse motion doesn't queue thousands of redraws
        now = time.monotonic()
        if now - _last_draw[0] >= min_interval:
            _last_draw[0] = now
            fig.canvas.draw_idle()

    def on_motion(event):
        if event.inaxes is ax_map and event.xdata is not None:
            cross_v.set_xdata([event.xdata])
            cross_h.set_ydata([event.ydata])
            update_hud(event.xdata, event.ydata)
        if _drag[0] is not None and event.inaxes is not ax_map:
            return
        if _drag[0] is not None:
            # pan: shift view by (cursor delta in display px -> map px)
            _, xl0, yl0, sx, sy = _drag[0]
            dx = event.x - sx           # screen px delta
            dy = event.y - sy
            # convert screen px to map px via current viewport size
            ax_rect = ax_map.get_window_extent()
            vw_screen = ax_rect.width; vh_screen = ax_rect.height
            vw_map = xl0[1] - xl0[0]; vh_map = yl0[0] - yl0[1]
            dxm = -dx * vw_map / vw_screen
            dym = +dy * vh_map / vh_screen          # screen y is flipped
            x0 = max(0, min(w - vw_map, xl0[0] + dxm))
            y1 = max(vh_map, min(h, yl0[0] + dym))
            ax_map.set_xlim(x0, x0 + vw_map)
            ax_map.set_ylim(y1, y1 - vh_map)
            if event.xdata is not None:
                update_hud(event.xdata, event.ydata)
            fig.canvas.draw_idle()      # pans redraw immediately
        else:
            throttled_draw()            # hover updates capped at 30 fps

    def on_press(event):
        if event.inaxes is not ax_map or event.button != MouseButton.LEFT:
            return
        _drag[0] = (event.button, ax_map.get_xlim(), ax_map.get_ylim(),
                    event.x, event.y)

    def on_release(event):
        if event.button != MouseButton.LEFT or _drag[0] is None:
            return
        btn, xl0, yl0, sx, sy = _drag[0]
        _drag[0] = None
        dist = np.hypot(event.x - sx, event.y - sy)
        if dist < 4 and event.inaxes is ax_map and event.xdata is not None:
            # Clicks without drag = confirmation.
            _state["selected"] = (event.xdata, event.ydata)
            plt.close(fig)            # breaks out of plt.show() below

    _state = {"selected": None}
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)

    print(">>> Select initial point on the map:")
    print("    SCROLL = zoom in/out  |  DRAG (left-mouse) = pan")
    print("    CLICK (no drag) = confirm the point under the cursor")
    print("    HUD shows cursor (map px / meters) + current zoom.")
    plt.show(block=True)     # maps the window; returns when fig is closed

    if _state["selected"] is None:
        raise RuntimeError("Initial point selection cancelled.")
    cx_sel, cy_sel = (int(round(_state["selected"][0])),
                      int(round(_state["selected"][1])))
    cx_sel = int(np.clip(cx_sel, 0, w - 1))
    cy_sel = int(np.clip(cy_sel, 0, h - 1))
    x_m = (cx_sel - cx_px) * gsd_map
    y_m = (cy_px - cy_sel) * gsd_map
    print(f"    Initial point: map px ({cx_sel}, {cy_sel}) "
          f"-> ({x_m:.1f}, {y_m:.1f}) m")
    return x_m, y_m


def local_yaw_scan(mcl, drone_patch, x0_m, y0_m, radius_m=48.0, step_m=12.0,
                   yaw_steps=12, topk=3, batch=256, yaw0=None, yaw_span=2*np.pi):
    """Fine position+yaw scan around a given map-frame point.

    Scans a position grid (±radius_m, step_m) x yaws and returns the top
    peaks (score, x_m, y_m, yaw). With yaw0 given, yaws are scanned in
    [yaw0 - yaw_span/2, yaw0 + yaw_span/2] instead of the full circle
    (used for re-localization, where gyro keeps a yaw hint)."""
    map_rgb = mcl.map_rgb
    drone_feat = mcl.branch(
        torch.from_numpy(drone_patch[None].astype(np.float32))
        .permute(0, 3, 1, 2).to(mcl.device))
    offs = np.arange(-radius_m, radius_m + 1e-6, step_m)
    positions = [(x0_m + dx, y0_m + dy) for dy in offs for dx in offs]
    px_list = [(mcl.cx_px + x / mcl.gsd_map, mcl.cy_px - y / mcl.gsd_map)
               for x, y in positions]
    if yaw0 is None:
        yaws = [yi * 2 * np.pi / yaw_steps for yi in range(yaw_steps)]
    else:
        yaws = np.linspace(yaw0 - yaw_span / 2, yaw0 + yaw_span / 2, yaw_steps)

    peaks = []
    for yaw in yaws:
        patches = np.empty((len(px_list), PATCH_PX, PATCH_PX, 3), dtype=np.uint8)
        for i, (px, py) in enumerate(px_list):
            patches[i] = cropRectangularSectionFromImage(
                map_rgb, px, py, -yaw, mcl.src_dim, PATCH_PX, 1.0)
        scores = np.zeros(len(px_list))
        for b0 in range(0, len(px_list), batch):
            t = torch.from_numpy(
                patches[b0:b0 + batch].astype(np.float32)).permute(0, 3, 1, 2).to(mcl.device)
            with torch.no_grad():
                mf = mcl.branch(t)
                s = mcl.decision(drone_feat.expand(t.shape[0], -1), mf).squeeze(-1)
            scores[b0:b0 + batch] = s.cpu().numpy()
        best = scores.argmax()
        peaks.append((float(scores[best]), positions[best][0], positions[best][1],
                      float(yaw)))
        print(f"  yaw {np.degrees(yaw):5.1f}°: max={scores.max():.4f} "
              f"at ({positions[best][0]:.0f}, {positions[best][1]:.0f}) m")
    peaks.sort(reverse=True)
    # drop near-duplicates (< 30 m apart with similar yaw)
    kept = []
    for p in peaks:
        if all(np.hypot(p[1] - q[1], p[2] - q[2]) > 30.0
               or abs(np.arctan2(np.sin(p[3] - q[3]), np.cos(p[3] - q[3]))) > np.radians(30)
               for q in kept):
            kept.append(p)
        if len(kept) >= topk:
            break
    return kept


def particles_from_peaks(mcl, peaks, n, std=(25.0, 25.0, 0.30, 0.03)):
    """Mixture of Gaussians over scan peaks, ~N/len(peaks) particles each."""
    sub = max(1, n // len(peaks))
    parts = {key: [] for key in ('x', 'y', 'yaw', 'scale', 'w')}
    for s, x, y, yaw in peaks:
        pk = mcl.init_particles(x, y, yaw, 1.0, std, sub)
        for key in parts:
            parts[key].append(pk[key])
    mix = {key: np.concatenate(v) for key, v in parts.items()}
    mix['w'] = np.full(mix['x'].shape[0], 1.0 / mix['x'].shape[0], dtype=np.float32)
    return mix


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('db3', help='rosbag2 .db3 file')
    ap.add_argument('--stride', type=int, default=5, help='camera frame stride (25 Hz / stride)')
    ap.add_argument('--alt', type=float, default=100.0, help='assumed altitude [m]')
    ap.add_argument('--frames', type=int, default=None, help='max frames to process')
    ap.add_argument('--n', type=int, default=1000, help='particle count')
    ap.add_argument('--out', default=None, help='output dir')
    ap.add_argument('--init', choices=('scan', 'click'), default='scan',
                    help='initialization: full-map scan, or interactive click '
                         'on the map (GUI; run outside the sandbox)')
    ap.add_argument('--init-px', nargs=2, type=int, metavar=('PX', 'PY'),
                    default=None, help='headless click: initial point in map '
                    'pixels (implies --init click, no GUI)')
    ap.add_argument('--ulog', default=None, metavar='PATH',
                    help='PX4 .ulg with the flight truth (e.g. '
                    'datasets/03_10_02.ulg): use its per-frame altitude '
                    '(vehicle_global_position) for the orthoprojection '
                    'instead of the fixed --alt')
    ap.add_argument('--ground-alt', type=float, default=None,
                    help='ground elevation [m ASL] for AGL conversion with '
                    '--ulog (default: 1st percentile of the flight alt, '
                    'i.e. the takeoff/landing level)')
    ap.add_argument('--seed', type=int, default=None,
                    help='RNG seed for reproducible runs (particle init, '
                    'process noise, resampling)')
    ap.add_argument('--no-ortho', action='store_true',
                    help='A/B test: bypass orthoprojection and crop the patch '
                         'directly from the raw frame (demo-style center '
                         'crop, coverage/GSD px square), no tilt correction')
    args = ap.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        print(f"RNG seed: {args.seed}")
    if args.init_px:
        args.init = 'click'

    out = args.out or os.path.join(
        OUTPUT_DIR, 'bag_' + os.path.basename(os.path.dirname(args.db3)))
    os.makedirs(out, exist_ok=True)

    map_rgb = load_rgb(MAP_PATH)
    h, w = map_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    print(f"Map: {w}x{h}px  gsd={MAP_GSD} m/px  |  bag: {args.db3}")

    mcl = MCL(map_rgb, MAP_GSD, (cx, cy), CHECKPOINT)
    K, dist, R_imu2cam, dist_model = load_kalibr()
    stream = BagStream(args.db3, stride=args.stride, max_frames=args.frames,
                       R_imu2cam=R_imu2cam)
    fx, fy, cx_cam, cy_cam = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # precompute undistortion maps (radtan or equidistant, from Kalibr)
    if dist_model == 'equidistant':
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), K, (752, 480), cv2.CV_32FC1)
    else:
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, dist, None, K, (752, 480), cv2.CV_32FC1)

    # Optional per-frame altitude from the flight ulog (truth trace): the
    # EKF altitude is ASL; AGL = ASL - ground. Clock alignment reuses
    # gt_eval's yaw-rate correlation (ulog clock has no valid UTC).
    alt_at = None
    if args.ulog:
        import struct as _struct
        import gt_eval as _gt
        _streams = _gt.ulog_streams(args.ulog)
        _bt, _bgz = _gt.bag_gyro_z(args.db3)
        _off, _peak = _gt.align_clocks(_streams, _bt, _bgz)
        _gp = _streams[_gt.ID_GPOS]
        _ft = np.array([x[0] for x in _gp])
        _fa = np.array([_struct.unpack_from('<f', b, 24)[0] for _, b in _gp])
        _o = np.argsort(_ft)
        _ft, _fa = _ft[_o], _fa[_o]
        _ground = (args.ground_alt if args.ground_alt is not None
                   else float(np.percentile(_fa, 1)))
        # bag frame/IMU header stamps are a bag-relative clock; bridge it to
        # the sqlite UTC scale used by the alignment
        _rel2utc = _bt[0] - stream.imu_t[0]
        print(f"ulog altitude profile: align peak {_peak:.2f}, "
              f"ground {_ground:.0f} m ASL, flight ASL "
              f"[{_fa.min():.0f}, {_fa.max():.0f}] m")

        def alt_at(t, _ft=_ft, _fa=_fa, _ground=_ground,
                   _rel2utc=_rel2utc, _off=_off):
            asl = float(np.interp(t + _rel2utc + _off, _ft, _fa))
            return max(20.0, asl - _ground)   # AGL with a safety floor
    foot_y = 480 * args.alt / fy          # across-track ground footprint [m]
    if foot_y < COVERAGE_M:
        print(f"  [INFO] altitude {args.alt:.0f} m -> {foot_y:.0f} m footprint "
              f"(< {COVERAGE_M:.0f} m): altitude-adaptive coverage engaged "
              f"(patch shrinks to the visible ground; map patches match)")
    print(f"IMU: {len(stream.imu_t)} samples, "
          f"{1.0 / np.mean(np.diff(stream.imu_t)):.0f} Hz  |  "
          f"alt={args.alt} m  fx={fx:.1f} (Kalibr)  stride={args.stride} "
          f"({25.0 / args.stride:.1f} Hz frames)")

    T = 0.02
    roughen_std = (5.0, 5.0, 0.03, 0.01)
    noise_std = np.array([1.5, 1.5, 0.02, 0.005])
    LOST_SCORE = 0.6        # below this the filter considers itself lost
    LOST_FRAMES = 5         # consecutive lost frames before re-localizing
    MIN_RESP = 0.50          # phase-correlation response gate (correct pairs
                             # score 0.9+; 0.1-0.4 pairs give garbage shifts)
    V_MAX = 15.0            # drone max speed [m/s] (shift + rescan radius)
    CONSENSUS_N = 7          # velocity buffer length (~1.4 s at 5 Hz)
    CONSENSUS_R = 0.60       # min direction consistency to inject translation
    HOVER_NOISE = 0.6        # position noise [m] when no translation is
                             # injected (hover jitter; was 3.0 — on hover/spin
                             # bags that diffusion alone random-walks the
                             # estimate hundreds of meters)

    def init_from_scan(patch):
        """(Re)initialize particles from a coarse scan of the given patch."""
        peaks = coarse_scan(mcl, patch, topk=5)
        print("  scan peaks (score, x, y, yaw°): " + "; ".join(
            f"({s:.3f},{x:.0f},{y:.0f},{np.degrees(yaw):.0f}°)"
            for s, x, y, yaw in peaks))
        return particles_from_peaks(mcl, peaks, args.n)

    particles = None
    prev = None            # (t, patch_gray_float)
    vel_buf = []           # recent candidate velocities, world frame [m/frame]
    yaw_est = 0.0          # last yaw estimate (for body->world rotation)
    lost_count = 0
    reloc_fails = 0        # consecutive failed re-localizations (widen search)
    last_good = None       # (x, y, yaw, t) at the last frame with score >= LOST_SCORE
    log = []

    time_start = time.time()

    frames = stream.frames()
    for k, t, gray in frames:
        roll, pitch = stream.tilt_at(t)
        gray = cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # Altitude-adaptive patch: shrink to the fully-valid footprint
        # (no zero padding), and match map-patch coverage to it.
        alt = alt_at(t) if alt_at is not None else args.alt
        if args.no_ortho:
            # Demo-style: direct center crop on the raw frame, no tilt
            # correction. Coverage limited by the vertical footprint.
            gsd = alt / fx
            coverage = min(COVERAGE_M, 480.0 * alt / fy)
            crop_px = max(32, int(round(coverage / gsd)))
            half = crop_px // 2
            x0 = int(np.clip(376 - half, 0, 752 - crop_px))
            y0 = int(np.clip(240 - half, 0, 480 - crop_px))
            patch = rgb[y0:y0 + crop_px, x0:x0 + crop_px]
            patch = cv2.resize(patch, (PATCH_PX, PATCH_PX),
                               interpolation=cv2.INTER_AREA)
            coverage = crop_px * gsd
            mcl.set_coverage(coverage)
        else:
            ortho, mask_o, extent = orthoproject(
                rgb, alt, roll, pitch, fx, fy, cx_cam, cy_cam)
            coverage = valid_coverage_m(mask_o, extent)
            mcl.set_coverage(coverage)
            patch, mask, center_ground = extract_patch(
                ortho, mask_o, extent, size_m=coverage)
            # resize to the CNN input size: smaller coverage = finer
            # effective GSD (coverage/PATCH_PX m/px), same 96x96 input
            # (validated by coverage_sweep_test.py)
            if patch.shape[0] != PATCH_PX:
                patch = cv2.resize(patch, (PATCH_PX, PATCH_PX),
                                   interpolation=cv2.INTER_LINEAR)
        mcl.set_drone_patch(patch)
        if k == 0:
            cv2.imwrite(os.path.join(out, 'frame0_patch.png'),
                        cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            if args.init == 'click':
                if args.init_px:
                    px_sel, py_sel = args.init_px
                    x0_m = (px_sel - cx) * MAP_GSD
                    y0_m = (cy - py_sel) * MAP_GSD
                    print(f"\nHeadless init point: map px ({px_sel}, {py_sel}) "
                          f"-> ({x0_m:.1f}, {y0_m:.1f}) m")
                else:
                    print("\nInteractive initial-point selection (two-stage click)...")
                    x0_m, y0_m = select_initial_point(
                        map_rgb, patch, MAP_GSD, cx, cy)
                print("\nLocal yaw scan around the clicked point "
                      "(±48 m, 12 yaws)...")
                peaks = local_yaw_scan(mcl, patch, x0_m, y0_m)
                print("  local peaks (score, x, y, yaw°): " + "; ".join(
                    f"({s:.3f},{x:.0f},{y:.0f},{np.degrees(yaw):.0f}°)"
                    for s, x, y, yaw in peaks))
                particles = particles_from_peaks(
                    mcl, peaks, args.n, std=(15.0, 15.0, 0.25, 0.03))
            else:
                print("\nInitial coarse scan (full map, 8 yaws)...")
                particles = init_from_scan(patch)

        patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)

        motion = None
        ns = noise_std.copy()
        if prev is not None:
            dyaw = stream.gyro_dyaw(prev[0], t)
            # NOTE: no yaw-noise inflation during spins. Gyro z between
            # frames is far more accurate than the CNN yaw hint; inflating
            # it by 0.5*|dyaw| let resampling drag yaw onto false peaks —
            # measured divergence reached 180° on bag_0002, which destroys
            # the map-patch rotation and with it the whole measurement.
            # Translation: phase correlation of consecutive patches. Two gates:
            #  per-frame (response + plausibility) feeds a candidate buffer;
            #  a CONSENSUS check then decides whether to actually inject it.
            #  Rationale (measured on bag_0002): on hover/spin bags the raw
            #  shifts are random-direction garbage (resultant 0.06) even at
            #  response 0.7+ — no per-frame gate can filter them. Real flight
            #  produces direction-persistent shifts; only those get injected.
            if abs(dyaw) < np.radians(10):
                # patches are 96x96 px regardless of coverage; effective GSD
                # is coverage/PATCH_PX m/px — convert shift px -> meters
                gs = coverage / PATCH_PX
                shift, resp = cv2.phaseCorrelate(prev[1], patch_gray)
                vx, vy = -shift[0] * gs, -shift[1] * gs   # -> body frame
                if resp >= MIN_RESP and np.hypot(vx, vy) <= V_MAX * (1.0 / (25.0 / args.stride)):
                    c, s = np.cos(yaw_est), np.sin(yaw_est)
                    vel_buf.append((vx * c - vy * s, vx * s + vy * c))
                    del vel_buf[:-CONSENSUS_N]
                a = np.asarray(vel_buf)
                if len(vel_buf) >= 4:
                    m = a.mean(0)
                    r = np.hypot(*m) / max(1e-9, np.mean(np.hypot(a[:, 0], a[:, 1])))
                    if r >= CONSENSUS_R:
                        # consistent flight: inject the robust median, rotated
                        # back from world into the current body frame
                        wx, wy = np.median(a[:, 0]), np.median(a[:, 1])
                        c, s = np.cos(yaw_est), np.sin(yaw_est)
                        motion = (wx * c + wy * s, -wx * s + wy * c, dyaw)
            if motion is None:   # no trusted translation: yaw-only + hover noise
                motion = (0.0, 0.0, dyaw)
                ns[:2] = HOVER_NOISE
        prev = (t, patch_gray)

        decay = max(0.2, 1.0 - 0.4 * k / 50.0)
        # While lost, widen the cloud progressively so it can re-acquire the
        # true peak locally instead of waiting for a full rescan.
        rs = roughen_std if lost_count == 0 else tuple(
            v * (1.0 + 2.0 * min(lost_count, 4)) for v in roughen_std)
        particles, est, scores = mcl.step(
            particles, T, rs, decay,
            motion=motion, noise_std=tuple(ns), resample_eff=0.3)
        yaw_est = est['yaw']
        log.append({'k': k, 't': t,
                    'x': est['x'], 'y': est['y'], 'yaw': est['yaw'],
                    'scale': est['scale'], 'spread': est['spread'],
                    'neff': mcl.last_neff, 'score': scores.max(),
                    'coverage': coverage, 'alt': alt,
                    'motion': motion if motion else (0, 0, 0),
                    'reloc': 0})
        # --- lost detection & re-localization ---
        if scores.max() >= LOST_SCORE:
            lost_count = 0
            reloc_fails = 0
            last_good = (est['x'], est['y'], est['yaw'], t)
        else:
            lost_count += 1
        if lost_count >= LOST_FRAMES:
            # Re-localize NEAR the last known position first: while lost, the
            # drone is bounded by V_MAX * (time since last good fix). Only
            # escalate to a full-map scan after repeated failures.
            if last_good is not None and reloc_fails < 2:
                radius = max(40.0, V_MAX * (t - last_good[3]))
                # anchor the yaw search to the GYRO-integrated yaw since the
                # last good fix, not to the (possibly diverged) estimate
                yaw_anchor = last_good[2] + stream.gyro_dyaw(last_good[3], t)
                print(f"  frame {k:4d}: LOST ({LOST_FRAMES} frames) -> local "
                      f"rescan (±{radius:.0f} m around last position)")
                peaks = local_yaw_scan(
                    mcl, patch, last_good[0], last_good[1],
                    radius_m=radius, step_m=radius / 8, yaw_steps=7,
                    topk=5, yaw0=yaw_anchor, yaw_span=np.radians(90))
                print("  local peaks (score, x, y, yaw°): " + "; ".join(
                    f"({s:.3f},{x:.0f},{y:.0f},{np.degrees(yaw):.0f}°)"
                    for s, x, y, yaw in peaks))
                reloc_fails += 1
            else:
                print(f"  frame {k:4d}: LOST -> GLOBAL rescan "
                      f"(reloc_fails={reloc_fails})")
                peaks = coarse_scan(mcl, patch, topk=5)
                reloc_fails += 1
            particles = particles_from_peaks(
                mcl, peaks, args.n, std=(10.0, 10.0, 0.15, 0.02))
            log[-1]['reloc'] = 1
            lost_count = 0
        if k % 10 == 0:
            mv = log[-1]['motion']
            print(f"  frame {k:4d} t={t:8.1f}s  x={est['x']:7.1f} y={est['y']:7.1f} "
                  f"yaw={np.degrees(est['yaw']):7.1f}°  spread={est['spread']:5.1f}m  "
                  f"neff={mcl.last_neff:6.1f}  score={scores.max():.4f}  "
                  f"mov=({mv[0]:+5.1f},{mv[1]:+5.1f},{np.degrees(mv[2]):+5.1f}°)")

    time_total = time.time() - time_start
    print(f"Total time: {time_total:.4f}s, Process {len(log)} frames")

    # --- save results ---
    np.savez(os.path.join(out, 'replay_log.npz'),
             **{key: np.array([r[key] for r in log]) for key in log[0]})
    print(f"\nSaved log: {out}/replay_log.npz  ({len(log)} frames)")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    # crop the map to the trace bounding box (full resolution, padded) so
    # small traces stay readable; falls back to the whole map if the crop
    # would cover most of it anyway
    tx = [cx + r['x'] / MAP_GSD for r in log]
    ty = [cy - r['y'] / MAP_GSD for r in log]
    mh, mw = map_rgb.shape[:2]
    pad_px = 150
    x0 = int(max(0, min(tx) - pad_px)); x1 = int(min(mw, max(tx) + pad_px))
    y0 = int(max(0, min(ty) - pad_px)); y1 = int(min(mh, max(ty) + pad_px))
    if (x1 - x0) * (y1 - y0) > 0.25 * mw * mh:      # trace spans most of map
        x0, y0, x1, y1 = 0, 0, mw, mh
    disp = map_rgb[y0:y1, x0:x1]
    ax.imshow(disp)
    ax.plot([t - x0 for t in tx], [t - y0 for t in ty], 'c-', lw=1)
    ax.plot(tx[0] - x0, ty[0] - y0, 'g+', ms=15, mew=2, label='start')
    ax.plot(tx[-1] - x0, ty[-1] - y0, 'r+', ms=15, mew=2, label='end')
    ax.set_title(f'Estimated trajectory on map '
                 f'({(x1 - x0) * MAP_GSD:.0f} x {(y1 - y0) * MAP_GSD:.0f} m crop)')
    ax.legend(loc='upper right')
    ax.axis('off')

    ax = axes[1]
    ax.plot([r['spread'] for r in log], label='spread (m)')
    ax.plot([r['neff'] for r in log], label='N_eff')
    ax.set_xlabel('frame')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Spread / N_eff')

    ax = axes[2]
    ax.plot([r['score'] for r in log], label='max score')
    ax.set_xlabel('frame')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Best-particle score')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'replay.png'), dpi=120)
    print(f"Saved plot: {out}/replay.png")


if __name__ == '__main__':
    main()
