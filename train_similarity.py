"""Universal SIVL-style similarity-model training for GNSS-denied localization.

Region-agnostic re-implementation of sivl/trainSimilarityModel.py's training
form — config-driven (train_config.yml, like sivl's configuration.yml),
anchor/positive/negative BCE training with a tqdm loop and sivl-compatible
checkpoints — extended with the domain-alignment stages this pipeline needs:

  1. --build-bank   multi-temporal map patch bank from the area folders:
                    every temporal layer SIFT-aligned to its area's newest
                    layer (failures kept at capture registration, like sivl)
                    and resampled to 1 m/px; cropped on sivl's dense
                    centerpoint grid (adjacent samples, no stride)
  2. (pretrain)     anchor/positive from DIFFERENT temporal layers of the
                    same location (sivl's BitmapDataset pairing — forces
                    time-invariant structural features instead of texture
                    matching); the anchor additionally gets the calibrated
                    drone-appearance simulation (sim_drone)
  3. --build-real   real drone patches + GPS-truth map pairs (per flight)
  4. --finetune     mixed real + simulated training from a pretrain ckpt

For a new region: edit train_config.yml, then run the stages in order.

Usage:
    python train_similarity.py --build-bank
    python train_similarity.py                       # pretrain
    python train_similarity.py --build-real
    python train_similarity.py --finetune
    python train_similarity.py --epochs 2 --max-batches 30   # smoke test
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
import zarr
from numcodecs import Blosc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
SIVL_DIR = os.path.join(ROOT, 'sivl')
sys.path.insert(0, SIVL_DIR)
sys.path.insert(0, ROOT)
from models.orthosimilarity import BranchNet, DecisionNet  # noqa: E402
from utils.utils import (cropRectangularSectionFromImage,  # noqa: E402
                         numpyImageToTensor)
from bag_reader import parse_imu  # noqa: E402


# ------------------------------------------------------------------ config
def _namespace(d):
    """dict/list tree -> nested SimpleNamespace (sivl uses Box for this)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_namespace(v) for v in d]
    return d


with open(os.path.join(ROOT, 'train_config.yml')) as _f:
    cfg = _namespace(yaml.safe_load(_f))

PATCH_PX = cfg.sampledimensions.dimension_px   # network input size

BANK_PX = 136           # stored bank crop (fits a rotated 96px square: 96*sqrt(2))
BANK_CODEC = Blosc(cname='zstd', clevel=3, shuffle=Blosc.NOSHUFFLE)
                        # measured on real crops: 1.34x (65.8 -> ~49 GB),
                        # decode 0.07 ms/crop — cheap in DataLoader workers
BANK_CHUNK = (1, BANK_PX, BANK_PX, 3)  # one crop = one compressed chunk file
                                        # (disjoint-row writes never share a
                                        # chunk -> contention-free parallelism)
ALIGN_RES = 0.5         # SIFT alignment detection scale (speed)
ALIGN_MIN_INLIERS = 15  # drop layers with fewer RANSAC inliers
ALIGN_MAX_CANDIDATES = 2  # chained alignment: try this many nearest-date
                          #  already-aligned layers before dropping


# ------------------------------------------------------ flights / georeference
def ulog_attitude_yaw(path):
    from pyulog import ULog
    u = ULog(path)
    for d in u.data_list:
        if d.name == 'vehicle_attitude':
            t = d.data['timestamp'] / 1e6
            q = np.stack([d.data[f'q[{i}]'] for i in range(4)], 1)
            yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                             1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            return t, np.unwrap(yaw)
    raise RuntimeError('no vehicle_attitude')


def ulog_gps(path):
    from pyulog import ULog
    u = ULog(path)
    for d in u.data_list:
        if d.name == 'vehicle_gps_position':
            ok = d.data['fix_type'] >= 3
            return (d.data['timestamp'][ok] / 1e6,
                    d.data['latitude_deg'][ok], d.data['longitude_deg'][ok])
    raise RuntimeError('no vehicle_gps_position')


def bag_gyro_z(db3):
    import sqlite3
    db = sqlite3.connect(f'file:{db3}?mode=ro', uri=True)
    tid = dict(db.execute('SELECT name,id FROM topics'))['/imu0']
    rows = db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? '
                      'ORDER BY timestamp', (tid,)).fetchall()
    db.close()
    # HEADER stamps (the bridge's own clock) — same clock as the camera
    # header stamps that mcl_node logs as t.
    t = np.array([parse_imu(b)[0] for _, b in rows])
    gz = np.array([parse_imu(b)[1][2] for _, b in rows])
    return t, gz


def align_clocks(t_att, yaw, t_imu, gz):
    """Offset (att_t = bag_t + off) via direct gyro-z / yaw-rate correlation.

    The clocks have unrelated bases (ulog boot time ~1e2 s vs bag epoch time
    ~1.8e9 s), so scan every offset that gives temporal overlap, then refine.
    """
    fr = np.gradient(yaw, t_att)

    def corr_at(off):
        x = t_imu + off
        m = (x >= t_att[0]) & (x <= t_att[-1])
        if m.sum() < 1000:
            return 0.0
        return abs(np.corrcoef(gz[m], np.interp(x[m], t_att, fr))[0, 1])

    lo = t_att[0] - t_imu[-1]
    hi = t_att[-1] - t_imu[0]
    best_c, best_off = 0.0, 0.0
    for off in np.arange(lo, hi, 1.0):
        c = corr_at(off)
        if c > best_c:
            best_c, best_off = c, off
    for off in best_off + np.arange(-1.5, 1.51, 0.05):
        c = corr_at(off)
        if c > best_c:
            best_c, best_off = c, off
    return best_off, best_c


def latlon_to_px(lat, lon, z=17):
    """WGS-84 lat/lon -> global web-mercator pixel coords at zoom z."""
    xt = (lon + 180.0) / 360.0 * 2 ** z * 256.0
    yt = ((1 - np.log(np.tan(np.radians(lat)) +
                      1 / np.cos(np.radians(lat))) / np.pi) / 2
          * 2 ** z * 256.0)
    return xt, yt


def ulog_altitude(path):
    """Altitude above takeoff (m) vs ulog time, from vehicle_local_position
    (NED z negated). Used to gate out takeoff/landing frames — too close to
    the ground for usable orthoprojections."""
    from pyulog import ULog
    u = ULog(path)
    for d in u.data_list:
        if d.name == 'vehicle_local_position':
            ok = np.isfinite(d.data['z'])
            return d.data['timestamp'][ok] / 1e6, -d.data['z'][ok]
    raise RuntimeError('no vehicle_local_position')


def flight_truth(flight, geo_offset):
    """Replay-log rows + GPS truth (deployment-map pixels) for one flight.

    Returns (row_of_k, truth_px, cov_all, alt_all): truth_px[i] is the (x, y)
    map pixel of log row i, cov_all[i] its drone-patch coverage in meters,
    alt_all[i] its altitude above takeoff (m, ulog local frame).
    """
    log = np.load(os.path.join(flight.run_dir, 'replay_log.npz'))
    k_all, t_all = log['k'], log['t']
    cov_all = log['coverage'].astype(float)
    row_of_k = {int(k): i for i, k in enumerate(k_all)}
    t_att, yaw = ulog_attitude_yaw(flight.ulog)
    t_gps, lat, lon = ulog_gps(flight.ulog)
    t_imu, gz = bag_gyro_z(flight.db3)
    off, peak = align_clocks(t_att, yaw, t_imu, gz)
    print(f'[{flight.name}] clock offset: att_t = bag_t + {off:+.3f} s '
          f'(corr {peak:.3f})')
    gx, gy = latlon_to_px(lat, lon)
    gi = np.interp(t_all, t_gps - off, gx)
    gj = np.interp(t_all, t_gps - off, gy)
    truth_px = np.stack([gi + geo_offset[0], gj + geo_offset[1]], 1)
    t_alt, alt = ulog_altitude(flight.ulog)
    alt_all = np.interp(t_all, t_alt - off, alt)
    return row_of_k, truth_px, cov_all, alt_all


# --------------------------------------------------------------- scoring
class Scorer:
    """BranchNet+DecisionNet similarity scorer (deployment-identical)."""

    def __init__(self, ckpt, device=None):
        self.device = torch.device(
            device or cfg.device if torch.cuda.is_available() else 'cpu')
        self.branch = BranchNet().to(self.device)
        self.decision = DecisionNet().to(self.device)
        c = torch.load(ckpt, map_location=self.device)
        self.branch.load_state_dict(c['branchmodel_state_dict'])
        self.decision.load_state_dict(c['decisionmodel_state_dict'])
        self.branch.eval()
        self.decision.eval()

    def score_many(self, drone_patch, map_patches, bs=1024):
        """drone_patch 96x96x3, map_patches (N,96,96,3) -> (N,) scores."""
        t = numpyImageToTensor(drone_patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            df = self.branch(t)
            out = []
            for i in range(0, len(map_patches), bs):
                mp = torch.from_numpy(
                    map_patches[i:i + bs].astype(np.float32)
                ).permute(0, 3, 1, 2).to(self.device)
                mf = self.branch(mp)
                out.append(self.decision(df.expand(len(mf), -1), mf)
                           .squeeze(-1).cpu())
            return torch.cat(out).numpy()

    def embed(self, patches, bs=1024):
        """patches (N,96,96,3) -> (N,D) branch features (diagnostics)."""
        with torch.no_grad():
            out = []
            for i in range(0, len(patches), bs):
                mp = torch.from_numpy(
                    patches[i:i + bs].astype(np.float32)
                ).permute(0, 3, 1, 2).to(self.device)
                out.append(self.branch(mp).cpu())
            return torch.cat(out).numpy()


def extract_all(map_rgb, px, py, src_dim, angles_deg):
    """Map patches at (px, py) for every rotation, coverage src_dim px."""
    n = len(angles_deg)
    patches = np.empty((n, PATCH_PX, PATCH_PX, 3), np.uint8)
    for i, a in enumerate(angles_deg):
        patches[i] = cropRectangularSectionFromImage(
            map_rgb, px, py, math.radians(a), src_dim, PATCH_PX, 1.0)
    return patches


# ------------------------------------------------- drone-side simulation
def rot_crop(src, theta_deg, scale=1.0, out=PATCH_PX, corner_noise=0.0,
             shift=(0.0, 0.0), rng=None):
    """Extract an `out`x`out` crop whose source is a `out*scale`-side square
    centered at src's center + shift, rotated by theta_deg. Fits in BANK_PX
    for scale <= 1 at any theta (bounding box <= 96*sqrt(2) <= 136)."""
    h, w = src.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    th = math.radians(theta_deg)
    d = out * scale / 2.0
    c, s = math.cos(th), math.sin(th)
    corners = np.array([[-d, -d], [d, -d], [d, d], [-d, d]], np.float32)
    R = np.array([[c, -s], [s, c]], np.float32)
    corners = corners @ R.T + np.array([cx + shift[0], cy + shift[1]],
                                       np.float32)
    if corner_noise > 0:
        corners = corners + rng.normal(0, corner_noise, corners.shape).astype(np.float32)
    dst = np.array([[0, 0], [out, 0], [out, out], [0, out]], np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(src, H, (out, out))


def sim_drone(img, rng):
    """Drone-side appearance simulation on an RGB crop -> gray 3ch.

    Calibrated on 246 real orthoprojection patches from bag_0001 (debug
    saves): mean 93-172 (med 142), std 19-71 (med 55), Laplacian var
    1195-4595 (med 2354) i.e. SHARPER than the map (~500). The real patches
    are contrasty (camera AGC) and sharp, so: light blur only, then affine
    normalization to sampled real (mean, std), then optional unsharp mask to
    cover the high-sharpness tail. Camera-specific (not region-specific).
    """
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if rng.random() < 0.25:                       # GSD loss: down+up scale
        f = rng.uniform(0.6, 0.85)
        small = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        g = cv2.resize(small, (g.shape[1], g.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:                        # light blur only
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 0.8))
    if rng.random() < 0.5:                        # sensor noise
        g = g + rng.normal(0, rng.uniform(1, 6), g.shape)
    if rng.random() < 0.3:                        # gamma (histogram shape)
        gamma = rng.uniform(0.7, 1.4)
        g = 255 * (np.clip(g, 0, 255) / 255.0) ** gamma
    # affine normalization to the measured real-patch statistics
    tm, ts = rng.uniform(95, 175), rng.uniform(25, 70)
    g = (g - g.mean()) / (g.std() + 1e-6) * ts + tm
    if rng.random() < 0.3:                        # sharpness boost
        blur = cv2.GaussianBlur(g, (0, 0), 1.0)
        g = g + rng.uniform(0.5, 1.5) * (g - blur)
    g = np.clip(g, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


# ------------------------------------------------------------------ bank
def _area_entries():
    """(name, path, split) per configured area folder (train + val).

    Mirrors sivl's trainingdata_bitmaps / testingdata_bitmaps: each entry is
    a folder of aligned, equally-sized, different-time images
    (year_month_N.jpg Google-Earth exports of one fixed viewport, ~equal
    GSD). Like sivl's BitmapDataset, the bank grids at the images' NATIVE
    pixel size — no GSD resampling.
    """
    p = cfg.data.path
    entries = []
    for split, key in (('train', 'trainingdata_bitmaps'),
                       ('val', 'testingdata_bitmaps')):
        areas = getattr(p, key, None)
        if not areas:
            continue
        for name, path in vars(areas).items():
            entries.append((name, path, split))
    return entries


def _image_files(path):
    """All image files under the area folder, recursive (Google-Earth exports
    sit at the top level; USGS-style downloads sit in an images/ subfolder)."""
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    found = []
    for sub, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(exts):
                found.append(os.path.join(sub, f))
    return sorted(os.path.relpath(f, path) for f in found)


def _file_date(fname):
    """Filename -> months since year 0 (for newest-layer selection and
    nearest-date chaining). Parses both export styles:

      year_month_N.jpg          (Google-Earth:  2012_10_1.jpg)
      *_YYYYMMDD.png            (USGS: m_4007430_ne_18_1_20100829.png)
    """
    try:
        return int(fname[:4]) * 12 + int(fname[5:7])
    except ValueError:
        m = re.search(r'_(\d{4})(\d{2})\d{2}\.', fname)
        return int(m.group(1)) * 12 + int(m.group(2)) if m else 0


def _match_translation(ref_feats, img_feats, rng):
    """RANSAC translation (ALIGN_RES-scale px) aligning img to ref, or None.

    t satisfies ref_coord = img_coord + t (a feature at p in img appears at
    p + t in ref). Translation-only: the Google-Earth viewport is fixed, so
    temporal layers differ only by source-imagery georegistration shifts
    (observed 10-25 px on old layers).
    """
    (kr, dr), (ki, di) = ref_feats, img_feats
    if dr is None or di is None:
        return None
    good = [m for m, n in cv2.BFMatcher().knnMatch(dr, di, k=2)
            if m.distance < 0.75 * n.distance]
    if len(good) < ALIGN_MIN_INLIERS:
        return None
    diff = (np.float32([kr[m.queryIdx].pt for m in good])
            - np.float32([ki[m.trainIdx].pt for m in good]))
    best_cnt, best_t = 0, None
    for i in rng.choice(len(diff), min(len(diff), 500), replace=False):
        cnt = int((np.linalg.norm(diff - diff[i], axis=1) < 2.0).sum())
        if cnt > best_cnt:
            best_cnt, best_t = cnt, diff[i]
    if best_cnt < ALIGN_MIN_INLIERS:
        return None
    inl = np.linalg.norm(diff - best_t, axis=1) < 2.0
    return diff[inl].mean(axis=0)


# ------------------------------------------------- parallel bank workers
# Worker counts are MEMORY-BUDGETED, not just cpu-counted (16 uncapped
# SIFT workers OOM'd the 32 GB machine; even 12 pushed it to 27 GB used —
# the budget must reserve the OS baseline ~5 GB AND the parent process
# ~3 GB with torch imported, which forked workers share copy-on-write).
# Measured private peak per worker: SIFT align ~1.8-2 GB (OpenCV DoG
# pyramid on the 0.5-scale image), warp/crop ~0.4 GB.
BANK_MEM_BUDGET_GB = 24        # TOTAL cap: system + parent + all workers
PARENT_SYSTEM_RESERVE_GB = 8   # OS/desktop baseline + parent (torch import)
ALIGN_WORKER_GB = 2.0          # pass-1 SIFT private peak (measured ~1.8)
CROP_WORKER_GB = 0.5           # pass-2 peak (imread + warpAffine buffers)


def _bank_align_task(args):
    """Pass-1 worker: SIFT + chained translation alignment + sivl node grid
    for ONE area (runs in a forked process; only small results return).

    Deterministic: each area seeds its own RNG (1000 + area index), so the
    bank does not depend on worker scheduling.
    """
    name, path, sp, ai, files, ap_std, step, g0, margin = args
    cv2.setNumThreads(1)  # parallelism is across workers, not inside cv2
    rng = np.random.default_rng(1000 + ai)
    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.02)
    feats, shape0 = {}, None
    for fname in files:
        g = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
        assert g is not None, fname
        if shape0 is None:
            shape0 = g.shape
        assert g.shape == shape0, \
            f'{name}/{fname}: layer sizes differ within the area'
        small = cv2.resize(g, None, fx=ALIGN_RES, fy=ALIGN_RES,
                           interpolation=cv2.INTER_AREA)
        del g  # free the full-res grayscale before SIFT's pyramid spike
        feats[fname] = sift.detectAndCompute(small, None)
        del small
    # chained translation alignment vs the newest layer (newest first).
    # Only REAL alignments may chain: a failed layer kept at (0,0) must
    # not become a candidate for older layers, or they all inherit its
    # unknown true offset (observed: area_9's every layer off by ~(-13,-19)
    # after chaining through a failed 2020_11).
    t_of = {files[-1]: np.zeros(2, np.float32)}  # ALIGN_RES-scale px
    aligned = {files[-1]}
    unaligned = []
    for i in range(len(files) - 2, -1, -1):
        f = files[i]
        cands = sorted(aligned, key=lambda c: abs(_file_date(c) - _file_date(f)))
        for c in cands[:ALIGN_MAX_CANDIDATES]:
            tv = _match_translation(feats[c], feats[f], rng)
            if tv is not None:
                t_of[f] = tv + t_of[c]
                aligned.add(f)
                break
        if f not in aligned:    # keep at capture registration, like sivl
            t_of[f] = np.zeros(2, np.float32)  # (which assumes aligned)
            unaligned.append(f)
    # node grid ONCE per area (sivl centerpoints + one-time apShifting
    # jitter), shared by all layers so pairs stay co-located. The grid
    # margin keeps every 136 px crop inside the image.
    ow, oh = shape0[1], shape0[0]               # native size, no resample
    nodes = []
    gy = g0
    while gy + PATCH_PX / 2 + margin < oh:
        gx = g0
        while gx + PATCH_PX / 2 + margin < ow:
            cx = int(np.clip(round(gx + rng.normal(0, ap_std)),
                             BANK_PX // 2, ow - BANK_PX // 2))
            cy = int(np.clip(round(gy + rng.normal(0, ap_std)),
                             BANK_PX // 2, oh - BANK_PX // 2))
            nodes.append((cx, cy))
            gx += step
        gy += step
    return dict(name=name, ai=ai, sp=sp, path=path, files=files,
                t_of=t_of, unaligned=unaligned, nodes=nodes, ow=ow, oh=oh)


def _bank_crop_task(args):
    """Pass-2 worker: warp + crop a BLOCK of nodes of one area into the
    node-major zarr bank (all layers of a node land on contiguous rows —
    the train-time anchor/positive read hits adjacent chunks instead of
    two random reads ~100k rows apart). Each row is its own compressed
    chunk file, so writes to disjoint rows are contention-free. Crops go
    straight from the warped layer image to the store (no block buffer;
    peak memory = one full-size image ~0.5 GB).
    """
    name, store, path, files, tvs, ow, oh, centers, row0, stride = args
    cv2.setNumThreads(1)
    bank = zarr.open(store, mode='r+')  # array created by the parent
    half = BANK_PX // 2
    for li, fname in enumerate(files):
        img = cv2.imread(os.path.join(path, fname))
        assert img is not None, f'{name}/{fname}'
        M = np.float32([[1, 0, tvs[li][0]], [0, 1, tvs[li][1]]])
        img = cv2.warpAffine(img, M, (ow, oh))  # frees the imread buffer
        for i, (cx, cy) in enumerate(centers):
            bank[row0 + i * stride + li] = cv2.cvtColor(
                img[cy - half: cy + half, cx - half: cx + half],
                cv2.COLOR_BGR2RGB)
    return len(centers) * len(files)


def build_bank():
    """Multi-temporal map patch bank from the area folders.

    sivl's BitmapDataset data model, precomputed once. Per area
    (cfg.data.path.trainingdata_bitmaps + testingdata_bitmaps):

      1. ALIGN: every temporal layer is translation-aligned to the area's
         NEWEST layer via SIFT+RANSAC (historical layers can be offset by
         10-25 px; unaligned anchor/positive pairs would teach translation
         invariance — fatal for localization). Layers that fail against the
         newest retry against the nearest-date ALREADY-ALIGNED layer
         (chained; translations compose). Layers that fail every candidate
         are KEPT at their capture registration (offset 0,0 — sivl assumes
         its bitmaps are aligned) and reported.
      2. WARP: each layer is shifted by its alignment translation only —
         NO GSD resampling (origin sivl grids at the images' native pixel
         size; our previous resample to 1 m/px shrank the Huizhou grid
         83x83 -> 49x49 and lost 61% of the samples).
      3. CROP: sivl's BitmapDataset centerpoint grid — starts at
         dim/2 + dim/sqrt(2), steps by dim + marginBetweenSamples_px (0 =
         ADJACENT samples, no stride), + N(0, apShiftingStd) jitter drawn
         ONCE per node (sivl draws it in __init__, not per epoch) — cropped
         from EVERY layer at the SAME centers so the dataset can pair
         different times of the same location. ALL crops are kept (sivl has
         no featureless-patch filter).

    Both passes run in PARALLEL worker processes (fork; one task per area
    in pass 1, one task per node block in pass 2). Worker counts derive
    from a TOTAL memory budget (BANK_MEM_BUDGET_GB) minus the OS+parent
    reserve, divided by each pass's measured per-worker peak, then capped
    by cpu count — NOT one-per-core (16 uncapped SIFT workers OOM'd the
    32 GB machine; 12 still pushed it to 27 GB). Row order is NODE-MAJOR
    (config area order -> node -> layer; deterministic, each area seeds
    its own RNG), so rebuilds are reproducible regardless of worker
    scheduling and a location's temporal layers sit on adjacent rows.

    Output: patch_bank/bank.zarr (N,136,136,3 uint8, one zstd-compressed
    chunk file per crop — ~1.34x measured, ~49 GB instead of 65.8 GB raw,
    and the anchor/positive read hits adjacent chunks) + meta.npz
    (per-patch area/node/layer-file provenance, centers, train/val split,
    area names).
    """
    bank_dir = cfg.data.path.patch_bank
    os.makedirs(bank_dir, exist_ok=True)
    t = cfg.training
    areas = _area_entries()
    if not areas:
        raise RuntimeError('no area folders configured — set '
                           'data.path.trainingdata_bitmaps in train_config.yml')
    # sivl BitmapDataset grid geometry (on the NATIVE-resolution image —
    # no GSD resampling, exactly like sivl's BitmapDataset)
    margin = PATCH_PX / math.sqrt(2)             # sivl marginForRotation
    step = PATCH_PX + t.marginBetweenSamples_px  # 0 margin = adjacent samples
    g0 = PATCH_PX / 2 + margin                   # first centerpoint (sivl)

    # ---- pass 1 (PARALLEL, one task per area): SIFT + chained alignment
    # + node grids + total patch count --------------------------------------
    tasks = []
    for ai, (name, path, sp) in enumerate(areas):
        files = _image_files(path)
        if len(files) < 2:
            print(f'WARNING: [{name}] has {len(files)} temporal layers '
                  f'(need >= 2 for pairing) — SKIPPED')
            continue
        tasks.append((name, path, sp, ai,
                      sorted(files, key=_file_date),  # chronological: [-1] =
                      float(t.apShiftingStd), step, g0, margin))  # the NEWEST
    if not tasks:
        raise RuntimeError('no bank patches — check the area folders')
    nw = max(1, min(os.cpu_count() or 1, len(tasks),
                    int((BANK_MEM_BUDGET_GB - PARENT_SYSTEM_RESERVE_GB)
                        // ALIGN_WORKER_GB)))
    print(f'pass 1/2: SIFT + alignment of {len(tasks)} areas on {nw} '
          f'workers (budget {BANK_MEM_BUDGET_GB:.0f} GB, reserve '
          f'{PARENT_SYSTEM_RESERVE_GB:.0f} GB, {ALIGN_WORKER_GB:.1f} GB '
          f'per SIFT worker)')
    plans = []
    with ProcessPoolExecutor(max_workers=nw,
                             mp_context=mp.get_context('fork')) as ex:
        futs = [ex.submit(_bank_align_task, a) for a in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc='align'):
            plans.append(fut.result())
    plans.sort(key=lambda p: p['ai'])  # config area order = row order
    total = sum(len(p['nodes']) * len(p['files']) for p in plans)
    for p in plans:  # per-area alignment report (same format as before)
        t_of, unaligned = p['t_of'], p['unaligned']
        offs = ' '.join(
            f'{os.path.basename(f)[:22]}:({t_of[f][0] / ALIGN_RES:+.1f},'
            f'{t_of[f][1] / ALIGN_RES:+.1f})'
            for f in sorted(t_of, key=_file_date))
        print(f"{p['name']} [{p['sp']}]: "
              f"{len(p['files']) - len(unaligned)}/{len(p['files'])} "
              f"layers aligned ({len(unaligned)} kept unaligned), "
              f"{len(p['nodes'])} nodes, "
              f"{len(p['nodes']) * len(p['files'])} patches")
        print(f'   offsets full-res px vs newest: {offs}')
        if unaligned:
            print(f"   UNALIGNED (offset assumed 0,0): {unaligned}")
    print(f'bank total: {total} patches, '
          f'{total * BANK_PX * BANK_PX * 3 / 1e9:.1f} GB — writing ...')

    # ---- pass 2 (PARALLEL, one task per node block): warp + crop into the
    # NODE-MAJOR compressed zarr bank. Row layout: area (config order) ->
    # node -> layer, so all temporal layers of one location are contiguous
    # rows (train-time anchor/positive reads hit adjacent chunks instead of
    # random rows ~100k apart). Each row is one zstd chunk file (~40 KB);
    # tasks write disjoint rows = disjoint chunks, so parallel writes are
    # contention-free. Compression (1.34x measured) shrinks the bank from
    # 65.8 GB raw to ~49 GB — the page cache covers far more of it, which
    # is what actually stops the disk thrash during training.
    store = os.path.join(bank_dir, 'bank.zarr')
    if os.path.exists(store):
        shutil.rmtree(store)   # stale/failed build; real_pairs.npz untouched
    zarr.open(store, mode='w', shape=(total, BANK_PX, BANK_PX, 3),
              chunks=BANK_CHUNK, dtype='u1', compressor=BANK_CODEC)
    area_of, node_of, centers, split, file_of = [], [], [], [], []
    crop_tasks = []
    BLOCK = 512               # nodes per task (amortizes the L image loads)
    for p in plans:
        L = len(p['files'])
        tvs = [p['t_of'][f] / ALIGN_RES for f in p['files']]  # full-res px
        for b0 in range(0, len(p['nodes']), BLOCK):
            crop_tasks.append((p['name'], store, p['path'], p['files'], tvs,
                               p['ow'], p['oh'], p['nodes'][b0:b0 + BLOCK],
                               sum(len(q['nodes']) * len(q['files'])
                                   for q in plans[:p['ai']]) + b0 * L, L))
        for nid, (cx, cy) in enumerate(p['nodes']):   # node-major meta rows
            for fname in p['files']:
                area_of.append(p['ai'])
                node_of.append(nid)
                centers.append((cx, cy))
                split.append(p['sp'])
                file_of.append(fname)
    assert len(area_of) == total
    nw = max(1, min(os.cpu_count() or 1, len(crop_tasks),
                    int((BANK_MEM_BUDGET_GB - PARENT_SYSTEM_RESERVE_GB)
                        // CROP_WORKER_GB)))
    print(f'pass 2/2: warp + crop of {len(crop_tasks)} node blocks on {nw} '
          f'workers (budget {BANK_MEM_BUDGET_GB:.0f} GB, reserve '
          f'{PARENT_SYSTEM_RESERVE_GB:.0f} GB, {CROP_WORKER_GB:.2f} GB '
          f'per crop worker)')
    done = 0
    with ProcessPoolExecutor(max_workers=nw,
                             mp_context=mp.get_context('fork')) as ex:
        futs = [ex.submit(_bank_crop_task, ct) for ct in crop_tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc='crop'):
            done += fut.result()
    assert done == total
    np.savez(os.path.join(bank_dir, 'meta.npz'),
             area=np.array(area_of, np.int32),
             node=np.array(node_of, np.int32),
             centers=np.array(centers, np.int32),
             split=np.array(split),
             layer_file=np.array(file_of),
             area_names=np.array([a[0] for a in areas]))
    used = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(store) for f in fs)
    print(f'bank: ({total}, {BANK_PX}, {BANK_PX}, 3) zstd zarr, '
          f'{used / 1e9:.1f} GB on disk (raw would be '
          f'{total * BANK_PX * BANK_PX * 3 / 1e9:.1f} GB)')


# --------------------------------------------------------------- datasets
def _tensor(img):
    return torch.from_numpy(
        np.ascontiguousarray(img.transpose(2, 0, 1))).float()


def worker_init_fn(wid):
    np.random.seed(torch.initial_seed() % 2 ** 32)
    try:    # one decode thread per worker (no blosc pool oversubscription)
        from numcodecs import blosc
        blosc.use_threads = False
    except Exception:
        pass


class TemporalBankDataset(Dataset):
    """Multi-temporal simulated-pair dataset from the map patch bank (pretrain).

    sivl's BitmapDataset semantics on the bank (one bank node = one sivl
    centerpoint; a node's layers = sivl's aligned images of one area):

      anchor/positive = two DIFFERENT random temporal layers of this node
        (sivl: random.sample(images, 2)) at the SAME centerpoint, sharing
        one random rotation uniform(0,360) and one random scale ~N(1,
        scaleStd) — the positive adds sivl's pair noise:
        rotationErrorStd_deg, translationErrorStd_px,
        homographyCornerErrorStd_px. The anchor additionally gets the
        calibrated drone-appearance simulation (sim_drone) — this
        pipeline's stand-in for sivl's albumentations albTransform.
        Different-time pairs force time-invariant STRUCTURAL matching;
        same-layer pairs would drift to texture matching and break under
        seasonal/appearance change at deployment.
      negative = a random OTHER node of the same area, random layer,
        independent rotation and scale (sivl: random other centerpoint,
        random image).

    Scales are clamped to <= 1.0: bank crops are 136 px, so an upscaled
    rotated 96 px window would not fit (sivl crops from full images and
    could allow N(1,0.1) upscale draws).

    mode 'train': fresh random draws per call (like sivl). mode 'val'
    (held-out testing areas): deterministic per node.
    """

    def __init__(self, mode):
        self.mode = mode                          # 'train' or 'val'
        t = cfg.training
        self.rot_std = t.rotationErrorStd_deg
        self.trans_std = t.translationErrorStd_px
        self.corner_std = t.homographyCornerErrorStd_px
        self.scale_std = t.scaleStd
        self.patches = zarr.open(os.path.join(cfg.data.path.patch_bank,
                                              'bank.zarr'), mode='r')
        meta = np.load(os.path.join(cfg.data.path.patch_bank, 'meta.npz'),
                       allow_pickle=True)
        sel = np.where(meta['split'] == mode)[0]
        # group bank patches of one location (area, node) across layers
        key = meta['area'][sel].astype(np.int64) * (1 << 20) + meta['node'][sel]
        order = np.argsort(key, kind='stable')
        _, starts = np.unique(key[order], return_index=True)
        bounds = list(starts) + [len(order)]
        self.groups = []           # node -> global patch indices (layer order)
        for a, b in zip(bounds[:-1], bounds[1:]):
            g = sel[order[a:b]]
            if len(g) >= 2:        # a single surviving layer cannot form a pair
                self.groups.append(g)
        if not self.groups:
            raise RuntimeError(
                f'split "{mode}" has no multi-layer bank nodes — check '
                'data.path.trainingdata_bitmaps / testingdata_bitmaps and '
                'the --build-bank report')
        self.group_area = meta['area'][[g[0] for g in self.groups]]
        # per-area group indices (sivl negatives come from the same area)
        self.by_area = {}
        for gi, a in enumerate(self.group_area):
            self.by_area.setdefault(int(a), []).append(gi)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, gi):
        grp = self.groups[gi]
        if self.mode == 'train':
            rng = np.random.default_rng()
        else:                                      # deterministic val
            rng = np.random.default_rng(1000 + gi)
        pair = rng.choice(len(grp), 2, replace=False)   # DIFFERENT layers

        # sivl: anchor/positive share rotation + scale; positive adds small
        # rotation/translation/corner noise
        ap_rot = rng.uniform(0, 360)
        ap_scale = min(rng.normal(1.0, self.scale_std), 1.0)
        anchor = sim_drone(rot_crop(np.array(self.patches[grp[pair[0]]]),
                                    ap_rot, ap_scale), rng)
        pos = rot_crop(np.array(self.patches[grp[pair[1]]]),
                       ap_rot + rng.normal(0, self.rot_std), ap_scale,
                       corner_noise=self.corner_std,
                       shift=(rng.normal(0, self.trans_std),
                              rng.normal(0, self.trans_std)), rng=rng)

        # sivl: negative = random other centerpoint of this area, random
        # image, independent rotation + scale
        cands = self.by_area[int(self.group_area[gi])]
        for _ in range(20):
            gj = cands[int(rng.integers(0, len(cands)))]
            if gj != gi:
                break
        neg_grp = self.groups[gj]
        neg = rot_crop(np.array(
            self.patches[neg_grp[int(rng.integers(0, len(neg_grp)))]]),
            rng.uniform(0, 360),
            min(rng.normal(1.0, self.scale_std), 1.0))

        return (_tensor(anchor), _tensor(pos), _tensor(neg))


class FinetuneDataset(Dataset):
    """Mixed real + simulated training set (finetune stage).

    p_real of items: REAL drone anchor + GPS-truth map positive (rotation
    label from the label model + jitter; scale jitter covers coverage
    estimation error) + near/far/rotated negative cropped from the
    deployment map. Rest: TemporalBankDataset pairs (different-time
    same-location — keeps structural matching, false-peak rejection and
    yaw-ambiguity handling general).

    mode: 'train' (mixture), 'realval' (held-out real frames, deterministic).

    Real train/val split is by FLIGHT (finetune.val_flights in train_config):
    every frame of a held-out flight is validation, so realval tests
    cross-flight generalization. An in-flight every-5th-frame split leaks:
    frames 0.2 s apart over a 96 m footprint are near-duplicates with the
    same lighting/terrain, so realval measures memorization of the training
    flight (a fine-tune on bag 184004 scored realval 1.000 yet truth 0.003 on
    the held-out 190020 flight, 2026-09-09). Falls back to the every-5th-frame
    split when val_flights is empty or the npz predates flight tags.
    """

    def __init__(self, mode, p_real=None):
        self.mode = mode
        self.p_real = cfg.finetune.p_real if p_real is None else p_real
        self.sim = TemporalBankDataset('train' if mode == 'train' else 'val')
        d = np.load(os.path.join(cfg.data.path.patch_bank, 'real_pairs.npz'))
        self.patches = d['patches']
        self.px, self.py = d['px'], d['py']
        self.rot, self.cov = d['rot'], d['cov']
        if 'flight' in d.files:
            self.flight = d['flight'].astype(str)
        else:
            self.flight = np.full(len(self.patches), 'flight0')
        val_flights = set(getattr(cfg.finetune, 'val_flights', []) or [])
        if val_flights:
            is_val = np.array([f in val_flights for f in self.flight])
            missing = val_flights - set(self.flight)
            if missing:
                print(f'  WARNING: val_flights {sorted(missing)} not in '
                      f'real_pairs.npz (flights present: '
                      f'{sorted(set(self.flight))})')
        else:
            is_val = np.arange(len(self.patches)) % 5 == 4   # legacy in-flight
        self.real_idx = (np.where(~is_val)[0] if mode != 'realval'
                         else np.where(is_val)[0])
        if mode == 'realval':
            vf = sorted(set(self.flight[self.real_idx]))
            print(f'  realval: {len(self.real_idx)} frames from flights {vf}')
            if len(self.real_idx) == 0:
                print('  WARNING: realval is EMPTY — check val_flights names')
        self.map_rgb = cv2.cvtColor(
            cv2.imread(cfg.data.path.map.path, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB)
        self.map_gsd = cfg.data.path.map.gsd
        t = cfg.training
        self.p_near, self.p_far = t.negativeMixture.near, t.negativeMixture.far
        self.near_min, self.near_max = t.nearNegativeMin_px, t.nearNegativeMax_px

    def __len__(self):
        return len(self.sim) if self.mode != 'realval' else len(self.real_idx)

    def _crop(self, px, py, rot_deg, src_dim, scale=1.0):
        return cropRectangularSectionFromImage(
            self.map_rgb, px, py, math.radians(rot_deg), src_dim,
            PATCH_PX, scale)

    def _real(self, ri, rng, train):
        patch = self.patches[ri]
        px, py = self.px[ri], self.py[ri]
        rot, src_dim = self.rot[ri], self.cov[ri] / self.map_gsd
        # positive: truth map patch; rotation/scale jitter in train mode
        pr = rot + (rng.normal(0, cfg.finetune.rotationErrorStd_deg)
                    if train else 2.0)
        ps = (rng.uniform(cfg.finetune.positiveScaleMin,
                          cfg.finetune.positiveScaleMax)
              if train else 1.0)
        pos = self._crop(px, py, pr, src_dim, ps)
        # negative: near ring / far / rotated-self (same mixture as sim)
        r = rng.random()
        h, w = self.map_rgb.shape[:2]
        if r < self.p_near:
            for _ in range(20):
                ang, dd = rng.uniform(0, 2 * np.pi), rng.uniform(
                    self.near_min, self.near_max)
                qx, qy = px + np.cos(ang) * dd, py + np.sin(ang) * dd
                if 100 < qx < w - 100 and 100 < qy < h - 100:
                    break
            neg = self._crop(qx, qy, rng.uniform(0, 360), src_dim,
                             rng.uniform(0.75, 1.1))
            if neg.mean() < 5.0:                    # OOB black — rotated-self
                neg = self._crop(px, py, rot + 90 * int(rng.integers(1, 4)),
                                 src_dim)
        elif r < self.p_near + self.p_far:
            neg = self._crop(rng.uniform(300, w - 300), rng.uniform(300, h - 300),
                             rng.uniform(0, 360), src_dim,
                             rng.uniform(0.75, 1.1))
        else:
            neg = self._crop(px, py, rot + 90 * int(rng.integers(1, 4)),
                             src_dim)
        return (_tensor(patch), _tensor(pos), _tensor(neg))

    def __getitem__(self, li):
        if self.mode == 'realval':
            rng = np.random.default_rng(5000 + int(self.real_idx[li]))
            return self._real(int(self.real_idx[li]), rng, train=False)
        rng = np.random.default_rng()
        if rng.random() < self.p_real:
            ri = int(self.real_idx[rng.integers(0, len(self.real_idx))])
            return self._real(ri, rng, train=True)
        return self.sim[li]


# --------------------------------------------------------------- real pairs
def build_real_pairs():
    """Pair every debug-saved real drone patch with its GPS-truth map patch.

    Per flight (cfg.data.path.flights): clock-align the ulog with the bag,
    map GPS truth into deployment-map pixels via geo_offset, then score the
    truth location over a 2-deg rotation sweep with the label model — its
    argmax (refined to 0.5 deg) is the rotation pseudo-label. Frames the
    label model cannot match (<0.5) are dropped (bad geometry/label).
    Frames below the flight's alt_min_m (takeoff/landing, too close to the
    ground for usable orthoprojections) are skipped.
    """
    map_rgb = cv2.cvtColor(cv2.imread(cfg.data.path.map.path, cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
    gsd = cfg.data.path.map.gsd
    geo = cfg.data.path.map.geo_offset
    angles = np.arange(180) * 2.0
    sc = Scorer(cfg.data.path.label_model_checkpoint)
    keep = []
    for fl in cfg.data.path.flights:
        alt_min = getattr(fl, 'alt_min_m', 30.0)
        row_of_k, truth_px, cov_all, alt_all = flight_truth(fl, geo)
        files = sorted(glob.glob(os.path.join(fl.run_dir, 'debug',
                                              'k*_4_patch.png')))
        matched = sum(1 for f in files
                      if int(os.path.basename(f)[1:6]) in row_of_k)
        if files and matched < 0.9 * len(files):
            raise RuntimeError(
                f'[{fl.name}] only {matched}/{len(files)} debug frames match '
                f'the replay log — run_dir was likely overwritten by a later '
                f'MCL run (its log no longer pairs with the debug patches). '
                f'Re-run MCL with save_debug into a FRESH out_dir and point '
                f'the flight entry at it.')
        n_alt = 0
        for f in files:
            k = int(os.path.basename(f)[1:6])
            if k not in row_of_k:
                continue
            row = row_of_k[k]
            if alt_all[row] < alt_min:
                n_alt += 1
                continue
            patch = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            px, py = truth_px[row]
            src_dim = cov_all[row] / gsd
            tp = extract_all(map_rgb, px, py, src_dim, angles)
            s = sc.score_many(patch, tp)
            s = np.where(tp.reshape(len(tp), -1).mean(1) >= 5.0, s, 0.0)
            i = int(np.argmax(s))
            fine = np.arange(angles[i] - 4.0, angles[i] + 4.5, 0.5)
            sf = sc.score_many(patch, extract_all(map_rgb, px, py, src_dim, fine))
            j = int(np.argmax(sf))
            score = max(float(s[i]), float(sf[j]))
            if score < 0.5:
                print(f'  [{fl.name}] k={k}: label-model truth score '
                      f'{score:.3f} < 0.5 — drop')
                continue
            keep.append((fl.name, k, patch, px, py, float(fine[j]), score,
                         float(cov_all[row])))
        print(f'[{fl.name}] {len(files)} debug frames: '
              f'{n_alt} below alt_min_m={alt_min} (takeoff/landing) skipped')
    np.savez(os.path.join(cfg.data.path.patch_bank, 'real_pairs.npz'),
             flight=np.array([r[0] for r in keep]),
             k=np.array([r[1] for r in keep]),
             patches=np.stack([r[2] for r in keep]),
             px=np.array([r[3] for r in keep], np.float32),
             py=np.array([r[4] for r in keep], np.float32),
             rot=np.array([r[5] for r in keep], np.float32),
             label_score=np.array([r[6] for r in keep], np.float32),
             cov=np.array([r[7] for r in keep], np.float32))
    print(f'real pairs: {len(keep)} kept '
          f'-> {os.path.join(cfg.data.path.patch_bank, "real_pairs.npz")}')


# ------------------------------------------------------------------ train
def getTrainingAndTestingDatasets(finetune=False, printStatistics=True):
    """Training dataset + named validation datasets, like the original
    getTrainingAndTestingDatasets (but validation is a list because the
    finetune stage tracks both sim and real metrics)."""
    if finetune:
        trainingdataset = FinetuneDataset('train')
        testingdatasets = [('simval', TemporalBankDataset('val')),
                           ('realval', FinetuneDataset('realval'))]
    else:
        trainingdataset = TemporalBankDataset('train')
        testingdatasets = [('simval', TemporalBankDataset('val'))]
    if printStatistics:
        print("Number of training samples: {}".format(len(trainingdataset)))
        for name, ds in testingdatasets:
            print("Number of testing samples ({}): {}".format(name, len(ds)))
    return trainingdataset, testingdatasets


def evaluate(branchmodel, decisionmodel, loader, device):
    """One testing pass (the original testing loop) + ranking metrics."""
    branchmodel.eval()
    decisionmodel.eval()
    criterion = torch.nn.BCELoss(reduction='none')
    tot_loss, n, correct = 0.0, 0, 0
    pos_scores, neg_scores = [], []
    with torch.no_grad():
        for anchorImage, positiveImage, negativeImage in loader:
            anchorImage = anchorImage.to(device)
            positiveImage = positiveImage.to(device)
            negativeImage = negativeImage.to(device)
            anc = branchmodel(anchorImage)
            pos = branchmodel(positiveImage)
            neg = branchmodel(negativeImage)
            ancToPos = decisionmodel(anc, pos)
            ancToNeg = decisionmodel(anc, neg)
            samples = torch.cat((ancToPos, ancToNeg)).squeeze(-1)
            targets = torch.cat((torch.ones_like(ancToPos.squeeze(-1)),
                                 torch.zeros_like(ancToNeg.squeeze(-1))))
            loss = criterion(samples, targets)
            numSamplesInBatch = anchorImage.shape[0]
            tot_loss += loss.mean().item() * numSamplesInBatch
            n += numSamplesInBatch
            correct += (samples[:len(targets) // 2]
                        > samples[len(targets) // 2:]).sum().item()
            pos_scores.append(samples[:len(targets) // 2].cpu())
            neg_scores.append(samples[len(targets) // 2:].cpu())
    pos = torch.cat(pos_scores)
    neg = torch.cat(neg_scores)
    return (tot_loss / max(n, 1), correct / max(len(pos), 1),
            float(pos.mean()), float(neg.mean()))


def main(args):
    finetune = args.finetune
    tcfg = cfg.finetune if finetune else cfg.training
    if args.experiment_name:
        tcfg = SimpleNamespace(**{**vars(tcfg),
                                  'experiment_name': args.experiment_name})
    numEpochs = args.epochs or tcfg.numEpochs

    device = torch.device(cfg.device)
    print("device={}".format(device))

    trainingdataset, testingdatasets = getTrainingAndTestingDatasets(finetune)
    trainingloader = DataLoader(trainingdataset,
                                batch_size=tcfg.batchsize, shuffle=True,
                                num_workers=tcfg.num_workers,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn,
                                drop_last=True,
                                persistent_workers=tcfg.num_workers > 0)
    testingloaders = [DataLoader(ds, batch_size=tcfg.batchsize, num_workers=2)
                      for _, ds in testingdatasets]
    testingloader_descriptions = [name for name, _ in testingdatasets]

    # Initialize model
    branchmodel = BranchNet().to(device)
    decisionmodel = DecisionNet().to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        list(branchmodel.parameters()) + list(decisionmodel.parameters()),
        lr=tcfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)

    training_losses_per_epoch = []
    testing_losses_per_epoch = [[] for _ in testingloaders]
    testing_metrics_per_epoch = [[] for _ in testingloaders]
    first_epoch = 0

    # If a checkpoint for training was specified, use it for initializing
    # the model and optimizer (like the original script).
    print("initial model checkpoint: {}".format(tcfg.initial_model_checkpoint))
    if (tcfg.initial_model_checkpoint
            and os.path.isfile(tcfg.initial_model_checkpoint)):
        checkpoint = torch.load(tcfg.initial_model_checkpoint,
                                map_location=device)
        branchmodel.load_state_dict(checkpoint['branchmodel_state_dict'])
        decisionmodel.load_state_dict(checkpoint['decisionmodel_state_dict'])

        # If we want to continue from an interrupted training session, load
        # also optimizer, epoch and loss history.
        if not tcfg.use_only_network_params:
            print("Loading optimizer state dict and epoch history")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            first_epoch = checkpoint['epoch'] + 1
            training_losses_per_epoch = checkpoint.get(
                'training_losses_per_epoch', [])
            testing_losses_per_epoch = checkpoint.get(
                'testing_losses_per_epoch', [[] for _ in testingloaders])
            testing_metrics_per_epoch = checkpoint.get(
                'testing_metrics_per_epoch', [[] for _ in testingloaders])
        print("file loaded")
    else:
        print("file not found or not set, starting with default initialization")

    # Initialize criterion for loss
    criterion = torch.nn.BCELoss(reduction='none')
    os.makedirs(tcfg.checkpoint_saving_path, exist_ok=True)

    for epoch in range(first_epoch, numEpochs):

        # Training loop
        branchmodel.train()
        decisionmodel.train()

        cumulative_loss_alltrls = 0
        cumulative_samples_alltrls = 0

        pbar = tqdm(enumerate(trainingloader), total=len(trainingloader),
                    desc="Epoch {}/{}".format(epoch + 1, numEpochs),
                    leave=True, mininterval=2, miniters=10)
        for i, data in pbar:
            if args.max_batches and i >= args.max_batches:
                break

            (anchorImage, positiveImage, negativeImage) = data
            anchorImage = anchorImage.to(device)
            positiveImage = positiveImage.to(device)
            negativeImage = negativeImage.to(device)

            optimizer.zero_grad()

            anc = branchmodel(anchorImage)
            pos = branchmodel(positiveImage)
            neg = branchmodel(negativeImage)

            ancToPos = decisionmodel(anc, pos)
            ancToNeg = decisionmodel(anc, neg)

            samples = torch.cat((ancToPos, ancToNeg))
            targets = torch.cat((torch.ones_like(ancToPos),
                                 torch.zeros_like(ancToNeg)))

            loss = criterion(samples, targets)

            numSamplesInBatch = anchorImage.shape[0]
            cumulative_loss_alltrls += loss.mean().item() * numSamplesInBatch
            cumulative_samples_alltrls += numSamplesInBatch

            loss.mean().backward()
            optimizer.step()

            pbar.set_postfix({
                "Batch": f"{i + 1}/{len(trainingloader)}",
                "train_loss":
                    f"{cumulative_loss_alltrls / cumulative_samples_alltrls:.4f}",
            })

        training_losses_per_epoch.append(
            cumulative_loss_alltrls / max(cumulative_samples_alltrls, 1))

        # Testing loop with all testing loaders
        branchmodel.eval()
        decisionmodel.eval()

        metrics_line = []
        for idx, testingloader in enumerate(testingloaders):
            l, acc, mpos, mneg = evaluate(branchmodel, decisionmodel,
                                          testingloader, device)
            testing_losses_per_epoch[idx].append(l)
            testing_metrics_per_epoch[idx].append((acc, mpos, mneg))
            metrics_line.append(
                "{} loss {:.4f} acc {:.3f} p {:.3f} n {:.3f}".format(
                    testingloader_descriptions[idx], l, acc, mpos, mneg))
        print(" | ".join(metrics_line), flush=True)

        # Only save every Nth epoch state on disk (as determined in
        # configuration) or the output of the last training epoch.
        if ((epoch + 1) % tcfg.save_every == 0
                or epoch == numEpochs - 1):
            ckptFilename = "{}_epoch_{:03d}_of_{}.pt".format(
                tcfg.experiment_name, epoch + 1, numEpochs)
            ckptFilenameWithPath = os.path.join(
                tcfg.checkpoint_saving_path, ckptFilename)
            torch.save({
                'epoch': epoch,
                'branchmodel_state_dict': branchmodel.state_dict(),
                'decisionmodel_state_dict': decisionmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_losses_per_epoch': training_losses_per_epoch,
                'testing_losses_per_epoch': testing_losses_per_epoch,
                'testing_metrics_per_epoch': testing_metrics_per_epoch,
                'testing_loss_descriptions': testingloader_descriptions,
            }, ckptFilenameWithPath)
            print("Saved state to {}".format(ckptFilenameWithPath))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build-bank', action='store_true',
                    help='build the multi-temporal map patch bank from the '
                         'configured area folders (align + resample + crop)')
    ap.add_argument('--build-real', action='store_true',
                    help='build real_pairs.npz (real drone patches + '
                         'GPS-truth map pairs, rotation-labeled)')
    ap.add_argument('--finetune', action='store_true',
                    help='train on the mixed real+simulated set '
                         '(default: pretrain on the map bank)')
    ap.add_argument('--epochs', type=int, default=0,
                    help='override the configured number of epochs')
    ap.add_argument('--max-batches', type=int, default=0,
                    help='debug: truncate each epoch')
    ap.add_argument('--experiment-name', default='',
                    help='override the configured experiment name')
    args = ap.parse_args()

    if args.build_bank:
        build_bank()
    elif args.build_real:
        build_real_pairs()
    else:
        main(args)
