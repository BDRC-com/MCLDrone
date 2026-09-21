"""MCL v1 — particle filter for GNSS-denied UAV localization.

Reuses the SIVL similarity model (BranchNet + DecisionNet) and
cropRectangularSectionFromImage for per-particle map-patch extraction.
Full predict -> update -> resample -> roughen loop, with orthoprojection
preprocessing of camera frames (see make_drone_patch / MCL.set_drone_image).

Run the trajectory test:
    ~/anaconda3/envs/sivl/bin/python mcl.py
"""

import os
import sys
import numpy as np
import cv2
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared RNG so runs are reproducible via set_seed() (bag_mcl --seed).
_rng = np.random.default_rng()


def set_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)

# Paths are resolved relative to this file so the deployed folder is portable
# (map, checkpoint and sivl/ ship inside it) — no absolute dev paths.
_HERE = os.path.dirname(os.path.abspath(__file__))

SIVL_DIR = os.path.join(_HERE, "sivl")
sys.path.insert(0, SIVL_DIR)
sys.path.insert(0, _HERE)
from utils.utils import cropRectangularSectionFromImage, numpyImageToTensor  # noqa: E402
from models.orthosimilarity import BranchNet, DecisionNet  # noqa: E402
from orthoprojection import orthoproject, extract_patch  # noqa: E402

MAP_PATH = os.path.join(_HERE, "maps", "z17_5120.png")
DRONE_IMAGE_PATH = os.path.join(_HERE, "09_frame_t68.png")   # demo frame only
OUTPUT_DIR = os.path.join(_HERE, "mcl_runs")
# Default similarity model: Huizhou map pretrain + real-pair fine-tune
# (eval on bag_0001 real patches: truth 1.000 vs false peaks ~0, win 100%;
# the original Sweden model scores false peaks 0.96 — the false-lock cause;
# the pretrain-ONLY model misses truth (med 0.000) AND fires ~1.0 on rare
# false sites — never deploy it, it is only an init for --finetune)
CHECKPOINT = os.path.join(_HERE, "checkpoints_huizhou",
                          "huizhou_ft_mt_1_epoch_040_of_40.pt")

PATCH_PX = 96            # CNN input size
COVERAGE_M = 96.0        # ground coverage of each patch
DRONE_GSD = 0.22         # verified drone GSD at 100m (m/px)
MAP_GSD = 1.10           # z17 at lat 23 (m/px)


def load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_centered(image, cx, cy, size_px):
    """Crop a square of size_px centered on (cx, cy). Reflect-pad if out of bounds."""
    h, w = image.shape[:2]
    half = size_px // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size_px, y0 + size_px
    pad_t, pad_l = max(0, -y0), max(0, -x0)
    pad_b, pad_r = max(0, y1 - h), max(0, x1 - w)
    if pad_t or pad_l or pad_b or pad_r:
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_REFLECT_101)
        x0 += pad_l
        y0 += pad_t
    return image[y0:y0 + size_px, x0:x0 + size_px]


def make_drone_patch(image, altitude, roll, pitch, fx, fy, cx, cy,
                     size_m=COVERAGE_M, resolution=1.0):
    """Orthoproject a camera frame and extract the model-input patch.

    This is the standard drone-side preprocessing: camera image + altitude
    + IMU roll/pitch -> top-down patch at model resolution (1 m/px).

    Returns (patch 96x96x3, patch_mask, center_ground) where center_ground
    is the patch center in camera-relative ground meters (x=right, y=down in
    image); (0, 0) = directly below the drone. A non-zero value means the
    patch was shifted (tilt + footprint clamp), so the observation is of
    that ground point, not of the drone's position.
    """
    ortho, mask, extent = orthoproject(
        image, altitude, roll, pitch, fx, fy, cx, cy, resolution)
    return extract_patch(ortho, mask, extent, size_m, resolution)


class MCL:
    """Particle filter with state (x, y, yaw, scale) in map-frame meters/rad."""

    def __init__(self, map_rgb, gsd_map, center_px, checkpoint_path,
                 device="cuda:0"):
        self.map_rgb = map_rgb
        self.gsd_map = gsd_map
        self.cx_px, self.cy_px = center_px
        self.src_dim = COVERAGE_M / gsd_map   # patch footprint in map px at scale 1.0
        self.coverage_m = COVERAGE_M          # current patch ground coverage [m]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"  device: {self.device}")

        self.branch = BranchNet().to(self.device)
        self.decision = DecisionNet().to(self.device)
        print(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.branch.load_state_dict(ckpt["branchmodel_state_dict"])
        self.decision.load_state_dict(ckpt["decisionmodel_state_dict"])
        self.branch.eval()
        self.decision.eval()
        self.drone_feat = None

    def set_coverage(self, coverage_m):
        """Set the map-patch ground coverage [m] to match the drone patch.

        Altitude-adaptive coverage: when flying below the altitude that
        gives a 96 m footprint, the drone patch shrinks to the visible
        ground and map patches must cover the same meters for the CNN
        comparison to stay consistent (verified by coverage_sweep_test.py:
        the model tolerates 32-96 m coverage with >= 0.94 true-match
        scores and large margins over wrong locations).
        """
        self.coverage_m = coverage_m
        self.src_dim = coverage_m / self.gsd_map

    def set_drone_patch(self, patch_rgb):
        """patch_rgb: 96x96x3. Compute drone branch features ONCE."""
        t = numpyImageToTensor(patch_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.drone_feat = self.branch(t)   # (1, 16384)

    def set_drone_image(self, image, altitude, roll, pitch, fx, fy, cx, cy):
        """Full drone-side preprocessing: orthoproject + extract patch.

        The patch is centered on the nadir (0,0) unless the footprint clamp
        shifted it; the shift is stored in self.patch_center_ground.
        """
        patch, _, center_ground = make_drone_patch(
            image, altitude, roll, pitch, fx, fy, cx, cy)
        self.patch_center_ground = center_ground
        if np.hypot(*center_ground) > 20.0:
            print(f"  WARNING: patch shifted {np.hypot(*center_ground):.0f} m "
                  f"from nadir (tilt/footprint clamp); observation is of the "
                  f"ground at that offset, not of the drone position")
        self.set_drone_patch(patch)

    def init_particles(self, x0, y0, yaw0, scale0, std, N=200):
        rng = _rng
        return {
            "x":     rng.normal(x0, std[0], N).astype(np.float32),
            "y":     rng.normal(y0, std[1], N).astype(np.float32),
            "yaw":   rng.normal(yaw0, std[2], N).astype(np.float32),
            "scale": np.clip(rng.normal(scale0, std[3], N), 0.7, 1.3).astype(np.float32),
            "w":     np.full(N, 1.0 / N, dtype=np.float32),
        }

    def state_to_pixel(self, x, y):
        px = self.cx_px + x / self.gsd_map
        py = self.cy_px - y / self.gsd_map
        return px, py

    def score_particles(self, particles):
        N = particles["x"].shape[0]
        patches = np.empty((N, PATCH_PX, PATCH_PX, 3), dtype=np.uint8)
        for i in range(N):
            px, py = self.state_to_pixel(particles["x"][i], particles["y"][i])
            patches[i] = cropRectangularSectionFromImage(
                self.map_rgb, px, py, -particles["yaw"][i],
                self.src_dim, PATCH_PX, particles["scale"][i])
        # NO /255 — match numpyImageToTensor (model trained on 0-255 float range)
        t = torch.from_numpy(patches.astype(np.float32)).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            map_feat = self.branch(t)                                   # (N, 16384)
            scores = self.decision(self.drone_feat.expand(N, -1), map_feat).squeeze(-1)
        scores = scores.cpu().numpy()
        # Edge-particle guard: black (OOB) patches score arbitrarily -> zero them
        means = patches.reshape(N, -1).mean(axis=1)
        scores = np.where(means < 5.0, 0.0, scores)
        return scores

    @staticmethod
    def update_weights(scores, T=0.05):
        s = scores.astype(np.float64)
        w = np.exp((s - s.max()) / T)
        w /= w.sum()
        return w.astype(np.float32)

    @staticmethod
    def effective_sample_size(particles):
        """N_eff = 1 / sum(w^2). 1.0 for uniform weights, -> 0 when collapsed."""
        w = particles["w"].astype(np.float64)
        return 1.0 / np.sum(w * w)

    @staticmethod
    def resample(particles):
        w = particles["w"]
        N = w.shape[0]
        positions = (np.arange(N) + _rng.uniform()) / N
        cumsum = np.cumsum(w)
        cumsum[-1] = 1.0  # guard against float drift
        idx = np.clip(np.searchsorted(cumsum, positions), 0, N - 1)
        return {k: v[idx] for k, v in particles.items()}

    @staticmethod
    def roughen(particles, std, decay=1.0):
        """Add Gaussian jitter after resampling to maintain diversity (prevents
        particle deprivation / premature collapse). `decay` anneals the jitter
        over iterations (explore early, converge late)."""
        s = np.array(std, dtype=np.float32) * decay
        rng = _rng
        out = dict(particles)
        out["x"] = particles["x"] + rng.normal(0, s[0], particles["x"].shape).astype(np.float32)
        out["y"] = particles["y"] + rng.normal(0, s[1], particles["y"].shape).astype(np.float32)
        out["yaw"] = particles["yaw"] + rng.normal(0, s[2], particles["yaw"].shape).astype(np.float32)
        out["scale"] = np.clip(particles["scale"] + rng.normal(0, s[3], particles["scale"].shape).astype(np.float32), 0.7, 1.3)
        return out

    @staticmethod
    def estimate(particles):
        w = particles["w"]
        x = np.sum(w * particles["x"])
        y = np.sum(w * particles["y"])
        scale = np.sum(w * particles["scale"])
        # circular mean for yaw
        c = np.sum(w * np.cos(particles["yaw"]))
        s = np.sum(w * np.sin(particles["yaw"]))
        yaw = np.arctan2(s, c)
        spread = np.sqrt(np.sum(w * (particles["x"] - x) ** 2)
                         + np.sum(w * (particles["y"] - y) ** 2))
        return {"x": x, "y": y, "yaw": yaw, "scale": scale, "spread": spread}

    @staticmethod
    def predict(particles, dx, dy, dyaw, noise_std):
        """Motion model: propagate particles by a body-frame delta (dx, dy, dyaw).

        (dx, dy) are in the body frame (x=forward, y=left). Rotated into the
        world frame by each particle's yaw. Adds zero-mean Gaussian process
        noise scaled by `noise_std` = (sx, sy, syaw, sscale) in meters/rad.
        """
        yaw = particles["yaw"]
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        rng = _rng
        out = dict(particles)
        out["x"] = (particles["x"] + dx * cos_y - dy * sin_y
                    + rng.normal(0, noise_std[0], yaw.shape).astype(np.float32))
        out["y"] = (particles["y"] + dx * sin_y + dy * cos_y
                    + rng.normal(0, noise_std[1], yaw.shape).astype(np.float32))
        out["yaw"] = (particles["yaw"] + dyaw
                      + rng.normal(0, noise_std[2], yaw.shape).astype(np.float32))
        out["scale"] = np.clip(
            particles["scale"] + rng.normal(0, noise_std[3], yaw.shape).astype(np.float32),
            0.7, 1.3)
        return out

    def step(self, particles, T=0.05, roughen_std=None, decay=1.0,
             motion=None, noise_std=None, resample_eff=0.5,
             accumulate=True, skip_update=False):
        """One filter step: predict -> update -> (gated) resample -> roughen.

        Temporal filtering: with `accumulate=True` the per-frame likelihood
        MULTIPLIES the existing weights instead of replacing them. A false
        peak can win a single frame, but only the hypothesis consistent
        with every observation (the true track) keeps a high score across
        frames as the drone moves — so accumulated weights discriminate
        them. Resampling is gated by the effective sample size
        (N_eff / N < resample_eff) to avoid collapsing the cloud onto a
        one-frame false peak before the drone has moved.

        With `skip_update=True` (fast-rotation gate): only the prediction
        runs. Particles follow the VIO motion, but no measurement is
        applied — during fast rotation the yaw estimate lags, map patches
        are extracted at wrong rotations, and the resulting garbage scores
        would poison the accumulated weights (observed: false re-locks
        right after spins). Returns scores=None.
        """
        # Predict (motion model) — optional: motion=(dx,dy,dyaw) body-frame
        if motion is not None and noise_std is not None:
            particles = self.predict(particles, *motion, noise_std)
        if skip_update:
            est = self.estimate(particles)
            self.last_neff = self.effective_sample_size(particles)
            self.last_scored = None
            return particles, est, None
        # Update (measurement model)
        scores = self.score_particles(particles)
        self.last_scored = particles    # scored set (for debug visualization)
        like = self.update_weights(scores, T)
        particles = dict(particles)
        if accumulate:
            w = particles["w"].astype(np.float64) * like
            particles["w"] = (w / w.sum()).astype(np.float32)
        else:
            particles["w"] = like
        est = self.estimate(particles)
        # Resample only when the weights have degenerated (temporal
        # accumulation keeps the cloud diverse across early frames)
        neff = self.effective_sample_size(particles)
        self.last_neff = neff
        if neff < resample_eff * particles["w"].shape[0]:
            particles = self.resample(particles)
            if roughen_std is not None:
                particles = self.roughen(particles, roughen_std, decay)
        return particles, est, scores


def run_convergence_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    map_rgb = load_rgb(MAP_PATH)
    h, w = map_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    print(f"Map: {w}x{h}px  center=({cx},{cy})  gsd={MAP_GSD} m/px")

    mcl = MCL(map_rgb, MAP_GSD, (cx, cy), CHECKPOINT)

    # Drone patch: crop 436px (=96m at GSD 0.22) at the known center, resize to 96
    drone_img = load_rgb(DRONE_IMAGE_PATH)
    drone_crop_px = int(round(COVERAGE_M / DRONE_GSD))   # 436
    drone_patch = crop_centered(drone_img, 386, 231, drone_crop_px)
    drone_patch = cv2.resize(drone_patch, (PATCH_PX, PATCH_PX),
                            interpolation=cv2.INTER_AREA)
    mcl.set_drone_patch(drone_patch)
    print(f"Drone patch: crop {drone_crop_px}px -> {PATCH_PX}px (GSD {DRONE_GSD} m/px)")

    # True pose (drone patch maps to map pixel 2700, 2734)
    x_true = (2700 - cx) * MAP_GSD
    y_true = (cy - 2734) * MAP_GSD
    print(f"True pose: x={x_true:.1f}m  y={y_true:.1f}m  yaw=0  scale=1.0")

    # Initial guess offset from true to test convergence.
    # SMALL offset (30m): validates the filter tracks to truth when initialized well.
    # (A large offset fails on a SINGLE frame because the similarity function has
    #  false peaks that outscore the true location at yaw=0 — this is the "warm-up
    #  distance" limitation the paper describes; the motion model over multiple
    #  frames is what disambiguates them. See mcl_grid_scan.py for the score map.)
    x0, y0, yaw0, scale0 = x_true + 30, y_true + 30, 0.1, 1.0
    std = (20.0, 20.0, 0.15, 0.03)         # tight spread around a good guess
    N, K, T = 500, 15, 0.005              # sharp T: scores are flat (0.85-0.99)
    roughen_std = (6.0, 6.0, 0.04, 0.01)   # jitter after resampling (annealed)
    print(f"Init: x0={x0:.1f} y0={y0:.1f} yaw0={yaw0}  std={std}  N={N} K={K} T={T}")
    print(f"Roughen std: {roughen_std} (annealed)")

    particles = mcl.init_particles(x0, y0, yaw0, scale0, std, N)
    est0 = mcl.estimate(particles)
    print(f"  iter  0: x={est0['x']:.1f} y={est0['y']:.1f} yaw={est0['yaw']:+.3f} "
          f"scale={est0['scale']:.3f} spread={est0['spread']:.1f}m")

    ests = [est0]
    for k in range(1, K + 1):
        decay = max(0.15, 1.0 - 0.6 * k / K)   # anneal: explore early, converge late
        particles, est, scores = mcl.step(particles, T, roughen_std, decay)
        ests.append(est)
        if k % 3 == 0 or k == K:
            err = np.hypot(est["x"] - x_true, est["y"] - y_true)
            print(f"  iter {k:2d}: x={est['x']:.1f} y={est['y']:.1f} "
                  f"yaw={est['yaw']:+.3f} scale={est['scale']:.3f} "
                  f"spread={est['spread']:.1f}m  err={err:.1f}m  decay={decay:.2f}  "
                  f"score[min={scores.min():.3f} max={scores.max():.3f} mean={scores.mean():.3f}]")

    final = ests[-1]
    err = np.hypot(final["x"] - x_true, final["y"] - y_true)
    yaw_err = abs(np.arctan2(np.sin(final["yaw"]), np.cos(final["yaw"])))
    print("\n" + "=" * 50)
    print(f"Final xy error:      {err:.1f} m  (target < 30)")
    print(f"Final spread:        {final['spread']:.1f} m  (target < 50)")
    print(f"Final yaw error:     {yaw_err:.3f} rad  (target < 0.3)")
    print(f"Final scale error:   {abs(final['scale'] - 1.0):.3f}  (target < 0.05)")
    print("=" * 50)

    visualize(mcl, particles, ests, x_true, y_true, drone_patch, scores)
    return final


def visualize(mcl, particles, ests, x_true, y_true, drone_patch, scores):
    # Downsample map for display
    disp = cv2.pyrDown(cv2.pyrDown(mcl.map_rgb))
    scale = 0.25

    def to_disp(x, y):
        px, py = mcl.state_to_pixel(x, y)
        return px * scale, py * scale

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: particles on map
    ax = axes[0]
    ax.imshow(disp)
    px, py = to_disp(particles["x"], particles["y"])
    sc = ax.scatter(px, py, c=particles["w"], s=8, cmap="hot", alpha=0.6)
    tx, ty = to_disp(x_true, y_true)
    ax.plot(tx, ty, "g+", ms=18, mew=2, label="true")
    ex, ey = to_disp(ests[-1]["x"], ests[-1]["y"])
    ax.plot(ex, ey, "c^", ms=12, label="estimate")
    ax.set_title("Particles on map (after resampling)")
    ax.legend(loc="upper right")
    ax.axis("off")

    # Middle: convergence over iterations
    ax = axes[1]
    errs = [np.hypot(e["x"] - x_true, e["y"] - y_true) for e in ests]
    spreads = [e["spread"] for e in ests]
    ax.plot(errs, "o-", label="xy error (m)")
    ax.plot(spreads, "s-", label="spread (m)")
    ax.set_xlabel("iteration")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: drone patch vs estimated best-match map patch
    ax = axes[2]
    est = ests[-1]
    px, py = mcl.state_to_pixel(est["x"], est["y"])
    map_patch = cropRectangularSectionFromImage(
        mcl.map_rgb, px, py, -est["yaw"], mcl.src_dim, PATCH_PX, est["scale"])
    pair = np.hstack([drone_patch, np.ones((PATCH_PX, 4, 3), dtype=np.uint8) * 255,
                      map_patch])
    ax.imshow(pair)
    ax.set_title("Drone patch  |  Estimated map patch")
    ax.axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "mcl_convergence.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved visualization: {out}")


def run_trajectory_test():
    """Multi-frame temporal filtering test: a curving trajectory on the map.

    Observations are map patches extracted at the true poses (synthetic — no
    camera/satellite domain gap), so real false peaks from the similarity
    landscape are present. The filter must converge from a wide initial
    uncertainty using the accumulated consistency of observations over
    frames: a false peak matches one frame, the true track matches all.
    This is the "warm-up distance" behavior described in the paper.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    map_rgb = load_rgb(MAP_PATH)
    h, w = map_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    print(f"Map: {w}x{h}px  center=({cx},{cy})  gsd={MAP_GSD} m/px")

    mcl = MCL(map_rgb, MAP_GSD, (cx, cy), CHECKPOINT)

    # --- Build a curving true trajectory (body-frame: 80m forward, +0.15 rad/step) ---
    STEP_M, DYAW = 80.0, 0.15
    N_FRAMES = 12
    x_t, y_t, yaw_t = (2700 - cx) * MAP_GSD, (cy - 2734) * MAP_GSD, 0.0   # start at true drone loc
    truth = [(x_t, y_t, yaw_t)]
    motions = []   # body-frame (dx, dy, dyaw) between consecutive frames
    for k in range(1, N_FRAMES):
        motions.append((STEP_M, 0.0, DYAW))
        x_t = x_t + STEP_M * np.cos(yaw_t)
        y_t = y_t + STEP_M * np.sin(yaw_t)
        yaw_t = yaw_t + DYAW
        truth.append((x_t, y_t, yaw_t))
    print(f"Trajectory: {N_FRAMES} frames, {STEP_M}m/step, +{DYAW} rad/step "
          f"(curving), {STEP_M * (N_FRAMES - 1):.0f} m flown")

    # --- Initial guess: wide offset (150 m + 150 m std) to test warm-up ---
    x0, y0, yaw0 = truth[0][0] + 150, truth[0][1] + 150, truth[0][2] + 0.4
    std = (150.0, 150.0, 0.5, 0.05)
    N, T = 1000, 0.02
    roughen_std = (5.0, 5.0, 0.03, 0.01)
    noise_std = (3.0, 3.0, 0.03, 0.01)   # process noise (odometry is fairly accurate)
    resample_eff = 0.3
    print(f"Init: offset=150m  std={std}  N={N} T={T}  process_noise={noise_std}")
    print(f"Temporal filtering: accumulate=True  resample_eff={resample_eff}")

    particles = mcl.init_particles(x0, y0, yaw0, 1.0, std, N)

    # --- Run predict + update per frame ---
    print(f"{'frame':>5} {'est_x':>8} {'est_y':>8} {'true_x':>8} {'true_y':>8} "
          f"{'err_m':>7} {'spread':>7} {'N_eff':>7} {'score_max':>9}")
    ests = []
    for k in range(N_FRAMES):
        # Observation: map patch at the true pose (synthetic drone view)
        tx, ty, tyaw = truth[k]
        px, py = mcl.state_to_pixel(tx, ty)
        obs = cropRectangularSectionFromImage(
            mcl.map_rgb, px, py, -tyaw, mcl.src_dim, PATCH_PX, 1.0)
        mcl.set_drone_patch(obs)
        # Motion model: apply inter-frame body delta (frame 0 = no motion)
        motion = motions[k - 1] if k > 0 else None
        ns = noise_std if k > 0 else None
        decay = max(0.2, 1.0 - 0.5 * k / N_FRAMES)
        particles, est, scores = mcl.step(
            particles, T, roughen_std, decay, motion=motion, noise_std=ns,
            resample_eff=resample_eff)
        ests.append(est)
        err = np.hypot(est["x"] - tx, est["y"] - ty)
        print(f"{k:5d} {est['x']:8.1f} {est['y']:8.1f} {tx:8.1f} {ty:8.1f} "
              f"{err:7.1f} {est['spread']:7.1f} {mcl.last_neff:7.1f} {scores.max():9.4f}")

    final_err = np.hypot(ests[-1]["x"] - truth[-1][0], ests[-1]["y"] - truth[-1][1])
    mean_err = np.mean([np.hypot(e["x"] - t[0], e["y"] - t[1])
                        for e, t in zip(ests, truth)])
    # warm-up: mean error over the last half (after the filter has locked on)
    warm_err = np.mean([np.hypot(e["x"] - t[0], e["y"] - t[1])
                        for e, t in zip(ests[N_FRAMES // 2:], truth[N_FRAMES // 2:])])
    print("\n" + "=" * 55)
    print(f"Final tracking error: {final_err:.1f} m  (target < 30)")
    print(f"Mean tracking error:  {mean_err:.1f} m  (all frames)")
    print(f"Warm tracking error:  {warm_err:.1f} m  (last {N_FRAMES - N_FRAMES // 2} frames)")
    print(f"Final spread:         {ests[-1]['spread']:.1f} m  (target < 50)")
    print("=" * 55)

    visualize_trajectory(mcl, ests, truth)
    return ests


def visualize_trajectory(mcl, ests, truth):
    disp = cv2.pyrDown(cv2.pyrDown(mcl.map_rgb))
    scale = 0.25

    def to_disp(x, y):
        px, py = mcl.state_to_pixel(x, y)
        return px * scale, py * scale

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: trajectory on map
    ax = axes[0]
    ax.imshow(disp)
    tx = [to_disp(t[0], t[1])[0] for t in truth]
    ty = [to_disp(t[0], t[1])[1] for t in truth]
    ax.plot(tx, ty, "g-", lw=2, label="true path")
    ax.plot(tx[0], ty[0], "g+", ms=15, mew=2)
    ex = [to_disp(e["x"], e["y"])[0] for e in ests]
    ey = [to_disp(e["x"], e["y"])[1] for e in ests]
    ax.plot(ex, ey, "c--o", ms=5, label="estimate")
    ax.set_title("Trajectory: true vs estimated")
    ax.legend(loc="upper right")
    ax.axis("off")

    # Right: tracking error over frames
    ax = axes[1]
    errs = [np.hypot(e["x"] - t[0], e["y"] - t[1]) for e, t in zip(ests, truth)]
    spreads = [e["spread"] for e in ests]
    ax.plot(errs, "o-", label="tracking error (m)")
    ax.plot(spreads, "s-", label="spread (m)")
    ax.axhline(30, color="r", ls=":", label="target (30m)")
    ax.set_xlabel("frame")
    ax.set_title("Tracking error over trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "mcl_trajectory.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved visualization: {out}")


if __name__ == "__main__":
    run_trajectory_test()
