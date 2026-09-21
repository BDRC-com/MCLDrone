# GNSS-denied Localization — Training & MCL Commands

All training/eval commands run from `/home/one/GNSS-denied-Localization` in the `sivl` conda env:

```bash
conda activate sivl
cd /home/one/GNSS-denied-Localization
```

---

## 1. Similarity Model Training

Configuration: [`train_config.yml`](train_config.yml) — all paths and hyperparameters
(multi-temporal area folders, deployment map + gsd + geo_offset, flights, label
model, lr, batchsize, jitter). **For a new region, edit this file only.**

Trainer: [`train_similarity.py`](train_similarity.py) (sivl-form: config-driven,
anchor/positive/negative BCE, tqdm loop, sivl-compatible checkpoints).

### Stage 1 — Build the multi-temporal map patch bank (once per region, ~30 min)

```bash
python train_similarity.py --build-bank
```

Reads the multi-temporal area folders (`data.path.trainingdata_bitmaps` /
`testingdata_bitmaps` — each folder is one fixed viewport holding temporal
layers; two layouts are auto-detected: Google-Earth `year_month_N.jpg` at the
top level, or USGS NAIP-style `images/*_YYYYMMDD.png` subfolders such as
`datasets/woodbridge` and `datasets/fountainhead`). Per area:
SIFT+RANSAC-aligns every layer to the area's newest (chained via
nearest-date already-aligned layer; layers that fail every candidate are
KEPT at their capture registration, like sivl assumes — reported in the
log), then crops sivl's BitmapDataset centerpoint grid — start
`dim/2 + dim/sqrt(2)`, step `dim + marginBetweenSamples_px` (0 = ADJACENT
samples, no stride), one-time `apShiftingStd` jitter — from EVERY layer at
the SAME centers so patches of one location stay co-located across time.
NO GSD resampling: exactly like origin sivl's BitmapDataset, the grid is
laid on the images' NATIVE pixel size (the earlier resample to 1 m/px
shrank the Huizhou 8192-px exports 83x83 -> 49x49 nodes and lost 61% of
the samples). ALL images and ALL crops are kept (sivl semantics — no featureless filter, no drops). Output: `data.path.patch_bank/bank.zarr`
+ `meta.npz` (~1.19M patches, ~49 GB — zstd-compressed zarr, 1.34x
measured; needs ~50 GB free disk). Rows are NODE-MAJOR (area -> node ->
layer): a location's temporal layers sit on adjacent rows, so the
train-time anchor/positive read hits adjacent chunks instead of random
rows ~100k apart, and the compressed bank (~49 GB vs 65.8 GB raw) lets
the page cache cover far more of it — together this removes the disk
bottleneck that made epochs swing 90 s to 5 min. The build is PARALLEL
(pass 1 SIFT alignment, pass 2 warp+crop) with worker counts derived
from a 24 GB total RAM budget minus an 8 GB OS+parent reserve (SIFT
workers peak ~2 GB private each; 16 uncapped workers OOM'd the 32 GB
machine). Requires `pip install "zarr<3" numcodecs` (sivl conda env).

### Stage 2 — Pretrain on multi-temporal pairs (default mode)

```bash
python train_similarity.py
```

RESUMES from the user's original-sivl 200-epoch checkpoint
(`sivl/checkpoints/training_4_epoch_00200_of_1000.pt`, trained with the
original sivl code): `use_only_network_params: False` loads its optimizer
state + epoch history, so training continues at epoch 200 and runs the
remaining 800 of 1000.

sivl's BitmapDataset pairing on the bank: anchor/positive = two different
random temporal layers of the same location sharing one random rotation +
scale (positive adds sivl's pair noise: rotation/translation/corner stds;
anchor additionally gets the calibrated drone-appearance simulation
`sim_drone`), negative = random other location of the same area. batchsize
200 (sivl's configuration.yml values). `simval` = held-out testing areas
(3, 7). Experiment `huizhou_mt_1` ->
`checkpoints_huizhou/huizhou_mt_1_epoch_{300..1000}_of_1000.pt`
(saved every 100 epochs; first save at epoch 300 since 200 is the resume
point). Note: the log's tqdm shows "Epoch 201/1000" for the first epoch
(epoch counter is 0-indexed + 1 display).

### Stage 3 — Build real-pair dataset (once per flight set)

```bash
python train_similarity.py --build-real
```

For each flight in `data.path.flights`: clock-aligns the ulog with the bag
(gyro-z / yaw-rate correlation), maps GPS truth to map pixels via
`geo_offset`, rotation-labels each debug patch with the label model
(`data.path.label_model_checkpoint`). Frames scoring < 0.5 are dropped.

Requirements per flight entry:
- `run_dir`: MCL run output with `replay_log.npz` + `debug/k*_4_patch.png`
  (i.e. an MCL run with `save_debug:=true`)
- `ulog`: flight log with GPS truth (`.ulg`)
- `db3`: rosbag db3 with `/imu0` (for clock alignment)

Output: `patch_bank/real_pairs.npz`.

### Stage 4 — Fine-tune on mixed real + simulated (~25 min)

```bash
python train_similarity.py --finetune
```

Init from `finetune.initial_model_checkpoint` (already set to the pretrain
output `huizhou_mt_1_epoch_1000_of_1000.pt`). Experiment `huizhou_ft_mt_1` ->
`checkpoints_huizhou/huizhou_ft_mt_1_epoch_040_of_40.pt` (the deployment
checkpoint; also the new `mcl.py`/launch default). Mixture:
`finetune.p_real` real pairs, rest simulated bank pairs. Tracks two
validation sets each epoch:
- `simval` — must stay acc 1.000 (false-peak rejection retained)
- `realval` — cross-FLIGHT validation: ALL frames of the flights listed in
  `finetune.val_flights` (currently 190020); the model never trains on them.
  This is the deployment gate — expect held-out pos ~0.9+. An in-flight
  every-5th-frame split leaks (frames 0.2 s apart over a 96 m footprint are
  near-duplicates): a fine-tune on 184004 reported realval 1.000 yet
  false-locked on 190020 (held-out truth med 0.001) — incident 2026-09-09.
  All flights in `data.path.flights` get pairs via `--build-real`;
  `val_flights` only controls which are excluded from training. Final
  acceptance: full MCL bag runs on the held-out flight (stronger than the
  offline patch metric), plus a never-used 3rd flight as true test set.

### Smoke tests

```bash
python train_similarity.py --epochs 2 --max-batches 30
python train_similarity.py --finetune --epochs 2 --max-batches 30
```

### Evaluate on real patches (truth vs false peaks)

```bash
# full eval: 2-deg rotation sweep at GPS truth vs near/far false locations
python eval_real_patches.py --frames 40 --save-sb 6 \
  --ckpt checkpoints_huizhou/huizhou_ft_1_epoch_040_of_40.pt

# failure diagnosis: scale sweep + patch stats + embedding domain gap
python eval_real_patches.py --profile --frames 40 --ckpt <ckpt>
```

---

## 2. Running MCL (online localization)

ROS2 Jazzy environment:

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/ovws/install/setup.bash
```

**Important**: kill stale nodes before relaunch (orphans with the same
`out_dir` corrupt `replay_log.npz`):

```bash
pkill -f mcl_node
```

### Standard run (VIO + MCL + bag playback)

```bash
ros2 launch ov_mcl mcl_localization.launch.py bag:=<bag_dir>
```

Example:

```bash
ros2 launch ov_mcl mcl_localization.launch.py \
  bag:=/home/one/workspace/datasets/bag_0001_20260831_184004
```

Defaults: `vio_config=cyperstereo_012_752x480_equi`, map `z17_5120.png`,
checkpoint `checkpoints_huizhou/huizhou_ft_mt_1_epoch_040_of_40.pt`.

> **rclpy type trap**: pass float launch args WITH a decimal point
> (`init_radius_m:=300.0`, not `300`) — a bare integer parses as INTEGER and
> rclpy rejects it against the node's DOUBLE declaration
> (`InvalidParameterTypeException`).

### Useful launch arguments

| arg | default | meaning |
|---|---|---|
| `bag:=` | — | rosbag2 directory to play (replay test). Also enables the GPS-truth overlay on `replay.png`: the ULog beside the bag (`*.ulg`) is aligned to the bag clock via /imu0 gyro-z ↔ yaw-rate correlation and the truth track is drawn (yellow dashed) on the trajectory panel with median/p90 error in the title. The filter NEVER sees the GPS (plotting only). No `.ulg`/`.db3` beside the bag or weak alignment (corr < 0.3) → plot without truth + a warning |
| `vio_config:=` | `cyperstereo_012_752x480_equi` | VIO config name or full path |
| `checkpoint:=` | huizhou_ft_mt_1 | similarity model checkpoint |
| `map_path:=` | `z17_5120.png` | deployment map |
| `out_dir:=` | `.../mcl_test/online` | output dir (replay_log.npz, plots) |
| `stride:=` | `5` | process every N-th camera frame |
| `frame_start:=` | `0` | first camera frame to process (skip takeoff; 0 = from start) |
| `frame_end:=` | `0` | one past last camera frame to process (skip landing; 0 = to end) |
| `n_particles:=` | `1000` | particle count |
| `init_mode:=` | `scan` | `scan` / `point` / `click` |
| `init_lat:=` / `init_lon:=` | `-999.0` | init WGS-84 lat/lon (deg) for `point` mode — **overrides** `init_px_x/y` |
| `init_px_x:=` / `init_px_y:=` | `0` | init point (map px) for `point` mode |
| `init_radius_m:=` | `300.0` | `point`-mode prior uncertainty radius [m]: init scan sweeps this region around the clicked point (compass-steered yaw window when FCU heading is available) |
| `yaw_cal:=` | `-1,1.3` | compass→MCL-map yaw calibration `"a,b"` (b in deg): `yaw_mcl = a*fcu_ned_heading + radians(b)`. Fitted on bag 190020 s2 replay (residual std 4.3°); a=−1 (NED heading is CW-positive), b = declination + camera-mounting offset — rig+map constant |
| `map_geo_offset_x/y:=` | z17_5120.png offset | map georeference for lat/lon conversion: local px = global z17 mercator px + offset (for a map stitched from z17 tiles with top-left tile origin `(tx,ty)`: `(-tx*256, -ty*256)`) |
| `alt_anchor:=` | `0.0` | altitude offset added to VIO z |
| `fcu_alt_topic:=` | `/mavros/local_position/pose` | live FCU baro height + EKF attitude source (MAVROS ENU local position pose: z = height, orientation = EKF attitude; empty disables). Replay reads the ULog beside the bag instead; live mode interpolates this topic (falls back to VIO z / VIO attitude if absent) |
| `fcu_imu_rot:=` | Wahba fit (bag 190020) | FCU-body(FRD) → /imu0 mounting rotation, 9 row-major values comma-separated (empty = built-in default; recalibrate if the rig is re-mounted — fit windowed /imu0 acc means against ULog-attitude gravity) |
| `save_debug:=` | `false` | save per-frame pipeline images (needed for `--build-real`) |
| `debug_stride:=` | `1` | save debug images every N-th processed frame |

### Health monitoring topic (`/mcl/health`)

`std_msgs/String` JSON at 1 Hz — everything a guard/mission-manager needs to
decide whether the localization output is currently trustworthy:

- `state`: `no_frames` (nothing processed yet) | `init_wait` (waiting for
  first fix — includes the low-coverage hold) | `motion_only` (tracking but
  current frame below the coverage gate: VIO-only predict, no measurement
  update) | `recovering` (fix found, confirming) | `lost` (rescanning) |
  `tracking` (healthy)
- `note` (why a frame was skipped / degraded): `ok` | `low_cov` |
  `scan_buffer` | `global_wait` | `no_geom` | `envelope` | `proc_fail`
- Raw facts: `k`, `t` (bag clock), `x`, `y`, `yaw_deg`, `score`, `spread`,
  `neff`, `coverage`, `alt`, `vio` (ok/drop/resume), `innov_m` (EMA of
  estimate-vs-VIO motion disagreement), `reloc_fails`, `locals_recent`
  (local rescans in the last 90 s — the tracking-offset-aliasing signal),
  `recover_n`, `cov_gated_run`, `vio_jumps`

```bash
ros2 topic echo /mcl/health    # note: echo truncates long strings — parse
                               # the JSON with a small rclpy subscriber if
                               # you need every field
```

### Low-coverage frame gating (SCAN_MIN_COV = 88 m)

Frames whose orthoprojected ground coverage is below 88 m (≈44 m altitude
for the 752×480 fisheye) MUST NOT update the particle filter: real drone
patches below that are mushy to the CNN (near-uniform ~0.99 scores) and
false-lock — the v7 replay init'd at 15 m altitude / cov 32 and burned 590 m
of aliasing within 45 frames. The 32–96 m tolerance in
`coverage_sweep_test.py` was synthetic-only and does NOT transfer to real
patches. The gate:

- **Altitude is honest** — the old `max(20, alt)` clamp is gone (it faked
  40 m coverage on the ground). Altitude floor `ALT_MIN = 3.0` m; the VIO-z
  gate is `3 ≤ z ≤ 400` m.
- **Gated frames** (coverage < 88): skip `set_drone_patch` (no CNN forward),
  run `mcl.step(skip_update=True)` = motion-only predict, freeze the
  lost/rescan machinery (no streak counting, no rescans).
- **Init holds** until the first frame with coverage ≥ 88 (`note=low_cov`),
  accumulating `preinit_motion` (map-frame VIO deltas) so the init region
  follows the drone during the climb; after 240 s (`SCAN_WAIT_CAP_S`) it
  falls back to a degraded scan at the current best position.
- Verified end-to-end on bag 190020 point-init replay v8: no false-track
  phase, init locked ~4 m from GPS truth at k=965, post-init median 5.3 m /
  p90 10.6 m / 94% < 20 m, zero LOST/global events, innov median 0.31 m
  (run kept at `sample_images/mcl_test/point_190020_noise170_v8/`).

### Running VIO, RVIZ, and MCL separately

Use this to inspect each stage independently (e.g. check VIO trajectory in
RVIZ before starting MCL). Each in its own terminal, all with the ROS2 env
sourced:

```bash
source /opt/ros/jazzy/setup.bash && source ~/workspace/ovws/install/setup.bash
```

If you hit `PermissionError: [Errno 13] Permission denied: '/home/one/.ros/log/...'`,
redirect the log dir first:

```bash
export ROS_LOG_DIR=/tmp/ros_log
```

**Terminal 1 — bag playback** (skip for live camera):

```bash
ros2 bag play /home/one/workspace/datasets/bag_0001_20260831_184004
```

**Terminal 2 — VIO only** (SchurVINS/OpenVINS, subscribes to
`/cam0/image_raw`, `/cam1/image_raw`, `/imu0`; publishes
`/ov_msckf/odomimu`):

```bash
ros2 launch ov_msckf subscribe.launch.py config:=cyperstereo_012_752x480_equi rviz_enable:=true
```

Add `rviz_enable:=true` to also start RVIZ with the VIO display config, or
run RVIZ standalone (below) to watch VIO and MCL at the same time.

**Terminal 3 — MCL only** (subscribes to `/cam0/image_raw` +
`/ov_msckf/odomimu`; start after VIO is publishing):

```bash
ros2 run ov_mcl mcl_node --ros-args \
  -p map_path:=/home/one/GNSS-denied-Localization/z17_5120.png \
  -p checkpoint:=/home/one/GNSS-denied-Localization/checkpoints_huizhou/huizhou_ft_1_epoch_040_of_40.pt \
  -p calib_path:=/home/one/workspace/ovws/install/ov_msckf/share/ov_msckf/config/cyperstereo_012_752x480_equi/kalibr_imucam_chain.yaml \
  -p out_dir:=/home/one/GNSS-denied-Localization/sample_images/mcl_test/online_separate \
  -p stride:=5 \
  -p n_particles:=1000 \
  -p init_mode:=point \
  -p init_lat:=22.842866 \
  -p init_lon:=114.525564
```

Useful extra `--ros-args -p` parameters (same meanings as the launch
table): `init_px_x`, `init_px_y` (with `init_mode:=point`), `alt_anchor`,
`save_debug:=true`, `debug_stride`.

Reference init point:
- Bag 184004: `init_lat:=22.842866 init_lon:=114.525564`
- Bag 190020: `init_lat:=22.842897 init_lon:=114.525573`
- Bag 174711: `init_lat:=22.842871 init_lon:=114.525563`

Notes:

- Start order matters: bag/VIO first, wait for `/ov_msckf/odomimu` messages
  (`ros2 topic hz /ov_msckf/odomimu`), then start MCL — the MCL node needs
  VIO odometry to drive its prediction step.
- To record the VIO odometry while it runs (for offline analysis):
  `ros2 bag record /ov_msckf/odomimu -o vio_odom`

### Truth-anchored run with debug saving (for building real pairs)

```bash
# by map pixel:
ros2 launch ov_mcl mcl_localization.launch.py \
  bag:=<bag_dir> \
  init_mode:=point init_px_x:=2686 init_px_y:=2728 \
  save_debug:=true \
  out_dir:=/home/one/GNSS-denied-Localization/sample_images/mcl_test/my_run

# or by WGS-84 lat/lon (overrides init_px; equivalent to the above):
ros2 launch ov_mcl mcl_localization.launch.py \
  bag:=<bag_dir> \
  init_mode:=point init_lat:=22.842871 init_lon:=114.525563 \
  save_debug:=true \
  out_dir:=/home/one/GNSS-denied-Localization/sample_images/mcl_test/my_run
```

Then add the run to `train_config.yml` under `data.path.flights` and
re-run Stage 3/4.

### Debug frame generation (bag_0001_20260831_184004, camera frames 2000-8200)

`k` in debug filenames IS the camera-frame index (k02000 = bag camera frame
2000; stride only filters which frames appear, it does not renumber). The
sivl (Sweden) checkpoint is only needed to make MCL run — the training
patches/truth for `--build-real` come from the bag + ulog, not from this model.

```bash
pkill -f mcl_node   # kill stale nodes first
ros2 launch ov_mcl mcl_localization.launch.py \
  bag:=/home/one/workspace/datasets/bag_0001_20260831_184004 \
  checkpoint:=/home/one/GNSS-denied-Localization/sivl/checkpoints/training_4_epoch_00200_of_200.pt \
  out_dir:=/home/one/GNSS-denied-Localization/sample_images/mcl_test/bag0001_debug \
  save_debug:=true debug_stride:=1 \
  init_mode:=point init_lat:=22.842866 init_lon:=114.525564 \
  frame_start:=2000 frame_end:=8201
```

- `frame_end` is EXCLUSIVE (`frame_idx < frame_end`): 8201 includes k08200
- With stride=5: processed frames k02000, k02005, ..., k08200 (1241 frames,
  all above the 30 m altitude gate — verified from the ulog)
- init point = drone GPS position AT frame 2000 (ulog: lat 22.842866,
  lon 114.525564 -> map px (2684,2728); the drone is still near-vertical at
  this point in the climb, ~58 m altitude)
- VIO still consumes all frames from bag start (only MCL skips the first
  2000), so VIO is warmed up before the first processed frame
- Bag 184004 frame_start:=2000 frame_end:=8201
- Bag 190020 frame_start:=1100 frame_end:=14801
- Bag 174711 frame_start:=900 frame_end:=10091

---

## 3. Rebuilding the workspace (after code changes)

Workspace: `/home/one/workspace/ovws` (packages: `ov_mcl`, `ov_SchurVINS`).

After editing Python/launch/config files in `src/ov_mcl/` or
`src/ov_SchurVINS/`, rebuild so the `install/` space picks up the changes:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/one/workspace/ovws
colcon build --packages-select ov_mcl --symlink-install
```

- `--symlink-install`: install symlinks instead of copies — Python source
  edits (`ov_mcl/ov_mcl/*.py`) take effect without rebuilding; only
  launch files / entry points / non-Python files need a rebuild
- Rebuild everything: `colcon build --symlink-install`
- Rebuild SchurVINS (C++, after VIO source changes — slow):

  ```bash
  cd /home/one/workspace/ovws
  colcon build --packages-select ov_msckf
  ```

Re-source after any rebuild (already-open terminals keep the old install):

```bash
source ~/workspace/ovws/install/setup.bash
```

Note: `mcl_node` imports `mcl.py`, `bag_mcl.py` etc. directly from
`/home/one/GNSS-denied-Localization` (see `GNSS_DIR` in
[mcl_node.py](/home/one/workspace/ovws/src/ov_mcl/ov_mcl/mcl_node.py)) —
changes to those files need no rebuild, just restart the node.

---

## 4. Typical full workflow (from scratch / new region)

For the CURRENT region the area exports, flights and config are already in
place — start at step 4:

1. Prepare multi-temporal area exports: Google-Earth fixed viewports
   (8192×8192, one folder per area under `datasets/huizhou/area_N/` with
   `year_month_N.jpg` layers, ≥2 layers per area). Train and val areas must
   be spatially disjoint. Measure `bitmap_gsd` by correlating one layer
   against the deployment map (Huizhou: 0.60)
2. Record a flight bag + ulog; run MCL once with `save_debug:=true`
   (see "Truth-anchored run" above)
3. Edit `train_config.yml`: area folders, `bitmap_gsd`, deployment map path,
   `gsd`, `geo_offset`, flights
4. `python train_similarity.py --build-bank` (rebuild the patch bank)
5. `python train_similarity.py` — pretrain → `huizhou_mt_1_epoch_XXXXX_of_1000.pt`
6. `python train_similarity.py --build-real` — rebuild `real_pairs.npz`
   (independent of pretrain; needs only the label model + flight debug runs)
7. `python train_similarity.py --finetune` — init from the pretrain output
   (config already points there) → `huizhou_ft_mt_1_epoch_040_of_40.pt`
8. `python eval_real_patches.py --ckpt checkpoints_huizhou/huizhou_ft_mt_1_epoch_040_of_40.pt`
   — expect win ~100%, held-out-flight truth score high
9. Deploy: `ros2 launch ov_mcl mcl_localization.launch.py bag:=<bag>`
   (default checkpoint is the mt finetune output)

---

## 5. PX4 SITL end-to-end test: MCL External Vision → EKF2 (step 4c)

Closes the loop in simulation: the recorded `/mcl/odom` EV stream (from the
verified round-2 replay, `/tmp/ev_record`) is fed by
[`sitl_ev_driver.py`](sitl_ev_driver.py) into
`/fmu/in/vehicle_visual_odometry`; PX4 EKF2 fuses it (GNSS-denied) while
the sim x500 flies the bag's motion profile in OFFBOARD; `/fmu/out/*` +
`/mcl/odom` are recorded and compared against bag GPS truth
(`from mcl_node import bag_gps_truth`). Validated 2026-09-17/18 (run 3,
bag `/tmp/sitl_result3`, 112k msgs / 410 s).

### Recipe (4 terminals — run the agent/SITL unsandboxed)

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/ovws/install/setup.bash                          # bag play/record
source /home/one/ros2_ws/install/setup.bash                         # px4_msgs (v1.17, prebuilt, Jazzy)
export ROS_LOG_DIR=/tmp/roslog
export PYTHONDONTWRITEBYTECODE=1   # sandbox: /opt/ros/jazzy __pycache__ writes are blocked
```

1. **uXRCE-DDS agent**: `MicroXRCEAgent udp4 -p 8888`
2. **PX4 SITL** (from `/home/one/Documents/PX4-Autopilot`):
   `make px4_sitl gz_x500_mono_cam` — target is `gz_<model dir under
   Tools/simulation/gz/models>`; boot ~40 s, SITL auto-starts the
   uXRCE-DDS client on localhost:8888.
3. **Recorder**:
   ```bash
   ros2 bag record -o /tmp/sitl_result3 \
     /fmu/out/vehicle_local_position_v1 /fmu/out/vehicle_global_position \
     /fmu/out/vehicle_odometry /fmu/out/vehicle_status_v1 \
     /fmu/out/vehicle_land_detected /fmu/out/failsafe_flags \
     /fmu/out/estimator_status_flags /fmu/out/vehicle_command_ack /mcl/odom
   ```
4. **EV stream + driver** (driver from `/home/one/GNSS-denied-Localization`):
   ```bash
   ros2 bag play /tmp/ev_record --rate 1
   python3 sitl_ev_driver.py
   ```

In the SITL console before the flight, configure EKF2 for EV-only fusion:
`EKF2_EV_CTRL=5` (horizontal position + 3D velocity; NO EV height/yaw —
baro owns z) and `EKF2_GPS_CTRL=0`.

Driver timeline (run 3): OFFBOARD engaged 3.0 s after the first EV message;
ARMED at EV+41.4 s after 37 `TEMPORARILY_REJECTED` retries; 263 s of armed
flight; 5 s after the stream ends it commands AUTO.LAND. It logs
`EV lock: map fix (143.8, -184.3) -> sim origin` — honest EV positions are
translated by the first honest fix so the bag track starts at the sim origin.

### Mode switching: DO_SET_MODE, NOT SET_NAV_STATE

`VEHICLE_CMD_SET_NAV_STATE (100001)` acks ACCEPTED but the mode is never
applied (observed twice). `VEHICLE_CMD_DO_SET_MODE (176)` with
`param1=MAV_MODE_FLAG_CUSTOM_ENABLED(1)` and `param2/3` = PX4 custom
main/sub mode (OFFBOARD `(1,6,0)`, AUTO.LAND `(1,4,6)`) is the working
path — driver v3 uses it.

### Arming in GNSS-denied SITL takes ~40 s of EV fusion

First ARM attempts are rejected with `Preflight Fail: heading estimate not
stable`, later `height estimate not stable` (ULog console); success came at
EV+41.4 s. The driver retries every 1 s — this delay is expected, not a bug.

### Gated poses must be NaN position, never (0,0,0)+huge covariance

EKF2 gates EV position and velocity independently; a zero position with 1e4
covariance is NOT "no data" — its Kalman gain still collapses the estimate
toward the EV origin between honest fixes (observed as ~140 xy resets in an
earlier run). [`sitl_ev_driver.py`](sitl_ev_driver.py) therefore encodes
gated poses as NaN position (twist is always honest, body FLU → FRD).
**TODO for real deployment**: the EV publisher in `mcl_node.py` must encode
gated poses as NaN too (currently zeros + huge covariance).

### Measured results (run 3, 12,682 armed + honest-EV samples)

| pair | median | p90 |
|---|---|---|
| EKF2 vs EV target (sim frame) | 2.3 m | 6.5 m |
| EKF2 vs bag GPS truth | 5.7 m | 10.2 m |
| EV (MCL, translated) vs GPS truth | 5.3 m | 9.2 m |

EV-vs-truth 5.3 m replicates the step-4b replay number exactly → map
georeference, frame conversions and fix translation are all correct; fusion
adds only +0.4 m median over the input. Health flags while flying (after
EV+60 s): `cs_ev_pos` 60%, `cs_ev_vel` 84%, `local_velocity_invalid` 2%,
inertial-dead-reckoning 19%. `SET_GPS_GLOBAL_ORIGIN (100000)` (param5/6 =
lat/lon; driver sends map center 22.8445297 / 114.5242310) was accepted 6/6
and EKF2 logged `New NED origin (LLA)` — this is what makes
`/fmu/out/vehicle_global_position` comparable to bag GPS truth.

### Known sim-only artifact: z over-climb

The bag's body twist z (median 1.6 m/s) carries real-flight tilt coupling;
the driver feeds it as the NED-down vertical for BOTH the feedforward
setpoint and the EV 3D velocity. In SITL the vehicle therefore climbed to
~169 m and EKF2 z drifted (late `Attitude failure (pitch)` in the ULog).
`/mcl/odom` orientation is yaw-only, so the bag tilt cannot be recovered
from the stream. Sim-only: the real FCU rotates body velocity with its own
attitude, and EV height is not fused (baro owns z).

### SITL memory leak — shutdown protocol (MANDATORY)

A leftover gz SITL run consumed ALL 31 GB RAM + 11 GB swap in ~15 min
(2026-09-17). NEVER leave it running unattended. After each test:

```bash
pgrep -af 'build/px4_sitl_default/bin/px4|gz sim|MicroXRCEAgent|sitl_ev_driver|ros2 bag'
pkill -f sitl_ev_driver; pkill -f MicroXRCEAgent
pkill -f 'ros2 bag record'; pkill -f 'build/px4_sitl_default/bin/px4'; pkill -f 'gz sim'
```

During a run, watchdog RSS every 15 s and kill if total > 6 GB (run 3 stayed
flat at ~260 MB → no leak that run). Watchdog threshold (run 3):
`pgrep -f 'build/px4_sitl_default/bin/px4|gz sim'`, sum `ps -o rss=`,
`pkill` at >6 GB.

### Runtime budget

Run 3 was cut 440.9 s after SITL boot while still airborne — AUTO.LAND was
therefore never exercised. For a full-mission run (incl. landing) start the
bag play promptly after boot and allow ≥ 600 s of SITL runtime (the job
wrapper's own timeout may be the hard cap; the watchdog had not fired and no
OOM appears in syslog).

---

## 6. Pre-staging a deployment map (step 6)

One command turns a centre + desired edge length into a ready-to-fly map, its
georeference sidecar and a map-edge margin report — no mid-mission downloads:

```bash
python3 prestage_map.py --center 22.8445297,114.5242310 --size 5 \
  [--zoom 17] [--margin-m 1000] [--waypoints "137,-185; 500,300"] \
  [--provider google_sat] [--no-download] [--rebuild-index] [--dry-run]
```

- **window**: `--size` km at z17 (native ~1.10 m/px), rounded OUT to whole
  256 px tiles centred on `--center`, so the georeference stays on the tile
  grid: `local px = global z17 px + geo_offset`, `geo_offset = -tile_origin*256`
  (exactly what `mcl_node`'s `map_geo_offset_x/y` expects).
- **tiles**: missing tiles of the window are fetched into the shared store
  `huizhou_map/tiles` (jpg, resumable — reruns only get what is new); the
  MapDB sqlite is rebuilt only when something was downloaded
  (or `--rebuild-index`).
- **outputs**: `<out>` PNG + `<out>.json` sidecar (actual centre, gsd, tile
  origin, geo offsets, `map_half_m`, missing fraction); default name
  `z17_<size>km_<lat>_<lon>.png` in the repo root.
- **margin report**: usable box `±(size/2 − margin_m)`; `--waypoints`
  (map ENU metres — the mission-manager convention) are checked and any
  violation exits 1.
- Prints the runtime parameters to paste in (`map_path`, `map_geo_offset_x/y`,
  `map_half_m`, `map_center_lat/lon` for the EKF2 origin).
- Needs only numpy + opencv (system `python3` works).
- Validated 2026-09-18: a 5 km window at the deployment centre is a
  pixel-exact crop of `z17_5120.png` (mean abs diff 0.000).

## Notes

- `geo_offset` for a map stitched from z17 tiles with top-left tile origin
  `(tx, ty)`: `[-tx*256, -ty*256]`
- `z17_5120.png`'s true georeference (pixel-exact validation, 2026-09-18) is
  origin tile `(107223, 56979)` → `geo_offset = (-27449088, -14586624)`. The
  `-14586625` default in `ov_mcl`'s launch file / `mcl_node.py` is 1 px
  (1.1 m) off — only lat/lon↔map-px conversions (point init, GPS-truth
  overlay) are affected, not the particle filter itself.
- Rotation sweeps for evaluation must use 2-deg steps (the model is
  rotation-sensitive; 10-deg steps miss peaks)
- Training on GPU: launch long runs with `nohup ... &` if needed; the model
  is ~212 MB per checkpoint
- Checkpoints live in `checkpoints_huizhou/`; the default deployment
  checkpoint is set in [`mcl.py`](mcl.py) `CHECKPOINT` and the launch file
