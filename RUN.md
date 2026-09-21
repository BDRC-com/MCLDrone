# RUN.md — deployment pipeline and start sequence (real airframe)

GPS substitute on the companion: the camera feeds VIO, VIO feeds MCL
(CNN map-matching + particle filter), and the mission manager turns MCL into
PX4 external vision, flies the mission in OFFBOARD and lands when the guard
says so.

```
 camera ─► VIO ─► MCL ─┬─ /mcl/odom   (map-ENU pose + body twist)
                       └─ /mcl/health (state + metrics, 1 Hz)      ┌─ /fmu/in/vehicle_visual_odometry (NaN-gated poses)
                              │                                     ├─ /fmu/in/offboard_control_mode + trajectory_setpoint (50 Hz)
                              ▼                                     │
                        mission_manager (guard) ───────────────────┴─ /fmu/in/vehicle_command (DO_SET_MODE / ARM / AUTO.LAND)
                              │  /mission/state (1 Hz JSON)
                              ▼
                        PX4 EKF2 (EV-only) ─► OFFBOARD flight
                        uXRCE-DDS via MicroXRCEAgent (udp4 8888)
```

Everything on the companion; PX4 has **no GNSS aiding** (EKF2_EV_CTRL=5,
EKF2_GPS_CTRL=0). Interface contract the pipeline expects:

| topic | source | notes |
|---|---|---|
| `/cam0/image_raw`, `/cam1/image_raw`, `/imu0` | camera driver | 752×480 YUYV, 25–60 Hz stereo + IMU (same rig as the training flights) |
| `/mavros/local_position/pose` | MAVROS | FCU baro height + EKF attitude (`fcu_alt_topic`) |
| `/fmu/out/*`, `/fmu/in/*` | uXRCE-DDS | px4_msgs v1.17 (`~/ros2_ws`) |

## 0. Before the flight (once per airframe / per area)

**FCU parameters** (QGC or pxh console, then `param save`):

```
param set EKF2_EV_CTRL 5     # horizontal position + 3D velocity; NO EV height/yaw
param set EKF2_GPS_CTRL 0    # GNSS-denied
```

**Stage the map** for the takeoff area (downloads + stitches + margin report):

```bash
python3 prestage_map.py --center <takeoff_lat>,<takeoff_lon> --size 5 \
  --waypoints "x,y; x,y"        # planned route, map ENU metres
```

Keep the printed runtime parameters — the MCL launch and the manager need
`map_path`, `map_geo_offset_x/y`, `map_half_m`, `map_center_lat/lon`. The
report fails (exit 1) if a waypoint leaves the ±(size/2 − 1 km) usable box.

**Checkpoints**: deployment model is
`checkpoints_huizhou/huizhou_ft_mt_1_epoch_040_of_40.pt` (never a
pretrain-only checkpoint); VIO config `cyperstereo_012_752x480_equi`.

## 1. Power-on and interface check

1. FCU, camera, companion powered; wait for the camera stream and MAVROS.
2. Verify the contract **before** launching anything:

```bash
ros2 topic hz /cam0/image_raw          # ~25-60 Hz
ros2 topic hz /imu0
ros2 topic echo /mavros/local_position/pose --once   # FCU height + attitude
```

## 2. Companion start sequence (4 terminals)

```bash
# every terminal:
source /opt/ros/jazzy/setup.bash
source ~/ovws/install/setup.bash             # VIO + MCL (dev machine: ~/workspace/ovws)
source ~/ros2_ws/install/setup.bash          # px4_msgs (uXRCE-DDS)
export ROS_LOG_DIR=/tmp/roslog
```

1. **uXRCE-DDS agent** — `MicroXRCEAgent udp4 -p 8888`
2. **MAVROS** — the drone's usual launch so `/mavros/local_position/pose`
   is live (baro height + EKF attitude).
3. **MCL node** (live mode; FCU topic from MAVROS):

```bash
ros2 launch ov_mcl mcl_localization.launch.py \
  map_path:=<staged map> \
  map_geo_offset_x:=<off_x> map_geo_offset_y:=<off_y> \
  init_mode:=point init_lat:=<takeoff_lat> init_lon:=<takeoff_lon> \
  init_radius_m:=300.0 \
  fcu_alt_topic:=/mavros/local_position/pose \
  out_dir:=~/mcl_runs/<flight_name>
```

   The initial scan waits for the first frame with coverage ≥ 88 m (≈44 m
   altitude) and runs **during the climb** — takeoff is not blocked by MCL.
   Watch `/mcl/health`: `no_frames → init_wait(low_cov) → recovering →
   tracking` (full argument list: COMMANDS.md §2).

4. **Mission manager** (map frame anchored at the take-off point):

```bash
python3 mission_manager.py --ros-args \
  -p mission_mode:=waypoints \
  -p 'waypoints:=x,y; x,y' \
  -p ev_frame:=map \
  -p local_origin_map_x:=<takeoff map ENU x> \
  -p local_origin_map_y:=<takeoff map ENU y> \
  -p cruise_alt:=50.0
```

   `ev_frame:=map` means the EV odometry is the map-ENU position minus the
   take-off point; `local_origin_map_x/y` is that point in map metres (the
   same frame `prestage_map.py --waypoints` uses). **Do not** run
   `sitl_ev_driver.py` at the same time — both publish
   `/fmu/in/vehicle_visual_odometry`.

> **Float launch args need a decimal point** — `map_geo_offset_x:=-27449088.0`,
> `init_radius_m:=300.0`, `cruise_alt:=50.0`. On Humble a value like
> `-27449088` is inferred as INTEGER and a DOUBLE parameter rejects it with
> `InvalidParameterTypeException` (Jazzy is more forgiving, which is why the
> dev machine did not show this).

## 3. Mission flow (what the manager does)

1. Publishes EV odometry (gated poses as **NaN position** — zeros + huge
   covariance would be fused) and 50 Hz OFFBOARD setpoints; engages
   `DO_SET_MODE(OFFBOARD)`, then retries ARM. In GNSS-denied arming takes
   ~40 s of EV fusion — `Preflight Fail: heading estimate not stable` /
   `height estimate not stable` messages are expected, not a bug.
2. `wait → takeoff` once armed+OFFBOARD; climbs at 2.5 m/s to `cruise_alt`,
   then `takeoff → mission` and flies the waypoints (P-controller toward each
   point, yaw along the route).
3. The **guard** watches `/mcl/health`, EV freshness and position:
   - `hold` on degraded localization (`lost`/`no_frames` immediately;
     `motion_only`/`recovering` after 25 s; `vio=drop` after 3 s), budget
     180 s cumulative — no infinite loitering;
   - `return` if the position leaves `map_half_m − margin_m` (default
     `2500 − 1000 = 1500 m`) for > 3 s, budget 60 s;
   - `land` when the last waypoint is reached, on any timeout, on a blind
     low-altitude state, or if PX4 drops out of armed+OFFBOARD for > 3 s
     (external takeover).
4. Last waypoint → `AUTO.LAND`; `land → done` on the land detector, or after
   `land_max_s` (120 s) the manager forces DISARM.
5. `/mission/state` (1 Hz JSON) carries state, reason, position, altitude and
   the armed/nav flags — record it for the post-flight review.

Abort rules: manager exits before flying if not armed within 180 s of the
first EV message; every guard path is bounded and ends in MISSION or LAND.

## 4. Landing and shutdown

1. After `done` (or a manual abort): kill the manager, then the MCL launch,
   then the agent; disarm if still armed.
2. Archive `out_dir/replay_log.npz`, the `/mission/state` recording and the
   logs; check `/mcl/health`'s `state` never sat in `lost` without recovery.

## 5. Parameter reference (defaults)

| parameter | default | meaning |
|---|---|---|
| `cruise_alt` | 50 m | mission altitude (must be ≥ ~44 m for scan coverage) |
| `mission_mode` / `waypoints` | feedforward / — | `waypoints` for real missions (map ENU metres) |
| `takeoff_max_s` | 120 s | takeoff timeout → LAND |
| `hold_budget_s` | 180 s | cumulative HOLD budget → LAND |
| `return_max_s` | 60 s | margin-return budget → LAND |
| `mission_max_s` | 900 s | whole-mission clock → LAND |
| `land_max_s` | 120 s | LAND without a land-detector trip → DONE (forced DISARM) |
| `margin_m` / `map_half_m` | 1000 / 2500 m | map-edge margin and half-size (from the staged map) |
| `offboard_lost_s` | 3 s | armed+OFFBOARD lost → LAND (external takeover) |
| `margin_persist_s` | 3 s | honest margin breach must persist (no gated-frame flapping) |
| `ev_stale_s` | 0.5 s | `/mcl/odom` silence → EV invalid |
| `preflight_max_s` | 180 s | not armed in time → abort on the ground |

`ev_vz_zero` / `ev_frame:=first_fix` are **SITL-only** hacks and stay off on
the real airframe (the FCU rotates the honest body velocity with attitude).

## 6. Pre-flight checklist

- [ ] map staged for this take-off point, margin report clean (≥1 km to the edge)
- [ ] `EKF2_EV_CTRL=5`, `EKF2_GPS_CTRL=0` on the FCU (and saved)
- [ ] deployment checkpoint (fine-tuned), not a pretrain-only one
- [ ] camera/IMU + MAVROS topics alive at the expected rates
- [ ] `/mcl/health` reaches `tracking` during the climb (SCAN_MIN_COV ≥ 88 m)
- [ ] take-off point map ENU matches `local_origin_map_x/y` and the init lat/lon
- [ ] only one EV publisher running (manager, never the SITL driver)

## 7. Shipping to the drone — the `MCLDrone/` folder

`MCLDrone/` is the deployable subset of this project (same file names and
structure) plus the ROS 2 package and the rig calibration:

```
MCLDrone/
├── RUN.md  COMMANDS.md                    # this doc + the full command reference
├── mcl.py  bag_mcl.py  bag_reader.py  orthoprojection.py  gt_eval.py
├── mission_manager.py  mission_guard.py  test_mission_guard.py
├── prestage_map.py                        # map staging (ground side)
├── sivl/models/orthosimilarity.py         # BranchNet + DecisionNet (CNN)
├── sivl/utils/utils.py
├── checkpoints_huizhou/huizhou_ft_mt_1_epoch_040_of_40.pt   # deployment model
├── maps/z17_5120.png (+ .json)            # staged map + georeference sidecar
├── gnss_free/gnss_free_core/mapdb/build_tile_index.py       # prestage_map helper
└── ovws/src/
    ├── ov_mcl/                            # colcon-buildable node + launch
    └── ov_SchurVINS/config/cyperstereo_012_752x480_equi/    # rig VIO calibration
```

Prerequisites **not** included (bring your own): PX4 + px4_msgs v1.17
(uXRCE-DDS), SchurVINS/`ov_msckf` VIO (install the included config dir into its
`share/ov_msckf/config/`), `MicroXRCEAgent`, MAVROS, the camera/IMU driver, and
python with **numpy + opencv-python + torch (CUDA build on the Jetson) +
matplotlib + pyulog** — `mcl.py` imports matplotlib at module level even for
headless runs, and the **replay** path reads the ULog beside the bag with
pyulog (`from pyulog import ULog` in `mcl_node.py`, for the FCU attitude and
the clock alignment). Live flight does not need pyulog — MAVROS supplies the
attitude.

**No absolute dev paths** — everything is resolved relative to the folder:

| where | how |
|---|---|
| `ovws/src/ov_mcl/ov_mcl/mcl_node.py` | first root that actually holds `mcl.py`, in order: `$MCL_ROOT`, walking up from the file (bundle layout), the dev checkout, `~/MCLDrone` (deployed layout) |
| `mcl.py` | its own directory: `sivl/`, `maps/z17_5120.png`, `checkpoints_huizhou/…`, `mcl_runs/` |
| `mcl_localization.launch.py` | same root search for the map / checkpoint / out_dir defaults — so **no** `checkpoint:=` override is needed on the drone |
| `mcl_node.py` venv | optional: `<root>/venv_mcl`, else the dev venv, else the system python |

So deploy the folder anywhere (`~/MCLDrone` as above, or elsewhere with
`export MCL_ROOT=/path/to/MCLDrone`) — no edits needed. After syncing an
edited `ov_mcl` to the drone, rebuild it, because the launch file and the node
are installed into `share/` and `site-packages`:

```bash
cd ~/ovws && colcon build --packages-select ov_mcl && source install/setup.bash
```

Sanity checks on the drone (no hardware, no bag needed):

```bash
cd ~/MCLDrone
python3 test_mission_guard.py                          # 22 pure-logic scenarios
python3 -c "import sys; sys.path.insert(0,'ovws/src/ov_mcl'); \
  from ov_mcl import mcl_node as n; import mcl; \
  print('root:', n.GNSS_DIR); print('calib:', n._default_calib()); \
  print('map:', mcl.MAP_PATH); print('ckpt:', mcl.CHECKPOINT)"
```
The last line must print paths inside your MCLDrone folder — that proves the
bundle is self-contained before you launch anything.