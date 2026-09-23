#!/usr/bin/env bash
#
# run_live.sh — live MCL localization + VIO + FCU bridge + bag recording
#               in ONE command (pure PX4 uXRCE-DDS vehicle, NO MAVROS).
#
# Starts (in order):
#   1. MicroXRCE-DDS agent      (unless --no-agent; it bridges the FCU)
#   2. TWO ros2 bag record      (unless --no-record; mcap): one for camera
#      images, one for high-rate small messages (imu/odom/fcu) — splitting
#      prevents IMU burst drops under image-serialize load (see IMG_TOPICS).
#      --topics "..." forces a single recorder instead.
#   3. fcu_pose_bridge.py       (unless --no-bridge; FCU NED/FRD -> ENU/FLU)
#   4. ros2 launch ov_mcl       (SchurVINS + MCL node, LIVE mode: no bag:=)
#
# --daemon / -D: detach the WHOLE stack (new session) so an SSH disconnect
#                 cannot stop it in flight; console -> /tmp/run_live_<name>.
#                 console.log. Control with --status and --stop (same --name).
#
# Ctrl-C once -> SIGINT the launch (MCL flushes replay_log.npz) -> stops the
# recorder cleanly -> kills SchurVINS/bridge/agent that survive launch exit.
#
# Prerequisites (power/bring-up, NOT started here):
#   - FCU powered (agent sees a client from 192.168.2.103)
#   - CyperStereo camera bridge publishing /cam0 /cam1 /imu0 on the SAME
#     ROS_DOMAIN_ID (default 0): start it in a domain-0 terminal
#
# Examples:
#   ./run_live.sh --name field01 --lat 22.842897 --lon 114.525573
#   ./run_live.sh --name bench01 --no-record                 # monitor only
#   ./run_live.sh --name t2 --lat .. --topics "/cam0/image_raw /imu0 /mcl/odom"
#   ./run_live.sh --name scan01 --init-mode scan
#
set -o pipefail
# This script always runs under bash (shebang), but a zsh parent exports
# ZSH_VERSION/ZSH_NAME (and sometimes BASH_ENV): Humble's ament setup.bash
# then redirects to setup.zsh, which BASH cannot parse ("bad substitution",
# "cd: -q"). Clear all zsh leakage before sourcing anything. ROS setup files
# are not `set -u`-clean either, so nounset stays off.
unset ZSH_VERSION ZSH_NAME BASH_ENV AMENT_CURRENT_PREFIX 2>/dev/null || true

# ----------------------------- configurable defaults ------------------------
NAME="live_$(date +%Y%m%d_%H%M%S)"
DOMAIN=0
BAG_ROOT="$HOME/ros2bag                   # ros2 bag output root
MCL_ROOT="$HOME/ros2bag/mcl_runs"         # MCL replay_log.npz / plots root
MAP_PATH="$HOME/MCLDrone/maps/z17_5120.png"
OFF_X="-27449088.0"
OFF_Y="-14586624.0"
CHECKPOINT=""                     # "" = launch default (fine-tuned ckpt)
INIT_MODE="point"
INIT_LAT="-999.0"
INIT_LON="-999.0"
INIT_RADIUS="300.0"
STRIDE="5"
YAW_CAL="-1,1.3"
FCU_TOPIC="/fcu/local_position/pose"
EXTRA_ARGS=""
# Recorders are SPLIT by design: heavy image frames in one bag, high-rate
# small messages in another. A single recorder loses IMU samples (the CyperStereo
# delivers /imu0 in ~50 ms bursts into a keep-last queue; while the writer
# serializes a 360 KB camera frame the burst is evicted — observed: 195 Hz
# published, only ~27 Hz captured). Two writers = two queues/two threads.
IMG_TOPICS="/cam0/image_raw /cam1/image_raw"
DATA_TOPICS="/imu0 \
/ov_msckf/odomimu /mcl/odom /mcl/health \
/fcu/local_position/pose \
/fmu/out/vehicle_local_position_v1 /fmu/out/vehicle_attitude \
/fmu/out/vehicle_status_v1 /fmu/in/vehicle_visual_odometry"
SINGLE_TOPICS=""               # non-empty -> ONE recorder with these topics

START_AGENT=1
START_BRIDGE=1
DO_RECORD=1
DAEMON=0
DO_STOP=0
DO_STATUS=0
ORIG_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$SCRIPT_DIR/fcu_pose_bridge.py"
# FastDDS UDP-only profile for the small-message recorder (SHM sheds bursty
# high-rate streams on FastDDS 2.6 — see fastdds_udp_only.xml header)
UDP_PROFILE="$SCRIPT_DIR/fastdds_udp_only.xml"
AGENT_BIN="$HOME/my_px4/install/microxrcedds_agent/bin/MicroXRCEAgent"
ROS_SETUP="/opt/ros/humble/setup.bash"
PX4_SETUP="$HOME/my_px4/install/setup.bash"
MCL_SETUP="$SCRIPT_DIR/ovws/install/setup.bash"

usage() { sed -n '2,36p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --name)        NAME="$2"; shift 2;;
    --domain)      DOMAIN="$2"; shift 2;;
    --bag-root)    BAG_ROOT="$2"; shift 2;;
    --mcl-root)    MCL_ROOT="$2"; shift 2;;
    --map)         MAP_PATH="$2"; shift 2;;
    --off-x)       OFF_X="$2"; shift 2;;
    --off-y)       OFF_Y="$2"; shift 2;;
    --checkpoint)  CHECKPOINT="$2"; shift 2;;
    --init-mode)   INIT_MODE="$2"; shift 2;;
    --lat)         INIT_LAT="$2"; shift 2;;
    --lon)         INIT_LON="$2"; shift 2;;
    --radius)      INIT_RADIUS="$2"; shift 2;;
    --stride)      STRIDE="$2"; shift 2;;
    --yaw-cal)     YAW_CAL="$2"; shift 2;;
    --fcu-topic)   FCU_TOPIC="$2"; shift 2;;
    --img-topics)  IMG_TOPICS="$2"; shift 2;;
    --data-topics) DATA_TOPICS="$2"; shift 2;;
    --topics)      SINGLE_TOPICS="$2"; shift 2;;
    --extra)       EXTRA_ARGS="$2"; shift 2;;
    --no-agent)    START_AGENT=0; shift;;
    --no-bridge)   START_BRIDGE=0; shift;;
    --no-record)   DO_RECORD=0; shift;;
    --daemon|-D)   DAEMON=1; shift;;
    --stop)        DO_STOP=1; shift;;
    --status)      DO_STATUS=1; shift;;
    -h|--help)     usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage; exit 1;;
  esac
done

PID_FILE="/tmp/run_live_${NAME}.pid"
CONSOLE_LOG="/tmp/run_live_${NAME}.console.log"

# ----------------------------- control commands -----------------------------
if [ "$DO_STATUS" = 1 ]; then
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    pid=$(cat "$PID_FILE")
    echo "run_live '$NAME' RUNNING (pid $pid) — console: $CONSOLE_LOG"
    ps -o pid,etime,cmd -g "$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null | \
      grep -E 'ros2|MicroXRCE|fcu_pose|run_live' | head -12
    echo "--- last console lines ---"; tail -8 "$CONSOLE_LOG" 2>/dev/null
  else
    echo "run_live '$NAME' not running"
    rm -f "$PID_FILE" 2>/dev/null
  fi
  exit 0
fi

if [ "$DO_STOP" = 1 ]; then
  if [ ! -f "$PID_FILE" ] || ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "run_live '$NAME' not running"
    rm -f "$PID_FILE" 2>/dev/null
    exit 0
  fi
  pid=$(cat "$PID_FILE")
  echo "SIGINT -> $pid (npz flush + recorder close can take ~30 s) ..."
  kill -INT "$pid" 2>/dev/null
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill "$pid" 2>/dev/null
  rm -f "$PID_FILE" 2>/dev/null
  echo "stopped. tail of console:"
  tail -10 "$CONSOLE_LOG" 2>/dev/null
  exit 0
fi

# ----------------------------- detach for flight ----------------------------
# Re-exec ourselves in a new session so loss of SSH (SIGHUP) cannot stop the
# flight stack or recorders; stdout/stderr go to a console log. The PID file
# is written by the detached child and used by --stop/--status.
if [ "$DAEMON" = 1 ] && [ "${RL_DAEMONIZED:-0}" != 1 ]; then
  echo "daemonizing run_live '$NAME' — survives SSH disconnect"
  echo "  console : $CONSOLE_LOG"
  echo "  stop    : $0 --name $NAME --stop"
  echo "  status  : $0 --name $NAME --status"
  exec setsid env RL_DAEMONIZED=1 "$0" "${ORIG_ARGS[@]}" \
    </dev/null >>"$CONSOLE_LOG" 2>&1
fi
if [ "$DAEMON" = 1 ]; then
  echo $$ > "$PID_FILE"
  echo "=== detached pid $$ started $(date) ==="
fi

# ----------------------------- environment ----------------------------------
for f in "$ROS_SETUP" "$PX4_SETUP" "$MCL_SETUP"; do
  if [ ! -f "$f" ]; then
    echo "[run_live] FATAL: missing ROS setup file: $f" >&2
    echo "           (build the deployed ov_mcl first: colcon build in $SCRIPT_DIR/ovws)" >&2
    exit 1
  fi
  unset AMENT_CURRENT_PREFIX        # each setup file derives its own prefix
  # shellcheck disable=SC1090
  source "$f"
done
export ROS_DOMAIN_ID="$DOMAIN"
export ROS_LOG_DIR=/tmp/roslog
mkdir -p "$BAG_ROOT" "$MCL_ROOT"

OUT_DIR="$MCL_ROOT/$NAME"
BAG_IMG_DIR="$BAG_ROOT/${NAME}_img"
BAG_DATA_DIR="$BAG_ROOT/${NAME}_data"
BAG_DIR="$BAG_ROOT/${NAME}_bag"       # single-recorder mode (--topics)
LOG_DIR="/tmp/run_live_$NAME"
mkdir -p "$OUT_DIR" "$LOG_DIR"

AGENT_PID=""; REC_PIDS=""; BRIDGE_PID=""; LAUNCH_PID=""
CLEANED=0

log() { echo "[run_live] $*"; }

stop_recorder() {
  # $1 = pid: SIGINT (flush mcap tail), wait, then KILL
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 0
  kill -INT "$pid" 2>/dev/null
  for _ in $(seq 1 15); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill "$pid" 2>/dev/null
}

cleanup() {
  [ "$CLEANED" = 1 ] && return
  CLEANED=1
  log "shutting down (MCL npz flush can take ~15 s) ..."
  # 1. SIGINT ros2 launch -> mcl_node writes replay_log.npz on shutdown.
  #    Signal the launch process itself (in backgrounded/daemon mode the
  #    tracked PID is the tee at the end of the pipeline), then the pipe.
  pkill -INT -f 'ros2 launch ov_mcl mcl_localization' 2>/dev/null
  if [ -n "$LAUNCH_PID" ] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -INT "$LAUNCH_PID" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "$LAUNCH_PID" 2>/dev/null || break
      sleep 1
    done
    kill "$LAUNCH_PID" 2>/dev/null
  fi
  # 2. stop recorder(s) so the bag tail is flushed (no useless tail data)
  for pid in $REC_PIDS; do stop_recorder "$pid"; done
  # 3. SchurVINS + bridge + agent do not always die with the launch
  pkill -f run_subscribe_msckf 2>/dev/null
  [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" 2>/dev/null
  if [ "$START_AGENT" = 1 ] && [ -n "$AGENT_PID" ]; then
    kill "$AGENT_PID" 2>/dev/null
  fi
  log "done. MCL output: $OUT_DIR"
  if [ "$DO_RECORD" = 1 ]; then
    if [ -n "$SINGLE_TOPICS" ]; then log "bag: $BAG_DIR"; else
      log "bags: $BAG_IMG_DIR  +  $BAG_DATA_DIR"; fi
  fi
  [ "$DAEMON" = 1 ] && rm -f "$PID_FILE"
}
trap cleanup INT TERM

# ----------------------------- preflight ------------------------------------
if ! ros2 topic list 2>/dev/null | grep -q '^/cam0/image_raw$'; then
  log "WARNING: /cam0/image_raw not visible on domain $DOMAIN — start the"
  log "         camera bridge in another terminal (same ROS_DOMAIN_ID=$DOMAIN)."
fi
if [ "$START_BRIDGE" = 1 ] && [ ! -f "$BRIDGE" ]; then
  log "FATAL: bridge not found: $BRIDGE"; exit 1
fi

# ----------------------------- 1. agent -------------------------------------
if [ "$START_AGENT" = 1 ]; then
  if ros2 topic list 2>/dev/null | grep -q '^/fmu/out/vehicle_attitude$'; then
    log "agent already serving /fmu topics — not starting another"
    START_AGENT=0
  else
    if [ ! -x "$AGENT_BIN" ]; then
      log "FATAL: agent binary not found: $AGENT_BIN"; exit 1
    fi
    log "starting uXRCE-DDS agent (domain $DOMAIN, port 8888)"
    "$AGENT_BIN" udp4 -p 8888 -d 0 >"$LOG_DIR/agent.log" 2>&1 &
    AGENT_PID=$!
    sleep 3
    if ! ros2 topic list 2>/dev/null | grep -q '^/fmu/out/vehicle_attitude$'; then
      log "WARNING: no /fmu/out/vehicle_attitude yet (FCU powered? client connected?)"
      log "         see $LOG_DIR/agent.log — continuing anyway"
    fi
  fi
fi

# ----------------------------- 2. recorder(s) -------------------------------
if [ "$DO_RECORD" = 1 ]; then
  start_rec() {  # $1 = bag dir, $2 = udp_only|shm, rest = topics
    local bdir="$1"; local transport="$2"; shift 2
    local env=()
    if [ "$transport" = udp_only ] && [ -f "$UDP_PROFILE" ]; then
      env=(env FASTRTPS_DEFAULT_PROFILES_FILE="$UDP_PROFILE")
    fi
    "${env[@]}" ros2 bag record -s mcap -o "$bdir" "$@" \
        >"$LOG_DIR/record_$(basename "$bdir").log" 2>&1 &
    local pid=$!
    sleep 3
    if ! kill -0 "$pid" 2>/dev/null; then
      log "FATAL: recorder for $bdir failed — $LOG_DIR/record_$(basename "$bdir").log"
      cleanup; exit 1
    fi
    REC_PIDS="$REC_PIDS $pid"
    log "recording -> $bdir [$transport] ($* )"
  }
  if [ -n "$SINGLE_TOPICS" ]; then
    # shellcheck disable=SC2086
    start_rec "$BAG_DIR" shm $SINGLE_TOPICS
  else
    # shellcheck disable=SC2086
    start_rec "$BAG_IMG_DIR"  shm      $IMG_TOPICS
    # shellcheck disable=SC2086
    start_rec "$BAG_DATA_DIR" udp_only $DATA_TOPICS
  fi
fi

# ----------------------------- 3. FCU bridge --------------------------------
if [ "$START_BRIDGE" = 1 ]; then
  python3 "$BRIDGE" >"$LOG_DIR/bridge.log" 2>&1 &
  BRIDGE_PID=$!
  sleep 2
  if ! timeout 8 ros2 topic hz "$FCU_TOPIC" 2>/dev/null | grep -q average; then
    log "WARNING: no data on $FCU_TOPIC yet (z_valid? attitude?) — MCL will"
    log "         fall back to VIO attitude/alt until it appears"
  else
    log "FCU bridge live on $FCU_TOPIC (~50 Hz)"
  fi
fi

# ----------------------------- 4. VIO + MCL ---------------------------------
LAUNCH_ARGS=(map_path:="$MAP_PATH"
             map_geo_offset_x:="$OFF_X" map_geo_offset_y:="$OFF_Y"
             init_mode:="$INIT_MODE"
             init_lat:="$INIT_LAT" init_lon:="$INIT_LON"
             init_radius_m:="$INIT_RADIUS"
             stride:="$STRIDE" yaw_cal:="$YAW_CAL"
             fcu_alt_topic:="$FCU_TOPIC"
             out_dir:="$OUT_DIR")
[ -n "$CHECKPOINT" ] && LAUNCH_ARGS+=(checkpoint:="$CHECKPOINT")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && LAUNCH_ARGS+=( $EXTRA_ARGS )

log "launching MCL (live) -> $OUT_DIR"
log "args: ${LAUNCH_ARGS[*]}"
ros2 launch ov_mcl mcl_localization.launch.py "${LAUNCH_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/mcl_launch.log" &
LAUNCH_PID=$!
wait "$LAUNCH_PID"
log "launch exited"
cleanup
