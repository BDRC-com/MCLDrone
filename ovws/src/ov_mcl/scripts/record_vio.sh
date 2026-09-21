#!/usr/bin/env bash
# Record VIO odometry (/ov_msckf/odomimu) for a drone bag.
#
# Plays the bag (default rate 2x — the VIO processes every message with
# correct stamps, it is not real-time constrained), runs the SchurVINS
# subscribe node, and records its odometry output to a new bag for
# offline use (validate_vio.py / bag_mcl --vio experiments).
#
# Usage:
#   bash record_vio.sh <bag_dir> [out_dir] [rate]
# Example:
#   bash record_vio.sh ~/bags/bag_0001_20260831_190020 ~/bags/vio_bag_0001 2.0
set -eo pipefail

BAG="${1:?usage: record_vio.sh <bag_dir> [out_dir] [rate]}"
OUT="${2:-$(dirname "$BAG")/vio_$(basename "$BAG")}"
RATE="${3:-2.0}"
CONFIG="${VIO_CONFIG:-cyperstereo_012_752x480_equi}"

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
WS="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$WS/install/setup.bash"

rm -rf "$OUT"

# 1) VIO (must be up before the bag starts feeding it)
ros2 launch ov_msckf subscribe.launch.py config:="$CONFIG" \
    > /tmp/record_vio_vio.log 2>&1 &
VIO_PID=$!
sleep 5

# 2) recorder BEFORE play: odomimu is only published while subscribed
#    (sqlite3 storage: the offline tools read .db3, not Jazzy's mcap default)
ros2 bag record --storage sqlite3 -o "$OUT" /ov_msckf/odomimu \
    > /tmp/record_vio_rec.log 2>&1 &
REC_PID=$!
sleep 3

# 3) play the bag (blocks until finished)
ros2 bag play "$BAG" --rate "$RATE"
echo "bag play finished"

# flush trailing messages, then tear down
sleep 8
kill "$REC_PID" 2>/dev/null || true
kill "$VIO_PID" 2>/dev/null || true
pkill -f run_subscribe_msckf 2>/dev/null || true
sleep 2
echo "VIO output recorded to: $OUT"
