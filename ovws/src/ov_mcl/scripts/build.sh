#!/usr/bin/env bash
# Build the ov_mcl package in the workspace that contains it
# (…/MCLDrone/ovws on the drone, ~/workspace/ovws on the dev machine).
#
# SchurVINS itself is already built there; this only rebuilds ov_mcl.
# To rebuild everything: cd <workspace> && colcon build
#
# Run after changing ov_mcl:
#   bash src/ov_mcl/scripts/build.sh
#
# Then in every terminal (substitute your ROS distro):
#   source /opt/ros/$ROS_DISTRO/setup.bash
#   source <workspace>/install/setup.bash
#   ros2 launch ov_mcl mcl_localization.launch.py
# NOTE: no `set -u` — ROS2 setup.bash references unbound vars
set -eo pipefail

WS="$(cd "$(dirname "$0")/../../.." && pwd)"
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
cd "$WS"
colcon build --packages-select ov_mcl "$@"
