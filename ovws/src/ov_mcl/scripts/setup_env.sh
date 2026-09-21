#!/usr/bin/env bash
# One-time setup for the online MCL node.
#
# Creates a python3.12 venv holding torch (CUDA) + opencv for the ROS2
# system python (rclpy itself comes from /opt/ros/jazzy via PYTHONPATH;
# the node appends the venv site-packages to sys.path at startup).
#
# Disk note: the CUDA torch install is ~6-7 GB. For a CPU-only variant
# replace the pip line with:
#   pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # ovws
source /opt/ros/jazzy/setup.bash

if [ -d "$WS/venv_mcl" ]; then
    echo "venv already exists at $WS/venv_mcl (delete it to recreate)"
else
    python3 -m venv --system-site-packages "$WS/venv_mcl"
fi

"$WS/venv_mcl/bin/pip" install --no-cache-dir torch opencv-python-headless

# verify rclpy (ROS2) + torch + cv2 coexist in one interpreter
PYTHONPATH="$WS/venv_mcl/lib/python3.12/site-packages:$PYTHONPATH" \
    python3 -c "import rclpy, torch, cv2, numpy; \
print('OK: rclpy | torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'| cv2', cv2.__version__)"
