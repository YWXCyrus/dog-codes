#!/bin/bash
# Optional: record a LIGHT rosbag of the survey so we can replay it offline and tune thresholds.
# Usage: bash scripts/record_bag.sh [max_seconds]      (default 240; stop earlier with Ctrl+C)
# Records only /odom /imu/data /joint_states /livox/points /tf /tf_static (no cameras).
# /livox/points is the big one (~5 MB/s), so a 4-minute run is roughly 1 GB.
MAXS=${1:-240}
source /opt/ros/humble/setup.bash 2>/dev/null || { echo "ROS 2 Humble not found"; exit 1; }
ROOT="${DOG_TOOLS_RUNS:-$HOME/dog_tools_runs}"
mkdir -p "$ROOT"
FREE_KB=$(df -k "$ROOT" | awk 'NR==2{print $4}')
NEED_KB=$((MAXS * 6000 + 2000000))
if [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt "$NEED_KB" ]; then
  echo "Not enough free disk for ${MAXS}s (have $(df -h "$ROOT" | awk 'NR==2{print $4}'), want ~$((NEED_KB/1024/1024)) GB). Use a shorter time."
  exit 1
fi
OUT="$ROOT/bag_$(date +%Y%m%d_%H%M%S)"
echo "ROS_DOMAIN_ID = ${ROS_DOMAIN_ID:-<unset>}; recording up to ${MAXS}s into $OUT"
exec timeout -s INT "$MAXS" ros2 bag record -o "$OUT" /odom /imu/data /joint_states /livox/points /tf /tf_static
