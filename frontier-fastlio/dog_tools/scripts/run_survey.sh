#!/bin/bash
# Start the READ-ONLY terrain survey (terminal 2 of the survey procedure).
# Usage: bash scripts/run_survey.sh [-p name:=value ...]      e.g.  -p body_h:=0.30
# It subscribes to /odom, /imu/data, /livox/points + TF and publishes NOTHING.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash 2>/dev/null || { echo "ROS 2 Humble not found"; exit 1; }
echo "ROS_DOMAIN_ID = ${ROS_DOMAIN_ID:-<unset>}   (must be the dog's domain, same as the bring-up terminal)"
python3 -c "import numpy, cv2" 2>/dev/null || { echo "python3 needs numpy and cv2 (OpenCV)"; exit 1; }
FREE_KB=$(df -k "$HOME" | awk 'NR==2{print $4}')
if [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt 2000000 ]; then
  echo "WARNING: less than 2 GB free in \$HOME ($(df -h "$HOME" | awk 'NR==2{print $4}')). Free some space first."
  exit 1
fi
cd "$HERE" && exec python3 -B terrain_survey.py --ros-args "$@"
