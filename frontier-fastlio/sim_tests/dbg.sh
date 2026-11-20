#!/bin/bash
# Debug harness: sim + cloud_proc + explorer only. Writes /tmp/grid.png. Usage: dbg.sh seconds
source /opt/ros/humble/setup.bash; source ~/ws/install/setup.bash; export ROS_DOMAIN_ID=42
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # = frontier-fastlio/
pkill -f sim_world.py; pkill -f dog_codes; sleep 1
python3 "$D/sim_tests/sim_world.py" > /tmp/dbg_sim.log 2>&1 &
python3 -u "$D/dog_codes/cloud_proc.py" > /tmp/dbg_cloud.log 2>&1 &
python3 -u "$D/dog_codes/explorer.py" --ros-args -p debug_png:=/tmp/grid.png > /tmp/dbg_expl.log 2>&1 &
sleep 5
(while true; do ros2 topic pub --once /quad/mode std_msgs/msg/String "{data: EXPLORE}" >/dev/null; sleep 1; done) &
PUB=$!
sleep ${1:-20}
timeout 3 ros2 topic echo --once /quad/explore_state
kill $PUB; pkill -f sim_world.py; pkill -f dog_codes
tail -5 /tmp/dbg_expl.log /tmp/dbg_cloud.log
