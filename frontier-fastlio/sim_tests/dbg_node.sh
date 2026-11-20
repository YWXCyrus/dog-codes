#!/bin/bash
# Run sim + cloud_proc + one dog_codes node, then echo a topic once. Usage: dbg_node.sh node.py /topic secs
source /opt/ros/humble/setup.bash; source ~/ws/install/setup.bash; export ROS_DOMAIN_ID=42; export DOG_DEBUG=1
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # = frontier-fastlio/
pkill -f sim_world.py; pkill -f dog_codes; sleep 1
python3 "$D/sim_tests/sim_world.py" $SIMARGS > /tmp/dbg_sim.log 2>&1 &
python3 -u "$D/dog_codes/cloud_proc.py" > /tmp/dbg_cloud.log 2>&1 &
python3 -u "$D/dog_codes/$1" > /tmp/dbg_node.log 2>&1 &
sleep ${3:-12}
timeout 5 ros2 topic echo --once $2
pkill -f sim_world.py; pkill -f dog_codes
