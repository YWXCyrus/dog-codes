#!/bin/bash
source /opt/ros/humble/setup.bash; source ~/ws/install/setup.bash; export ROS_DOMAIN_ID=42
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # = frontier-fastlio/
pkill -f safety_gate.py; sleep 1
python3 -u "$D/dog_codes/safety_gate.py" > /tmp/gate.log 2>&1 &
G=$!
sleep 3
python3 "$D/sim_tests/test_gate.py"
kill $G
