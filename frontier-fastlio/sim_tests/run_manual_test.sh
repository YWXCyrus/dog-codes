#!/bin/bash
# Joystick takeover test: while the mission is running publish /joy_cmd_vel for ~2 s, expect PAUSED then resume.
source /opt/ros/humble/setup.bash; source ~/ws/install/setup.bash; export ROS_DOMAIN_ID=42
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # = frontier-fastlio/
pkill -f sim_world.py; pkill -f dog_codes; sleep 1
python3 "$D/sim_tests/sim_world.py" > /tmp/m_sim.log 2>&1 &
sleep 4
ros2 launch "$D/dog_codes/mission.launch.py" group_id:=group02_TEST scene_server:=http://127.0.0.1:8000 report_url:=http://127.0.0.1:8001/task_report > /tmp/m_mission.log 2>&1 &
ML=$!
sleep 12
ros2 topic pub --once /quad/arm std_msgs/msg/Bool "{data: true}" >/dev/null
sleep 25
echo "--- joystick takeover"
timeout 2.5 ros2 topic pub -r 10 /joy_cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}" >/dev/null
sleep 1.5
echo "mode during/after: $(timeout 2 ros2 topic echo --once /quad/supervisor | grep data)"
sleep 6
echo "mode 6 s later:   $(timeout 2 ros2 topic echo --once /quad/supervisor | grep data)"
grep -E "MANUAL|resuming|PAUSED" /tmp/m_mission.log | sed -E 's/\[[0-9.]+\] //' | cut -c1-160
kill -INT $ML; sleep 2; pkill -f sim_world.py; pkill -f dog_codes
