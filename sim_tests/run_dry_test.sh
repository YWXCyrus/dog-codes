#!/bin/bash
source /opt/ros/humble/setup.bash; source ~/ws/install/setup.bash; export ROS_DOMAIN_ID=42
D="/mnt/c/Users/Admin/Desktop/IR 2D/codes"
pkill -f safety_gate.py; sleep 1
python3 -u "$D/dog_codes/safety_gate.py" --ros-args -p dry_run:=true > /tmp/gate_dry.log 2>&1 &
G=$!
sleep 3
ros2 topic pub --once /quad/arm std_msgs/msg/Bool "{data: true}" >/dev/null
(ros2 topic pub -r 10 /quad/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}" >/dev/null &) 
sleep 1
N=$(timeout 3 ros2 topic echo /nav_cmd_vel 2>/dev/null | grep -c linear)
ros2 topic pub --once /quad/gait_req std_msgs/msg/String "{data: hstep}" >/dev/null
sleep 1
echo "dry_run: messages on /nav_cmd_vel while armed+commanded = $N (expect 0)"
grep -E "dry_run" /tmp/gate_dry.log | tail -2
pkill -f "ros2 topic pub"; kill $G
