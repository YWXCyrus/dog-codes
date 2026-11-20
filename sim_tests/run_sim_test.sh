#!/bin/bash
# End-to-end run of dog_codes against sim_world (WSL, domain 42). Usage: run_sim_test.sh [duration_s] [sim args...]
DUR=${1:-150}; shift
SIMARGS="$@"
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
export ROS_DOMAIN_ID=42
export DOG_DEBUG_PNG=/tmp/grid.png
D="/mnt/c/Users/Admin/Desktop/IR 2D/codes"
DC="${DOG_CODES:-$D/dog_codes}"      # override to run another copy of dog_codes (A/B tests)
LOG=/tmp/simtest; mkdir -p $LOG; rm -f $LOG/*.log
pkill -f sim_world.py; pkill -f dog_codes; pkill -f mission.launch; sleep 1
python3 "$D/sim_tests/sim_world.py" $SIMARGS > $LOG/sim.log 2>&1 &
SIM=$!
sleep 4
ros2 launch "$DC/mission.launch.py" group_id:=group02_TEST scene_server:=http://127.0.0.1:8000 report_url:=http://127.0.0.1:8001/task_report > $LOG/mission.log 2>&1 &
ML=$!
sleep 14
echo "--- arming"
ros2 topic pub --once /quad/arm std_msgs/msg/Bool "{data: true}" > /dev/null
T=0
while [ $T -lt $DUR ]; do
  sleep 10; T=$((T+10))
  echo "t=$T $(timeout 2 ros2 topic echo --once /quad/supervisor 2>/dev/null | grep data | head -1)"
done
echo "--- summary"
curl -s http://127.0.0.1:8001/summary | python3 -m json.tool
kill -INT $ML; sleep 2; kill $SIM 2>/dev/null; pkill -f dog_codes; pkill -f sim_world.py
echo "logs in $LOG"
