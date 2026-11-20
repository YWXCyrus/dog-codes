#!/bin/bash
# End-to-end test of dog_tools against sim_world (WSL, domain 42 ONLY). Usage: run_survey_sim.sh
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
export ROS_DOMAIN_ID=42
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # = frontier-fastlio/
OUT=${OUT:-/tmp/surveytest}; V=${DRIVE_V:-0.3}
rm -rf "$OUT"; mkdir -p "$OUT/runs"
pkill -f sim_world.py; pkill -f terrain_survey.py; sleep 1
python3 -B "$D/sim_tests/sim_world.py" > "$OUT/sim.log" 2>&1 &
SIM=$!
sleep 4
python3 -B "$D/dog_tools/terrain_survey.py" --ros-args -p run_root:="$OUT/runs" > "$OUT/survey.log" 2>&1 &
SV=$!
sleep 9
( echo 1; sleep 6; echo 5; sleep 2; echo m; sleep 2; echo n curtain ahead; sleep 6; echo -; sleep 4 ) \
  | python3 -B "$D/dog_tools/gait_label.py" "$OUT/runs" > "$OUT/label.log" 2>&1 &
LB=$!
python3 -B "$D/sim_tests/drive_joy.py" 16 $V
sleep 2
ros2 node info /terrain_survey > "$OUT/nodeinfo.txt" 2>&1
sleep 6
kill -INT $SV
wait $SV
kill $SIM 2>/dev/null; wait $LB 2>/dev/null
echo "=== survey log (tail)"; tail -n 25 "$OUT/survey.log"
echo "=== node info"; cat "$OUT/nodeinfo.txt"
echo "=== checks"
[ -n "$SKIP_CHECKS" ] || python3 -B "$D/sim_tests/check_survey_output.py" "$OUT/runs" "$OUT/nodeinfo.txt"
