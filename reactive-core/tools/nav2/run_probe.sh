#!/usr/bin/env bash
# Nav2-as-planner viability experiment (WSL, ROS_DOMAIN_ID 42, never on the dog's domain 1).
#   bash tools/nav2/run_probe.sh
# Starts map_server + planner_server (+ lifecycle manager + a static map->base_link TF), runs probe.py, stops all.

cd "$(dirname "$0")/../.."                     # reactive-core/
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
OUT=${TMPDIR:-/tmp}/nav2_probe
mkdir -p "$OUT"
python3 tools/nav2/make_map.py "$OUT" arena || exit 1
sed "s#MAP_YAML#$OUT/map.yaml#" tools/nav2/nav2_params.yaml > "$OUT/params.yaml"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link > "$OUT/tf.log" 2>&1 & pids+=($!)
ros2 run nav2_map_server map_server --ros-args --params-file "$OUT/params.yaml" > "$OUT/map_server.log" 2>&1 & pids+=($!)
ros2 run nav2_planner planner_server --ros-args --params-file "$OUT/params.yaml" > "$OUT/planner.log" 2>&1 & pids+=($!)
ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args --params-file "$OUT/params.yaml" \
  -r __node:=lifecycle_manager_planner > "$OUT/lifecycle.log" 2>&1 & pids+=($!)
sleep 8
python3 tools/nav2/probe.py
rc=$?
echo "(logs in $OUT)"
exit $rc
