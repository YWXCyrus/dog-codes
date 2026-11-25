#!/usr/bin/env bash
# End-to-end test of the navigation cues without the dog: fake camera (QR) + fake /odom + fake camera
# model detections (stairs) -> perception -> stairs_seen event, stairs answer, /quad/mission_status.
# ROS_DOMAIN_ID=42 only. Usage: ROS_DOMAIN_ID=42 bash perception/test_cues_ros.sh
if [ "${ROS_DOMAIN_ID:-}" != "42" ]; then
  echo "refusing: run with ROS_DOMAIN_ID=42 (never on the dog domain)"; exit 2
fi
source /opt/ros/humble/setup.bash
cd "$(dirname "$0")" || exit 1
python3 -u fake_feeds.py > /tmp/pcue_fake.log 2>&1 &
FAKE=$!
setsid ros2 launch ./perception.launch.py group_id:=group02_TEST dry_run:=true scene_enabled:=false > /tmp/pcue_launch.log 2>&1 &
LAUNCH=$!
sleep 4
python3 -u check_cues.py 16
RC=$?
# the launch runs in its own process group (setsid): stop exactly that group
kill -INT -- -$LAUNCH 2>/dev/null; kill -INT $FAKE 2>/dev/null
for i in 1 2 3 4 5 6; do kill -0 $LAUNCH 2>/dev/null || break; sleep 1; done
kill -TERM -- -$LAUNCH 2>/dev/null; kill -TERM $FAKE 2>/dev/null
wait $LAUNCH $FAKE 2>/dev/null
exit $RC
