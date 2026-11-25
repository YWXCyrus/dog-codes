#!/usr/bin/env bash
# End-to-end test of perception.launch.py without the dog: fake camera (QR) + fake /odom ->
# perception -> reporter (dry_run, scene queries off) -> the JSON must appear in the reporter log.
# ROS_DOMAIN_ID=42 only. Usage: ROS_DOMAIN_ID=42 bash perception/test_standalone.sh [seconds]
if [ "${ROS_DOMAIN_ID:-}" != "42" ]; then
  echo "refusing: run with ROS_DOMAIN_ID=42 (never on the dog domain)"; exit 2
fi
source /opt/ros/humble/setup.bash
cd "$(dirname "$0")" || exit 1
RUN_S=${1:-25}
TEXT="Survivor is in the kitchen"
LOGS=../frontier-fastlio/dog_codes/run_logs
before=$(ls -1 "$LOGS"/reports_*.jsonl 2>/dev/null | wc -l)

python3 -u fake_feeds.py --text "$TEXT" > /tmp/pst_fake.log 2>&1 &
FAKE=$!
sleep 2
timeout -s INT "$RUN_S" ros2 launch ./perception.launch.py group_id:=group02_TEST dry_run:=true \
  scene_enabled:=false > /tmp/pst_launch.log 2>&1
kill -INT $FAKE 2>/dev/null; wait $FAKE 2>/dev/null

latest=$(ls -1t "$LOGS"/reports_*.jsonl 2>/dev/null | head -1)
after=$(ls -1 "$LOGS"/reports_*.jsonl 2>/dev/null | wc -l)
echo "perception: $(grep -c 'REPORT' /tmp/pst_launch.log) REPORT lines; $(grep -m1 'perception up' /tmp/pst_launch.log | sed 's/.*perception up/perception up/')"
echo "reporter log: $latest"
python3 - "$latest" "$TEXT" "$before" "$after" <<'EOF'
import json, sys
path, text, before, after = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ok = False
if after > before and path:
    for line in open(path):
        d = json.loads(line).get('dry_run')
        if d and d.get('task_type') == 'qr':
            print('payload:', json.dumps(d))
            ok = (d == {'group_id': 'group02_TEST', 'task_id': 'M04', 'task_type': 'qr',
                        'scene_type': d['scene_type'], 'label': '', 'pos': None, 'message': text})
print('PASS' if ok else 'FAIL (see /tmp/pst_launch.log)')
sys.exit(0 if ok else 1)
EOF
