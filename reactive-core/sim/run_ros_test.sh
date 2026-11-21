#!/usr/bin/env bash
# ROS wiring test: fake dog + run.py --sim. ONLY on ROS_DOMAIN_ID=42 (WSL), never the dog.
#   ROS_DOMAIN_ID=42 sim/run_ros_test.sh [default|stairs] [run seconds]
if [ "${ROS_DOMAIN_ID:-}" != "42" ]; then
  echo "refusing: run with ROS_DOMAIN_ID=42 (this test must never reach the dog)"; exit 2
fi
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
set -u
cd "$(dirname "$0")/.." || exit 1
WORLD=${1:-default}
RUN_S=${2:-60}
OUT=$(mktemp -d /tmp/rc_rostest_XXXX)
echo "output: $OUT"

python3 -B sim/fake_dog_node.py --world "$WORLD" --sitting --summary "$OUT/dog.json" > "$OUT/dog.log" 2>&1 &
DOG=$!
sleep 3
timeout -s INT $((RUN_S + 45)) python3 -B run.py --sim --auto-go --no-viewer \
  --set run_time_s="$RUN_S" --runs-dir "$OUT/runs" < /dev/null > "$OUT/run.log" 2>&1
RC=$?
sleep 1
kill -INT $DOG
wait $DOG
echo "run.py exit code: $RC (124 = stopped by timeout/SIGINT as intended)"
tail -n 25 "$OUT/run.log"

python3 - "$OUT" "$WORLD" <<'EOF'
import glob, json, os, sys
out, world = sys.argv[1], sys.argv[2]
dog = json.load(open(os.path.join(out, 'dog.json')))
runs = glob.glob(os.path.join(out, 'runs', '*'))
fails = []
def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond: fails.append(msg)
codes = [c[1] for c in dog['codes']]
check(len(runs) == 1, 'one run folder')
r = runs[0] if runs else ''
check(os.path.exists(os.path.join(r, 'decisions.jsonl')), 'decisions.jsonl written')
check(os.path.exists(os.path.join(r, 'snapshots.jsonl')), 'snapshots.jsonl written')
check(os.path.exists(os.path.join(r, 'bag', 'metadata.yaml')), 'recorder bag closed (metadata.yaml)')
check(codes.count('stand_sit') == 1, 'exactly one Stand/Sit toggle (dog started sitting): %s' % codes)
check(dog['posture'] == 'standing', 'dog standing at the end')
check(dog['collisions'] == 0, 'no collisions (%d)' % dog['collisions'])
check(not dog['reversed'], 'never reversed')
check(dog['n_cmd'] > 100, 'commands received (%d)' % dog['n_cmd'])
check(dog['last_cmd'] == [0.0, 0.0], 'last command after Ctrl+C is zero: %s' % dog['last_cmd'])
if world == 'stairs':
    check('enter_ai' in codes and 'exit_ai' in codes, 'AI entered and exited: %s' % codes)
    check(dog['max_support'] >= 0.44, 'reached the platform (%.2f)' % dog['max_support'])
print('RESULT:', 'FAIL' if fails else 'ALL PASS')
sys.exit(1 if fails else 0)
EOF
