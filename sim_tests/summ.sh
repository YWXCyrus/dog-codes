#!/bin/bash
# Condensed view of the last run_sim_test logs.
echo "== sim.log (errors/collisions/events)"; grep -E "Traceback|Error|COLLISION|SIM:|rror" /tmp/simtest/sim.log | cut -c1-200 | head -${1:-25}
echo "== mission.log (events)"
grep -E "MODE|STAIRS|CURTAIN|stuck|QR|survey|goto|exhaust|Traceback|Error|rror|REPORT" /tmp/simtest/mission.log | grep -v "M03\|M05" | sed -E 's/\[[0-9.]+\] //' | cut -c1-210 | head -${2:-70}
