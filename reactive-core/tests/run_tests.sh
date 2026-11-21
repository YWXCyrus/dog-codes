#!/usr/bin/env bash
# Run the reactive-core test suite (pure Python, no ROS needed). Usage: tests/run_tests.sh [pytest args]
cd "$(dirname "$0")/.." || exit 1
exec python3 -B -m pytest -q -p no:cacheprovider tests "$@"
