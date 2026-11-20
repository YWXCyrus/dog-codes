#!/bin/bash
# Rebuild the report (summary tables + map PNGs) of a survey run. No ROS needed.
# Usage: bash scripts/make_report.sh [run_folder]     (default: the latest run)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${DOG_TOOLS_RUNS:-$HOME/dog_tools_runs}"
RUN="$1"
[ -z "$RUN" ] && RUN="$(cat "$ROOT/LATEST" 2>/dev/null)"
[ -d "$RUN" ] || { echo "no run folder found (give it as argument)"; exit 1; }
exec python3 -B "$HERE/report.py" "$RUN"
