#!/bin/bash
# One-shot status of gate + supervisor.
timeout 3 ros2 topic echo --once /quad/status
timeout 3 ros2 topic echo --once /quad/supervisor
