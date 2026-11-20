#!/bin/bash
# Software stop: zero velocity + latch (needs disarm.sh before re-arming). The physical STOP is separate.
ros2 topic pub --once /quad/stop std_msgs/msg/Empty "{}"
