#!/bin/bash
# Disarm (clears a latched fault too).
ros2 topic pub --once /quad/arm std_msgs/msg/Bool "{data: false}"
