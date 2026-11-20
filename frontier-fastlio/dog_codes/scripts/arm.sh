#!/bin/bash
# ARM the autonomy (the dog will stand if sitting and start exploring). Hoist + someone on STOP first.
ros2 topic pub --once /quad/arm std_msgs/msg/Bool "{data: true}"
