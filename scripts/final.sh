#!/bin/bash
# Captain/Marshal: tell the robot it is in the final cell (stop + face the opening).
ros2 topic pub --once /quad/final_manual std_msgs/msg/Empty "{}"
