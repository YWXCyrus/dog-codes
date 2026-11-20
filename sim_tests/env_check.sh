#!/bin/bash
python3 -c 'import scipy,yaml,cv2,numpy;print("scipy",scipy.__version__,"cv2",cv2.__version__,"numpy",numpy.__version__)'
source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash
python3 -c 'import transfer_interfaces.msg as m;print(m.MotionSimpleCMD)'
python3 -c 'import cv2;print("QRenc",hasattr(cv2,"QRCodeEncoder_create"),"Aruco",hasattr(cv2,"QRCodeDetectorAruco"))'
ros2 pkg list | grep -E "^(tf2_ros|launch_ros|twist_mux)$"
