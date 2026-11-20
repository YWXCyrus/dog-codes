#!/usr/bin/env python3
"""Plays the part of the controller in the SIMULATOR ONLY (domain 42): /joy_cmd_vel at 20 Hz.
Usage: drive_joy.py <seconds> <vx> [wz]"""
import os
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '42':
        print('refusing: ROS_DOMAIN_ID must be 42 (simulator domain)')
        return 2
    secs = float(sys.argv[1])
    vx = float(sys.argv[2])
    wz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    rclpy.init()
    n = Node('drive_joy')
    pub = n.create_publisher(Twist, '/joy_cmd_vel', 10)
    t0 = time.time()
    while time.time() - t0 < secs:
        m = Twist()
        m.linear.x, m.angular.z = vx, wz
        pub.publish(m)
        time.sleep(0.05)
    pub.publish(Twist())
    n.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
