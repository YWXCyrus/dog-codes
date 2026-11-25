#!/usr/bin/env python3
"""Fake camera + /odom for testing perception.launch.py WITHOUT the dog (ROS_DOMAIN_ID=42 only).

Publishes a 640x480 rgb8 camera frame with a QR code (text --text, ~250 px wide) on
/camera/color/image_raw at 10 Hz and a still nav_msgs/Odometry on /odom at 20 Hz.
"""
import argparse
import os
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'qr-test'))


def frame_with_qr(text, width_px):
    import qr_test as T
    import qr_encode as E
    bgr = T.place(E.render(E.encode(text)), width_px)
    return bgr[:, :, ::-1].copy()                     # -> rgb8 like the RealSense driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='Survivor is in the kitchen')
    ap.add_argument('--width', type=float, default=250.0)
    a = ap.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '42':
        print('refusing: run with ROS_DOMAIN_ID=42 (never on the dog domain)')
        return 2
    rgb = frame_with_qr(a.text, a.width)
    rclpy.init()
    n = Node('fake_feeds')
    ip = n.create_publisher(Image, '/camera/color/image_raw', 5)
    op = n.create_publisher(Odometry, '/odom', 20)

    def pub_img():
        m = Image()
        m.header.stamp = n.get_clock().now().to_msg()
        m.header.frame_id = 'camera_color_optical_frame'
        m.height, m.width, m.encoding, m.step = 480, 640, 'rgb8', 640 * 3
        m.data = rgb.tobytes()
        ip.publish(m)

    def pub_odom():
        m = Odometry()
        m.header.stamp = n.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = 'odom', 'base_link'
        m.pose.pose.orientation.w = 1.0
        op.publish(m)

    n.create_timer(0.1, pub_img)
    n.create_timer(0.05, pub_odom)
    print('fake camera (QR %r, %.0f px) + /odom on domain 42' % (a.text, a.width), flush=True)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
