#!/usr/bin/env python3
"""cloud_proc: /livox/points (livox_frame) -> /quad/cloud (odom frame, z = height above the
dog's foot plane). Shared by explorer, terrain and curtain so the transform is done once.

Height reference: h = z_world - (robot_z - body_h). body_h (standing body height above the
floor) is calibrated from the floor seen around the robot while standing still
(/quad/recalibrate, also automatically on the first clouds).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

import common as C


class CloudProc(Node):

    def __init__(self):
        super().__init__('cloud_proc')
        p = C.declare(self, {
            'cloud_topic': '/livox/points', 'odom_frame': 'odom', 'sensor_frame': 'livox_frame',
            'max_range': 10.0, 'max_points': 12000, 'body_h_default': 0.30,
            'auto_calibrate_n': 8, 'min_floor_pts': 150,
        })
        self.p = p
        self.body_h = p['body_h_default']
        self.calib_n = 0
        self.calib_samples = []
        self.calibrated = False
        self.robot_z = 0.0
        self.have_odom = False
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)
        self.pub = self.create_publisher(PointCloud2, C.T_CLOUD, C.SENSOR_QOS)
        self.calib_pub = self.create_publisher(C.String, C.T_CALIB, C.LATCHED_QOS)
        self.create_subscription(Odometry, '/odom', self.on_odom, C.SENSOR_QOS)
        self.create_subscription(PointCloud2, p['cloud_topic'], self.on_cloud, C.SENSOR_QOS)
        self.create_subscription(Empty, '/quad/recalibrate', self.on_recal, 10)
        self.rng = np.random.default_rng(1)
        self.last_warn = 0.0
        self.calib_pub.publish(C.jmsg({'body_h': self.body_h, 'calibrated': False}))
        self.get_logger().info('cloud_proc up (%s -> %s)' % (p['cloud_topic'], C.T_CLOUD))

    def on_odom(self, m):
        self.robot_z = m.pose.pose.position.z
        self.px = m.pose.pose.position.x
        self.py = m.pose.pose.position.y
        self.have_odom = True

    def on_recal(self, _):
        self.calibrated = False
        self.calib_n = 0
        self.calib_samples = []
        self.get_logger().info('recalibrating body height')

    def on_cloud(self, msg):
        if not self.have_odom:
            return
        try:
            try:
                tf = self.tfbuf.lookup_transform(self.p['odom_frame'], self.p['sensor_frame'],
                                                 rclpy.time.Time.from_msg(msg.header.stamp))
            except Exception:
                tf = self.tfbuf.lookup_transform(self.p['odom_frame'], self.p['sensor_frame'],
                                                 rclpy.time.Time())
        except Exception as e:
            if time.monotonic() - self.last_warn > 5.0:
                self.last_warn = time.monotonic()
                self.get_logger().warn('no TF %s->%s: %s' % (self.p['odom_frame'],
                                                            self.p['sensor_frame'], e))
            return
        xyz = C.cloud_to_xyz(msg)
        if xyz.shape[0] == 0:
            return
        q = tf.transform.rotation
        t = tf.transform.translation
        R = C.quat_to_rot(q.x, q.y, q.z, q.w).astype(np.float32)
        w = xyz @ R.T + np.array([t.x, t.y, t.z], np.float32)
        d = np.hypot(w[:, 0] - self.px, w[:, 1] - self.py)
        w = w[d < self.p['max_range']]
        if not self.calibrated:
            self.calibrate(w)
        w = w.copy()
        w[:, 2] -= (self.robot_z - self.body_h)
        if w.shape[0] > self.p['max_points']:
            idx = self.rng.choice(w.shape[0], self.p['max_points'], replace=False)
            w = w[idx]
        out = C.xyz_to_cloud(w, self.p['odom_frame'], msg.header.stamp)
        self.pub.publish(out)

    def calibrate(self, w):
        d = np.hypot(w[:, 0] - self.px, w[:, 1] - self.py)
        sel = w[(d > 0.5) & (d < 2.5) & (w[:, 2] < self.robot_z)]
        if sel.shape[0] < self.p['min_floor_pts']:
            return
        ground = float(np.percentile(sel[:, 2], 10))
        bh = self.robot_z - ground
        if not (0.12 < bh < 0.50):
            return
        self.calib_samples.append(bh)
        if len(self.calib_samples) >= self.p['auto_calibrate_n']:
            self.body_h = float(np.median(self.calib_samples))
            self.calibrated = True
            self.calib_pub.publish(C.jmsg({'body_h': round(self.body_h, 3), 'calibrated': True}))
            self.get_logger().info('body height calibrated: %.3f m' % self.body_h)


def main():
    rclpy.init()
    n = CloudProc()
    try:
        C.safe_spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
