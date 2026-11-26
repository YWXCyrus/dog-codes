#!/usr/bin/env python3
"""Fake Lite3 for the ROS wiring test (ROS_DOMAIN_ID must NOT be 61 - the dog - or 1, its old domain).

Publishes like the dog: /livox/points (PointCloud2, livox_frame, 10 Hz), /imu/data, /joint_states,
/leg_odom (PoseWithCovarianceStamped), /tf_static. Subscribes /nav_cmd_vel (stands in for the TA
twist_mux + the dog; 0.5 s timeout like twist_mux) and /simple_cmd. Writes a JSON summary on exit.

  python3 sim/fake_dog_node.py --world default|stairs [--sitting] [--ai-ignores-cmd] --summary out.json
"""
import argparse
import json
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Imu, JointState, PointCloud2  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from transfer_interfaces.msg import MotionSimpleCMD  # noqa: E402

from rc.pc2 import xyz_to_cloud  # noqa: E402
from sim import world as W  # noqa: E402
from sim.dogmodel import DogModel  # noqa: E402
from sim.lidar import Lidar  # noqa: E402

BE = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class FakeDog(Node):

    def __init__(self, a):
        super().__init__('fake_dog')
        self.world = W.stairs_world() if a.world == 'stairs' else W.arena_default()
        self.dog = DogModel(self.world, sitting=a.sitting, ai_ignores_cmd=a.ai_ignores_cmd)
        self.lidar = Lidar()
        self.cmd = (0.0, 0.0)
        self.cmd_t = -1e9
        self.last_cmd = None
        self.n_cmd = 0
        self.summary = a.summary
        self.p_cloud = self.create_publisher(PointCloud2, '/livox/points', BE)
        self.p_imu = self.create_publisher(Imu, '/imu/data', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_odom = self.create_publisher(PoseWithCovarianceStamped, '/leg_odom', 10)
        self.p_tf = self.create_publisher(TFMessage, '/tf_static', LATCHED)
        self.create_subscription(Twist, '/nav_cmd_vel', self.on_cmd, 10)
        self.create_subscription(MotionSimpleCMD, '/simple_cmd', self.on_simple, 10)
        self.publish_tf()
        self.k = 0
        self.dt = 0.05
        self.create_timer(self.dt, self.step)
        self.get_logger().info('fake dog up (world %s, sitting %s)' % (a.world, a.sitting))

    def publish_tf(self):
        tf = TransformStamped()
        tf.header.frame_id = 'base_link'
        tf.child_frame_id = 'livox_frame'
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = 0.18678, 0.0, 0.12498
        p = math.radians(15.0)
        tf.transform.rotation.y, tf.transform.rotation.w = math.sin(p / 2), math.cos(p / 2)
        self.p_tf.publish(TFMessage(transforms=[tf]))

    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.angular.z)
        self.cmd_t = time.monotonic()
        self.last_cmd = [m.linear.x, m.angular.z]
        self.n_cmd += 1

    def on_simple(self, m):
        self.dog.on_code(int(m.cmd_code))
        self.get_logger().info('simple_cmd 0x%08X' % int(m.cmd_code))

    def step(self):
        vx, wz = self.cmd if time.monotonic() - self.cmd_t < 0.5 else (0.0, 0.0)
        d = self.dog
        d.step(vx, wz, self.dt)
        stamp = self.get_clock().now().to_msg()
        acc, gyro = d.imu(self.dt)
        im = Imu()
        im.header.stamp = stamp
        im.linear_acceleration.x, im.linear_acceleration.y, im.linear_acceleration.z = acc
        im.angular_velocity.x, im.angular_velocity.y, im.angular_velocity.z = gyro
        im.orientation.w = 1.0
        self.p_imu.publish(im)
        n, pos, vel = d.joints()
        js = JointState()
        js.header.stamp = stamp
        js.name, js.position, js.velocity = n, pos, vel
        self.p_js.publish(js)
        od = PoseWithCovarianceStamped()
        od.header.stamp = stamp
        x, y, yaw = d.leg_odom()
        od.pose.pose.position.x, od.pose.pose.position.y = x, y
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.p_odom.publish(od)
        if self.k % 2 == 0:
            pts = self.lidar.scan(self.world, d.x, d.y, d.base_z(), d.yaw, d.roll, d.pitch)
            self.p_cloud.publish(xyz_to_cloud(pts, 'livox_frame', stamp))
        self.k += 1

    def write_summary(self):
        d = self.dog
        s = {'collisions': d.collisions, 'reversed': d.reversed, 'codes': d.code_log,
             'last_cmd': self.last_cmd, 'n_cmd': self.n_cmd, 'pose': [d.x, d.y, d.yaw],
             'max_support': d.max_support, 'posture': d.posture, 'state': d.state}
        if self.summary:
            with open(self.summary, 'w') as f:
                json.dump(s, f, indent=1)
        print(json.dumps(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default='default', choices=['default', 'stairs'])
    ap.add_argument('--sitting', action='store_true')
    ap.add_argument('--ai-ignores-cmd', action='store_true')
    ap.add_argument('--summary', default=None)
    a = ap.parse_args()
    dom = os.environ.get('ROS_DOMAIN_ID', '0').strip()
    if dom in ('61', '1'):
        print('refusing: the fake dog must not run on ROS_DOMAIN_ID=%s (a dog domain)' % dom)
        return 2
    rclpy.init()
    n = FakeDog(a)
    stop = {'v': False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    while rclpy.ok() and not stop['v']:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.write_summary()
    n.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
