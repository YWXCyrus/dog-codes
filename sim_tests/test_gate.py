#!/usr/bin/env python3
"""Behavioural test of safety_gate (run with the gate already started on the same domain).

Checks: silent while disarmed; ramped output when armed; watchdog stops stale commands; speed caps;
joystick override; tilt latch (and arm refusal while latched); /quad/stop latch.
Usage:  python3 safety_gate.py &   then   python3 test_gate.py
"""
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Empty, String


class T(Node):
    def __init__(self):
        super().__init__('gate_test')
        self.out = []
        self.status = {}
        self.create_subscription(Twist, '/nav_cmd_vel', lambda m: self.out.append((time.time(), m.linear.x, m.angular.z)), 10)
        self.create_subscription(String, '/quad/status', lambda m: setattr(self, 'status', json.loads(m.data)), 10)
        self.cmd = self.create_publisher(Twist, '/quad/cmd_vel', 10)
        self.arm = self.create_publisher(Bool, '/quad/arm', 10)
        self.stop = self.create_publisher(Empty, '/quad/stop', 10)
        self.lim = self.create_publisher(String, '/quad/limits', 10)
        self.joy = self.create_publisher(Twist, '/joy_cmd_vel', 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=2)
        self.imu = self.create_publisher(Imu, '/imu/data', qos)
        self.odom = self.create_publisher(Odometry, '/odom', qos)
        self.ox = 0.0
        self.last_odom = 0.0

    def spin_for(self, s, drive=None, imu=None, odom_v=None):
        """odom_v: publish an /odom at 10 Hz whose x advances at this speed (m/s) -- 0.0 = standing still."""
        t0 = time.time()
        while time.time() - t0 < s:
            if odom_v is not None and time.time() - self.last_odom >= 0.1:
                self.ox += odom_v * (time.time() - self.last_odom if self.last_odom else 0.1)
                self.last_odom = time.time()
                o = Odometry()
                o.pose.pose.position.x = self.ox
                self.odom.publish(o)
            if drive is not None:
                m = Twist()
                m.linear.x, m.angular.z = drive
                self.cmd.publish(m)
            if imu is not None:
                i = Imu()
                roll, pitch = imu
                i.linear_acceleration.x = -9.81 * math.sin(pitch)
                i.linear_acceleration.y = 9.81 * math.sin(roll) * math.cos(pitch)
                i.linear_acceleration.z = 9.81 * math.cos(roll) * math.cos(pitch)
                self.imu.publish(i)
            rclpy.spin_once(self, timeout_sec=0.02)


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (' ' + str(info) if info else ''))
    return ok


def main():
    rclpy.init()
    n = T()
    good = True
    n.spin_for(1.0, imu=(0.0, 0.0))
    n.out.clear()
    n.spin_for(1.0, drive=(0.2, 0.0), imu=(0.0, 0.0))
    good &= check('silent while disarmed', len(n.out) == 0, len(n.out))
    n.arm.publish(Bool(data=True))
    n.out.clear()
    n.spin_for(1.5, drive=(0.5, 3.0), imu=(0.0, 0.0))
    mx = max([o[1] for o in n.out] or [0])
    mw = max([abs(o[2]) for o in n.out] or [0])
    good &= check('armed -> publishes', len(n.out) > 10, len(n.out))
    good &= check('speed cap vx<=0.25 (init limit)', mx <= 0.2501, round(mx, 3))
    good &= check('speed cap wz<=0.6', mw <= 0.6001, round(mw, 3))
    first = [o for o in n.out if o[1] > 0][:1]
    good &= check('ramped (first sample < cap)', bool(first) and first[0][1] < 0.2, first)
    n.lim.publish(String(data=json.dumps({'vx': 0.1, 'wz': 0.2})))
    n.spin_for(0.5, drive=(0.5, 0.0), imu=(0.0, 0.0))
    n.out.clear()
    n.spin_for(1.0, drive=(0.5, 0.0), imu=(0.0, 0.0))
    good &= check('limits topic lowers cap', max([o[1] for o in n.out] or [9]) <= 0.1001, max([o[1] for o in n.out] or [0]))
    # watchdog: stop commanding
    n.spin_for(1.2, imu=(0.0, 0.0))
    n.out.clear()
    n.spin_for(0.6, imu=(0.0, 0.0))
    good &= check('watchdog -> zero output after stale cmd', all(abs(o[1]) < 0.01 for o in n.out) and len(n.out) > 3,
                  [round(o[1], 3) for o in n.out[-3:]])
    # joystick override
    n.spin_for(0.3, drive=(0.1, 0.0), imu=(0.0, 0.0))
    j = Twist()
    j.linear.x = 0.4
    for _ in range(10):
        n.joy.publish(j)
        n.spin_for(0.05, drive=(0.1, 0.0), imu=(0.0, 0.0))
    n.out.clear()
    for _ in range(10):
        n.joy.publish(j)
        n.spin_for(0.05, drive=(0.1, 0.0), imu=(0.0, 0.0))
    good &= check('joystick active -> gate outputs zero', all(abs(o[1]) < 0.01 for o in n.out), [round(o[1], 3) for o in n.out[-3:]])
    good &= check('status.manual flag', n.status.get('manual') is True, n.status.get('manual'))
    n.spin_for(1.5, drive=(0.1, 0.0), imu=(0.0, 0.0))
    # tilt latch
    n.spin_for(1.0, drive=(0.1, 0.0), imu=(math.radians(40), 0.0))
    good &= check('tilt 40deg -> latched + disarmed', n.status.get('latched') is True and n.status.get('armed') is False,
                  (n.status.get('latched'), n.status.get('armed'), n.status.get('reason')))
    n.arm.publish(Bool(data=True))
    n.spin_for(0.5, imu=(0.0, 0.0))
    good &= check('arm refused while latched', n.status.get('armed') is False)
    n.arm.publish(Bool(data=False))
    n.spin_for(0.5, imu=(0.0, 0.0))
    good &= check('disarm clears latch', n.status.get('latched') is False)
    n.arm.publish(Bool(data=True))
    n.spin_for(0.5, imu=(0.0, 0.0))
    good &= check('re-arm works', n.status.get('armed') is True)
    # odometry watchdog (armed here, limit vx = 0.1 -> threshold stays at the 0.6 m/s floor)
    n.spin_for(2.0, drive=(0.1, 0.0), imu=(0.0, 0.0), odom_v=0.0)
    good &= check('odom standing still -> stays armed', n.status.get('armed') is True, n.status.get('reason'))
    n.spin_for(3.0, drive=(0.1, 0.0), imu=(0.0, 0.0), odom_v=0.3)
    good &= check('odom walking 0.3 m/s -> stays armed', n.status.get('armed') is True, n.status.get('reason'))
    n.spin_for(3.0, drive=(0.1, 0.0), imu=(0.0, 0.0), odom_v=1.6)
    good &= check('odom runaway 1.6 m/s -> disarmed + latched',
                  n.status.get('armed') is False and n.status.get('latched') is True and 'odometry' in str(n.status.get('reason')),
                  (n.status.get('armed'), n.status.get('latched'), n.status.get('reason')))
    n.out.clear()
    n.spin_for(0.6, drive=(0.1, 0.0), imu=(0.0, 0.0), odom_v=0.0)
    good &= check('after the odom fault the output is zero', all(abs(o[1]) < 0.01 for o in n.out), [round(o[1], 3) for o in n.out[-3:]])
    n.arm.publish(Bool(data=False))
    n.spin_for(0.5, imu=(0.0, 0.0), odom_v=0.0)
    n.arm.publish(Bool(data=True))
    n.spin_for(0.5, imu=(0.0, 0.0), odom_v=0.0)
    good &= check('operator can re-arm after an odom fault', n.status.get('armed') is True)
    n.stop.publish(Empty())
    n.spin_for(0.5, imu=(0.0, 0.0))
    good &= check('/quad/stop -> latched', n.status.get('latched') is True and n.status.get('armed') is False)
    print('ALL PASS' if good else 'SOME FAILED')
    rclpy.shutdown()
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
