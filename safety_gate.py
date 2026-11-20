#!/usr/bin/env python3
"""Safety gate: the ONLY node that publishes velocity / gait commands towards the dog.

  /quad/cmd_vel (behaviours) --> [arm, caps, ramp, watchdog, tilt guard] --> /nav_cmd_vel
  /quad/gait_req (names)     --> /simple_cmd (transfer_interfaces/MotionSimpleCMD)

/nav_cmd_vel is muxed by the TAs' twist_mux (priority 10) below the joystick (100), so the
operator's joystick always wins. Starts DISARMED: nothing is published until
  ros2 topic pub --once /quad/arm std_msgs/msg/Bool "{data: true}"
Disarm / stop: /quad/arm false, or /quad/stop.
"""
import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, String, Empty

import common as C

try:
    from transfer_interfaces.msg import MotionSimpleCMD
except Exception:  # pragma: no cover
    MotionSimpleCMD = None

GAIT_CODES = {
    'flat_slow': 0x21010300, 'flat_medium': 0x21010307, 'flat_fast': 0x21010303,
    'crawl': 0x21010406, 'grip': 0x21010402, 'general': 0x21010401, 'hstep': 0x21010407,
    'move_mode': 0x21010D06, 'pose_mode': 0x21010D05,
    'stand_toggle': 0x21010202,          # Stand/Sit TOGGLE
    'nav_mode': 0x21010C03, 'manual_mode': 0x21010C02,
    'estop': 0x21020C0E,
}


class SafetyGate(Node):

    def __init__(self):
        super().__init__('safety_gate')
        p = C.declare(self, {
            'out_topic': '/nav_cmd_vel',
            'simple_cmd_topic': '/simple_cmd',
            'rate_hz': 20.0,
            'watchdog_s': 0.35,
            'abs_max_vx': 0.6, 'abs_max_vy': 0.3, 'abs_max_wz': 1.0, 'abs_min_vx': -0.3,
            'init_max_vx': 0.25, 'init_max_vy': 0.15, 'init_max_wz': 0.6,
            'acc_lin': 0.6, 'acc_ang': 2.0,
            'tilt_limit_deg': 30.0, 'tilt_hold_s': 0.3,
            'knee_joint_names': ['LF_Joint_2', 'RF_Joint_2', 'LB_Joint_2', 'RB_Joint_2'],
            'stand_knee_thresh': -2.3,      # mean knee angle above this => standing
            'fault_sends_estop': False,
            'simple_cmd_min_gap_s': 0.4,
            'joy_topic': '/joy_cmd_vel',
            'dry_run': False,              # True: never publish anything towards the dog
            # odometry watchdog: FAST-LIO can diverge (20 Nov 10:07: pose reached 284 m while the dog stood still).
            # Net speed over 1 s above odom_fault_speed for odom_fault_s while ARMED => disarm + latch.
            'odom_fault_enabled': True, 'odom_fault_speed': 0.6, 'odom_fault_s': 0.5,
        })
        self.p = p
        self.codes = dict(GAIT_CODES)
        for k in list(self.codes):
            self.declare_parameter('code_' + k, int(self.codes[k]))
            self.codes[k] = int(self.get_parameter('code_' + k).value)

        self.armed = False
        self.latched = False
        self.latch_reason = ''
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_t = 0.0
        self.out = [0.0, 0.0, 0.0]
        self.lim = dict(vx=p['init_max_vx'], vy=p['init_max_vy'], wz=p['init_max_wz'],
                        tilt=p['tilt_limit_deg'])
        self.roll = self.pitch = 0.0
        self.imu_t = 0.0
        self.standing = None
        self.joint_t = 0.0
        self.gait = 'unknown'
        self.last_simple = 0.0
        self.tilt_since = None
        self.manual_t = 0.0
        self.odom_speed = 0.0
        self.odom_t = 0.0
        self.push_since = None
        self.responding = True
        self.mode = 'IDLE'
        self.pending = []
        self.odom_wd = C.OdomWatchdog(max_speed=p['odom_fault_speed'], hold_s=p['odom_fault_s'])

        self.pub =self.create_publisher(Twist, p['out_topic'], 10)
        self.simple_pub = (self.create_publisher(MotionSimpleCMD, p['simple_cmd_topic'], 10)
                           if MotionSimpleCMD else None)
        if MotionSimpleCMD is None:
            self.get_logger().error('transfer_interfaces not found: gait/stand commands DISABLED')
        self.status_pub = self.create_publisher(String, C.T_STATUS, 10)

        self.create_subscription(Twist, C.T_CMD, self.on_cmd, 10)
        self.create_subscription(String, C.T_GAIT_REQ, self.on_gait, 10)
        self.create_subscription(Bool, C.T_ARM, self.on_arm, 10)
        self.create_subscription(Empty, '/quad/stop', self.on_stop, 10)
        self.create_subscription(String, C.T_LIMITS, self.on_limits, 10)
        self.create_subscription(String, C.T_MODE, lambda m: setattr(self, 'mode', m.data), 10)
        self.create_subscription(Imu, '/imu/data', self.on_imu, C.SENSOR_QOS)
        self.create_subscription(JointState, '/joint_states', self.on_joints, C.SENSOR_QOS)
        self.create_subscription(Odometry, '/odom', self.on_odom, C.SENSOR_QOS)
        self.create_subscription(Twist, p['joy_topic'], self.on_joy, 10)

        self.create_timer(1.0 / p['rate_hz'], self.tick)
        self.create_timer(0.2, self.publish_status)
        self.get_logger().info('safety_gate up: DISARMED. dry_run=%s out=%s'
                               % (p['dry_run'], p['out_topic']))

    # ------------------------------------------------------------------ inputs
    def now(self):
        return time.monotonic()

    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.linear.y, m.angular.z)
        self.cmd_t = self.now()

    def on_arm(self, m):
        if m.data:
            if self.latched:
                self.get_logger().warn('arm refused: latched (%s). Send /quad/arm false first.'
                                       % self.latch_reason)
                return
            self.armed = True
            self.out = [0.0, 0.0, 0.0]
            self.get_logger().warn('ARMED')
        else:
            self.disarm('operator disarm', latch=False)
            self.latched = False
            self.latch_reason = ''

    def on_stop(self, _):
        self.disarm('operator /quad/stop', latch=True)

    def on_limits(self, m):
        d = C.jload(m.data, {})
        p = self.p
        if 'vx' in d:
            self.lim['vx'] = C.clamp(float(d['vx']), 0.0, p['abs_max_vx'])
        if 'vy' in d:
            self.lim['vy'] = C.clamp(float(d['vy']), 0.0, p['abs_max_vy'])
        if 'wz' in d:
            self.lim['wz'] = C.clamp(float(d['wz']), 0.0, p['abs_max_wz'])
        if 'tilt' in d:
            self.lim['tilt'] = C.clamp(float(d['tilt']), 10.0, 60.0)

    def on_imu(self, m):
        a = m.linear_acceleration
        n = math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2)
        if n < 1e-3:
            return
        k = 1.0 if n > 3.0 else 9.81   # accel may be in g
        ax, ay, az = a.x * k, a.y * k, a.z * k
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        alpha = 0.3
        self.roll += alpha * (roll - self.roll)
        self.pitch += alpha * (pitch - self.pitch)
        self.imu_t = self.now()

    def on_joints(self, m):
        names = self.p['knee_joint_names']
        vals = [m.position[i] for i, n in enumerate(m.name) if n in names and i < len(m.position)]
        if vals:
            mean = sum(vals) / len(vals)
            self.standing = mean > self.p['stand_knee_thresh']
            self.joint_t = self.now()

    def on_odom(self, m):
        v = m.twist.twist.linear
        self.odom_speed = math.hypot(v.x, v.y)
        t = self.now()
        self.odom_t = t
        pos = m.pose.pose.position
        # odometry watchdog (not while the operator drives: the joystick can legitimately go faster)
        if (self.p['odom_fault_enabled'] and self.armed and not self.latched
                and (t - self.manual_t) >= 1.5):
            # the limit follows the speed cap, so a faster run (lim vx up to 0.6) cannot trip it with real motion
            self.odom_wd.max_speed = max(self.p['odom_fault_speed'], 1.5 * self.lim['vx'])
            if self.odom_wd.update(t, pos.x, pos.y):
                self.disarm('odometry diverged: %.1f m/s net over 1 s' % self.odom_wd.speed, latch=True)
                self.odom_wd.reset()
        else:
            self.odom_wd.reset()

    def on_joy(self, m):
        if abs(m.linear.x) + abs(m.linear.y) + abs(m.angular.z) > 0.02:
            self.manual_t = self.now()

    # ------------------------------------------------------------------ outputs
    def send_simple(self, name_or_code, typ=0):
        if self.p['dry_run']:
            self.get_logger().info('[dry_run] simple_cmd %s' % name_or_code)
            return False
        if self.simple_pub is None:
            return False
        t = self.now()
        if t - self.last_simple < self.p['simple_cmd_min_gap_s']:
            if len(self.pending) < 8:
                self.pending.append((name_or_code, typ))     # sent from tick() when the gap is over
                return True
            return False
        if isinstance(name_or_code, str) and name_or_code in self.codes:
            code = self.codes[name_or_code]
        elif isinstance(name_or_code, str) and name_or_code.startswith('code:'):
            code = int(name_or_code[5:], 0)
        else:
            self.get_logger().error('unknown gait request %r' % (name_or_code,))
            return False
        msg = MotionSimpleCMD()
        msg.cmd_code = int(code)
        msg.size = 0
        msg.type = int(typ)
        self.simple_pub.publish(msg)
        self.last_simple = t
        self.get_logger().info('simple_cmd %s = 0x%08X' % (name_or_code, code))
        return True

    def on_gait(self, m):
        if self.latched:
            self.get_logger().warn('gait request %r ignored (latched)' % m.data)
            return
        name = m.data
        if name == 'estop':
            self.disarm('gait_req estop', latch=True)
            return
        if self.send_simple(name):
            self.gait = name

    def disarm(self, why, latch):
        was = self.armed
        self.armed = False
        self.cmd = (0.0, 0.0, 0.0)
        self.out = [0.0, 0.0, 0.0]
        if latch:
            self.latched = True
            self.latch_reason = why
        if was or latch:
            self.get_logger().error('DISARMED: %s' % why)
        self.pub.publish(Twist()) if not self.p['dry_run'] else None

    def tick(self):
        t = self.now()
        dt = 1.0 / self.p['rate_hz']
        p = self.p
        if self.pending and t - self.last_simple >= p['simple_cmd_min_gap_s'] and not self.latched:
            nm, ty = self.pending.pop(0)
            self.send_simple(nm, ty)
        # tilt guard
        tilt = max(abs(math.degrees(self.roll)), abs(math.degrees(self.pitch)))
        if self.armed and t - self.imu_t < 1.0 and tilt > self.lim['tilt']:
            if self.tilt_since is None:
                self.tilt_since = t
            elif t - self.tilt_since > p['tilt_hold_s']:
                self.disarm('tilt %.0f deg > %.0f' % (tilt, self.lim['tilt']), latch=True)
                if p['fault_sends_estop']:
                    self.send_simple('estop')
        else:
            self.tilt_since = None
        manual = (t - self.manual_t) < 1.0
        if not self.armed:
            return
        age = t - self.cmd_t
        vx, vy, wz = self.cmd if age < p['watchdog_s'] and not manual else (0.0, 0.0, 0.0)
        vx = C.clamp(vx, p['abs_min_vx'], self.lim['vx'])
        vy = C.clamp(vy, -self.lim['vy'], self.lim['vy'])
        wz = C.clamp(wz, -self.lim['wz'], self.lim['wz'])
        # ramp
        tgt = [vx, vy, wz]
        for i, (acc) in enumerate((p['acc_lin'], p['acc_lin'], p['acc_ang'])):
            d = tgt[i] - self.out[i]
            step = acc * dt
            self.out[i] += C.clamp(d, -step, step)
        if manual:
            self.out = [0.0, 0.0, 0.0]
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = self.out
        # "is the dog responding" diagnostic
        if abs(self.out[0]) > 0.1 and self.odom_speed < 0.02 and t - self.odom_t < 1.0:
            if self.push_since is None:
                self.push_since = t
            self.responding = (t - self.push_since) < 3.0
        else:
            self.push_since = None
            self.responding = True
        if p['dry_run']:
            return
        self.pub.publish(msg)

    def publish_status(self):
        t = self.now()
        self.status_pub.publish(C.jmsg({
            'armed': self.armed, 'latched': self.latched, 'reason': self.latch_reason,
            'standing': self.standing, 'joint_age': round(t - self.joint_t, 1),
            'roll': round(math.degrees(self.roll), 1), 'pitch': round(math.degrees(self.pitch), 1),
            'gait': self.gait, 'manual': (t - self.manual_t) < 1.0,
            'cmd_age': round(t - self.cmd_t, 2), 'responding': self.responding,
            'lim': self.lim, 'out': [round(x, 2) for x in self.out],
            'dry_run': self.p['dry_run']}))


def main():
    rclpy.init()
    node = SafetyGate()
    try:
        C.safe_spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
