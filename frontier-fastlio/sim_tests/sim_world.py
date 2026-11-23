#!/usr/bin/env python3
"""Offline simulator for dog_codes (WSL, ROS 2 Humble, run on a NON-dog domain, e.g. 42).

Emulates what the dog + TA bringup provide: /odom, TF, /livox/points (3D raycast of walls,
rubble, stairs, a see-through curtain), /camera/color/image_raw (QR + hazard sign textures),
/imu/data, /joint_states, /simple_cmd handling (stand toggle, gaits), /nav_cmd_vel, and the
organiser's /final_cell_reached. Also runs MOCK scene-understanding (:8000) and scoring (:8001)
HTTP servers that answer from ground truth and score the run.

Arena (cells 1.02 m, start cell (0,0) facing +X):
   corridor along +X: (1,0) rubble, curtain at x=2.55, (3,0) stairs up 3x0.10 m, (4,0) final
   room N (1,1): kitchen + survivor + high_voltage sign + QR "My friend is in the clinic."
   room S (1,-1): clinic + survivor + medical_equipment
"""
import argparse
import json
import math
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu, JointState, PointCloud2, PointField
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from transfer_interfaces.msg import MotionSimpleCMD

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'dog_codes'))
import vision as V  # noqa: E402

CELL = 1.02
H = CELL / 2
BODY_H = 0.30
MOUNT = np.array([0.187, 0.0, 0.125])
MOUNT_PITCH = math.radians(15.0)
CAM_OFF = np.array([0.25, 0.0, 0.074])
LOW_H = 0.30            # height of a 'low' wall (below the lidar, which is 0.425 m up)
FX = 462.6

GAIT_CODES = {0x21010300: 'flat_slow', 0x21010307: 'flat_medium', 0x21010303: 'flat_fast',
              0x21010406: 'crawl', 0x21010402: 'grip', 0x21010401: 'general', 0x21010407: 'hstep'}


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


# =============================================================================== world
class World:

    def __init__(self, layout='basic', stair_ok=('general', 'grip'), low_back=False):
        self.stair_ok = set(stair_ok)
        x0, x1 = -H, 5 * CELL + H
        y0, y1 = -3 * H, 3 * H
        W = []          # (x1,y1,x2,y2, kind)  kind: 'wall' | 'curtain' | 'low'
        # 'low' = solid but only LOW_H high: the pitched-down lidar looks UP behind the dog, so it never sees it
        # (real arena, 20 Nov trial 2: the dog pushed into something behind its start pose that the cloud never showed)
        W += [(x0, y0, x1, y0, 'wall'), (x0, y1, x1, y1, 'wall'),
              (x0, y0, x0, y1, 'low' if low_back else 'wall')]
        W += [(x1, y0, x1, -H, 'wall'), (x1, H, x1, y1, 'wall')]          # final-cell opening
        # corridor side walls
        W += [(-H, -H, H, -H, 'wall'), (-H, H, H, H, 'wall')]             # start cell
        W += [(H, H, H, y1, 'wall'), (H, y0, H, -H, 'wall')]              # room west walls
        W += [(3 * H, H, 3 * H, y1, 'wall'), (3 * H, y0, 3 * H, -H, 'wall')]   # room east walls
        W += [(3 * H, H, x1, H, 'wall'), (3 * H, -H, x1, -H, 'wall')]     # corridor walls (x>1.53)
        # curtain across the corridor between cells 2 and 3
        W += [(5 * H, -H, 5 * H, H, 'curtain')]
        self.walls = np.array([w[:4] for w in W], float)
        self.kinds = [w[4] for w in W]
        self.start_walls_n = len(W)
        self.rooms = {(1, 1): {'scene': 'kitchen', 'objects': [], 'survivor': True},
                      (1, -1): {'scene': 'clinic', 'objects': ['medical_equipment'], 'survivor': True}}
        # markers: (kind, centre xyz, right xy, size, texture)
        self.markers = []
        self.qr_text = 'My friend is in the clinic.'
        self.markers.append(('qr', np.array([H, CELL, 0.34]), np.array([0.0, 1.0]), 0.12,
                             self.make_qr(self.qr_text)))
        self.markers.append(('hazard:high_voltage', np.array([CELL, 3 * H - 0.01, 0.36]),
                             np.array([1.0, 0.0]), 0.16, self.make_hazard('high_voltage')))
        self.final_box = (5 * CELL - 0.45, 5 * CELL + 0.45, -0.45, 0.45)

    def make_qr(self, text):
        import qrcode
        qr = qrcode.QRCode(border=0, box_size=8, error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(text)
        qr.make(fit=True)
        arr = np.array(qr.get_matrix(), dtype=np.uint8)
        img = np.where(arr > 0, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (arr.shape[1] * 8, arr.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
        img = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def make_hazard(self, name):
        sheet = cv2.imread(os.path.join(HERE, '..', 'dog_codes', 'assets', 'hazard_sheet.jpg'))
        if name == 'flammable':
            return V._flame_template()
        tris = V.find_triangles(sheet, 1500)
        tris.sort(key=lambda t: (round(t['bbox'][1] / 120), t['bbox'][0]))
        i = V.SHEET_NAMES.index(name)
        x, y, w, h = tris[i]['bbox']
        pad = 8
        return sheet[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].copy()

    # ---- ground
    def ground(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        g = np.zeros_like(x)
        inb = np.abs(y) < H
        # stairs (cell 4): three 0.10 m steps 0.30 m deep, then platform
        for k, xs in enumerate((3.77, 4.07, 4.37)):
            g = np.where(inb & (x >= xs), 0.10 * (k + 1), g)
        # rubble (cell 1)
        rub = inb & (x > H) & (x < 3 * H)
        bump = 0.08 * (np.sin(2 * math.pi * x / 0.21) * np.sin(2 * math.pi * y / 0.17) + 1) / 2
        g = np.where(rub, bump, g)
        return g

    def in_final(self, x, y):
        b = self.final_box
        return b[0] < x < b[1] and b[2] < y < b[3]

    # ---- geometry helpers
    def seg_dist(self, px, py):
        A = self.walls[:, :2]
        B = self.walls[:, 2:]
        AB = B - A
        t = np.clip(((px - A[:, 0]) * AB[:, 0] + (py - A[:, 1]) * AB[:, 1]) /
                    np.maximum((AB ** 2).sum(1), 1e-9), 0, 1)
        cx = A[:, 0] + t * AB[:, 0]
        cy = A[:, 1] + t * AB[:, 1]
        return np.hypot(px - cx, py - cy)

    def los_blocked(self, p, q):
        """2D line of sight p->q blocked by a solid wall?"""
        for (ax, ay, bx, by), kind in zip(self.walls, self.kinds):
            if kind in ('curtain', 'low'):
                continue
            d1 = (q[0] - p[0], q[1] - p[1])
            d2 = (bx - ax, by - ay)
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9:
                continue
            t = ((ax - p[0]) * d2[1] - (ay - p[1]) * d2[0]) / den
            s = ((ax - p[0]) * d1[1] - (ay - p[1]) * d1[0]) / den
            if 0.02 < t < 0.98 and 0 <= s <= 1:
                return True
        return False

    def raycast(self, o, dirs, rng):
        """o: (3,), dirs: (N,3) unit. Returns (N,) range (inf if none) and is_curtain mask."""
        N = dirs.shape[0]
        best = np.full(N, np.inf)
        curt = np.zeros(N, bool)
        A = self.walls[:, :2]
        B = self.walls[:, 2:]
        for k in range(len(self.walls)):
            a, b = A[k], B[k]
            e = b - a
            h = dirs[:, :2]
            den = h[:, 0] * e[1] - h[:, 1] * e[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                t = ((a[0] - o[0]) * e[1] - (a[1] - o[1]) * e[0]) / den
                s = ((a[0] - o[0]) * h[:, 1] - (a[1] - o[1]) * h[:, 0]) / den
            ok = (np.abs(den) > 1e-9) & (t > 0.05) & (s >= 0) & (s <= 1) & (t < rng)
            if self.kinds[k] == 'curtain':
                ok &= np.random.rand(N) < 0.82          # cloth is partly see-through
            z = o[2] + t * dirs[:, 2]
            px = o[0] + t * dirs[:, 0]
            py = o[1] + t * dirs[:, 1]
            ok &= (z >= self.ground(px, py) - 0.02) & (z <= (LOW_H if self.kinds[k] == 'low' else 1.10))
            upd = ok & (t < best)
            if self.kinds[k] == 'curtain':
                t = t + np.random.randn(N) * 0.035
            best = np.where(upd, t, best)
            curt = np.where(upd, self.kinds[k] == 'curtain', curt)
        # ground
        down = dirs[:, 2] < -0.02
        if down.any():
            T = np.arange(0.3, min(rng, 7.0), 0.05)
            idx = np.where(down)[0]
            P = o[None, None, :] + dirs[idx][:, None, :] * T[None, :, None]
            gz = self.ground(P[:, :, 0], P[:, :, 1])
            below = P[:, :, 2] <= gz
            first = np.where(below.any(1), below.argmax(1), -1)
            ok = first >= 0
            tg = np.where(ok, T[np.maximum(first, 0)], np.inf)
            upd = tg < best[idx]
            best[idx] = np.where(upd, tg, best[idx])
            curt[idx] = np.where(upd, False, curt[idx])
        return best, curt


# =============================================================================== sim node
class Sim(Node):

    def __init__(self, world, args):
        super().__init__('sim_world')
        self.w = world
        self.args = args
        sx, sy, syaw = [float(v) for v in args.start.split(',')]
        self.x, self.y, self.yaw = sx, sy, math.radians(syaw)
        self.zb = BODY_H           # base z in world
        self.pitch = 0.0
        self.v = np.zeros(3)       # vx, vy, wz (smoothed)
        self.cmd = np.zeros(3)
        self.cmd_t = 0.0
        self.standing = args.start_standing
        self.stand_change_t = 0.0
        self.gait = 'flat_slow'
        self.collisions = 0
        self.coll_t = 0.0
        self.t_final = None
        self.t0 = time.time()
        self.max_z = 0.0
        self.gait_log = []
        self.blocked_stairs = 0
        self.curtain_passed = False
        self.pose_hist = []
        self.first_move_t = None           # sim wall time of the first commanded motion
        self.odom_off = np.zeros(2)        # injected odometry error (--odom-fault-at)

        qos_b = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos_b)
        self.cloud_pub = self.create_publisher(PointCloud2, '/livox/points', qos_b)
        self.img_pub = self.create_publisher(Image, '/camera/color/image_raw', qos_b)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', qos_b)
        self.js_pub = self.create_publisher(JointState, '/joint_states', qos_b)
        self.final_pub = self.create_publisher(Bool, '/final_cell_reached', 10)
        self.tfb = TransformBroadcaster(self)
        self.stfb = StaticTransformBroadcaster(self)
        self.create_subscription(Twist, '/nav_cmd_vel', self.on_cmd, 10)
        self.create_subscription(Twist, '/joy_cmd_vel', self.on_joy, 10)
        self.create_subscription(MotionSimpleCMD, '/simple_cmd', self.on_simple, 10)
        self.g_slow = MutuallyExclusiveCallbackGroup()
        self.g_fast = MutuallyExclusiveCallbackGroup()
        self.create_timer(0.02, self.physics, callback_group=self.g_fast)
        self.create_timer(0.1, self.lidar, callback_group=self.g_slow)
        self.create_timer(0.1, self.camera, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(0.02, self.fast_sensors, callback_group=self.g_fast)
        self.create_timer(0.5, self.final_check, callback_group=self.g_fast)
        self.publish_static()
        self.joy = np.zeros(3)
        self.joy_t = 0.0
        self.get_logger().info('sim_world up: standing=%s stair_ok=%s' % (self.standing, world.stair_ok))

    # ------------------------------------------------------------ inputs
    def on_cmd(self, m):
        self.cmd = np.array([m.linear.x, m.linear.y, m.angular.z])
        self.cmd_t = time.time()

    def on_joy(self, m):
        self.joy = np.array([m.linear.x, m.linear.y, m.angular.z])
        self.joy_t = time.time()

    def on_simple(self, m):
        c = m.cmd_code
        if c == 0x21010202:
            self.standing = not self.standing
            self.stand_change_t = time.time()
            self.get_logger().info('SIM: stand toggle -> standing=%s' % self.standing)
        elif c in GAIT_CODES:
            self.gait = GAIT_CODES[c]
            self.gait_log.append((round(time.time() - self.t0, 1), self.gait))
            self.get_logger().info('SIM: gait -> %s' % self.gait)
        else:
            self.get_logger().warn('SIM: unknown simple_cmd 0x%08X' % c)

    # ------------------------------------------------------------ kinematics
    def physics(self):
        dt = 0.02
        now = time.time()
        use_joy = now - self.joy_t < 0.5 and np.abs(self.joy).sum() > 0.01
        cmd = self.joy if use_joy else (self.cmd if now - self.cmd_t < 0.5 else np.zeros(3))
        can_move = self.standing and (now - self.stand_change_t > 1.5)
        if not can_move:
            cmd = np.zeros(3)
        if self.first_move_t is None and np.abs(cmd).sum() > 0.02:
            self.first_move_t = now
        if (self.args.odom_fault_at > 0 and self.first_move_t is not None
                and now - self.first_move_t > self.args.odom_fault_at):
            # FAST-LIO divergence (real run 10:07): the pose runs away at ~1.5 m/s while the dog is where it was
            self.odom_off += np.array([0.0, -1.5 * dt])
            if now - getattr(self, '_of_log', 0) > 5.0:
                self._of_log = now
                self.get_logger().warn('SIM ODOM FAULT active: offset (%.1f, %.1f)' % tuple(self.odom_off))
        self.v += (cmd - self.v) * min(1.0, dt / 0.25)
        vx, vy, wz = self.v
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        nx = self.x + (vx * c - vy * s) * dt
        ny = self.y + (vx * s + vy * c) * dt
        # curtain: slows down
        d = self.w.seg_dist(nx, ny)
        kinds = np.array(self.w.kinds)
        solid = d[kinds != 'curtain'].min() if (kinds != 'curtain').any() else 9
        curt = d[kinds == 'curtain'].min()
        if curt < 0.22:
            nx = self.x + (nx - self.x) * 0.6
            ny = self.y + (ny - self.y) * 0.6
            if abs(vx) > 0.02:
                self.curtain_passed = self.curtain_passed or (self.x > 2.55 + 0.2)
        # stairs: need a gait that can climb; each step edge blocks otherwise
        g_now = float(self.w.ground(self.x, self.y))
        g_new = float(self.w.ground(nx + 0.22 * c, ny + 0.22 * s))
        blocked = False
        if g_new > g_now + 0.07 and 3.5 < nx < 4.7 and self.gait not in self.w.stair_ok:
            blocked = True
            self.blocked_stairs += 1
        if solid < 0.20:
            blocked = True
            if abs(vx) + abs(vy) > 0.05 and now - self.coll_t > 1.0:
                self.collisions += 1
                self.coll_t = now
                self.get_logger().warn('SIM COLLISION #%d at (%.2f,%.2f)' % (self.collisions, self.x, self.y))
        if blocked and now - getattr(self, '_blk_log', 0) > 3.0 and abs(vx) > 0.02:
            self._blk_log = now
            self.get_logger().info('SIM blocked at (%.2f,%.2f) solid=%.2f stairs_gait=%s g_new-g_now=%.2f vx=%.2f' % (
                self.x, self.y, solid, self.gait, g_new - g_now, vx))
        if not blocked:
            self.x, self.y = nx, ny
        self.yaw += wz * dt
        # body height / pitch follow ground
        gb = float(self.w.ground(self.x, self.y))
        # rubble bumps are filtered by the legs: use a smoothed value
        gs = float(np.mean(self.w.ground(self.x + np.array([-0.2, 0, 0.2]) * c, self.y + np.array([-0.2, 0, 0.2]) * s)))
        tgt = gs + BODY_H
        self.zb += (tgt - self.zb) * min(1.0, dt / 0.4)
        def sg(d):
            o = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
            return float(np.mean(self.w.ground(self.x + (d + o) * c, self.y + (d + o) * s)))
        slope = (sg(0.3) - sg(-0.3)) / 0.6
        if self.gait not in self.w.stair_ok:
            slope = 0.0
        ptgt = -math.atan(slope * 0.8)
        self.pitch += (ptgt - self.pitch) * min(1.0, dt / 0.3)
        self.max_z = max(self.max_z, self.zb - BODY_H)
        if now - getattr(self, '_pose_log', 0) > 5.0:
            self._pose_log = now
            self.get_logger().info('SIM pose (%.2f,%.2f) yaw=%.0f z=%.2f gait=%s v=%.2f' % (self.x, self.y, math.degrees(self.yaw), self.zb - BODY_H, self.gait, self.v[0]))

    def R_base(self):
        return Rz(self.yaw) @ Ry(self.pitch)

    # ------------------------------------------------------------ publishers
    def publish_static(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'livox_frame'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = MOUNT
        q = rot_to_quat(Ry(MOUNT_PITCH))
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.stfb.sendTransform(t)

    def fast_sensors(self):
        now = self.get_clock().now().to_msg()
        R = self.R_base()
        q = rot_to_quat(R)
        zodom = self.zb - BODY_H
        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = 'odom'
        od.child_frame_id = 'base_link'
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = (
            self.x + self.odom_off[0], self.y + self.odom_off[1], zodom)
        od.pose.pose.orientation.x, od.pose.pose.orientation.y, od.pose.pose.orientation.z, od.pose.pose.orientation.w = q
        od.twist.twist.linear.x, od.twist.twist.linear.y = float(self.v[0]), float(self.v[1])
        od.twist.twist.angular.z = float(self.v[2])
        self.odom_pub.publish(od)
        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            self.x + self.odom_off[0], self.y + self.odom_off[1], zodom)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = q
        self.tfb.sendTransform(tf)
        # imu (accel in m/s^2, body frame)
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = 'imu'
        a = R.T @ np.array([0, 0, 9.81]) + np.random.randn(3) * 0.05
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = a
        self.imu_pub.publish(imu)
        # joints
        js = JointState()
        js.header.stamp = now
        names = ['LF_Joint', 'LF_Joint_1', 'LF_Joint_2', 'RF_Joint', 'RF_Joint_1', 'RF_Joint_2',
                 'LB_Joint', 'LB_Joint_1', 'LB_Joint_2', 'RB_Joint', 'RB_Joint_1', 'RB_Joint_2']
        knee = -1.55 if self.standing else -2.75
        hipy = 0.85 if self.standing else 1.17
        js.name = names
        js.position = [0.0, hipy, knee] * 4
        self.js_pub.publish(js)

    def lidar(self):
        R = self.R_base()
        o = np.array([self.x, self.y, self.zb - BODY_H]) + R @ (np.array([0, 0, BODY_H]) + MOUNT)
        # base frame origin is at height zb above the ground-datum (ground z = 0 world)
        o = np.array([self.x, self.y, self.zb]) + R @ MOUNT
        Rs = R @ Ry(MOUNT_PITCH)
        az = np.radians(np.arange(0, 360, 0.7))
        el = np.radians(np.linspace(-7, 52, 14))
        AZ, EL = np.meshgrid(az, el)
        ds = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
        dw = ds @ Rs.T
        rng, curt = self.w.raycast(o, dw, 12.0)
        ok = np.isfinite(rng)
        r = rng[ok] + np.random.randn(ok.sum()) * 0.008
        pw = o + dw[ok] * r[:, None]
        ps = (pw - o) @ Rs                     # world -> sensor frame (R^T applied on the right)
        pts = ps.astype(np.float32)
        m = PointCloud2()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'livox_frame'
        m.height = 1
        m.width = pts.shape[0]
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
        m.fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
        inten = np.where(curt[ok], 20.0, 120.0).astype(np.float32)
        arr = np.concatenate([pts, inten[:, None]], 1).astype(np.float32)
        m.is_bigendian = False
        m.point_step = 16
        m.row_step = 16 * m.width
        m.is_dense = True
        m.data = arr.tobytes()
        self.cloud_pub.publish(m)

    # ------------------------------------------------------------ camera
    def cam_pose(self):
        R = Rz(self.yaw)
        pos = np.array([self.x, self.y, self.zb]) + R @ CAM_OFF
        return pos, self.yaw

    def project(self, P, pos, yaw):
        d = P - pos
        c, s = math.cos(yaw), math.sin(yaw)
        f = d[..., 0] * c + d[..., 1] * s
        l = -d[..., 0] * s + d[..., 1] * c
        u = d[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            px = 320.0 + FX * (-l) / f
            py = 240.0 + FX * (-u) / f
        return np.stack([px, py], -1), f

    def marker_corners(self, mk):
        kind, c, right, size, tex = mk
        r = np.array([right[0], right[1], 0.0])
        up = np.array([0, 0, 1.0])
        hw = size / 2
        return np.array([c - r * hw + up * hw, c + r * hw + up * hw, c + r * hw - up * hw, c - r * hw - up * hw])

    def visible_markers(self):
        pos, yaw = self.cam_pose()
        out = []
        for mk in self.w.markers:
            C = self.marker_corners(mk)
            px, f = self.project(C, pos, yaw)
            if (f < 0.25).any():
                continue
            if (px[:, 0] < -50).any() and (px[:, 0] > 690).all():
                continue
            cen = mk[1]
            if (cen[0] - pos[0]) * math.cos(yaw) + (cen[1] - pos[1]) * math.sin(yaw) <= 0.25:
                continue
            inside = ((px[:, 0] > 5) & (px[:, 0] < 635) & (px[:, 1] > 5) & (px[:, 1] < 475)).all()
            if not inside:
                continue
            # facing the camera? (marker normal = -(view direction) for the wall it sits on)
            right = mk[2]
            nrm = np.array([-right[1], right[0]])       # one of the two normals
            to_cam = pos[:2] - cen[:2]
            if abs(np.dot(nrm, to_cam)) < 0.05:
                continue
            if self.w.los_blocked(pos[:2], cen[:2] + (to_cam / (np.linalg.norm(to_cam) + 1e-9)) * 0.03):
                continue
            out.append((mk, px))
        return out

    def camera(self):
        rng = np.random.default_rng(7)
        img = (110 + rng.integers(-18, 18, (480, 640, 1))).astype(np.uint8).repeat(3, axis=2)
        for mk, px in self.visible_markers():
            tex = mk[4]
            h, w = tex.shape[:2]
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            Mh = cv2.getPerspectiveTransform(src, px.astype(np.float32))
            warped = cv2.warpPerspective(tex, Mh, (640, 480), flags=cv2.INTER_AREA)
            mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Mh, (640, 480))
            img = np.where(mask[:, :, None] > 0, warped, img)
        m = Image()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'camera_color_optical_frame'
        m.height, m.width = 480, 640
        m.encoding = 'bgr8'
        m.step = 640 * 3
        m.data = img.tobytes()
        self.img_pub.publish(m)

    def final_check(self):
        if self.w.in_final(self.x, self.y) and self.zb - BODY_H > 0.2:
            if self.t_final is None:
                self.t_final = time.time() - self.t0
                self.get_logger().warn('SIM: robot reached FINAL cell at t=%.1f s' % self.t_final)
            self.final_pub.publish(Bool(data=True))


# =============================================================================== servers
class Truth:
    """Ground truth + mock servers' state."""

    def __init__(self, sim):
        self.sim = sim
        self.reports = {}          # task_id -> payload (latest)
        self.reject_log = []
        self.vlm_busy = False
        self.vlm_calls = 0
        self.vlm_lock = threading.Lock()

    def view(self):
        s = self.sim
        w = s.w
        cx, cy = cell = (int(round(s.x / CELL)), int(round(s.y / CELL)))
        room = w.rooms.get(cell)
        if room is None:
            # looking through the door into a room?
            for dist in (0.8, 1.3, 1.8):
                px, py = s.x + dist * math.cos(s.yaw), s.y + dist * math.sin(s.yaw)
                c2 = (int(round(px / CELL)), int(round(py / CELL)))
                if c2 in w.rooms and not w.los_blocked((s.x, s.y), (px, py)):
                    room = w.rooms[c2]
                    break
        vis = s.visible_markers()
        haz = [m[0][0].split(':')[1] for m in vis if m[0][0].startswith('hazard:')]
        return room, haz

    def vlm_answer(self, body):
        s = self.sim
        txt = body.decode('latin1')
        room, haz = self.view()
        if 'hanging fabric curtain' in txt:
            # curtain within 1.6 m ahead?
            ahead = False
            for (ax, ay, bx, by), k in zip(s.w.walls, s.w.kinds):
                if k != 'curtain':
                    continue
                if abs(ax - s.x) < 1.6 and abs(s.y) < 0.8 and ax > s.x:
                    ahead = True
            res = {'scene_description': 'a hanging cloth curtain' if ahead else 'a plain wall',
                   'relevant_to_mission': ahead, 'evidence': ['fabric folds'] if ahead else ['flat surface'],
                   'confidence': 0.9}
        else:
            if room is None:
                res = {'scene_description': 'corridor with plain walls', 'relevant_to_mission': False,
                       'evidence': [], 'confidence': 0.8}
            else:
                ev = list(room['objects']) + (['survivor'] if room['survivor'] else []) + haz
                res = {'scene_description': '%s room' % room['scene'], 'relevant_to_mission': bool(ev),
                       'evidence': ev[:3], 'confidence': 0.92}
        return res


def make_handlers(truth):
    VALID_SCENE = {'clothes_shop', 'fitness_centre', 'kitchen', 'market', 'workshop', 'clinic', 'cleaning_room',
                   'furniture_shop'}
    VALID_LABEL = {'medical_equipment', 'hand_tools', 'food', 'water', 'utensils', 'high_voltage',
                   'radioactive', 'chemical', 'biological', 'flammable'}

    class Scene(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == '/openapi.json':
                spec = {'paths': {'/api/v1/analyze': {'post': {'requestBody': {'content': {
                    'multipart/form-data': {'schema': {'properties': {
                        'file': {'type': 'string'}, 'mission_prompt': {'type': 'string'},
                        'group': {'type': 'string'}}}}}}}}}}
                b = json.dumps(spec).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b)
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(n)
            if self.path != '/api/v1/analyze':
                self.send_response(404)
                self.end_headers()
                return
            with truth.vlm_lock:
                if truth.vlm_busy:
                    self.send_response(429)
                    self.send_header('Retry-After', '1')
                    self.end_headers()
                    self.wfile.write(b'{"error":"group_busy"}')
                    return
                truth.vlm_busy = True
            try:
                res = truth.vlm_answer(body)          # truth as of the REQUEST time
                time.sleep(truth.sim.args.vlm_latency)
                truth.vlm_calls += 1
                env = {'success': True, 'group_id': 'group02', 'request_id': 'sim',
                       'result': json.dumps(res), 'processing_time_s': truth.sim.args.vlm_latency}
                b = json.dumps(env).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b)
            finally:
                truth.vlm_busy = False

    class Score(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == '/summary':
                b = json.dumps(truth.summary()).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b)

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            try:
                d = json.loads(self.rfile.read(n))
            except Exception:
                d = {}
            reason = None
            gid = str(d.get('group_id', ''))
            if not re.fullmatch(r'group\d\d_[A-Za-z0-9]{4}', gid):
                reason = 'Invalid group_id'
            elif d.get('task_id') not in ('M01', 'M02', 'M03', 'M04', 'M05'):
                reason = 'Invalid task_id'
            elif d.get('task_type') not in ('object', 'hazard', 'survivor', 'qr'):
                reason = 'Invalid task_type'
            elif d.get('scene_type') not in VALID_SCENE | {''}:
                reason = 'Invalid scene_type'
            elif d.get('label') not in VALID_LABEL | {''}:
                reason = 'Invalid label'
            elif not (d.get('pos') is None or (isinstance(d['pos'], list) and len(d['pos']) == 2)):
                reason = 'Invalid pos'
            if reason:
                out = {'status': 'rejected', 'group_id': gid[:7], 'task_id': d.get('task_id'), 'reason': reason}
                truth.reject_log.append((d, reason))
            else:
                truth.reports[d['task_id']] = d
                out = {'status': 'accepted', 'group_id': gid[:7], 'task_id': d['task_id'],
                       'message': 'Submission recorded'}
            b = json.dumps(out).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b)

    return Scene, Score


def summary_fn(self):
    s = self.sim
    R = self.reports
    pts = {}

    def near(p, c):
        return p is not None and abs(p[0] - c[0]) <= 0.5 and abs(p[1] - c[1]) <= 0.5

    k = (CELL, CELL)
    c_ = (CELL, -CELL)
    m = R.get('M01')
    pts['M01'] = (5 * bool(m and m['label'] == 'medical_equipment' and m['scene_type'] == 'clinic') +
                  5 * bool(m and m['label'] == 'medical_equipment' and m['scene_type'] == 'clinic' and near(m['pos'], c_)))
    m = R.get('M02')
    pts['M02'] = (5 * bool(m and m['label'] == 'high_voltage' and m['scene_type'] == 'kitchen') +
                  5 * bool(m and m['label'] == 'high_voltage' and m['scene_type'] == 'kitchen' and near(m['pos'], k)))
    m = R.get('M03')
    pts['M03'] = (5 * bool(m and m['scene_type'] == 'kitchen') +
                  5 * bool(m and m['scene_type'] == 'kitchen' and near(m['pos'], k)))
    m = R.get('M04')
    pts['M04'] = 5 * bool(m and m['message'] == s.w.qr_text)
    m = R.get('M05')
    pts['M05'] = (5 * bool(m and m['scene_type'] == 'clinic') +
                  5 * bool(m and m['scene_type'] == 'clinic' and near(m['pos'], c_)))
    return {'reports': R, 'points': pts, 'scene_points_total': sum(pts.values()), 'of': 45,
            'rejected': [(d, r) for d, r in self.reject_log][-5:],
            'vlm_calls': self.vlm_calls, 'collisions': s.collisions, 'final_t': None if s.t_final is None else round(float(s.t_final), 1),
            'max_height_gain': round(s.max_z, 3), 'gait_log': s.gait_log,
            'curtain_passed_seen': bool(s.curtain_passed), 'pose': [round(s.x, 2), round(s.y, 2)],
            'blocked_stair_ticks': s.blocked_stairs}


Truth.summary = summary_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-sitting', dest='start_standing', action='store_false', default=True)
    ap.add_argument('--stair-ok', default='general,grip')
    ap.add_argument('--start', default='0,0,0', help='x,y,yaw_deg start pose (odom frame offset)')
    ap.add_argument('--vlm-latency', type=float, default=1.2)
    ap.add_argument('--low-back', action='store_true',
                    help='the wall behind the start is only 0.30 m high: solid, but invisible to the lidar')
    ap.add_argument('--odom-fault-at', type=float, default=0.0,
                    help='seconds after the first commanded motion at which the odometry starts to run away (0 = never)')
    ap.add_argument('--scene-port', type=int, default=8000)
    ap.add_argument('--score-port', type=int, default=8001)
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    world = World(stair_ok=args.stair_ok.split(','), low_back=args.low_back)
    sim = Sim(world, args)
    truth = Truth(sim)
    Scene, Score = make_handlers(truth)
    for port, H_ in ((args.scene_port, Scene), (args.score_port, Score)):
        srv = ThreadingHTTPServer(('127.0.0.1', port), H_)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(sim)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print('SIM SUMMARY', json.dumps(truth.summary(), indent=1))
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
