#!/usr/bin/env python3
"""curtain: find passable hanging obstacles (curtains) and push through them.

Perception (2 Hz): nearest-surface scan from /quad/cloud (obstacle height window) is split
into straight segments; for each segment a curtain score in [0,1] is computed from
  * thickness/roughness of the returns around the fitted line (cloth flutters, walls don't),
  * penetration (returns BEHIND the surface in the same bearings: cloth is see-through).
Published as /quad/curtain_info (top patches). If a patch ahead scores high while exploring,
event {"type":"curtain_candidate", ...} is sent to the supervisor.

Executor (mode CURTAIN, command /quad/behaviour {"cmd":"push",...}): optional camera check
(/quad/vlm_ask), then drive straight at v_push. The TOUCH TEST decides: if the dog stops making
progress the obstacle is a wall -> back off, event curtain_failed. Otherwise after push_dist
-> curtain_passed.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

import common as C

NBIN = 360


class Curtain(Node):

    def __init__(self):
        super().__init__('curtain')
        p = C.declare(self, {
            'analysis_hz': 2.0, 'ctrl_hz': 10.0,
            'obs_lo': 0.30, 'obs_hi': 1.50, 'min_r': 0.40, 'max_r': 3.5,
            'seg_jump': 0.18, 'min_seg_bins': 6, 'min_width': 0.35,
            'rough_lo': 0.012, 'rough_hi': 0.045, 'pen_hi': 0.12,
            'candidate_thr': 0.45, 'candidate_range': 1.8, 'candidate_cone': 0.55,
            'v_push': 0.12, 'push_dist': 1.45, 'stall_s': 2.5, 'stall_dist': 0.035,
            'grace_s': 1.5, 'backoff_s': 2.5, 'vlm_wait_s': 8.0, 'yaw_kp': 1.5,
            'timeout_s': 30.0,
        })
        self.p = p
        self.px = self.py = self.yaw = 0.0
        self.have_odom = False
        self.mode = 'IDLE'
        self.patches = []
        self.last_cand = 0.0
        self.phase = None
        self.job = {}
        self._last_an = 0.0
        self.vlm_id = 0
        self.vlm_ans = None

        self.cmd_pub = self.create_publisher(Twist, C.T_CMD, 10)
        self.evt_pub = self.create_publisher(String, C.T_EVENT, 10)
        self.info_pub = self.create_publisher(String, '/quad/curtain_info', 10)
        self.ask_pub = self.create_publisher(String, '/quad/vlm_ask', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, C.SENSOR_QOS)
        self.create_subscription(PointCloud2, C.T_CLOUD, self.on_cloud, C.SENSOR_QOS)
        self.create_subscription(String, C.T_MODE, self.on_mode, 10)
        self.create_subscription(String, C.T_BEHAV, self.on_behav, 10)
        self.create_subscription(String, '/quad/vlm_answer', self.on_vlm, 10)
        self.create_timer(1.0 / p['ctrl_hz'], self.control)
        self.get_logger().info('curtain up')

    def on_odom(self, m):
        self.px, self.py = m.pose.pose.position.x, m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.yaw = C.quat_to_rpy(q.x, q.y, q.z, q.w)[2]
        self.have_odom = True

    def on_mode(self, m):
        if self.mode == 'CURTAIN' and m.data != 'CURTAIN':
            self.phase = None
        self.mode = m.data

    def on_vlm(self, m):
        d = C.jload(m.data, {})
        if d.get('id') == self.vlm_id:
            self.vlm_ans = d

    def on_behav(self, m):
        d = C.jload(m.data, {})
        if d.get('cmd') == 'push':
            t = time.monotonic()
            self.job = {'heading': float(d.get('heading', self.yaw)),
                        'vlm': bool(d.get('vlm', False)),
                        'last_resort': bool(d.get('last_resort', False)),
                        'x0': self.px, 'y0': self.py, 't0': t}
            if self.job['vlm']:
                self.vlm_id += 1
                self.vlm_ans = None
                self.ask_pub.publish(C.jmsg({'id': self.vlm_id, 'kind': 'curtain'}))
                self.phase = 'INSPECT'
            else:
                self.phase = 'PUSH'
            self.job['phase_t'] = t
            self.job['prog'] = (t, self.px, self.py)
            self.get_logger().info('curtain push job %s' % self.job)

    # --------------------------------------------------------------- perception
    def on_cloud(self, msg):
        if not self.have_odom:
            return
        t = time.monotonic()
        if t - self._last_an < 1.0 / self.p['analysis_hz']:
            return
        self._last_an = t
        pts = C.cloud_to_xyz(msg)
        if pts.shape[0] == 0:
            return
        self.patches = self.find_patches(pts)
        self.info_pub.publish(C.jmsg({'patches': self.patches[:8]}))
        # live candidate while exploring
        if self.mode == 'EXPLORE' and t - self.last_cand > 3.0:
            p = self.p
            for pa in self.patches:
                bear = C.wrap(math.atan2(pa['y'] - self.py, pa['x'] - self.px) - self.yaw)
                if (pa['score'] >= p['candidate_thr'] and pa['rng'] < p['candidate_range']
                        and abs(bear) < p['candidate_cone']):
                    self.last_cand = t
                    self.evt_pub.publish(C.jmsg({'type': 'curtain_candidate', **pa}))
                    break

    def find_patches(self, pts):
        p = self.p
        pts = C.filter_obstacle_points(pts, lo=p['obs_lo'], hi=p['obs_hi'], rmin=p['min_r'],
                                       origin=(self.px, self.py))
        if pts.shape[0] == 0:
            return []
        dx = pts[:, 0] - self.px
        dy = pts[:, 1] - self.py
        r = np.hypot(dx, dy)
        sel = r < p['max_r']
        dx, dy, r = dx[sel], dy[sel], r[sel]
        if r.size < 30:
            return []
        ang = np.arctan2(dy, dx)
        bins = ((ang + math.pi) / (2 * math.pi) * NBIN).astype(int) % NBIN
        rmin = np.full(NBIN, np.inf)
        np.minimum.at(rmin, bins, r)
        # segments of consecutive bins with continuous nearest range
        segs = []
        cur = []
        for b in range(NBIN):
            if np.isfinite(rmin[b]) and (not cur or abs(rmin[b] - rmin[cur[-1]]) < p['seg_jump']
                                         and (b - cur[-1]) <= 2):
                cur.append(b)
            else:
                if len(cur) >= p['min_seg_bins']:
                    segs.append(cur)
                cur = [b] if np.isfinite(rmin[b]) else []
        if len(cur) >= p['min_seg_bins']:
            segs.append(cur)
        out = []
        flat = []
        for sg in segs:
            flat.extend(self.split_straight(sg, rmin))
        for sg in flat:
            b0, b1 = sg[0], sg[-1]
            inb = (bins >= b0) & (bins <= b1)
            rs = rmin[sg]
            a = (np.array(sg) + 0.5) / NBIN * 2 * math.pi - math.pi
            P = np.stack([self.px + rs * np.cos(a), self.py + rs * np.sin(a)], 1)
            c = P.mean(0)
            u, s_, vt = np.linalg.svd(P - c, full_matrices=False)
            d = vt[0]
            n = np.array([-d[1], d[0]])
            width = float(np.ptp((P - c) @ d))
            if width < p['min_width']:
                continue
            if np.dot(n, np.array([self.px, self.py]) - c) < 0:
                n = -n                      # normal points towards the robot
            # refit the line on ALL returns close to it (nearest-per-bin points are biased at
            # oblique incidence), then measure thickness / see-through on the inner bearings
            allb = (bins >= b0) & (bins <= b1)
            Qa = np.stack([self.px + dx[allb], self.py + dy[allb]], 1)
            for _ in range(2):
                band = np.abs((Qa - c) @ n) < 0.15
                if band.sum() < 10:
                    break
                c = Qa[band].mean(0)
                u2, s2, vt2 = np.linalg.svd(Qa[band] - c, full_matrices=False)
                d = vt2[0]
                n = np.array([-d[1], d[0]])
                if np.dot(n, np.array([self.px, self.py]) - c) < 0:
                    n = -n
            inner = (bins >= b0 + 1) & (bins <= b1 - 1)
            Q = np.stack([self.px + dx[inner], self.py + dy[inner]], 1)
            if Q.shape[0] < 12:
                continue
            dist = (Q - c) @ n              # >0: in front of the line, <0: behind it
            near = np.abs(dist) < 0.12
            if near.sum() < 10:
                continue
            thick = float(np.std(dist[near]))
            behind = float(np.sum((dist < -0.30) & (dist > -2.0)))
            pen = behind / max(1.0, float(near.sum()) + behind)
            rough_n = C.clamp((thick - p['rough_lo']) / (p['rough_hi'] - p['rough_lo']), 0, 1)
            pen_n = C.clamp(pen / p['pen_hi'], 0, 1)
            score = 0.20 * pen_n + 0.80 * rough_n
            out.append({'x': round(float(c[0]), 2), 'y': round(float(c[1]), 2),
                        'nx': round(float(n[0]), 3), 'ny': round(float(n[1]), 3),
                        'w': round(width, 2), 'thick': round(thick, 3), 'pen': round(pen, 3),
                        'score': round(score, 3),
                        'rng': round(float(np.hypot(c[0] - self.px, c[1] - self.py)), 2)})
        out.sort(key=lambda d: (-d['score'], d['rng']))
        return out

    def split_straight(self, sg, rmin, tol=0.06):
        """Split a run of bins at corners so that each piece is a straight wall segment."""
        if len(sg) < 2 * self.p['min_seg_bins']:
            return [sg]
        a = (np.array(sg) + 0.5) / NBIN * 2 * math.pi - math.pi
        rs = rmin[sg]
        P = np.stack([rs * np.cos(a), rs * np.sin(a)], 1)
        A, B = P[0], P[-1]
        AB = B - A
        L = np.linalg.norm(AB)
        if L < 1e-6:
            return [sg]
        nrm = np.array([-AB[1], AB[0]]) / L
        dev = np.abs((P - A) @ nrm)
        i = int(np.argmax(dev))
        if dev[i] > tol and self.p['min_seg_bins'] <= i <= len(sg) - self.p['min_seg_bins']:
            return self.split_straight(sg[:i + 1], rmin, tol) + self.split_straight(sg[i:], rmin, tol)
        return [sg]

    # --------------------------------------------------------------- executor
    def finish(self, evt, **kw):
        self.cmd_pub.publish(Twist())
        self.phase = None
        self.evt_pub.publish(C.jmsg({'type': evt, **kw}))
        self.get_logger().warn('CURTAIN %s %s' % (evt, kw))

    def control(self):
        if self.mode != 'CURTAIN' or self.phase is None or not self.have_odom:
            return
        p = self.p
        t = time.monotonic()
        j = self.job
        tw = Twist()
        if t - j['t0'] > p['timeout_s']:
            return self.finish('curtain_failed', why='timeout')
        if self.phase == 'INSPECT':
            self.cmd_pub.publish(tw)
            ans = self.vlm_ans
            if ans is not None:
                if ans.get('answer') or j['last_resort']:
                    self.phase = 'PUSH'
                    j['prog'] = (t, self.px, self.py)
                    j['x0'], j['y0'] = self.px, self.py
                else:
                    return self.finish('curtain_failed', why='vlm_says_not_curtain')
            elif t - j['phase_t'] > p['vlm_wait_s']:
                self.phase = 'PUSH'            # no answer: let the touch test decide
                j['prog'] = (t, self.px, self.py)
                j['x0'], j['y0'] = self.px, self.py
            return
        if self.phase == 'PUSH':
            travelled = math.hypot(self.px - j['x0'], self.py - j['y0'])
            if travelled >= p['push_dist']:
                return self.finish('curtain_passed', x=round(self.px, 2), y=round(self.py, 2),
                                   heading=round(j['heading'], 2))
            # touch test: progress over the last stall_s
            t0, x0, y0 = j['prog']
            if math.hypot(self.px - x0, self.py - y0) > p['stall_dist']:
                j['prog'] = (t, self.px, self.py)
            elif t - t0 > p['stall_s'] and t - j['phase_t'] > p['grace_s']:
                self.phase = 'BACKOFF'
                j['back_until'] = t + p['backoff_s']
                return
            tw.linear.x = p['v_push']
            tw.angular.z = C.clamp(p['yaw_kp'] * C.wrap(j['heading'] - self.yaw), -0.4, 0.4)
            self.cmd_pub.publish(tw)
        elif self.phase == 'BACKOFF':
            if t >= j['back_until']:
                return self.finish('curtain_failed', why='blocked_wall',
                                   x=round(self.px, 2), y=round(self.py, 2),
                                   heading=round(j['heading'], 2))
            tw.linear.x = -0.10
            self.cmd_pub.publish(tw)


def main():
    rclpy.init()
    n = Curtain()
    try:
        C.safe_spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
