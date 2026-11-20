#!/usr/bin/env python3
"""terrain: stairs + rough-terrain perception, and the stair climbing executor.

Perception (always on, 5 Hz, from /quad/cloud = odom frame, z = height above the foot plane):
  * stairs_ahead : a monotonic rise (or drop) of the floor profile in the corridor ahead.
                   The lidar is blind to the floor within ~1 m of the feet, so the distance to the
                   first riser is estimated far away and then dead-reckoned with odometry.
  * rough        : uneven/rubble floor ahead (height scatter) -> {"type":"rough","on":bool}

Executor (active only while /quad/mode == STAIRS; command via /quad/behaviour):
  APPROACH (square up to the nearest axis, dead-reckon to the riser) -> GEAR (switch gait) ->
  CLIMB (closed loop on body-height gain) -> FINISH (back to the flat gait) ->
  event stairs_done / stairs_failed.
Gait names are tried in order of the `stair_gaits` parameter, one per attempt.
ASSUMPTION: stairs are square to the arena grid (yaw error is measured to the nearest 90 deg).
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


class Terrain(Node):

    def __init__(self):
        super().__init__('terrain')
        p = C.declare(self, {
            'analysis_hz': 5.0, 'ctrl_hz': 10.0,
            'corr_x0': 0.25, 'corr_x1': 2.6, 'corr_half_w': 0.40, 'bin_m': 0.06,
            'trend_min_rise': 0.10, 'trend_min_incr': 2, 'trend_max_drop': 0.045,
            'blind_bias_m': 0.35, 'trend_slope_min': 0.10, 'trend_slope_max': 1.2, 'detect_down': True,
            'confirm_n': 4, 'confirm_window': 6,
            'rough_std': 0.050, 'rough_frac': 0.35, 'rough_on_n': 4, 'rough_off_n': 15,
            # the start area (odom origin = start pose) holds the wall behind the dog, never stairs:
            # a detection whose foot lies inside this radius is ignored (0 disables)
            'start_exclusion_m': 0.9,
            'stair_gaits': ['general'],
            'flat_gait': 'general',
            'approach_dist': 0.45, 'v_approach': 0.12, 'v_climb': 0.22, 'v_descend': 0.15,
            'min_rise': 0.18, 'top_margin_m': 0.35, 'gear_wait_s': 1.8, 'top_still_s': 2.5,
            'approach_timeout_s': 30.0, 'climb_timeout_s': 40.0, 'no_progress_s': 9.0,
            'yaw_kp': 1.6, 'debug': False,
        })
        self.p = p
        self.px = self.py = self.pz = self.yaw = 0.0
        self.have_odom = False
        self.mode = 'IDLE'
        self.hist = []        # recent stairs detections
        self.rough_hits = 0
        self.rough_miss = 0
        self.rough_on = False
        self.last_stairs = None
        self.pitch = 0.0
        self._latest = None
        self._last_an = 0.0
        # executor
        self.phase = None
        self.attempt = 0
        self.dirn = 'up'
        self.phase_t = 0.0
        self.t_start = 0.0
        self.z0 = 0.0
        self.yaw0 = 0.0
        self.progress_t = 0.0
        self.best_dz = 0.0
        self.top_xy = None
        self.app_start = (0.0, 0.0)
        self.app_dist = 0.0
        self.dz_hist = []

        self.cmd_pub = self.create_publisher(Twist, C.T_CMD, 10)
        self.evt_pub = self.create_publisher(String, C.T_EVENT, 10)
        self.gait_pub = self.create_publisher(String, C.T_GAIT_REQ, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, C.SENSOR_QOS)
        self.create_subscription(PointCloud2, C.T_CLOUD, self.on_cloud, C.SENSOR_QOS)
        self.create_subscription(String, C.T_MODE, self.on_mode, 10)
        self.create_subscription(String, C.T_BEHAV, self.on_behav, 10)
        self.create_subscription(String, C.T_STATUS, self.on_status, 10)
        self.create_timer(1.0 / p['ctrl_hz'], self.control)
        self.get_logger().info('terrain up')

    # ------------------------------------------------------------ inputs
    def on_odom(self, m):
        self.px, self.py, self.pz = (m.pose.pose.position.x, m.pose.pose.position.y,
                                     m.pose.pose.position.z)
        q = m.pose.pose.orientation
        self.yaw = C.quat_to_rpy(q.x, q.y, q.z, q.w)[2]
        self.have_odom = True

    def on_status(self, m):
        d = C.jload(m.data, {})
        self.pitch = d.get('pitch', 0.0)

    def on_mode(self, m):
        if m.data != self.mode:
            if self.mode == 'STAIRS' and m.data != 'STAIRS':
                self.phase = None
            self.mode = m.data

    def on_behav(self, m):
        d = C.jload(m.data, {})
        if d.get('cmd') == 'stairs_start':
            self.attempt = int(d.get('attempt', 0))
            self.dirn = d.get('dir') or (self.last_stairs.get('dir') if self.last_stairs else 'up')
            fresh = self.last_stairs and (time.monotonic() - self.last_stairs.get('t', 0) < 3.0)
            if fresh:
                self.app_dist = float(self.last_stairs.get('dist', 1.0))
            elif 'dist' in d:
                self.app_dist = float(d['dist'])
                self.last_stairs = None
            elif self.last_stairs:
                self.app_dist = float(self.last_stairs.get('dist', 1.0))
            self.begin_approach()
            if d.get('retry'):
                # already at the foot of the stairs: just switch gait and try again
                self.phase = 'GEAR'
                self.phase_t = time.monotonic()
                self.gait_pub.publish(C.String(data=self.gaits()))

    # ------------------------------------------------------------ perception
    def axis_err(self):
        """Heading error to the nearest arena axis (stairs assumed square to the grid)."""
        ang = round(self.yaw / (math.pi / 2)) * (math.pi / 2)
        return C.wrap(ang - self.yaw)

    def analyse(self, pts):
        p = self.p
        dx = pts[:, 0] - self.px
        dy = pts[:, 1] - self.py
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        lx = cy * dx + sy * dy
        ly = -sy * dx + cy * dy
        h = pts[:, 2]
        res = {'stairs': None, 'rough': None}
        wide = (np.abs(ly) < 0.9) & (lx > 0.1) & (lx < p['corr_x1'] + 0.3) & (h < 0.6) & (h > -0.5)
        lx, ly, h = lx[wide], ly[wide], h[wide]
        if lx.size < 30:
            return res
        cen = np.abs(ly) < p['corr_half_w']
        bx = ((lx[cen] - p['corr_x0']) / p['bin_m']).astype(int)
        hc = h[cen]
        nb = int((p['corr_x1'] - p['corr_x0']) / p['bin_m'])
        prof = np.full(nb, np.nan)
        for b in range(nb):
            sel = hc[bx == b]
            if sel.size >= 2:
                prof[b] = np.percentile(sel, 85)
        self._dbg_prof = ' '.join('%.2f' % v if v == v else '--' for v in prof)
        res['stairs'] = self.find_trend(prof)
        near =(lx > 0.9) & (lx < 2.0) & (np.abs(ly) < 0.4) & (h > -0.10) & (h < 0.30)
        hn = h[near]
        if hn.size >= 25:
            res['rough'] = (float(np.std(hn)), float(np.mean(hn > 0.04)))
        return res

    def find_trend(self, prof):
        """Stairs = monotonic rise (or drop) of the floor profile with >=2 increments."""
        p = self.p
        valid = np.where(~np.isnan(prof))[0]
        if valid.size < 6:
            return None
        v = prof[valid]
        xs = p['corr_x0'] + (valid + 0.5) * p['bin_m']
        # near-field clutter (walls, props, curtain): leading bins that tower over what follows
        drops = np.where(v[:-1] - v[1:] > 0.2)[0]
        drops = drops[drops < len(v) // 2]
        if drops.size:
            v = v[drops[-1] + 1:]
            xs = xs[drops[-1] + 1:]
        if v.size < 6:
            return None
        # reference: the dog's own foot plane (0) when the first visible floor is already well above/below
        # it (we are standing at the foot of the stairs, inside the lidar's blind zone), else the first bin
        base = 0.0 if abs(float(v[0])) >= 0.08 else float(v[0])
        for sign, name in ((1.0, 'up'), (-1.0, 'down')):
            if name == 'down' and not p['detect_down']:
                continue
            w = sign * (v - base)
            run = np.maximum.accumulate(w)
            if np.any(run - w > p['trend_max_drop']):      # not monotonic (rubble, clutter)
                continue
            rise = float(run[-1])
            if rise < p['trend_min_rise']:
                continue
            incr = int(np.sum(np.diff(run) >= 0.025))
            elevated = bool(w[0] >= 0.08 and w[0] <= 0.40)
            if incr < p['trend_min_incr'] and not elevated:
                continue
            i0 = int(np.argmax(run >= 0.03))
            x_start = float(xs[i0])
            span = float(xs[-1] - x_start)
            if span < 0.2 and not elevated:
                continue
            slope = rise / max(span, 0.3)
            if not (p['trend_slope_min'] <= slope <= p['trend_slope_max']) and not elevated:
                continue
            clipped = i0 < 3 or elevated         # foot of the stairs is in the blind zone
            dist = x_start - 0.5 * p['bin_m'] - (p['blind_bias_m'] if clipped else 0.0)
            return {'dir': name, 'dist': max(dist, 0.1), 'rise': sign * rise,
                    'n': incr, 'clipped': clipped, 'yaw_err': self.axis_err()}
        return None

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
        a = self.analyse(pts)
        p = self.p
        st = a['stairs']
        if st and p['start_exclusion_m'] > 0 and self.mode != 'STAIRS':
            fx = self.px + math.cos(self.yaw) * st['dist']
            fy = self.py + math.sin(self.yaw) * st['dist']
            if math.hypot(fx, fy) < p['start_exclusion_m']:
                self.get_logger().info('stairs candidate at foot (%.2f, %.2f) is inside the start area: ignored'
                                       % (fx, fy), throttle_duration_sec=5.0)
                st = None
        if p['debug'] or os.environ.get('DOG_DEBUG'):
            self.get_logger().info('terrain: stairs=%s rough=%s prof=%s' % (
                st, a['rough'], getattr(self, '_dbg_prof', '')), throttle_duration_sec=1.0)
        self.hist.append(st)
        self.hist = self.hist[-p['confirm_window']:]
        hits = [h for h in self.hist if h]
        if len(hits) >= p['confirm_n'] and st:
            self.last_stairs = dict(st)
            self.last_stairs['t'] = t
            self.last_stairs['px'] = self.px
            self.last_stairs['py'] = self.py
            if self.mode != 'STAIRS':
                self.evt_pub.publish(C.jmsg({'type': 'stairs_ahead', **{
                    k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()},
                    'pose': [round(self.px, 2), round(self.py, 2), round(self.yaw, 2)]}))
        rg = a['rough']
        is_rough = bool(rg and (rg[0] > p['rough_std'] or rg[1] > p['rough_frac']) and not st)
        if is_rough:
            self.rough_hits += 1
            self.rough_miss = 0
        else:
            self.rough_miss += 1
            self.rough_hits = 0
        if not self.rough_on and self.rough_hits >= p['rough_on_n']:
            self.rough_on = True
            self.evt_pub.publish(C.jmsg({'type': 'rough', 'on': True,
                                         'std': round(rg[0], 3) if rg else 0}))
        elif self.rough_on and self.rough_miss >= p['rough_off_n']:
            self.rough_on = False
            self.evt_pub.publish(C.jmsg({'type': 'rough', 'on': False}))
        self._latest = st

    # ------------------------------------------------------------ executor
    def gaits(self):
        g = list(self.p['stair_gaits'])
        return g[self.attempt % len(g)]

    def begin_approach(self):
        t = time.monotonic()
        self.phase = 'APPROACH'
        self.phase_t = t
        self.t_start = t
        self.z0 = self.pz
        self.yaw0 = self.yaw + self.axis_err()          # hold the arena axis
        self.best_dz = 0.0
        self.app_start = (self.px, self.py)
        if self.last_stairs and (t - self.last_stairs.get('t', 0) < 3.0):
            # distance was measured where the stairs were last confirmed: dead-reckon from there
            self.app_start = (self.last_stairs['px'], self.last_stairs['py'])
        self.dz_hist = []
        self.get_logger().info('STAIRS approach (attempt %d, gait %s, dir %s, dist %.2f)'
                               % (self.attempt, self.gaits(), self.dirn, self.app_dist))

    def finish(self, ok, why=''):
        self.cmd_pub.publish(Twist())
        self.gait_pub.publish(C.String(data=self.p['flat_gait']))
        self.phase = None
        dz = self.pz - self.z0
        self.evt_pub.publish(C.jmsg({'type': 'stairs_done' if ok else 'stairs_failed',
                                     'why': why, 'dz': round(dz, 3), 'attempt': self.attempt}))
        self.get_logger().warn('STAIRS %s (%s) dz=%.3f' % ('DONE' if ok else 'FAILED', why, dz))

    def control(self):
        if self.mode != 'STAIRS' or self.phase is None or not self.have_odom:
            return
        p = self.p
        t = time.monotonic()
        tw = Twist()
        st = self._latest if (t - self._last_an < 1.0) else None
        err = C.wrap(self.yaw0 - self.yaw)
        if self.phase == 'APPROACH':
            if t - self.t_start > p['approach_timeout_s']:
                return self.finish(False, 'approach_timeout')
            travelled = math.hypot(self.px - self.app_start[0], self.py - self.app_start[1])
            remaining = self.app_dist - travelled
            if st and not st.get('clipped'):             # fresh far-field measurement wins
                remaining = st['dist']
            self.get_logger().info('approach: travelled=%.2f remaining=%.2f err=%.2f st=%s' % (travelled, remaining, err, st), throttle_duration_sec=1.5)
            if abs(err) > 0.12:
                tw.angular.z = C.clamp(p['yaw_kp'] * err, -0.5, 0.5)
            elif remaining > p['approach_dist']:
                tw.linear.x = p['v_approach']
                tw.angular.z = C.clamp(p['yaw_kp'] * err, -0.4, 0.4)
            else:
                self.cmd_pub.publish(Twist())
                self.phase = 'GEAR'
                self.phase_t = t
                self.gait_pub.publish(C.String(data=self.gaits()))
                return
            self.cmd_pub.publish(tw)
        elif self.phase == 'GEAR':
            self.cmd_pub.publish(tw)
            if t - self.phase_t > p['gear_wait_s']:
                self.phase = 'CLIMB'
                self.phase_t = t
                self.progress_t = t
                self.z0 = self.pz
                self.best_dz = 0.0
                self.top_xy = None
                self.dz_hist = []
        elif self.phase == 'CLIMB':
            sign = 1.0 if self.dirn == 'up' else -1.0
            dz = (self.pz - self.z0) * sign
            self.dz_hist.append((t, dz))
            self.dz_hist = [(a, b) for a, b in self.dz_hist if t - a <= p['top_still_s']]
            if dz > self.best_dz + 0.02:
                self.best_dz = dz
                self.progress_t = t
            if t - self.phase_t > p['climb_timeout_s']:
                return self.finish(False, 'climb_timeout')
            if self.best_dz < p['min_rise'] and t - self.progress_t > p['no_progress_s']:
                return self.finish(False, 'no_progress')
            v = p['v_climb'] if self.dirn == 'up' else p['v_descend']
            if self.best_dz >= p['min_rise']:
                recent = [b for a, b in self.dz_hist]
                still = (t - self.dz_hist[0][0] > p['top_still_s'] - 0.3 and
                         max(recent) - min(recent) < 0.02)
                if still or self.top_xy is not None:
                    if self.top_xy is None:
                        self.top_xy = (self.px, self.py)
                    moved = math.hypot(self.px - self.top_xy[0], self.py - self.top_xy[1])
                    if moved >= p['top_margin_m'] or t - self.progress_t > p['top_still_s'] + 6.0:
                        return self.finish(True, 'climbed')
            tw.linear.x = v
            tw.angular.z = C.clamp(p['yaw_kp'] * err, -0.4, 0.4)
            self.cmd_pub.publish(tw)


def main():
    rclpy.init()
    n = Terrain()
    try:
        C.safe_spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
