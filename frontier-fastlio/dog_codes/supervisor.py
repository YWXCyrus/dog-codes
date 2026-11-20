#!/usr/bin/env python3
"""supervisor: mission state machine. Decides which behaviour node is active (/quad/mode).

 IDLE -> STANDUP -> EXPLORE <-> { STAIRS | GOTO->CURTAIN | SCAN | QR_APPROACH | GEAR }
                       |  final-cell signal
                       v
                     FINAL (stop, face the opening) -> DONE (publishes /quad/ready_to_launch)
 PAUSED   : operator is driving with the joystick (everything yields, resumes 3 s after release)
 FAULT    : gate latched (tilt / operator stop) - needs a human.

Mode owners: EXPLORE/GOTO = explorer, STAIRS = terrain, CURTAIN = curtain, QR_APPROACH = perception,
SCAN/FINAL/STANDUP = this node. Gate caps speeds per mode via /quad/limits.
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
from std_msgs.msg import Bool, Empty, Int32, String

import common as C


class Supervisor(Node):

    def __init__(self):
        super().__init__('supervisor')
        p = C.declare(self, {
            'run_budget_s': 360.0,
            'start_gait': 'general', 'flat_gait': 'general', 'rough_gait': 'general',
            'rough_enabled': False, 'gear_wait_s': 1.2, 'gear_min_dwell_s': 8.0,
            'final_topic': '/final_cell_reached', 'final_type': 'Bool',
            'defer_obstacles': True, 'defer_until_s': 170.0, 'stairs_standoff_m': 1.3,
            'stairs_enabled': True, 'stairs_max_attempts': 3, 'stairs_cooldown_s': 45.0,
            'curtain_enabled': True, 'curtain_live': False, 'curtain_use_vlm': True, 'curtain_vlm_below': 0.85,
            'curtain_standoff_m': 0.70, 'axis_snap': True, 'curtain_max_tries': 4,
            'scan_enabled': True, 'scan_yaw_deg': 90.0, 'scan_hold_s': 1.5,
            'qr_assist_enabled': True, 'qr_assist_below_px': 140.0,
            'lim_explore_vx': 0.25, 'lim_explore_wz': 0.6,
            'lim_rough_vx': 0.15, 'lim_rough_wz': 0.4,
            'lim_stairs_vx': 0.28, 'lim_stairs_wz': 0.5, 'tilt_stairs': 50.0,
            'lim_curtain_vx': 0.16, 'lim_curtain_wz': 0.5,
            'tilt_default': 32.0,
            'final_open_min_m': 2.5, 'final_face_tol': 0.08,
            'stand_wait_s': 7.0, 'calib_wait_s': 8.0,
            'resume_after_manual_s': 3.0,
        })
        self.p = p
        self.mode = 'IDLE'
        self.prev_mode = 'IDLE'
        self.mode_t = time.monotonic()
        self.status = {}
        self.px = self.py = self.yaw = 0.0
        self.have_odom = False
        self.t_mission = None
        self.calibrated = False
        self.stairs_attempt = 0
        self.stairs_block_until = 0.0
        self.stairs_done = 0
        self.curtain_black = []       # (x, y)
        self.curtain_tries = 0
        self.curtain_info = []
        self.curtain_mem = []
        self.stairs_mem = None
        self.cur_patch = None
        self.goto_purpose = None
        self.qr_text = None
        self.scanned_cells = set()
        self.scan_state = None
        self.scan_cooldown = 0.0
        self.qr_cooldown = 0.0
        self.rough_on = False
        self.exh_seen = 0.0
        self.exh_first = 0.0
        self.last_exh_action = 0.0
        self.reset_tried = False
        self.manual_count = 0
        self.manual_t = 0.0
        self.paused_from = None
        self.stand_t0 = None
        self.stand_sent = 0
        self.cloud = None
        self.final_started = None
        self.final_face = None
        self.gear_resume = None
        self.gear_until = 0.0
        self.last_gear_t = 0.0
        self.resume_mode = 'STANDUP'
        self.explore_state = {}
        self.ready_sent = False
        self.budget_warned = False
        self._calib_t0 = None

        self.mode_pub = self.create_publisher(String, C.T_MODE, 10)
        self.limits_pub = self.create_publisher(String, C.T_LIMITS, 10)
        self.cmd_pub = self.create_publisher(Twist, C.T_CMD, 10)
        self.gait_pub = self.create_publisher(String, C.T_GAIT_REQ, 10)
        self.behav_pub = self.create_publisher(String, C.T_BEHAV, 10)
        self.ctrl_pub = self.create_publisher(String, C.T_EXPLORE_CTRL, 10)
        self.recal_pub = self.create_publisher(Empty, '/quad/recalibrate', 10)
        self.ready_pub = self.create_publisher(String, '/quad/ready_to_launch', C.LATCHED_QOS)
        self.sup_pub = self.create_publisher(String, '/quad/supervisor', 10)

        self.create_subscription(String, C.T_STATUS, lambda m: setattr(self, 'status', C.jload(m.data, {})), 10)
        self.create_subscription(String, C.T_EVENT, self.on_event, 50)
        self.create_subscription(String, C.T_EXPLORE_STATE,
                                 lambda m: setattr(self, 'explore_state', C.jload(m.data, {})), 10)
        self.create_subscription(String, '/quad/curtain_info', self.on_curtain_info, 10)
        self.create_subscription(String, C.T_CALIB, self.on_calib, C.LATCHED_QOS)
        self.create_subscription(Odometry, '/odom', self.on_odom, C.SENSOR_QOS)
        self.create_subscription(PointCloud2, C.T_CLOUD, self.on_cloud, C.SENSOR_QOS)
        self.create_subscription(Empty, '/quad/final_manual', lambda m: self.on_final('manual'), 10)
        ft = p['final_topic']
        if ft:
            typ = {'Bool': Bool, 'Empty': Empty, 'String': String, 'Int32': Int32}.get(p['final_type'], Bool)
            self.create_subscription(typ, ft, self.on_final_msg, 10)
        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.heartbeat)
        self.get_logger().info('supervisor up: waiting for /quad/arm')

    # ------------------------------------------------------------- inputs
    def on_odom(self, m):
        self.px, self.py = m.pose.pose.position.x, m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.yaw = C.quat_to_rpy(q.x, q.y, q.z, q.w)[2]
        self.have_odom = True

    def on_cloud(self, m):
        self.cloud = C.cloud_to_xyz(m)

    def on_calib(self, m):
        d = C.jload(m.data, {})
        self.calibrated = bool(d.get('calibrated'))

    def on_curtain_info(self, m):
        self.curtain_info = (C.jload(m.data, {}) or {}).get('patches', [])
        # remember promising patches (arena frame) so they can be tried when exploration is exhausted
        for pa in self.curtain_info:
            if pa.get('score', 0) < 0.25 or pa.get('w', 0) < 0.5:
                continue
            for q in self.curtain_mem:
                if math.hypot(q['x'] - pa['x'], q['y'] - pa['y']) < 0.6:
                    if pa['score'] >= q['score']:
                        q.update(pa)
                    break
            else:
                self.curtain_mem.append(dict(pa))

    def on_final_msg(self, m):
        v = getattr(m, 'data', True)
        if isinstance(v, bool) and not v:
            return
        if isinstance(v, str) and not v.strip():
            return
        self.on_final('topic %s' % self.p['final_topic'])

    def on_final(self, why):
        if self.mode in ('FINAL', 'DONE', 'IDLE', 'FAULT'):
            return
        self.get_logger().warn('FINAL CELL signal (%s) -> stopping' % why)
        self.set_mode('FINAL')
        self.final_started = time.monotonic()
        self.final_face = None

    # ------------------------------------------------------------- mode handling
    def limits_for(self, mode):
        p = self.p
        base = {'tilt': p['tilt_default']}
        if mode == 'STAIRS':
            return {**base, 'vx': p['lim_stairs_vx'], 'vy': 0.0, 'wz': p['lim_stairs_wz'],
                    'tilt': p['tilt_stairs']}
        if mode in ('CURTAIN',):
            return {**base, 'vx': p['lim_curtain_vx'], 'vy': 0.0, 'wz': p['lim_curtain_wz']}
        if mode in ('SCAN', 'FINAL', 'QR_APPROACH', 'STANDUP'):
            return {**base, 'vx': 0.10, 'vy': 0.0, 'wz': 0.5}
        if self.rough_on and mode in ('EXPLORE', 'GOTO'):
            return {**base, 'vx': p['lim_rough_vx'], 'vy': 0.0, 'wz': p['lim_rough_wz']}
        return {**base, 'vx': p['lim_explore_vx'], 'vy': 0.0, 'wz': p['lim_explore_wz']}

    def set_mode(self, mode):
        if mode == self.mode:
            return
        self.get_logger().info('MODE %s -> %s' % (self.mode, mode))
        if self.mode not in ('PAUSED', 'GEAR'):
            self.prev_mode = self.mode
        self.mode = mode
        self.mode_t = time.monotonic()
        self.cmd_pub.publish(Twist())
        self.limits_pub.publish(C.jmsg(self.limits_for(mode)))
        self.mode_pub.publish(String(data=mode))

    def heartbeat(self):
        self.mode_pub.publish(String(data=self.mode))
        self.limits_pub.publish(C.jmsg(self.limits_for(self.mode)))
        el = None if self.t_mission is None else round(time.monotonic() - self.t_mission, 1)
        self.sup_pub.publish(C.jmsg({'mode': self.mode, 'elapsed': el, 'stairs_done': self.stairs_done,
                                     'manual': self.manual_count, 'rough': self.rough_on,
                                     'qr': self.qr_text, 'curtain_tries': self.curtain_tries}))

    def gait(self, name):
        if name:
            self.gait_pub.publish(String(data=name))

    def gear_switch(self, name, then=None):
        """Pause movers, change gait, resume."""
        self.gear_resume = then or (self.mode if self.mode != 'GEAR' else 'EXPLORE')
        self.set_mode('GEAR')
        self.gait(name)
        self.last_gear_t = time.monotonic()
        self.gear_until = self.last_gear_t + self.p['gear_wait_s']

    def maybe_gear(self, now):
        """Keep the gait in line with the rough-ground flag in every walking mode (EXPLORE and GOTO).

        Level-triggered (checked every tick) instead of one-shot on the event, so a rough flag that
        arrives while the dog is in GOTO / QR_APPROACH / ... is still honoured, and rate-limited
        (gear_min_dwell_s) so a flickering detector cannot make the dog change gait every few seconds.
        Returns True when a gear switch was started."""
        p = self.p
        if not p['rough_enabled']:
            return False
        cur = self.status.get('gait')
        want = p['rough_gait'] if self.rough_on else p['flat_gait']
        if cur in (None, 'unknown') or cur == want:
            return False
        if now - self.last_gear_t < p['gear_min_dwell_s']:
            return False
        self.get_logger().warn('gait %s -> %s (rough=%s, mode %s)' % (cur, want, self.rough_on, self.mode))
        self.gear_switch(want, then=self.mode)
        return True

    # ------------------------------------------------------------- events
    def on_event(self, m):
        e = C.jload(m.data, {})
        t = e.get('type')
        p = self.p
        now = time.monotonic()
        if t == 'rough' and p['rough_enabled']:
            if e.get('on') != self.rough_on:
                self.rough_on = bool(e.get('on'))     # the gait follows in tick() via maybe_gear()
        elif t == 'stairs_ahead':
            if (self.mode == 'EXPLORE' and p['stairs_enabled'] and now > self.stairs_block_until):
                if self.deferring():
                    self.remember_stairs(e)
                else:
                    self.stairs_attempt = 0
                    self.start_stairs(e)
        elif t == 'stairs_done':
            self.stairs_done += 1
            self.stairs_attempt = 0
            self.stairs_block_until = now + p['stairs_cooldown_s']
            self.rough_on = False
            self.set_mode('EXPLORE')
        elif t == 'stairs_failed':
            self.stairs_attempt += 1
            self.get_logger().warn('stairs failed (%s), attempt %d' % (e.get('why'), self.stairs_attempt))
            if (self.stairs_attempt < p['stairs_max_attempts']
                    and e.get('why') in ('no_progress', 'climb_timeout')):
                self.start_stairs(e, retry=True)
            else:
                self.stairs_block_until = now + 90.0
                self.set_mode('EXPLORE')
        elif t == 'curtain_candidate':
            if (self.mode == 'EXPLORE' and p['curtain_enabled'] and p['curtain_live']
                    and not self.near_black(e) and not self.deferring()):
                self.start_curtain(e, last_resort=False)
        elif t == 'goto_done' and self.mode == 'GOTO':
            if self.goto_purpose == 'survey':
                self.goto_purpose = None
                self.start_scan()
            elif self.goto_purpose == 'stairs':
                self.goto_purpose = None
                self.stairs_attempt = 0
                self.set_mode('STAIRS')
                self.behav_pub.publish(C.jmsg({'cmd': 'stairs_start', 'attempt': 0,
                                               'dir': self.stairs_mem.get('dir', 'up'),
                                               'dist': p['stairs_standoff_m'], 'retry': False}))
                self.stairs_mem = None
            elif self.cur_patch:
                patch = self.cur_patch
                heading = math.atan2(-patch['ny'], -patch['nx'])
                use_vlm = (p['curtain_use_vlm'] and not patch.get('last_resort')
                           and patch.get('score', 0) < p['curtain_vlm_below'])
                self.set_mode('CURTAIN')
                self.behav_pub.publish(C.jmsg({'cmd': 'push', 'heading': heading, 'vlm': use_vlm,
                                               'last_resort': bool(patch.get('last_resort'))}))
        elif t == 'goto_failed' and self.mode == 'GOTO':
            if self.goto_purpose in ('survey', 'stairs'):
                if self.goto_purpose == 'stairs':
                    self.stairs_mem = None
                    self.stairs_block_until = time.monotonic() + 60.0
                self.goto_purpose = None
            elif self.cur_patch:
                self.curtain_black.append((self.cur_patch['x'], self.cur_patch['y']))
            self.set_mode('EXPLORE')
        elif t == 'curtain_passed':
            if self.cur_patch:
                self.ctrl_pub.publish(C.jmsg({'cmd': 'clear', 'x': self.cur_patch['x'],
                                              'y': self.cur_patch['y'], 'r': 0.8}))
                self.curtain_black.append((self.cur_patch['x'], self.cur_patch['y']))
            self.curtain_tries = 0
            self.set_mode('EXPLORE')
        elif t == 'curtain_failed':
            if self.cur_patch:
                self.curtain_black.append((self.cur_patch['x'], self.cur_patch['y']))
            self.set_mode('EXPLORE')
        elif t == 'exhausted':
            if now - self.exh_seen > 7.0:
                self.exh_first = now
            self.exh_seen = now
        elif t == 'survivor_seen' and p['scan_enabled'] and self.mode == 'EXPLORE':
            cell = tuple(e.get('cell', []))
            if (self.qr_text is None and cell not in self.scanned_cells and now > self.scan_cooldown):
                self.scanned_cells.add(cell)
                if C.cell_of(self.px, self.py) == cell:
                    self.start_scan()
                else:
                    cx, cy = C.cell_centre(cell)
                    self.goto_purpose = 'survey'
                    self.get_logger().warn('survivor seen in cell %s: going in to scan the walls for a QR'
                                           % (cell,))
                    self.set_mode('GOTO')
                    self.ctrl_pub.publish(C.jmsg({'cmd': 'goto', 'x': cx, 'y': cy, 'face': None,
                                                  'timeout': 30}))
        elif t == 'qr_decoded':
            self.qr_text = e.get('text')
            if self.mode == 'SCAN':
                self.end_scan()
        elif t == 'qr_seen':
            if (self.mode in ('EXPLORE', 'SCAN') and p['qr_assist_enabled'] and self.qr_text is None
                    and e.get('w', 999) < p['qr_assist_below_px'] and now > self.qr_cooldown):
                self.scan_state = None
                self.set_mode('QR_APPROACH')
        elif t == 'qr_assist_done' and self.mode == 'QR_APPROACH':
            self.qr_cooldown = now + 10.0
            self.set_mode('EXPLORE')

    def deferring(self):
        """Explore the open area (rooms, survivors, signs) before taking on stairs/curtains."""
        if not self.p['defer_obstacles'] or self.t_mission is None:
            return False
        return (time.monotonic() - self.t_mission) < self.p['defer_until_s'] and not self.explore_state.get('exhausted')

    def remember_stairs(self, e):
        pose = e.get('pose', [self.px, self.py, self.yaw])
        ang = round(pose[2] / (math.pi / 2)) * (math.pi / 2)
        foot = (pose[0] + max(e.get('dist', 1.0), 0.3) * math.cos(ang),
                pose[1] + max(e.get('dist', 1.0), 0.3) * math.sin(ang))
        if self.stairs_mem and math.hypot(self.stairs_mem['fx'] - foot[0], self.stairs_mem['fy'] - foot[1]) < 0.8:
            return
        self.stairs_mem = {'fx': foot[0], 'fy': foot[1], 'ang': ang, 'dir': e.get('dir', 'up')}
        self.get_logger().warn('stairs seen at foot (%.2f, %.2f): deferred, keep-out set' % foot)
        self.ctrl_pub.publish(C.jmsg({'cmd': 'keepout', 'id': 'stairs', 'x': foot[0], 'y': foot[1], 'r': 0.95}))

    def start_stairs_from_memory(self):
        m = self.stairs_mem
        so = self.p['stairs_standoff_m']
        gx = m['fx'] - so * math.cos(m['ang'])
        gy = m['fy'] - so * math.sin(m['ang'])
        # centre on the lattice line across the stairs
        if abs(math.cos(m['ang'])) > 0.5:
            gy = round(gy / C.CELL) * C.CELL
        else:
            gx = round(gx / C.CELL) * C.CELL
        self.ctrl_pub.publish(C.jmsg({'cmd': 'keepout_clear', 'id': 'stairs'}))
        self.goto_purpose = 'stairs'
        self.get_logger().warn('going back to the stairs: stand-off (%.2f, %.2f)' % (gx, gy))
        self.set_mode('GOTO')
        self.ctrl_pub.publish(C.jmsg({'cmd': 'goto', 'x': gx, 'y': gy, 'face': m['ang'], 'timeout': 60}))

    def near_black(self, patch):
        return any(math.hypot(patch['x'] - bx, patch['y'] - by) < 0.7 for bx, by in self.curtain_black)

    def start_stairs(self, e, retry=False):
        self.set_mode('STAIRS')
        self.behav_pub.publish(C.jmsg({'cmd': 'stairs_start', 'attempt': self.stairs_attempt,
                                       'dir': e.get('dir', 'up'), 'retry': bool(retry)}))

    def start_curtain(self, patch, last_resort):
        self.curtain_tries += 1
        self.cur_patch = dict(patch)
        self.cur_patch['last_resort'] = last_resort
        so = self.p['curtain_standoff_m']
        if self.p['axis_snap']:
            # the arena is a grid aligned with the start pose: push along an axis, centred in the cell
            ang = round(math.atan2(patch['ny'], patch['nx']) / (math.pi / 2)) * (math.pi / 2)
            self.cur_patch['nx'], self.cur_patch['ny'] = round(math.cos(ang), 3), round(math.sin(ang), 3)
            patch = self.cur_patch
        gx = patch['x'] + patch['nx'] * so
        gy = patch['y'] + patch['ny'] * so
        if self.p['axis_snap']:
            if abs(patch['nx']) > 0.5:
                gy = round(gy / C.CELL) * C.CELL
            else:
                gx = round(gx / C.CELL) * C.CELL
        face = math.atan2(-patch['ny'], -patch['nx'])
        self.get_logger().warn('CURTAIN candidate %s (last_resort=%s) -> goto (%.2f, %.2f)'
                               % (patch, last_resort, gx, gy))
        self.goto_purpose = 'curtain'
        self.set_mode('GOTO')
        self.ctrl_pub.publish(C.jmsg({'cmd': 'goto', 'x': gx, 'y': gy, 'face': face, 'timeout': 40}))

    # ------------------------------------------------------------- SCAN
    def start_scan(self):
        a = math.radians(self.p['scan_yaw_deg'])
        self.scan_state = {'y0': self.yaw, 'targets': [a, -a, 0.0], 'i': 0, 'hold_until': None,
                           't0': time.monotonic()}
        self.set_mode('SCAN')

    def end_scan(self):
        self.scan_state = None
        self.scan_cooldown = time.monotonic() + 15.0
        self.set_mode('EXPLORE')

    def run_scan(self):
        s = self.scan_state
        if s is None or time.monotonic() - s['t0'] > 30.0:
            return self.end_scan()
        if s['i'] >= len(s['targets']):
            return self.end_scan()
        tgt = C.wrap(s['y0'] + s['targets'][s['i']])
        e = C.wrap(tgt - self.yaw)
        tw = Twist()
        if abs(e) > 0.08 and s['hold_until'] is None:
            tw.angular.z = C.clamp(1.5 * e, -0.45, 0.45)
            if abs(tw.angular.z) < 0.15:
                tw.angular.z = math.copysign(0.15, e)
        else:
            if s['hold_until'] is None:
                s['hold_until'] = time.monotonic() + self.p['scan_hold_s']
            elif time.monotonic() > s['hold_until']:
                s['i'] += 1
                s['hold_until'] = None
        self.cmd_pub.publish(tw)

    # ------------------------------------------------------------- FINAL
    def opening_bearing(self):
        """World bearing of the widest open sector (the missing wall), or None."""
        pts = self.cloud
        if pts is None or pts.shape[0] < 50:
            return None
        dx, dy, h = pts[:, 0] - self.px, pts[:, 1] - self.py, pts[:, 2]
        r = np.hypot(dx, dy)
        sel = (h > 0.3) & (h < 1.5) & (r > 0.3)
        a = np.arctan2(dy[sel], dx[sel])
        rr = r[sel]
        nb = 72
        b = ((a + math.pi) / (2 * math.pi) * nb).astype(int) % nb
        rmin = np.full(nb, 9.0)
        np.minimum.at(rmin, b, rr)
        open_ = rmin >= self.p['final_open_min_m']
        if not open_.any():
            return None
        # widest contiguous open run (circular)
        best, best_len, i = None, 0, 0
        o2 = np.concatenate([open_, open_])
        while i < nb:
            if o2[i]:
                j = i
                while j < i + nb and o2[j]:
                    j += 1
                if j - i > best_len:
                    best, best_len = i, j - i
                i = j
            else:
                i += 1
        if best is None or best_len < 2:
            return None
        mid = (best + best_len / 2.0) % nb
        return (mid + 0.0) / nb * 2 * math.pi - math.pi

    def run_final(self):
        now = time.monotonic()
        tw = Twist()
        if now - self.final_started < 1.2:       # stand still: autonomous stop in the cell
            self.cmd_pub.publish(tw)
            return
        if self.final_face is None:
            self.final_face = self.opening_bearing()
            self.final_try = getattr(self, 'final_try', 0) + 1
            if self.final_face is None:
                if now - self.final_started > 6.0:
                    self.get_logger().warn('no opening found - staying put, ready anyway')
                    return self.finish_final(None)
                self.cmd_pub.publish(tw)
                return
            self.get_logger().warn('facing opening at bearing %.0f deg' % math.degrees(self.final_face))
        e = C.wrap(self.final_face - self.yaw)
        if abs(e) > self.p['final_face_tol']:
            tw.angular.z = C.clamp(1.4 * e, -0.5, 0.5)
            if abs(tw.angular.z) < 0.15:
                tw.angular.z = math.copysign(0.15, e)
            self.cmd_pub.publish(tw)
            if now - self.final_started > 25.0:
                self.finish_final(self.final_face)
            return
        self.finish_final(self.final_face)

    def finish_final(self, face):
        self.cmd_pub.publish(Twist())
        el = None if self.t_mission is None else round(time.monotonic() - self.t_mission, 1)
        self.ready_pub.publish(C.jmsg({'ready': True, 'yaw': round(self.yaw, 3), 'x': round(self.px, 2),
                                       'y': round(self.py, 2), 'faced_opening': face is not None,
                                       'elapsed': el}))
        self.get_logger().warn('READY TO LAUNCH (elapsed %s s). Launch the drone manually.' % el)
        self.set_mode('DONE')

    # ------------------------------------------------------------- main tick
    def tick(self):
        p = self.p
        now = time.monotonic()
        st = self.status
        armed = bool(st.get('armed'))
        if st.get('latched') and self.mode not in ('FAULT',):
            self.get_logger().error('FAULT: gate latched (%s)' % st.get('reason'))
            self.set_mode('FAULT')
            return
        if self.mode == 'FAULT':
            if not st.get('latched'):
                self.set_mode('IDLE')
            return
        if not armed:
            if self.mode not in ('IDLE', 'DONE'):
                self.resume_mode = self.mode if self.mode in ('EXPLORE',) else 'STANDUP'
                self.set_mode('IDLE')
            return
        if self.mode == 'IDLE':
            self.stand_t0 = None
            self.set_mode('STANDUP')
            return
        # manual joystick override
        if st.get('manual') and self.mode not in ('PAUSED', 'DONE', 'STANDUP'):
            self.paused_from = self.mode
            self.manual_count += 1
            self.manual_t = now
            self.get_logger().warn('MANUAL OVERRIDE #%d: pausing autonomy' % self.manual_count)
            self.set_mode('PAUSED')
            return
        if self.mode == 'PAUSED':
            if st.get('manual'):
                self.manual_t = now
            elif now - self.manual_t > p['resume_after_manual_s']:
                back = self.paused_from or 'EXPLORE'
                if back in ('STAIRS', 'CURTAIN', 'GOTO', 'QR_APPROACH', 'SCAN', 'FINAL'):
                    back = 'EXPLORE' if back != 'FINAL' else 'FINAL'
                self.get_logger().warn('resuming in %s' % back)
                if back == 'FINAL':
                    self.final_started = now
                self.set_mode(back)
            return
        if (self.t_mission is not None and now - self.t_mission > p['run_budget_s']
                and not self.budget_warned):
            self.budget_warned = True
            self.get_logger().warn('run budget %.0fs exceeded (still running)' % p['run_budget_s'])
        # ---- per-mode logic
        if self.mode in ('EXPLORE', 'GOTO') and self.maybe_gear(now):
            return
        if self.mode == 'STANDUP':
            self.run_standup(now)
        elif self.mode == 'GEAR':
            self.cmd_pub.publish(Twist())
            if now > self.gear_until:
                self.set_mode(self.gear_resume or 'EXPLORE')
        elif self.mode == 'SCAN':
            self.run_scan()
        elif self.mode == 'FINAL':
            self.run_final()
        elif self.mode == 'EXPLORE':
            self.run_explore(now)
        elif self.mode in ('GOTO', 'CURTAIN', 'STAIRS', 'QR_APPROACH'):
            # watchdog for stuck sub-behaviours
            lim = {'GOTO': 60, 'CURTAIN': 45, 'STAIRS': 90, 'QR_APPROACH': 20}[self.mode]
            if now - self.mode_t > lim:
                self.get_logger().error('%s timed out -> EXPLORE' % self.mode)
                self.set_mode('EXPLORE')

    def run_standup(self, now):
        st = self.status
        if self.stand_t0 is None:
            self.stand_t0 = now
            self.stand_sent = 0
        standing = st.get('standing')
        age = st.get('joint_age', 99)
        if standing is False and age < 2.0 and self.stand_sent < 2 and now - self.stand_t0 > 1.0:
            self.get_logger().warn('dog is sitting: sending Stand toggle (%d)' % (self.stand_sent + 1))
            self.gait('stand_toggle')
            self.stand_sent += 1
            self.stand_t0 = now
            return
        ready = (standing is True) or (standing is None and now - self.stand_t0 > 3.0)
        if standing is False and now - self.stand_t0 > self.p['stand_wait_s'] and self.stand_sent >= 2:
            self.get_logger().error('dog not standing after 2 toggles - need a human; waiting')
            self.stand_t0 = now
            return
        if not ready:
            return
        if self._calib_t0 is None:
            self._calib_t0 = now
            self.recal_pub.publish(Empty())
            self.gait(self.p['start_gait'])
            self.calibrated = False
            return
        if self.calibrated or now - self._calib_t0 > self.p['calib_wait_s']:
            self._calib_t0 = None
            if self.t_mission is None:
                self.t_mission = now
                self.get_logger().warn('MISSION START')
            self.set_mode('EXPLORE')

    def run_explore(self, now):
        p = self.p
        # defer window over: go and take on the remembered obstacles
        if (p['defer_obstacles'] and self.t_mission is not None
                and now - self.t_mission > p['defer_until_s'] and now - self.last_exh_action > 8.0):
            if self.stairs_mem and p['stairs_enabled']:
                self.last_exh_action = now
                self.start_stairs_from_memory()
                return
            cands = sorted([c for c in self.curtain_mem if c['score'] >= 0.45 and not self.near_black(c)],
                           key=lambda c: -c['score'])
            if cands and p['curtain_enabled'] and self.curtain_tries < p['curtain_max_tries']:
                self.last_exh_action = now
                self.start_curtain(cands[0], last_resort=False)
                return
        # exploration exhausted -> try curtain candidates, then a map reset, then ask for help
        if (self.exh_seen and now - self.exh_seen < 5.0 and now - self.exh_first > 6.0
                and now - self.last_exh_action > 6.0 and self.explore_state.get('exhausted')):
            self.last_exh_action = now
            if self.stairs_mem and p['stairs_enabled']:
                self.start_stairs_from_memory()
                return
            if p['curtain_enabled'] and self.curtain_tries < p['curtain_max_tries']:
                cands = sorted([c for c in self.curtain_mem if not self.near_black(c)],
                               key=lambda c: -c['score'])
                if cands:
                    self.start_curtain(cands[0], last_resort=True)
                    return
            if not self.reset_tried:
                self.reset_tried = True
                self.get_logger().warn('exploration exhausted, no candidates: resetting map once')
                self.ctrl_pub.publish(C.jmsg({'cmd': 'reset_map'}))
            else:
                self.get_logger().error('EXPLORATION EXHAUSTED and no curtain/stairs candidate: '
                                        'Captain decision needed (manual / final signal)')


def main():
    rclpy.init()
    n = Supervisor()
    try:
        C.safe_spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
