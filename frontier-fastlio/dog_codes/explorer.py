#!/usr/bin/env python3
"""explorer: self-contained mapping + frontier exploration + path following.

Uses only /quad/cloud (height-above-feet point cloud in the odom/arena frame) and /odom,
so it does not depend on Nav2 / slam_toolbox / the TAs' /scan parameters.

* 2D log-odds grid (0.1 m). Obstacles = cloud points with height in [obs_lo, obs_hi]
  (so floor, rubble and stair steps are NOT obstacles; walls, props and curtains are).
* Frontier = known-free cell next to unknown. Goal = cheapest reachable frontier cluster.
* Dijkstra over inflated free space, pure-pursuit follower -> /quad/cmd_vel.
* Active only while /quad/mode == EXPLORE (mapping runs always).
* Publishes /quad/explore_state (JSON) and an 'exhausted' event when no frontier is left.
"""
import heapq
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy import ndimage as ndi
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

import common as C

NB = 360  # angular bins for free-space carving


class Explorer(Node):

    def __init__(self):
        super().__init__('explorer')
        p = C.declare(self, {
            'res': 0.1, 'size_m': 40.0,
            'obs_lo': 0.30, 'obs_hi': 1.60, 'obs_extent': 0.18, 'obs_tall': 0.70, 'min_r': 0.35, 'free_range': 4.0,
            'inflate_m': 0.20, 'plan_hz': 1.5, 'ctrl_hz': 10.0,
            'v_max': 0.25, 'wz_max': 0.6, 'lookahead': 0.45, 'goal_tol': 0.30,
            'min_frontier_cells': 5, 'frontier_unknown_min': 12,
            'front_stop_m': 0.38, 'front_slow_m': 0.8,
            'stuck_s': 7.0, 'stuck_dist': 0.06, 'blacklist_s': 45.0,
            'exhausted_after': 3, 'visit_radius': 0.40, 'coverage': True, 'window_m': 10.0,
            # the lidar cannot see the floor within ~1 m of the dog or behind it. At the START nothing around the dog
            # has ever been seen from a distance, so unknown cells there say nothing about the map and a frontier
            # closer than front_min_robot_m to the robot is no real frontier (trial 1: the dog chased a "frontier"
            # between itself and the wall 0.3 m behind it). Applied only while the dog is within front_guard_travel_m
            # of its start pose, and switched off for good once it has travelled front_guard_travel_m from it, so the
            # behaviour after the start is exactly the old one. front_min_robot_m = 0 disables.
            'front_min_robot_m': 1.0, 'front_guard_travel_m': 1.5,
            # 20 Nov trial 2: behind the start line the lidar only looks UP (it is pitched down at the front), so a low
            # wall there is invisible and the unknown area behind the dog is no frontier. During the same start phase
            # (until front_guard_travel_m is covered) cells within rear_fence_m BEHIND the start line are not traversable.
            # If that leaves no goal for rear_fence_timeout_s the start guards are lifted (old behaviour). 0 disables.
            'rear_fence_m': 1.5, 'rear_fence_timeout_s': 8.0,
            # bump detection: commanded forward (>= bump_min_vx) for bump_window_s but moved < bump_ratio of the commanded
            # distance => something unseen is in the way: mark a bump obstacle bump_d0..bump_d1 m ahead (valid bump_ttl_s),
            # blacklist the goal and back off.
            'bump_enabled': True, 'bump_window_s': 3.5, 'bump_min_vx': 0.10, 'bump_ratio': 0.2,
            'bump_d0': 0.35, 'bump_d1': 0.50, 'bump_half_w': 0.15, 'bump_ttl_s': 60.0,
            # keep a chosen goal at least this long (stops the goal flipping between left / right / behind)
            'goal_commit_s': 3.0,
            'occ_thresh': 1.0, 'free_thresh': -0.4,
            'start_rotate_scan': True, 'debug_png': '',
        })
        p['debug_png'] = p['debug_png'] or os.environ.get('DOG_DEBUG_PNG', '')
        self.p = p
        r = p['res']
        self.n = int(p['size_m'] / r)
        self.half = self.n // 2
        self.L = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros((self.n, self.n), bool)
        ir = max(1, int(round(p['inflate_m'] / r)))
        yy, xx = np.mgrid[-ir:ir + 1, -ir:ir + 1]
        self.disk = (xx * xx + yy * yy) <= ir * ir

        self.px = self.py = self.pz = 0.0
        self.yaw = 0.0
        self.have_odom = False
        self.mode = 'IDLE'
        self.path = []
        self.goal = None
        self.frontier_goal_cell = None
        self.blacklist = []  # (x, y, expiry)
        self.exhausted_n = 0
        self.exhausted = False
        self.visited = set()
        self.start_xy = None
        self.start_yaw = 0.0
        self.guard_done = False
        self.fence_idle_since = None
        self.bump = C.BumpDetector(window_s=p['bump_window_s'], min_cmd_vx=p['bump_min_vx'], ratio=p['bump_ratio'])
        self.bump_t = np.full((self.n, self.n), -1e9, np.float32)   # time each cell was marked by a bump
        self.n_bumps = 0
        self.last_vx = 0.0
        self.gate_ok = False          # gate armed, not latched, operator not driving: our commands really reach the dog
        self.goal_t = 0.0
        self.turn_sign = 0.0
        self.n_front = 0
        self.last_move = (time.monotonic(), 0.0, 0.0)
        self.backoff_until = 0.0
        self.cmd_avg = 0.0
        self.last_exh_evt = 0.0
        self.n_clouds = 0
        self.scan_phase = None
        self.goto = None
        self.goto_done_sent = False
        self.keepout = {}
        self.visited_c = set()
        self.goal_kind = ''

        self.g_plan = MutuallyExclusiveCallbackGroup()
        self.g_map = MutuallyExclusiveCallbackGroup()
        self.cmd_pub = self.create_publisher(Twist, C.T_CMD, 10)
        self.state_pub = self.create_publisher(String, C.T_EXPLORE_STATE, 10)
        self.evt_pub = self.create_publisher(String, C.T_EVENT, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, C.SENSOR_QOS)
        self.create_subscription(PointCloud2, C.T_CLOUD, self.on_cloud, C.SENSOR_QOS,
                                 callback_group=self.g_map)
        self.create_subscription(String, C.T_MODE, self.on_mode, 10)
        self.create_subscription(String, C.T_EXPLORE_CTRL, self.on_ctrl, 10)
        self.create_subscription(String, C.T_STATUS, self.on_status, 10)
        self.create_timer(1.0 / p['plan_hz'], self.plan, callback_group=self.g_plan)
        self.create_timer(1.0 / p['ctrl_hz'], self.control)
        self.create_timer(0.5, self.publish_state)
        self.get_logger().info('explorer up (grid %dx%d @ %.2f m)' % (self.n, self.n, r))

    # ---------------------------------------------------------------- grid helpers
    def w2g(self, x, y):
        r = self.p['res']
        return int(round(x / r)) + self.half, int(round(y / r)) + self.half  # (col,row)

    def g2w(self, c, rw):
        r = self.p['res']
        return (c - self.half) * r, (rw - self.half) * r

    def in_grid(self, c, r):
        return 0 <= c < self.n and 0 <= r < self.n

    # ---------------------------------------------------------------- inputs
    def on_odom(self, m):
        self.px = m.pose.pose.position.x
        self.py = m.pose.pose.position.y
        self.pz = m.pose.pose.position.z
        q = m.pose.pose.orientation
        self.yaw = C.quat_to_rpy(q.x, q.y, q.z, q.w)[2]
        self.have_odom = True
        if self.start_xy is None:
            self.start_xy = (self.px, self.py)
            self.start_yaw = self.yaw
        self.visited.add(C.cell_of(self.px, self.py))
        ci = C.cell_of(self.px, self.py)
        cx, cy = C.cell_centre(ci)
        if math.hypot(self.px - cx, self.py - cy) < self.p['visit_radius']:
            self.visited_c.add(ci)

    def on_mode(self, m):
        if m.data != self.mode:
            if m.data in ('EXPLORE', 'GOTO'):
                self.path = []
                self.exhausted_n = 0
                self.exhausted = False
                self.last_move = (time.monotonic(), self.px, self.py)
                self.bump.reset()
            self.mode = m.data

    def on_status(self, m):
        d = C.jload(m.data, {})
        self.gate_ok = bool(d.get('armed')) and not d.get('latched') and not d.get('manual')

    def on_ctrl(self, m):
        d = C.jload(m.data, {})
        if d.get('cmd') == 'clear':
            x, y, rad = d['x'], d['y'], d.get('r', 0.6)
            c0, r0 = self.w2g(x, y)
            k = int(rad / self.p['res'])
            sl = (slice(max(r0 - k, 0), r0 + k + 1), slice(max(c0 - k, 0), c0 + k + 1))
            self.L[sl] = 0.0
            self.seen[sl] = False
            self.path = []
            self.exhausted = False
            self.exhausted_n = 0
        elif d.get('cmd') == 'keepout':
            self.keepout[d['id']] = (float(d['x']), float(d['y']), float(d.get('r', 0.9)))
        elif d.get('cmd') == 'keepout_clear':
            self.keepout.pop(d.get('id'), None)
        elif d.get('cmd') == 'goto':
            self.goto = {'x': float(d['x']), 'y': float(d['y']), 'face': d.get('face'),
                         't0': time.monotonic(), 'timeout': float(d.get('timeout', 45.0)),
                         'arrived': False}
            self.goto_done_sent = False
            self.path = []
        elif d.get('cmd') == 'blacklist_here':
            self.blacklist.append((self.px, self.py, time.monotonic() + 120.0))
        elif d.get('cmd') == 'reset_map':
            self.L[:] = 0
            self.seen[:] = False
            self.path = []

    def on_cloud(self, msg):
        if not self.have_odom:
            return
        pts = C.cloud_to_xyz(msg)
        if pts.shape[0] == 0:
            return
        self.n_clouds += 1
        p = self.p
        dx = pts[:, 0] - self.px
        dy = pts[:, 1] - self.py
        rr = np.hypot(dx, dy)
        h = pts[:, 2]
        # obstacle = a map cell whose points reach above obs_lo AND span a vertical extent (walls,
        # props, curtains) -- a raised flat floor (stair tops, platforms) has no vertical extent.
        cand = (h >= -0.15) & (h <= p['obs_hi']) & (rr > p['min_r'])
        ci = np.round(pts[cand, 0] / p['res']).astype(np.int64) + self.half
        cj = np.round(pts[cand, 1] / p['res']).astype(np.int64) + self.half
        hh = h[cand]
        okc = (ci >= 0) & (ci < self.n) & (cj >= 0) & (cj < self.n)
        ci, cj, hh = ci[okc], cj[okc], hh[okc]
        key = cj * self.n + ci
        obs_idx = np.zeros(0, np.int64)
        if key.size:
            order = np.argsort(key, kind='stable')
            ks, hs = key[order], hh[order]
            starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
            hmax = np.maximum.reduceat(hs, starts)
            hmin = np.minimum.reduceat(hs, starts)
            cnt = np.diff(np.r_[starts, ks.size])
            isobs = (hmax >= p['obs_lo']) & (((hmax - hmin) >= p['obs_extent']) | (hmax >= p['obs_tall'])) \
                & (cnt >= 2)
            obs_idx = ks[starts][isobs]
        ocx = (obs_idx % self.n - self.half) * p['res']
        ocy = (obs_idx // self.n - self.half) * p['res']
        sel_r = np.hypot(ocx - self.px, ocy - self.py)
        sel = np.zeros(pts.shape[0], bool)      # (kept for the free-space code below)
        ang = np.arctan2(ocy - self.py, ocx - self.px)
        rng = sel_r
        # --- free space: along each bearing up to the nearest obstacle; where there is no obstacle,
        #     only as far as FLOOR returns prove it (rays that pass over wall tops prove nothing)
        bins = ((ang + math.pi) / (2 * math.pi) * NB).astype(int) % NB
        rob = np.full(NB, np.inf, np.float32)
        if rng.size:
            np.minimum.at(rob, bins, rng)
        fl = (np.abs(h) < 0.12) & (rr > p['min_r'])
        rfl = np.zeros(NB, np.float32)
        if fl.any():
            bf = ((np.arctan2(dy[fl], dx[fl]) + math.pi) / (2 * math.pi) * NB).astype(int) % NB
            np.maximum.at(rfl, bf, rr[fl])
        free_to = np.where(np.isfinite(rob), np.minimum(rob - 0.12, p['free_range']),
                           np.minimum(rfl, p['free_range']) - 0.1)
        free_to = np.maximum(free_to, 0.0)
        stepn = int(p['free_range'] / (p['res'] * 0.7)) + 1
        ts = np.linspace(0.2, p['free_range'], stepn, dtype=np.float32)[None, :]
        centre = (np.arange(NB) + 0.5) / NB * 2 * math.pi - math.pi
        m_ok = ts <= free_to[:, None]
        fx = self.px + np.cos(centre)[:, None] * ts
        fy = self.py + np.sin(centre)[:, None] * ts
        fx, fy = fx[m_ok], fy[m_ok]
        c = np.round(fx / p['res']).astype(int) + self.half
        rw = np.round(fy / p['res']).astype(int) + self.half
        ok = (c >= 0) & (c < self.n) & (rw >= 0) & (rw < self.n)
        flat = np.unique(rw[ok] * self.n + c[ok])
        self.L.flat[flat] = np.maximum(self.L.flat[flat] - 0.35, -4.0)
        self.seen.flat[flat] = True
        # --- occupied endpoints
        flat = obs_idx
        self.L.flat[flat] = np.minimum(self.L.flat[flat] + 0.9, 4.0)
        self.seen.flat[flat] = True

    # ---------------------------------------------------------------- planning
    def window(self):
        k = int(self.p['window_m'] / self.p['res'])
        c0, r0 = self.w2g(self.px, self.py)
        c0 = C.clamp(c0, k, self.n - k - 1)
        r0 = C.clamp(r0, k, self.n - k - 1)
        return int(c0 - k), int(c0 + k + 1), int(r0 - k), int(r0 + k + 1)

    def plan(self):
        if not self.have_odom or self.mode not in ('EXPLORE', 'GOTO') or self.n_clouds < 3:
            return
        p = self.p
        t = time.monotonic()
        if self.mode == 'GOTO' and (self.goto is None or self.goto_done_sent):
            return
        self.blacklist = [b for b in self.blacklist if b[2] > t]
        if (not self.guard_done and self.start_xy is not None and
                math.hypot(self.px - self.start_xy[0], self.py - self.start_xy[1]) >= p['front_guard_travel_m']):
            self.guard_done = True          # the dog has left its start: from now on the map is seen from a distance
        fence_on = (p['rear_fence_m'] > 0 and not self.guard_done and self.start_xy is not None
                    and self.mode == 'EXPLORE')
        c_lo, c_hi, r_lo, r_hi = self.window()
        L = self.L[r_lo:r_hi, c_lo:c_hi]
        seen = self.seen[r_lo:r_hi, c_lo:c_hi]
        occ = (L > p['occ_thresh']) | ((t - self.bump_t[r_lo:r_hi, c_lo:c_hi]) < p['bump_ttl_s'])
        infl = ndi.binary_dilation(occ, structure=self.disk)
        free = seen & ~infl & (L < 0.5)
        for (kx, ky, kr) in self.keepout.values():
            kc, kr_ = self.w2g(kx, ky)
            kc -= c_lo
            kr_ -= r_lo
            rad = int(kr / p['res'])
            y0, y1 = max(kr_ - rad, 0), min(kr_ + rad + 1, free.shape[0])
            x0, x1 = max(kc - rad, 0), min(kc + rad + 1, free.shape[1])
            if y0 < y1 and x0 < x1:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                free[y0:y1, x0:x1] &= ((yy - kr_) ** 2 + (xx - kc) ** 2) > rad * rad
        if fence_on:
            xs = (np.arange(free.shape[1]) + c_lo - self.half) * p['res']
            ys = (np.arange(free.shape[0]) + r_lo - self.half) * p['res']
            X, Y = np.meshgrid(xs, ys)
            free &= ~C.rear_fence_mask(X, Y, self.start_xy, self.start_yaw, p['rear_fence_m'])
        # robot cell and neighbours always traversable
        rc, rr_ = self.w2g(self.px, self.py)
        rc -= c_lo
        rr_ -= r_lo
        for dc in range(-2, 3):
            for dr in range(-2, 3):
                if 0 <= rr_ + dr < free.shape[0] and 0 <= rc + dc < free.shape[1]:
                    free[rr_ + dr, rc + dc] = True
        dist_obs = ndi.distance_transform_edt(~occ) * p['res']
        pen = 1.0 + 3.0 * np.clip((0.7 - dist_obs) / 0.4, 0.0, 1.0)

        # ---- Dijkstra from the robot over free cells
        H, W = free.shape
        INF = 1e9
        dist = np.full((H, W), INF, np.float32)
        parent = np.full((H, W, 2), -1, np.int16)
        dist[rr_, rc] = 0.0
        pq = [(0.0, rr_, rc)]
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        while pq:
            d, r0, c0 = heapq.heappop(pq)
            if d > dist[r0, c0]:
                continue
            for dr, dc, w in nbrs:
                r1, c1 = r0 + dr, c0 + dc
                if r1 < 0 or c1 < 0 or r1 >= H or c1 >= W or not free[r1, c1]:
                    continue
                nd = d + w * pen[r1, c1]
                if nd < dist[r1, c1]:
                    dist[r1, c1] = nd
                    parent[r1, c1] = (r0, c0)
                    heapq.heappush(pq, (nd, r1, c1))
        reach = dist < INF / 2

        if self.mode == 'GOTO':
            g = self.goto
            if t - g['t0'] > g['timeout']:
                self.goto_done_sent = True
                self.path = []
                self.evt_pub.publish(C.jmsg({'type': 'goto_failed', 'why': 'timeout'}))
                return
            idx = np.argwhere(reach)
            if len(idx) == 0:
                return
            wx = (idx[:, 1] + c_lo - self.half) * p['res']
            wy = (idx[:, 0] + r_lo - self.half) * p['res']
            dd = (wx - g['x']) ** 2 + (wy - g['y']) ** 2
            k = int(np.argmin(dd))
            if math.sqrt(float(dd[k])) > g.get('max_miss', 0.7):
                g['miss_n'] = g.get('miss_n', 0) + 1
                if g['miss_n'] >= 3:
                    self.goto_done_sent = True
                    self.path = []
                    self.evt_pub.publish(C.jmsg({'type': 'goto_failed', 'why': 'unreachable'}))
                return
            r1, c1 = int(idx[k][0]), int(idx[k][1])
            path = []
            guard = 0
            while (r1, c1) != (rr_, rc) and guard < 4000:
                path.append(self.g2w(c1 + c_lo, r1 + r_lo))
                pr, pc = parent[r1, c1]
                if pr < 0:
                    break
                r1, c1 = int(pr), int(pc)
                guard += 1
            path.reverse()
            self.path = path
            self.goal = (g['x'], g['y'])
            return

        # ---- frontiers
        unknown = ~seen
        unk_n = ndi.uniform_filter(unknown.astype(np.float32), size=7) * 49.0
        near_unknown = ndi.binary_dilation(unknown, structure=np.ones((3, 3), bool))
        front = reach & near_unknown & (unk_n >= p['frontier_unknown_min']) & (dist_obs > 0.3)
        if p['front_min_robot_m'] > 0 and not self.guard_done:
            gr, gc = np.ogrid[:H, :W]
            front &= ((gr - rr_) ** 2 + (gc - rc) ** 2) * (p['res'] ** 2) > p['front_min_robot_m'] ** 2
        if p['debug_png']:
            self.get_logger().info('plan: reach=%d free=%d near_unk&reach=%d front=%d' % (
                int(reach.sum()), int(free.sum()), int((reach & near_unknown).sum()), int(front.sum())),
                throttle_duration_sec=2.0)
        lab, nlab = ndi.label(front, structure=np.ones((3, 3), bool))
        best, best_cost = None, INF
        self.n_front = 0
        for i in range(1, nlab + 1):
            cells = np.argwhere(lab == i)
            if len(cells) < p['min_frontier_cells']:
                continue
            self.n_front += 1
            # reachable member closest to robot
            ds = dist[cells[:, 0], cells[:, 1]]
            j = int(np.argmin(ds))
            rr0, cc0 = cells[j]
            gx, gy = self.g2w(cc0 + c_lo, rr0 + r_lo)
            if any(math.hypot(gx - bx, gy - by) < 0.6 for bx, by, _ in self.blacklist):
                continue
            ang = math.atan2(gy - self.py, gx - self.px)
            turn = abs(C.wrap(ang - self.yaw))
            cost = float(ds[j]) * p['res'] + 0.25 * turn - 0.02 * min(len(cells), 30)
            if cost < best_cost:
                best_cost, best = cost, (rr0, cc0, gx, gy)
        if best is None and p['coverage']:
            # no frontier left: walk to the centre of every reachable lattice cell not yet visited
            # (rooms, survivors, signs and QR codes need a close look)
            ci0 = C.cell_of(self.px, self.py)
            bcost, bc = INF, None
            for i in range(ci0[0] - 6, ci0[0] + 7):
                for j in range(ci0[1] - 6, ci0[1] + 7):
                    if (i, j) in self.visited_c:
                        continue
                    gx, gy = C.cell_centre((i, j))
                    cc, rw = self.w2g(gx, gy)
                    cc -= c_lo
                    rw -= r_lo
                    if not (0 <= rw < H and 0 <= cc < W) or not reach[rw, cc]:
                        continue
                    if any(math.hypot(gx - bx, gy - by) < 0.5 for bx, by, _ in self.blacklist):
                        continue
                    cost = float(dist[rw, cc])
                    if cost < bcost:
                        bcost, bc = cost, (rw, cc, gx, gy)
            if bc is not None:
                best = bc
                self.goal_kind = 'coverage'
        elif best is not None:
            self.goal_kind = 'frontier'
        if best is None:
            self.path = []
            self.goal = None
            if fence_on:
                # nothing to do while the rear is fenced off: after rear_fence_timeout_s give the start guards up
                if self.fence_idle_since is None:
                    self.fence_idle_since = t
                elif t - self.fence_idle_since > p['rear_fence_timeout_s']:
                    self.guard_done = True
                    self.get_logger().warn('no goal for %.0f s with the start guards on: guards lifted'
                                           % p['rear_fence_timeout_s'])
                return
            self.exhausted_n += 1
            if self.exhausted_n >= p['exhausted_after']:
                self.exhausted = True
                if t - self.last_exh_evt > 4.0:
                    self.last_exh_evt = t
                    self.evt_pub.publish(C.jmsg({'type': 'exhausted', 'src': 'explorer',
                                                 'visited': len(self.visited)}))
            return
        self.exhausted_n = 0
        self.exhausted = False
        self.fence_idle_since = None
        # keep the current goal for goal_commit_s while it is still reachable and not blacklisted: the best goal can
        # flip between left / right / behind from one plan to the next (20 Nov 10:07: 8 turn reversals in 12 s)
        if self.goal is not None and self.path and (t - self.goal_t) < p['goal_commit_s']:
            gcx, gcy = self.w2g(*self.goal)
            gcx -= c_lo
            gcy -= r_lo
            if (0 <= gcy < H and 0 <= gcx < W and reach[gcy, gcx]
                    and not any(math.hypot(self.goal[0] - bx, self.goal[1] - by) < 0.6
                                for bx, by, _ in self.blacklist)):
                best = (gcy, gcx, self.goal[0], self.goal[1])
        rr0, cc0, gx, gy = best
        if self.goal is None or math.hypot(gx - self.goal[0], gy - self.goal[1]) > 0.5:
            self.goal_t = t
        path = []
        r1, c1 = rr0, cc0
        guard = 0
        while (r1, c1) != (rr_, rc) and guard < 4000:
            path.append(self.g2w(c1 + c_lo, r1 + r_lo))
            pr, pc = parent[r1, c1]
            if pr < 0:
                break
            r1, c1 = int(pr), int(pc)
            guard += 1
        path.reverse()
        self.path = path
        self.goal = (gx, gy)

    # ---------------------------------------------------------------- control
    def front_clearance(self):
        """Distance to the nearest raw occupied cell in a +/-35 deg cone ahead (<=1.2 m)."""
        r = self.p['res']
        best = 9.9
        for a in np.linspace(-0.45, 0.45, 7):
            ang = self.yaw + a
            for d in np.arange(0.25, 1.2, r):
                c, rw = self.w2g(self.px + math.cos(ang) * d, self.py + math.sin(ang) * d)
                if self.in_grid(c, rw) and (self.L[rw, c] > self.p['occ_thresh']
                                            or time.monotonic() - self.bump_t[rw, c] < self.p['bump_ttl_s']):
                    best = min(best, d)
                    break
        return best

    def send(self, tw):
        self.last_vx = tw.linear.x
        self.cmd_pub.publish(tw)

    def on_bump(self, t):
        """The dog pushed against something the map does not show. Mark it (sticky for bump_ttl_s: the free-space carving
        of the cloud would otherwise erase it again), blacklist the goal, back off."""
        p = self.p
        r = p['res']
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        n = 0
        for d in np.arange(p['bump_d0'], p['bump_d1'] + 1e-6, r):
            for s in np.arange(-p['bump_half_w'], p['bump_half_w'] + 1e-6, r):
                c, rw = self.w2g(self.px + cy * d - sy * s, self.py + sy * d + cy * s)
                if self.in_grid(c, rw):
                    self.bump_t[rw, c] = t
                    n += 1
        self.n_bumps += 1
        cmd_dist, moved = self.bump.last or (0.0, 0.0)
        self.get_logger().warn('BUMP #%d at (%.2f,%.2f) yaw=%.0f deg: commanded %.2f m, moved %.2f m in %.1f s -> '
                               'obstacle marked ahead, goal %s blacklisted' % (
                                   self.n_bumps, self.px, self.py, math.degrees(self.yaw), cmd_dist, moved,
                                   p['bump_window_s'], None if not self.goal else
                                   (round(self.goal[0], 2), round(self.goal[1], 2))))
        if self.goal:
            self.blacklist.append((self.goal[0], self.goal[1], t + p['blacklist_s']))
        self.evt_pub.publish(C.jmsg({'type': 'bump', 'x': round(self.px, 2), 'y': round(self.py, 2),
                                     'yaw': round(self.yaw, 2), 'cmd_dist': round(cmd_dist, 2),
                                     'moved': round(moved, 2)}))
        self.backoff_until = t + 1.5
        self.last_move = (t, self.px, self.py)
        self.path = []

    def control(self):
        if not self.have_odom or self.mode not in ('EXPLORE', 'GOTO'):
            return
        p = self.p
        t = time.monotonic()
        tw = Twist()
        if self.mode == 'GOTO':
            g = self.goto
            if g is None or self.goto_done_sent:
                self.send(tw)
                return
            if math.hypot(g['x'] - self.px, g['y'] - self.py) < 0.18 or (
                    not self.path and math.hypot(g['x'] - self.px, g['y'] - self.py) < 0.45):
                g['arrived'] = True
            if g['arrived']:
                if g.get('face') is not None:
                    e = C.wrap(float(g['face']) - self.yaw)
                    if abs(e) > 0.07:
                        tw.angular.z = C.clamp(1.5 * e, -0.5, 0.5)
                        if abs(tw.angular.z) < 0.15:
                            tw.angular.z = math.copysign(0.15, e)
                        self.send(tw)
                        return
                self.send(tw)
                self.goto_done_sent = True
                self.evt_pub.publish(C.jmsg({'type': 'goto_done'}))
                return
        if p['bump_enabled'] and self.mode == 'EXPLORE' and self.gate_ok:
            # (a disarmed gate, a latched fault or the joystick means our commands do not reach the dog: no bump then)
            if t >= self.backoff_until and self.bump.update(t, self.px, self.py, self.last_vx):
                self.on_bump(t)
        else:
            self.bump.reset()
        if t < self.backoff_until:
            tw.linear.x = -0.10
            self.send(tw)
            return
        if not self.path:
            self.send(tw)   # stand still (keeps watchdog fed with zeros)
            return
        # stuck detection
        t0, x0, y0 = self.last_move
        if math.hypot(self.px - x0, self.py - y0) > p['stuck_dist']:
            self.last_move = (t, self.px, self.py)
        elif t - t0 > p['stuck_s']:
            self.get_logger().warn('stuck: backing off, blacklisting goal %s' % (self.goal,))
            if self.goal:
                self.blacklist.append((self.goal[0], self.goal[1], t + p['blacklist_s']))
            self.backoff_until = t + 1.5
            self.last_move = (t, self.px, self.py)
            self.path = []
            return
        # lookahead target
        tgt = self.path[-1]
        for pt in self.path:
            if math.hypot(pt[0] - self.px, pt[1] - self.py) >= p['lookahead']:
                tgt = pt
                break
        gdist = math.hypot(self.path[-1][0] - self.px, self.path[-1][1] - self.py)
        if gdist < p['goal_tol']:
            self.path = []
            self.send(tw)
            return
        e = C.wrap(math.atan2(tgt[1] - self.py, tgt[0] - self.px) - self.yaw)
        clr = self.front_clearance()
        if abs(e) > 0.7:
            # keep turning the same way while the target is nearly behind (the sign of e flips there with tiny yaw noise)
            if self.turn_sign == 0.0 or (abs(e) < 2.5 and math.copysign(1.0, e) != self.turn_sign):
                self.turn_sign = math.copysign(1.0, e)
            tw.angular.z = self.turn_sign * p['wz_max']
        else:
            self.turn_sign = 0.0
            sp = p['v_max'] * (1.0 - abs(e) / 0.9)
            if clr < p['front_stop_m']:
                sp = 0.0
            elif clr < p['front_slow_m']:
                sp *= 0.5 + 0.5 * (clr - p['front_stop_m']) / (p['front_slow_m'] - p['front_stop_m'])
            tw.linear.x = max(sp, 0.0)
            tw.angular.z = C.clamp(1.6 * e, -p['wz_max'], p['wz_max'])
        self.send(tw)

    def dump_png(self):
        import cv2
        np.savez('/tmp/grid.npz', L=self.L, seen=self.seen, p=np.array([self.px, self.py, self.yaw]))
        k = int(12.0 / self.p['res'])
        c0, r0 = self.w2g(self.px, self.py)
        c0 = int(C.clamp(c0, k, self.n - k - 1))
        r0 = int(C.clamp(r0, k, self.n - k - 1))
        L = self.L[r0 - k:r0 + k, c0 - k:c0 + k]
        seen = self.seen[r0 - k:r0 + k, c0 - k:c0 + k]
        img = np.full(L.shape + (3,), 128, np.uint8)
        img[seen & (L < 0.5)] = (255, 255, 255)
        infl = ndi.binary_dilation(L > self.p['occ_thresh'], structure=self.disk)
        img[seen & (L < 0.5) & infl] = (190, 190, 255)
        img[L > self.p['occ_thresh']] = (0, 0, 0)
        for pt in self.path:
            cc, rr = self.w2g(*pt)
            cc -= c0 - k
            rr -= r0 - k
            if 0 <= cc < img.shape[1] and 0 <= rr < img.shape[0]:
                img[rr, cc] = (0, 200, 0)
        if self.goal:
            cc, rr = self.w2g(*self.goal)
            cv2.circle(img, (cc - (c0 - k), rr - (r0 - k)), 3, (0, 0, 255), -1)
        cc, rr = self.w2g(self.px, self.py)
        cv2.circle(img, (cc - (c0 - k), rr - (r0 - k)), 3, (255, 0, 0), -1)
        img = cv2.flip(img, 0)          # y up
        cv2.imwrite(self.p['debug_png'], cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    def publish_state(self):
        if self.p['debug_png']:
            try:
                self.dump_png()
            except Exception as e:
                self.get_logger().warn('png: %s' % e, throttle_duration_sec=10)
        self.state_pub.publish(C.jmsg({
            'pose': [round(self.px, 2), round(self.py, 2), round(self.yaw, 2)],
            'cell': list(C.cell_of(self.px, self.py)), 'visited': len(self.visited),
            'frontiers': self.n_front, 'goal': None if not self.goal else
            [round(self.goal[0], 2), round(self.goal[1], 2)],
            'exhausted': self.exhausted, 'path_len': len(self.path), 'kind': self.goal_kind,
            'cells_visited': len(self.visited_c), 'bumps': self.n_bumps,
            'start_guard': not self.guard_done}))


def main():
    rclpy.init()
    n = Explorer()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(n)
    try:
        C.safe_spin(n, ex)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
