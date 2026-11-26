"""Cell navigator (nav_mode: cells, 2026-11-25).

Why: the lane rule (rc/lanes.py + Driver._navigate) could not get round a 90-degree corner of a
0.9 m corridor in the simulator (main and testing, every wall side; a shorter lane_d did not help):
the straight lane closes 1 m before the end wall, the dog turns a cell early onto a diagonal that
clips the inner wall's end, the guard stops it, and the progress check (which reads the guarded
command) never notices.

Idea (micromouse-like, but no fixed lattice, so it does not need a regular grid):
  ALIGN    arena walls are straight and mostly at right angles: the dominant wall direction mod
           90 deg (live scan) gives four AXES. Its slow change in the odom frame is the gyro /
           odometry yaw drift, which is taken out of the dog's own position (frame A).
  MOVE     the dog only walks straight along an axis and only turns on the spot while stopped
           (the 'walk OR turn' rule of 2026-11-23 is kept).
  STOP     at decision points: before a wall (about a cell centre), at the centre of a side
           opening (a junction), and every cells_seg_m of corridor (to re-centre).
  DECIDE   at a stop: which axis directions are open (a body-wide strip free for cells_open_m)?
           Memory = a graph of stop points (nodes) and the walks between them: take an open
           direction that leads somewhere not yet walked (straight first, then the wall side),
           else head along the graph to the nearest node that still has an untried opening, else
           (all explored) the least-used direction. Dead end -> turn round.
  SAFETY   no on-the-spot turn while tilted (walk straight off first, else HOLD); a stall
           watchdog on the command BEFORE the guard; a spin limit. The guard stays the last layer.
Frames: body (x forward, y left), odom (di.pose), A = odom with the estimated yaw drift removed.
"""
import collections
import math

import numpy as np

from rc import geometry as G
from rc.guard import Cmd

Q = math.pi / 2.0


def wrap90(a):
    """Angle wrapped to [-45, 45) deg (axis directions are only defined mod 90 deg)."""
    return (a + math.pi / 4.0) % Q - math.pi / 4.0


def wall_angle(pts, r_max=3.0, dmin=0.15, dmax=0.45, n_max=400):
    """Dominant wall direction mod 90 deg in the frame of `pts`, and a confidence 0..1.
    Mean of exp(4i*theta) over point pairs 0.15-0.45 m apart (0.1 m cells: ~0.3 deg mean error)."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    p = p[np.hypot(p[:, 0], p[:, 1]) < r_max]
    if len(p) < 10:
        return None, 0.0
    if len(p) > n_max:
        p = p[::int(math.ceil(len(p) / n_max))]
    d = p[None, :, :] - p[:, None, :]
    L = np.hypot(d[..., 0], d[..., 1])
    m = np.triu(np.ones(L.shape, dtype=bool), 1) & (L > dmin) & (L < dmax)
    if m.sum() < 10:
        return None, 0.0
    a = np.arctan2(d[..., 1][m], d[..., 0][m])
    w = L[m]
    z = np.sum(w * np.exp(4j * a)) / np.sum(w)
    return float(np.angle(z) / 4.0), float(abs(z))


def to_axis(pts, ang):
    """Body points -> (u along the direction at body angle `ang`, v to its left)."""
    c, s = math.cos(ang), math.sin(ang)
    return c * pts[:, 0] + s * pts[:, 1], -s * pts[:, 0] + c * pts[:, 1]


def free_along(u, v, width, dmax=3.0):
    m = (u > 0.0) & (np.abs(v) < width / 2.0)
    return float(min(dmax, u[m].min())) if m.any() else dmax


def side_gaps(u, v, side, clear_m, u_lo=-0.4, u_hi=2.2, bin_m=0.05, vmin=0.12):
    """Openings beside the axis line on one side (+1 left, -1 right): (u0, u1, closed_lo, closed_hi)
    for each run of u where nothing lies between vmin and clear_m on that side. Obstacle cells are
    0.1 m apart, so occupied bins are widened by one bin."""
    nb = int(round((u_hi - u_lo) / bin_m))
    s = side * v
    m = (s > vmin) & (s < clear_m) & (u >= u_lo) & (u < u_hi)
    occ = np.zeros(nb, dtype=bool)
    if m.any():
        idx = np.clip(((u[m] - u_lo) / bin_m).astype(int), 0, nb - 1)
        occ[idx] = True
    occ = occ | np.concatenate([[False], occ[:-1]]) | np.concatenate([occ[1:], [False]])
    gaps, start = [], None
    for i in range(nb + 1):
        free = i < nb and not occ[i]
        if free and start is None:
            start = i
        elif not free and start is not None:
            gaps.append((u_lo + start * bin_m, u_lo + i * bin_m, start > 0, i < nb))
            start = None
    return gaps


def centre_offset(u, v, near=0.85, one_wall=0.50, u0=-0.2, u1=1.0):
    """Lateral offset (to the left, m) of the corridor centre line from the dog."""
    m = (u > u0) & (u < u1)
    lv, rv = v[m & (v > 0.05)], -v[m & (v < -0.05)]
    dl = float(lv.min()) if lv.size else math.inf
    dr = float(rv.min()) if rv.size else math.inf
    if dl < near and dr < near:
        return (dl - dr) / 2.0
    if dl < one_wall + 0.15:
        return dl - one_wall
    if dr < one_wall + 0.15:
        return one_wall - dr
    return 0.0


class Node:
    __slots__ = ('x', 'y', 'open', 'edge', 'used', 'free', 'id')

    def __init__(self, nid, x, y):
        self.id, self.x, self.y = nid, x, y
        self.open = [None] * 4          # axis k: True open / False closed / None unknown
        self.edge = [None] * 4          # axis k: neighbour node id
        self.used = [0] * 4             # times walked out along k
        self.free = [0.0] * 4           # free distance seen along k (short branches are explored first)


class CellNav:
    """One step() per driver decision (rate_hz). Returns the command; sets driver.mode/phase."""

    def __init__(self, cfg, driver):
        self.c = cfg
        self.drv = driver
        self.side = 1 if cfg['wall_side'] == 'left' else -1
        self.alpha_ref = None           # axis 0 in frame A (fixed after the first estimate)
        self.delta = 0.0                # estimated yaw drift: odom yaw - A yaw
        self.pA = None                  # dog position in frame A
        self.last_xy = None
        self.nodes = []
        self.visited = set()            # 0.25 m cells the dog's path covered (frame A)
        self.tilt_hist = collections.deque()
        self._reset(None)

    # ------------------------------------------------------------------ state
    def _reset(self, t):
        self.state = 'DECIDE'
        self.t_state = t
        self.axis = None                # travel axis index 0..3
        self.target_yaw = None          # A frame
        self.from_node = None
        self.seg_start = None
        self.stall_t = None
        self.stall_xy = None
        self.stalls = 0
        self.shifts = 0                 # sideways shuffles in a row (no room to turn)
        self.go_kind = 'normal'         # GO: normal | shift | tilted
        self.jmem = []
        self.next_check = 0.0
        self.after = 'GO'               # what to do after ROTATE: 'GO' | 'DECIDE' (a look turn)
        self.turn_dir = 1
        self.why = ''                   # reason of the last decision ('new', 'frontier', ...)
        self.spin = 0.0
        self.spin_xy = None
        self.decisions = 0              # decisions since the last cells_spin_m of progress
        self.dec_xy = None

    def _set(self, t, state):
        self.state, self.t_state = state, t
        self.drv.mode = 'TURN' if state == 'ROTATE' else 'CRUISE'
        self.drv.phase = 'cells:' + state

    # ------------------------------------------------------------------ geometry helpers
    def _yawA(self, di):
        return G.wrap(di.pose[2] - self.delta)

    def _axis_yaw(self, k):
        return G.wrap(self.alpha_ref + k * Q)

    def _axis_body(self, di, k):
        return G.wrap(self._axis_yaw(k) - self._yawA(di))

    def _nearest_axis(self, yawA):
        return int(round(G.wrap(yawA - self.alpha_ref) / Q)) % 4

    def _points(self, di):
        parts = [np.asarray(a, dtype=float).reshape(-1, 2) for a in (di.obs, di.mem, getattr(di, 'drops', None))
                 if a is not None]
        p = np.vstack(parts) if parts else np.zeros((0, 2))
        if p.shape[0]:
            b = np.degrees(np.abs(np.arctan2(p[:, 1], p[:, 0])))
            p = p[b <= self.c['cells_fov_deg']]               # the rear is blind on the real dog
        return p

    def _vkey(self, x, y):
        r = self.c['cells_visit_res_m']
        return int(math.floor(x / r)), int(math.floor(y / r))

    def _is_new(self, x, y):
        """No walked path within cells_new_m of (x, y)."""
        r = self.c['cells_visit_res_m']
        n = int(math.ceil(self.c['cells_new_m'] / r))
        i0, j0 = self._vkey(x, y)
        for i in range(i0 - n, i0 + n + 1):
            for j in range(j0 - n, j0 + n + 1):
                if (i, j) in self.visited and math.hypot((i + 0.5) * r - x, (j + 0.5) * r - y) < self.c['cells_new_m']:
                    return False
        return True

    def _target(self, node, k):
        a = self._axis_yaw(k)
        d = self.c['cells_look_m']
        return node.x + d * math.cos(a), node.y + d * math.sin(a)

    # ------------------------------------------------------------------ frame A / memory
    def _track(self, di, pts):
        x, y, yaw = di.pose
        turning = abs(di.last_cmd.wz) > 0.05 or self.state == 'ROTATE'
        a_body, conf = wall_angle(pts) if not turning else (None, 0.0)
        if self.alpha_ref is None:
            if a_body is not None and conf >= self.c['cells_align_conf']:
                self.alpha_ref = wrap90(yaw + a_body)
            else:
                self.alpha_ref = G.wrap(yaw)                    # no walls yet: start heading
            self.delta = 0.0
        elif a_body is not None and conf >= self.c['cells_align_conf']:
            meas = wrap90(yaw + a_body - self.alpha_ref)
            self.delta += self.c['cells_align_gain'] * wrap90(meas - self.delta)
        if self.pA is None:
            self.pA = [0.0, 0.0]
        elif self.last_xy is not None and di.odom_ok:
            dx, dy = x - self.last_xy[0], y - self.last_xy[1]
            if math.hypot(dx, dy) < 0.3:                        # odometry jumps are not walked
                c, s = math.cos(-self.delta), math.sin(-self.delta)
                self.pA[0] += c * dx - s * dy
                self.pA[1] += s * dx + c * dy
        self.last_xy = (x, y)
        self.visited.add(self._vkey(*self.pA))

    def _node_here(self):
        best, bd = None, self.c['cells_merge_m']
        for n in self.nodes:
            d = math.hypot(n.x - self.pA[0], n.y - self.pA[1])
            if d < bd:
                best, bd = n, d
        if best is None:
            best = Node(len(self.nodes), self.pA[0], self.pA[1])
            self.nodes.append(best)
        return best

    def _link(self, a, k, b):
        if a is None or b is None or a is b:
            return
        a.edge[k], a.open[k] = b.id, True
        r = (k + 2) % 4
        b.edge[r], b.open[r] = a.id, True

    def _frontier_dirs(self, n):
        return [k for k in range(4) if n.open[k] and n.edge[k] is None and self._is_new(*self._target(n, k))]

    def _bfs_first_step(self, start):
        """First axis to walk from `start` towards the nearest node with a frontier direction."""
        prev = {start.id: None}
        q = collections.deque([start.id])
        while q:
            nid = q.popleft()
            n = self.nodes[nid]
            if nid != start.id and self._frontier_dirs(n):
                while prev[nid][0] != start.id:
                    nid = prev[nid][0]
                return prev[nid][1]
            for k in range(4):
                m = n.edge[k]
                if m is not None and n.open[k] and m not in prev:
                    prev[m] = (nid, k)
                    q.append(m)
        return None

    # ------------------------------------------------------------------ decisions
    def _observe(self, di, pts, node):
        """Open / closed for the axis directions the lidar can see from here."""
        c = self.c
        for k in range(4):
            rel = abs(self._axis_body(di, k))
            if rel > math.radians(c['cells_see_deg']):
                continue
            u, v = to_axis(pts, self._axis_body(di, k))
            free = free_along(u, v, c['cells_strip_w'])
            node.free[k] = free
            is_open = free >= c['cells_open_m']
            if node.edge[k] is not None and not is_open:
                continue                                         # walked before: keep it open
            if node.open[k] is False and is_open and node.used[k] > 0:
                continue                                         # tried from here and failed: stays closed
            node.open[k] = is_open

    def _choose(self, node, came, heading):
        """came = axis we arrived along (None: start / no move); heading = preferred 'straight'."""
        opts = [k for k in range(4) if node.open[k]]
        back = None if came is None else (came + 2) % 4
        fwd = [k for k in opts if k != back]

        def order(ks):
            pref = []
            if heading is not None:
                pref = [heading, (heading + self.side) % 4, (heading - self.side) % 4, (heading + 2) % 4]
            return sorted(ks, key=lambda k: pref.index(k) if k in pref else 9)
        # new ways: the shorter branch first (finish in the long one, no walk back out of it);
        # within cells_short_tie_m: straight, then the wall side
        new = [k for k in order(fwd) if node.edge[k] is None and self._is_new(*self._target(node, k))]
        tie = self.c['cells_short_tie_m']
        new.sort(key=lambda k: math.floor(min(node.free[k], 3.0) / tie))
        if new:
            return new[0], 'new'
        if back is not None and node.open[back] and node.edge[back] is None and \
                self._is_new(*self._target(node, back)):
            return back, 'new'
        k = self._bfs_first_step(node)
        if k is not None:
            return k, 'frontier'
        if fwd:
            return min(order(fwd), key=lambda kk: node.used[kk]), 'explored'
        if back is not None and node.open[back] is not False:
            return back, 'dead_end'
        return None, 'look'

    # ------------------------------------------------------------------ main step
    def step(self, di, dyaw):
        c = self.c
        t = di.t
        if not self.drv.phase.startswith('cells:'):              # first call, or back from HOLD / stairs
            self._reset(t)
            self._set(t, 'DECIDE')
        pts = self._points(di)
        self._track(di, pts)
        self.tilt_hist.append((t, di.tilt_deg))
        while self.tilt_hist and t - self.tilt_hist[0][0] > c['cells_tilt_window_s']:
            self.tilt_hist.popleft()
        # tilt that matters for an on-the-spot turn: the median of the last cells_tilt_window_s (the dog
        # has just stopped: gait wobble of a few degrees is not a slab under the feet)
        tilt = float(np.median([v for _, v in self.tilt_hist]))
        # spin limit: more than cells_spin_deg of turning without cells_spin_m of progress -> HOLD
        if self.spin_xy is None or math.hypot(self.pA[0] - self.spin_xy[0], self.pA[1] - self.spin_xy[1]) > c['cells_spin_m']:
            self.spin, self.spin_xy = 0.0, tuple(self.pA)
        self.spin += abs(dyaw)
        if self.spin > math.radians(c['cells_spin_deg']):
            self.drv._hold(t, 'spin')
            return Cmd()
        fn = getattr(self, '_st_' + self.state.lower())
        return fn(di, pts, tilt)

    # DECIDE: stand still for cells_settle_s, then look and choose
    def _st_decide(self, di, pts, tilt):
        c, t = self.c, di.t
        if t - self.t_state < c['cells_settle_s']:
            return Cmd()
        node = self._node_here()
        came = self.axis if self.from_node is not None and self.from_node is not node else None
        if self.from_node is not None and self.axis is not None:
            self._link(self.from_node, self.axis, node)
        # decide-loop watchdog: many decisions without progress -> HOLD (never a silent livelock)
        if self.dec_xy is None or math.hypot(self.pA[0] - self.dec_xy[0], self.pA[1] - self.dec_xy[1]) > c['cells_spin_m']:
            self.decisions, self.dec_xy = 0, tuple(self.pA)
        self.decisions += 1
        if self.decisions > c['cells_decide_max']:
            self.drv._hold(t, 'stuck')
            return Cmd()
        self._observe(di, pts, node)
        here = self._nearest_axis(self._yawA(di))
        k, why = self._choose(node, came, came if came is not None else here)
        if k is None:                                           # nothing open in view (start, rear blind)
            u, v = to_axis(pts, 0.0)
            k = (here + (1 if free_along(v, u, 0.5) >= free_along(-v, u, 0.5) else -1)) % 4
            why = 'look'
        self.drv._ev(t, cells=why, node=node.id, axis=int(k), at=[round(float(self.pA[0]), 2), round(float(self.pA[1]), 2)])
        self.from_node = node
        self.axis = k
        self.why = why
        node.used[k] += 1
        if why == 'look':
            self.target_yaw = self._axis_yaw(k)
            self.after = 'DECIDE'
        else:
            # aim at the corridor centre line cells_aim_m ahead along the chosen axis
            u, v = to_axis(pts, self._axis_body(di, k))
            off = max(-0.25, min(0.25, centre_offset(u, v)))
            self.target_yaw = G.wrap(self._axis_yaw(k) + math.atan2(off, c['cells_aim_m']))
            self.after = 'GO'
        return self._start_turn(di, pts, tilt)

    def _start_turn(self, di, pts, tilt):
        """Turn on the spot to target_yaw - only if the dog is level and the body has room to turn."""
        c, t = self.c, di.t
        err = G.wrap(self.target_yaw - self._yawA(di))
        if abs(err) <= math.radians(c['turn_done_deg']):
            return self._after_turn(di)
        if tilt > c['cells_tilt_turn_deg']:
            return self._tilted(di, pts)
        if not self._sweep_ok(pts, err):
            return self._shift(di, pts)
        if self.after != 'SHIFT':
            self.shifts = 0
        self._set(t, 'ROTATE')
        self.turn_dir = 1 if err > 0 else -1
        return self._st_rotate(di, pts, tilt)

    def _after_turn(self, di):
        if self.after == 'DECIDE':
            self._set(di.t, 'DECIDE')
            return Cmd()
        return self._go(di, 'shift' if self.after == 'SHIFT' else 'normal')

    def _sweep_ok(self, pts, turn):
        """Does the body (cells_body_l x cells_body_w + cells_sweep_margin_m), turning on the spot about
        base_link by `turn` rad, stay clear of every obstacle cell in view? (The rear is blind: the dog
        came from there.) A 0.6 m dog sweeps ~0.35 m: in a 0.9 m corridor that leaves ~0.1 m."""
        if pts.shape[0] == 0:
            return True
        c = self.c
        L = c['cells_body_l'] / 2.0 + c['cells_sweep_margin_m']
        W = c['cells_body_w'] / 2.0 + c['cells_sweep_margin_m']
        x, y = pts[:, 0], pts[:, 1]
        near = np.hypot(x, y) < math.hypot(L, W) + 0.01
        if not near.any():
            return True
        x, y = x[near], y[near]
        for a in np.linspace(0.0, turn, max(2, int(abs(turn) / math.radians(5.0)) + 1)):
            ca, sa = math.cos(a), math.sin(a)
            xr, yr = ca * x + sa * y, -sa * x + ca * y
            if np.any((np.abs(xr) <= L) & (np.abs(yr) <= W)):
                return False
        return True

    def _shift(self, di, pts):
        """No room to turn here (off-centre in a narrow corridor): turn cells_shift_deg towards the
        corridor centre, walk cells_shift_m, then decide again. If even that does not fit: HOLD."""
        c, t = self.c, di.t
        here = self._nearest_axis(self._yawA(di))
        u, v = to_axis(pts, self._axis_body(di, here))
        off = centre_offset(u, v)
        if abs(off) >= 0.02 and self.shifts < c['cells_shift_max']:
            tgt = G.wrap(self._axis_yaw(here) + math.copysign(math.radians(c['cells_shift_deg']), off))
            err = G.wrap(tgt - self._yawA(di))
            ub, vb = to_axis(pts, err)
            room = free_along(ub, vb, c['footprint_w']) - c['cells_stop_front_m']
            if room >= c['cells_shift_m'] and self._sweep_ok(pts, err):
                self.shifts += 1
                self.axis, self.target_yaw, self.after = here, tgt, 'SHIFT'
                self.drv._ev(t, cells='shift', off=round(off, 2))
                self._set(t, 'ROTATE')
                self.turn_dir = 1 if err > 0 else -1
                return Cmd()
        self.drv._hold(t, 'tight')
        return Cmd()

    def _tilted(self, di, pts):
        """A turn is wanted while the dog is tilted (slab, bar, strips): never turn on the spot then
        (both falls came from that). Carry on straight along the nearest axis if that way is open,
        else HOLD."""
        c, t = self.c, di.t
        here = self._nearest_axis(self._yawA(di))
        if free_along(pts[:, 0], pts[:, 1], c['footprint_w']) > c['cells_tilt_free_m']:
            self.axis, self.target_yaw, self.why = here, self._yawA(di), 'tilted'
            self.drv._ev(t, cells='tilted_straight', axis=int(here))
            return self._go(di, 'tilted')
        self.drv._hold(t, 'tilted')
        return Cmd()

    def _st_rotate(self, di, pts, tilt):
        c, t = self.c, di.t
        if tilt > c['cells_tilt_turn_deg']:
            self._set(t, 'DECIDE')                              # stop turning; DECIDE goes straight or HOLDs
            return Cmd()
        err = G.wrap(self.target_yaw - self._yawA(di))
        if abs(err) <= math.radians(c['turn_done_deg']):
            return self._after_turn(di)
        if abs(err) < Q:
            self.turn_dir = 1 if err > 0 else -1                # overshoot: turn back, not round
        rate = c['turn_wz'] * abs(err) / math.radians(c['turn_slow_deg'])
        rate = max(min(c['turn_wz_min'], c['turn_wz']), min(c['turn_wz'], rate))
        return Cmd(0.0, self.turn_dir * rate)

    def _go(self, di, kind='normal'):
        t = di.t
        self.go_kind = kind
        self.seg_start = tuple(self.pA)
        self.stall_t, self.stall_xy = None, None
        self.jmem = []                                          # junction centres ahead (along-axis, frame A)
        aim = abs(G.wrap(self.target_yaw - self._axis_yaw(self.axis)))
        # after an aim correction, look again once the centre line should be reached
        self.next_check = self.c['cells_aim_m'] * 0.9 if aim > math.radians(2.0) else self.c['cells_recentre_every_m']
        self._set(t, 'GO')
        return Cmd(self.c['cruise_vx'], 0.0)

    # GO: walk straight along the axis until a stop point
    def _st_go(self, di, pts, tilt):
        c, t = self.c, di.t
        k = self.axis
        travelled = math.hypot(self.pA[0] - self.seg_start[0], self.pA[1] - self.seg_start[1])
        front = free_along(pts[:, 0], pts[:, 1], c['footprint_w'])
        u, v = to_axis(pts, self._axis_body(di, k))
        ax = self._axis_yaw(k)
        s_dog = self.pA[0] * math.cos(ax) + self.pA[1] * math.sin(ax)
        # junctions: a side opening gap_min..gap_max long, seen closed at both ends -> remember its
        # centre (along the axis, frame A) and stop there; near the centre its far edge is out of view
        if self.go_kind == 'normal':
            for side in (1, -1):
                for (u0, u1, lo, hi) in side_gaps(u, v, side, c['cells_side_clear_m']):
                    if not (lo and hi) or not (c['cells_gap_min_m'] <= u1 - u0 <= c['cells_gap_max_m']):
                        continue
                    uc = (u0 + u1) / 2.0
                    if travelled + uc < c['cells_leave_m'] or uc <= -0.05:
                        continue
                    if self.why == 'new':
                        # exploring straight on: an opening into places already walked is not worth a stop
                        px = self.pA[0] + uc * math.cos(ax) - side * c['cells_look_m'] * math.sin(ax)
                        py = self.pA[1] + uc * math.sin(ax) + side * c['cells_look_m'] * math.cos(ax)
                        if not self._is_new(px, py):
                            continue
                    sc = s_dog + uc
                    for i, s0 in enumerate(self.jmem):
                        if abs(s0 - sc) < 0.3:
                            self.jmem[i] = sc                   # nearer view = better estimate
                            break
                    else:
                        self.jmem.append(sc)
        self.jmem = [s for s in self.jmem if s - s_dog > -0.3]
        jc = min((s - s_dog for s in self.jmem), default=math.inf)
        stop_front = front - c['cells_stop_front_m']
        head_err = abs(G.wrap(self._yawA(di) - self.target_yaw))
        why = None
        if stop_front <= 0.0:
            why = 'front'
        elif self.go_kind == 'shift' and travelled >= c['cells_shift_m']:
            why = 'shift'
        elif jc <= 0.03:
            why = 'junction'
        elif self.go_kind == 'normal' and travelled >= self.next_check and \
                not (jc < c['cells_extend_m'] or stop_front < c['cells_extend_m']):
            # re-centre only when needed (each stop costs ~1 s). Before a stop where the dog will turn
            # (wall ahead / junction within cells_pre_centre_m) be stricter: turning needs the room
            off = abs(centre_offset(u, v))
            tol = c['cells_pre_centre_tol_m'] if min(jc, stop_front) < c['cells_pre_centre_m'] else c['cells_recentre_m']
            if off > tol or abs(G.wrap(self._yawA(di) - ax)) > math.radians(c['cells_recentre_deg']):
                why = 'recentre'
            else:
                self.next_check = travelled + c['cells_recentre_every_m']
        if why is None and head_err > math.radians(c['cells_head_err_deg']):
            why = 'heading'
        # stall watchdog: GO sends vx > 0 (before the guard); if the dog does not move -> another way
        if why is None:
            if self.stall_t is None or math.hypot(self.pA[0] - self.stall_xy[0], self.pA[1] - self.stall_xy[1]) > 0.05:
                self.stall_t, self.stall_xy = t, tuple(self.pA)
            elif t - self.stall_t > c['cells_stall_s']:
                why = 'stall'
        if why is None:
            return Cmd(c['cruise_vx'], 0.0)
        failed = self.go_kind == 'normal' and (
            why == 'stall' or (why in ('front', 'heading') and travelled < c['cells_fail_m']))
        if failed and self.from_node is not None:
            # this way does not work from here (e.g. a node merged from a slightly different spot):
            # do not choose it again from this node
            self.from_node.open[k] = False
            self.from_node.edge[k] = None
            self.drv._ev(t, cells_failed=why, node=self.from_node.id, axis=int(k))
        if why == 'stall':
            self.stalls += 1
            if self.stalls >= c['cells_stall_max']:
                self.drv._hold(t, 'stall')
                return Cmd()
        elif travelled > 0.3:
            self.stalls = 0
        self._set(t, 'DECIDE')
        return Cmd()
