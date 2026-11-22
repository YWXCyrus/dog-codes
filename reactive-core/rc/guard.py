"""Guard (spec section 7): the last layer between every command and the dog.

Order: stale feeds -> tilt -> no reverse -> FRONT (raw scan only) -> TURN SWEEP (raw scan +
short-term memory). Never invents motion: it only scales or zeroes the command.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from rc import geometry as G


@dataclass
class Cmd:
    vx: float = 0.0
    wz: float = 0.0


@dataclass
class GuardInput:
    obs_now: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    mem: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    mem_ok: bool = True
    ages: dict = field(default_factory=dict)       # feed name -> seconds since last arrival
    tilt_deg: float = 0.0                          # max(|roll|, |pitch|)
    stairs: bool = False
    roll_deg: float = None                         # separate roll / pitch (stairs mode); None = use tilt_deg
    pitch_deg: float = None
    t: float = None                                # clock (s); None = no tilt push / latch delay


class Guard:

    def __init__(self, cfg, sensor_x=0.18678):
        self.c = cfg
        self.sensor_x = sensor_x
        self.latched = False
        self.rough_since = None      # tilt first above tilt_slow_deg (flat-ground mode)
        self.rough_fwd = False       # ... while walking forward -> push through for tilt_push_s
        self.latch_since = None      # tilt first above tilt_latch_deg

    def _flat_tilt(self, gi, vx):
        """Flat-ground tilt rule. Returns (action, reason): action in None | 'slow' | 'stop' | 'push'.

        Crossing an obstacle in RUG general tilts the body 15-28 deg for a moment (hand-driven
        runs, 2026-11-22). Zeroing the command mid-step left the dog stuck on the wooden frame
        with its legs inside (run 110743: 22 deg -> stop -> sagged to 46 deg). So: if the dog was
        walking forward when the tilt rose, keep the forward command (no turning) for up to
        tilt_push_s; only a tilt that LASTS slows / stops it. The latch needs tilt_latch_s above
        tilt_latch_deg (a fall stays tilted, a step does not); above tilt_latch_hard_deg it is
        immediate."""
        c, tilt, t = self.c, gi.tilt_deg, gi.t
        if tilt > c['tilt_latch_hard_deg']:
            self.latched = True
        if tilt > c['tilt_latch_deg']:
            if t is None:
                self.latched = True
            else:
                if self.latch_since is None:
                    self.latch_since = t
                if t - self.latch_since >= c['tilt_latch_s']:
                    self.latched = True
        else:
            self.latch_since = None
        if self.latched:
            return 'stop', 'tilt_latch'
        if tilt <= c['tilt_slow_deg']:
            self.rough_since = None
            return None, None
        if self.rough_since is None and t is not None:
            self.rough_since = t
            self.rough_fwd = vx > 0.0
        if (t is not None and self.rough_fwd and vx > 0.0
                and t - self.rough_since <= c['tilt_push_s']):
            return 'push', 'tilt_push'
        if tilt > c['tilt_stop_deg'] or tilt > c['tilt_latch_deg']:
            return 'stop', 'tilt_stop'
        return 'slow', 'tilt_slow'

    # ---- helpers ----
    def _front_dist(self, obs):
        if obs.shape[0] == 0:
            return math.inf
        half = self.c['footprint_w'] / 2.0 + self.c['guard_corridor_margin_m']
        m = (obs[:, 0] > self.sensor_x) & (np.abs(obs[:, 1]) < half)
        return float(obs[m, 0].min() - self.sensor_x) if m.any() else math.inf

    @staticmethod
    def _rect_dist(pts, cx, cy, a, L, W):
        """Distance of each point to a rectangle (0 inside)."""
        ca, sa = math.cos(a), math.sin(a)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u = np.abs(ca * dx + sa * dy) - L / 2.0
        v = np.abs(-sa * dx + ca * dy) - W / 2.0
        return np.hypot(np.maximum(u, 0.0), np.maximum(v, 0.0))

    def _sweep_hit(self, pts, vx, wz):
        """Does the motion of the next sweep_dt_s bring the footprint CLOSER than sweep_margin_m to
        any point? Points the motion moves away from never block (so the dog can always turn away
        from a wall it is close to); points inside the footprint itself are ignored (self hits)."""
        c = self.c
        if pts.shape[0] == 0 or wz == 0.0:
            return False
        L, W, off, margin = c['footprint_l'], c['footprint_w'], c['base_offset_x'], c['sweep_margin_m']
        d0 = self._rect_dist(pts, off, 0.0, 0.0, L, W)
        pts, d0 = pts[d0 > 0.0], d0[d0 > 0.0]
        if pts.shape[0] == 0:
            return False
        total = wz * c['sweep_dt_s']
        n = max(2, int(math.ceil(abs(math.degrees(total)) / 5.0)) + 1)
        for k in range(1, n):
            f = k / (n - 1)
            a = total * f
            shift = max(0.0, vx) * c['sweep_dt_s'] * f
            d = self._rect_dist(pts, (off + shift) * math.cos(a), (off + shift) * math.sin(a), a, L, W)
            if np.any((d < margin) & (d < d0 - 1e-3)):
                return True
        return False

    def _sweep_pts(self, gi):
        parts = [np.asarray(gi.obs_now, dtype=float).reshape(-1, 2)]
        if gi.mem_ok:
            parts.append(np.asarray(gi.mem, dtype=float).reshape(-1, 2))
        return np.vstack(parts)

    def blocked_both(self, gi):
        pts = self._sweep_pts(gi)
        w = self.c['turn_wz']
        return self._sweep_hit(pts, 0.0, w) and self._sweep_hit(pts, 0.0, -w)

    # ---- main ----
    def filter(self, cmd, gi):
        c = self.c
        reasons = []
        vx, wz = float(cmd.vx), float(cmd.wz)

        stale = [k for k, a in sorted(gi.ages.items()) if not (a <= c['stale_s'])]
        if stale:
            return Cmd(0.0, 0.0), ['stale:%s' % k for k in stale]

        roll = gi.tilt_deg if gi.roll_deg is None else abs(gi.roll_deg)
        pitch = gi.tilt_deg if gi.pitch_deg is None else abs(gi.pitch_deg)
        if gi.stairs:
            # on stairs the body pitches with the slope (0.15/0.30 m = 27 deg); roll keeps normal limits
            self.rough_since = self.latch_since = None
            latch = roll > c['tilt_latch_deg'] or pitch > c['tilt_latch_stairs_deg']
            stop = roll > c['tilt_stop_deg'] or pitch > c['tilt_stop_stairs_deg']
            if latch:
                self.latched = True
            if self.latched:
                return Cmd(0.0, 0.0), ['tilt_latch']
            if stop:
                return Cmd(0.0, 0.0), ['tilt_stop']
        else:
            action, why = self._flat_tilt(gi, vx)
            if action == 'stop':
                return Cmd(0.0, 0.0), [why]
            if action == 'slow':
                vx *= 0.5
                reasons.append(why)
            elif action == 'push':
                wz = 0.0                     # straight over the obstacle, full forward command
                reasons.append(why)

        if vx < 0.0:
            vx = 0.0
            reasons.append('no_reverse')

        if vx > 0.0:
            d = self._front_dist(np.asarray(gi.obs_now, dtype=float).reshape(-1, 2))
            if d < c['stop_m']:
                vx = 0.0
                reasons.append('front_stop')
            elif d < c['slow_m']:
                vx *= (d - c['stop_m']) / (c['slow_m'] - c['stop_m'])
                reasons.append('front_slow')

        if wz != 0.0:
            if not gi.mem_ok and abs(wz) > c['wz_no_memory']:
                wz = math.copysign(c['wz_no_memory'], wz)
                reasons.append('no_memory')
            if self._sweep_hit(self._sweep_pts(gi), vx, wz):
                reasons.append('sweep_block_left' if wz > 0 else 'sweep_block_right')
                wz = 0.0
        return Cmd(vx, wz), reasons
