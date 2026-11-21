"""Leg-odometry pose with a consistency check (spec section 8).

Input: pose (x, y, yaw) in the odom frame at arrival time t, plus the commanded speed.
Inconsistent if one step jumps more than odom_jump_m, or the measured speed exceeds
odom_speed_factor * commanded + odom_speed_add for > 0.5 s. Once flagged, `ok` stays False until
5 s after the last bad sample. Travel distance only counts consistent steps.
"""
import collections
import math

BAD_HOLD_S = 5.0
FAST_FOR_S = 0.5
SPEED_WINDOW_S = 0.3


class LegOdom:

    def __init__(self, cfg):
        self.c = cfg
        self.pose = None
        self.travelled = 0.0
        self.speed = 0.0
        self.reason = ''
        self.have = False
        self._last_bad = -1e9
        self._fast_since = None
        self._hist = collections.deque()
        self._t = None

    @property
    def ok(self):
        return self.have and (self._t is None or self._t - self._last_bad > BAD_HOLD_S)

    def _bad(self, t, why):
        self._last_bad = t
        self.reason = why

    def update(self, x, y, yaw, t, cmd_speed):
        c = self.c
        self._t = t
        if self.pose is not None:
            step = math.hypot(x - self.pose[0], y - self.pose[1])
            if step > c['odom_jump_m']:
                self._bad(t, 'jump %.2f m' % step)
                self._hist.clear()
            else:
                self.travelled += step
        self.pose = (x, y, yaw)
        self.have = True
        self._hist.append((t, x, y))
        while self._hist and t - self._hist[0][0] > SPEED_WINDOW_S:
            self._hist.popleft()
        t0, x0, y0 = self._hist[0]
        if t - t0 >= 0.2:
            self.speed = math.hypot(x - x0, y - y0) / (t - t0)
            limit = c['odom_speed_factor'] * abs(cmd_speed) + c['odom_speed_add']
            if self.speed > limit:
                if self._fast_since is None:
                    self._fast_since = t
                if t - self._fast_since > FAST_FOR_S:
                    self._bad(t, 'speed %.2f m/s > %.2f' % (self.speed, limit))
            else:
                self._fast_since = None
