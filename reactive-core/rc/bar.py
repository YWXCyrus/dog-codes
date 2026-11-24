"""Low bar / cylinder across the lane (2026-11-25, design B of session 10).

Seen on the levelled lidar points (body frame, floor at floor_z). A bar is an object

  * bar_h_min .. bar_h_max (0.10 .. 0.32 m) above the floor,
  * at most bar_depth_max (0.35 m) deep along the walking direction,
  * across most of the lane (>= bar_cover of the 0.1 m slices over +-bar_half_w),
  * with NOTHING higher within bar_clear_m behind its front edge (a wall, the stairs and the
    0.5 m deep wooden slab of run 181509 all fail one of these tests),
  * 0.4 .. bar_range_m ahead, and seen in bar_persist of the last bar_hist scans.

The front edge also gives the angle of the bar's line (x = a + k*y -> square-up correction -atan(k)),
so the dog can face it squarely before crossing. Thresholds are guesses until a real cylinder is
scanned (tools/measure.py).
"""
import collections
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BarResult:
    found: bool = False
    confident: bool = False
    dist: float = math.inf        # front edge, body frame x
    depth: float = 0.0
    height: float = 0.0
    corr: float = 0.0             # yaw correction to face the bar (rad, + = left)
    cover: float = 0.0
    too_deep: bool = False        # a wide, bar-high object that is deeper than bar_depth_max (a slab / platform)


def analyse(p_level, floor_z, cfg):
    p = np.asarray(p_level, dtype=float).reshape(-1, 3)
    if p.shape[0] == 0:
        return BarResult()
    h = p[:, 2] - floor_z
    hw = cfg['bar_half_w']
    m = (p[:, 0] > 0.4) & (p[:, 0] < cfg['bar_range_m'] + 1.0) & (np.abs(p[:, 1]) < hw) & (h > 0.04)
    if m.sum() < 8:
        return BarResult()
    x, y, hh = p[m, 0], p[m, 1], h[m]
    band = (hh >= cfg['bar_h_min']) & (hh <= cfg['bar_h_max'])
    high = (hh > cfg['bar_h_max'] + 0.03) & (hh < cfg['obs_h_max'])
    fronts, ys, tops, depths = [], [], [], []
    deep_n = 0
    edges = np.arange(-hw, hw + 1e-9, 0.1)
    nslices = len(edges) - 1
    for i in range(nslices):
        s = (y >= edges[i]) & (y < edges[i + 1])
        sb = s & band
        if sb.sum() < 3:
            continue
        xr = float(x[sb].min())
        if xr > cfg['bar_range_m']:
            continue
        depth = float(x[sb].max() - xr)
        if depth > cfg['bar_depth_max']:
            deep_n += 1
            continue
        if np.any(s & high & (x > xr - 0.1) & (x < xr + cfg['bar_clear_m'])):
            continue
        fronts.append(xr)
        ys.append(0.5 * (edges[i] + edges[i + 1]))
        tops.append(float(hh[sb].max()))
        depths.append(depth)
    cover = len(fronts) / float(nslices)
    too_deep = deep_n / float(nslices) >= 0.4
    if cover < cfg['bar_cover']:
        return BarResult(cover=cover, too_deep=too_deep)
    corr = 0.0
    if len(fronts) >= 4:
        k = float(np.polyfit(np.array(ys), np.array(fronts), 1)[0])
        corr = max(-math.radians(20.0), min(math.radians(20.0), -math.atan(k)))
    return BarResult(True, False, float(np.median(fronts)), float(np.median(depths)),
                     float(np.median(tops)), corr, cover, too_deep)


class BarDetector:

    def __init__(self, cfg):
        self.c = cfg
        self.hist = collections.deque(maxlen=int(cfg['bar_hist']))
        # a slab seen deep in ONE scan stays "deep": from a moving dog the far part of its top is
        # sampled sparsely and a single scan can show only 0.3 m of a 0.6 m slab (sim, 2026-11-25)
        self.deep = collections.deque(maxlen=int(cfg['bar_deep_memory']))

    def reset(self):
        self.hist.clear()
        self.deep.clear()

    def update(self, p_level, floor_z):
        r = analyse(p_level, floor_z, self.c)
        self.hist.append(r.found)
        self.deep.append(r.too_deep)
        if r.found:
            r.confident = sum(self.hist) >= self.c['bar_persist'] and not any(self.deep)
        return r
