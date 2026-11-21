"""Stairs detection from the levelled cloud (spec section 9.3) - "height levels" method.

Per candidate heading, the points in the lane are histogrammed by height. Horizontal surfaces
(floor, treads, platform) collect many points at one height -> peaks = LEVELS; risers and walls
spread over all heights and give no level. Each level has a distance range (u10..u90) along the
lane. This works with ring-structured lidar data (only 1-2 rings per tread).

  up      >= stair_min_rises consecutive levels, each one riser (stair_rise_min..max) higher and
          clearly FURTHER away than the previous, ranges not interleaved; rises similar (<= 0.05 m).
          A rise from the dog's own floor counts only if that floor is visible ahead.
  unsure  one such counted rise, a tread >= 0.25 m, and nothing seen beyond it
  down    a level one riser or more BELOW the stance floor, further away than the visible stance
  none    anything else (strip boards interleave with the floor; walls give no level)

StairDetector: headings -20..+20 deg vote; heading refined by an edge-line fit; persistence for
a confident 'up'; nothing reported inside the start-exclusion radius.
"""
import collections
import math
from dataclasses import dataclass

import numpy as np

HEADINGS_DEG = range(-20, 21, 5)
HIST_BIN = 0.02
LEVEL_HALF = 0.03            # points within this of a peak belong to the level
MIN_LEVEL_SPAN = 0.06        # a level must extend at least this far along the lane (walls do not)
STANCE_TOL = 0.04


@dataclass
class StairsResult:
    kind: str = 'none'
    heading: float = 0.0
    edge_x: float = math.inf
    confident: bool = False
    rises: int = 0


def lane_points(p_level, heading, width, floor_z, x_min=0.3, x_max=3.0):
    """(u, v, z_rel) of the points in the lane rotated to the heading; z relative to the stance floor."""
    p = np.asarray(p_level, dtype=float).reshape(-1, 3)
    if p.shape[0] == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    c, s = math.cos(heading), math.sin(heading)
    u = c * p[:, 0] + s * p[:, 1]
    v = -s * p[:, 0] + c * p[:, 1]
    z = p[:, 2] - floor_z
    m = (np.abs(v) < width / 2.0) & (u >= x_min) & (u < x_max) & (z > -0.65) & (z < 0.8)
    return u[m], v[m], z[m]


def profile(p_level, heading, width, bin_m, floor_z, x_min=0.3, x_max=3.0):
    """Median height per distance bin (absolute levelled z). Used by tools/measure.py."""
    u, _, z = lane_points(p_level, heading, width, floor_z, x_min, x_max)
    edges = np.arange(x_min, x_max + 1e-9, bin_m)
    centres = (edges[:-1] + edges[1:]) / 2.0
    heights = np.full(centres.shape, np.nan)
    if u.size:
        idx = np.floor((u - x_min) / bin_m).astype(int)
        for b in np.unique(idx):
            sel = z[idx == b]
            if sel.size >= 3:
                heights[b] = float(np.median(sel)) + floor_z
    return centres, heights


def levels(u, z):
    """Height levels [(z, n, u10, u50, u90)] sorted by height."""
    n = z.size
    if n < 20:
        return []
    edges = np.arange(-0.65, 0.8 + 1e-9, HIST_BIN)
    cnt, _ = np.histogram(z, edges)
    sm = np.convolve(cnt, [1, 2, 1], mode='same') / 4.0
    thr = max(8.0, 0.03 * n)
    peaks = [i for i in range(1, len(sm) - 1) if sm[i] >= thr and sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]]
    out = []
    for i in sorted(peaks, key=lambda i: -sm[i]):
        zc = (edges[i] + edges[i + 1]) / 2.0
        if any(abs(zc - o[0]) < 0.06 for o in out):
            continue
        sel = np.abs(z - zc) < LEVEL_HALF
        if sel.sum() < 8:
            continue
        us = u[sel]
        u10, u50, u90 = np.percentile(us, [10, 50, 90])
        if u90 - u10 < MIN_LEVEL_SPAN:
            continue
        out.append((float(np.median(z[sel])), int(sel.sum()), float(u10), float(u50), float(u90)))
    return sorted(out, key=lambda l: l[0])


def classify(lv, u, z, cfg):
    rmin, rmax = cfg['stair_rise_min'], cfg['stair_rise_max']
    res = {'kind': 'none', 'rises': 0, 'edge_x': math.inf}
    if not lv:
        return res
    stance = [l for l in lv if abs(l[0]) < STANCE_TOL]
    stance = stance[0] if stance else None
    # ---- down: a level one riser (or more) below the stance, beyond the visible stance ----
    below = [l for l in lv if -(rmax + 0.2) <= l[0] <= -rmin]
    if below:
        low = max(below, key=lambda l: l[0])          # the nearest level below
        if stance is None or low[3] > stance[3] + 0.1:
            edge = (stance[4] + low[2]) / 2.0 if stance is not None else max(0.3, low[2] - 0.3)
            res.update(kind='down', edge_x=edge)
            return res
    # ---- up: chain of levels, each one riser higher and further away ----
    prev = stance if stance is not None else (0.0, 0, 0.0, 0.0, 0.0)
    chain, counted = [], []
    for l in [l for l in lv if l[0] >= rmin - 0.02]:
        d = l[0] - prev[0]
        if d > rmax:
            break
        if d < rmin - 0.02:
            continue
        if l[3] < prev[3] + 0.1:
            return res                                  # higher but not further away: not stairs
        if prev is stance or chain:
            overlap = min(prev[4], l[4]) - max(prev[2], l[2])
            if overlap > 0.5 * min(prev[4] - prev[2], l[4] - l[2]):
                return res                              # interleaved ranges: strip boards
        if prev is stance or chain:                     # rise between two VISIBLE levels
            counted.append(d)
        chain.append(l)
        prev = l
    if not chain:
        return res
    if len(counted) >= cfg['stair_min_rises'] and max(counted) - min(counted) <= 0.05:
        res.update(kind='up', rises=len(counted), edge_x=chain[0][2] if stance is None else
                   (stance[4] + chain[0][2]) / 2.0)
        return res
    if len(counted) == 1 and len(chain) == 1 and chain[0][4] - chain[0][2] >= 0.25:
        if not np.any(u > chain[0][4] + 0.15):          # nothing seen beyond the single step
            res.update(kind='unsure', rises=1, edge_x=(stance[4] + chain[0][2]) / 2.0)
    return res


def analyse_lane(p_level, heading, floor_z, cfg):
    u, _, z = lane_points(p_level, heading, cfg['lane_w'], floor_z)
    return classify(levels(u, z), u, z, cfg)


class StairDetector:

    def __init__(self, cfg):
        self.c = cfg
        self.hist = collections.deque(maxlen=int(cfg['stair_persist']) + 2)

    def reset(self):
        self.hist.clear()

    def _edge_angle(self, p_level, heading, edge_x, floor_z, win=0.3):
        """Correction to face the first riser squarely. The lane is cut into 0.1 m lateral slices;
        in each, the nearest point of the first step (riser face / first tread) is found; a line
        u = a + k*v is fitted through them -> correction = -atan(k). Needs >= 4 slices."""
        c = self.c
        p = np.asarray(p_level, dtype=float).reshape(-1, 3)
        cs, sn = math.cos(heading), math.sin(heading)
        u = cs * p[:, 0] + sn * p[:, 1]
        v = -sn * p[:, 0] + cs * p[:, 1]
        z = p[:, 2] - floor_z
        rise = (z > c['stair_rise_min'] / 2.0) & (z < c['stair_rise_max'] + 0.05) & (np.abs(u - edge_x) < win)
        vs, us = [], []
        for vc in np.arange(-0.3, 0.301, 0.1):
            m = rise & (np.abs(v - vc) < 0.05)
            if m.sum() >= 2:
                vs.append(vc)
                us.append(float(np.percentile(u[m], 10)))
        if len(vs) < 4:
            return 0.0
        k = float(np.polyfit(np.array(vs), np.array(us), 1)[0])
        corr = -math.atan(k)
        return max(-math.radians(20), min(math.radians(20), corr))

    def update(self, p_level, floor_z, t, dist_from_start):
        c = self.c
        best = StairsResult()
        if dist_from_start >= c['stair_start_excl_m']:
            found = []
            for hd in HEADINGS_DEG:
                h = math.radians(hd)
                a = analyse_lane(p_level, h, floor_z, c)
                if a['kind'] != 'none':
                    found.append((a, h))
            order = {'up': 0, 'down': 1, 'unsure': 2}
            if found:
                found.sort(key=lambda fh: (order[fh[0]['kind']], fh[0]['edge_x'], abs(fh[1])))
                a, h = found[0]
                if a['kind'] == 'up':
                    # all headings that see the stairs vote; then refine with the edge line fit
                    ups = [fh for fh in found if fh[0]['kind'] == 'up']
                    h = float(np.mean([fh[1] for fh in ups]))
                    edge = float(np.median([fh[0]['edge_x'] for fh in ups]))
                    h = h + self._edge_angle(p_level, h, edge, floor_z)
                    a = dict(a, edge_x=edge)
                best = StairsResult(a['kind'], h, a['edge_x'], False, a['rises'])
        self.hist.append(best.kind)
        if best.kind == 'up':
            best.confident = sum(1 for k in self.hist if k == 'up') >= c['stair_persist']
        return best
