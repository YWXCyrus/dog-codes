"""Curtain (2026-11-25): a hanging cloth that looks like a ~1 m tall wall to the lidar.

The lidar cannot tell a curtain from a wall, so the decision needs the camera. best.pt (YOLO26n,
classes 0 curtain / 1 human / 2 stairs) runs in a separate node (rc/vision_node.py) and publishes
detections with a bearing; this module only holds them and checks them against the lidar:

  * the curtain box must be seen in >= curtain_persist detections within curtain_window_s,
    confidence >= curtain_conf, bearing within +-curtain_cone_deg;
  * the lidar must show an obstacle face in that bearing, curtain_min_m .. curtain_max_m ahead and
    at least curtain_min_w wide (so a camera false positive in open space is ignored).

The face line gives the square-up angle (x = a + k*y -> -atan(k)). The push itself is in driver.py.
"""
import collections
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CurtainResult:
    found: bool = False
    dist: float = math.inf          # nearest face point along the bearing, body frame
    bearing: float = 0.0            # rad, + = left
    width: float = 0.0
    corr: float = 0.0               # yaw correction from the face line (rad)
    conf: float = 0.0


class Vision:
    """Holds the latest detections: [{'cls': 'curtain', 'conf': 0.9, 'bearing': 0.1}, ...]."""

    def __init__(self, cfg):
        self.c = cfg
        self.hist = collections.deque()          # (t, conf, bearing)
        self.last_t = None
        self.counts = {}

    def on_detections(self, dets, t):
        self.last_t = t
        for d in dets or []:
            cls = d.get('cls')
            self.counts[cls] = self.counts.get(cls, 0) + 1
            if cls == 'curtain' and d.get('conf', 0.0) >= self.c['curtain_conf']:
                self.hist.append((t, float(d['conf']), float(d['bearing'])))
        self._trim(t)

    def _trim(self, t):
        w = self.c['curtain_window_s']
        while self.hist and t - self.hist[0][0] > w:
            self.hist.popleft()

    def curtain_bearing(self, t):
        """(conf, bearing) of the curtain if seen persistently inside the cone, else None."""
        self._trim(t)
        cone = math.radians(self.c['curtain_cone_deg'])
        sel = [(c, b) for _, c, b in self.hist if abs(b) <= cone]
        if len(sel) < self.c['curtain_persist']:
            return None
        return float(np.mean([c for c, _ in sel])), float(np.median([b for _, b in sel]))


def analyse(vision, obs_now, t, cfg):
    cb = vision.curtain_bearing(t)
    if cb is None:
        return CurtainResult()
    conf, bearing = cb
    o = np.asarray(obs_now, dtype=float).reshape(-1, 2)
    if o.shape[0] == 0:
        return CurtainResult(conf=conf, bearing=bearing)
    r = np.hypot(o[:, 0], o[:, 1])
    a = np.arctan2(o[:, 1], o[:, 0])
    cone = math.radians(cfg['curtain_cone_deg'])
    m = (np.abs(a - bearing) <= cone) & (o[:, 0] > 0.0) & (r > cfg['curtain_min_m']) & (r < cfg['curtain_max_m'])
    if m.sum() < 4:
        return CurtainResult(conf=conf, bearing=bearing)
    f = o[m]
    near = f[:, 0] <= f[:, 0].min() + 0.35               # the face, not whatever is behind it
    face = f[near]
    width = float(np.ptp(face[:, 1]))
    if width < cfg['curtain_min_w']:
        return CurtainResult(conf=conf, bearing=bearing, width=width)
    corr = 0.0
    if face.shape[0] >= 6 and width > 0.3:
        k = float(np.polyfit(face[:, 1], face[:, 0], 1)[0])
        corr = max(-math.radians(20.0), min(math.radians(20.0), -math.atan(k)))
    return CurtainResult(True, float(np.median(face[:, 0])), bearing, width, corr, conf)
