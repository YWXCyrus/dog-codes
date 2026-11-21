"""Lane rule (spec section 6): a heading is walkable if a width x depth rectangle starting at
base_link along that heading contains no obstacle / drop cell."""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Lane:
    heading: float
    free: bool
    free_dist: float


def _local(pts, heading):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    c, s = math.cos(heading), math.sin(heading)
    return c * pts[:, 0] + s * pts[:, 1], -s * pts[:, 0] + c * pts[:, 1]


def free_distance(pts, heading, width, dmax=3.0):
    if len(pts) == 0:
        return dmax
    u, v = _local(pts, heading)
    m = (u > 0.0) & (np.abs(v) < width / 2.0)
    return float(min(dmax, u[m].min())) if m.any() else dmax


def lane_free(pts, heading, width, depth):
    return free_distance(pts, heading, width, dmax=depth + 1.0) >= depth


def candidate_headings(cfg):
    fov, step = cfg['lane_fov_deg'], cfg['lane_step_deg']
    return np.radians(np.arange(-fov, fov + 1e-6, step))


def evaluate(obs_xy, drops_xy, cfg, width=None, extra=None):
    """Lanes for all candidate headings. `extra` = more blocking points (e.g. short-term memory)."""
    width = cfg['lane_w'] if width is None else width
    parts = [np.asarray(a, dtype=float).reshape(-1, 2) for a in (obs_xy, drops_xy, extra) if a is not None]
    pts = np.vstack(parts) if parts else np.zeros((0, 2))
    out = []
    for h in candidate_headings(cfg):
        d = free_distance(pts, h, width)
        out.append(Lane(float(h), d >= cfg['lane_d'], d))
    return out
