"""Local 3-D point map placed with the FAST-LIO pose (replaces the old short-term obstacle memory).

FAST-LIO (/fastlio/odometry) gives the 6-DoF lidar pose in its start frame (lidar_init -> livox_frame).
Its own clouds are thinned to 0.5 m voxels - too coarse for 0.6 m lanes - so the map is built from our
raw /livox/points (sensor frame), placed in the world with that pose, kept on a lio_voxel_m grid within
lio_radius_m of the dog for lio_time_s. Points nearer than lio_blind_m to the sensor (the dog's own legs,
the zero-range blob; FAST-LIO's own `blind` is 0.5 m) are never stored.

The map is handed back in the CURRENT sensor frame, so core.py levels it exactly like a live scan and
the same obstacle rule applies; ground seen earlier fills the blind zone near the dog.

FAST-LIO diverged by kilometres in earlier runs (2026-11-22/23, a copy left running across runs). A pose
step faster than lio_max_speed or longer than lio_jump_m marks the map BAD for the rest of the run:
cleared, never used again (the dog then drives on the live scan only).
"""
import collections
import math

import numpy as np

from rc import geometry as G

_OFF = 1 << 19                    # voxel index offset (+-26 km at 0.05 m)
_B = 1 << 20


def _keys(ijk):
    ijk = np.clip(ijk + _OFF, 0, _B - 1)
    return (ijk[:, 0] * _B + ijk[:, 1]) * _B + ijk[:, 2]


class LioMap:

    def __init__(self, cfg):
        self.c = cfg
        self.ok = True                 # False after a divergence (for the rest of the run)
        self.reason = ''
        self.pose_t = None             # arrival time of the latest pose
        self.R = np.eye(3)
        self.p = np.zeros(3)
        self.hist = collections.deque()
        self.pts = np.zeros((0, 3))
        self.ts = np.zeros(0)
        self.keys = np.zeros(0, dtype=np.int64)

    # ---------------------------------------------------------------- pose
    def on_pose(self, pos, quat, t):
        """Returns a reason string when this pose made the map BAD, else None."""
        if not self.ok:
            return None
        p = np.asarray(pos, dtype=float)
        c = self.c
        bad = None
        if self.pose_t is not None:
            step = float(np.linalg.norm(p - self.p))
            if step > c['lio_jump_m']:
                bad = 'pose jump %.2f m' % step
        self.hist.append((t, p))
        while len(self.hist) > 2 and t - self.hist[1][0] >= 0.3:
            self.hist.popleft()
        t0, p0 = self.hist[0]
        if bad is None and t - t0 >= 0.3:
            v = float(np.linalg.norm(p - p0)) / (t - t0)
            if v > c['lio_max_speed']:
                bad = 'speed %.2f m/s' % v
        if bad is not None:
            self.ok, self.reason = False, bad
            self.clear()
            return bad
        self.R = G.quat_to_rot(*quat)
        self.p = p
        self.pose_t = t
        return None

    def usable(self, t):
        return self.ok and self.pose_t is not None and t - self.pose_t <= self.c['lio_stale_s']

    def status(self, t):
        if not self.ok:
            return 'BAD'
        return 'ok' if self.usable(t) else '-'

    # ---------------------------------------------------------------- map
    def clear(self):
        self.pts = np.zeros((0, 3))
        self.ts = np.zeros(0)
        self.keys = np.zeros(0, dtype=np.int64)

    def add(self, xyz_sensor, t):
        if not self.usable(t):
            return
        c = self.c
        s = np.asarray(xyz_sensor, dtype=float).reshape(-1, 3)
        r = np.sqrt((s * s).sum(axis=1))
        s = s[(r >= c['lio_blind_m']) & (r <= c['max_range_m'])]
        w = s @ self.R.T + self.p
        k = _keys(np.floor(w / c['lio_voxel_m']).astype(np.int64))
        # newest first so np.unique keeps the newest point of every voxel
        allk = np.concatenate([k, self.keys])
        allp = np.vstack([w, self.pts])
        allt = np.concatenate([np.full(k.shape[0], t), self.ts])
        _, idx = np.unique(allk, return_index=True)
        keep = idx[(t - allt[idx] <= c['lio_time_s'])
                   & (np.hypot(allp[idx, 0] - self.p[0], allp[idx, 1] - self.p[1]) <= c['lio_radius_m'])]
        self.keys, self.pts, self.ts = allk[keep], allp[keep], allt[keep]

    def points_sensor(self, t):
        """Map points in the current sensor frame (empty when the pose is BAD / stale)."""
        if not self.usable(t) or self.pts.shape[0] == 0:
            return np.zeros((0, 3))
        return (self.pts - self.p) @ self.R

    def __len__(self):
        return int(self.pts.shape[0])
