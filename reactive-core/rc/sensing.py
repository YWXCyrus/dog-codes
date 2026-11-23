"""Raw lidar -> levelled body-frame points -> OBSTACLE and DROP cells (spec section 5).

Levelled body frame: origin base_link, x forward, y left, z up along gravity, yaw not removed
(it is the dog's own heading). The stance floor sits at floor_z (= -body height, ~ -0.32 m).
"""
import collections
from dataclasses import dataclass, field

import numpy as np

from rc import geometry as G

EMPTY2 = np.zeros((0, 2))


@dataclass
class Mount:
    xyz: np.ndarray          # sensor position in base_link
    pitch: float             # sensor pitch (positive = looking down), radians


@dataclass
class Scan:
    pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    obs_now: np.ndarray = field(default_factory=lambda: EMPTY2.copy())   # this scan only
    obs: np.ndarray = field(default_factory=lambda: EMPTY2.copy())       # persistent cells
    drops: np.ndarray = field(default_factory=lambda: EMPTY2.copy())     # persistent drop patches
    ground_now: np.ndarray = field(default_factory=lambda: EMPTY2.copy())  # this scan: walkable returns
    high_now: np.ndarray = field(default_factory=lambda: EMPTY2.copy())    # this scan: obstacle-height returns
    floor_z: float = -0.32
    t: float = 0.0


def filter_sensor(xyz, near_m, max_range_m):
    """Drop non-finite points, the zero-range blob (< near_m from the sensor) and far returns."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    ok = np.isfinite(xyz).all(axis=1)
    xyz = xyz[ok]
    r = np.sqrt((xyz * xyz).sum(axis=1))
    return xyz[(r >= near_m) & (r <= max_range_m)]


def level_points(xyz_sensor, mount, roll, pitch):
    """Sensor frame -> base_link (fixed mount) -> levelled by the IMU roll/pitch."""
    p_body = xyz_sensor @ G.rot_y(mount.pitch).T + mount.xyz
    R = G.rot_y(pitch) @ G.rot_x(roll)
    return p_body @ R.T


def calibrate_floor(p_level):
    """Stance floor height from the floor seen 1.4-3.0 m ahead; None if not plausible."""
    if p_level.shape[0] == 0:
        return None
    m = (p_level[:, 0] >= 1.4) & (p_level[:, 0] <= 3.0) & (np.abs(p_level[:, 1]) <= 1.0)
    z = p_level[m, 2]
    if z.size < 150:
        return None
    f = float(np.percentile(z, 10))
    return f if -0.50 < f < -0.15 else None


def _cells(xy, size):
    return np.floor(xy / size).astype(np.int64)


def _keys(ij):
    return (ij[:, 0] + 100000) * 200003 + (ij[:, 1] + 100000)


def _unkey(k):
    return np.stack([k // 200003 - 100000, k % 200003 - 100000], axis=1)


class ObstacleDetector:

    def __init__(self, cfg):
        self.c = cfg
        self.hist_obs = collections.deque(maxlen=int(cfg['persist_n']))
        self.hist_drop = collections.deque(maxlen=int(cfg['persist_n']))

    def reset(self):
        self.hist_obs.clear()
        self.hist_drop.clear()

    def _patch_ground(self, p, floor_z, trust):
        """Per-point local ground (with the stance-floor fallback) + drop patch keys."""
        c = self.c
        n = p.shape[0]
        pij = _cells(p[:, :2], c['patch_m'])
        key = _keys(pij)
        order = np.lexsort((p[:, 2], key))
        ks = key[order]
        zs = p[order, 2]
        starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
        cnt = np.diff(np.r_[starts, n])
        idx = starts + np.floor((c['ground_pct'] / 100.0) * (cnt - 1)).astype(np.int64)
        g_raw = zs[idx]
        trusted = (cnt >= 3) & (np.abs(g_raw - floor_z) <= trust)
        g = np.where(trusted, g_raw, floor_z)
        ground = np.empty(n)
        ground[order] = np.repeat(g, cnt)
        drop_keys = ks[starts][(cnt >= 10) & (g_raw < floor_z - c['step_limit'])]
        return ground, drop_keys

    def update(self, p_level, floor_z, t, stairs_mode=False):
        c = self.c
        p = np.asarray(p_level, dtype=float).reshape(-1, 3)
        scan = Scan(pts=p, floor_z=floor_z, t=t)
        obs_keys = np.zeros((0,), np.int64)
        drop_keys = np.zeros((0,), np.int64)
        if p.shape[0] > 0:
            trust = c['stairs_trust_m'] if stairs_mode else c['step_limit']
            ground, dk = self._patch_ground(p, floor_z, trust)
            h = p[:, 2] - ground
            m = (h > c['step_limit']) & (h < c['obs_h_max'])
            scan.ground_now = p[np.abs(h) <= c['step_limit'], :2]
            scan.high_now = p[m, :2]
            if m.any():
                k, cnt = np.unique(_keys(_cells(p[m, :2], c['obs_cell_m'])), return_counts=True)
                obs_keys = k[cnt >= c['n_min']]
            if not stairs_mode:
                drop_keys = dk
        self.hist_obs.append(set(obs_keys.tolist()))
        self.hist_drop.append(set(drop_keys.tolist()))
        scan.obs_now = self._centres(obs_keys, c['obs_cell_m'])
        scan.obs = self._centres(self._persistent(self.hist_obs), c['obs_cell_m'])
        scan.drops = self._centres(self._persistent(self.hist_drop), c['patch_m'])
        return scan

    def _persistent(self, hist):
        cnt = collections.Counter()
        for s in hist:
            cnt.update(s)
        k = int(self.c['persist_k'])
        return np.array(sorted(key for key, v in cnt.items() if v >= k), dtype=np.int64)

    @staticmethod
    def _centres(keys, size):
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size == 0:
            return EMPTY2.copy()
        return (_unkey(keys) + 0.5) * size
