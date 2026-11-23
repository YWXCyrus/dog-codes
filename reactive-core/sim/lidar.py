"""Fake Livox: rays from the sensor (real mount geometry) against the floor and the boxes.
Returns points in the SENSOR frame, like /livox/points, including a zero-range blob."""
import math

import numpy as np

from rc import geometry as G


class Lidar:

    def __init__(self, mount_xyz=(0.18678, 0.0, 0.12498), mount_pitch_deg=15.0,
                 elev_deg=None, az_step_deg=1.0, max_range=10.0, noise=0.01, blob=2000, seed=0,
                 elev_step_deg=1.0, cloth_fold_m=0.03, cloth_period_m=0.22, cloth_flutter_m=0.02,
                 cloth_see_through=0.15):
        # cloth (boxes named in world.CLOTH): folds (sine along the bar), flutter (the folds drift
        # and wobble from scan to scan), a share of rays passing through the fabric
        self.cloth = (cloth_fold_m, cloth_period_m, cloth_flutter_m, cloth_see_through)
        self.phase = 0.0
        self.cloth_rng = np.random.default_rng(seed + 7919)   # own stream: worlds without cloth stay
                                                              # bit-identical to before cloth existed
        # ~21k rays per scan: the real /livox/points has ~20k points per 0.1 s frame (session 1)
        self.mount_xyz = np.array(mount_xyz, dtype=float)
        self.mount_pitch = math.radians(mount_pitch_deg)
        el = np.radians(np.arange(-7.0, 52.1, elev_step_deg) if elev_deg is None else np.asarray(elev_deg))
        az = np.radians(np.arange(0.0, 360.0, az_step_deg))
        E, A = np.meshgrid(el, az)
        E, A = E.ravel(), A.ravel()
        self.dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=1)
        self.max_range = max_range
        self.noise = noise
        self.blob = blob
        self.rng = np.random.default_rng(seed)

    def scan(self, world, x, y, z_base, yaw, roll=0.0, pitch=0.0):
        """z_base = height of base_link above the world floor."""
        R_body = G.rot_z(yaw) @ G.rot_y(pitch) @ G.rot_x(roll)
        R_s = R_body @ G.rot_y(self.mount_pitch)
        o = np.array([x, y, z_base]) + R_body @ self.mount_xyz
        d = self.dirs @ R_s.T
        n = d.shape[0]
        t = np.full(n, np.inf)
        down = d[:, 2] < -1e-9
        t[down] = -o[2] / d[down, 2]
        for b in world.boxes:
            c, s = math.cos(b.yaw), math.sin(b.yaw)
            ox, oy = o[0] - b.cx, o[1] - b.cy
            lo_ = np.array([c * ox + s * oy, -s * ox + c * oy, o[2]])
            ld = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
            lo = np.array([-b.sx / 2, -b.sy / 2, b.z0])
            hi = np.array([b.sx / 2, b.sy / 2, b.z1])
            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = (lo - lo_) / ld
                t2 = (hi - lo_) / ld
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0.0)
            if b.name in ('curtain', 'stiff_curtain'):
                fold, period, flutter, see = self.cloth
                u = lo_[0] + tmin * ld[:, 0]                        # position along the bar
                off = fold * np.sin(2 * math.pi * u / period + self.phase)
                off += self.cloth_rng.normal(0.0, flutter * 0.5, off.shape)
                with np.errstate(divide='ignore', invalid='ignore'):
                    tmin = tmin + off / np.maximum(np.abs(ld[:, 1]), 0.25)
                hit &= self.cloth_rng.random(hit.shape) >= see
            t = np.where(hit & (tmin < t), tmin, t)
        if any(b.name in ('curtain', 'stiff_curtain') for b in world.boxes):
            self.phase += self.cloth_rng.normal(0.0, 0.6)          # the folds move (flutter)
        ok = t < self.max_range
        r = t[ok][:, None]
        pts = self.dirs[ok] * r
        pts += self.rng.normal(0.0, self.noise, pts.shape)
        if self.blob:
            pts = np.vstack([pts, np.zeros((self.blob, 3))])
        return pts
