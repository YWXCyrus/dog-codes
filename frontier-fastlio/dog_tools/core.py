"""dog_tools core: pure python + numpy, NO ROS import (so it is unit-testable anywhere).

Conventions (same as dog_codes): odom frame, origin = start pose, +X = initial forward, +Y = left,
lattice cells CELL = 1.02 m centred on (i*CELL, j*CELL). h = height above the dog's foot plane.
"""
import math

import numpy as np

CELL = 1.02
MOVING_SPEED = 0.05          # m/s: a sample counts as "moving" above this

SAMPLE_COLS = ['t_wall', 't_rel', 'x', 'y', 'yaw_deg', 'speed', 'roll_odom', 'pitch_odom',
               'roll_imu', 'pitch_imu', 'gyro_dps', 'rough_std', 'rough_frac', 'near_n',
               'rough_hit', 'rough_on', 'label']

KNOWN_GAITS = ['flat_slow', 'flat_medium', 'flat_fast', 'crawl', 'general', 'grip', 'hstep']
UNLABELLED = 'unlabelled'


# ------------------------------------------------------------------ small math
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_to_rpy(x, y, z, w):
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tilt_from_accel(ax, ay, az):
    """(roll, pitch) in rad from a mean accelerometer vector; None if it is ~zero. Scale free."""
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-6:
        return None
    return math.atan2(ay, az), math.atan2(-ax, math.sqrt(ay * ay + az * az))


# ------------------------------------------------------------------ point cloud
_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32,
       7: np.float32, 8: np.float64}


def cloud_to_xyz(msg):
    """sensor_msgs/PointCloud2 -> (N,3) float32 of finite x,y,z (numpy only)."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32)
    off = {f.name: f for f in msg.fields}
    if not all(k in off for k in 'xyz'):
        return np.zeros((0, 3), np.float32)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = msg.point_step
    buf = buf[:n * step].reshape(n, step)
    cols = []
    for k in 'xyz':
        f = off[k]
        dt = np.dtype(_DT[f.datatype]).newbyteorder('>' if msg.is_bigendian else '<')
        col = np.ascontiguousarray(buf[:, f.offset:f.offset + dt.itemsize]).view(dt).reshape(n)
        cols.append(col.astype(np.float32))
    xyz = np.stack(cols, axis=1)
    return xyz[np.isfinite(xyz).all(axis=1)]


class BodyHeightCalib:
    """Standing body height above the floor, from the floor seen around the dog (same method as
    dog_codes/cloud_proc.py). fixed>0 skips the calibration."""

    def __init__(self, default=0.30, n_needed=8, min_floor_pts=150, fixed=0.0):
        self.value = float(fixed) if fixed > 0 else float(default)
        self.calibrated = fixed > 0
        self.n_needed = n_needed
        self.min_floor_pts = min_floor_pts
        self.samples = []

    def update(self, w, px, py, robot_z):
        """w: (N,3) world points (odom frame, absolute z)."""
        if self.calibrated:
            return True
        d = np.hypot(w[:, 0] - px, w[:, 1] - py)
        sel = w[(d > 0.5) & (d < 2.5) & (w[:, 2] < robot_z)]
        if sel.shape[0] < self.min_floor_pts:
            return False
        bh = robot_z - float(np.percentile(sel[:, 2], 10))
        if not (0.12 < bh < 0.50):
            return False
        self.samples.append(bh)
        if len(self.samples) >= self.n_needed:
            self.value = float(np.median(self.samples))
            self.calibrated = True
        return self.calibrated


# ------------------------------------------------------------------ rough detector (mirrors terrain.py)
def rough_features(xyh, px, py, yaw):
    """Exactly the 'rough' measurement of dog_codes/terrain.py analyse(): floor-height scatter in a
    window 0.9-2.0 m ahead, +-0.4 m wide. xyh: (N,3) x,y,h. Returns (std, frac_above_4cm, n) or None."""
    if xyh.shape[0] == 0:
        return None
    dx = xyh[:, 0] - px
    dy = xyh[:, 1] - py
    cy, sy = math.cos(yaw), math.sin(yaw)
    lx = cy * dx + sy * dy
    ly = -sy * dx + cy * dy
    h = xyh[:, 2]
    wide = (np.abs(ly) < 0.9) & (lx > 0.1) & (lx < 2.9) & (h < 0.6) & (h > -0.5)
    if int(wide.sum()) < 30:
        return None
    lx, ly, h = lx[wide], ly[wide], h[wide]
    near = (lx > 0.9) & (lx < 2.0) & (np.abs(ly) < 0.4) & (h > -0.10) & (h < 0.30)
    hn = h[near]
    if hn.size < 25:
        return None
    return float(np.std(hn)), float(np.mean(hn > 0.04)), int(hn.size)


class RoughDetector:
    """The supervisor's rough flag: ON after on_n consecutive hits, OFF after off_n consecutive
    misses (5 Hz updates). Stairs suppression of terrain.py is NOT included (stairs not analysed)."""

    def __init__(self, std=0.050, frac=0.35, on_n=4, off_n=15):
        self.std, self.frac, self.on_n, self.off_n = std, frac, on_n, off_n
        self.hits = 0
        self.miss = 0
        self.on = False

    def update(self, feat):
        hit = bool(feat and (feat[0] > self.std or feat[1] > self.frac))
        if hit:
            self.hits += 1
            self.miss = 0
        else:
            self.miss += 1
            self.hits = 0
        flip = None
        if not self.on and self.hits >= self.on_n:
            self.on = True
            flip = 'on'
        elif self.on and self.miss >= self.off_n:
            self.on = False
            flip = 'off'
        return hit, self.on, flip


# ------------------------------------------------------------------ grid map
def box_sum(a, r):
    """Sum over the (2r+1)x(2r+1) window around every element (zero padded)."""
    ny, nx = a.shape
    ap = np.zeros((ny + 1, nx + 1))
    ap[1:, 1:] = a.cumsum(0).cumsum(1)
    ar = np.arange(ny)
    ac = np.arange(nx)
    y0 = np.clip(ar - r, 0, ny)
    y1 = np.clip(ar + r + 1, 0, ny)
    x0 = np.clip(ac - r, 0, nx)
    x1 = np.clip(ac + r + 1, 0, nx)
    return ap[y1][:, x1] - ap[y0][:, x1] - ap[y1][:, x0] + ap[y0][:, x0]


class GridMap:
    """Top-down accumulation of the cloud in the odom frame. Per 0.1 m cell: floor-layer point
    count/sum/sum^2/min/max of h (h in [floor_lo, wall_h]) and the number of points above wall_h
    (walls). Array index is [iy, ix]; iy grows with +Y, ix with +X."""

    def __init__(self, x0=-6.0, x1=14.0, y0=-8.0, y1=8.0, res=0.1, floor_lo=-0.35,
                 wall_h=0.60, wall_max=2.0):
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        self.nx = int(round((x1 - x0) / res))
        self.ny = int(round((y1 - y0) / res))
        self.floor_lo, self.wall_h, self.wall_max = floor_lo, wall_h, wall_max
        N = self.nx * self.ny
        self.n = np.zeros(N)
        self.s = np.zeros(N)
        self.s2 = np.zeros(N)
        self.hmin = np.full(N, np.inf)
        self.hmax = np.full(N, -np.inf)
        self.nwall = np.zeros(N)
        self.scans = 0

    # -- accumulate
    def add(self, xyh):
        if xyh.shape[0] == 0:
            return
        x, y, h = xyh[:, 0], xyh[:, 1], xyh[:, 2].astype(np.float64)
        ix = np.floor((x - self.x0) / self.res).astype(np.int64)
        iy = np.floor((y - self.y0) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        fl = ok & (h >= self.floor_lo) & (h <= self.wall_h)
        wl = ok & (h > self.wall_h) & (h <= self.wall_max)
        N = self.nx * self.ny
        if fl.any():
            f = iy[fl] * self.nx + ix[fl]
            hh = h[fl]
            self.n += np.bincount(f, minlength=N)
            self.s += np.bincount(f, weights=hh, minlength=N)
            self.s2 += np.bincount(f, weights=hh * hh, minlength=N)
            np.minimum.at(self.hmin, f, hh)
            np.maximum.at(self.hmax, f, hh)
        if wl.any():
            self.nwall += np.bincount(iy[wl] * self.nx + ix[wl], minlength=N)
        self.scans += 1

    # -- views
    def g(self, a):
        return a.reshape(self.ny, self.nx)

    def wall_cells(self, min_pts=3):
        return self.g(self.nwall) >= min_pts

    def observed(self):
        return (self.g(self.n) > 0) | (self.g(self.nwall) > 0)

    def centres(self):
        cx = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        cy = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        return cx, cy

    def _stats(self, r, min_n):
        wc = self.wall_cells()
        n = self.g(self.n).copy()
        s = self.g(self.s).copy()
        s2 = self.g(self.s2).copy()
        n[wc] = 0
        s[wc] = 0
        s2[wc] = 0
        N = box_sum(n, r)
        S = box_sum(s, r)
        S2 = box_sum(s2, r)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = S / N
            var = S2 / N - mean * mean
        mean[N < min_n] = np.nan
        std = np.sqrt(np.clip(var, 0, None))
        std[N < min_n] = np.nan
        return mean, std

    def local_rough(self, r=2, min_n=12):
        """std of floor height in a (2r+1) cell block (r=2 -> 0.5 m). NaN where too few points."""
        return self._stats(r, min_n)[1]

    def mean_height(self, r=1, min_n=4):
        return self._stats(r, min_n)[0]

    def bounds(self):
        """(xmin, xmax, ymin, ymax) of observed cells in metres, or None."""
        obs = self.observed()
        if not obs.any():
            return None
        ys, xs = np.where(obs)
        return (self.x0 + xs.min() * self.res, self.x0 + (xs.max() + 1) * self.res,
                self.y0 + ys.min() * self.res, self.y0 + (ys.max() + 1) * self.res)

    def wall_iou(self, other):
        """Overlap (intersection over union) of the wall cells of two maps with the same geometry."""
        a, b = self.wall_cells(), other.wall_cells()
        u = int((a | b).sum())
        return float((a & b).sum()) / u if u else float('nan')

    # -- persistence
    def to_arrays(self):
        return dict(n=self.n, s=self.s, s2=self.s2, hmin=self.hmin, hmax=self.hmax,
                    nwall=self.nwall,
                    geom=np.array([self.x0, self.y0, self.res, self.nx, self.ny, self.floor_lo,
                                   self.wall_h, self.wall_max, self.scans]))

    @classmethod
    def from_arrays(cls, d):
        geom = d['geom']
        x0, y0, res, nx, ny, flo, wh, wm, scans = [float(v) for v in geom]
        m = cls(x0=x0, x1=x0 + int(nx) * res, y0=y0, y1=y0 + int(ny) * res, res=res,
                floor_lo=flo, wall_h=wh, wall_max=wm)
        for k in ('n', 's', 's2', 'hmin', 'hmax', 'nwall'):
            setattr(m, k, np.array(d[k], dtype=np.float64))
        m.scans = int(scans)
        return m


def merge_grids(grids):
    """Sum several maps of identical geometry (same origin) into one."""
    g0 = grids[0]
    out = GridMap(x0=g0.x0, x1=g0.x0 + g0.nx * g0.res, y0=g0.y0, y1=g0.y0 + g0.ny * g0.res, res=g0.res,
                  floor_lo=g0.floor_lo, wall_h=g0.wall_h, wall_max=g0.wall_max)
    for g in grids:
        if ((g.nx, g.ny) != (g0.nx, g0.ny) or abs(g.x0 - g0.x0) > 1e-9 or abs(g.y0 - g0.y0) > 1e-9
                or abs(g.res - g0.res) > 1e-9):
            raise ValueError('maps have different geometry (survey run with different map_* parameters)')
        out.n += g.n
        out.s += g.s
        out.s2 += g.s2
        out.nwall += g.nwall
        out.hmin = np.minimum(out.hmin, g.hmin)
        out.hmax = np.maximum(out.hmax, g.hmax)
        out.scans += g.scans
    return out


# ------------------------------------------------------------------ label file
def read_new_rows(path, offset):
    """Complete csv rows appended to `path` after byte `offset` -> (rows, new_offset).
    A half-written last line is left for the next call; the header row is skipped."""
    import csv
    import io
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b'\n')
    if end < 0:
        return [], offset
    text = data[:end + 1].decode('utf-8', 'replace')
    rows = [r for r in csv.reader(io.StringIO(text, newline='')) if len(r) >= 3 and r[0] != 't_wall']
    return rows, offset + end + 1


# ------------------------------------------------------------------ segment statistics
def _finite(vals):
    a = np.array([v for v in vals if v == v and math.isfinite(v)], dtype=float)
    return a


def _nanmean(vals):
    a = _finite(vals)
    return float(a.mean()) if a.size else float('nan')


def path_length(run):
    """Walked distance: steps count only while moving (speed >= MOVING_SPEED), because odometry jitters by a
    few mm per sample even at rest and that would add metres over minutes; a jump >= 0.5 m is a relocalisation."""
    d = 0.0
    for a, b in zip(run[:-1], run[1:]):
        step = math.hypot(b['x'] - a['x'], b['y'] - a['y'])
        if step < 0.5 and b['speed'] >= MOVING_SPEED:
            d += step
    return d


def runs_by_label(samples, max_gap_s=2.0):
    """Split into runs of consecutive samples with the same label (a time gap also splits)."""
    runs = []
    for s in samples:
        if (runs and runs[-1][-1]['label'] == s['label'] and
                s['t_wall'] - runs[-1][-1]['t_wall'] <= max_gap_s):
            runs[-1].append(s)
        else:
            runs.append([s])
    return runs


def run_stats(runs, dt, events=()):
    """Statistics over a list of runs (all one label, or a single run)."""
    allr = [s for r in runs for s in r]
    mov = [s for s in allr if s['speed'] >= MOVING_SPEED]
    dur = sum(r[-1]['t_wall'] - r[0]['t_wall'] + dt for r in runs)
    tilt = [max(abs(s['roll_odom']), abs(s['pitch_odom'])) for s in allr
            if s['roll_odom'] == s['roll_odom'] and s['pitch_odom'] == s['pitch_odom']]
    ri = _finite([s['roll_imu'] for s in mov])
    pi = _finite([s['pitch_imu'] for s in mov])
    wob = float('nan')
    if ri.size >= 3 and pi.size >= 3:
        wob = math.sqrt(float(np.var(ri)) + float(np.var(pi)))
    gy = _finite([s['gyro_dps'] for s in mov])
    gyro = float(np.sqrt(np.mean(gy ** 2))) if gy.size else float('nan')
    flips = 0
    for r in runs:
        for a, b in zip(r[:-1], r[1:]):
            if int(a['rough_on']) != int(b['rough_on']):
                flips += 1
    marks = 0
    for r in runs:
        t0, t1 = r[0]['t_wall'], r[-1]['t_wall'] + dt
        marks += sum(1 for (t, kind, _txt) in events if kind == 'mark' and t0 <= t < t1)
    nm = len(mov)
    return {
        'label': allr[0]['label'], 'dur_s': dur, 'moving_s': nm * dt,
        'dist_m': sum(path_length(r) for r in runs),
        'speed_mean': _nanmean([s['speed'] for s in mov]),
        'tilt_max_deg': max(tilt) if tilt else float('nan'),
        'wobble_deg': wob, 'gyro_rms_dps': gyro,
        'rough_std_mean': _nanmean([s['rough_std'] for s in mov]),
        'rough_hit_pct': 100.0 * sum(int(s['rough_hit']) for s in mov) / nm if nm else float('nan'),
        'rough_on_pct': 100.0 * sum(int(s['rough_on']) for s in mov) / nm if nm else float('nan'),
        'rough_flips': flips, 'marks': marks,
    }


def sample_dt(samples):
    if len(samples) < 2:
        return 0.2
    d = np.diff([s['t_wall'] for s in samples])
    d = d[(d > 0) & (d < 2.0)]
    return float(np.median(d)) if d.size else 0.2


def summarize(samples, events=()):
    """Rows: one per contiguous run of a label ('segment'), then one per label ('total')."""
    if not samples:
        return []
    dt = sample_dt(samples)
    t_ref = samples[0]['t_wall']
    runs = runs_by_label(samples)
    rows = []
    for i, r in enumerate(runs, 1):
        st = run_stats([r], dt, events)
        st.update(kind='segment', seg=i, t_start=r[0]['t_wall'] - t_ref, t_end=r[-1]['t_wall'] - t_ref)
        rows.append(st)
    by = {}
    for r in runs:
        by.setdefault(r[0]['label'], []).append(r)
    for lab, rs in by.items():
        st = run_stats(rs, dt, events)
        st.update(kind='total', seg=len(rs), t_start=rs[0][0]['t_wall'] - t_ref,
                  t_end=rs[-1][-1]['t_wall'] - t_ref)
        rows.append(st)
    return rows


SUMMARY_COLS = ['kind', 'seg', 'label', 't_start', 't_end', 'dur_s', 'moving_s', 'dist_m', 'speed_mean',
                'tilt_max_deg', 'wobble_deg', 'gyro_rms_dps', 'rough_std_mean', 'rough_hit_pct',
                'rough_on_pct', 'rough_flips', 'marks']


def _f(v, w, p):
    if isinstance(v, float) and v != v:
        return '-'.rjust(w)
    return ('%*.*f' % (w, p, v)) if isinstance(v, float) else str(v).rjust(w)


def format_summary(rows):
    head = ('%-8s %3s %-14s %6s %6s %6s %7s %6s %6s %6s %7s %6s %7s %6s %6s %5s %5s' % (
        'kind', 'seg', 'label', 't0', 't1', 'dur', 'moving', 'dist', 'speed', 'tiltmx', 'wobble', 'gyro',
        'rough', 'hit%', 'on%', 'flips', 'marks'))
    out = [head]
    for r in rows:
        out.append('%-8s %3d %-14s %s %s %s %s %s %s %s %s %s %s %s %s %5d %5d' % (
            r['kind'], r['seg'], r['label'][:14], _f(r['t_start'], 6, 1), _f(r['t_end'], 6, 1),
            _f(r['dur_s'], 6, 1), _f(r['moving_s'], 7, 1), _f(r['dist_m'], 6, 2),
            _f(r['speed_mean'], 6, 2), _f(r['tilt_max_deg'], 6, 1), _f(r['wobble_deg'], 7, 2),
            _f(r['gyro_rms_dps'], 6, 1), _f(r['rough_std_mean'], 7, 3), _f(r['rough_hit_pct'], 6, 0),
            _f(r['rough_on_pct'], 6, 0), r['rough_flips'], r['marks']))
    return '\n'.join(out)
