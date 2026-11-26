"""Arena pose: dead reckoning in the booklet arena frame, snapped to the cell borders.

  heading  = gyro heading (relative to the heading at Enter) + cyaw (yaw corrections from the snap)
  position = sum of the odometry's BODY-FRAME steps (forward / sideways, computed with the source's own
             yaw) rotated by that heading, + position corrections from the snap.
Only body-frame steps are taken from the odometry, so a drifting odometry heading (the leg odometry's
own yaw, FAST-LIO's) does not turn into a growing sideways error; the source can change (leg <-> LIO)
without a jump. Origin = the pose at Enter = the start cell centre; +X = start heading, +Y = left.

GRID SNAP (the arena is Manhattan: every panel stands on a line x = (k + 1/2) L or y = (k + 1/2) L):
  1. wall returns from the levelled scan (height above the stance floor snap_z_lo..hi, within
     snap_range_m), placed with the current arena pose;
  2. yaw: rotate them about the dog by -snap_yaw_max..+snap_yaw_max; the rotation that makes the x and
     y histograms sharpest (walls collapse into thin peaks) is the heading error;
  3. position: strong, long, continuous peaks of those histograms are panel faces; their offset from
     the nearest border line (within snap_max_off_m, agreeing within snap_agree_m) is the position error.
  Corrections are applied with a gain and a cap per update, only while the dog is level, not on the
  stairs and not turning fast (a turning scan is smeared). Drift between updates must stay below
  ~0.25 m (half the distance to the next border line) - at 2-5 Hz it is a few cm.
"""
import math

import numpy as np

from rc import geometry as G


def _sharp(v, lo, bin_m, nb):
    idx = np.floor((v - lo) / bin_m).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < nb)]
    h = np.bincount(idx, minlength=nb).astype(float)
    return float((h * h).sum())


class GridLocalizer:

    def __init__(self, cfg):
        self.c = cfg
        self.L = float(cfg['arena_cell_m'])
        self.started = False
        self.x = 0.0
        self.y = 0.0
        self.h0 = 0.0                        # gyro heading at Enter
        self.cyaw = 0.0                      # yaw correction (snap)
        self.prev = {}                       # source name -> last source pose
        self.src = None                      # active source name
        self.jumps = 0
        self.step_mode = 'odom'              # odom: steps split with the source's own yaw | forward: |step| along the heading
        self.turn_g = 0.0                    # gyro / source yaw turned (rad), for the source-yaw check
        self.turn_s = 0.0
        self.events = []
        self.last_t = -1e9
        self.n_snaps = 0
        self.last = {}                       # what the last update saw (status / viewer)
        self.total_yaw = 0.0                 # sum of applied yaw corrections (deg) - drift indicator
        self.total_xy = 0.0

    # ------------------------------------------------------------------ dead reckoning
    def start(self, heading):
        self.started = True
        self.x = self.y = self.cyaw = 0.0
        self.h0 = heading
        self.prev = {}

    def yaw(self, heading):
        return G.wrap(heading - self.h0 + self.cyaw)

    def pose(self, heading):
        """Arena pose (x, y, yaw) for the current gyro heading."""
        if not self.started or heading is None:
            return None
        return self.x, self.y, self.yaw(heading)

    def feed(self, name, src, heading, vx_cmd=None):
        """One pose of source `name` (x, y, its own yaw). Only the active source moves the dog."""
        if not self.started or src is None:
            return
        if name != self.src:
            self.src = name
            self.prev[name] = tuple(src)              # a new source starts from here: no jump
            self._ph = heading
            return
        p = self.prev.get(name)
        self.prev[name] = tuple(src)
        if p is None:
            return
        # source-yaw check: while the gyro turns, the source's own yaw must follow (a frozen / lagging
        # odometry yaw would split the steps wrongly) - else forward-only steps along the gyro heading
        ph = getattr(self, '_ph', heading)
        self._ph = heading
        if heading is not None and ph is not None:
            self.turn_g += abs(G.wrap(heading - ph))
            self.turn_s += abs(G.wrap(src[2] - p[2]))
            if self.turn_g > math.radians(120.0):
                if self.step_mode == 'odom' and self.turn_s < 0.5 * self.turn_g and self.c['loc_step'] == 'auto':
                    self.step_mode = 'forward'
                    self.events.append('the %s yaw turned %.0f deg while the gyro turned %.0f: forward-only steps'
                                       % (name, math.degrees(self.turn_s), math.degrees(self.turn_g)))
                self.turn_g = self.turn_s = 0.0
        dx, dy = src[0] - p[0], src[1] - p[1]
        if self.step_mode == 'forward' or self.c['loc_step'] == 'forward':
            # the dog never walks sideways: the step length along the heading, signed by the command
            d = math.hypot(dx, dy)
            sgn = -1.0 if (vx_cmd is not None and vx_cmd < -0.01) else 1.0
            fwd, lat = (sgn * d if vx_cmd is None or abs(vx_cmd) > 0.01 else 0.0), 0.0
        else:
            c, s = math.cos(p[2]), math.sin(p[2])
            fwd, lat = c * dx + s * dy, -s * dx + c * dy
        if math.hypot(fwd, lat) > self.c['loc_jump_m']:
            self.jumps += 1                          # a jump in the source: not a motion
            return
        a = self.yaw(heading)
        ca, sa = math.cos(a), math.sin(a)
        self.x += ca * fwd - sa * lat
        self.y += sa * fwd + ca * lat

    # ------------------------------------------------------------------ snap
    def wall_points(self, p_level, floor_z, apose):
        """Wall returns (N, 2) in the arena frame."""
        c = self.c
        p = np.asarray(p_level, dtype=float).reshape(-1, 3)
        if p.shape[0] == 0:
            return np.zeros((0, 2))
        h = p[:, 2] - floor_z
        r = np.hypot(p[:, 0], p[:, 1])
        m = (h >= c['snap_z_lo']) & (h <= c['snap_z_hi']) & (r >= 0.35) & (r <= c['snap_range_m'])
        q = p[m, :2]
        if q.shape[0] > c['snap_max_pts']:
            q = q[np.linspace(0, q.shape[0] - 1, c['snap_max_pts']).astype(np.int64)]
        return G.body_to_world(q, apose)

    def update(self, p_level, floor_z, heading, t, gyro_z=0.0, tilt_deg=0.0, allowed=True):
        """One snap step. Returns a dict describing it (also kept in self.last)."""
        c = self.c
        if (not c['snap'] or not self.started or heading is None
                or t - self.last_t < 1.0 / c['snap_hz'] - 1e-6):
            return None
        self.last_t = t
        out = {'t': round(t, 2), 'used': False}
        if not allowed:
            out['why'] = 'not allowed (stairs / curtain push)'
            self.last = out
            return out
        if abs(gyro_z) > c['snap_max_wz'] or tilt_deg > c['snap_max_tilt_deg']:
            out['why'] = 'turning' if abs(gyro_z) > c['snap_max_wz'] else 'tilted'
            self.last = out
            return out
        ap = self.pose(heading)
        W = self.wall_points(p_level, floor_z, ap)
        out['n'] = int(W.shape[0])
        if W.shape[0] < c['snap_min_pts']:
            out['why'] = 'few wall points'
            self.last = out
            return out
        px, py = ap[0], ap[1]
        bin_m = 0.03
        R = c['snap_range_m'] + 0.2
        nb = int(2 * R / bin_m) + 1
        # --- yaw: sharpest histograms
        best, best_d = -1.0, 0.0
        scores = {}
        D = W - np.array([px, py])
        step = math.radians(0.5)
        dmax = math.radians(c['snap_yaw_max_deg'])
        for d in np.arange(-dmax, dmax + 1e-9, step):
            cs, sn = math.cos(d), math.sin(d)
            X = cs * D[:, 0] - sn * D[:, 1] + px
            Y = sn * D[:, 0] + cs * D[:, 1] + py
            sc = _sharp(X, px - R, bin_m, nb) + _sharp(Y, py - R, bin_m, nb)
            scores[round(d, 5)] = sc
            if sc > best:
                best, best_d = sc, d
        s0 = scores.get(0.0, best)
        dyaw = 0.0
        if best > s0 * (1.0 + c['snap_yaw_gain_min']) and abs(best_d) < dmax - 1e-6:
            # parabolic refinement
            sm, sp = scores.get(round(best_d - step, 5)), scores.get(round(best_d + step, 5))
            if sm is not None and sp is not None and (sm - 2 * best + sp) < 0:
                best_d += 0.5 * step * (sm - sp) / (sm - 2 * best + sp)
            dyaw = best_d
        cs, sn = math.cos(dyaw), math.sin(dyaw)
        X = cs * D[:, 0] - sn * D[:, 1] + px
        Y = sn * D[:, 0] + cs * D[:, 1] + py
        self._dbg = []
        ex, nx = self._axis_offset(X, Y, px)
        dbg_x = self._dbg
        self._dbg = []
        ey, ny = self._axis_offset(Y, X, py)
        out['dbg'] = {'x': dbg_x, 'y': self._dbg}
        out.update(dyaw_deg=round(math.degrees(dyaw), 2), ex=None if ex is None else round(ex, 3),
                   ey=None if ey is None else round(ey, 3), peaks=[nx, ny])
        # --- apply (gain + cap)
        cap_y = math.radians(c['snap_cap_deg'])
        a_yaw = max(-cap_y, min(cap_y, c['snap_gain'] * dyaw))
        self.cyaw = G.wrap(self.cyaw + a_yaw)
        cap = c['snap_cap_m']
        if ex is not None:
            self.x -= max(-cap, min(cap, c['snap_gain'] * ex))
        if ey is not None:
            self.y -= max(-cap, min(cap, c['snap_gain'] * ey))
        used = bool(a_yaw) or ex is not None or ey is not None
        if used:
            self.n_snaps += 1
        self.total_yaw += abs(math.degrees(a_yaw))
        self.total_xy += abs(ex or 0.0) * c['snap_gain'] + abs(ey or 0.0) * c['snap_gain']
        out['used'] = used
        self.last = out
        return out

    def _axis_offset(self, V, U, p_v):
        """Offset of the panel faces along one axis from their border lines. V: coordinate across the
        panels, U: along them. Returns (weighted offset or None, number of peaks used)."""
        c = self.c
        L = self.L
        bin_m = 0.02
        lo = float(V.min()) - bin_m
        nb = int((float(V.max()) - lo) / bin_m) + 2
        idx = np.floor((V - lo) / bin_m).astype(np.int64)
        h = np.bincount(idx, minlength=nb)
        offs, ws = [], []
        used_bins = set()
        for b in np.argsort(-h):
            if h[b] < c['snap_peak_min']:
                break
            if any(abs(b - u) <= 3 for u in used_bins):
                continue
            used_bins.add(int(b))
            centre = lo + (b + 0.5) * bin_m
            m = np.abs(V - centre) <= 0.03
            if np.count_nonzero(m) < c['snap_peak_min']:
                continue
            u = U[m]
            u10, u90 = np.percentile(u, 10), np.percentile(u, 90)
            if u90 - u10 < c['snap_peak_span_m']:
                continue                                       # short: a prop, a post, not a panel
            # a panel is CONTINUOUS along its length; the scan stripes of the two side walls next to
            # the dog line up across it too, but with nothing in between (sim 2026-11-26: 0.4 m error)
            occ = np.unique(np.floor(u[(u >= u10) & (u <= u90)] / 0.05)).size * 0.05
            if occ < c['snap_peak_fill'] * (u90 - u10):
                continue
            v = float(np.median(V[m]))
            border = (round(v / L - 0.5) + 0.5) * L          # nearest border line (k + 1/2) L
            face = border + (0.01 if p_v > border else -0.01)  # panels are ~0.02 m thick
            off = v - face
            if abs(off) > c['snap_max_off_m']:
                continue
            offs.append(off)
            ws.append(float(np.count_nonzero(m)))
            self._dbg.append((round(v, 3), round(border, 3), round(off, 3), int(np.count_nonzero(m)),
                              round(float(np.percentile(u, 10)), 2), round(float(np.percentile(u, 90)), 2)))
            if len(offs) >= 4:
                break
        if not offs:
            return None, 0
        if len(offs) > 1 and max(offs) - min(offs) > c['snap_agree_m']:
            return None, -len(offs)                 # panels disagree: something else is in the picture
        return float(np.average(offs, weights=ws)), len(offs)

    def status(self):
        return {'snaps': self.n_snaps, 'yaw_fix_deg': round(self.total_yaw, 1), 'xy_fix_m': round(self.total_xy, 2),
                'last': self.last}
