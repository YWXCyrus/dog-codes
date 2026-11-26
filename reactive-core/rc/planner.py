"""Frontier planner (2026-11-25): where to go next, as a heading hint for the driver.

Replaces "leftmost free lane" wall following as the first choice. Pure Python + numpy, no ROS.

World frame = the leg-odometry frame (x, y, yaw) the driver already uses. The planner keeps its own
persistent 2-D grid (the FAST-LIO map in liomap.py is local and forgets):
  occ   - cells with obstacle / drop points (from the scan cells the lanes use)
  seen  - cells the lidar looked at: ray-cast over +-plan_fov_deg up to plan_seen_r_m, stopped at the
          first obstacle in each bearing bin; a disc around the dog and its own trail count as seen
          (the rear is blind, the dog came from there).
Frontier = reachable free cell with an unseen 4-neighbour. Goal = the cheapest frontier (Dijkstra
distance + a turn penalty, so it does not choose something behind the dog while one is ahead),
committed for plan_commit_s. Output = yaw (odom frame) of the point plan_lookahead_m along the path.

The planner only advises: the driver still picks a FREE lane of the live scan, so a path through
something the lidar sees blocked is never followed.
"""
import heapq
import math

import numpy as np

from rc import geometry as G

_N4 = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0))
_N8 = _N4 + ((1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))


def lookahead_yaw(path, pose, look):
    """Yaw (odom frame) from `pose` to the first path point at least `look` m away (the last one if all are
    nearer). `path` = [(x, y), ...] in the odom frame. Shared by the own planner and an external (Nav2) path."""
    x, y = pose[0], pose[1]
    tgt = path[-1]
    for p in path:
        if math.hypot(p[0] - x, p[1] - y) >= look:
            tgt = p
            break
    return math.atan2(tgt[1] - y, tgt[0] - x)


class PlanResult:
    __slots__ = ('yaw', 'goal', 'path', 'reason')

    def __init__(self, yaw=None, goal=None, path=None, reason=''):
        self.yaw, self.goal, self.path, self.reason = yaw, goal, path or [], reason


class Planner:

    def __init__(self, cfg):
        self.c = cfg
        self.res = float(cfg['plan_res_m'])
        self.n = int(2 * cfg['plan_half_m'] / self.res)
        self.half = self.n // 2
        self.reset()

    def reset(self):
        self.occ = np.zeros((self.n, self.n), dtype=bool)
        self.seen = np.zeros((self.n, self.n), dtype=bool)
        self.goal = None                 # (gx, gy) cell
        self.goal_t = -1e9
        self.blacklist = []              # (gx, gy, until_t)
        self.last = PlanResult(reason='reset')
        self.last_plan_t = -1e9
        self.updates = 0
        self.external = None             # (path, t) from Nav2 via set_external_path

    # ------------------------------------------------------------ grid helpers
    def to_cell(self, x, y):
        return (int(math.floor(x / self.res)) + self.half, int(math.floor(y / self.res)) + self.half)

    def to_world(self, cx, cy):
        return ((cx - self.half + 0.5) * self.res, (cy - self.half + 0.5) * self.res)

    def _in(self, cx, cy):
        return 0 <= cx < self.n and 0 <= cy < self.n

    def _cells_xy(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        c = np.floor(xy / self.res).astype(np.int64) + self.half
        ok = (c[:, 0] >= 0) & (c[:, 0] < self.n) & (c[:, 1] >= 0) & (c[:, 1] < self.n)
        return c[ok]

    # ------------------------------------------------------------ mapping
    def update(self, pose, obs_body, drops_body):
        """pose = (x, y, yaw) odom frame; obstacle / drop cell centres in the body frame."""
        c = self.c
        x, y, yaw = pose
        self.updates += 1
        ob = np.vstack([np.asarray(obs_body, dtype=float).reshape(-1, 2),
                        np.asarray(drops_body, dtype=float).reshape(-1, 2)])
        if ob.shape[0]:
            cells = self._cells_xy(G.body_to_world(ob, pose))
            if cells.shape[0]:
                self.occ[cells[:, 1], cells[:, 0]] = True
                self.seen[cells[:, 1], cells[:, 0]] = True
        # rays over the field of view stop at the first occupied cell of the persistent grid (dilated by
        # one cell: obstacle cells are ~0.1 m apart, 2-degree rays would slip between them)
        fov = math.radians(c['plan_fov_deg'])
        rmax = float(c['plan_seen_r_m'])
        k = int(math.ceil(rmax / self.res)) + 2
        cx, cy = self.to_cell(x, y)
        x0, x1 = max(0, cx - k), min(self.n, cx + k + 1)
        y0, y1 = max(0, cy - k), min(self.n, cy + k + 1)
        w = self.occ[y0:y1, x0:x1]
        blk = w.copy()
        blk[1:] |= w[:-1]
        blk[:-1] |= w[1:]
        blk[:, 1:] |= w[:, :-1]
        blk[:, :-1] |= w[:, 1:]
        ang = yaw + np.arange(-fov, fov + 1e-9, math.radians(2.0))
        ds = np.arange(0.2, rmax + 1e-9, self.res * 0.5)
        gx = np.floor((x + np.outer(np.cos(ang), ds)) / self.res).astype(np.int64) + self.half - x0
        gy = np.floor((y + np.outer(np.sin(ang), ds)) / self.res).astype(np.int64) + self.half - y0
        inside = (gx >= 0) & (gx < x1 - x0) & (gy >= 0) & (gy < y1 - y0)
        hit = np.zeros(gx.shape, dtype=bool)
        hit[inside] = blk[gy[inside], gx[inside]]
        passed = np.cumsum(hit, axis=1) > 0                  # at or beyond the first blocked sample
        ok = inside & ~passed
        self.seen[(gy[ok] + y0), (gx[ok] + x0)] = True
        # the dog's own surroundings (blind rear) and trail are known free ground
        rr = float(c['plan_near_seen_m'])
        k = int(rr / self.res)
        cx, cy = self.to_cell(x, y)
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        m = (xx * xx + yy * yy) * self.res ** 2 <= rr * rr
        gx, gy = cx + xx[m], cy + yy[m]
        okc = (gx >= 0) & (gx < self.n) & (gy >= 0) & (gy < self.n)
        gx, gy = gx[okc], gy[okc]
        free_near = ~self.occ[gy, gx]
        self.seen[gy[free_near], gx[free_near]] = True

    # ------------------------------------------------------------ planning
    def _inflated(self, lo_x, hi_x, lo_y, hi_y):
        k = int(math.ceil(self.c['plan_inflate_m'] / self.res))
        sub = self.occ[lo_y:hi_y, lo_x:hi_x]
        out = sub.copy()
        ys, xs = np.nonzero(sub)
        h, w = sub.shape
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if dx * dx + dy * dy > k * k:
                    continue
                yy, xx = ys + dy, xs + dx
                ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                out[yy[ok], xx[ok]] = True
        return out

    def set_external_path(self, path, t):
        """A path from Nav2's planner server (odom-frame (x, y) points). While it is fresh
        (plan_external_s) it replaces the own path; the own frontier goal is still what Nav2 is asked for."""
        self.external = (list(path), t)

    def plan(self, pose, t, force=False):
        """Recompute at plan_hz; returns the latest PlanResult (yaw None = no advice)."""
        c = self.c
        if self.external is not None and t - self.external[1] <= c['plan_external_s'] and self.external[0]:
            path = self.external[0]
            self.last = PlanResult(yaw=lookahead_yaw(path, pose, float(c['plan_lookahead_m'])),
                                   goal=tuple(path[-1]), path=path, reason='nav2')
            return self.last
        if not force and t - self.last_plan_t < 1.0 / c['plan_hz']:
            return self.last
        self.last_plan_t = t
        self.last = self._plan(pose, t)
        return self.last

    def _plan(self, pose, t):
        c = self.c
        x, y, yaw = pose
        if self.updates < 3:
            return PlanResult(reason='warming up')
        rcx, rcy = self.to_cell(x, y)
        wk = int(c['plan_window_m'] / self.res)
        lo_x, hi_x = max(0, rcx - wk), min(self.n, rcx + wk + 1)
        lo_y, hi_y = max(0, rcy - wk), min(self.n, rcy + wk + 1)
        infl = self._inflated(lo_x, hi_x, lo_y, hi_y)
        seen = self.seen[lo_y:hi_y, lo_x:hi_x]
        free = seen & ~infl
        ox, oy = rcx - lo_x, rcy - lo_y
        free[max(0, oy - 2):oy + 3, max(0, ox - 2):ox + 3] = True      # start cell and neighbours
        H, W = free.shape
        INF = 1e18
        dist = np.full((H, W), INF)
        par = np.full((H, W, 2), -1, dtype=np.int32)
        dist[oy, ox] = 0.0
        pq = [(0.0, oy, ox)]
        while pq:
            d, cy, cx = heapq.heappop(pq)
            if d > dist[cy, cx]:
                continue
            for dx, dy, wgt in _N8:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or nx < 0 or ny >= H or nx >= W or not free[ny, nx]:
                    continue
                if dx and dy and not (free[cy, nx] and free[ny, cx]):
                    continue                                       # no diagonal through a corner
                nd = d + wgt
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    par[ny, nx] = (cy, cx)
                    heapq.heappush(pq, (nd, ny, nx))
        reach = dist < INF / 2
        # frontier: reachable free cell with an unseen 4-neighbour
        unseen = ~seen
        near_unseen = np.zeros_like(unseen)
        near_unseen[1:, :] |= unseen[:-1, :]
        near_unseen[:-1, :] |= unseen[1:, :]
        near_unseen[:, 1:] |= unseen[:, :-1]
        near_unseen[:, :-1] |= unseen[:, 1:]
        # a frontier at the window edge is not a frontier
        near_unseen[0, :] = near_unseen[-1, :] = False
        near_unseen[:, 0] = near_unseen[:, -1] = False
        front = reach & near_unseen & free
        self.blacklist = [b for b in self.blacklist if b[2] > t]
        for bx, by, _ in self.blacklist:
            ex, ey = bx - lo_x, by - lo_y
            r = int(0.5 / self.res)
            front[max(0, ey - r):ey + r + 1, max(0, ex - r):ex + r + 1] = False
        ys, xs = np.nonzero(front)
        if ys.size == 0:
            return PlanResult(reason='no frontier')
        wx = (xs + lo_x - self.half + 0.5) * self.res
        wy = (ys + lo_y - self.half + 0.5) * self.res
        dx_, dy_ = wx - x, wy - y
        rr = np.hypot(dx_, dy_)
        keep = rr >= c['plan_front_min_m']
        if not keep.any():
            return PlanResult(reason='frontier too close')
        ys, xs, wx, wy, dx_, dy_, rr = ys[keep], xs[keep], wx[keep], wy[keep], dx_[keep], dy_[keep], rr[keep]
        ang = np.abs(np.array([G.wrap(math.atan2(b, a) - yaw) for a, b in zip(dx_, dy_)]))
        cost = dist[ys, xs] * self.res + c['plan_turn_cost'] * ang
        # keep the committed goal while it is still a frontier
        goal_cell = None
        if self.goal is not None and t - self.goal_t < c['plan_commit_s']:
            gx, gy = self.goal[0] - lo_x, self.goal[1] - lo_y
            if 0 <= gy < H and 0 <= gx < W and front[max(0, gy - 3):gy + 4, max(0, gx - 3):gx + 4].any():
                goal_cell = (gy, gx)
        if goal_cell is None:
            i = int(np.argmin(cost))
            goal_cell = (int(ys[i]), int(xs[i]))
            self.goal = (goal_cell[1] + lo_x, goal_cell[0] + lo_y)
            self.goal_t = t
        if math.hypot(*(np.array(self.to_world(self.goal[0], self.goal[1])) - (x, y))) < 0.6:
            self.blacklist.append((self.goal[0], self.goal[1], t + 30.0))
            self.goal = None
        # path from the goal back to the dog
        path = []
        cy, cx = goal_cell
        guard = 0
        while (cy, cx) != (oy, ox) and guard < 4000:
            path.append(self.to_world(cx + lo_x, cy + lo_y))
            py, px = par[cy, cx]
            if py < 0:
                return PlanResult(reason='no path')
            cy, cx = int(py), int(px)
            guard += 1
        path.reverse()
        if not path:
            return PlanResult(reason='at goal')
        gx, gy = self.to_world(*self.goal) if self.goal else path[-1]
        return PlanResult(yaw=lookahead_yaw(path, pose, float(c['plan_lookahead_m'])), goal=(gx, gy),
                          path=path, reason='ok')

    def mark_failed(self, t):
        """The driver could not follow the hint (blocked): do not pick this goal again for a while."""
        if self.goal is not None:
            self.blacklist.append((self.goal[0], self.goal[1], t + 20.0))
            self.goal = None
