"""Arena cell map in the booklet frame (v6.4 E): what the dog knows about the 3 x 5 (+1) grid.

Frame: origin = centre of the start cell, +X = the dog's start heading, +Y = its left. Cell (i, j) is
centred at (i * L, j * L), L = cell_m = 1.02 m (1.00 m panels + 0.02 m posts). Directions are named
like a map with the start heading up: N = +X, S = -X, W = +Y (left), E = -Y (right).

EDGES (cell borders). Each border is a vertical rectangle on the line x = (i + 0.5) L (between (i, j)
and (i + 1, j), key (i, j, 'x')) or y = (j + 0.5) L (between (i, j) and (i, j + 1), key (i, j, 'y')).
Every lidar scan votes, per border, in two height bands (heights above the stance floor):
  MID  edge_mid_lo..hi (0.22-0.42 m): a panel is there, a hanging curtain is not (hem ~0.45 m)
  UP   edge_up_lo..hi  (0.62-0.95 m): a hanging curtain is there, a 0.6 m panel is not
  hit  = a return ON the border line (within edge_tol_m) in the band
  pass = a ray that CROSSES the border line in the band and ends beyond it (it saw through)
State: WALL (MID mostly hit), OPEN (MID mostly pass and UP not blocked), CURTAIN (MID pass + UP hit =
cloth hanging from a bar, open below: booklet D.4; or a camera cue on a wall-like border), UNKNOWN.
A border the dog walked through is OPEN for good (CURTAIN if it was a curtain).

CELLS: discovered (floor returns seen inside), visited (the dog's base was within visit_r_m of the
centre), terrain (flat / uneven / elevated from the heights inside), labels from perception.

BOUNDS. The arena is 3 x 5 cells; the start cell is in the first row (an end row of the long side),
the start heading may point anywhere (booklet H). All placements of the 3 x 5 rectangle consistent with
that are kept as hypotheses (12 at the start). A visited cell outside a rectangle kills it unless it is
one of that rectangle's 3 final-cell candidates (the +1 cell beyond the far row). A strong OPEN border
from inside a rectangle to a cell outside it (not a final candidate) is a penalty point. The planner
explores the union of the live rectangles; the final cell comes from the live hypotheses.
"""
import math
from dataclasses import dataclass, field

import numpy as np

UNKNOWN, OPEN, WALL, CURTAIN = 'unknown', 'open', 'wall', 'curtain'
DIRS = {'N': (1, 0), 'S': (-1, 0), 'W': (0, 1), 'E': (0, -1)}
DIR_YAW = {'N': 0.0, 'W': math.pi / 2, 'S': math.pi, 'E': -math.pi / 2}


def yaw_dir(yaw):
    """Arena yaw (rad, CCW from +X) -> nearest of N/W/S/E."""
    k = int(round(yaw / (math.pi / 2))) % 4
    return ['N', 'W', 'S', 'E'][k]


def step(cell, d):
    di, dj = DIRS[d]
    return cell[0] + di, cell[1] + dj


def dir_between(a, b):
    di, dj = b[0] - a[0], b[1] - a[1]
    for k, v in DIRS.items():
        if v == (di, dj):
            return k
    return None


def edge_key(a, b):
    """Key of the border between two 4-neighbour cells (None if not neighbours)."""
    di, dj = b[0] - a[0], b[1] - a[1]
    if abs(di) + abs(dj) != 1:
        return None
    if di:
        i = min(a[0], b[0])
        return (i, a[1], 'x')
    j = min(a[1], b[1])
    return (a[0], j, 'y')


def edge_cells(k):
    i, j, ax = k
    return ((i, j), (i + 1, j)) if ax == 'x' else ((i, j), (i, j + 1))


@dataclass
class Edge:
    mh: float = 0.0          # MID band hits / passes
    mp: float = 0.0
    uh: float = 0.0          # UP band hits / passes
    up: float = 0.0
    traversed: bool = False
    cue: float = 0.0         # camera curtain cue (persistent detections pointing at this border)
    result: str = ''         # '' | passed | rigid | camera_no | camera_yes
    blocked_until: float = -1e9
    blocked_n: int = 0
    state: str = UNKNOWN
    was_curtain: bool = False


@dataclass
class Cell:
    floor_n: float = 0.0     # floor returns seen inside (discovered)
    visited: bool = False
    visit_t: float = None
    flat_n: float = 0.0
    raised_n: float = 0.0    # h 0.04..0.22: boards / rubble
    high_n: float = 0.0      # h 0.25..0.60: steps / platform
    stairs: bool = False     # stairs in this cell (lidar stairs detector, camera, perception)
    stairs_src: str = ''
    labels: dict = field(default_factory=dict)   # perception: scene, survivor, hazard, object, qr
    obstructed: int = 0      # times an obstacle inside it stopped the dog on the way through


class ArenaMap:

    def __init__(self, cfg):
        self.c = cfg
        self.L = float(cfg['arena_cell_m'])
        self.edges = {}
        self.cells = {}
        self.order = []                    # visit order
        self.hyps = self._hypotheses(int(cfg['arena_long']), int(cfg['arena_short']))
        self.penalised = set()
        self.final_cell = None
        self.no_bounds = False             # every hypothesis died (bad localization): bounds ignored
        self.events = []

    # ------------------------------------------------------------------ basic access
    def cell(self, c):
        c = (int(c[0]), int(c[1]))
        if c not in self.cells:
            self.cells[c] = Cell()
        return self.cells[c]

    def edge(self, k):
        if k not in self.edges:
            self.edges[k] = Edge()
        return self.edges[k]

    def edge_between(self, a, b):
        k = edge_key(a, b)
        return None if k is None else self.edge(k)

    def centre(self, c):
        return c[0] * self.L, c[1] * self.L

    def cell_of(self, x, y):
        return int(round(x / self.L)), int(round(y / self.L))

    def visited(self, c):
        return c in self.cells and self.cells[c].visited

    # ------------------------------------------------------------------ hypotheses (bounds)
    @staticmethod
    def _hypotheses(n_long, n_short):
        hs = []
        for axis in ('x', 'y'):
            for sgn in (1, -1):                # far end along +axis or -axis
                for off in range(-(n_short - 1), 1):
                    if axis == 'x':
                        i0, i1 = (0, n_long - 1) if sgn > 0 else (-(n_long - 1), 0)
                        j0, j1 = off, off + n_short - 1
                        fi = i1 + 1 if sgn > 0 else i0 - 1
                        finals = [(fi, j) for j in range(j0, j1 + 1)]
                    else:
                        j0, j1 = (0, n_long - 1) if sgn > 0 else (-(n_long - 1), 0)
                        i0, i1 = off, off + n_short - 1
                        fj = j1 + 1 if sgn > 0 else j0 - 1
                        finals = [(i, fj) for i in range(i0, i1 + 1)]
                    far = [(c[0] - (sgn if axis == 'x' else 0), c[1] - (sgn if axis == 'y' else 0)) for c in finals]
                    hs.append({'rect': (i0, i1, j0, j1), 'axis': axis, 'sgn': sgn, 'finals': finals,
                               'far': far, 'alive': True, 'pen': 0})
        return hs

    @staticmethod
    def _in_rect(h, c):
        i0, i1, j0, j1 = h['rect']
        return i0 <= c[0] <= i1 and j0 <= c[1] <= j1

    def live(self):
        if self.no_bounds:
            return []
        al = [h for h in self.hyps if h['alive']]
        if not al:
            return []
        m = min(h['pen'] for h in al)
        return [h for h in al if h['pen'] <= m + int(self.c['hyp_pen_slack'])]

    def possible(self, c):
        """Could c be one of the 15 arena cells? (True for everything when the bounds are lost.)"""
        lv = self.live()
        if not lv:
            return True
        return any(self._in_rect(h, c) for h in lv)

    def certain(self, c):
        lv = self.live()
        return bool(lv) and all(self._in_rect(h, c) for h in lv)

    def final_candidates(self):
        """Cells that may be the +1 final cell, best first."""
        if self.final_cell is not None:
            return [self.final_cell]
        lv = self.live()
        out = []
        for h in lv:
            for f, far in zip(h['finals'], h['far']):
                if f not in out and not self.possible(f):
                    out.append(f)
        # an OPEN border from the far-row cell into the candidate first, then candidates seen as floor
        def score(f):
            s = 0.0
            for h in lv:
                if f in h['finals']:
                    far = h['far'][h['finals'].index(f)]
                    e = self.edges.get(edge_key(far, f))
                    if e is not None:
                        s += {OPEN: 3.0, CURTAIN: 1.0, UNKNOWN: 0.5, WALL: -3.0}[e.state]
            s += min(1.0, self.cells[f].floor_n / 200.0) if f in self.cells else 0.0
            return -s
        out.sort(key=score)
        return out

    def bounds_known(self):
        lv = self.live()
        return bool(lv) and len({h['rect'] for h in lv}) == 1

    def rect_union(self):
        lv = self.live()
        if not lv:
            return None
        return (min(h['rect'][0] for h in lv), max(h['rect'][1] for h in lv),
                min(h['rect'][2] for h in lv), max(h['rect'][3] for h in lv))

    def _update_hyps(self, t):
        if self.no_bounds:
            return
        vis = [c for c, v in self.cells.items() if v.visited]
        for h in self.hyps:
            if not h['alive']:
                continue
            outside = [c for c in vis if not self._in_rect(h, c)]
            if any(c not in h['finals'] for c in outside) or len(outside) > 1:
                h['alive'] = False
        for k, e in self.edges.items():
            if e.state != OPEN or k in self.penalised or (e.mp < self.c['hyp_open_min'] and not e.traversed):
                continue
            a, b = edge_cells(k)
            for h in self.hyps:
                if h['alive'] and self._in_rect(h, a) != self._in_rect(h, b):
                    out = b if self._in_rect(h, a) else a
                    if out not in h['finals']:
                        h['pen'] += 1
            self.penalised.add(k)
        if not any(h['alive'] for h in self.hyps):
            self.no_bounds = True
            self.events.append({'t': round(t, 2), 'arena': 'bounds lost (no 3x5 placement fits the visited cells) - '
                                                         'exploring without bounds'})
        # the final cell: a visited cell outside every live rectangle
        lv = self.live()
        if self.final_cell is None and lv:
            for c in vis:
                if not any(self._in_rect(h, c) for h in lv) and any(c in h['finals'] for h in lv):
                    self.final_cell = c
                    self.events.append({'t': round(t, 2), 'arena': 'final cell found %s' % (c,)})
                    break

    # ------------------------------------------------------------------ dog position
    def on_pose(self, x, y, t):
        """Mark visits and traversed borders. Returns the cell entered for the first time, or None."""
        c = self.cell_of(x, y)
        cx, cy = self.centre(c)
        new = None
        if math.hypot(x - cx, y - cy) <= self.c['visit_r_m']:
            cl = self.cell(c)
            if not cl.visited:
                cl.visited, cl.visit_t = True, t
                self.order.append(c)
                new = c
                self._update_hyps(t)
        # crossed a border: counted only when the dog is well inside the new cell (no jitter on the line)
        if math.hypot(x - cx, y - cy) <= 0.40 * self.L:
            prev = self._last_cell
            if prev is not None and prev != c:
                k = edge_key(prev, c)
                if k is not None:
                    e = self.edge(k)
                    if not e.traversed:
                        e.traversed = True
                        self._classify(e)
                        self._update_hyps(t)
            self._last_cell = c
        return new

    _last_cell = None

    def force_visit(self, c, t):
        """Count a cell as visited without its centre being reached (a prop at the centre, the stairs)."""
        cl = self.cell(c)
        if not cl.visited:
            cl.visited, cl.visit_t = True, t
            self.order.append(tuple(c))
            self._update_hyps(t)

    def weight(self, c):
        """Share of the live placements that contain c (1.0 without bounds)."""
        lv = self.live()
        if not lv:
            return 1.0
        return sum(1 for h in lv if self._in_rect(h, c)) / float(len(lv))

    # ------------------------------------------------------------------ lidar evidence
    def observe(self, P, S, t):
        """P: (N, 3) points in the arena frame, z = height above the stance floor. S: sensor (x, y, h)."""
        c = self.c
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        if P.shape[0] == 0:
            return
        L = self.L
        sx, sy, sh = float(S[0]), float(S[1]), float(S[2])
        R = c['edge_range_m']
        d2 = (P[:, 0] - sx) ** 2 + (P[:, 1] - sy) ** 2
        P = P[(d2 <= (R + 1.0) ** 2) & (d2 >= 0.35 ** 2)]
        if P.shape[0] > c['edge_max_pts']:
            idx = np.random.default_rng(int(t * 1000) & 0xffff).choice(P.shape[0], c['edge_max_pts'], replace=False)
            P = P[idx]
        bands = ((c['edge_mid_lo'], c['edge_mid_hi']), (c['edge_up_lo'], c['edge_up_hi']))
        tol, marg, cap = c['edge_tol_m'], c['edge_margin_m'], c['edge_cap_per_scan']
        ci, cj = self.cell_of(sx, sy)
        n = int(math.ceil(R / L)) + 1
        for i in range(ci - n, ci + n + 1):
            for j in range(cj - n, cj + n + 1):
                for ax in ('x', 'y'):
                    if ax == 'x':
                        pc, a0, a1 = (i + 0.5) * L, (j - 0.5) * L + marg, (j + 0.5) * L - marg
                        dp, along, ds, s_al = P[:, 0] - pc, P[:, 1], sx - pc, sy
                    else:
                        pc, a0, a1 = (j + 0.5) * L, (i - 0.5) * L + marg, (i + 0.5) * L - marg
                        dp, along, ds, s_al = P[:, 1] - pc, P[:, 0], sy - pc, sx
                    # nearest point of the border segment to the sensor
                    if math.hypot(ds, min(max(s_al, a0), a1) - s_al) > R or abs(ds) < 0.10:
                        continue
                    hit_m = (np.abs(dp) < tol) & (along > a0) & (along < a1)
                    cross = (np.sign(dp) != np.sign(ds)) & (np.abs(dp) > 0.10)
                    if cross.any():
                        f = ds / (ds - dp[cross])
                        al_c = s_al + f * (along[cross] - s_al)
                        h_c = sh + f * (P[cross, 2] - sh)
                        in_span = (al_c > a0) & (al_c < a1)
                    else:
                        h_c = np.zeros(0)
                        in_span = np.zeros(0, dtype=bool)
                    counts = []
                    for lo, hi in bands:
                        hh = P[hit_m, 2]
                        nh = int(np.count_nonzero((hh >= lo) & (hh <= hi)))
                        np_ = int(np.count_nonzero(in_span & (h_c >= lo) & (h_c <= hi)))
                        counts += [min(cap, nh), min(cap, np_)]
                    if sum(counts) == 0:
                        continue
                    e = self.edge((i, j, ax))
                    e.mh += counts[0]
                    e.mp += counts[1]
                    e.uh += counts[2]
                    e.up += counts[3]
                    tot = e.mh + e.mp + e.uh + e.up
                    if tot > c['edge_count_max']:          # keep it responsive: old evidence fades
                        k = c['edge_count_max'] / tot
                        e.mh, e.mp, e.uh, e.up = e.mh * k, e.mp * k, e.uh * k, e.up * k
                    self._classify(e)
        self._observe_cells(P, sx, sy)
        self._update_hyps(t)

    def _observe_cells(self, P, sx, sy):
        L = self.L
        ij = np.round(P[:, :2] / L).astype(np.int64)
        # interior only (not the border band)
        off = P[:, :2] - ij * L
        inner = (np.abs(off[:, 0]) < 0.38 * L) & (np.abs(off[:, 1]) < 0.38 * L)
        h = P[:, 2]
        # terrain only from near returns: 2 deg of walking wobble is +-7 cm of height at 2 m
        near = (P[:, 0] - sx) ** 2 + (P[:, 1] - sy) ** 2 <= self.c['terrain_range_m'] ** 2
        for name, m in (('floor_n', inner & (np.abs(h) < 0.10)),
                        ('flat_n', inner & near & (np.abs(h) < 0.04)),
                        ('raised_n', inner & near & (h >= 0.06) & (h <= 0.22)),
                        ('high_n', inner & near & (h >= 0.25) & (h <= 0.60))):
            if not m.any():
                continue
            keys, cnt = np.unique(ij[m], axis=0, return_counts=True)
            for (a, b), n in zip(keys.tolist(), cnt.tolist()):
                cl = self.cell((a, b))
                setattr(cl, name, min(5000.0, getattr(cl, name) + n))

    def _classify(self, e):
        c = self.c
        prev = e.state
        if e.traversed:
            e.state = CURTAIN if (e.was_curtain or e.result == 'passed') else OPEN
            return
        if e.result == 'rigid':
            e.state = WALL
            return
        nm = e.mh + e.mp
        nu = e.uh + e.up
        st = UNKNOWN
        if nm >= c['edge_n_min']:
            rb = e.mh / nm
            if rb >= c['edge_wall_ratio']:
                st = WALL
            elif rb <= c['edge_open_ratio']:
                if nu >= c['edge_n_min'] and e.uh / nu >= c['edge_curtain_up_ratio']:
                    st = CURTAIN                      # open below, cloth above: a hanging curtain
                else:
                    st = OPEN
        if st == WALL and e.cue >= c['cur_cue_min'] and e.result != 'camera_no':
            st = CURTAIN                              # wall-like, but the camera says curtain
        if st != WALL and st != UNKNOWN and e.result == 'camera_yes':
            st = CURTAIN
        if st == CURTAIN:
            e.was_curtain = True
        e.state = st
        return prev != st

    def add_curtain_cue(self, x, y, bearing_abs, t, dist_max=2.5):
        """A persistent camera curtain detection along bearing_abs (arena yaw) from (x, y): the first border
        the ray crosses that is not OPEN gets a cue - only if the ray meets it fairly square (within
        cur_cue_angle_deg of its normal) and away from its ends (a grazing ray past a curtain's edge would
        credit the panel next to it; sim 2026-11-26: a pushed wall)."""
        L = self.L
        cb, sb = math.cos(bearing_abs), math.sin(bearing_abs)
        prev = self.cell_of(x, y)
        for d in np.arange(0.1, dist_max, 0.04):
            b = self.cell_of(x + d * cb, y + d * sb)
            if b == prev:
                continue
            a, prev = prev, b
            k = edge_key(a, b)
            if k is None:
                return None                              # through a corner: ambiguous
            e = self.edge(k)
            if e.state == OPEN or e.traversed:
                continue
            i, j, ax = k
            if ax == 'x':
                along, centre, normal_c = y + d * sb, j * L, abs(cb)
            else:
                along, centre, normal_c = x + d * cb, i * L, abs(sb)
            if abs(along - centre) > 0.5 * L - 0.20:
                return None                              # near a post: could be either border
            if normal_c < math.cos(math.radians(self.c['cur_cue_angle_deg'])):
                return None                              # grazing
            e.cue += 1.0
            self._classify(e)
            return k
        return None

    # ------------------------------------------------------------------ terrain / labels
    def terrain(self, c):
        cl = self.cells.get(c)
        if cl is None:
            return ''
        if cl.stairs:
            return 'stairs'
        n = cl.flat_n + cl.raised_n + cl.high_n
        if n < 40:
            return ''
        if cl.high_n / n > 0.30:
            return 'elevated'
        if cl.raised_n / n > 0.30:
            return 'uneven'
        return 'flat'

    def mark_stairs(self, c, src):
        """Mark a stairs cell. The stairs are ONE straight run (booklet: 2-3 steps up, platform, 2-3 down): a new
        cell must be in line with and next to the cells already marked (else it is refused)."""
        c = (int(c[0]), int(c[1]))
        cl = self.cell(c)
        have = [k for k, v in self.cells.items() if v.stairs]
        if have and not cl.stairs and src != 'crossed':
            nb = any(abs(c[0] - h[0]) + abs(c[1] - h[1]) == 1 for h in have)
            line = all(h[0] == c[0] for h in have) or all(h[1] == c[1] for h in have)
            if not (nb and line):
                self.events.append({'arena': 'stairs mark refused for %s (%s): not in line with %s' % (c, src, have)})
                return False
        if not cl.stairs:
            cl.stairs, cl.stairs_src = True, src
            return True
        return False

    def is_stairs(self, c):
        return c in self.cells and self.cells[c].stairs

    def set_label(self, c, key, value):
        self.cell(c).labels[key] = value

    # ------------------------------------------------------------------ edge states for the planner
    def state(self, a, b):
        k = edge_key(a, b)
        e = self.edges.get(k)
        return UNKNOWN if e is None else e.state

    def block(self, a, b, t, secs):
        e = self.edge_between(a, b)
        if e is not None:
            e.blocked_until = t + secs
            e.blocked_n += 1

    def is_blocked(self, a, b, t):
        e = self.edges.get(edge_key(a, b))
        return e is not None and t < e.blocked_until

    # ------------------------------------------------------------------ viewer / logs
    def snapshot(self, t=0.0):
        rect = self.rect_union()
        cells = []
        keys = set(self.cells)
        if rect is not None:
            for i in range(rect[0] - 1, rect[1] + 2):
                for j in range(rect[2] - 1, rect[3] + 2):
                    keys.add((i, j))
        lv = self.live()
        for c in sorted(keys):
            cl = self.cells.get(c, Cell())
            st = 'visited' if cl.visited else ('seen' if cl.floor_n >= self.c['seen_min_n'] else 'unknown')
            cells.append({'i': c[0], 'j': c[1], 'state': st, 'terrain': self.terrain(c),
                          'in': (self.certain(c) and 2) or (self.possible(c) and 1) or 0,
                          'final': c == self.final_cell or (c in self.final_candidates()[:1] and self.bounds_known()),
                          'labels': dict(cl.labels), 'order': self.order.index(c) if c in self.order else None})
        edges = []
        for k, e in sorted(self.edges.items()):
            if e.state == UNKNOWN and e.mh + e.mp + e.uh + e.up < 3:
                continue
            edges.append({'k': [k[0], k[1], k[2]], 'state': e.state, 'trav': e.traversed,
                          'res': e.result, 'blocked': t < e.blocked_until,
                          'ev': [round(e.mh), round(e.mp), round(e.uh), round(e.up)], 'cue': round(e.cue, 1)})
        return {'L': self.L, 'cells': cells, 'edges': edges,
                'bounds': {'known': self.bounds_known(), 'lost': self.no_bounds,
                           'rects': [list(h['rect']) for h in lv],
                           'finals': [list(f) for f in self.final_candidates()[:3]]},
                'final': list(self.final_cell) if self.final_cell else None}
