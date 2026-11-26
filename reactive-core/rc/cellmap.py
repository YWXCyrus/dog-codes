"""Cell map and lattice localisation for nav_mode cells (2026-11-26).

The arena (booklet v6.4 E, H) is 3 x 5 cells plus one final cell, cell centres 1.02 m apart (1.00 m
wall sections + 0.02 m posts). Walls stand on the cell boundaries. Arena frame (the booklet's
reporting frame): origin = centre of the start cell, +x = the dog's start heading, +y = left.

  CellMap    the dog's notebook, one entry per cell: visited or not, and for each of its 4 sides
             wall / open / unknown (read at a stop near the cell centre - map by cell, not as it
             goes). The arena size bounds where it may go: the start cell is on the edge of a
             3 x 5 rectangle ('first row'), every visited cell rules placements out; one visited
             cell outside is allowed (the final cell). frontier_path() = nearest cell still to see.
  wall_lines / lattice_errors
             walls lie on the lines x, y = (k + 0.5) * pitch: a straight wall line seen at x_meas
             means the position estimate is off by (wall centre line - nearest boundary).

Arena directions d: 0 = +x, 1 = +y, 2 = -x, 3 = -y.
"""
import collections
import math

import numpy as np

STEP = ((1, 0), (0, 1), (-1, 0), (0, -1))


def nb(cell, d):
    return cell[0] + STEP[d][0], cell[1] + STEP[d][1]


def placements(nx, ny):
    """Every nx x ny (or ny x nx) rectangle of cells that holds (0, 0) on its edge."""
    out = set()
    for a, b in ((nx, ny), (ny, nx)):
        for x0 in range(-(a - 1), 1):
            for y0 in range(-(b - 1), 1):
                if x0 in (0, -(a - 1)) or y0 in (0, -(b - 1)):
                    out.add((x0, y0, a, b))
    return sorted(out)


def _inside(p, cell):
    x0, y0, a, b = p
    return x0 <= cell[0] < x0 + a and y0 <= cell[1] < y0 + b


class CellMap:

    def __init__(self, pitch=1.02, arena=(3, 5)):
        self.P = float(pitch)
        self.vis = set()
        self.edges = {}                   # (cell, d) -> True open / False wall
        self.places = placements(int(arena[0]), int(arena[1]))
        self._alive = list(self.places)

    def cell_of(self, x, y):
        return int(round(x / self.P)), int(round(y / self.P))

    def centre(self, cell):
        return cell[0] * self.P, cell[1] * self.P

    def edge(self, cell, d):
        return self.edges.get((cell, d))

    def set_edge(self, cell, d, is_open):
        self.edges[(cell, d)] = bool(is_open)
        self.edges[(nb(cell, d), (d + 2) % 4)] = bool(is_open)

    def visit(self, cell):
        if cell in self.vis:
            return
        self.vis.add(cell)
        # placements that leave more than one visited cell outside are impossible
        alive = [p for p in self.places if sum(not _inside(p, c) for c in self.vis) <= 1]
        self._alive = alive or self._alive

    def alive(self):
        return self._alive

    def in_bounds(self, cell):
        return any(_inside(p, cell) for p in self._alive)

    def to_see(self, cell):
        """Unvisited and inside the arena."""
        return cell not in self.vis and self.in_bounds(cell)

    def unknown_sides(self, cell):
        """Sides of a visited cell not seen yet that lead to a cell still to see."""
        return [d for d in range(4) if self.edge(cell, d) is None and self.to_see(nb(cell, d))]

    def frontier_path(self, start):
        """Shortest path (list of cells, start first) through known-open sides and visited cells to
        the nearest cell still to see, or to a visited cell with an unknown side. None: all seen."""
        prev = {start: None}
        q = collections.deque([start])
        while q:
            c = q.popleft()
            if c != start and (self.to_see(c) or (c in self.vis and self.unknown_sides(c))):
                path = [c]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            if c != start and c not in self.vis:
                continue                                        # do not plan through unseen cells
            for d in range(4):
                n = nb(c, d)
                if n in prev or not self.edge(c, d):
                    continue
                if n in self.vis or self.in_bounds(n):
                    prev[n] = c
                    q.append(n)
        return None

    def path_to(self, start, goal):
        """Shortest path (start first) through known-open sides and visited cells to `goal`, which may
        be unvisited or outside the arena. None if there is none."""
        prev = {start: None}
        q = collections.deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                path = [c]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            if c != start and c not in self.vis:
                continue
            for d in range(4):
                n = nb(c, d)
                if n in prev or not self.edge(c, d):
                    continue
                if n in self.vis or n == goal:
                    prev[n] = c
                    q.append(n)
        return None

    def exit_cell(self, view, view_min=1.8):
        """The final cell (booklet H: one cell outside the 3 x 5, the side facing the drone target not
        built), or None when it is not certain:
          1. a visited cell that EVERY remaining placement leaves outside (all agree on it);
          2. else an unvisited cell outside every placement, behind a known-open side of a visited cell,
             through which the dog saw >= view_min free (past the cell: no far wall); the longest view.
        view: (cell, d) -> free distance seen from that cell's centre along d."""
        outs = []
        for p in self._alive:
            o = [c for c in self.vis if not _inside(p, c)]
            outs.append(o[0] if o else None)
        if outs and None not in outs and len(set(outs)) == 1:
            return outs[0]
        cands = []
        for c in self.vis:
            for d in range(4):
                n = nb(c, d)
                v = view.get((c, d), 0.0)
                if self.edge(c, d) and n not in self.vis and not self.in_bounds(n) and v >= view_min:
                    cands.append((-v, n))
        return min(cands)[1] if cands else None

    def summary(self):
        return {'visited': len(self.vis), 'placements': len(self._alive)}


# ------------------------------------------------------------------ lattice localisation
def wall_lines(along, across, bin_m=0.04, min_n=12, min_span=0.35, thin=0.025, fill=0.6, fill_bin=0.05):
    """Straight wall lines perpendicular to an axis: (position across, n points) for groups of points
    sharing `across` (within +-thin of their median) that run CONTINUOUSLY along it for >= min_span
    (>= fill of the fill_bin bins between the ends occupied). Without the continuity test the two side
    walls of a corridor (points at every `across`, but only at two `along` values) looked like a wall
    across it at every position and held the estimate still (sim W3, 2026-11-26)."""
    if across.size < min_n:
        return []
    k = np.floor(across / bin_m).astype(int)
    ks, cnt = np.unique(k, return_counts=True)
    out = []
    for kk in ks[cnt >= min_n // 2]:
        m = np.abs(across - (kk + 0.5) * bin_m) <= 1.5 * bin_m
        if m.sum() < min_n:
            continue
        v = float(np.median(across[m]))
        m = np.abs(across - v) <= thin
        if m.sum() < min_n:
            continue
        a = np.sort(along[m])
        span = a[-1] - a[0]
        if span < min_span:
            continue
        nbins = int(math.ceil(span / fill_bin)) or 1
        occ = len(np.unique(np.minimum(((a - a[0]) / fill_bin).astype(int), nbins - 1)))
        if occ < fill * nbins:
            continue
        out.append((float(np.median(across[m])), int(m.sum())))
    out.sort()
    merged = []
    for v, n in out:
        if merged and abs(v - merged[-1][0]) < 0.08:
            if n > merged[-1][1]:
                merged[-1] = (v, n)
        else:
            merged.append((v, n))
    return merged


def lattice_errors(along, across, here, pitch, half_t, gate):
    """Position errors along one axis from the wall lines seen: wall face -> centre line (half_t
    further from the dog) -> nearest boundary (k + 0.5) * pitch. Lines further than `gate` from a
    boundary (props, stairs, a curtain off the grid) are not used."""
    errs = []
    for v, _ in wall_lines(along, across):
        face = v + (half_t if v > here else -half_t)
        bnd = (math.floor(face / pitch - 0.5) + 0.5) * pitch
        if abs(face - bnd - pitch) < abs(face - bnd):
            bnd += pitch
        e = face - bnd
        if abs(e) < gate:
            errs.append(e)
    return errs


def side_cover(X, Y, cell, d, pitch, band=0.10, end=0.12, bin_m=0.10):
    """Share (0..1) of side d of `cell` covered by wall points (arena frame): points within `band` of
    the boundary line, the `end` m next to each post left out, counted in bin_m bins."""
    i, j = cell
    if d in (0, 2):
        line = (i + (0.5 if d == 0 else -0.5)) * pitch
        a0, a1 = (j - 0.5) * pitch + end, (j + 0.5) * pitch - end
        m = (np.abs(X - line) < band) & (Y > a0) & (Y < a1)
        along = Y[m]
    else:
        line = (j + (0.5 if d == 1 else -0.5)) * pitch
        a0, a1 = (i - 0.5) * pitch + end, (i + 0.5) * pitch - end
        m = (np.abs(Y - line) < band) & (X > a0) & (X < a1)
        along = X[m]
    nbins = max(1, int(round((a1 - a0) / bin_m)))
    if along.size == 0:
        return 0.0
    idx = np.clip(((along - a0) / bin_m).astype(int), 0, nbins - 1)
    return float(len(np.unique(idx))) / nbins
