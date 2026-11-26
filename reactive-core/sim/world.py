"""Worlds made of boxes standing on a flat floor (z = 0)."""
import math
from dataclasses import dataclass

import numpy as np

SOLID = ('wall', 'low_back', 'block')   # never climbed, never stood on
SOFT = ('curtain',)                     # seen by the lidar, the dog walks through (cloth on a bar)


@dataclass
class Box:
    cx: float
    cy: float
    sx: float          # size along the box's own x
    sy: float
    z0: float
    z1: float
    yaw: float = 0.0
    name: str = ''

    def contains_xy(self, x, y):
        """Mask: (x, y) inside the footprint of the box."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = np.asarray(x) - self.cx, np.asarray(y) - self.cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        return (np.abs(u) <= self.sx / 2.0) & (np.abs(v) <= self.sy / 2.0)


class World:

    def __init__(self, boxes=None):
        self.boxes = list(boxes or [])

    def add(self, *b):
        self.boxes.extend(b)
        return self

    # ---- builders ----
    def wall(self, x0, y0, x1, y1, h=0.6, t=0.05, name='wall'):
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        L = math.hypot(x1 - x0, y1 - y0)
        return self.add(Box(cx, cy, L, t, 0.0, h, math.atan2(y1 - y0, x1 - x0), name))

    def board(self, cx, cy, sx, sy, h, name='board'):
        return self.add(Box(cx, cy, sx, sy, 0.0, h, 0.0, name))

    def stairs(self, x0, cy, n_up=3, rise=0.15, tread=0.30, platform=0.8, n_down=3, width=1.0):
        """Along +x from x0: n_up steps, a platform, n_down steps (each step a box from the floor)."""
        x = x0
        for i in range(n_up):
            self.add(Box(x + tread / 2.0, cy, tread, width, 0.0, rise * (i + 1), 0.0, 'step'))
            x += tread
        top = rise * n_up
        self.add(Box(x + platform / 2.0, cy, platform, width, 0.0, top, 0.0, 'platform'))
        x += platform
        for i in range(n_down):
            self.add(Box(x + tread / 2.0, cy, tread, width, 0.0, top - rise * (i + 1), 0.0, 'step'))
            x += tread
        return self

    # ---- queries ----
    def support(self, x, y, current, max_step):
        """Height the dog stands on at (x, y): the highest box top under it that is reachable
        from `current` (<= current + max_step); the floor otherwise. Walls are never stood on."""
        best = 0.0
        for b in self.boxes:
            if b.name in SOLID or b.name in SOFT:
                continue
            if b.z1 <= current + max_step + 1e-9 and b.contains_xy(x, y):
                best = max(best, b.z1)
        return best

    def blocking(self, xs, ys, support, max_step, body_top=0.55):
        """Boxes that the footprint sample points (xs, ys) would hit at this support height."""
        hits = []
        for b in self.boxes:
            if b.name in SOFT:
                continue
            step = 0.02 if b.name in SOLID else max_step
            if b.z1 > support + step + 1e-9 and b.z0 < support + body_top:
                if np.any(b.contains_xy(xs, ys)):
                    hits.append(b)
        return hits


def corridor_world(length=10.0, width=1.4, h=0.6):
    w = World()
    w.wall(-1.0, width / 2, length, width / 2, h)
    w.wall(-1.0, -width / 2, length, -width / 2, h)
    w.wall(-1.0, -width / 2, -1.0, width / 2, h)
    w.wall(length, -width / 2, length, width / 2, h)
    return w


def stairs_world():
    """Corridor 1.4 m wide with stairs (3 up, platform 0.8 m, 3 down) starting 3 m ahead."""
    w = corridor_world(12.0, 1.4)
    w.stairs(3.0, 0.0, 3, 0.15, 0.30, 0.8, 3, 1.4)
    return w


def arena_default():
    """8 x 6 m arena: start pit with low strips, a low wall behind the start (invisible to the
    lidar), a room with a 0.9 m door, an L corridor. Start at (0, 0) facing +x."""
    w = World()
    # outer walls
    w.wall(-1.0, -1.0, 7.0, -1.0)
    w.wall(7.0, -1.0, 7.0, 5.0)
    w.wall(7.0, 5.0, -1.0, 5.0)
    w.wall(-1.0, 5.0, -1.0, -1.0)
    # start pit: low wall behind, side wall on the left, strips ahead
    w.wall(-0.6, -1.0, -0.6, 1.0, h=0.30, name='low_back')
    w.wall(-1.0, 1.0, 1.5, 1.0)
    for k, x in enumerate((0.9, 1.5, 2.1)):
        w.board(x, 0.0, 0.25, 1.9, 0.10, 'strip')
    # room in the top-right with a 0.9 m door
    w.wall(3.5, 2.5, 3.5, 5.0)
    w.wall(3.5, 2.5, 4.8, 2.5)
    w.wall(5.7, 2.5, 7.0, 2.5)
    # a free-standing block
    w.add(Box(5.0, 0.6, 0.6, 0.6, 0.0, 0.6, 0.0, 'block'))
    return w


# ---------------------------------------------------------------------------------------------
# Competition-like arenas (booklet v6.4): 3 x 5 cells of 1.02 m + the final cell, panels on the
# cell borders, start cell (0, 0) in the first row, the dog facing +X (arena frame = world frame).
L_CELL = 1.02


def edge_segment(k, L=L_CELL):
    """Border key (i, j, 'x'|'y') -> wall segment (x0, y0, x1, y1)."""
    i, j, ax = k
    if ax == 'x':
        x = (i + 0.5) * L
        return x, (j - 0.5) * L, x, (j + 0.5) * L
    y = (j + 0.5) * L
    return (i - 0.5) * L, y, (i + 0.5) * L, y


def _ekey(a, b):
    if a[0] != b[0]:
        return (min(a[0], b[0]), a[1], 'x')
    return (a[0], min(a[1], b[1]), 'y')


def _nbrs(c):
    i, j = c
    return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]


def build_arena(rect, final, open_edges, curtains=(), boards=(), stairs=None, wall_h=0.43, curtain_low=0.15,
                floor_curtains=(), props=(), L=L_CELL, t=0.05, curtain_top=0.93):
    """rect = (i0, i1, j0, j1) arena cells; final = the +1 cell (its far side open); open_edges /
    curtains = border keys without panel / with a hanging curtain (open below curtain_low);
    floor_curtains = curtains down to the floor; boards = cells with low strips (uneven terrain);
    stairs = {'cells': [...], 'dir': 'x'|'y', 'sign': +1|-1, 'n': 2, 'rise': 0.12, 'tread': 0.25,
    'platform': 0.5}; props = cells with a 0.4 m box at the centre."""
    i0, i1, j0, j1 = rect
    cells = {(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}
    allc = cells | {final}
    w = World()
    w.layout = {'rect': rect, 'final': final, 'open': set(open_edges), 'curtains': set(curtains),
                'cells': cells}
    done = set()
    soft = set(curtains) | set(floor_curtains)
    for c in allc:
        for n in _nbrs(c):
            k = _ekey(c, n)
            if k in done:
                continue
            done.add(k)
            inside = n in allc
            if inside and (k in open_edges or k in soft):
                if k in soft:
                    x0, y0, x1, y1 = edge_segment(k, L)
                    z0 = 0.02 if k in floor_curtains else curtain_low
                    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                    w.add(Box(cx, cy, L - 0.06, 0.03, z0, curtain_top, math.atan2(y1 - y0, x1 - x0), 'curtain'))
                continue
            if c == final and not inside:
                # the final cell: walls on its sides, its far side (towards the drone target) open
                far = (final[0] + (final[0] - _closest(final, cells)[0]), final[1] + (final[1] - _closest(final, cells)[1]))
                if n == far:
                    continue
            if n == final and c not in cells:
                continue
            x0, y0, x1, y1 = edge_segment(k, L)
            w.wall(x0, y0, x1, y1, h=wall_h, t=t)
    for c in boards:
        cx, cy = c[0] * L, c[1] * L
        for k in range(3):
            w.board(cx - 0.3 + 0.3 * k, cy, 0.12, 0.9, 0.08, 'strip')
    for c in props:
        w.add(Box(c[0] * L, c[1] * L, 0.4, 0.4, 0.0, 0.5, 0.0, 'block'))
    if stairs:
        c0 = stairs['cells'][0]
        n, rise, tread, plat = stairs.get('n', 2), stairs.get('rise', 0.12), stairs.get('tread', 0.25), stairs.get('platform', 0.5)
        sg = stairs.get('sign', 1)
        if stairs['dir'] == 'x':
            x = c0[0] * L - sg * 0.5 * L + sg * 0.20
            pos = lambda u: (x + sg * u, c0[1] * L)
            yaw = 0.0
        else:
            y = c0[1] * L - sg * 0.5 * L + sg * 0.20
            pos = lambda u: (c0[0] * L, y + sg * u)
            yaw = math.pi / 2
        u = 0.0
        width = 0.9
        for k in range(n):
            px, py = pos(u + tread / 2.0)
            w.add(Box(px, py, tread, width, 0.0, rise * (k + 1), yaw, 'step'))
            u += tread
        px, py = pos(u + plat / 2.0)
        w.add(Box(px, py, plat, width, 0.0, rise * n, yaw, 'platform'))
        u += plat
        for k in range(n):
            px, py = pos(u + tread / 2.0)
            w.add(Box(px, py, tread, width, 0.0, rise * (n - k - 1), yaw, 'step'))
            u += tread
    return w


def _closest(final, cells):
    for n in _nbrs(final):
        if n in cells:
            return n
    return final


def random_layout(seed, axis='x', sign=1, offset=0, loops=0.25, n_long=5, n_short=3, curtain=False,
                  final_col=None):
    """A random perfect maze on the 3 x 5 cells (+ a few extra openings) with the start cell (0, 0)
    in the first row. Returns the kwargs of build_arena."""
    import random
    rnd = random.Random(seed)
    if axis == 'x':
        i0, i1 = (0, n_long - 1) if sign > 0 else (-(n_long - 1), 0)
        j0, j1 = offset, offset + n_short - 1
    else:
        j0, j1 = (0, n_long - 1) if sign > 0 else (-(n_long - 1), 0)
        i0, i1 = offset, offset + n_short - 1
    cells = [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    cs = set(cells)
    seen, stack, opened = {(0, 0)}, [(0, 0)], set()
    while stack:
        c = stack[-1]
        nb = [n for n in _nbrs(c) if n in cs and n not in seen]
        if not nb:
            stack.pop()
            continue
        n = rnd.choice(nb)
        opened.add(_ekey(c, n))
        seen.add(n)
        stack.append(n)
    for c in cells:
        for n in _nbrs(c):
            if n in cs and rnd.random() < loops / 2:
                opened.add(_ekey(c, n))
    if axis == 'x':
        far_i = i1 if sign > 0 else i0
        col = rnd.randint(j0, j1) if final_col is None else final_col
        final = (far_i + sign, col)
        far = (far_i, col)
    else:
        far_j = j1 if sign > 0 else j0
        col = rnd.randint(i0, i1) if final_col is None else final_col
        final = (col, far_j + sign)
        far = (col, far_j)
    opened.add(_ekey(far, final))
    curtains = set()
    if curtain:
        # 'leaf': a dead-end cell (one opening, not the start / the final's neighbour) is reachable only through
        # a curtain; 'alt' (or True): an extra passage through a wall, hung with a curtain (an alternative route)
        deg = {c: [k for k in opened if c in ((k[0], k[1]), (k[0] + (k[2] == 'x'), k[1] + (k[2] == 'y')))] for c in cells}
        if curtain == 'leaf':
            leaves = [c for c in cells if len(deg[c]) == 1 and c != (0, 0) and c != far]
            rnd.shuffle(leaves)
            if leaves:
                k = deg[leaves[0]][0]
                opened.discard(k)
                curtains.add(k)
        else:
            walls = sorted({_ekey(c, n) for c in cells for n in _nbrs(c) if n in cs} - opened)
            walls = [k for k in walls if (0, 0) not in ((k[0], k[1]), (k[0] + (k[2] == 'x'), k[1] + (k[2] == 'y')))]
            if walls:
                curtains.add(rnd.choice(walls))
    return {'rect': (i0, i1, j0, j1), 'final': final, 'open_edges': opened, 'curtains': curtains}


def _connected(cells, opened):
    cs = set(cells)
    seen, stack = {(0, 0)}, [(0, 0)]
    while stack:
        c = stack.pop()
        for n in _nbrs(c):
            if n in cs and n not in seen and _ekey(c, n) in opened:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cs)
