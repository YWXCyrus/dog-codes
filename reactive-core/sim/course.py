"""Course-shaped worlds (2026-11-25): narrow channels on a ~1 m grid, like the SUTD arena.

The default sim arena (world.arena_default) is an open 8 x 6 m room with a 2 m start pit; none of
its tests has a 0.9-1.0 m corridor with a corner, which is where the course runs failed. Every
world here returns a CourseWorld: the boxes, the dog's start pose and a success test.

Start: the dog stands at (0, 0) facing +x unless stated. Walls are 0.10 m thick and 0.6 m high,
so a 1.02 m cell leaves a 0.92 m free corridor.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from sim.world import Box, World

PITCH = 1.02
WALL_T = 0.10


@dataclass
class CourseWorld:
    world: World
    name: str
    start: tuple = (0.0, 0.0, 0.0)
    kind: str = 'corridor'            # corridor | maze | safety | regression
    goal: object = None               # corridor: f(x, y) -> True once the goal is reached
    xb: list = field(default_factory=list)     # maze: cell boundary x coordinates (maze frame)
    yb: list = field(default_factory=list)
    rot: float = 0.0                  # maze frame -> world rotation (about the origin)
    cells: int = 0                    # maze: reachable cells
    dog_kwargs: dict = field(default_factory=dict)
    T: float = 61.0


def _seg(w, x0, y0, x1, y1, h=0.6, t=WALL_T, rot=0.0, name='wall'):
    """Wall from (x0, y0) to (x1, y1), extended by t/2 at both ends (closed corners), rotated."""
    L = math.hypot(x1 - x0, y1 - y0) + t
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    c, s = math.cos(rot), math.sin(rot)
    w.add(Box(c * cx - s * cy, s * cx + c * cy, L, t, 0.0, h, math.atan2(y1 - y0, x1 - x0) + rot, name))


# ------------------------------------------------------------------ corridor worlds (W1-W5)
def l_corner(turn='left', run=3.0, W=0.92):
    """W1/W2: corridor along +x from the start, 90-degree turn at the end; goal = 0.5 m into the
    side corridor."""
    s = 1 if turn == 'left' else -1
    h = W / 2.0
    E = run + h                                   # end wall face at x = run + W/2 (+ half thickness)
    w = World()
    o = h + WALL_T / 2.0                          # wall centre line offset
    _seg(w, -0.8, -s * o, E + WALL_T / 2, -s * o)              # outer wall
    _seg(w, E + WALL_T / 2, -s * o, E + WALL_T / 2, s * 4.0)   # end wall, continues as side corridor wall
    _seg(w, -0.8, s * o, run - h - WALL_T / 2, s * o)          # inner wall, ends at the opening
    _seg(w, run - h - WALL_T / 2, s * o, run - h - WALL_T / 2, s * 4.0)
    _seg(w, -0.8, -s * o, -0.8, s * o)                          # behind the start
    _seg(w, run - h - WALL_T / 2, s * 4.0, E + WALL_T / 2, s * 4.0)
    return CourseWorld(w, 'W%d_corner_%s' % (1 if turn == 'left' else 2, turn),
                       goal=lambda x, y: s * y > h + 0.5 and abs(x - run) < h)


def t_junction(run=3.0, W=0.92):
    """W3: corridor along +x ending in a T (branches to +y and -y); goal = 0.5 m into either branch."""
    h = W / 2.0
    o = h + WALL_T / 2.0
    w = World()
    xe = run + h + WALL_T / 2.0
    _seg(w, -0.8, o, run - h - WALL_T / 2, o)
    _seg(w, -0.8, -o, run - h - WALL_T / 2, -o)
    _seg(w, -0.8, -o, -0.8, o)
    _seg(w, xe, -4.0, xe, 4.0)                                  # top of the T
    for s in (1, -1):
        _seg(w, run - h - WALL_T / 2, s * o, run - h - WALL_T / 2, s * 4.0)
        _seg(w, run - h - WALL_T / 2, s * 4.0, xe, s * 4.0)
    return CourseWorld(w, 'W3_t_junction', goal=lambda x, y: abs(y) > h + 0.5 and abs(x - run) < h)


def dead_end(ahead=2.0, behind=3.0, W=0.92):
    """W4: the dog faces a dead end `ahead` m away; the only way out is behind it, where the
    corridor turns left after `behind` m. Goal = round that far corner (rear is blind at the start)."""
    h = W / 2.0
    o = h + WALL_T / 2.0
    w = World()
    xd = ahead + h + WALL_T / 2.0                 # dead-end wall
    xb = -behind                                  # corner centre behind the start
    _seg(w, xd, -o, xd, o)
    _seg(w, xb - h - WALL_T / 2, -o, xd, -o)      # right wall (when facing +x), full length
    _seg(w, xb + h + WALL_T / 2, o, xd, o)        # left wall, open at the far corner
    _seg(w, xb - h - WALL_T / 2, -o, xb - h - WALL_T / 2, 4.0)
    _seg(w, xb + h + WALL_T / 2, o, xb + h + WALL_T / 2, 4.0)
    _seg(w, xb - h - WALL_T / 2, 4.0, xb + h + WALL_T / 2, 4.0)
    return CourseWorld(w, 'W4_dead_end', goal=lambda x, y: y > h + 0.5 and abs(x - xb) < h)


def start_bay(cylinder=True, W=0.92):
    """W5: the start bay - one cell wide, a 20 cm cylinder (modelled as a 0.20 x 0.20 m bar)
    lying across it 0.75 m ahead, the bay turns left 2 cells ahead. The real dog crossed a 20 cm
    slab in RUG general, so this world lets the model step 0.22 m."""
    cw = l_corner('left', run=2 * PITCH, W=W)
    if cylinder:
        cw.world.add(Box(0.75, 0.0, 0.20, W, 0.0, 0.20, 0.0, 'cylinder'))
    cw.name = 'W5_start_bay'
    cw.dog_kwargs = {'step_rug': 0.22}
    return cw


def slab_wall(W=0.92):
    """W5b (safety, the 24 Nov fall): a 20 cm slab 0.5 m deep with a wall right behind it in a
    dead-end bay. Expected: no on-the-spot turn while tilted (HOLD is a pass)."""
    h = W / 2.0
    o = h + WALL_T / 2.0
    w = World()
    xs = 1.0                                       # slab front edge
    xd = xs + 0.5 + WALL_T / 2.0
    _seg(w, -0.8, o, xd, o)
    _seg(w, -0.8, -o, xd, -o)
    _seg(w, -0.8, -o, -0.8, o)
    _seg(w, xd, -o, xd, o)
    w.add(Box(xs + 0.25, 0.0, 0.5, W, 0.0, 0.20, 0.0, 'slab'))
    return CourseWorld(w, 'W5b_slab_wall', kind='safety', dog_kwargs={'step_rug': 0.22})


# ------------------------------------------------------------------ mazes (W6, W7)
def _maze_walls(n, rng, extra_open, room):
    """Randomised depth-first maze on n x n cells + extra openings (loops) + optional 2x2 room.
    Returns the set of closed interior walls as ('v', i, j) = between (i, j) and (i+1, j),
    ('h', i, j) = between (i, j) and (i, j+1)."""
    walls = {('v', i, j) for i in range(n - 1) for j in range(n)} | {('h', i, j) for i in range(n) for j in range(n - 1)}
    seen, stack = {(0, 0)}, [(0, 0)]
    while stack:
        i, j = stack[-1]
        nb = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= i + di < n and 0 <= j + dj < n and (i + di, j + dj) not in seen]
        if not nb:
            stack.pop()
            continue
        a, b = nb[rng.integers(len(nb))]
        walls.discard(('v', min(i, a), j) if a != i else ('h', i, min(j, b)))
        seen.add((a, b))
        stack.append((a, b))
    inner = sorted(walls)
    for k in rng.permutation(len(inner))[:int(round(extra_open * len(inner)))]:
        walls.discard(inner[k])
    if room:
        ri, rj = int(rng.integers(1, n - 1)), int(rng.integers(1, n - 1))
        for wkey in (('v', ri, rj), ('v', ri, rj + 1), ('h', ri, rj), ('h', ri + 1, rj)):
            walls.discard(wkey)
    return walls


def _build_maze(name, n, xb, yb, walls, rot, kind='maze'):
    w = World()
    # outer boundary
    for (x0, y0, x1, y1) in ((xb[0], yb[0], xb[-1], yb[0]), (xb[-1], yb[0], xb[-1], yb[-1]),
                             (xb[-1], yb[-1], xb[0], yb[-1]), (xb[0], yb[-1], xb[0], yb[0])):
        _seg(w, x0, y0, x1, y1, rot=rot)
    # interior walls, merged along each boundary line (fewer boxes -> faster lidar)
    for i in range(n - 1):              # vertical lines x = xb[i+1]
        j = 0
        while j < n:
            if ('v', i, j) in walls:
                j0 = j
                while j < n and ('v', i, j) in walls:
                    j += 1
                _seg(w, xb[i + 1], yb[j0], xb[i + 1], yb[j], rot=rot)
            else:
                j += 1
    for j in range(n - 1):              # horizontal lines y = yb[j+1]
        i = 0
        while i < n:
            if ('h', i, j) in walls:
                i0 = i
                while i < n and ('h', i, j) in walls:
                    i += 1
                _seg(w, xb[i0], yb[j + 1], xb[i], yb[j + 1], rot=rot)
            else:
                i += 1
    cx, cy = (xb[0] + xb[1]) / 2.0, (yb[0] + yb[1]) / 2.0          # start cell (0, 0) centre
    c, s = math.cos(rot), math.sin(rot)
    start = (c * cx - s * cy, s * cx + c * cy, 0.0)                # facing world +x (maze is rotated)
    return CourseWorld(w, name, start=start, kind=kind, xb=list(xb), yb=list(yb), rot=rot, cells=n * n, T=301.0)


def maze(seed, n=5, extra_open=0.15):
    """W6: random n x n maze on the regular 1.02 m grid with ~15% extra openings (loops)."""
    rng = np.random.default_rng(1000 + seed)
    walls = _maze_walls(n, rng, extra_open, room=False)
    b = [(k - 0.5) * PITCH for k in range(n + 1)]
    return _build_maze('W6_maze_s%d' % seed, n, b, b, walls, 0.0)


def messy(seed, n=4, extra_open=0.15):
    """W7: random n x n maze, NOT a regular grid: cell boundaries jittered (cell pitch 1.0-1.5 m,
    corridors 0.9-1.4 m), one open 2 x 2 room, the whole layout rotated 5-10 deg against the dog."""
    rng = np.random.default_rng(2000 + seed)

    def bounds():
        while True:
            b = np.cumsum(np.concatenate([[0.0], rng.uniform(1.0, 1.5, n)]))
            b = b - (b[0] + b[1]) / 2.0                 # start cell centred on 0
            return list(b)
    xb, yb = bounds(), bounds()
    walls = _maze_walls(n, rng, extra_open, room=True)
    rot = math.radians(rng.uniform(5.0, 10.0)) * (1 if rng.random() < 0.5 else -1)
    return _build_maze('W7_messy_s%d' % seed, n, xb, yb, walls, rot)


def maze_coverage(cw, path, margin=0.15):
    """Cells of a maze world whose interior (boundary - margin) the dog's true path entered."""
    c, s = math.cos(-cw.rot), math.sin(-cw.rot)
    xb, yb = np.asarray(cw.xb), np.asarray(cw.yb)
    got = set()
    for x, y in path:
        mx, my = c * x - s * y, s * x + c * y
        i = int(np.searchsorted(xb, mx)) - 1
        j = int(np.searchsorted(yb, my)) - 1
        if 0 <= i < len(xb) - 1 and 0 <= j < len(yb) - 1:
            if (xb[i] + margin < mx < xb[i + 1] - margin) and (yb[j] + margin < my < yb[j + 1] - margin):
                got.add((i, j))
    return got


def corridor_worlds():
    return [l_corner('left'), l_corner('right'), t_junction(), dead_end(), start_bay(), slab_wall()]


# ------------------------------------------------------------------ booklet arena (A, 2026-11-26)
def _rect_walls(nx, ny, rng, extra_open):
    """Randomised depth-first maze on nx x ny cells + extra openings; closed interior walls as in
    _maze_walls: ('v', i, j) between (i, j) and (i+1, j), ('h', i, j) between (i, j) and (i, j+1)."""
    walls = {('v', i, j) for i in range(nx - 1) for j in range(ny)} | \
            {('h', i, j) for i in range(nx) for j in range(ny - 1)}
    seen, stack = {(0, 0)}, [(0, 0)]
    while stack:
        i, j = stack[-1]
        nb = [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= i + a < nx and 0 <= j + b < ny and (i + a, j + b) not in seen]
        if not nb:
            stack.pop()
            continue
        a, b = nb[rng.integers(len(nb))]
        walls.discard(('v', min(i, a), j) if a != i else ('h', i, min(j, b)))
        seen.add((a, b))
        stack.append((a, b))
    inner = sorted(walls)
    for k in rng.permutation(len(inner))[:int(round(extra_open * len(inner)))]:
        walls.discard(inner[k])
    return walls


def arena35(seed, nx=3, ny=5, extra_open=0.15):
    """A: the booklet arena (v6.4 H) - 3 x 5 cells on the 1.02 m grid plus ONE final cell outside the
    last row, whose far side is open (no wall: line of sight to the drone target). The dog starts at
    the centre of a first-row cell, facing a random axis direction (the booklet: orientation varies)."""
    rng = np.random.default_rng(3000 + seed)
    walls = _rect_walls(nx, ny, rng, extra_open)
    xb = [(k - 0.5) * PITCH for k in range(nx + 1)]
    yb = [(k - 0.5) * PITCH for k in range(ny + 1)]
    fi = int(rng.integers(nx))                                  # final cell column, beyond row ny-1
    w = World()
    # outer boundary, with a gap into the final cell
    _seg(w, xb[0], yb[0], xb[-1], yb[0])
    _seg(w, xb[0], yb[0], xb[0], yb[-1])
    _seg(w, xb[-1], yb[0], xb[-1], yb[-1])
    if fi > 0:
        _seg(w, xb[0], yb[-1], xb[fi], yb[-1])
    if fi < nx - 1:
        _seg(w, xb[fi + 1], yb[-1], xb[-1], yb[-1])
    ytop = yb[-1] + PITCH                                       # final cell: side walls only
    _seg(w, xb[fi], yb[-1], xb[fi], ytop)
    _seg(w, xb[fi + 1], yb[-1], xb[fi + 1], ytop)
    for key in sorted(walls):
        kind, i, j = key
        if kind == 'v':
            _seg(w, xb[i + 1], yb[j], xb[i + 1], yb[j + 1])
        else:
            _seg(w, xb[i], yb[j + 1], xb[i + 1], yb[j + 1])
    si = int(rng.integers(nx))
    yaw = float(rng.integers(4)) * math.pi / 2.0
    cw = CourseWorld(w, 'A_arena35_s%d' % seed, start=((xb[si] + xb[si + 1]) / 2.0, 0.0, yaw), kind='arena',
                     xb=xb, yb=yb, cells=nx * ny, T=241.0)
    cw.final = (xb[fi], yb[-1], xb[fi + 1], ytop)               # final cell box (maze frame)
    return cw


def outside_arena(cw, x, y, margin=0.3):
    """True when (x, y) is more than `margin` outside the 3 x 5 arena and its final cell."""
    in_a = cw.xb[0] - margin <= x <= cw.xb[-1] + margin and cw.yb[0] - margin <= y <= cw.yb[-1] + margin
    fx0, fy0, fx1, fy1 = cw.final
    in_f = fx0 - margin <= x <= fx1 + margin and fy0 - margin <= y <= fy1 + margin
    return not (in_a or in_f)


def arena_to_world(cw, ax, ay):
    """A point of the dog's arena frame (origin = start, +x = start heading) in world coordinates."""
    x0, y0, yaw0 = cw.start
    c, s = math.cos(yaw0), math.sin(yaw0)
    return x0 + c * ax - s * ay, y0 + s * ax + c * ay


def looked_cells(cw, events, pitch=PITCH):
    """Maze cells looked at from the doorway (nav_mode cells 'room_done' events): seen, not entered."""
    got = set()
    for e in events:
        if e.get('cells') == 'room_done':
            got |= maze_coverage(cw, [arena_to_world(cw, e['room'][0] * pitch, e['room'][1] * pitch)], margin=0.0)
    return got
