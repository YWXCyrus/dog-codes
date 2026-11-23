"""Worlds made of boxes standing on a flat floor (z = 0)."""
import math
from dataclasses import dataclass

import numpy as np

SOLID = ('wall', 'low_back', 'block')   # never climbed, never stood on
CLOTH = ('curtain', 'stiff_curtain')     # lidar: folds, flutter, see-through (sim/lidar.py)
PASSABLE = ('curtain',)                  # the dog walks through (cloth); 'stiff_curtain' is a rigid decoy


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

    def curtain(self, x0, y0, x1, y1, z0=0.45, z1=1.0, stiff=False):
        """Cloth hanging from a bar between (x0, y0) and (x1, y1), hem z0 above the floor (booklet v6.3
        D.4 photo: bar ~1.0 m, hem ~0.45 m; z0=0.03 = a floor-length curtain)."""
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        L = math.hypot(x1 - x0, y1 - y0)
        return self.add(Box(cx, cy, L, 0.02, z0, z1, math.atan2(y1 - y0, x1 - x0),
                            'stiff_curtain' if stiff else 'curtain'))

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
            if b.name in SOLID or b.name in CLOTH:
                continue
            if b.z1 <= current + max_step + 1e-9 and b.contains_xy(x, y):
                best = max(best, b.z1)
        return best

    def blocking(self, xs, ys, support, max_step, body_top=0.55):
        """Boxes that the footprint sample points (xs, ys) would hit at this support height."""
        hits = []
        for b in self.boxes:
            if b.name in PASSABLE:
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
