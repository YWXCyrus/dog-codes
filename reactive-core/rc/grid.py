"""1 m grid memory fixed to the start pose (spec section 8).

Grid frame: start cell (0,0), +y = start heading ("forward", N), +x = right (E). Odometry poses
(odom frame) are converted with the start pose. The grid only breaks ties between safe lanes.
"""
import math

from rc import geometry as G

DIRS = ['N', 'E', 'S', 'W']
STEP = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}


class Grid:

    def __init__(self, cell_m=1.0):
        self.cell_m = cell_m
        self.origin = None
        self.visited = set()
        self.dead = set()          # (cx, cy, dir)
        self.current = None

    def set_origin(self, pose):
        self.origin = tuple(pose)

    def clear(self):
        self.visited.clear()
        self.dead.clear()
        self.current = None

    def to_grid(self, pose):
        """(gx, gy) in metres: gx = right of the start heading, gy = along it."""
        ox, oy, oyaw = self.origin
        dx, dy = pose[0] - ox, pose[1] - oy
        c, s = math.cos(oyaw), math.sin(oyaw)
        fwd = c * dx + s * dy
        left = -s * dx + c * dy
        return -left, fwd

    def cell(self, pose):
        gx, gy = self.to_grid(pose)
        return int(round(gx / self.cell_m)), int(round(gy / self.cell_m))

    def update(self, pose):
        """Returns the cell when a cell is entered for the first time, else None."""
        if self.origin is None:
            return None
        c = self.cell(pose)
        self.current = c
        if c in self.visited:
            return None
        self.visited.add(c)
        return c

    def heading_dir(self, yaw):
        """Odom-frame yaw -> N/E/S/W of the grid (N = start heading)."""
        rel = G.wrap(yaw - self.origin[2])         # positive = to the left of the start heading
        k = int(round(-rel / (math.pi / 2))) % 4  # 0 N, 1 E (right), 2 S, 3 W
        return DIRS[k]

    def grid_yaw(self, yaw):
        """Heading in the grid frame, radians, 0 = N, positive = towards E (clockwise)."""
        return G.wrap(-(yaw - self.origin[2]))

    @staticmethod
    def neighbour(cell, d):
        sx, sy = STEP[d]
        return cell[0] + sx, cell[1] + sy

    def mark_dead_end(self, cell, d):
        self.dead.add((cell[0], cell[1], d))

    def is_dead_end(self, cell, d):
        return (cell[0], cell[1], d) in self.dead

    def all_marked(self, cell, dirs):
        return all(self.is_dead_end(cell, d) for d in dirs)

    def snapshot(self):
        return {'cell_m': self.cell_m,
                'visited': sorted([list(c) for c in self.visited]),
                'dead_ends': sorted([list(d) for d in self.dead]),
                'current': list(self.current) if self.current else None}
