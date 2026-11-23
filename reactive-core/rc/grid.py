"""Cell grid fixed to the start pose, in the ARENA frame of booklet v6.3 section E:
origin = centre of the start cell, +X = the robot's initial forward direction, +Y = its left at the
start, cell centres 1.02 m apart (1.00 m walls + 0.02 m posts). Cell (i, j) has its centre at
(1.02 i, 1.02 j) - the 'pos' the scoring server expects. Odometry poses (odom frame) are converted
with the start pose. Compass names stay relative to the start: N = start heading (+X), E = its right
(-Y), S = behind (-X), W = its left (+Y). The driver uses the grid to break ties between safe lanes.
"""
import math

from rc import geometry as G

DIRS = ['N', 'E', 'S', 'W']
STEP = {'N': (1, 0), 'E': (0, -1), 'S': (-1, 0), 'W': (0, 1)}      # (di, dj) in the arena frame


class Grid:

    def __init__(self, cell_m=1.02):
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
        """Arena frame (X, Y) in metres: X = along the start heading, Y = to the start's left."""
        ox, oy, oyaw = self.origin
        dx, dy = pose[0] - ox, pose[1] - oy
        c, s = math.cos(oyaw), math.sin(oyaw)
        fwd = c * dx + s * dy
        left = -s * dx + c * dy
        return fwd, left

    def centre(self, cell):
        """Arena-frame centre (X, Y) of cell (i, j) - the 'pos' to report for that cell."""
        return round(cell[0] * self.cell_m, 2), round(cell[1] * self.cell_m, 2)

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
        """Heading for the viewer, which draws X to the right and Y up with 0 = up, clockwise:
        the start heading (+X) is pi/2 (pointing right, as in the booklet's coordinate figure)."""
        return G.wrap(math.pi / 2 - G.wrap(yaw - self.origin[2]))

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
