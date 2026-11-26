"""Grid snap + dead reckoning (rc/localize.py)."""
import math

import numpy as np

from rc import config
from rc.localize import GridLocalizer

CFG = config.load()
L = CFG['arena_cell_m']
FLOOR = -0.32


def wall_pts(x0, y0, x1, y1, step=0.02):
    n = int(math.hypot(x1 - x0, y1 - y0) / step)
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    zs = np.arange(FLOOR + 0.25, FLOOR + 0.6, 0.04)
    X = np.repeat(xs, zs.size)
    Y = np.repeat(ys, zs.size)
    Z = np.tile(zs, xs.size)
    return np.stack([X, Y, Z], axis=1)


def room_scan(true_pose):
    """Cell (0,0) with panels behind and on both sides, the border ahead open: the scan in the body
    frame of the dog at true_pose (arena frame)."""
    h = L / 2
    W = np.vstack([wall_pts(-h, -h, -h, h), wall_pts(-h, h, 1.5 * L, h), wall_pts(-h, -h, 1.5 * L, -h),
                   wall_pts(1.5 * L, -h, 1.5 * L, h)])
    x, y, yaw = true_pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = W[:, 0] - x, W[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy, W[:, 2]], axis=1)


def test_snap_pulls_a_wrong_position_and_heading_back_onto_the_panels():
    loc = GridLocalizer(CFG)
    loc.start(0.0)
    true = (0.10, -0.05, math.radians(3.0))
    loc.x, loc.y = 0.0, 0.10                       # estimate 0.10 m / 0.15 m / 3 deg off
    t = 0.0
    for _ in range(40):
        t += 1.0
        loc.update(room_scan(true), FLOOR, 0.0, t)
    x, y, yaw = loc.pose(0.0)
    assert abs(x - true[0]) < 0.03 and abs(y - true[1]) < 0.03
    assert abs(math.degrees(yaw - true[2])) < 1.0


def test_two_side_walls_alone_do_not_pull_along_the_corridor():
    loc = GridLocalizer(CFG)
    loc.start(0.0)
    h = L / 2
    W = np.vstack([wall_pts(-2, h, 2, h), wall_pts(-2, -h, 2, -h)])       # a corridor along X, nothing across
    for k in range(20):
        loc.update(W, FLOOR, 0.0, float(k + 1))
    assert abs(loc.x) < 1e-9                       # no peak across X: x untouched


def test_dead_reckoning_uses_body_steps_so_a_drifting_odometry_heading_does_not_bend_the_path():
    loc = GridLocalizer(CFG)
    loc.start(0.0)
    od = 0.0
    ox = oy = 0.0
    loc.feed('leg', (ox, oy, od), 0.0)
    for k in range(100):                           # walk 2 m straight; the odometry's own yaw drifts 0.5 rad
        od += 0.005
        ox += 0.02 * math.cos(od)
        oy += 0.02 * math.sin(od)
        loc.feed('leg', (ox, oy, od), 0.0)        # the gyro says: still heading 0
    x, y, _ = loc.pose(0.0)
    assert abs(x - 2.0) < 0.02 and abs(y) < 0.02


def test_source_switch_has_no_jump():
    loc = GridLocalizer(CFG)
    loc.start(0.0)
    loc.feed('leg', (5.0, 5.0, 1.0), 0.0)
    loc.feed('leg', (5.0 + 0.1 * math.cos(1.0), 5.0 + 0.1 * math.sin(1.0), 1.0), 0.0)
    before = loc.pose(0.0)
    loc.feed('lio', (100.0, -3.0, 2.0), 0.0)       # another frame entirely
    assert loc.pose(0.0) == before
    loc.feed('lio', (100.0 + 0.1 * math.cos(2.0), -3.0 + 0.1 * math.sin(2.0), 2.0), 0.0)
    assert abs(loc.pose(0.0)[0] - before[0] - 0.1) < 1e-6


def test_frozen_odometry_yaw_switches_to_forward_steps_along_the_gyro_heading():
    loc = GridLocalizer(CFG)
    loc.start(0.0)
    h = 0.0
    loc.feed('leg', (0.0, 0.0, 0.0), h, 0.0)
    for k in range(80):                            # turn on the spot 160 deg: the odometry yaw stays 0
        h += math.radians(2.0)
        loc.feed('leg', (0.0, 0.0, 0.0), h, 0.0)
    assert loc.step_mode == 'forward' and loc.events
    x0, y0, _ = loc.pose(h)
    ox = 0.0
    for k in range(50):                            # walk 1 m: the odometry x grows (its frozen frame)
        ox += 0.02
        loc.feed('leg', (ox, 0.0, 0.0), h, 0.2)
    x, y, _ = loc.pose(h)
    assert abs(math.hypot(x - x0, y - y0) - 1.0) < 0.02
    assert abs(math.atan2(y - y0, x - x0) - h) < 0.02         # along the gyro heading
