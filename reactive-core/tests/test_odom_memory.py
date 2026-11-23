import math

import numpy as np
import pytest

from rc import config
from rc.memory import ObstacleMemory
from rc.odom import LegOdom

CFG = config.load()


def feed(o, poses, t0=0.0, dt=0.02, cmd=0.15):
    t = t0
    for p in poses:
        o.update(p[0], p[1], p[2], t, cmd)
        t += dt
    return t


def test_odom_travel_and_speed():
    o = LegOdom(CFG)
    t = feed(o, [(0.003 * i, 0.0, 0.0) for i in range(100)])   # 0.15 m/s
    assert o.ok
    assert o.travelled == pytest.approx(0.297, abs=0.01)
    assert o.speed == pytest.approx(0.15, abs=0.02)
    assert o.pose[0] == pytest.approx(0.297)


def test_odom_jump_flags_and_recovers():
    o = LegOdom(CFG)
    t = feed(o, [(0.0, 0.0, 0.0)] * 10)
    o.update(0.5, 0.0, 0.0, t, 0.15)
    assert not o.ok and 'jump' in o.reason
    t += 0.02
    t = feed(o, [(0.5, 0.0, 0.0)] * 200, t0=t)                   # 4 s of good data
    assert not o.ok
    feed(o, [(0.5, 0.0, 0.0)] * 60, t0=t)                        # > 5 s after the jump
    assert o.ok


def test_odom_too_fast_for_command():
    o = LegOdom(CFG)
    feed(o, [(0.02 * i, 0.0, 0.0) for i in range(60)], cmd=0.15)   # 1.0 m/s, commanded 0.15
    assert not o.ok and 'speed' in o.reason


def test_odom_jump_does_not_count_as_travel():
    o = LegOdom(CFG)
    t = feed(o, [(0.0, 0.0, 0.0)] * 5)
    o.update(5.0, 0.0, 0.0, t, 0.1)
    assert o.travelled < 0.01


def test_memory_moves_with_the_dog():
    m = ObstacleMemory(CFG)
    m.add(np.array([[1.0, 0.0]]), (0.0, 0.0, 0.0), 0.0, 0.0)
    q = m.query((1.0, 0.0, math.pi / 2), 1.0, 1.0)
    assert q.shape == (1, 2)
    assert np.allclose(q, [[0.0, 0.0]], atol=0.06)
    q = m.query((0.0, 0.0, math.pi / 2), 1.0, 1.0)              # world (1,0) is now on the right
    assert np.allclose(q, [[0.0, -1.0]], atol=0.06)


def test_memory_expiry_time_and_distance():
    m = ObstacleMemory(CFG)
    m.add(np.array([[1.0, 0.0]]), (0.0, 0.0, 0.0), 0.0, 0.0)
    assert m.query((0, 0, 0), 9.0, 0.0).shape[0] == 1
    assert m.query((0, 0, 0), 11.0, 0.0).shape[0] == 0
    m.add(np.array([[1.0, 0.0]]), (0.0, 0.0, 0.0), 20.0, 0.0)
    assert m.query((0, 0, 0), 21.0, 3.1).shape[0] == 0


def test_memory_dedup_and_refresh():
    m = ObstacleMemory(CFG)
    for k in range(10):
        m.add(np.array([[1.0, 0.0], [1.02, 0.01]]), (0.0, 0.0, 0.0), float(k), 0.0)
    q = m.query((0, 0, 0), 15.0, 0.0)                              # refreshed at t=9 -> still alive
    assert q.shape[0] == 1


def test_memory_clear():
    m = ObstacleMemory(CFG)
    m.add(np.array([[1.0, 0.0]]), (0, 0, 0), 0.0, 0.0)
    m.clear()
    assert m.query((0, 0, 0), 0.0, 0.0).shape[0] == 0


# ---- clearing by the current scan (ghost cells must not block lanes the lidar sees as free) ----

def _ground(x, y, n=4):
    return np.array([[x + 0.01 * i, y] for i in range(n)])


def test_memory_cleared_where_scan_sees_open_ground():
    m = ObstacleMemory(CFG)
    m.add(np.array([[0.55, 0.05]]), (0, 0, 0), 0.0, 0.0)
    assert m.clear_seen(_ground(0.52, 0.03), np.zeros((0, 2)), (0, 0, 0)) == 1
    assert m.query((0, 0, 0), 0.1, 0.0).shape[0] == 0


def test_memory_kept_where_scan_sees_an_obstacle_too():
    m = ObstacleMemory(CFG)
    m.add(np.array([[0.55, 0.05]]), (0, 0, 0), 0.0, 0.0)
    assert m.clear_seen(_ground(0.52, 0.03), np.array([[0.56, 0.06]]), (0, 0, 0)) == 0
    assert m.query((0, 0, 0), 0.1, 0.0).shape[0] == 1


def test_memory_kept_where_scan_sees_nothing_or_too_little():
    m = ObstacleMemory(CFG)
    m.add(np.array([[0.05, -0.29]]), (0, 0, 0), 0.0, 0.0)        # blind zone beside the body
    assert m.clear_seen(_ground(1.0, 0.0), np.zeros((0, 2)), (0, 0, 0)) == 0
    assert m.clear_seen(_ground(0.02, -0.25, n=1), np.zeros((0, 2)), (0, 0, 0)) == 0   # one stray point
    assert m.query((0, 0, 0), 0.1, 0.0).shape[0] == 1


def test_memory_clearing_uses_the_dog_pose():
    m = ObstacleMemory(CFG)
    m.add(np.array([[1.05, 0.05]]), (0, 0, 0), 0.0, 0.0)           # world (1.05, 0.05)
    # dog moved to (1, 0) facing +y: the cell is now 0.05 m ahead, 0.05 m to the right
    assert m.clear_seen(_ground(0.02, -0.03), np.zeros((0, 2)), (1.0, 0.0, math.pi / 2)) == 1
