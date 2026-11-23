import math

import numpy as np
import pytest

from rc import config
from rc.liomap import LioMap

CFG = config.load()
Q0 = (0.0, 0.0, 0.0, 1.0)


def qz(a):
    return (0.0, 0.0, math.sin(a / 2.0), math.cos(a / 2.0))


def test_point_seen_from_another_pose():
    m = LioMap(CFG)
    m.on_pose((0.0, 0.0, 0.0), Q0, 0.0)
    m.add(np.array([[2.0, 0.0, 0.1]]), 0.0)           # 2 m ahead at the start
    m.on_pose((0.2, 0.0, 0.0), Q0, 1.0)                # 0.2 m forward (0.2 m/s)
    p = m.points_sensor(1.0)
    assert p.shape == (1, 3) and p[0] == pytest.approx([1.8, 0.0, 0.1], abs=0.05)


def test_point_after_turning_90_deg():
    m = LioMap(CFG)
    m.on_pose((0.0, 0.0, 0.0), Q0, 0.0)
    m.add(np.array([[1.0, 0.0, 0.0]]), 0.0)
    for k in range(1, 11):                              # turn left 90 deg in 4 s, on the spot
        m.on_pose((0.0, 0.0, 0.0), qz(math.pi / 2 * k / 10), 0.4 * k)
    p = m.points_sensor(4.0)
    assert p[0] == pytest.approx([0.0, -1.0, 0.0], abs=0.05)   # the point is now on the right


def test_points_near_the_sensor_never_stored():
    m = LioMap(CFG)
    m.on_pose((0.0, 0.0, 0.0), Q0, 0.0)
    m.add(np.array([[0.30, 0.15, -0.2], [0.2, 0.0, 0.0], [1.5, 0.0, 0.0]]), 0.0)   # legs, legs, wall
    assert len(m) == 1


def test_voxel_keeps_one_point():
    m = LioMap(CFG)
    m.on_pose((0.0, 0.0, 0.0), Q0, 0.0)
    m.add(np.array([[1.501, 0.001, 0.001], [1.502, 0.002, 0.002]]), 0.0)
    m.add(np.array([[1.503, 0.003, 0.003]]), 0.1)
    assert len(m) == 1


def test_pruned_by_radius_and_age():
    m = LioMap(CFG)
    m.on_pose((0.0, 0.0, 0.0), Q0, 0.0)
    m.add(np.array([[2.5, 0.0, 0.0], [-2.9, 0.0, 0.0]]), 0.0)
    for k in range(1, 21):                              # walk 2 m forward at 0.2 m/s
        m.on_pose((0.1 * k, 0.0, 0.0), Q0, 0.5 * k)
    m.add(np.zeros((0, 3)), 10.0)
    assert len(m) == 1                                  # the point behind is now > 3 m away
    m.on_pose((2.0, 0.0, 0.0), Q0, 10.0 + CFG['lio_time_s'] + 1.0)
    m.add(np.zeros((0, 3)), 10.0 + CFG['lio_time_s'] + 1.0)
    assert len(m) == 0                                  # too old


def test_pose_jump_marks_bad_for_the_run():
    m = LioMap(CFG)
    m.on_pose((0.0, 0.0, 0.0), Q0, 0.0)
    m.add(np.array([[2.0, 0.0, 0.0]]), 0.0)
    assert m.on_pose((40.0, 0.0, 0.0), Q0, 0.1) is not None      # 400 m/s (seen 2026-11-23)
    assert not m.ok and m.status(0.1) == 'BAD' and len(m) == 0
    m.on_pose((0.0, 0.0, 0.0), Q0, 0.2)                           # never comes back
    assert m.points_sensor(0.2).shape[0] == 0


def test_fast_drift_marks_bad():
    m = LioMap(CFG)
    bad = None
    for k in range(20):                                 # 0.3 m per 0.1 s = 3 m/s, each step < lio_jump_m
        bad = bad or m.on_pose((0.3 * k, 0.0, 0.0), Q0, 0.1 * k)
    assert bad is not None and 'speed' in bad and not m.ok


def test_stale_pose_not_used():
    m = LioMap(CFG)
    m.on_pose((0.0, 0.0, 0.0), Q0, 0.0)
    m.add(np.array([[2.0, 0.0, 0.0]]), 0.0)
    assert m.points_sensor(0.1).shape[0] == 1
    assert m.points_sensor(CFG['lio_stale_s'] + 0.1).shape[0] == 0 and m.status(1.0) == '-'


def test_core_without_fastlio_is_scan_only():
    from rc.core import Core
    core = Core(CFG)
    pts = np.array([[2.0, y, -0.3] for y in np.linspace(-1, 1, 50)])
    core.on_cloud(pts, 0.0)
    assert core.lio.status(0.0) == '-' and core._mem_body(0.0).shape == (0, 2)


def test_core_map_fills_lanes_with_earlier_wall():
    from rc.core import Core
    core = Core(CFG)
    core.on_lio((0.0, 0.0, 0.0), Q0, 0.0)
    wall = np.array([[1.2, y, z] for y in np.linspace(-0.6, 0.6, 40) for z in np.linspace(-0.3, 0.3, 12)])
    core.on_cloud(wall, 0.0)                            # seen once ...
    core.on_lio((0.0, 0.0, 0.0), Q0, 0.1)
    core.on_cloud(np.zeros((0, 3)), 0.1)                # ... then not in the live scan any more
    mb = core._mem_body(0.1)
    assert mb.shape[0] > 0 and np.all(np.abs(mb[:, 0] - 1.2 - 0.187) < 0.4)
