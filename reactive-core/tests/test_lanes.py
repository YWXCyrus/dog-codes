import math
import os

import numpy as np
import pytest

from rc import config
from rc import lanes as L
from rc.sensing import ObstacleDetector

CFG = config.load()


def line(x0, y0, x1, y1, n=60):
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


def test_empty_all_free():
    ls = L.evaluate(np.zeros((0, 2)), np.zeros((0, 2)), CFG)
    assert all(l.free for l in ls)
    assert len(ls) == len(L.candidate_headings(CFG))


def test_candidate_headings_limits():
    h = L.candidate_headings(CFG)
    assert math.degrees(h.max()) == pytest.approx(135)
    assert math.degrees(h.min()) == pytest.approx(-135)
    assert np.any(np.isclose(h, 0.0))


def test_wall_across():
    wall = line(0.8, -2, 0.8, 2)
    assert not L.lane_free(wall, 0.0, 0.70, 1.0)
    assert L.lane_free(wall, math.pi / 2, 0.70, 1.0)
    assert L.lane_free(wall, -math.pi / 2, 0.70, 1.0)
    assert L.free_distance(wall, 0.0, 0.70) == pytest.approx(0.8, abs=0.01)


def test_doorway_width():
    door = np.vstack([line(0.6, -2.0, 0.6, -0.40), line(0.6, 0.40, 0.6, 2.0)])   # 0.80 m opening
    assert L.lane_free(door, 0.0, 0.70, 1.0)
    assert not L.lane_free(door, 0.0, 0.90, 1.0)


def test_drop_vetoes():
    ls = L.evaluate(np.zeros((0, 2)), np.array([[0.5, 0.0]]), CFG)
    straight = [l for l in ls if abs(l.heading) < 1e-9][0]
    assert not straight.free


def test_points_behind_do_not_block():
    assert L.lane_free(np.array([[-0.3, 0.0]]), 0.0, 0.70, 1.0)


def test_width_override():
    door = np.vstack([line(0.6, -2.0, 0.6, -0.30), line(0.6, 0.30, 0.6, 2.0)])   # 0.60 m
    s0 = [l for l in L.evaluate(door, np.zeros((0, 2)), CFG) if abs(l.heading) < 1e-9][0]
    s1 = [l for l in L.evaluate(door, np.zeros((0, 2)), CFG, width=0.50) if abs(l.heading) < 1e-9][0]
    assert not s0.free and s1.free


DATA = os.path.join(os.path.dirname(__file__), 'data', 'trial3_run1.npz')


@pytest.mark.skipif(not os.path.exists(DATA), reason='fixture missing')
@pytest.mark.parametrize('t', [2.0, 3.0, 4.0])
def test_real_start_lane_free(t):
    d = np.load(DATA)
    i = int(np.argmin(np.abs(d['times'] - t)))
    det = ObstacleDetector(CFG)
    for k in range(5):
        s = det.update(d['pts_%d' % i].astype(float), -0.32, k * 0.1)
    assert L.lane_free(s.obs, 0.0, CFG['lane_w'], CFG['lane_d'])
