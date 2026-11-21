"""Replays of real trial-3 frames (tests/data/trial3_run1.npz, built by tools/extract_fixtures.py).

Frames are in the levelled body frame with the floor at -0.32 m (BODY_H of the old cloud_proc).
times: 0,1,2,3,4 (start, strip pit) and 39,40,40.5,41,41.5,42 (approach to the wall).
"""
import os

import numpy as np
import pytest

from rc import config
from rc.sensing import ObstacleDetector

DATA = os.path.join(os.path.dirname(__file__), 'data', 'trial3_run1.npz')
pytestmark = pytest.mark.skipif(not os.path.exists(DATA), reason='fixture missing')
CFG = config.load()
FLOOR = -0.32
SENSOR_X = 0.18678


def frame(t):
    d = np.load(DATA)
    i = int(np.argmin(np.abs(d['times'] - t)))
    return d['pts_%d' % i].astype(float)


def scan_of(t, **kw):
    det = ObstacleDetector(CFG)
    s = None
    for k in range(5):
        s = det.update(frame(t), FLOOR, k * 0.1, **kw)
    return s


def in_box(c, x0, x1, ymax):
    if c.shape[0] == 0:
        return c
    return c[(c[:, 0] > x0) & (c[:, 0] < x1) & (np.abs(c[:, 1]) < ymax)]


@pytest.mark.parametrize('t', [40.5, 41.5])
def test_wall_in_front_is_seen(t):
    s = scan_of(t)
    near = in_box(s.obs, SENSOR_X, SENSOR_X + 0.5, 0.3)
    assert near.shape[0] > 0, 'trial-3 wall must be an obstacle within 0.5 m of the sensor'


@pytest.mark.parametrize('t', [2.0, 3.0, 4.0])
def test_strip_pit_boards_are_not_obstacles(t):
    s = scan_of(t)
    lane = in_box(s.obs, 0.5, 1.4, 0.35)
    assert lane.shape[0] < 3, 'strip boards below step_limit must not block the start lane: %s' % lane
