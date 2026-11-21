import math
import os

import numpy as np
import pytest

from rc import config
from rc import stairs as S
from tests import helpers as H

CFG = config.load()


def up_stairs(x0=1.6, n=3, rise=0.15, tread=0.30, angle=0.0):
    pts = np.vstack([H.floor(1.4, x0 - 0.01), H.stairs(x0, n, rise, tread, -0.8, 0.8),
                     H.grid_pts(x0 + n * tread, x0 + n * tread + 1.2, -0.8, 0.8, H.FLOOR + n * rise)])
    return H.rotate_z(pts, angle) if angle else pts


def run(det, pts, n=7, dist=5.0):
    r = None
    for k in range(n):
        r = det.update(pts, H.FLOOR, k * 0.1, dist)
    return r


BIN = CFG['stair_bin_m']


def test_profile_bins():
    c, h = S.profile(up_stairs(), 0.0, 0.70, BIN, H.FLOOR)
    assert np.nanmin(h) == pytest.approx(H.FLOOR, abs=0.02)
    assert np.nanmax(h) == pytest.approx(H.FLOOR + 0.45, abs=0.02)


@pytest.mark.parametrize('x0', [1.6, 1.63, 1.66, 1.71])      # riser positions vs bin edges
def test_analyse_up_any_alignment(x0):
    a = S.analyse_lane(up_stairs(x0=x0), 0.0, H.FLOOR, CFG)
    assert a['kind'] == 'up' and a['rises'] >= 2


def test_analyse_up():
    a = S.analyse_lane(up_stairs(), 0.0, H.FLOOR, CFG)
    assert a['kind'] == 'up' and a['rises'] >= 2
    assert a['edge_x'] == pytest.approx(1.6, abs=0.16)


def test_levels_found():
    u, _, z = S.lane_points(up_stairs(), 0.0, 0.70, H.FLOOR)
    lv = [round(l[0], 2) for l in S.levels(u, z)]
    for want in (0.0, 0.15, 0.30, 0.45):
        assert any(abs(l - want) < 0.02 for l in lv), lv


def test_detector_confident_after_persistence():
    det = S.StairDetector(CFG)
    r = det.update(up_stairs(), H.FLOOR, 0.0, 5.0)
    assert r.kind == 'up' and not r.confident
    r = run(det, up_stairs(), n=5)
    assert r.kind == 'up' and r.confident
    assert r.edge_x == pytest.approx(1.6, abs=0.16)


@pytest.mark.parametrize('deg', [-15, -7, 0, 10, 15])
def test_detector_squares_up_to_rotated_stairs(deg):
    r = run(S.StairDetector(CFG), up_stairs(angle=math.radians(deg)))
    assert r.kind == 'up'
    assert math.degrees(r.heading) == pytest.approx(deg, abs=3.0)


def test_strip_boards_are_not_stairs():
    pts = np.vstack([H.floor(1.4, 4.0),
                     H.box_top(1.6, 1.9, -0.6, 0.6, 0.10), H.box_top(2.2, 2.5, -0.6, 0.6, 0.10),
                     H.box_top(2.8, 3.1, -0.6, 0.6, 0.10)])
    r = run(S.StairDetector(CFG), pts)
    assert r.kind != 'up'


def test_single_board_never_up():
    pts = np.vstack([H.floor(1.4, 4.0), H.box_top(1.8, 2.1, -0.6, 0.6, 0.15)])
    r = run(S.StairDetector(CFG), pts)
    assert r.kind in ('none', 'unsure')


def test_one_step_then_unseen_is_unsure():
    pts = np.vstack([H.floor(1.4, 1.79), H.stairs(1.8, 1, 0.15, 0.6, -0.6, 0.6)])
    r = run(S.StairDetector(CFG), pts)
    assert r.kind == 'unsure'


def test_wall_is_not_stairs():
    pts = np.vstack([H.floor(1.4, 2.0), H.wall_x(2.0, -1, 1, H.FLOOR, H.FLOOR + 1.0)])
    r = run(S.StairDetector(CFG), pts)
    assert r.kind == 'none'


def test_down_edge():
    pts = np.vstack([H.floor(1.4, 2.0), H.floor(2.31, 3.5, z=H.FLOOR - 0.15)])
    r = run(S.StairDetector(CFG), pts)
    assert r.kind == 'down'
    assert r.edge_x == pytest.approx(2.0, abs=0.35)


def test_down_with_gap_then_lower():
    pts = np.vstack([H.floor(1.4, 1.9), H.floor(2.6, 3.5, z=H.FLOOR - 0.30)])
    r = run(S.StairDetector(CFG), pts)
    assert r.kind == 'down'


def test_start_exclusion():
    r = run(S.StairDetector(CFG), up_stairs(), dist=1.0)
    assert r.kind == 'none'


DATA = os.path.join(os.path.dirname(__file__), 'data', 'trial3_run1.npz')


@pytest.mark.skipif(not os.path.exists(DATA), reason='fixture missing')
@pytest.mark.parametrize('t', [2.0, 3.0, 4.0])
def test_real_start_frames_never_up(t):
    d = np.load(DATA)
    i = int(np.argmin(np.abs(d['times'] - t)))
    p = d['pts_%d' % i].astype(float)
    r = run(S.StairDetector(CFG), p, dist=0.0)
    assert r.kind == 'none'                              # inside the start exclusion radius


@pytest.mark.skipif(not os.path.exists(DATA), reason='fixture missing')
@pytest.mark.parametrize('t', [0.0, 1.0, 2.0, 3.0, 4.0, 39.0, 40.0, 40.5, 41.0, 41.5, 42.0])
def test_real_frames_not_stairs_even_without_exclusion(t):
    # regression: the strip pit (boards +0.10 m, then a wall face) once read as 2-4 "steps"
    d = np.load(DATA)
    i = int(np.argmin(np.abs(d['times'] - t)))
    r = run(S.StairDetector(CFG), d['pts_%d' % i].astype(float), dist=10.0)
    assert r.kind != 'up'


def test_wall_face_is_not_a_staircase():
    # a sloped / vertical face sampled into short plateaus must not count as treads
    face = np.vstack([H.grid_pts(1.4 + 0.05 * k, 1.44 + 0.05 * k, -0.6, 0.6, H.FLOOR + 0.12 * (k + 1)) for k in range(5)])
    r = run(S.StairDetector(CFG), np.vstack([H.floor(1.4, 1.39 + 1e-3), face]))
    assert r.kind != 'up'


def test_rise_from_unseen_floor_not_counted():
    # dog in a trough: first visible surface is a board +0.10 m, then ONE real step: not 'up'
    pts = np.vstack([H.grid_pts(0.7, 1.39, -0.6, 0.6, H.FLOOR + 0.10),
                     H.grid_pts(1.4, 2.0, -0.6, 0.6, H.FLOOR + 0.25)])
    r = run(S.StairDetector(CFG), pts)
    assert r.kind != 'up'
