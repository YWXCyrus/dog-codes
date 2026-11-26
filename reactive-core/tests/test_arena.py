"""Arena cell map (rc/arena.py): border evidence, placements (bounds), final cell, camera cues."""
import math

import numpy as np

from rc import arena as A
from rc import config

CFG = config.load()
L = CFG['arena_cell_m']
SENSOR = (0.19, 0.0, 0.45)          # arena frame, height above the stance floor


def plane_x(x, y0, y1, z0, z1, step=0.02):
    ys, zs = np.meshgrid(np.arange(y0, y1, step), np.arange(z0, z1, step))
    return np.stack([np.full(ys.size, x), ys.ravel(), zs.ravel()], axis=1)


def floor_patch(x0, x1, y0, y1, step=0.04):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)


def observe(m, P, n=3):
    for k in range(n):
        m.observe(P, SENSOR, float(k))


def test_panel_on_the_border_ahead_is_a_wall():
    m = A.ArenaMap(CFG)
    observe(m, plane_x(0.51, -0.5, 0.5, 0.0, 0.6))
    assert m.state((0, 0), (1, 0)) == A.WALL


def test_floor_seen_beyond_the_border_is_open():
    m = A.ArenaMap(CFG)
    observe(m, floor_patch(0.9, 2.0, -0.4, 0.4))
    assert m.state((0, 0), (1, 0)) == A.OPEN


def test_hanging_cloth_open_below_is_a_curtain():
    m = A.ArenaMap(CFG)
    cloth = plane_x(0.51, -0.5, 0.5, 0.47, 1.0)                  # hem ~0.45 m, bar ~1 m (booklet D.4)
    beyond = floor_patch(0.9, 2.2, -0.4, 0.4)                   # seen under the hem
    observe(m, np.vstack([cloth, beyond]))
    assert m.state((0, 0), (1, 0)) == A.CURTAIN


def test_low_panel_with_nothing_above_is_still_a_wall_not_a_curtain():
    m = A.ArenaMap(CFG)
    wall = plane_x(0.51, -0.5, 0.5, 0.0, 0.6)
    far_wall = plane_x(2.5, -1.0, 1.0, 0.65, 1.0)               # seen over the panel
    observe(m, np.vstack([wall, far_wall]))
    assert m.state((0, 0), (1, 0)) == A.WALL


def test_twelve_placements_at_the_start_and_visits_kill_them():
    m = A.ArenaMap(CFG)
    assert len(m.live()) == 12
    for c in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]:           # five cells along +X: long axis is X, far end +X
        m.force_visit(c, 0.0)
    assert all(h['axis'] == 'x' and h['sgn'] == 1 for h in m.live())
    m.force_visit((0, 1), 0.0)
    m.force_visit((0, -1), 0.0)
    assert m.bounds_known() and m.rect_union() == (0, 4, -1, 1)
    assert set(m.final_candidates()) == {(5, -1), (5, 0), (5, 1)}
    m.force_visit((5, 1), 0.0)
    assert m.final_cell == (5, 1) and m.final_candidates() == [(5, 1)]


def test_a_cell_outside_every_placement_loses_the_bounds_gracefully():
    m = A.ArenaMap(CFG)
    for c in [(0, 0), (3, 3), (-3, -3)]:
        m.force_visit(c, 0.0)
    assert m.no_bounds and m.possible((9, 9))


def test_traversed_border_is_open_for_good():
    m = A.ArenaMap(CFG)
    m.on_pose(0.0, 0.0, 0.0)
    m.on_pose(L, 0.0, 1.0)
    assert m.state((0, 0), (1, 0)) == A.OPEN and m.edges[A.edge_key((0, 0), (1, 0))].traversed


def test_camera_cue_goes_to_the_border_it_points_at_square_and_not_to_a_grazed_neighbour():
    m = A.ArenaMap(CFG)
    observe(m, plane_x(0.51, -0.5, 0.5, 0.0, 0.6))              # a wall-like border ahead
    for _ in range(int(CFG['cur_cue_min'])):
        assert m.add_curtain_cue(0.0, 0.0, 0.0, 0.0) == (0, 0, 'x')
    assert m.state((0, 0), (1, 0)) == A.CURTAIN                  # floor-length curtain: wall-like + camera
    m2 = A.ArenaMap(CFG)
    assert m2.add_curtain_cue(0.0, 0.0, math.radians(45.0), 0.0) is None     # through the corner post: no credit


def test_snapshot_is_json_ready():
    import json
    m = A.ArenaMap(CFG)
    observe(m, plane_x(0.51, -0.5, 0.5, 0.0, 0.6))
    m.force_visit((0, 0), 0.0)
    s = m.snapshot(1.0)
    json.dumps(s)
    assert any(e['state'] == 'wall' for e in s['edges']) and s['bounds']['rects']
