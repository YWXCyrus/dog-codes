import json
import math

import pytest

from rc.grid import Grid

ORIGIN = (2.0, 3.0, math.pi / 2)       # start pose in the odom frame, facing odom +y


def test_arena_frame_x_forward_y_left():
    # booklet v6.3 E: origin = start cell centre, +X = initial forward, +Y = left, 1.02 m cells
    g = Grid(1.02)
    g.set_origin(ORIGIN)
    assert g.cell(ORIGIN) == (0, 0)
    assert g.cell((2.0, 4.02, 0.0)) == (1, 0)         # 1.02 m along the start heading (odom +y)
    assert g.cell((0.98, 3.0, 0.0)) == (0, 1)         # 1.02 m to the start's left (odom -x)
    assert g.cell((3.02, 3.0, 0.0)) == (0, -1)        # to the right
    X, Y = g.to_grid((1.0, 5.0, 0.0))
    assert (X, Y) == (pytest.approx(2.0), pytest.approx(1.0))


def test_cell_centre_is_the_reported_pos():
    g = Grid(1.02)
    assert g.centre((3, 2)) == (3.06, 2.04)           # booklet example pos [3.06, 2.04]
    assert g.centre((0, 0)) == (0.0, 0.0)


def test_cells_after_several_metres_with_1_02():
    # 1.0 m cells would be one cell off after ~25 cells; with 1.02 the 4th cell centre is exact
    g = Grid(1.02)
    g.set_origin((0.0, 0.0, 0.0))
    assert g.cell((4.08 + 0.45, 0.0, 0.0)) == (4, 0)
    assert g.cell((4.08 - 0.45, 0.0, 0.0)) == (4, 0)


def test_viewer_yaw():
    g = Grid(1.02)
    g.set_origin(ORIGIN)
    assert g.grid_yaw(math.pi / 2) == pytest.approx(math.pi / 2)     # start heading: drawn to the right
    assert g.grid_yaw(math.pi) == pytest.approx(0.0)                 # left of the start (+Y): up


def test_update_reports_new_cells_once():
    g = Grid(1.0)
    g.set_origin(ORIGIN)
    assert g.update(ORIGIN) == (0, 0)
    assert g.update((2.0, 3.2, 0.0)) is None
    assert g.update((2.0, 4.0, 0.0)) == (1, 0)
    assert g.update((2.0, 3.0, 0.0)) is None          # back in a visited cell: no change event
    assert g.visited == {(0, 0), (1, 0)}


def test_heading_dir():
    g = Grid(1.0)
    g.set_origin(ORIGIN)
    assert g.heading_dir(math.pi / 2) == 'N'
    assert g.heading_dir(0.0) == 'E'                  # odom +x = the start's right
    assert g.heading_dir(-math.pi / 2) == 'S'
    assert g.heading_dir(math.pi) == 'W'


def test_dead_ends():
    g = Grid(1.0)
    g.set_origin(ORIGIN)
    g.mark_dead_end((0, 1), 'N')
    assert g.is_dead_end((0, 1), 'N')
    assert not g.is_dead_end((0, 1), 'E')
    assert not g.all_marked((0, 1), ['N', 'E'])
    g.mark_dead_end((0, 1), 'E')
    assert g.all_marked((0, 1), ['N', 'E'])


def test_neighbour():
    g = Grid(1.0)
    assert g.neighbour((0, 0), 'N') == (1, 0)          # start heading = +X
    assert g.neighbour((0, 0), 'E') == (0, -1)         # right = -Y
    assert g.neighbour((0, 0), 'W') == (0, 1)          # left = +Y
    assert g.neighbour((0, 0), 'S') == (-1, 0)


def test_snapshot_and_clear():
    g = Grid(1.0)
    g.set_origin(ORIGIN)
    g.update(ORIGIN)
    g.mark_dead_end((0, 0), 'N')
    s = g.snapshot()
    json.dumps(s)
    assert s['visited'] == [[0, 0]]
    assert s['dead_ends'] == [[0, 0, 'N']]
    g.clear()
    assert g.visited == set() and not g.is_dead_end((0, 0), 'N')


def test_no_origin_yet():
    g = Grid(1.0)
    assert g.update((0.0, 0.0, 0.0)) is None
