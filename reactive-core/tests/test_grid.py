import json
import math

import pytest

from rc.grid import Grid

ORIGIN = (2.0, 3.0, math.pi / 2)       # start pose in the odom frame, facing odom +y


def test_forward_and_right():
    g = Grid(1.0)
    g.set_origin(ORIGIN)
    assert g.cell(ORIGIN) == (0, 0)
    assert g.cell((2.0, 4.0, 0.0)) == (0, 1)          # 1 m along the start heading
    assert g.cell((3.0, 3.0, 0.0)) == (1, 0)          # 1 m to the start's right (odom +x)
    gx, gy = g.to_grid((3.0, 4.0, 0.0))
    assert (gx, gy) == (pytest.approx(1.0), pytest.approx(1.0))


def test_update_reports_new_cells_once():
    g = Grid(1.0)
    g.set_origin(ORIGIN)
    assert g.update(ORIGIN) == (0, 0)
    assert g.update((2.0, 3.2, 0.0)) is None
    assert g.update((2.0, 4.0, 0.0)) == (0, 1)
    assert g.update((2.0, 3.0, 0.0)) is None          # back in a visited cell: no change event
    assert g.visited == {(0, 0), (0, 1)}


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
    assert g.neighbour((0, 0), 'N') == (0, 1)
    assert g.neighbour((0, 0), 'E') == (1, 0)
    assert g.neighbour((0, 0), 'W') == (-1, 0)


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
