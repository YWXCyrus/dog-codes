"""rc/cellmap.py: the cell map (visited cells, sides, arena bounds, frontier) and lattice localisation."""
import numpy as np

from rc import cellmap as CM

P = 1.02


def test_placements_hold_the_start_on_an_edge():
    pl = CM.placements(3, 5)
    assert pl
    for x0, y0, a, b in pl:
        assert {a, b} == {3, 5}
        assert x0 <= 0 < x0 + a and y0 <= 0 < y0 + b
        assert x0 in (0, -(a - 1)) or y0 in (0, -(b - 1))


def test_visits_narrow_the_arena_and_one_cell_may_lie_outside():
    m = CM.CellMap(P, (3, 5))
    n0 = len(m.alive())
    for c in [(1, 0), (2, 0), (3, 0), (4, 0)]:          # a straight run of 5 cells along +x
        m.visit(c)
    m.visit((0, 0))
    assert len(m.alive()) < n0
    assert not m.in_bounds((5, 0))                      # a 6th cell in a row: only allowed as the final cell
    assert not m.in_bounds((0, -3)) and not m.in_bounds((0, 3))   # 3 wide: at most 2 cells to either side
    m.visit((5, 0))                                     # the final cell (one outside) keeps the arena alive
    assert m.alive()


def test_frontier_path_goes_through_known_open_sides_to_the_nearest_unseen_cell():
    m = CM.CellMap(P, (3, 5))
    for c in [(0, 0), (1, 0), (2, 0)]:
        m.visit(c)
    m.set_edge((0, 0), 0, True)
    m.set_edge((1, 0), 0, True)
    m.set_edge((2, 0), 0, False)
    m.set_edge((1, 0), 1, True)                         # (1, 1) unseen behind an open side
    for c, d in [((0, 0), 1), ((0, 0), 3), ((0, 0), 2), ((1, 0), 3), ((2, 0), 1), ((2, 0), 3)]:
        m.set_edge(c, d, False)
    path = m.frontier_path((2, 0))
    assert path == [(2, 0), (1, 0), (1, 1)]
    m.visit((1, 1))
    for d in range(4):
        if m.edge((1, 1), d) is None:
            m.set_edge((1, 1), d, False)
    assert m.frontier_path((2, 0)) is None              # everything reachable is seen


def _wall_points(x0, y0):
    """Wall faces of a 1.02 m corridor cell around arena (0, 0) seen by a dog believing it is at
    (x0, y0) while it really is at (0, 0): faces 0.01 m off the boundary lines."""
    ys = np.arange(-0.4, 0.4, 0.02)
    left = np.stack([ys, np.full_like(ys, 0.51 - 0.01)], 1)
    right = np.stack([ys, np.full_like(ys, -0.51 + 0.01)], 1)
    front = np.stack([np.full_like(ys, 0.51 - 0.01), ys], 1)
    pts = np.vstack([left, right, front]) + [x0, y0]
    return pts[:, 0], pts[:, 1]


def test_lattice_errors_measure_the_offset_of_the_estimate():
    X, Y = _wall_points(0.12, -0.07)                    # estimate off by (+0.12, -0.07)
    ex = CM.lattice_errors(Y, X, 0.12, P, 0.01, 0.2)
    ey = CM.lattice_errors(X, Y, -0.07, P, 0.01, 0.2)
    assert ex and abs(np.median(ex) - 0.12) < 0.02
    assert ey and abs(np.median(ey) + 0.07) < 0.02


def test_lattice_errors_ignore_lines_off_the_grid():
    ys = np.arange(-0.4, 0.4, 0.02)
    prop = np.stack([np.full_like(ys, 0.26), ys], 1)    # a prop face 0.25 m inside the cell
    assert CM.lattice_errors(prop[:, 1], prop[:, 0], 0.0, P, 0.01, 0.2) == []


def test_corridor_side_walls_are_not_a_wall_across():
    # the two side walls of a long corridor have points at every x, but only at y = +-0.5: no line
    # perpendicular to x may come out of them (W3 2026-11-26: they held the estimate still)
    xs = np.arange(-0.5, 2.5, 0.015)
    X = np.concatenate([xs, xs])
    Y = np.concatenate([np.full_like(xs, 0.5), np.full_like(xs, -0.5)])
    assert CM.wall_lines(Y, X) == []
    assert CM.lattice_errors(Y, X, 0.3, P, 0.01, 0.2) == []
    assert len(CM.wall_lines(X, Y)) == 2                # ... but they are two lines along x


def test_side_cover_sees_walls_and_openings():
    X, Y = _wall_points(0.0, 0.0)
    assert CM.side_cover(X, Y, (0, 0), 1, P) > 0.9      # left wall
    assert CM.side_cover(X, Y, (0, 0), 3, P) > 0.9      # right wall
    assert CM.side_cover(X, Y, (0, 0), 0, P) > 0.9      # front wall
    assert CM.side_cover(X, Y, (0, 0), 2, P) == 0.0     # behind: open


def _explored_3x5():
    """All 15 cells of a 3 x 5 arena visited (x 0..4, y 0..2; start (0, 0) on its edge), every side known:
    open inside the arena, walls on the border - except (4, 1) +x, the way into the final cell (5, 1)."""
    m = CM.CellMap(P, (3, 5))
    cells = [(x, y) for x in range(5) for y in range(3)]
    for c in cells:
        m.visit(c)
    for c in cells:
        for d in range(4):
            n = CM.nb(c, d)
            m.set_edge(c, d, n in cells)
    m.set_edge((4, 1), 0, True)
    return m


def test_final_cell_from_the_view_through_the_exit():
    m = _explored_3x5()
    assert m.frontier_path((0, 0)) is None                        # the 3 x 5 is explored
    assert m.exit_cell({((4, 1), 0): 3.0}) == (5, 1)              # long view: no far wall
    assert m.exit_cell({((4, 1), 0): 1.0}) is None                # a wall 1 m on: not sure -> no guess
    assert m.path_to((0, 0), (5, 1))[-2:] == [(4, 1), (5, 1)]


def test_final_cell_from_the_map_when_already_visited():
    m = _explored_3x5()
    m.visit((5, 1))                                               # exploration went in: the 16th cell
    assert m.exit_cell({}) == (5, 1)                              # every placement leaves it outside
