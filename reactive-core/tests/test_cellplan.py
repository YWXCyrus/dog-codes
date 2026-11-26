"""Cell planner (rc/cellplan.py) on hand-made arena maps."""
from rc import arena as A
from rc import config
from rc.cellplan import CellPlanner

CFG = config.load()


def known_map(walls=(), opens=None, rect=(0, 4, -1, 1), visited=((0, 0),)):
    """A map whose bounds are known (rect) with every inner border OPEN except `walls`."""
    m = A.ArenaMap(CFG)
    cells = [(i, j) for i in range(rect[0], rect[1] + 1) for j in range(rect[2], rect[3] + 1)]
    for c in cells:
        for d in 'NSWE':
            n = A.step(c, d)
            e = m.edge(A.edge_key(c, n))
            if n in cells:
                e.mp, e.state = 50.0, A.OPEN
            else:
                e.mh, e.state = 50.0, A.WALL
    for a, b in walls:
        e = m.edge(A.edge_key(a, b))
        e.mp, e.mh, e.state = 0.0, 50.0, A.WALL
    for c in [rect[:1] + (rect[2],), (rect[1], rect[3]), (rect[1], rect[2]), (rect[0], rect[3])]:
        m.force_visit(c, 0.0) if False else None
    m.no_bounds = False
    for h in m.hyps:
        h['alive'] = h['rect'] == rect
    for c in visited:
        m.force_visit(c, 0.0)
    return m


def test_path_goes_round_a_wall_not_through_it():
    m = known_map(walls=[((0, 0), (1, 0))])
    p = CellPlanner(CFG, m)
    pl = p.plan_to((0, 0), 'N', 0.0, (1, 0))
    assert pl is not None and pl.path[0] == (0, 0) and pl.path[-1] == (1, 0)
    assert (0, 0) in pl.path and len(pl.path) == 4                # (0,0) -> (0,+-1) -> (1,+-1) -> (1,0)


def test_tour_takes_the_dead_end_beside_the_dog_first_instead_of_coming_back_for_it():
    # dog in (1,0) facing N; (1,-1) is a dead end next to it, the rest of the arena lies ahead
    walls = [((1, -1), (0, -1)), ((1, -1), (2, -1))]
    m = known_map(walls=walls, visited=((0, 0), (1, 0), (0, 1), (0, -1)))
    pl = CellPlanner(CFG, m).plan((1, 0), 'N', 0.0)
    assert pl.reason == 'explore' and pl.goal == (1, -1)


def test_stairs_are_left_for_last():
    m = known_map(visited=((0, 0),))
    m.mark_stairs((1, 0), 'lidar')
    pl = CellPlanner(CFG, m).plan((0, 0), 'N', 0.0)
    assert pl.goal != (1, 0) and (1, 0) not in pl.path


def test_no_turning_on_the_stairs():
    m = known_map(visited=((0, 0),))
    m.mark_stairs((1, 0), 'lidar')
    p = CellPlanner(CFG, m)
    best, _ = p.search((0, 0), 'N', 0.0)
    # (1,1) is reachable, but never by turning inside the stairs cell
    pl = p.plan_to((0, 0), 'N', 0.0, (1, 1))
    assert (1, 0) not in pl.path


def test_final_phase_goes_to_the_final_cell_through_the_far_row():
    m = known_map(visited=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1)))
    m.edge(A.edge_key((4, 1), (5, 1))).state = A.OPEN
    pl = CellPlanner(CFG, m).plan((0, 0), 'N', 0.0, 'final')
    assert pl.reason == 'final' and pl.goal == (5, 1) and pl.path[-2] == (4, 1)


def test_a_detected_curtain_is_crossed_once_even_with_a_way_round():
    m = known_map(visited=[(i, j) for i in range(5) for j in (-1, 0, 1)])
    e = m.edge(A.edge_key((0, 0), (0, 1)))
    e.state, e.was_curtain = A.CURTAIN, True
    pl = CellPlanner(CFG, m).plan((0, 0), 'N', 0.0)
    assert pl.reason == 'curtain' and pl.path[-2:] == [(0, 0), (0, 1)]


def test_probe_a_wall_when_arena_cells_are_left_behind_it():
    m = known_map(walls=[((4, 1), (3, 1)), ((4, 1), (4, 0))],
                  visited=[(i, j) for i in range(5) for j in (-1, 0, 1) if (i, j) != (4, 1)])
    pl = CellPlanner(CFG, m).plan((0, 0), 'N', 0.0)
    assert pl.reason == 'probe' and pl.probe_edge in (A.edge_key((4, 1), (3, 1)), A.edge_key((4, 1), (4, 0)))
