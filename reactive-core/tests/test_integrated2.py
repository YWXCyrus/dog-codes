"""Branch `integrated`, second round (2026-11-26 evening, live runs 093607 / 094043): scene test by coverage
(fence walls leak lidar rays), the camera's view cell for perception, stairs only from a confident lidar /
the camera, one '>> ...' message per event. Hand-made maps + toy kinematics (no sim)."""
import math

import numpy as np

from rc import arena as A
from rc import config
from rc.cellnav import CellNav
from tests.test_integrated import CFG, Dog, boxed_map, drive, events, make_nav, scene_map


# ---------------------------------------------------------------- scene test by coverage (fence walls)
def leaky_room_map():
    """(2,0)'s three sides UNKNOWN by the ray test (fence walls leak rays), doorway (1,0)-(2,0) open."""
    m = scene_map(props=100.0)
    for k in ((2, 0, 'x'), (2, 0, 'y'), (2, -1, 'y')):
        e = m.edge(k)
        e.mh, e.mp, e.state = 50.0, 60.0, A.UNKNOWN
    return m


def side_points(k, n=40):
    """Obstacle returns along a border line (a fence seen along its length)."""
    i, j, ax = k
    s = np.linspace(-0.38, 0.38, n)
    if ax == 'x':
        return np.column_stack([np.full(n, (i + 0.5) * 1.02), j * 1.02 + s])
    return np.column_stack([i * 1.02 + s, np.full(n, (j + 0.5) * 1.02)])


def see_room(m, times=3):
    for _ in range(times):
        for k in ((2, 0, 'x'), (2, 0, 'y'), (2, -1, 'y')):
            m.observe_cover(side_points(k))


def test_fence_sides_seen_along_their_length_make_a_scene():
    m = leaky_room_map()
    assert m.scene_kind((2, 0), (1, 0)) is None                   # ray test alone: unknown sides
    see_room(m)
    assert m.cover((2, 0), (3, 0)) == 1.0
    assert m.scene_kind((2, 0), (1, 0)) == 'scene'


def test_a_side_with_a_few_returns_is_not_closed():
    m = leaky_room_map()
    for _ in range(3):
        for k in ((2, 0, 'y'), (2, -1, 'y')):
            m.observe_cover(side_points(k))
        m.observe_cover(side_points((2, 0, 'x'))[:6])              # something near one end of the far side only
    assert m.cover((2, 0), (3, 0)) < CFG['scene_cover_min']
    assert m.scene_kind((2, 0), (1, 0)) is None


def test_a_traversed_border_is_never_closed():
    m = leaky_room_map()
    see_room(m)
    m.edge((2, 0, 'x')).traversed = True
    assert m.scene_kind((2, 0), (1, 0)) is None


def test_room_closed_by_fences_is_never_entered_not_even_to_reach_cells_behind_it():
    # its sides are UNKNOWN to the ray test: without closing them the planner routed THROUGH the room
    # to the unvisited cells behind it (found by this test, 2026-11-26)
    m = leaky_room_map()
    see_room(m)
    nav = make_nav(m, (1, 0))
    dog = Dog(1.02, 0.0, 0.0)
    outs = drive(nav, dog, 40.0, answer_after=0.0)
    assert events(outs, 'scene_done') and m.visited((2, 0))
    assert m.is_blocked((1, 0), (2, 0), dog.t)
    assert all(m.is_blocked((2, 0), n, dog.t) for n in ((3, 0), (2, 1), (2, -1)))
    assert max(o.cmd.vx for o in outs) <= 0.0 and dog.x < 1.2


def test_boards_below_the_prop_line_are_not_scene_furniture():
    m = A.ArenaMap(CFG)
    xy = np.column_stack([np.full(100, 2.04), np.linspace(-0.3, 0.3, 100)])
    m._observe_cells(np.column_stack([xy, np.full(100, 0.30)]), 1.02, 0.0)     # top of a wooden board
    assert m.cell((2, 0)).prop_n == 0
    m._observe_cells(np.column_stack([xy, np.full(100, 0.70)]), 1.02, 0.0)     # a trolley / cabinet
    assert m.cell((2, 0)).prop_n >= CFG['scene_prop_min']


def test_coverage_comes_from_the_driving_obstacles_in_core():
    from rc.core import Core
    core = Core(config.load())
    core.scan.obs = np.array([[1.53, y] for y in np.linspace(-0.38, 0.38, 30)])   # a fence 1.53 m ahead
    ob = core._off_body(core.scan.obs)
    for _ in range(3):
        core.amap.observe_cover(ob)                                # what _grid_cloud does at the start pose
    assert core.amap.cover((1, 0), (2, 0)) == 1.0


# ---------------------------------------------------------------- the camera's view cell
def test_view_cell_runs_to_the_first_wall():
    m = boxed_map(opens=[((0, 0), (1, 0)), ((1, 0), (2, 0))], visited=[(0, 0)])
    assert m.view_cell(0.0, 0.0, 0.0, 2.0) == (2, 0)                # open to (2,0), its far side is a wall
    assert m.view_cell(0.0, 0.0, 0.0, 0.8) == (1, 0)                # capped
    assert m.view_cell(0.0, 0.0, math.pi / 2, 2.0) == (0, 0)        # a wall right beside: own cell


def test_core_sends_the_view_cell():
    from rc.core import Core
    core = Core(config.load())
    core.amap = boxed_map(opens=[((0, 0), (1, 0)), ((1, 0), (2, 0))], visited=[(0, 0)])
    core.driver.nav = CellNav(core.c, core.amap)
    core.scan.obs_now = np.array([[1.9, 0.0]])                     # the far wall of (2, 0), 1.9 m ahead
    assert core._view_look((0.0, 0.0, 0.0)) == {'cell': [2, 0], 'look': -1, 'kind': 'view'}
    core.scan.obs_now = np.array([[0.6, 0.0]])                     # a prop 0.6 m ahead: the camera stops there
    assert core._view_look((0.0, 0.0, 0.0))['cell'] == [1, 0]
    core.driver.nav = None                                         # wall mode: nothing sent
    assert core._view_look((0.0, 0.0, 0.0)) is None


def test_grid_mode_status_uses_arena_cells():
    from rc.core import Core
    core = Core(config.load())
    core.started = True
    core.apose = lambda t=None: (1.02, -1.0, -math.pi / 2)         # one forward, one right, facing right
    st = core.status(0.0)
    assert st['cell'] == [1, -1] and st['dir'] == 'E'


# ---------------------------------------------------------------- stairs: one raised board is a question
def test_one_raised_level_asks_the_camera_instead_of_marking_stairs():
    m = boxed_map(opens=[((0, 0), (1, 0))], visited=[(0, 0)])
    nav = make_nav(m, (0, 0), came=None)
    ni = Dog(0.0, 0.0, 0.0).ni()
    ni.stairs_cue = {'bearing': 0.0, 'src': 'lidar_unsure'}
    assert nav._stairs_check(ni, (0, 0), (1, 0)) is True           # handled: a question for the camera
    assert not m.is_stairs((1, 0)) and nav.phase == 'STAIRS_ASK'
    m2 = boxed_map(opens=[((0, 0), (1, 0))], visited=[(0, 0)])
    nav2 = make_nav(m2, (0, 0), came=None)
    ni.stairs_cue = {'bearing': 0.0, 'src': 'lidar_up'}            # a confident 'up' is stairs
    assert nav2._stairs_check(ni, (0, 0), (1, 0)) is True and m2.is_stairs((1, 0))


def test_lidar_cue_is_unsure_unless_the_detector_is_confident():
    from rc.core import Core
    from rc.stairs import StairsResult
    core = Core(config.load())
    core.stairs = StairsResult(kind='up', heading=0.0, edge_x=1.0, confident=False)
    assert core._stairs_cue(0.0)['src'] == 'lidar_unsure'
    core.stairs = StairsResult(kind='up', heading=0.0, edge_x=1.0, confident=True)
    assert core._stairs_cue(0.0)['src'] == 'lidar_up'


# ---------------------------------------------------------------- messages
def test_messages_for_the_main_events_cells_only():
    from rc.say import Say
    s = Say()
    assert s.lines({'t': 1, 'scene_state': [2, 0], 'scene': 'cleaning_room', 'survivor': True}) == \
        ['cleaning_room, (2,0), survivor']
    assert s.lines({'t': 2, 'scene_state': [2, 0], 'scene': 'cleaning_room', 'survivor': True}) == []
    assert s.lines({'t': 3, 'scene_state': [1, 1], 'scene': '', 'survivor': False}) == []
    assert s.lines({'t': 4, 'stairs_seen': 'up', 'confident': False, 'src': 'lidar'}) == \
        ['stairs detected ahead (lidar up)']
    assert s.lines({'t': 5, 'stairs_cell': [3, 1], 'src': 'lidar'}) == ['stairs confirm (3,1) (lidar)']
    assert s.lines({'t': 6, 'stairs': 'ALIGN'}) == ['ai mode activated']
    assert s.lines({'t': 7, 'stairs': 'DONE'}) == ['ai mode exited, walking again']
    assert s.lines({'t': 8, 'arena_cell': [1, 0], 'pos': [1.02, 0.0]}) == ['cell (1,0)']
    assert s.lines({'t': 9, 'grid': 'x', 'scene': [2, 0], 'kind': 'scene', 'pos': [2.04, 0.0]}) == \
        ['scene detected (2,0): looking from the doorway']
    assert s.lines({'t': 10, 'report': 'M01', 'type': 'object', 'label': 'utensils', 'scene': 'cleaning_room',
                    'cell': [2, 0], 'message': ''}) == ['M01 reported: utensils, cleaning_room, (2,0)']
    assert s.lines({'t': 11, 'curtain': 'push', 'edge': [1, -1, 'y']}) == \
        ['pushing through the curtain (1,-1)-(1,0)']
    assert s.lines({'t': 12, 'odom_frozen': True}) == ['odom_frozen True']                      # anything else
    assert s.lines({'t': 13, 'hold': 'blocked'}) == []                                          # run.py banner
    assert s.lines({'t': 14, 'stairs_answer': 'no', 'cell': [1, -1]}) == ['not stairs (1,-1) (camera)']


def test_core_turns_perception_reports_into_events_once():
    from rc.core import Core
    core = Core(config.load())
    st = {'tasks': {'M01': {'type': 'object', 'reported': True, 'label': 'utensils', 'scene_type': 'cleaning_room',
                            'pos': [2.04, 0.0]}}, 'qr': {'read': True, 'text': 'My friend is in the market.'}}
    core.on_mission_status(st, 1.0)
    core.on_mission_status(st, 2.0)
    ev = core.pending_events
    reps = [e for e in ev if 'report' in e]
    assert [e['report'] for e in reps] == ['M01'] and reps[0]['cell'] == [2, 0]
    assert [e['qr_decoded'] for e in ev if 'qr_decoded' in e] == ['My friend is in the market.']
    for t in (3.0, 3.5, 9.0):
        core.on_quad_event({'type': 'qr_seen'}, t)
    assert len([e for e in core.pending_events if e.get('perception') == 'qr_seen']) == 2      # 3.0 and 9.0


def test_stairs_seen_event_once_per_change():
    from rc.core import Core
    from rc.stairs import StairsResult
    core = Core(config.load())
    core.t_go = 0.0
    core.stairs = StairsResult(kind='up', heading=0.0, edge_x=1.5, confident=False)
    core._stairs_seen_event(1.0)
    core._stairs_seen_event(1.1)                                   # same: no new event
    core.stairs = StairsResult(kind='up', heading=0.0, edge_x=1.2, confident=True)
    core._stairs_seen_event(5.0)                                   # now sure
    ev = [e for e in core.pending_events if 'stairs_seen' in e]
    assert [e['confident'] for e in ev] == [False, True]
