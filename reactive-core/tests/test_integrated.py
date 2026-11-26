"""Branch `integrated` (2026-11-26): scene rooms looked at from the doorway, reversing over walked ground,
no on-the-spot turn while tilted, nav_mode values checked. Hand-made arena maps + toy kinematics (no sim)."""
import math

import numpy as np
import pytest

from rc import arena as A
from rc import config
from rc.cellnav import CellNav, NavIn

CFG = dict(config.load(), grid_reverse=True)     # the reversing tests switch it on (default off)
RECT = (0, 4, -1, 1)                      # the one live 3 x 5 placement in these maps


def boxed_map(opens, visited, rect=RECT):
    """Every border of the rect's cells is WALL except `opens` (pairs of cells); bounds = rect."""
    m = A.ArenaMap(CFG)
    cells = [(i, j) for i in range(rect[0], rect[1] + 1) for j in range(rect[2], rect[3] + 1)]
    for c in cells:
        for d in 'NSWE':
            e = m.edge(A.edge_key(c, A.step(c, d)))
            e.mh, e.state = 50.0, A.WALL
    for a, b in opens:
        e = m.edge(A.edge_key(a, b))
        e.mh, e.mp, e.state = 0.0, 50.0, A.OPEN
    for h in m.hyps:
        h['alive'] = h['rect'] == rect
    for c in visited:
        m.force_visit(c, 0.0)
    return m


def scene_map(props=100.0):
    """Start (0,0) -> (1,0) -> doorway -> (2,0) with WALL on its other three sides."""
    m = boxed_map(opens=[((0, 0), (1, 0)), ((1, 0), (2, 0))], visited=[(0, 0), (1, 0)])
    m.cell((2, 0)).prop_n = props
    return m


def make_nav(m, here, came='S'):
    nav = CellNav(CFG, m)
    nav.paused = set(m.order)
    nav.last_cell = here
    if came:
        nav.came_from[here] = came
    return nav


class Dog:
    """Toy kinematics: the command is followed exactly, except `reverses=False` (the real dog 2026-11-26)."""

    def __init__(self, x, y, yaw, reverses=True, guard_passes_reverse=True):
        self.x, self.y, self.yaw = x, y, yaw
        self.travelled = 0.0
        self.last_vx = 0.0
        self.reverses = reverses
        self.guard_passes = guard_passes_reverse
        self.t = 0.0

    def ni(self, perception=True, scene_seen=None):
        return NavIn(t=self.t, apose=(self.x, self.y, self.yaw), pts=np.zeros((0, 2)), travelled=self.travelled,
                     odom_ok=True, elapsed=self.t, last_vx=self.last_vx, perception=perception,
                     scene_seen=scene_seen or {})

    def apply(self, cmd, dt=0.1):
        vx = cmd.vx
        if vx < 0 and not self.guard_passes:
            vx = 0.0                                       # the guard stopped it (stale feeds)
        self.last_vx = vx
        if vx < 0 and not self.reverses:
            vx = 0.0                                       # commanded, passed, but the dog does not move
        self.yaw = math.atan2(math.sin(self.yaw + cmd.wz * dt), math.cos(self.yaw + cmd.wz * dt))
        self.x += vx * dt * math.cos(self.yaw)
        self.y += vx * dt * math.sin(self.yaw)
        self.travelled += abs(vx) * dt
        self.t += dt


def drive(nav, dog, secs, dt=0.1, answer_after=None, perception=True):
    outs, seen = [], {}
    for _ in range(int(round(secs / dt))):
        if answer_after is not None and dog.t >= answer_after and not seen:
            seen = {(2, 0): dog.t}
        out = nav.step(dog.ni(perception=perception, scene_seen=seen))
        outs.append(out)
        dog.apply(out.cmd, dt)
        if out.hold:
            break
    return outs


def events(outs, key):
    return [e for o in outs for e in o.events if key in e]


# ---------------------------------------------------------------- map: scene / dead end
def test_scene_kind_three_walls_with_props_is_a_scene_without_a_dead_end():
    m = scene_map(props=100.0)
    assert m.scene_kind((2, 0), (1, 0)) == 'scene'
    m.cell((2, 0)).prop_n = 0.0
    assert m.scene_kind((2, 0), (1, 0)) == 'dead_end'


def test_scene_kind_needs_all_three_other_sides_seen_as_wall():
    m = scene_map()
    m.edge(A.edge_key((2, 0), (3, 0))).state = A.OPEN
    assert m.scene_kind((2, 0), (1, 0)) is None                  # a corridor going on: drive through
    m.edge(A.edge_key((2, 0), (3, 0))).state = A.UNKNOWN
    assert m.scene_kind((2, 0), (1, 0)) is None                  # far wall not seen yet: as before


def test_scene_kind_never_for_visited_or_stairs_cells():
    m = scene_map()
    m.mark_stairs((2, 0), 'lidar')
    assert m.scene_kind((2, 0), (1, 0)) is None
    m = scene_map()
    m.force_visit((2, 0), 1.0)
    assert m.scene_kind((2, 0), (1, 0)) is None


def test_props_counted_inside_the_cell_above_the_rubble_height():
    m = A.ArenaMap(CFG)
    rng = np.random.default_rng(0)
    xy = np.column_stack([2.04 + rng.uniform(-0.3, 0.3, 200), rng.uniform(-0.3, 0.3, 200)])
    m._observe_cells(np.column_stack([xy, np.full(200, 0.6)]), 1.02, 0.0)       # a cabinet
    assert m.cell((2, 0)).prop_n >= CFG['scene_prop_min']
    m2 = A.ArenaMap(CFG)
    m2._observe_cells(np.column_stack([xy, np.full(200, 0.15)]), 1.02, 0.0)     # rubble / boards
    assert m2.cell((2, 0)).prop_n == 0


# ---------------------------------------------------------------- the doorway look
def test_scene_is_looked_at_from_the_doorway_front_left_right_and_never_entered():
    m = scene_map(props=100.0)
    nav = make_nav(m, (1, 0))
    dog = Dog(1.02, 0.0, 0.0)
    outs = drive(nav, dog, 30.0, answer_after=1.0)
    assert events(outs, 'scene') and events(outs, 'scene')[0]['kind'] == 'scene'
    assert events(outs, 'scene_done')
    assert m.visited((2, 0)) and m.is_blocked((1, 0), (2, 0), dog.t)
    assert max(o.cmd.vx for o in outs) <= 0.0                    # never walked towards the scene
    looks = [o.look for o in outs if o.look]
    assert {tuple(lk['cell']) for lk in looks} == {(2, 0)}
    assert {lk['look'] for lk in looks} >= {0, 1, 2}
    assert any(o.cmd.wz > 0 for o in outs) and any(o.cmd.wz < 0 for o in outs)   # left and right looks
    assert m.cells[(2, 0)].labels.get('look') == 'scene'


def test_front_look_waits_for_perception_then_times_out():
    m = scene_map(props=100.0)
    nav = make_nav(m, (1, 0))
    dog = Dog(1.02, 0.0, 0.0)
    outs = drive(nav, dog, 6.0)                                   # no answer yet: still on the front look
    assert all(o.look and o.look['look'] == 0 for o in outs if o.look) and not events(outs, 'scene_done')
    outs = drive(nav, dog, 25.0)                                  # scene_look_max_s, then the side looks
    assert events(outs, 'scene_done') and events(outs, 'scene_done')[0]['answered'] is False


def test_dead_end_gets_the_front_look_only():
    m = scene_map(props=0.0)
    nav = make_nav(m, (1, 0))
    dog = Dog(1.02, 0.0, 0.0)
    outs = drive(nav, dog, 10.0, perception=False)
    assert events(outs, 'scene')[0]['kind'] == 'dead_end' and events(outs, 'scene_done')
    assert all(o.cmd.wz == 0.0 for o in outs[:len(outs) - 1] if o.look)       # no side looks
    assert max(o.cmd.vx for o in outs) <= 0.0


def test_scene_seen_late_dog_past_the_centre_backs_to_the_centre_first():
    m = scene_map(props=100.0)
    nav = make_nav(m, (1, 0))
    dog = Dog(1.02 + 0.35, 0.0, 0.0)                              # walked on towards the doorway
    outs = drive(nav, dog, 1.0)
    assert outs[0].cmd.vx < 0 and outs[0].cmd.wz == 0.0 and outs[0].reverse_ok
    outs = drive(nav, dog, 6.0, answer_after=0.0)
    assert events(outs, 'scene') and abs(dog.x - 1.02) < 0.15


def test_base_over_the_doorway_backs_out_of_the_scene_cell():
    m = scene_map(props=100.0)
    nav = make_nav(m, (2, 0), came='S')                          # base_link already in (2, 0)
    dog = Dog(1.60, 0.0, 0.0)
    outs = drive(nav, dog, 8.0)
    assert outs[0].cmd.vx < 0 and outs[0].reverse_ok
    assert m.cell_of(dog.x, dog.y) == (1, 0) and events(outs, 'scene')


def test_scene_look_not_for_a_cell_outside_the_arena():
    m = scene_map(props=100.0)
    for h in m.hyps:
        h['alive'] = h['rect'] == (-1, 1, 0, 4)                   # i -1..1: (2, 0) is outside the arena
    assert not m.possible((2, 0)) and m.possible((1, 0))
    nav = make_nav(m, (1, 0))
    assert nav._scene_at((1, 0), (2, 0)) is None


def test_cell_behind_a_curtain_is_not_a_scene_look():
    m = scene_map(props=100.0)
    e = m.edge(A.edge_key((1, 0), (2, 0)))
    e.state, e.was_curtain = A.CURTAIN, True
    nav = make_nav(m, (1, 0))
    assert nav._scene_at((1, 0), (2, 0)) is None                 # push through it (15 points)
    nav2 = make_nav(m, (2, 0), came='S')                         # pushed through: no backing out
    assert nav2._scene_at((1, 0), (2, 0)) is None


# ---------------------------------------------------------------- reversing
def back_way_map():
    """(1,0) is a visited dead end; the unexplored cell (0,-1) is back through (0,0)."""
    return boxed_map(opens=[((0, 0), (1, 0)), ((0, 0), (0, -1))], visited=[(0, 0), (1, 0)])


def test_plan_back_the_way_it_came_reverses_to_the_previous_centre_then_turns():
    m = back_way_map()
    nav = make_nav(m, (1, 0), came='S')
    dog = Dog(1.02, 0.0, 0.0)
    outs = drive(nav, dog, 12.0)
    assert outs[0].cmd.vx < 0 and outs[0].cmd.wz == 0.0 and outs[0].reverse_ok
    rev = [o for o in outs if o.cmd.vx < 0]
    assert abs(dog.x) < 0.30 or dog.y < -0.2                      # back in (0, 0) (or already heading on)
    assert any(o.cmd.wz != 0 for o in outs[len(rev):])            # then turns towards (0, -1)


def test_dog_that_does_not_reverse_turns_round_instead():
    m = back_way_map()
    nav = make_nav(m, (1, 0), came='S')
    dog = Dog(1.02, 0.0, 0.0, reverses=False)
    outs = drive(nav, dog, 8.0)
    assert events(outs, 'grid') and any('does not move backwards' in e['grid'] for e in events(outs, 'grid'))
    assert nav.back_ok is False
    after = outs[-20:]
    assert all(o.cmd.vx >= 0 for o in after) and any(o.cmd.wz != 0 for o in after)


def test_guard_stopping_the_reverse_for_stale_feeds_is_not_a_failed_reverse():
    m = back_way_map()
    nav = make_nav(m, (1, 0), came='S')
    dog = Dog(1.02, 0.0, 0.0, guard_passes_reverse=False)
    drive(nav, dog, 6.0)
    assert nav.back_ok is True


def test_no_reverse_when_switched_off():
    assert config.load()['grid_reverse'] is False                 # default since run 093607
    cfg = dict(CFG, grid_reverse=False)
    m = back_way_map()
    nav = CellNav(cfg, m)
    nav.paused, nav.last_cell, nav.came_from[(1, 0)] = set(m.order), (1, 0), 'S'
    dog = Dog(1.02, 0.0, 0.0)
    out = nav.step(dog.ni())
    assert out.cmd.vx >= 0 and out.cmd.wz != 0                    # turns round as before


# ---------------------------------------------------------------- tilt (rule removed 2026-11-26)
def test_tilted_dog_still_turns_on_the_spot():
    """The no-turn-while-tilted rule held the dog on the uneven terrain (run 094043): removed."""
    m = scene_map(props=100.0)
    nav = make_nav(m, (1, 0))
    dog = Dog(1.02, 0.0, 0.0)
    outs = drive(nav, dog, 30.0, answer_after=0.0)
    assert any(o.cmd.wz != 0.0 for o in outs) and not any(o.hold for o in outs)
    assert not hasattr(nav, '_tilt_turn') and 'tilt_turn_deg' not in config.load()


# ---------------------------------------------------------------- config
@pytest.mark.parametrize('mode', ['cells', 'planner', 'Grid', 'off'])
def test_unknown_nav_mode_is_refused(mode):
    with pytest.raises(ValueError):
        config.load(None, ['nav_mode=%s' % mode])


def test_known_nav_modes_load():
    assert config.load(None, ['nav_mode=wall'])['nav_mode'] == 'wall'
    assert config.load()['nav_mode'] == 'grid'
