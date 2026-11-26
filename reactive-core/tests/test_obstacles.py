"""Planner, low bar (cylinder) and curtain: detectors on synthetic points, then closed-loop sim runs."""
import collections
import math

import numpy as np
import pytest

from rc import config
from rc import geometry as G
from rc import curtain as CU
from rc.bar import BarDetector, analyse as bar_analyse
from rc.driver import Driver, DriverInput
from rc.grid import Grid
from rc.guard import Cmd
from rc.curtain import Vision
from rc.planner import Planner, lookahead_yaw
from sim import harness as H
from sim import world as W
from sim.harness import simulate
from tests import helpers as T
from tests.test_driver import line

CFG = config.load()


def names(res):
    return [n for _, n in res.codes]


def evs(res, key):
    return [e[key] for e in res.events if key in e]


# ---------------------------------------------------------------- bar detector
def bar_pts(h=0.20, depth=0.15, x0=1.0, y0=-0.6, y1=0.6, extra=None):
    parts = [T.floor(1.4, 4.0, -1.0, 1.0), T.box_top(x0, x0 + depth, y0, y1, h)]
    if extra is not None:
        parts.append(extra)
    return np.vstack(parts)


def test_bar_found_with_distance_height_and_depth():
    r = bar_analyse(bar_pts(), T.FLOOR, CFG)
    assert r.found and abs(r.dist - 1.0) < 0.1
    assert 0.15 < r.height < 0.25 and r.depth < 0.25 and r.cover >= 0.8
    assert abs(r.corr) < math.radians(3)


def test_bar_not_found_for_wall_slab_half_width_or_stairs():
    assert not bar_analyse(np.vstack([T.floor(), T.wall_x(1.0, -0.6, 0.6)]), T.FLOOR, CFG).found
    # the 0.5 m deep wooden slab of run 181509 (20 cm tall)
    assert not bar_analyse(bar_pts(depth=0.5), T.FLOOR, CFG).found
    # a bar across only half the lane
    assert not bar_analyse(bar_pts(y0=0.0, y1=0.6), T.FLOOR, CFG).found
    # the slab with a tall wall 0.5 m behind it
    assert not bar_analyse(bar_pts(depth=0.3, extra=T.wall_x(1.5, -0.6, 0.6)), T.FLOOR, CFG).found
    assert not bar_analyse(np.vstack([T.floor(), T.stairs(1.0, 4, 0.15, 0.30)]), T.FLOOR, CFG).found
    # too low to matter / too high to be a bar
    assert not bar_analyse(bar_pts(h=0.05), T.FLOOR, CFG).found
    assert not bar_analyse(bar_pts(h=0.45), T.FLOOR, CFG).found


def test_bar_squares_up_to_a_skewed_bar():
    p = bar_pts(depth=0.15)
    skew = T.rotate_z(p[(p[:, 2] > T.FLOOR + 0.05)], math.radians(10))
    r = bar_analyse(np.vstack([T.floor(), skew]), T.FLOOR, CFG)
    assert r.found and abs(abs(r.corr) - math.radians(10)) < math.radians(4)


def test_bar_detector_needs_persistence():
    d = BarDetector(CFG)
    out = [d.update(bar_pts(), T.FLOOR) for _ in range(CFG['bar_persist'])]
    assert not out[0].confident and out[-1].confident


# ---------------------------------------------------------------- curtain detector
def face(x=1.0, w=1.0):
    return np.stack([np.full(40, x), np.linspace(-w / 2, w / 2, 40)], axis=1)


def vision_with(n, conf=0.9, bearing=0.0, t0=0.0):
    v = Vision(CFG)
    for i in range(n):
        v.on_detections([{'cls': 'curtain', 'conf': conf, 'bearing': bearing}], t0 + 0.2 * i)
    return v


def test_curtain_needs_camera_and_lidar_face():
    t = 1.0
    assert CU.analyse(vision_with(4), face(), t, CFG).found
    assert not CU.analyse(vision_with(0), face(), t, CFG).found                      # no camera
    assert not CU.analyse(vision_with(2), face(), t, CFG).found                      # not persistent
    assert not CU.analyse(vision_with(4, conf=0.3), face(), t, CFG).found            # low confidence
    assert not CU.analyse(vision_with(4), np.zeros((0, 2)), t, CFG).found            # open space
    assert not CU.analyse(vision_with(4), face(w=0.2), t, CFG).found                 # a pole, not a face
    assert not CU.analyse(vision_with(4), face(x=3.0), t, CFG).found                 # too far
    assert not CU.analyse(vision_with(4, bearing=math.radians(60)), face(), t, CFG).found   # outside the cone


def test_curtain_detections_expire():
    v = vision_with(4)
    assert CU.analyse(v, face(), 1.0, CFG).found
    assert not CU.analyse(v, face(), 5.0, CFG).found


# ---------------------------------------------------------------- planner
def corridor_cells(x0=-1.0, x1=3.0, half=0.7, closed_at=None):
    parts = [line(x0, half, x1, half, 60), line(x0, -half, x1, -half, 60)]
    if closed_at is not None:
        parts.append(line(closed_at, -half, closed_at, half, 30))
    return np.vstack(parts)


def feed(pl, pose, cells, n=4):
    for _ in range(n):
        pl.update(pose, cells, np.zeros((0, 2)))
    return pl.plan(pose, 100.0 + pl.updates, force=True)


def test_planner_heads_down_an_open_corridor():
    pl = Planner(CFG)
    r = feed(pl, (0.0, 0.0, 0.0), corridor_cells())
    assert r.yaw is not None and abs(r.yaw) < math.radians(15)
    assert r.goal[0] > 1.0


def test_planner_turns_for_the_only_way_out():
    # closed ahead at 1.8 m, a side opening to the left (wall on the left stops at x=0.8)
    cells = np.vstack([line(-1.0, -0.7, 1.8, -0.7, 50), line(1.8, -0.7, 1.8, 0.7, 20),
                       line(-1.0, 0.7, 0.8, 0.7, 30)])
    pl = Planner(CFG)
    r = feed(pl, (0.0, 0.0, 0.0), cells)
    assert r.yaw is not None and r.yaw > math.radians(8)         # bends towards the left opening
    assert r.goal[1] > 0.5


def test_planner_has_no_advice_in_a_closed_room():
    ring = np.vstack([line(-1.5, -1.5, 1.5, -1.5, 40), line(1.5, -1.5, 1.5, 1.5, 40),
                      line(1.5, 1.5, -1.5, 1.5, 40), line(-1.5, 1.5, -1.5, -1.5, 40)])
    pl = Planner(CFG)
    # facing +x the rear is blind: a frontier behind the dog is correct, so look round first
    r = None
    for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
        pose = (0.0, 0.0, yaw)
        body = G.world_to_body(ring, pose)
        for _ in range(3):
            pl.update(pose, body, np.zeros((0, 2)))
        r = pl.plan(pose, 100.0 + pl.updates, force=True)
    assert r.yaw is None and r.reason in ('no frontier', 'frontier too close')


def test_lookahead_yaw_picks_the_first_point_a_metre_away():
    path = [(0.3, 0.0), (0.6, 0.0), (1.0, 0.1), (1.5, 0.6)]
    assert lookahead_yaw(path, (0.0, 0.0, 0.0), 1.0) == pytest.approx(math.atan2(0.1, 1.0))
    assert lookahead_yaw(path[:2], (0.0, 0.0, 0.0), 1.0) == pytest.approx(0.0)          # all nearer: the last


def test_external_nav2_path_replaces_the_own_one_while_fresh():
    pl = Planner(CFG)
    feed(pl, (0.0, 0.0, 0.0), corridor_cells())
    pl.set_external_path([(0.5, 0.2), (1.0, 0.8), (1.2, 1.5)], 200.0)
    r = pl.plan((0.0, 0.0, 0.0), 200.5, force=True)
    assert r.reason == 'nav2' and r.yaw == pytest.approx(math.atan2(0.8, 1.0))
    r = pl.plan((0.0, 0.0, 0.0), 205.0, force=True)               # stale: back to the own planner
    assert r.reason != 'nav2'


def test_planner_blacklists_a_failed_goal_and_resets():
    pl = Planner(CFG)
    feed(pl, (0.0, 0.0, 0.0), corridor_cells())
    assert pl.goal is not None
    pl.mark_failed(10.0)
    assert pl.goal is None and pl.blacklist
    pl.reset()
    assert not pl.seen.any() and not pl.blacklist


# ---------------------------------------------------------------- closed loop
def test_sim_curtain_is_pushed_through():
    r = simulate(W.curtain_world(4.0), CFG, T=70.0)
    assert r.startup == 'ready'
    assert evs(r, 'curtain')[:3] == ['ALIGN', 'PUSH', 'DONE']
    assert r.end_pose[0] > 5.0 and r.collisions == 0
    assert r.final_mode != 'HOLD'


def test_sim_curtain_mode_detect_never_pushes():
    r = simulate(W.curtain_world(4.0), config.load(overrides=['curtain=detect']), T=45.0)
    assert 'PUSH' not in evs(r, 'curtain') and evs(r, 'curtain_seen')
    assert r.end_pose[0] < 4.0


def test_sim_solid_wall_is_not_a_curtain():
    # the camera says nothing (the box is a wall): the dog must never push
    r = simulate(W.curtain_world(4.0, cloth=False), CFG, T=60.0)
    assert evs(r, 'curtain') == [] and r.collisions == 0


def _push_driver():
    d = Driver(CFG, Grid(1.0))
    d.mode, d.phase = 'CURTAIN', 'PUSH'
    d.curtain_s = {'t0': 0.0, 'tp': 0.0, 'dist': 1.2, 'bearing': 0.0, 'yaw': 0.0, 'd0': 0.0,
                   'tc': 0.0, 'fd': collections.deque(), 'limit': 1.3 * (1.2 + CFG['curtain_through_m']) / CFG['curtain_vx'] + 3.0}
    return d


def _push(d, front_of, secs=30.0):
    ev, t = [], 0.0
    while t < secs and d.mode == 'CURTAIN':
        di = DriverInput(t=t, lanes=[], obs=np.zeros((0, 2)), mem=np.zeros((0, 2)), pose=(0.0, 0.0, 0.0),
                         odom_ok=True, travelled=0.12 * t, last_cmd=Cmd(0.12, 0.0), front_dist=front_of(t))
        d.events = []
        d._curtain(di)
        ev += [e['curtain'] for e in d.events if 'curtain' in e]
        t += 0.1
    return ev, d


def test_curtain_touch_test_flags_a_wall_that_does_not_come_closer():
    ev, d = _push(_push_driver(), lambda t: 1.0)          # walking, but the face stays 1.0 m away
    assert ev == ['WALL'] and d.mode == 'CRUISE' and d.front_ignore_x == 0.0
    assert d.curtain_cool_until > 20.0                     # the camera is ignored for a while


def test_curtain_touch_test_lets_a_closing_face_through():
    ev, d = _push(_push_driver(), lambda t: 1.2 - 0.12 * t if t < 7.5 else 3.0)   # closes at 0.12 m/s, then the sensor is inside
    assert ev == ['DONE'] and d.mode == 'CRUISE'


def test_curtain_touch_test_ignores_a_face_that_vanishes_inside_the_cloth():
    # inside ~0.3 m the lidar drops the face (zero-range filter): front_dist jumps to 3.0, no stall
    ev, d = _push(_push_driver(), lambda t: 3.0 if t > 6.0 else 1.2 - 0.12 * t)
    assert ev == ['DONE']


def test_sim_wall_taken_for_a_curtain_is_caught_by_the_touch_test(monkeypatch):
    # the camera claims a curtain in front of a solid wall: the dog walks up to it, the face stops
    # closing (a tall wall stays in the lidar until contact: front 0.25 m), WALL, never DONE, then
    # it turns away instead of pushing again (cooldown)
    monkeypatch.setattr(H, 'fake_camera', lambda w, d: [{'cls': 'curtain', 'conf': 0.9, 'bearing': 0.0}])
    r = simulate(W.curtain_world(4.0, cloth=False), CFG, T=80.0)
    seq = evs(r, 'curtain')
    assert seq[:3] == ['ALIGN', 'PUSH', 'WALL'] and 'DONE' not in seq and seq.count('PUSH') == 1
    assert not r.reversed and r.end_pose[0] < 4.0


@pytest.mark.parametrize('h', [0.20, 0.30])
def test_sim_bar_is_crossed(h):
    r = simulate(W.bar_world(3.0, h=h), CFG, T=60.0)
    assert evs(r, 'bar')[:3] == ['SQUARE', 'CROSS', 'DONE']
    assert r.end_pose[0] > 4.0 and r.collisions == 0 and not r.reversed
    assert r.final_mode != 'HOLD'


def test_sim_bar_mode_detect_never_crosses():
    r = simulate(W.bar_world(3.0, h=0.30), config.load(overrides=['bar=detect']), T=40.0)
    assert 'CROSS' not in evs(r, 'bar') and evs(r, 'bar_seen')
    assert r.end_pose[0] < 3.0                                      # a 0.30 m bar blocks the lane


def test_sim_deep_slab_is_not_taken_for_a_bar():
    r = simulate(W.bar_world(3.0, h=0.20, depth=0.6), CFG, T=40.0)
    assert evs(r, 'bar') == []


def test_bar_and_curtain_off_switches_accept_yaml_off():
    cfg = config.load(overrides=['bar=off', 'curtain=off'])         # YAML turns bare `off` into False
    assert config.mode(cfg, 'bar') == 'off' and config.mode(cfg, 'curtain') == 'off'
    r = simulate(W.curtain_world(4.0), cfg, T=40.0)
    assert evs(r, 'curtain') == []


def test_sim_planner_explores_the_arena_without_collisions():
    r = simulate(W.arena_default(), CFG, T=150.0)
    assert r.collisions == 0 and not r.reversed
    assert r.cells >= 8
