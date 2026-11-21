"""Closed-loop simulator tests (pure Python, ~1-2 min in total)."""
import math

import numpy as np
import pytest

from rc import config
from rc.sensing import Mount, level_points
from sim import world as W
from sim.harness import simulate
from sim.lidar import Lidar

CFG = config.load()


def names(res):
    return [n for _, n in res.codes]


def test_lidar_floor_starts_like_the_real_one():
    pts = Lidar(noise=0.0, blob=0).scan(W.World(), 0, 0, 0.32, 0.0)
    p = level_points(pts, Mount(np.array(CFG['mount_xyz']), math.radians(15)), 0, 0)
    fl = p[(np.abs(p[:, 1]) < 0.3) & (p[:, 0] > 0)]
    assert 1.1 < fl[:, 0].min() < 1.5          # real dog: floor returns from ~1.4 m


@pytest.fixture(scope='module')
def arena():
    return simulate(W.arena_default(), CFG, T=150.0)


def test_arena_no_collision_no_reverse(arena):
    assert arena.startup == 'ready'
    assert arena.collisions == 0
    assert not arena.reversed


def test_arena_explores(arena):
    assert arena.cells >= 8
    far = max(math.hypot(x, y) for x, y in arena.path)
    assert far > 2.0                          # left the start pit


def test_arena_never_enters_ai(arena):
    assert 'enter_ai' not in names(arena)


@pytest.fixture(scope='module')
def stairs():
    return simulate(W.stairs_world(), CFG, T=100.0)


def test_stairs_full_sequence(stairs):
    n = names(stairs)
    assert n[:2] == ['rug_general', 'rug_general']
    seq = [x for x in n[2:]]
    assert seq[:2] == ['enter_ai', 'ai_basic']
    assert seq[2:6] == ['exit_ai', 'move_mode', 'rug_general', 'rug_general']
    assert stairs.collisions == 0 and not stairs.reversed
    assert stairs.max_support >= 0.44                     # was on the platform
    assert stairs.end_pose[0] > 6.0                        # crossed and continued in RUG general
    assert any(e.get('stairs') == 'DONE' for e in stairs.events)


def test_stairs_ai_ignores_commands():
    r = simulate(W.stairs_world(), CFG, T=45.0, dog_kwargs={'ai_ignores_cmd': True})
    n = names(r)
    assert 'enter_ai' in n and 'exit_ai' in n
    assert r.final_mode == 'HOLD' and r.hold_reason == 'ai_no_response'
    assert r.max_support == 0.0 and r.collisions == 0      # never set foot on the stairs


def test_sitting_start_one_toggle():
    r = simulate(W.corridor_world(), CFG, T=15.0, dog_kwargs={'sitting': True})
    assert r.startup == 'ready'
    assert names(r).count('stand_sit') == 1


def test_leg_odom_yaw_frozen_still_turns():
    # heading comes from the gyro: a /leg_odom whose yaw never changes must not break turning
    r = simulate(W.arena_default(), CFG, T=90.0, dog_kwargs={'odom_mode': 'frozen_yaw'})
    assert r.startup == 'ready' and r.collisions == 0 and not r.reversed
    assert max(math.hypot(x, y) for x, y in r.path) > 2.0


def test_leg_odom_position_frozen_detected():
    # position never updates: the lidar sees the dog approach walls -> odometry declared frozen,
    # no phantom memory obstacles, no endless no_progress HOLDs, no collisions
    r = simulate(W.arena_default(), CFG, T=120.0, dog_kwargs={'odom_mode': 'frozen_xy'}, resume_on_hold=True)
    assert r.collisions == 0 and not r.reversed
    assert any('odom_frozen' in e for e in r.events)
    assert max(math.hypot(x, y) for x, y in r.path) > 2.0


def test_stairs_with_frozen_leg_odom():
    r = simulate(W.stairs_world(), CFG, T=110.0, dog_kwargs={'odom_mode': 'frozen_xy'}, resume_on_hold=True)
    assert r.collisions == 0
    n = names(r)
    if 'enter_ai' in n:                                    # if it climbed, it also got out of AI
        assert 'exit_ai' in n


def test_detect_mode_never_climbs():
    cfg = config.load(overrides=['stairs=detect'])
    r = simulate(W.stairs_world(), cfg, T=40.0)
    assert 'enter_ai' not in names(r)
    assert any('stairs_seen' in e for e in r.events)
