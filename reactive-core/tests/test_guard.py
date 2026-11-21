import os

import numpy as np
import pytest

from rc import config
from rc.guard import Cmd, Guard, GuardInput
from rc.sensing import ObstacleDetector

CFG = config.load()
E = np.zeros((0, 2))
FRESH = {'livox': 0.05, 'imu': 0.01, 'joints': 0.01, 'odom': 0.02}


def gi(obs=E, mem=E, mem_ok=True, ages=None, tilt=0.0, stairs=False):
    return GuardInput(obs_now=obs, mem=mem, mem_ok=mem_ok, ages=dict(FRESH if ages is None else ages),
                      tilt_deg=tilt, stairs=stairs)


def test_passthrough():
    c, r = Guard(CFG).filter(Cmd(0.15, 0.1), gi())
    assert (c.vx, c.wz) == (0.15, 0.1) and r == []


def test_stale_feed_stops():
    ages = dict(FRESH, imu=0.4)
    c, r = Guard(CFG).filter(Cmd(0.15, 0.2), gi(ages=ages))
    assert (c.vx, c.wz) == (0.0, 0.0) and 'stale:imu' in r


def test_missing_feed_counts_as_stale():
    ages = dict(FRESH, odom=float('inf'))
    c, r = Guard(CFG).filter(Cmd(0.15, 0.0), gi(ages=ages))
    assert c.vx == 0.0


def test_tilt_levels():
    g = Guard(CFG)
    c, r = g.filter(Cmd(0.2, 0.0), gi(tilt=16))
    assert c.vx == pytest.approx(0.1) and 'tilt_slow' in r
    c, r = g.filter(Cmd(0.2, 0.0), gi(tilt=22))
    assert c.vx == 0.0 and 'tilt_stop' in r
    c, r = g.filter(Cmd(0.2, 0.0), gi(tilt=32))
    assert g.latched
    c, r = g.filter(Cmd(0.2, 0.0), gi(tilt=0))
    assert (c.vx, c.wz) == (0.0, 0.0) and 'tilt_latch' in r


def stairs_gi(roll, pitch):
    x = gi(tilt=max(abs(roll), abs(pitch)), stairs=True)
    x.roll_deg, x.pitch_deg = roll, pitch
    return x


def test_stairs_pitch_allowed_roll_not():
    g = Guard(CFG)
    c, r = g.filter(Cmd(0.15, 0.0), stairs_gi(2.0, 31.0))      # 0.15/0.30 stairs pitch ~27-31 deg
    assert c.vx == pytest.approx(0.15) and not g.latched
    c, r = g.filter(Cmd(0.15, 0.0), stairs_gi(22.0, 5.0))      # rolling over: stop
    assert c.vx == 0.0 and 'tilt_stop' in r
    c, r = g.filter(Cmd(0.15, 0.0), stairs_gi(2.0, 36.0))
    assert c.vx == 0.0 and not g.latched
    c, r = g.filter(Cmd(0.15, 0.0), stairs_gi(2.0, 41.0))
    assert g.latched


def test_no_reverse():
    c, r = Guard(CFG).filter(Cmd(-0.1, 0.0), gi())
    assert c.vx == 0.0 and 'no_reverse' in r


def test_front_stop_and_slow():
    g = Guard(CFG)
    wall = np.array([[0.18678 + 0.30, 0.0]])
    c, r = g.filter(Cmd(0.2, 0.0), gi(obs=wall))
    assert c.vx == 0.0 and 'front_stop' in r
    mid = np.array([[0.18678 + 0.575, 0.0]])            # halfway between 0.35 and 0.80
    c, r = g.filter(Cmd(0.2, 0.0), gi(obs=mid))
    assert c.vx == pytest.approx(0.1, abs=0.005) and 'front_slow' in r
    side = np.array([[0.5, 0.5]])                       # outside the corridor
    c, r = g.filter(Cmd(0.2, 0.0), gi(obs=side))
    assert c.vx == 0.2


def test_front_ignores_memory():
    c, _ = Guard(CFG).filter(Cmd(0.2, 0.0), gi(mem=np.array([[0.4, 0.0]])))
    assert c.vx == 0.2


def test_turn_sweep_blocks_only_the_side_with_the_obstacle():
    g = Guard(CFG)
    # low box just left of the rear-left corner (only known from memory: blind zone)
    box = np.array([[-0.38, 0.28], [-0.36, 0.30]])
    c, r = g.filter(Cmd(0.0, 0.4), gi(mem=box))        # turning left swings the rear to the right...
    c2, r2 = g.filter(Cmd(0.0, -0.4), gi(mem=box))     # ...turning right swings the rear left into it
    assert (c.wz == 0.0) != (c2.wz == 0.0)
    assert c2.wz == 0.0 and 'sweep_block_right' in r2
    assert c.wz == 0.4


def test_turn_sweep_without_memory():
    g = Guard(CFG)
    box = np.array([[-0.38, 0.28]])
    c, r = g.filter(Cmd(0.0, -0.4), gi(mem=box, mem_ok=False))
    assert c.wz == pytest.approx(-0.2) and 'no_memory' in r


def test_points_inside_footprint_ignored():
    c, r = Guard(CFG).filter(Cmd(0.0, 0.4), gi(obs=np.array([[0.0, 0.0], [0.1, 0.1]])))
    assert c.wz == 0.4


def test_blocked_both():
    g = Guard(CFG)
    walls = np.array([[0.2, 0.30], [0.2, -0.30], [-0.2, 0.30], [-0.2, -0.30]])
    assert g.blocked_both(gi(obs=walls))
    assert not g.blocked_both(gi())


DATA = os.path.join(os.path.dirname(__file__), 'data', 'trial3_run1.npz')


@pytest.mark.skipif(not os.path.exists(DATA), reason='fixture missing')
@pytest.mark.parametrize('t', [40.5, 41.5])
def test_real_wall_stops(t):
    d = np.load(DATA)
    i = int(np.argmin(np.abs(d['times'] - t)))
    s = ObstacleDetector(CFG).update(d['pts_%d' % i].astype(float), -0.32, 0.0)   # single scan
    c, r = Guard(CFG).filter(Cmd(0.2, 0.0), gi(obs=s.obs_now))
    assert c.vx < 0.05, (c, r)
