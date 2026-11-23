import os

import numpy as np
import pytest

from rc import config
from rc.guard import Cmd, Guard, GuardInput
from rc.sensing import ObstacleDetector

CFG = config.load()
E = np.zeros((0, 2))
FRESH = {'livox': 0.05, 'imu': 0.01, 'joints': 0.01, 'odom': 0.02}
SX = 0.18678                                   # sensor x in base_link


def gi(obs=E, ages=None, tilt=0.0):
    return GuardInput(obs_now=obs, ages=dict(FRESH if ages is None else ages), tilt_deg=tilt)


def test_passthrough():
    c, r = Guard(CFG).filter(Cmd(0.15, 0.0), gi())
    assert (c.vx, c.wz) == (0.15, 0.0) and r == []
    c, r = Guard(CFG).filter(Cmd(0.0, 0.3), gi())
    assert (c.vx, c.wz) == (0.0, 0.3) and r == []


def test_walk_and_turn_never_both_turn_is_dropped():
    # 2026-11-23: the dog slipped at ledges while walking and turning at once. Backstop: a forward
    # command that also turns loses the turn (the driver should never send one)
    c, r = Guard(CFG).filter(Cmd(0.15, 0.1), gi())
    assert (c.vx, c.wz) == (0.15, 0.0) and r == ['walk_no_turn']


def test_stale_feed_stops():
    ages = dict(FRESH, imu=0.4)
    c, r = Guard(CFG).filter(Cmd(0.15, 0.2), gi(ages=ages))
    assert (c.vx, c.wz) == (0.0, 0.0) and 'stale:imu' in r


def test_missing_feed_counts_as_stale():
    ages = dict(FRESH, odom=float('inf'))
    c, r = Guard(CFG).filter(Cmd(0.15, 0.0), gi(ages=ages))
    assert c.vx == 0.0


def test_tilt_below_stop_passes_at_full_speed():
    # crossing boards / the hump tilts the body 15-28 deg (hand-driven runs): no slowing at all
    g = Guard(CFG)
    for tilt in (15, 22, 28, CFG['tilt_stop_deg'] - 0.5):
        c, r = g.filter(Cmd(0.2, 0.0), gi(tilt=tilt))
        assert (c.vx, c.wz) == (0.2, 0.0) and r == []


def test_tilt_stop_resumes_by_itself_with_hysteresis():
    g = Guard(CFG)
    c, r = g.filter(Cmd(0.2, 0.1), gi(tilt=CFG['tilt_stop_deg'] + 1))
    assert (c.vx, c.wz) == (0.0, 0.0) and r == ['tilt_stop']
    mid = (CFG['tilt_stop_deg'] + CFG['tilt_resume_deg']) / 2.0
    c, r = g.filter(Cmd(0.2, 0.1), gi(tilt=mid))            # still above the resume level: stays stopped
    assert c.vx == 0.0 and r == ['tilt_stop']
    c, r = g.filter(Cmd(0.2, 0.0), gi(tilt=CFG['tilt_resume_deg'] - 1))
    assert (c.vx, c.wz) == (0.2, 0.0) and r == [] and not g.latched


def test_tilt_latch_only_when_going_over():
    g = Guard(CFG)
    g.filter(Cmd(0.2, 0.0), gi(tilt=CFG['tilt_latch_deg'] - 1))
    assert not g.latched
    c, r = g.filter(Cmd(0.2, 0.0), gi(tilt=CFG['tilt_latch_deg'] + 1))
    assert g.latched and c.vx == 0.0 and r == ['tilt_latch']
    c, r = g.filter(Cmd(0.2, 0.0), gi(tilt=0))               # latched until restart
    assert (c.vx, c.wz) == (0.0, 0.0) and r == ['tilt_latch']


def test_stairs_slope_after_a_spike_resumes():
    # sim stairs test 2026-11-23: a spike over tilt_stop on the first step, then the 26.6 deg slope;
    # with the resume level below the slope the dog stood on the stairs until stairs_timeout
    g = Guard(CFG)
    g.filter(Cmd(0.15, 0.0), gi(tilt=CFG['tilt_stop_deg'] + 2))
    c, r = g.filter(Cmd(0.15, 0.0), gi(tilt=26.6))
    assert c.vx == pytest.approx(0.15) and r == []


def test_stairs_slope_passes():
    c, r = Guard(CFG).filter(Cmd(0.15, 0.0), gi(tilt=31.0))  # 0.15/0.30 stairs pitch ~27-31 deg
    assert c.vx == pytest.approx(0.15) and r == []


def test_no_reverse():
    c, r = Guard(CFG).filter(Cmd(-0.1, 0.0), gi())
    assert c.vx == 0.0 and 'no_reverse' in r


def test_front_stop_without_slow_zone():
    g = Guard(CFG)
    c, r = g.filter(Cmd(0.2, 0.0), gi(obs=np.array([[SX + CFG['stop_m'] - 0.02, 0.0]])))
    assert c.vx == 0.0 and r == ['front_stop']
    c, r = g.filter(Cmd(0.2, 0.0), gi(obs=np.array([[SX + CFG['stop_m'] + 0.02, 0.0]])))
    assert c.vx == 0.2 and r == []                          # just outside: full speed, no slowing
    c, r = g.filter(Cmd(0.2, 0.0), gi(obs=np.array([[0.5, 0.5]])))   # outside the corridor
    assert c.vx == 0.2


def test_front_stop_keeps_turning():
    # the front stop runs before the walk/turn backstop: a blocked front leaves a pure turn
    c, r = Guard(CFG).filter(Cmd(0.2, 0.4), gi(obs=np.array([[SX + 0.1, 0.0]])))
    assert (c.vx, c.wz) == (0.0, 0.4) and r == ['front_stop']


def test_turning_never_refused():
    # no turn sweep any more: walls right beside the body do not block turning
    walls = np.array([[0.2, 0.30], [0.2, -0.30], [-0.2, 0.30], [-0.2, -0.30]])
    c, r = Guard(CFG).filter(Cmd(0.0, 0.4), gi(obs=walls))
    assert c.wz == 0.4 and r == []


DATA = os.path.join(os.path.dirname(__file__), 'data', 'trial3_run1.npz')


@pytest.mark.skipif(not os.path.exists(DATA), reason='fixture missing')
@pytest.mark.parametrize('t', [40.5, 41.5])
def test_real_wall_seen_by_front_check(t):
    d = np.load(DATA)
    i = int(np.argmin(np.abs(d['times'] - t)))
    s = ObstacleDetector(CFG).update(d['pts_%d' % i].astype(float), -0.32, 0.0)   # single scan
    assert Guard(CFG).front_dist(s.obs_now) < 0.8
