"""Curtain (rc/curtain.py): detector on real walls and simulated cloth, camera answers, guard
exceptions, and the whole pass in the simulator (cloth lidar model, camera check, touch test)."""
import math
import os

import numpy as np
import pytest

from rc import config
from rc.curtain import CurtainDetector, CurtainPass
from rc.guard import Cmd, Guard, GuardInput
from rc.sensing import Mount, level_points
from sim import world as W
from sim.harness import simulate
from sim.lidar import Lidar
from sim.vlmsim import VlmSim

CFG = config.load()
FLOOR = -0.32
DATA = os.path.join(os.path.dirname(__file__), 'data', 'curtain_negatives.npz')
FRESH = {'livox': 0.05, 'imu': 0.01, 'joints': 0.01, 'odom': 0.02}


def scans(world, n=8, x=0.0, seed=0, fold=0.03):
    lid = Lidar(mount_xyz=CFG['mount_xyz'], mount_pitch_deg=CFG['mount_pitch_deg'], seed=seed, cloth_fold_m=fold)
    m = Mount(np.array(CFG['mount_xyz']), math.radians(CFG['mount_pitch_deg']))
    return [level_points(lid.scan(world, x, 0.0, 0.32, 0.0), m, 0.0, 0.0) for _ in range(n)]


def run_det(frames, floor=FLOOR):
    d = CurtainDetector(CFG)
    out = [d.update(p, floor, 0.1 * k) for k, p in enumerate(frames)]
    return d, out


def corridor(kind=None):
    w = W.World()
    w.wall(-1.0, -0.7, 8.0, -0.7, h=1.0)
    w.wall(-1.0, 0.7, 8.0, 0.7, h=1.0)
    w.wall(8.0, -0.7, 8.0, 0.7, h=1.0)
    w.wall(-1.0, -0.7, -1.0, 0.7, h=1.0)
    if kind == 'curtain':
        w.curtain(2.5, -0.7, 2.5, 0.7)
    elif kind == 'stiff':
        w.curtain(2.5, -0.7, 2.5, 0.7, stiff=True)
    elif kind == 'wall':
        w.wall(2.5, -0.7, 2.5, 0.7, h=1.0)
    return w


# ---------------------------------------------------------------- detector
def test_flat_wall_is_not_cloth():
    d, out = run_det(scans(corridor('wall'), x=1.3))
    assert out[-1] is not None and out[-1].tall >= CFG['cur_tall_min']
    assert not d.confident()


def taut_scans(hem, n=8):
    from sim.lidar import Lidar as L_
    m = Mount(np.array(CFG['mount_xyz']), math.radians(CFG['mount_pitch_deg']))
    lid = L_(mount_xyz=CFG['mount_xyz'], mount_pitch_deg=CFG['mount_pitch_deg'], cloth_fold_m=0.01,
             cloth_flutter_m=0.0, cloth_see_through=0.0)
    w = corridor()
    w.curtain(2.5, -0.7, 2.5, 0.7, z0=hem)
    return [level_points(lid.scan(w, 1.3, 0.0, 0.32, 0.0), m, 0.0, 0.0) for _ in range(n)]


def test_booklet_curtain_is_hanging_even_taut():
    # booklet v6.3 D.4: hung from a bar at ~1 m, hem ~0.45 m. Even taut (no folds, no flutter, not
    # see-through) the lidar sees it OPEN BELOW - a wall reaches the floor - from 1.2 m standing
    d, out = run_det(taut_scans(hem=0.45))
    s = out[-1]
    assert s is not None and s.dist == pytest.approx(2.5 - 1.3, abs=0.1)
    assert abs(s.bearing) < math.radians(5)
    assert s.open_below >= CFG['cur_open_push']
    assert d.confident() and d.hanging()


def test_floor_length_taut_sheet_needs_the_camera():
    # a taut sheet down to the floor is a wall to any lidar: never 'hanging' (no push without the
    # camera); the 'stuck + tall surface -> ask the camera' path handles it (closed loop)
    d, out = run_det(taut_scans(hem=0.03))
    assert out[-1] is not None and out[-1].tall >= CFG['cur_tall_min']
    assert out[-1].open_below < CFG['cur_open_lo']
    assert not d.hanging()


def test_low_obstacle_is_never_a_curtain():
    w = corridor()
    w.board(2.5, 0.0, 0.3, 1.3, 0.25)                   # wide but low (a box / ramp edge)
    d, out = run_det(scans(w, x=1.3))
    assert all(s is None or s.score == 0.0 for s in out) and not d.confident()


@pytest.mark.skipif(not os.path.exists(DATA), reason='fixture missing')
def test_real_walls_are_not_candidates():
    # real scans: start-bay walls (095924), the most cloth-like real wall (113959 t=32.5 s: one
    # scan scores 0.52), and the toy market stall + wire shopping trolley (113959 t=63.5-64.8 s:
    # rough AND see-through - only the unstable plane gives it away)
    z = np.load(DATA)
    for w in (0, 1, 2):
        meta = z['w%d_meta' % w]
        d = CurtainDetector(CFG)
        for j in range(meta.shape[0]):
            d.update(z['w%d_f%d' % (w, j)].astype(float), float(meta[j, 1]), float(meta[j, 0]), float(meta[j, 2]))
            assert not d.hanging(), str(z['w%d_bag' % w])        # never a push without the camera
            assert not d.confident(), str(z['w%d_bag' % w])


# ---------------------------------------------------------------- camera answers
def test_camera_cannot_check_is_unknown_not_no():
    p = CurtainPass(CFG)
    p.ask_id = 'rc1'
    p.on_answer({'id': 'rc1', 'answer': False, 'text': 'no image'})
    assert p.answer == 'unknown'
    p.on_answer({'id': 'rc1', 'answer': False, 'text': 'query failed: timeout', 'failed': True})
    assert p.answer == 'unknown'
    p.on_answer({'id': 'other', 'answer': True})
    assert p.answer == 'unknown'                        # not our question
    p.on_answer({'id': 'rc1', 'answer': True, 'text': 'a hanging cloth curtain'})
    assert p.answer is True


# ---------------------------------------------------------------- guard exceptions
def test_guard_push_relaxes_front_stop_at_creeping_speed_only():
    g = Guard(CFG)
    wall = np.array([[0.30, 0.0]])                      # 0.11 m in front of the lidar
    c, r = g.filter(Cmd(0.2, 0.0), GuardInput(obs_now=wall, ages=dict(FRESH)))
    assert c.vx == 0.0 and 'front_stop' in r
    c, r = g.filter(Cmd(0.2, 0.0), GuardInput(obs_now=wall, ages=dict(FRESH), push=True))
    assert c.vx == pytest.approx(CFG['cur_push_vx']) and 'push_cap' in r


def test_guard_reverse_only_while_retreating():
    g = Guard(CFG)
    c, r = g.filter(Cmd(-0.2, 0.3), GuardInput(ages=dict(FRESH)))
    assert c.vx == 0.0 and 'no_reverse' in r
    c, r = g.filter(Cmd(-0.2, 0.3), GuardInput(ages=dict(FRESH), retreat=True))
    assert c.vx == pytest.approx(-CFG['cur_retreat_vx']) and c.wz == 0.0
    c, r = g.filter(Cmd(-0.2, 0.0), GuardInput(ages=dict(FRESH, imu=0.5), retreat=True))
    assert c.vx == 0.0                                  # stale feeds still stop everything


# ---------------------------------------------------------------- closed loop
def contacts_hook(monkeypatch):
    import sim.dogmodel as D
    orig = D.DogModel.step

    def step(self, vx, wz, dt):
        before = self.collisions
        r = orig(self, vx, wz, dt)
        if self.collisions > before:
            self.contacts = getattr(self, 'contacts', []) + [self.t]
        return r
    monkeypatch.setattr(D.DogModel, 'step', step)


def go(kind, camera, monkeypatch, T=45.0, extra=()):
    contacts_hook(monkeypatch)
    w = corridor(kind)
    r = simulate(w, config.load(overrides=['run_time_s=%d' % T] + list(extra)), T=T, vlm=VlmSim(w, mode=camera))
    ev = [e for e in r.events if 'curtain' in e]
    return r, ev, getattr(r.dog, 'contacts', [])


def first(ev, what):
    hit = [e for e in ev if e.get('curtain') == what]
    return hit[0] if hit else None


@pytest.mark.parametrize('camera', ['ok', 'down'])
def test_sim_curtain_is_passed(camera, monkeypatch):
    # camera ok: lidar AND camera AND touch test; camera down: lidar + touch test (spec fallback)
    r, ev, contacts = go('curtain', camera, monkeypatch)
    assert first(ev, 'passed') is not None and first(ev, 'passed')['t'] < 35.0
    assert max(p[0] for p in r.path) > 4.0              # through, and on down the corridor
    assert contacts == [] and not r.reversed


def test_sim_camera_says_no_is_not_pushed(monkeypatch):
    r, ev, contacts = go('curtain', 'no', monkeypatch)
    assert first(ev, 'camera_no') is not None and first(ev, 'push') is None
    assert max(p[0] for p in r.path) < 2.5 and contacts == []


@pytest.mark.parametrize('camera', ['ok', 'down'])
def test_sim_rigid_look_alike_fails_the_touch_test(camera, monkeypatch):
    # looks like cloth to the lidar AND the camera ('ok' says yes): only touching tells
    r, ev, contacts = go('stiff', camera, monkeypatch)
    push, touch, wall = first(ev, 'push'), first(ev, 'touch_wall'), first(ev, 'wall')
    assert push and touch and wall and first(ev, 'passed') is None
    assert touch['t'] - push['t'] < CFG['cur_stall_s'] + CFG['cur_grace_s'] + 2.5
    assert contacts and all(push['t'] <= c <= touch['t'] + 0.1 for c in contacts)   # touched only while pushing
    assert wall['back'] <= CFG['cur_retreat_m'] + 0.03
    assert sum(1 for e in ev if e.get('curtain') == 'push') == 1                     # marked: never pushed again


def doorway_room():
    w = W.World()
    w.wall(-1.0, -2.0, 6.0, -2.0, h=1.0)
    w.wall(6.0, -2.0, 6.0, 2.0, h=1.0)
    w.wall(6.0, 2.0, -1.0, 2.0, h=1.0)
    w.wall(-1.0, 2.0, -1.0, -2.0, h=1.0)
    w.wall(2.5, -2.0, 2.5, -0.5, h=1.0)                 # panel wall with a 1 m doorway ...
    w.wall(2.5, 0.5, 2.5, 2.0, h=1.0)
    w.curtain(2.5, -0.5, 2.5, 0.5)                      # ... closed by a curtain, flush with the wall
    return w


@pytest.mark.parametrize('yaw,camera', [(0.0, 'ok'), (-25.0, 'ok'), (20.0, 'ok'), (0.0, 'down')])
def test_sim_curtained_doorway_in_a_wall_is_passed(yaw, camera, monkeypatch):
    # the cloth is a patch of a longer surface: found by the sliding windows, squared up, and - when
    # the dog's line would put its body on the door post (-25, +20 deg) - lined up in front of it
    contacts_hook(monkeypatch)
    w = doorway_room()
    r = simulate(w, config.load(overrides=['run_time_s=55']), T=55.0, start=(0.0, 0.0, math.radians(yaw)),
                 vlm=VlmSim(w, mode=camera))
    ev = [e for e in r.events if 'curtain' in e]
    assert first(ev, 'passed') is not None
    assert max(p[0] for p in r.path) > 3.5 and getattr(r.dog, 'contacts', []) == []


def test_sim_plain_wall_across_is_never_pushed(monkeypatch):
    r, ev, contacts = go('wall', 'ok', monkeypatch)
    assert first(ev, 'push') is None and contacts == []


def test_sim_curtain_off(monkeypatch):
    r, ev, contacts = go('curtain', 'ok', monkeypatch, extra=['curtain=off'])
    assert ev == [] and r.asks == 0 and max(p[0] for p in r.path) < 2.5
