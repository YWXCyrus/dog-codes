"""QR approach (rc/qr.py): state machine, geometry, and closed loop in the simulator with a simulated
camera + perception (sim/qrcam.py)."""
import math

import numpy as np
import pytest

from rc import config
from rc.qr import CAM_X, QrApproach
from sim import world as W
from sim.harness import simulate
from sim.qrcam import QrCam

CFG = config.load()
POSE = (0.0, 0.0, 0.0)
NO_OBS = np.zeros((0, 2))


def seen(q, t, w=60.0, cx=320.0, cy=200.0):
    q.on_event({'type': 'qr_seen', 'w': w, 'cx': cx, 'cy': cy}, t)


# ---------------------------------------------------------------- trigger
def test_wide_located_shapes_are_false_locations():
    q = QrApproach(CFG)
    for k in range(6):
        seen(q, 0.1 * k, w=300.0)              # OpenCV locator on a panel: a real 300 px QR is read
    assert not q.wants(0.6, POSE, NO_OBS)


def test_needs_two_stable_detections():
    q = QrApproach(CFG)
    seen(q, 0.0)
    assert not q.wants(0.1, POSE, NO_OBS)
    seen(q, 0.2, cx=500.0)                     # jumped across the picture: not the same thing
    assert not q.wants(0.3, POSE, NO_OBS)
    seen(q, 0.4, cx=505.0)
    assert q.wants(0.5, POSE, NO_OBS)


def test_never_after_a_qr_was_read():
    q = QrApproach(CFG)
    assert q.on_event({'type': 'qr_decoded', 'text': 'kitchen'}, 0.0) == 'kitchen'
    assert q.on_event({'type': 'qr_decoded', 'text': 'kitchen'}, 0.1) is None     # first only
    seen(q, 0.2)
    seen(q, 0.3)
    assert not q.wants(0.4, POSE, NO_OBS)


def test_bad_event_ignored():
    q = QrApproach(CFG)
    q.on_event({'type': 'qr_seen', 'w': 'x'}, 0.0)
    q.on_event({'type': 'other'}, 0.0)
    assert q.obs == []


# ---------------------------------------------------------------- geometry
def test_locate_uses_the_lidar_hit_on_the_camera_ray():
    q = QrApproach(CFG)
    wall = np.array([[2.0, yy] for yy in np.arange(-1.0, 1.0, 0.1)])
    o = type('O', (), {'w': 40.0, 'cx': CFG['qr_cx0'], 'cy': 200.0})()
    p, d = q.locate(o, wall)
    assert d == pytest.approx(2.0 - CAM_X, abs=0.06) and abs(p[1]) < 1e-9


def test_locate_falls_back_to_the_width():
    q = QrApproach(CFG)
    o = type('O', (), {'w': 69.4, 'cx': CFG['qr_cx0'], 'cy': 200.0})()
    _, d = q.locate(o, NO_OBS)
    assert d == pytest.approx(CFG['qr_fx'] * CFG['qr_size_m'] / 69.4, rel=1e-6)    # ~1.0 m


def test_turn_bearing_is_seen_from_base_link():
    # QR 0.5 m to the left of the camera axis, 0.5 m ahead of the camera: base_link sees it at a
    # smaller angle than the camera (the dog turns about base_link)
    q = QrApproach(CFG)
    cx = CFG['qr_cx0'] - CFG['qr_fx'] * 1.0             # 45 deg left of the camera
    wall = np.array([[CAM_X + 0.5, 0.5]])
    o = type('O', (), {'w': 80.0, 'cx': cx, 'cy': 200.0})()
    p, _ = q.locate(o, wall)
    assert math.degrees(math.atan2(p[1], p[0])) == pytest.approx(math.degrees(math.atan2(0.5, 0.752)), abs=1.0)


def test_closest_in_picture():
    q = QrApproach(CFG)
    assert q.closest_in_picture(400.0, 1.0) == 0.0            # top edge below the horizon: no limit
    # top edge 0.10 m above the camera, 1.0 m away: elevation 5.7 deg -> row from the 20 deg tilt
    row = CFG['qr_cy0'] - CFG['qr_fx'] * math.tan(math.atan2(0.10, 1.0) + math.radians(20))
    elev_max = math.atan2(CFG['qr_cy0'] - CFG['qr_top_margin_px'], CFG['qr_fx']) - math.radians(20)
    assert q.closest_in_picture(row, 1.0) == pytest.approx(0.10 / math.tan(elev_max), rel=1e-6)


# ---------------------------------------------------------------- state machine
def test_lost_target_is_blacklisted_with_cooldown():
    q = QrApproach(CFG)
    seen(q, 0.0)
    seen(q, 0.1)
    assert q.wants(0.2, POSE, NO_OBS)
    q.start(0.2, POSE)
    t = 0.2
    while q.active and t < 10.0:
        cmd = q.step(t, POSE, 3.0, NO_OBS)
        assert cmd is None or (cmd.vx, cmd.wz) == (0.0, 0.0)      # never moves without a measurement
        t += 0.1
    assert q.result == 'lost' and q.cooldown_until > t
    seen(q, t + 0.1)
    seen(q, t + 0.2)
    assert not q.wants(t + 0.3, POSE, NO_OBS)                  # cool-down
    t2 = q.cooldown_until + 0.1
    seen(q, t2)
    seen(q, t2 + 0.1)
    assert q.wants(t2 + 0.2, POSE, NO_OBS)                     # afterwards: a new chance


def test_failed_target_is_not_tried_again():
    q = QrApproach(CFG)
    wall = np.array([[2.0, yy] for yy in np.arange(-1.0, 1.0, 0.05)])
    q.blacklist.append((2.0, 0.0))                             # given up on this spot before
    seen(q, 0.0)
    seen(q, 0.1)
    assert not q.wants(0.2, POSE, wall)
    seen(q, 0.3, cx=CFG['qr_cx0'] - CFG['qr_fx'] * 0.6)        # another place on the wall (~1 m left)
    seen(q, 0.4, cx=CFG['qr_cx0'] - CFG['qr_fx'] * 0.6)
    assert q.wants(0.5, POSE, wall)


def test_decoded_during_approach_ends_it():
    q = QrApproach(CFG)
    seen(q, 0.0)
    seen(q, 0.1)
    q.start(0.2, POSE)
    q.on_event({'type': 'qr_decoded', 'text': 'M05 clinic'}, 0.3)
    assert q.step(0.4, POSE, 3.0, NO_OBS) is None and q.result == 'decoded'


def test_no_turn_on_the_spot_next_to_an_obstacle():
    q = QrApproach(CFG)
    cx = CFG['qr_cx0'] - CFG['qr_fx'] * math.tan(math.radians(40))     # 40 deg left: needs a turn
    for k in range(6):
        seen(q, 0.1 * k, cx=cx)
    q.start(0.6, POSE)
    near = np.array([[0.0, -0.35]])                                     # 0.35 m to the right
    t = 0.6
    while q.active and t < 5.0:
        cmd = q.step(t, POSE, 3.0, near)
        assert cmd is None or cmd.wz == 0.0
        seen(q, t, cx=cx)
        t += 0.1
    assert q.result == 'no_room_to_turn'


# ---------------------------------------------------------------- closed loop (simulator)
def room():
    w = W.World()
    w.wall(-1.0, -2.0, 6.0, -2.0)
    w.wall(6.0, -2.0, 6.0, 2.5)
    w.wall(6.0, 2.5, -1.0, 2.5)
    w.wall(-1.0, 2.5, -1.0, -2.0)
    w.wall(3.0, -0.6, 3.0, 1.6, h=1.0, name='wall')     # QR on its face at x = 2.97, facing -x
    return w


def qr_events(r):
    return [e for e in r.events if 'qr' in e]


@pytest.mark.parametrize('y', [0.5, -0.4])
def test_sim_low_qr_off_axis_is_read(y):
    w = room()
    cam = QrCam(w, 2.97, y, 0.30, math.pi, size=0.15)
    r = simulate(w, config.load(overrides=['run_time_s=45']), T=45.0, qr_cam=cam)
    ev = qr_events(r)
    assert r.collisions == 0 and not r.reversed
    assert cam.read
    done = [e for e in ev if e.get('qr') == 'decoded']
    assert done and done[0]['t'] < 25.0
    assert sum(1 for e in ev if e.get('qr') == 'start') == 1


def test_sim_without_approach_the_same_qr_is_not_read():
    w = room()
    cam = QrCam(w, 2.97, 0.5, 0.30, math.pi, size=0.15)
    r = simulate(w, config.load(overrides=['run_time_s=45', 'qr_approach=false']), T=45.0, qr_cam=cam)
    assert not cam.read and not qr_events(r)


def test_sim_qr_too_high_gives_up_cleanly():
    # 0.60 m: visible from afar, but its top leaves the picture before it is big enough to read
    w = room()
    cam = QrCam(w, 2.97, 0.5, 0.60, math.pi, size=0.15)
    r = simulate(w, config.load(overrides=['run_time_s=45']), T=45.0, qr_cam=cam)
    ev = qr_events(r)
    assert r.collisions == 0 and not cam.read
    ends = [e for e in ev if e.get('qr') in ('unreadable', 'lost', 'timeout')]
    assert ends and ends[0].get('why') == 'top_of_picture'
    assert sum(1 for e in ev if e.get('qr') == 'start') == 1            # blacklisted: not retried
    assert r.final_mode != 'QR'


def test_sim_false_locations_cost_little():
    w = W.arena_default()
    cam = QrCam(w, 50.0, 50.0, 0.3, 0.0, false_wide_rate=0.3, false_small_rate=0.05)   # no real QR
    r = simulate(w, config.load(overrides=['run_time_s=90']), T=90.0, qr_cam=cam)
    starts = sum(1 for e in qr_events(r) if e.get('qr') == 'start')
    assert r.collisions == 0 and not r.reversed
    assert starts <= CFG['qr_max_attempts']
    assert r.cells >= 5                                                 # still explores


def test_sim_wide_false_locations_alone_never_trigger():
    w = W.arena_default()
    cam = QrCam(w, 50.0, 50.0, 0.3, 0.0, false_wide_rate=0.5)
    r = simulate(w, config.load(overrides=['run_time_s=40']), T=40.0, qr_cam=cam)
    assert not qr_events(r)
