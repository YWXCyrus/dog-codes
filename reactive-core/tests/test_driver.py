import math

import numpy as np
import pytest

from rc import config
from rc import gait as Gt
from rc import lanes as L
from rc.driver import Driver, DriverInput
from rc.grid import Grid
from rc.guard import Cmd
from rc.stairs import StairsResult

E = np.zeros((0, 2))
INV = {v: k for k, v in Gt.CODES.items()}


def cfg(**kw):
    return config.load(overrides=['%s=%s' % (k, v) for k, v in kw.items()])


def line(x0, y0, x1, y1, n=80):
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


class Loop:
    """Scripted world: obstacles fixed in the BODY frame (or a function of the pose)."""

    def __init__(self, c, obs=E, stairs=None, gyro_gain=1.0, moves=True, tilt=0.0, pitch=None, drift=0.0):
        self.c = c
        self.drift = drift              # rad/s the heading creeps while walking (gait drift)
        self.pitch = pitch
        self.grid = Grid(1.0)
        self.grid.set_origin((0.0, 0.0, 0.0))
        self.d = Driver(c, self.grid)
        self.obs = obs
        self.stairs = stairs or (lambda loop: StairsResult())
        self.gain = gyro_gain
        self.moves = moves
        self.tilt = tilt
        self.pose = [0.0, 0.0, 0.0]
        self.trav = 0.0
        self.t = 0.0
        self.cmd = Cmd()
        self.codes, self.events, self.modes, self.cmds = [], [], [], []

    def step(self):
        obs = self.obs(self) if callable(self.obs) else self.obs
        lanes = L.evaluate(obs, E, self.c)
        tilt = self.tilt(self) if callable(self.tilt) else self.tilt
        pitch = self.pitch(self) if callable(self.pitch) else 0.0
        di = DriverInput(t=self.t, lanes=lanes, obs=obs, mem=E, pose=tuple(self.pose), odom_ok=True,
                         travelled=self.trav, stairs=self.stairs(self), gyro_z=self.cmd.wz * self.gain,
                         tilt_deg=tilt, pitch_deg=pitch, last_cmd=self.cmd)
        out = self.d.step(di)
        self.cmd = out.cmd
        self.codes += [INV[c] for c in out.codes]
        self.events += out.events
        self.modes.append((out.mode, out.phase))
        self.cmds.append(out.cmd)
        dt = 0.1
        if self.moves:
            self.pose[2] += out.cmd.wz * dt + (self.drift * dt if out.cmd.vx > 0 else 0.0)
            dx = out.cmd.vx * dt
            self.pose[0] += dx * math.cos(self.pose[2])
            self.pose[1] += dx * math.sin(self.pose[2])
            self.trav += abs(dx)
        self.t += dt
        return out

    def run(self, seconds):
        out = None
        for _ in range(int(round(seconds / 0.1))):
            out = self.step()
        return out


def test_open_floor_cruises_forward():
    lp = Loop(cfg())
    out = lp.run(1.0)
    assert out.mode == 'CRUISE' and out.cmd.vx > 0.1 and abs(out.cmd.wz) < 0.05


def test_wall_ahead_turns_in_place_never_reverses():
    lp = Loop(cfg(), obs=line(0.6, -2, 0.6, 2))
    out = lp.run(0.5)
    assert out.mode in ('TURN', 'BLOCKED') and out.cmd.vx == 0.0 and out.cmd.wz != 0.0
    assert all(c.vx >= 0.0 for c in lp.cmds)


def test_left_wall_keeps_going_along_it():
    lp = Loop(cfg(), obs=line(-1, 0.6, 3, 0.6))
    out = lp.run(1.0)
    assert out.mode == 'CRUISE' and out.cmd.vx > 0.0 and out.cmd.wz >= -0.05


def test_right_side_prefers_right():
    # opening to both sides at a T junction: left-hand rule turns left, right-hand rule right
    t_obs = np.vstack([line(0.8, -3, 0.8, 3)])
    out_l = Loop(cfg(), obs=t_obs).run(0.3)
    out_r = Loop(cfg(wall_side='right'), obs=t_obs).run(0.3)
    assert out_l.cmd.wz > 0 and out_r.cmd.wz < 0


def test_boxed_in_turns_on_the_spot_then_holds_and_resume():
    # no turn sweep: boxed in, the dog first turns on the spot looking for a lane (a full circle),
    # only then HOLD blocked
    box = np.vstack([line(0.5, -0.5, 0.5, 0.5), line(-0.5, 0.5, 0.5, 0.5), line(-0.5, -0.5, 0.5, -0.5)])
    lp = Loop(cfg(), obs=box)
    out = lp.run(1.0)
    assert out.mode == 'BLOCKED' and out.cmd.vx == 0.0 and out.cmd.wz != 0.0
    lp.run(2 * math.pi / cfg()['turn_wz'] + 1.0)
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'blocked'
    assert lp.cmd.vx == 0 and lp.cmd.wz == 0
    lp.obs = E
    lp.d.resume(lp.t)
    out = lp.run(0.3)
    assert out.mode == 'CRUISE'


def mixed(cmds):
    """Commands that walk and turn at once (never allowed since 2026-11-23: slips at ledges)."""
    return [(round(c.vx, 3), round(c.wz, 3)) for c in cmds if c.vx > 0.0 and c.wz != 0.0]


def ring(r=0.5, gaps=(), n=180, half=45):
    """Obstacle ring around the dog (body frame) with open sectors at the given headings (deg)."""
    a = np.radians(np.linspace(-180, 180, n, endpoint=False))
    keep = np.ones(n, bool)
    for g in gaps:
        keep &= np.abs((np.degrees(a) - g + 180) % 360 - 180) > half
    return np.stack([r * np.cos(a[keep]), r * np.sin(a[keep])], axis=1)


def test_blocked_flickering_lane_keeps_one_turn_direction_then_holds():
    # 061517: boxed in, a lane flickered free for single scans alternately left and right; every
    # re-entry into BLOCKED re-picked the direction -> left, back, left, back, never a full circle
    def obs(loop):
        k = int(round(loop.t / 0.1))
        return ring(gaps=(60,) if k % 6 == 0 else (-60,) if k % 6 == 3 else ())
    lp = Loop(cfg(), obs=obs)
    lp.run(2 * math.pi / cfg()['turn_wz'] + 6.0)
    turning = [c.wz for c in lp.cmds if c.wz != 0.0]
    assert turning and (all(w > 0 for w in turning) or all(w < 0 for w in turning))
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'blocked'
    assert all(c.vx == 0.0 for c in lp.cmds)


def test_blocked_leaves_only_for_a_lane_free_several_decisions():
    c = cfg()
    lp = Loop(c, obs=lambda loop: ring() if loop.t < 1.0 - 1e-6 else ring(gaps=(0,)))
    lp.run(1.0)
    assert lp.d.mode == 'BLOCKED'
    lp.step()                                            # the lane is free for the first time
    assert lp.d.mode == 'BLOCKED'
    lp.run(0.1 * c['blocked_confirm_n'])
    assert lp.d.mode in ('TURN', 'CRUISE')


def test_blocked_again_soon_keeps_direction_and_turned_angle():
    c = cfg()
    state = {'open': False}
    lp = Loop(c, obs=lambda loop: ring(gaps=(0,)) if state['open'] else ring())
    lp.run(2.0)
    d0, turned0 = lp.d.blocked_dir, lp.d.blocked_turned
    assert lp.d.mode == 'BLOCKED' and turned0 > 0
    state['open'] = True
    lp.run(1.0)                                          # free long enough: leaves BLOCKED
    assert lp.d.mode != 'BLOCKED'
    state['open'] = False
    lp.run(0.3)                                          # blocked again within blocked_memory_s
    assert lp.d.mode == "BLOCKED" and lp.d.blocked_dir == d0 and lp.d.blocked_turned >= turned0


def test_heading_error_over_head_err_turns_on_the_spot_small_one_walks_straight():
    c = cfg()
    lp = Loop(c)
    lp.run(0.2)
    lp.d.target_yaw, lp.d.target_t = math.radians(c['head_err_turn_deg'] + 10), lp.t
    out = lp.step()
    assert out.mode == 'TURN' and out.cmd.vx == 0.0 and out.cmd.wz > 0
    lp2 = Loop(c)
    lp2.run(0.2)
    lp2.d.target_yaw, lp2.d.target_t = math.radians(c['head_err_turn_deg'] - 4), lp2.t
    out = lp2.step()
    assert out.mode == 'CRUISE' and out.cmd.vx == pytest.approx(c['cruise_vx']) and out.cmd.wz == 0.0


def test_turn_goes_on_until_turn_done_deg_then_walks_straight():
    c = cfg()
    lp = Loop(c)
    lp.run(0.2)
    target = math.radians(c['head_err_turn_deg'] + 15)
    for _ in range(100):
        lp.d.target_yaw, lp.d.target_t = target, lp.t          # keep the target committed
        out = lp.step()
        if out.mode == 'CRUISE':
            break
    assert out.mode == 'CRUISE' and out.cmd.wz == 0.0 and out.cmd.vx > 0
    # the turn ended inside turn_done_deg, not at head_err_turn_deg
    assert abs(math.degrees(target - lp.pose[2])) <= c['turn_done_deg'] + 1e-6
    assert mixed(lp.cmds) == []


def test_turn_slows_near_the_target_and_turns_back_after_an_overshoot():
    # the dog answers a turn command ~0.3 s late (043747): full turn_wz up to the last degrees would
    # overshoot past turn_done_deg; an overshoot is turned back, not carried round a full circle
    c = cfg()
    lp = Loop(c)
    lp.run(0.2)
    lp.d.target_yaw, lp.d.target_t = math.radians(40), lp.t
    out = lp.step()
    assert out.mode == 'TURN' and out.cmd.wz == pytest.approx(c['turn_wz'])
    lp.pose[2] = math.radians(40 - (c['turn_done_deg'] + 2))       # nearly there
    lp.d.target_yaw, lp.d.target_t = math.radians(40), lp.t
    out = lp.step()
    assert out.mode == 'TURN' and 0 < out.cmd.wz < c['turn_wz']
    lp.pose[2] = math.radians(40 + c['turn_done_deg'] + 2)         # overshot
    lp.d.target_yaw, lp.d.target_t = math.radians(40), lp.t
    out = lp.step()
    assert out.mode == 'TURN' and out.cmd.vx == 0.0 and out.cmd.wz < 0


def test_driving_never_walks_and_turns_with_gait_drift():
    # corridor with a left wall and a T junction ahead, the gait creeps the heading to the right
    obs = lambda loop: np.vstack([line(-1, 0.6, 3, 0.6), line(2.5, -3, 2.5, 3)])
    lp = Loop(cfg(), obs=obs, drift=-0.05)
    lp.run(30.0)
    assert any(c.vx > 0 for c in lp.cmds) and any(c.wz != 0 for c in lp.cmds)
    assert mixed(lp.cmds) == []


def test_constant_cruise_speed_near_obstacles():
    c = cfg()
    out = Loop(c, obs=line(1.3, -0.2, 1.3, 0.2)).run(0.3)     # something 1.3 m ahead, lane still free
    assert out.cmd.vx == pytest.approx(c['cruise_vx']) or out.mode != 'CRUISE'


def test_full_circle_without_free_lane_holds():
    ring = np.array([[math.cos(a) * 0.7, math.sin(a) * 0.7] for a in np.linspace(0, 2 * math.pi, 200)])
    lp = Loop(cfg(), obs=ring)
    lp.run(25.0)
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'blocked'


def test_no_progress_holds():
    lp = Loop(cfg(), moves=False)
    lp.run(4.0)
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'no_progress'
    assert lp.grid.is_dead_end((0, 0), 'N')


def test_run_timer():
    lp = Loop(cfg(run_time_s=2.0))
    lp.run(2.5)
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'time'


def stairs_world(edge_x0=2.0, fixed=False):
    """Stairs whose first edge is at odom x = edge_x0 straight ahead; 'down' appears from the
    platform (odom x in [edge+0.9, edge+1.5]); past that the floor is lower and flat.
    fixed=True: the stairs face odom +x whatever the dog's heading (heading reported body-relative)."""
    def st(loop):
        x = loop.pose[0]
        e = edge_x0 - x
        if loop.d.phase in ('', 'APPROACH') and e > 0.6:
            return StairsResult('up', -loop.pose[2] if fixed else 0.0, e, True, 3)
        if edge_x0 + 0.9 < x < edge_x0 + 1.5:
            return StairsResult('down', 0.0, edge_x0 + 1.6 - x, False)
        return StairsResult()

    def pitch(loop):
        x = loop.pose[0]
        if edge_x0 < x < edge_x0 + 0.9:
            return -12.0                       # climbing: nose up
        if edge_x0 + 1.6 < x < edge_x0 + 2.5:
            return 12.0                        # descending: nose down
        return 1.0

    def tilt(loop):
        return abs(pitch(loop))
    return st, tilt, pitch


def test_full_stairs_sequence():
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(), stairs=st, tilt=tilt, pitch=pitch)
    lp.run(80.0)
    assert lp.codes[:3] == ['manual', 'enter_ai', 'ai_basic']
    assert lp.codes[3:] == ['exit_ai', 'move_mode', 'rug_general', 'rug_general', 'auto']
    phases = [p for _, p in lp.modes]
    for ph in ('APPROACH', 'ENTER', 'CLIMB', 'DESCEND', 'EXIT'):
        assert ph in phases
    assert lp.d.mode == 'CRUISE' and lp.d.stairs_done
    assert lp.pose[0] > 2.0 + 2.5 + 0.9                     # walked off the lower flat
    # AI Basic was entered on flat ground, short of the stairs edge
    i_enter = phases.index('ENTER')
    assert all(c.vx >= 0 for c in lp.cmds)


def test_stairs_never_walk_and_turn_with_gait_drift():
    # approach: stop and turn on the spot, then walk straight; on the steps: straight, no steering
    st, tilt, pitch = stairs_world(fixed=True)
    lp = Loop(cfg(), stairs=st, tilt=tilt, pitch=pitch, drift=0.005)
    lp.run(90.0)
    assert 'exit_ai' in lp.codes and lp.d.stairs_done
    assert mixed(lp.cmds) == []
    on_steps = [c for c, (m, p) in zip(lp.cmds, lp.modes) if p in ('CLIMB', 'DESCEND')]
    assert on_steps and all(c.wz == 0.0 for c in on_steps)


def test_stairs_approach_off_axis_turns_on_the_spot_first():
    # stairs 12 deg to the left: turn on the spot until within turn_done_deg, then walk straight
    def st(loop):
        e = 2.0 - loop.pose[0]
        if loop.d.phase in ('', 'APPROACH') and e > 0.6:
            return StairsResult('up', math.radians(12) - loop.pose[2], e, True, 3)
        return StairsResult()
    c = cfg()
    lp = Loop(c, stairs=st)
    for _ in range(300):
        lp.step()
        if lp.d.phase == 'ENTER':
            break
    appr = [cm for cm, (m, p) in zip(lp.cmds, lp.modes) if p == 'APPROACH']
    assert any(cm.wz > 0 and cm.vx == 0 for cm in appr) and any(cm.vx > 0 for cm in appr)
    assert mixed(lp.cmds) == []
    assert lp.d.phase == 'ENTER'


def test_stairs_heading_drift_on_steps_holds_in_ai():
    # no turning on a step: a heading error over stair_hold_err_deg there is a HOLD (stay in AI)
    st, tilt, pitch = stairs_world(fixed=True)
    lp = Loop(cfg(), stairs=st, tilt=tilt, pitch=pitch)
    for _ in range(600):
        lp.step()
        if lp.d.phase == 'CLIMB':
            break
    lp.drift = 0.08
    lp.run(10.0)
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'stairs_heading'
    assert 'exit_ai' not in lp.codes
    assert mixed(lp.cmds) == []


def test_stairs_exit_despite_walking_wobble():
    # trial 2: 5-12 deg wobble while walking on flat floor - single samples must not block the exit
    st, tilt, pitch = stairs_world()

    def wobbly(loop):
        return pitch(loop) + 7.0 * math.sin(loop.t * 7.3)
    lp = Loop(cfg(), stairs=st, tilt=lambda loop: abs(wobbly(loop)), pitch=wobbly)
    lp.run(90.0)
    assert 'exit_ai' in lp.codes and lp.d.stairs_done


def no_down(st):
    """The lidar never sees the way down (the real Mid-360 may miss the lower treads)."""
    def f(loop):
        r = st(loop)
        return StairsResult() if r.kind == 'down' else r
    return f


def descend_events(lp):
    return [e for e in lp.events if e.get('stairs') == 'DESCEND']


def test_climb_switches_to_descend_on_nose_down_pitch_without_lidar_down():
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(), stairs=no_down(st), tilt=tilt, pitch=pitch)
    lp.run(90.0)
    ev = descend_events(lp)
    assert len(ev) == 1 and ev[0]['by'] == 'imu'
    assert 'exit_ai' in lp.codes and lp.d.stairs_done and lp.d.mode == 'CRUISE'
    assert mixed(lp.cmds) == []


def test_lidar_down_still_triggers_descend_first():
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(), stairs=st, tilt=tilt, pitch=pitch)
    lp.run(80.0)
    ev = descend_events(lp)
    assert len(ev) == 1 and ev[0]['by'] == 'lidar'


def test_nose_up_climb_never_switches_to_descend():
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(), stairs=no_down(st), tilt=tilt, pitch=lambda loop: min(0.0, pitch(loop)))
    lp.run(40.0)
    assert descend_events(lp) == []
    assert 'exit_ai' not in lp.codes


def test_walking_wobble_on_the_platform_does_not_switch_to_descend():
    # 5-12 deg single-sample wobble (trial 2) must not fire the IMU switch before the real nose-down
    edge = 2.0
    st, tilt, pitch = stairs_world(edge)

    def wobbly(loop):
        return pitch(loop) + 7.0 * math.sin(loop.t * 7.3)
    xs = []
    lp = Loop(cfg(), stairs=no_down(st), tilt=lambda loop: abs(wobbly(loop)), pitch=wobbly)
    while lp.t < 90.0:
        lp.step()
        if lp.d.phase == 'DESCEND' and not xs:
            xs.append(lp.pose[0])
    assert xs and xs[0] > edge + 1.6                        # only once the dog really pitched nose-down
    assert 'exit_ai' in lp.codes and lp.d.stairs_done


def test_descend_without_going_down_holds():
    # 'down' seen but the dog never pitches nose-down for 4 m: something is wrong -> HOLD
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(), stairs=st, tilt=tilt, pitch=lambda loop: min(0.0, pitch(loop)))
    lp.run(90.0)
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'stairs_lost'
    assert 'exit_ai' not in lp.codes


def test_stairs_ai_ignores_cmd_aborts_to_hold():
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(), stairs=st, tilt=tilt, pitch=pitch, gyro_gain=0.0)
    lp.run(40.0)
    assert lp.codes[:3] == ['manual', 'enter_ai', 'ai_basic']
    assert 'exit_ai' in lp.codes
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason in ('ai_no_response', 'rug_no_response')
    assert lp.pose[0] < 2.0                                  # never walked onto the stairs


def test_run_timer_waits_for_the_stairs():
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(run_time_s=25.0), stairs=st, tilt=tilt, pitch=pitch)
    for _ in range(2000):
        lp.step()
        if lp.d.mode == 'HOLD':
            break
    assert lp.d.hold_reason == 'time'
    assert lp.d.stairs_done and 'exit_ai' in lp.codes        # the stairs were finished first


def test_resume_after_hold_in_enter_exits_ai():
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(), stairs=st, tilt=tilt, pitch=pitch)
    for _ in range(2000):
        lp.step()
        if lp.d.phase == 'ALIGN':
            break
    lp.d._hold(lp.t, 'tilt_latch', keep_stairs=True)
    lp.d.resume(lp.t)
    assert lp.d.phase == 'EXIT'
    lp.run(10.0)
    assert lp.codes[-5:] == ['exit_ai', 'move_mode', 'rug_general', 'rug_general', 'auto']


def test_unsure_looks_then_holds():
    lp = Loop(cfg(), stairs=lambda loop: StairsResult('unsure', 0.0, 1.2, False, 1))
    lp.run(3.0)
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'stairs_unsure'
    assert all(c.vx == 0 for c in lp.cmds[1:])


def test_unsure_disappears_continue():
    calls = {'n': 0}

    def st(loop):
        calls['n'] += 1
        return StairsResult('unsure', 0.0, 1.2, False, 1) if calls['n'] < 3 else StairsResult()
    lp = Loop(cfg(), stairs=st)
    lp.run(3.0)
    assert lp.d.mode == 'CRUISE'


def test_detect_mode_only_logs():
    lp = Loop(cfg(stairs='detect'), stairs=lambda loop: StairsResult('up', 0.0, 2.0, True, 3))
    lp.run(1.0)
    assert lp.codes == [] and any('stairs_seen' in e for e in lp.events)
    assert lp.d.mode == 'CRUISE'


def test_stairs_no_progress_on_step_holds_in_ai():
    st, tilt, pitch = stairs_world()
    lp = Loop(cfg(), stairs=st, tilt=tilt, pitch=pitch)
    # run until CLIMB, then freeze the dog
    for _ in range(600):
        lp.step()
        if lp.d.phase == 'CLIMB':
            break
    lp.moves = False
    lp.run(5.0)
    assert lp.d.mode == 'HOLD' and lp.d.hold_reason == 'stairs_no_progress'
    assert 'exit_ai' not in lp.codes                         # never Exit AI on a step
    lp.moves = True
    lp.d.resume(lp.t)
    assert lp.d.mode == 'STAIRS' and lp.d.phase == 'CLIMB'
