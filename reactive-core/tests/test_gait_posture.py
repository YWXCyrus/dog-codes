import pytest

from rc import config
from rc import gait as Gt
from rc import posture as P

CFG = config.load()


def drive(seq, gyro=0.3, t_end=30.0, dt=0.05):
    """Run a sequencer; gyro follows the commanded wz times `gyro/0.3`."""
    t, codes, status, cmds = 0.0, [], 'running', []
    while t < t_end:
        status, cmd, code = seq.tick(t, gyro_z=cmds[-1].wz * gyro / 0.3 if cmds else 0.0)
        cmds.append(cmd)
        if code is not None:
            codes.append((round(t, 2), code))
        if status != 'running':
            break
        t += dt
    return status, codes, cmds, t


def names(codes):
    inv = {v: k for k, v in Gt.CODES.items()}
    return [inv[c] for _, c in codes]


def test_enter_ai_sequence_passes():
    st, codes, cmds, t = drive(Gt.GaitSequencer(Gt.ENTER_AI, CFG))
    assert st == 'done'
    assert names(codes) == ['enter_ai', 'ai_basic']
    assert codes[1][0] - codes[0][0] >= 2.0                 # waits between codes
    assert any(c.wz > 0 for c in cmds)                       # the turn test turned


def test_enter_ai_fails_without_response():
    seq = Gt.GaitSequencer(Gt.ENTER_AI, CFG)
    st, codes, cmds, t = drive(seq, gyro=0.0)
    assert st == 'failed' and 'turn test' in seq.failed_reason


def test_exit_ai_sequence():
    st, codes, _, _ = drive(Gt.GaitSequencer(Gt.EXIT_AI, CFG))
    assert st == 'done'
    assert names(codes) == ['exit_ai', 'move_mode', 'rug_general', 'rug_general']


def test_codes_respect_gap():
    seq = Gt.GaitSequencer([('code', 'rug_general'), ('code', 'rug_general')], CFG)
    st, codes, _, _ = drive(seq)
    assert len(codes) == 2 and codes[1][0] - codes[0][0] >= CFG['simple_cmd_gap_s'] - 1e-9


def test_zero_step_commands_zero():
    seq = Gt.GaitSequencer([('zero', 1.0)], CFG)
    st, codes, cmds, t = drive(seq)
    assert st == 'done' and all(c.vx == 0 and c.wz == 0 for c in cmds) and t >= 1.0


def test_turn_sign():
    seq = Gt.GaitSequencer([('turn_test',)], CFG, turn_sign=-1)
    st, _, cmds, _ = drive(seq, gyro=-0.3)
    assert st == 'done' and any(c.wz < 0 for c in cmds)


def test_posture_classify():
    n = CFG['knee_names']
    assert P.classify(n, [-2.75] * 4, CFG) == 'sitting'
    assert P.classify(n, [-1.29] * 4, CFG) == 'standing'
    assert P.classify(n, [-2.0] * 4, CFG) == 'unclear'
    assert P.classify(['a', 'b'], [0.0, 0.0], CFG) == 'unclear'
    # names in another order with extra joints
    names_ = ['LF_Joint', 'LF_Joint_1', 'LF_Joint_2', 'RF_Joint_2', 'LB_Joint_2', 'RB_Joint_2']
    assert P.classify(names_, [0.4, 1.1, -1.3, -1.3, -1.3, -1.3], CFG) == 'standing'


def test_settle():
    s = P.Settle(CFG)
    assert not s.update([0.0] * 12, 0.0)
    assert not s.update([0.0] * 12, 0.5)
    assert s.update([0.0] * 12, 1.01)
    assert not s.update([0.2] + [0.0] * 11, 1.1)
    assert not s.update([0.0] * 12, 1.5)
    assert not s.update([0.0] * 12, 2.2)       # only 0.7 s since 1.5
    assert s.update([0.0] * 12, 2.6)


# ---- body height from the leg joints (run 113959, means over 3.5 s of standing) ----
LEG_NAMES = [p + s for p in ('LF', 'RF', 'LB', 'RB') for s in ('_Joint', '_Joint_1', '_Joint_2')]
DEFAULT_H = [0.069, 0.648, -1.299, -0.072, 0.642, -1.305, 0.084, 0.637, -1.293, -0.084, 0.614, -1.282]
LOWERED = [0.081, 0.844, -1.681, -0.080, 0.845, -1.715, 0.103, 0.846, -1.704, -0.100, 0.807, -1.664]
LYING_KNEE = -2.75


def test_leg_height_real_default_and_lowered():
    c = dict(CFG, leg_h_add=0.0)
    assert P.leg_height(LEG_NAMES, DEFAULT_H, c) == pytest.approx(0.327, abs=0.003)
    assert P.leg_height(LEG_NAMES, LOWERED, c) == pytest.approx(0.274, abs=0.004)


def test_leg_height_uses_stance_legs():
    # trotting: two legs lifted (shorter) -> the two longest legs give the height
    q = list(DEFAULT_H)
    q[1], q[2] = 1.2, -2.2          # LF swing
    q[10], q[11] = 1.2, -2.2        # RB swing
    assert P.leg_height(LEG_NAMES, q, dict(CFG, leg_h_add=0.0)) == pytest.approx(0.327, abs=0.003)


def test_leg_height_invalid():
    lying = [v if i % 3 != 2 else LYING_KNEE for i, v in enumerate(DEFAULT_H)]
    lying = [1.15 if i % 3 == 1 else v for i, v in enumerate(lying)]
    assert P.leg_height(LEG_NAMES, lying, CFG) is None                  # too low
    assert P.leg_height(LEG_NAMES[:9], DEFAULT_H[:9], CFG) is None      # a leg missing


def test_core_floor_follows_the_legs():
    from rc.core import Core
    core = Core(dict(CFG, leg_h_add=0.0))
    vel = [0.0] * 12
    core.on_joints(LEG_NAMES, DEFAULT_H, vel, 0.0)
    assert core.floor_src == 'legs' and -core.floor_z == pytest.approx(0.327, abs=0.003)
    for k in range(1, 60):                                               # 3 s lowered at 20 Hz
        core.on_joints(LEG_NAMES, LOWERED, vel, k * 0.05)
    assert -core.floor_z == pytest.approx(0.274, abs=0.006)
    core2 = Core(dict(CFG, floor_src='fixed'))
    core2.on_joints(LEG_NAMES, LOWERED, vel, 0.0)
    assert -core2.floor_z == CFG['body_h_default']
