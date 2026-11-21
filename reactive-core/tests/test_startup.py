from rc import config
from rc import gait as Gt
from rc.startup import Startup

CFG = config.load()
ALL = {'livox': True, 'imu': True, 'joints': True, 'odom': True}
INV = {v: k for k, v in Gt.CODES.items()}


def run(su, posture_fn, feeds=ALL, gyro_gain=1.0, floor_ok=True, T=40.0):
    t, codes, out, wz = 0.0, [], None, 0.0
    while t < T:
        p, settled = posture_fn(t, codes)
        out = su.tick(t, feeds, p, settled, wz * gyro_gain, floor_ok)
        wz = out.cmd.wz
        codes += [INV[c] for c in out.codes]
        if out.status != 'running':
            break
        t += 0.05
    return out, codes


def standing(t, codes):
    return 'standing', True


def test_standing_dog_goes_ready_without_toggle():
    out, codes = run(Startup(CFG), standing)
    assert out.status == 'ready'
    assert codes == ['rug_general', 'rug_general']


def test_missing_odom_fails_without_codes():
    out, codes = run(Startup(CFG), standing, feeds=dict(ALL, odom=False))
    assert out.status == 'failed' and 'odom' in out.msg and codes == []


def test_sitting_one_toggle_then_ready():
    def post(t, codes):
        if 'stand_sit' in codes and t > 4.0:
            return 'standing', t > 5.0
        return 'sitting', True
    out, codes = run(Startup(CFG), post)
    assert out.status == 'ready'
    assert codes.count('stand_sit') == 1


def test_sitting_never_stands_fails_after_one_toggle():
    out, codes = run(Startup(CFG), lambda t, c: ('sitting', True))
    assert out.status == 'failed' and codes.count('stand_sit') == 1


def test_unclear_posture_fails_no_codes():
    out, codes = run(Startup(CFG), lambda t, c: ('unclear', True))
    assert out.status == 'failed' and codes == []


def test_turn_test_failure():
    out, codes = run(Startup(CFG), standing, gyro_gain=0.0)
    assert out.status == 'failed' and 'auto mode' in out.msg


def test_dry_run_skips_toggle_and_turn_test():
    out, codes = run(Startup(CFG, dry_run=True), lambda t, c: ('sitting', True), gyro_gain=0.0)
    assert out.status == 'ready'
    assert 'stand_sit' not in codes


def test_never_settled_still_starts():
    out, codes = run(Startup(CFG), lambda t, c: ('standing', False))
    assert out.status == 'ready'


def test_turn_blocked_clear_message():
    su = Startup(CFG)
    t, out = 0.0, None
    while t < 20:
        out = su.tick(t, ALL, 'standing', True, 0.0, True, turn_blocked=True)
        if out.status != 'running':
            break
        t += 0.05
    assert out.status == 'failed' and 'turning circle' in out.msg


def test_floor_not_found():
    out, codes = run(Startup(CFG), standing, floor_ok=False)
    assert out.status == 'failed' and 'floor' in out.msg
