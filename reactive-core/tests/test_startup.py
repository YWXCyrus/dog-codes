"""Start-up (rc/startup.py): feeds only. All four feeds for 1 s -> ready; nothing is ever sent
(no Stand/Sit, no gait code, no turn test, no floor calibration) - see commit 9ee2599."""
from rc import config
from rc.startup import Startup

CFG = config.load()
ALL = {'livox': True, 'imu': True, 'joints': True, 'odom': True}


def run(su, feeds_fn, T=10.0):
    t, out, codes, cmds = 0.0, None, [], []
    while t < T:
        out = su.tick(t, feeds_fn(t))
        codes += out.codes
        cmds.append((out.cmd.vx, out.cmd.wz))
        if out.status != 'running':
            break
        t += 0.05
    return out, t, codes, cmds


def test_all_feeds_ready_after_one_second():
    out, t, codes, _ = run(Startup(CFG), lambda t: ALL)
    assert out.status == 'ready'
    assert 1.0 <= t < 1.2


def test_nothing_sent_during_startup():
    out, _, codes, cmds = run(Startup(CFG), lambda t: ALL)
    assert out.status == 'ready'
    assert codes == []
    assert all(c == (0.0, 0.0) for c in cmds)


def test_missing_feed_fails_after_5_s_and_names_it():
    out, t, codes, _ = run(Startup(CFG), lambda t: dict(ALL, odom=False))
    assert out.status == 'failed'
    assert 'odom' in out.msg and 'livox' not in out.msg
    assert 5.0 < t < 5.2
    assert codes == []


def test_feed_dropout_restarts_the_one_second_wait():
    # odom drops for one tick at 0.8 s: ready only 1 s after it is back
    out, t, _, _ = run(Startup(CFG), lambda t: dict(ALL, odom=not (0.79 < t < 0.81)))
    assert out.status == 'ready'
    assert t >= 1.8


def test_late_feeds_still_ready():
    out, t, _, _ = run(Startup(CFG), lambda t: ALL if t >= 3.0 else dict(ALL, livox=False))
    assert out.status == 'ready'
    assert 4.0 <= t < 4.2


def test_status_sticks_after_ready():
    su = Startup(CFG)
    run(su, lambda t: ALL)
    assert su.tick(20.0, {}).status == 'ready'


def test_dry_run_same_sequence():
    out, _, codes, _ = run(Startup(CFG, dry_run=True), lambda t: ALL)
    assert out.status == 'ready' and codes == []
