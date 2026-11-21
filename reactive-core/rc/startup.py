"""Start-up sequence (spec section 4), pure logic, ticked by run.py / the simulator.

FEEDS -> POSTURE (one Stand/Sit toggle at most) -> GAIT (RUG general x2) -> TURN TEST -> FLOOR -> ready.
Never sends a second Stand/Sit; an unclear posture fails (after STOP: Lose Control Protection).
"""
from dataclasses import dataclass, field

from rc import gait as Gt
from rc.guard import Cmd

FEEDS = ('livox', 'imu', 'joints', 'odom')


@dataclass
class StartOut:
    status: str                      # running | ready | failed
    cmd: Cmd = field(default_factory=Cmd)
    codes: list = field(default_factory=list)
    msg: str = ''


class Startup:

    def __init__(self, cfg, dry_run=False):
        self.c = cfg
        self.dry = dry_run          # nothing reaches the dog: posture toggle and turn test are skipped
        self.step = 'FEEDS'
        self.t0 = None
        self.feeds_since = None
        self.toggled = False
        self.toggle_t = None
        self.seq = None
        self.msg = ''
        self.last_code_t = -1e9
        self.stand_t = None

    def _go(self, step, t):
        self.step = step
        self.t0 = t

    def tick(self, t, feeds_ok, posture, settled, gyro_z, floor_ok, turn_sign=1, turn_blocked=False):
        c = self.c
        if self.t0 is None:
            self.t0 = t
        if self.step in ('READY', 'FAILED'):
            return StartOut('ready' if self.step == 'READY' else 'failed', msg=self.msg)

        if self.step == 'FEEDS':
            missing = [k for k in FEEDS if not feeds_ok.get(k, False)]
            if missing:
                self.feeds_since = None
                if t - self.t0 > 5.0:
                    return self._fail('no data from: %s (dog feed dead after STOP? bring-up running?)'
                                      % ', '.join(missing))
                return StartOut('running', msg='waiting for feeds: %s' % ', '.join(missing))
            if self.feeds_since is None:
                self.feeds_since = t
            if t - self.feeds_since >= 1.0:
                self._go('POSTURE', t)
            return StartOut('running', msg='feeds ok')

        if not all(feeds_ok.get(k, False) for k in FEEDS):
            return self._fail('feed lost during start-up')

        if self.step == 'POSTURE':
            if self.dry:
                self._go('GAIT', t)
                self.seq = Gt.GaitSequencer(Gt.SET_RUG, c, last_code_t=self.last_code_t)
                return StartOut('running', msg='dry run: posture %s (no Stand/Sit would be needed: %s)'
                                % (posture, posture == 'standing'))
            if posture == 'standing':
                if self.stand_t is None:
                    self.stand_t = t
            else:
                self.stand_t = None
            # 'settled' = joint speeds ~0 for 1 s; after 4 s standing go anyway (the real
            # /joint_states carries no velocities; noisy ones must not block the start)
            long_standing = self.stand_t is not None and t - self.stand_t > 4.0
            if posture == 'standing' and (settled or long_standing) and (self.toggle_t is None or t - self.toggle_t > 1.0):
                self._go('GAIT', t)
                self.seq = Gt.GaitSequencer(Gt.SET_RUG, c, last_code_t=self.last_code_t)
                return StartOut('running', msg='standing')
            if posture == 'unclear' and not self.toggled:
                return self._fail('posture unclear (knees neither sitting nor standing) - '
                                  'after a STOP the dog needs a hand Stand/Sit; check and restart')
            if posture == 'sitting' and not self.toggled:
                self.toggled = True
                self.toggle_t = t
                self.last_code_t = t
                return StartOut('running', codes=[Gt.CODES['stand_sit']], msg='sitting: standing up (one toggle)')
            if self.toggled and t - self.toggle_t > c['stand_timeout_s']:
                return self._fail('not standing %.0f s after the Stand/Sit toggle - not toggling again'
                                  % c['stand_timeout_s'])
            return StartOut('running', msg='waiting for standing / settled (%s)' % posture)

        if self.step == 'TEST' and turn_blocked:
            return self._fail('turn test impossible: obstacles inside the turning circle on both sides '
                              '(start the dog with ~0.5 m free around it)')
        if self.step in ('GAIT', 'TEST'):
            st, cmd, code = self.seq.tick(t, gyro_z)
            codes = [code] if code is not None else []
            if code is not None:
                self.last_code_t = t
            if st == 'failed':
                return self._fail('dog does not respond to commands (%s): auto mode on in the app? '
                                  'feed alive after a STOP?' % self.seq.failed_reason)
            if st == 'done':
                if self.step == 'GAIT' and self.dry:
                    self._go('FLOOR', t)
                    return StartOut('running', codes=codes, msg='dry run: turn test skipped')
                if self.step == 'GAIT':
                    self._go('TEST', t)
                    self.seq = Gt.GaitSequencer(Gt.TURN_TEST, c, turn_sign=turn_sign, last_code_t=self.last_code_t)
                else:
                    self._go('FLOOR', t)
            return StartOut('running', cmd=cmd, codes=codes, msg=self.step.lower())

        if self.step == 'FLOOR':
            if floor_ok:
                self.step = 'READY'
                self.msg = 'ready'
                return StartOut('ready', msg='ready')
            if t - self.t0 > 5.0:
                return self._fail('floor not found ahead (calibration) - is the dog facing open floor?')
            return StartOut('running', msg='calibrating floor')
        return self._fail('unknown step %s' % self.step)

    def _fail(self, msg):
        self.step = 'FAILED'
        self.msg = msg
        return StartOut('failed', msg=msg)
