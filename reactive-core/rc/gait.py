"""Gait table and timed gait sequences (spec section 9.2). Codes from doc 03.

AI mode is ONLY for the stairs (AI Basic); everything else is RUG general.
A sequence is a list of steps:
  ('zero', s)     command zero speed for s seconds
  ('code', name)  send one /simple_cmd code (respecting simple_cmd_gap_s)
  ('wait', s)     wait s seconds (zero speed)
  ('turn_test',)  turn at test_wz for test_s; the gyro must show >= test_ratio of it

Driving in AI (2026-11-23 hand tests on the dog): /cmd_vel is ignored there (speed commands need
navigation mode, which AI does not support). The dog follows AXIS commands on /simple_cmd - the
value goes in the 'size' field (SDK CommandHead value / paramters_size union) - sent at >= 20 Hz,
forward and turn axis together, and forward only moves the dog in MANUAL control mode. So Enter AI
switches to manual first and Exit AI switches back to navigation (auto) mode after leaving AI.
"""
from rc.guard import Cmd

CODES = {
    'stand_sit': 0x21010202,     # Stand/Sit TOGGLE
    'rug_general': 0x21010401,
    'move_mode': 0x21010D06,
    'enter_ai': 0x21010528,
    'exit_ai': 0x2101052B,
    'ai_basic': 0x2101052A,
    'manual': 0x21010C02,        # control mode manual (lite3_manual)
    'auto': 0x21010C03,          # control mode navigation (lite3_auto); not allowed inside AI
    'fwd_axis': 0x21010130,      # axis: forward (+) / backward
    'turn_axis': 0x21010135,     # axis: turn, + = to the RIGHT
}

SET_RUG = [('code', 'rug_general'), ('wait', 0.5), ('code', 'rug_general'), ('wait', 1.0)]
ENTER_AI = [('zero', 1.0), ('code', 'manual'), ('wait', 0.5), ('code', 'enter_ai'), ('wait', 2.0),
            ('code', 'ai_basic'), ('wait', 1.0), ('turn_test',)]
EXIT_AI = [('zero', 1.0), ('code', 'exit_ai'), ('wait', 2.0), ('code', 'move_mode'), ('wait', 0.5),
           ('code', 'rug_general'), ('wait', 0.5), ('code', 'rug_general'), ('wait', 0.5),
           ('code', 'auto'), ('wait', 1.0), ('turn_test',)]
TURN_TEST = [('turn_test',)]
AXIS_MAX = 32767


def axis_values(cmd, cfg):
    """(forward, turn) axis values for a guarded command: dead zone + gain * speed, clamped.
    Never backwards; our +wz turns left, the dog's turn axis is positive to the right."""
    c = cfg
    fwd = 0
    if cmd.vx > 0.0:
        fwd = int(min(AXIS_MAX, round(c['ai_fwd_dz'] + c['ai_fwd_gain'] * cmd.vx)))
    turn = 0
    if cmd.wz != 0.0:
        mag = int(min(AXIS_MAX, round(c['ai_turn_dz'] + c['ai_turn_gain'] * abs(cmd.wz))))
        turn = -mag if cmd.wz > 0 else mag
    return fwd, turn


class AiAxis:
    """Knows from the codes actually sent whether the dog is in AI; there every command becomes
    both axis commands (forward, turn) instead of /cmd_vel."""

    def __init__(self, cfg):
        self.c = cfg
        self.active = False

    def on_code(self, code):
        if code == CODES['enter_ai']:
            self.active = True
        elif code == CODES['exit_ai']:
            self.active = False

    def messages(self, cmd):
        if not self.active:
            return []
        fwd, turn = axis_values(cmd, self.c)
        return [(CODES['fwd_axis'], fwd), (CODES['turn_axis'], turn)]


class GaitSequencer:

    def __init__(self, steps, cfg, turn_sign=1, last_code_t=-1e9):
        self.steps = list(steps)
        self.c = cfg
        self.sign = 1 if turn_sign >= 0 else -1
        self.i = 0
        self.t0 = None
        self.last_code_t = last_code_t
        self.samples = []
        self.failed_reason = ''
        self.status = 'running'

    def _next(self, now):
        self.i += 1
        self.t0 = now
        self.samples = []

    def tick(self, now, gyro_z=0.0):
        """-> (status, Cmd, code or None)"""
        c = self.c
        zero = Cmd(0.0, 0.0)
        while self.status == 'running':
            if self.i >= len(self.steps):
                self.status = 'done'
                break
            if self.t0 is None:
                self.t0 = now
            step = self.steps[self.i]
            kind = step[0]
            el = now - self.t0
            if kind in ('zero', 'wait'):
                if el >= step[1]:
                    self._next(now)
                    continue
                return self.status, zero, None
            if kind == 'code':
                if now - self.last_code_t < c['simple_cmd_gap_s']:
                    return self.status, zero, None
                self.last_code_t = now
                self._next(now)
                return self.status, zero, CODES[step[1]]
            if kind == 'turn_test':
                T = c['test_s']
                if el < T:
                    if el >= T / 2.0:
                        self.samples.append(abs(gyro_z))
                    return self.status, Cmd(0.0, self.sign * c['test_wz']), None
                if el < T + 0.5:
                    return self.status, zero, None
                mean = sum(self.samples) / len(self.samples) if self.samples else 0.0
                if mean < c['test_ratio'] * c['test_wz']:
                    self.failed_reason = 'turn test: gyro %.2f rad/s for a %.2f command' % (mean, c['test_wz'])
                    self.status = 'failed'
                    break
                self._next(now)
                continue
            raise ValueError('unknown step %r' % (step,))
        return self.status, zero, None
