"""Start-up sequence, pure logic, ticked by run.py / the simulator.

FEEDS (all four topics arriving for 1 s) -> ready. Nothing is sent to the dog during start-up:
no Stand/Sit, no gait code, no turn test, no floor calibration. Stand the dog up and pick the
gait with the controller before pressing Enter.
"""
from dataclasses import dataclass, field

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
        self.dry = dry_run
        self.step = 'FEEDS'
        self.t0 = None
        self.feeds_since = None
        self.msg = ''

    def tick(self, t, feeds_ok, *_unused, **_unused_kw):
        if self.t0 is None:
            self.t0 = t
        if self.step in ('READY', 'FAILED'):
            return StartOut('ready' if self.step == 'READY' else 'failed', msg=self.msg)
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
            self.step = 'READY'
            self.msg = 'feeds ok - ready'
            return StartOut('ready', msg=self.msg)
        return StartOut('running', msg='feeds ok')

    def _fail(self, msg):
        self.step = 'FAILED'
        self.msg = msg
        return StartOut('failed', msg=msg)
