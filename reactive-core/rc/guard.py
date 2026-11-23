"""Guard (spec section 7): the last layer between every command and the dog.

Order: stale feeds -> tilt -> no reverse -> FRONT STOP (raw scan only) -> WALK OR TURN.
Curtain exceptions (rc/curtain.py, spec 9.4): PUSH relaxes the front stop at creeping speed;
RETREAT (after a touch-test wall verdict only) allows <= cur_retreat_vx backwards, straight. Nothing else
(2026-11-23: the turn sweep, the front slow zone and the tilt slow / push-through rules made the dog
stall on the course; the speed of the dog matters more). Never invents motion: it only zeroes the command.

Walk or turn (2026-11-23, the dog slipped at ledges while walking and turning at once): the driver
never sends both; if a command still walks AND turns, the turn is dropped ('walk_no_turn' - a sign
that some driver path was missed). After the front stop, so a blocked front leaves a pure turn.

Tilt (flat ground and stairs alike; stair slope ~27 deg): above tilt_stop_deg the command is zeroed
until the tilt is back below tilt_resume_deg, then the dog drives on by itself (no HOLD). Above
tilt_latch_deg the dog is going over: stopped until restart.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cmd:
    vx: float = 0.0
    wz: float = 0.0


@dataclass
class GuardInput:
    obs_now: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    ages: dict = field(default_factory=dict)       # feed name -> seconds since last arrival
    tilt_deg: float = 0.0                          # max(|roll|, |pitch|)
    t: float = None
    push: bool = False                             # curtain PUSH (rc/curtain.py): front stop relaxed, vx capped
    retreat: bool = False                          # curtain RETREAT after a wall verdict: slow reverse allowed


class Guard:

    def __init__(self, cfg, sensor_x=0.18678):
        self.c = cfg
        self.sensor_x = sensor_x
        self.latched = False
        self.tilted = False          # stopped by tilt, waiting for tilt < tilt_resume_deg

    def front_dist(self, obs):
        """Distance from the sensor to the nearest obstacle ahead in the body corridor."""
        obs = np.asarray(obs, dtype=float).reshape(-1, 2)
        if obs.shape[0] == 0:
            return math.inf
        half = self.c['footprint_w'] / 2.0 + self.c['guard_corridor_margin_m']
        m = (obs[:, 0] > self.sensor_x) & (np.abs(obs[:, 1]) < half)
        return float(obs[m, 0].min() - self.sensor_x) if m.any() else math.inf

    def filter(self, cmd, gi):
        c = self.c
        vx, wz = float(cmd.vx), float(cmd.wz)

        stale = [k for k, a in sorted(gi.ages.items()) if not (a <= c['stale_s'])]
        if stale:
            return Cmd(0.0, 0.0), ['stale:%s' % k for k in stale]

        tilt = gi.tilt_deg
        if tilt > c['tilt_latch_deg']:
            self.latched = True
        if self.latched:
            return Cmd(0.0, 0.0), ['tilt_latch']
        if tilt > c['tilt_stop_deg']:
            self.tilted = True
        elif tilt < c['tilt_resume_deg']:
            self.tilted = False
        if self.tilted:
            return Cmd(0.0, 0.0), ['tilt_stop']

        reasons = []
        if vx < 0.0 and gi.retreat:
            # the only reverse: <= cur_retreat_m straight back over the ground the dog just walked in
            # on (curtain touch test said "wall"; turning with the nose on it would hit it)
            vx = max(vx, -c['cur_retreat_vx'])
            wz = 0.0
        elif vx < 0.0:
            vx = 0.0
            reasons.append('no_reverse')
        if gi.push and vx > 0.0:
            # curtain touch test (spec 9.4: "the guard front zone is relaxed only inside this mode"):
            # the cloth is touched on purpose; creeping speed only
            if vx > c['cur_push_vx']:
                vx = c['cur_push_vx']
                reasons.append('push_cap')
        elif vx > 0.0 and self.front_dist(gi.obs_now) < c['stop_m']:
            vx = 0.0
            reasons.append('front_stop')
        if vx > 0.0 and wz != 0.0:
            wz = 0.0
            reasons.append('walk_no_turn')
        return Cmd(vx, wz), reasons
