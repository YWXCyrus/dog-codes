"""Very simple Lite3 model: posture toggle, normal/AI state, gaits, unicycle motion with lag,
terrain height and pitch from the support under the front / rear feet, collisions."""
import math

import numpy as np

from rc import geometry as G
from rc.gait import CODES

INV = {v: k for k, v in CODES.items()}
JOINTS = [p + s for p in ('LF', 'RF', 'LB', 'RB') for s in ('_Joint', '_Joint_1', '_Joint_2')]
BODY_H = 0.32
BODY_L, BODY_W = 0.548, 0.37
STEP_RUG = 0.12           # largest step RUG general climbs in the model
STEP_AI = 0.22            # AI Basic climbs stairs


class DogModel:

    def __init__(self, world, x=0.0, y=0.0, yaw=0.0, sitting=False, ai_ignores_cmd=False,
                 wobble_deg=2.0, seed=0, stand_time=3.0, odom_mode='ok', step_rug=STEP_RUG,
                 odom_drift=None):
        self.w = world
        self.x, self.y, self.yaw = x, y, yaw
        self.x0, self.y0, self.yaw0 = x, y, yaw
        self.odom_mode = odom_mode
        self.step_rug = step_rug        # largest step in RUG general (the real dog put its front legs on a 0.20 m slab)
        # odom_drift = (distance scale, gyro yaw bias rad/s): leg odometry integrates the true motion
        # scaled and rotated by the accumulated bias; /imu/data yaw rate carries the same bias
        self.odom_scale, self.gyro_bias = odom_drift if odom_drift else (1.0, 0.0)
        self.drift_yaw = 0.0
        self.ox, self.oy = x, y
        self.support = 0.0
        self.front_support = 0.0
        self.vx = 0.0
        self.wz = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.prev_pitch = 0.0
        self.posture = 'sitting' if sitting else 'standing'
        self.toggle_until = None
        self.stand_time = stand_time
        self.state = 'normal'        # normal | ai
        self.gait = 'rug'            # rug | ai_basic | None
        self.ai_ignores_cmd = ai_ignores_cmd
        self.wobble = math.radians(wobble_deg)
        self.rng = np.random.default_rng(seed)
        self.collisions = 0
        self.in_contact = False
        self.reversed = False
        self.code_log = []
        self.t = 0.0
        self.max_support = 0.0

    # ---- commands ----
    def on_code(self, code, t=None):
        name = INV.get(code, hex(code))
        self.code_log.append((round(self.t if t is None else t, 2), name))
        if name == 'stand_sit' and self.toggle_until is None:
            self.toggle_until = self.t + self.stand_time
        elif name == 'enter_ai' and self.posture == 'standing' and self.state == 'normal':
            self.state, self.gait = 'ai', None
        elif name == 'ai_basic' and self.state == 'ai':
            self.gait = 'ai_basic'
        elif name == 'exit_ai' and self.state == 'ai':
            self.state, self.gait = 'normal', None
        elif name == 'rug_general' and self.state == 'normal':
            self.gait = 'rug'

    def _responds(self):
        if self.posture != 'standing' or self.toggle_until is not None:
            return False
        if self.state == 'ai':
            return self.gait == 'ai_basic' and not self.ai_ignores_cmd
        return self.gait == 'rug'

    def _footprint(self, x, y, yaw):
        u, v = np.meshgrid(np.linspace(-BODY_L / 2, BODY_L / 2, 14), np.linspace(-BODY_W / 2, BODY_W / 2, 10))
        u, v = u.ravel(), v.ravel()
        c, s = math.cos(yaw), math.sin(yaw)
        return x + c * u - s * v, y + s * u + c * v

    def step(self, cmd_vx, cmd_wz, dt):
        self.t += dt
        if self.toggle_until is not None and self.t >= self.toggle_until:
            self.posture = 'standing' if self.posture == 'sitting' else 'sitting'
            self.toggle_until = None
        if cmd_vx < -1e-6:
            self.reversed = True
        tvx, twz = (cmd_vx, cmd_wz) if self._responds() else (0.0, 0.0)
        a = min(1.0, dt / 0.2)
        self.vx += a * (tvx - self.vx)
        self.wz += a * (twz - self.wz)
        max_step = STEP_AI if self.state == 'ai' else self.step_rug
        nyaw = self.yaw + self.wz * dt
        nx = self.x + self.vx * dt * math.cos(nyaw)
        ny = self.y + self.vx * dt * math.sin(nyaw)
        xs, ys = self._footprint(nx, ny, nyaw)
        # legs climb one at a time: a step is reachable from the front feet's height
        hits = self.w.blocking(xs, ys, max(self.support, self.front_support), max_step)
        self.drift_yaw += self.gyro_bias * dt
        if hits:
            if not self.in_contact:
                self.collisions += 1
            self.in_contact = True
            self.vx = 0.0
        else:
            self.in_contact = False
            dx, dy = nx - self.x, ny - self.y
            c, s = math.cos(self.drift_yaw), math.sin(self.drift_yaw)
            self.ox += self.odom_scale * (c * dx - s * dy)
            self.oy += self.odom_scale * (s * dx + c * dy)
            self.x, self.y, self.yaw = nx, ny, G.wrap(nyaw)
        # support under the front and rear feet -> height and pitch
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        zf = self.w.support(self.x + 0.30 * c, self.y + 0.30 * s, max(self.support, self.front_support), max_step)
        zr = self.w.support(self.x - 0.30 * c, self.y - 0.30 * s, self.support, max_step)
        self.front_support = zf
        self.support = max(zf, zr) if abs(zf - zr) < 0.05 else (zf + zr) / 2.0
        self.max_support = max(self.max_support, self.support)
        moving = abs(self.vx) > 0.02 or abs(self.wz) > 0.05
        wob = self.rng.normal(0.0, self.wobble) if moving else 0.0
        self.prev_pitch = self.pitch
        self.pitch = math.atan2(zr - zf, 0.6) + wob        # nose up (front higher) = negative pitch
        self.roll = (self.rng.normal(0.0, self.wobble) if moving else 0.0)

    # ---- sensors ----
    def base_z(self):
        return self.support + (BODY_H if self.posture == 'standing' else 0.12)

    def imu(self, dt):
        g = 9.81
        th, ph = self.pitch, self.roll
        acc = (-g * math.sin(th), g * math.sin(ph) * math.cos(th), g * math.cos(ph) * math.cos(th))
        gyro = (0.0, (self.pitch - self.prev_pitch) / max(dt, 1e-3), self.wz + self.gyro_bias)
        return acc, gyro

    def joints(self):
        knee = -1.29 if self.posture == 'standing' else -2.75
        moving = self.toggle_until is not None
        pos, vel = [], []
        for n in JOINTS:
            if n.endswith('_2'):
                pos.append(knee)
            elif n.endswith('_1'):
                pos.append(0.64 if self.posture == 'standing' else 1.15)   # real standing hip Y (run 113959): legs 0.327 m
            else:
                pos.append(0.0)
            vel.append(0.5 if moving else 0.0)
        return JOINTS, pos, vel

    def leg_odom(self):
        if self.odom_mode == 'frozen_xy':          # position never updates (yaw does)
            return self.x0, self.y0, self.yaw
        if self.odom_mode == 'frozen_yaw':         # yaw never updates (position does)
            return self.x, self.y, self.yaw0
        if self.odom_scale != 1.0 or self.gyro_bias != 0.0:
            return self.ox, self.oy, G.wrap(self.yaw + self.drift_yaw)
        return self.x, self.y, self.yaw
