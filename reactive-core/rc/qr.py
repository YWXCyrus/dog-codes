"""QR approach (driver mode QR): bring the camera close to, centred on and still in front of a QR
code that perception located but could not read, so that perception reads it (task M04).

Perception (frontier-fastlio/dog_codes/perception.py, separate process, perception/README.md)
publishes on /quad/event: {'type': 'qr_seen', 'w', 'cx', 'cy'} for a QR it located but could not
read, and {'type': 'qr_decoded', 'text', ...} once it read one. This module only moves the dog.

Why "look - turn - step" and not continuous visual servoing:
* perception decodes at ~6 Hz; an event arrives 0.15-0.4 s after its picture, and a moving camera
  blurs the modules. Measuring only while the dog stands still (LOOK, after qr_settle_s) makes every
  measurement sharp and free of latency. The motions in between (TURN, STEP) are closed on the gyro
  yaw and the leg odometry, which run at 200 Hz, not on the 6 Hz picture.
* the target is placed in the world: the camera ray (bearing from cx) is intersected with the lidar
  obstacles; without a lidar hit the distance comes from the QR's width and qr_size_m. The dog turns
  about base_link, the camera sits CAM_X in front of it: the turn uses the bearing seen FROM
  base_link to that point, not the camera bearing (they differ by up to ~10 deg close up).
* false targets: OpenCV's locator fires on panels and frames (recordings 2026-11-22: 27 false
  locations in 90 s, mostly 200-470 px wide). A real QR that wide would already be read (reads
  reliably from ~165 px), so located-but-unread shapes wider than qr_seek_w_max are never chased;
  a candidate needs qr_trigger_n events within 1 s at a stable image position.
* the camera looks 20 deg down: the top of the picture is +7.3 deg above horizontal. Getting closer
  moves a QR up in the picture; when its top edge reaches the border the dog stops advancing and
  reads from there (a QR mounted higher than ~0.45 m cannot be read from close up at all).
* budget: qr_max_s per attempt, qr_max_attempts per run; a failed target is blacklisted (world
  position) and followed by a cool-down. Once any QR is read, the approach is off for the run.
* never turns on the spot with an obstacle closer than qr_turn_clear_m (the guard has no turn
  sweep any more), never walks closer than qr_front_min_m; the guard still applies to every command.
"""
import math
from dataclasses import dataclass

import numpy as np

from rc import geometry as G
from rc.guard import Cmd

IDLE, LOOK, TURN, STEP, READ = 'IDLE', 'LOOK', 'TURN', 'STEP', 'READ'
CAM_X = 0.252           # base_link -> camera_link along x (/tf_static, 2026-11-22)


@dataclass
class Obs:
    t: float
    w: float
    cx: float
    cy: float


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class QrApproach:

    def __init__(self, cfg, ev=None):
        self.c = cfg
        self.ev = ev or (lambda t, **kw: None)
        self.phase = IDLE
        self.obs = []                # recent qr_seen observations
        self.decoded = None          # text, once perception read a QR
        self.attempts = 0
        self.cooldown_until = -1e9
        self.blacklist = []          # world (x, y) of targets given up on
        self.target = None           # world (x, y) of the current target
        self.t0 = self.t_phase = 0.0
        self.goal_yaw = 0.0
        self.step_d = 0.0
        self.step_from = (0.0, 0.0)
        self.read_why = ''
        self.result = ''

    @property
    def active(self):
        return self.phase != IDLE

    # ------------------------------------------------------------------ inputs
    def on_event(self, d, t):
        """A perception event (dict from /quad/event). Returns the text when a QR was read first."""
        typ = d.get('type')
        if typ == 'qr_decoded':
            first = self.decoded is None
            self.decoded = str(d.get('text', ''))
            return self.decoded if first else None
        if typ == 'qr_seen':
            try:
                o = Obs(t, float(d['w']), float(d['cx']), float(d.get('cy', self.c['qr_cy0'])))
            except (KeyError, TypeError, ValueError):
                return None
            self.obs.append(o)
            self.obs = [x for x in self.obs if t - x.t <= 3.0]
        return None

    # ------------------------------------------------------------------ geometry
    def bearing_cam(self, cx):
        """Bearing of an image column seen from the camera (+ = left)."""
        return math.atan2(self.c['qr_cx0'] - cx, self.c['qr_fx'])

    def locate(self, o, obs_body):
        """QR position in the body frame and its distance from the camera."""
        b = self.bearing_cam(o.cx)
        u = np.array([math.cos(b), math.sin(b)])
        cam = np.array([CAM_X, 0.0])
        d = None
        p = np.asarray(obs_body, dtype=float).reshape(-1, 2)
        if p.shape[0]:
            rel = p - cam
            along = rel @ u
            perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
            m = (along > 0.05) & (along < 4.0) & (perp < 0.08)
            if m.any():
                d = float(along[m].min())
        if d is None:
            d = self.c['qr_fx'] * self.c['qr_size_m'] / max(o.w, 1.0)
        return cam + d * u, d

    def closest_in_picture(self, top_px, dcam):
        """Closest camera distance at which the QR's top edge (now at row top_px, dcam away) stays
        qr_top_margin_px inside the picture. The camera is pitched qr_cam_pitch_deg down, so a code
        above the camera rises in the picture as the dog gets closer."""
        c = self.c
        tilt = math.radians(c['qr_cam_pitch_deg'])
        elev = math.atan2(c['qr_cy0'] - top_px, c['qr_fx']) - tilt        # above horizontal, now
        dz = dcam * math.tan(elev)                                          # height of the top edge
        elev_max = math.atan2(c['qr_cy0'] - c['qr_top_margin_px'], c['qr_fx']) - tilt
        if dz <= 0.0 or elev_max <= 0.0:
            return 0.0 if dz <= 0.0 else dcam
        return dz / math.tan(elev_max)

    @staticmethod
    def _median(obs):
        return Obs(obs[-1].t, float(np.median([o.w for o in obs])), float(np.median([o.cx for o in obs])),
                   float(np.median([o.cy for o in obs])))

    def _blacklisted(self, w_xy):
        return any(math.hypot(w_xy[0] - x, w_xy[1] - y) < self.c['qr_blacklist_m'] for x, y in self.blacklist)

    # ------------------------------------------------------------------ trigger
    def wants(self, t, pose, obs_body):
        c = self.c
        if (self.active or self.decoded is not None or self.attempts >= c['qr_max_attempts']
                or t < self.cooldown_until):
            return False
        rec = [o for o in self.obs if t - o.t <= 1.0 and c['qr_seek_w_min'] <= o.w <= c['qr_seek_w_max']]
        n = int(c['qr_trigger_n'])
        if len(rec) < n:
            return False
        rec = rec[-n:]
        if max(o.cx for o in rec) - min(o.cx for o in rec) > 60.0:
            return False                          # not a stable image position (or turning)
        p, _ = self.locate(self._median(rec), obs_body)
        return not self._blacklisted(G.body_to_world(p[None], pose)[0])

    def start(self, t, pose):
        self.attempts += 1
        self.t0 = t
        self.target = None
        self.result = ''
        self._phase(LOOK, t)
        self.ev(t, qr='start', attempt=self.attempts)

    def abort(self, t, why='interrupted'):
        """Stopped from outside (HOLD): no blacklist, short cool-down."""
        if self.active:
            self.ev(t, qr=why)
            self.phase, self.result = IDLE, why
            self.cooldown_until = t + 3.0

    # ------------------------------------------------------------------ control
    def _phase(self, ph, t):
        self.phase, self.t_phase = ph, t

    def _finish(self, t, result, **info):
        if result != 'decoded':
            if self.target is not None:
                self.blacklist.append((float(self.target[0]), float(self.target[1])))
            self.cooldown_until = t + self.c['qr_cooldown_s']
        self.ev(t, qr=result, s=round(t - self.t0, 1), **info)
        self.phase, self.result = IDLE, result
        self.obs = []
        return None

    def step(self, t, pose, front, obs_body):
        """One driver tick while active. Returns the command, or None when the approach is over."""
        c = self.c
        x, y, yaw = pose
        if self.decoded is not None:
            return self._finish(t, 'decoded', text=self.decoded)
        if t - self.t0 > c['qr_max_s']:
            return self._finish(t, 'timeout')

        if self.phase == LOOK:
            win = [o for o in self.obs if o.t >= self.t_phase + c['qr_settle_s']]
            waited = t - self.t_phase
            if not win:
                if waited > c['qr_look_max_s']:
                    return self._finish(t, 'lost')
                return Cmd()
            if len(win) < 2 and waited < c['qr_settle_s'] + c['qr_look_s']:
                return Cmd()                      # a second sharp measurement if it comes in time
            o = self._median(win)
            p, dcam = self.locate(o, obs_body)
            self.target = G.body_to_world(p[None], pose)[0]
            bearing = math.atan2(p[1], p[0])      # seen from base_link (the dog turns about it)
            top = o.cy - o.w / 2.0
            if abs(bearing) > math.radians(c['qr_align_deg']):
                obs = np.asarray(obs_body, dtype=float).reshape(-1, 2)
                if obs.shape[0] and np.hypot(obs[:, 0], obs[:, 1]).min() < c['qr_turn_clear_m']:
                    return self._finish(t, 'no_room_to_turn')
                self.goal_yaw = G.wrap(yaw + bearing)
                self._phase(TURN, t)
                return Cmd()
            why = ('top_of_picture' if top <= c['qr_top_margin_px'] else
                   'close' if o.w >= c['qr_target_w'] else
                   'wall' if front <= c['qr_front_min_m'] else '')
            want = dcam * (1.0 - o.w / c['qr_target_w'])          # size ~ 1/distance
            # long steps while far (the heading error of one measurement grows little over them),
            # short ones close up, where the next look must catch the code before it leaves the picture
            cap = max(c['qr_step_max_m'], c['qr_step_frac'] * dcam)
            d_top = dcam - self.closest_in_picture(top, dcam)
            d = min(cap, want, front - c['qr_front_min_m'], d_top)
            if why or d < 0.03:
                self.read_why = why or ('top_of_picture' if d_top < 0.03 else 'close')
                self.ev(t, qr='read', why=self.read_why, w=round(o.w), top=round(top),
                        front=round(front, 2))
                self._phase(READ, t)
                return Cmd()
            self.step_d, self.step_from, self.goal_yaw = d, (x, y), G.wrap(yaw + bearing)
            self._phase(STEP, t)
            return Cmd()

        if self.phase == TURN:
            err = G.wrap(self.goal_yaw - yaw)
            if abs(err) < math.radians(2.0) or t - self.t_phase > 4.0:
                self._phase(LOOK, t)
                return Cmd()
            wz = _clamp(1.5 * err, -c['qr_wz'], c['qr_wz'])
            if abs(wz) < 0.12:
                wz = math.copysign(0.12, err)
            return Cmd(0.0, wz)

        if self.phase == STEP:
            moved = math.hypot(x - self.step_from[0], y - self.step_from[1])
            if (moved >= self.step_d or front <= c['qr_front_min_m']
                    or t - self.t_phase > self.step_d / c['qr_vx'] + 2.0):
                self._phase(LOOK, t)
                return Cmd()
            err = G.wrap(self.goal_yaw - yaw)
            vx = c['qr_vx_far'] if self.step_d - moved > 0.4 else c['qr_vx']
            return Cmd(vx, _clamp(1.0 * err, -c['walk_wz_max'], c['walk_wz_max']))

        if self.phase == READ:
            if t - self.t_phase > c['qr_read_s']:
                return self._finish(t, 'unreadable', why=self.read_why)
            return Cmd()
        return None
