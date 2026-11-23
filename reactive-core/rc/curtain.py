"""Curtain: detect a hanging cloth ahead (lidar), confirm it (camera), push through it (touch test).

Spec section 9.4: "lidar candidate AND camera confirmation AND a slow touch test (creep <= 0.15 m/s,
<= ~1 m; if progress stops it was a wall: back off by turning, mark as wall). The guard front zone is
relaxed only inside this mode. Fallback if the scene server is unusable: lidar + touch test."

CurtainDetector (every scan): the surface straight ahead in the levelled body frame, from the
nearest return per 2-degree bearing bin (obstacle heights), split into continuous segments; the one
ahead is fitted with a line. Features, each a physical difference between cloth and a panel:
  wavy    robust spread of the nearest-surface profile around the line (folds; panels are flat)
  thick   robust spread of ALL returns near the line (fabric texture / sway within one scan)
  pen     share of returns BEHIND the surface in its bearings (see-through cloth, gaps at the hem)
  flutter robust spread of the per-bin change between scans after removing the dog's own motion
score = weighted sum of the features normalised by cur_*_lo/hi. The lo values sit above what the
real panels and walls of the course produced (runs 2026-11-22/23, tools/curtain_probe.py); the hi
values are guesses until a real curtain has been recorded - the camera and the touch test decide.

CurtainPass (driver mode CURTAIN): ALIGN (face the surface square: shortest way through the
cloth, pure turn) -> INSPECT (stand still, ask the camera via perception /quad/vlm_ask; no answer
in time -> lidar + touch test) -> APPROACH (straight, normal guard, until the front stop) -> PUSH
(guard front stop relaxed, straight at cur_push_vx; TOUCH TEST: progress measured by FAST-LIO when
it is usable, else by leg odometry; < cur_stall_m in cur_stall_s = rigid -> marked as a wall) ->
done when base_link is cur_clear_m past the surface. Never reverses; walk OR turn, never both.
"""
import collections
import math
from dataclasses import dataclass, field

import numpy as np

from rc import geometry as G
from rc.guard import Cmd

BIN_DEG = 2.0
HALF_FOV_DEG = 50.0
NB = int(2 * HALF_FOV_DEG / BIN_DEG)


@dataclass
class Surface:
    t: float = 0.0
    dist: float = math.inf           # base_link -> surface line (m)
    bearing: float = 0.0             # direction INTO the surface, normal to it (rad, body frame, + = left)
    centre: tuple = (0.0, 0.0)       # body frame
    width: float = 0.0
    wavy: float = 0.0
    thick: float = 0.0
    pen: float = 0.0
    flutter: float = 0.0
    tall: float = 0.0               # share of 0.1 m height slices cur_h_lo..cur_tall_top_m with returns ON the surface
    score: float = 0.0
    n_bins: int = 0
    axis: tuple = (0.0, 1.0)         # unit vector along the surface (body frame)
    span: tuple = (-0.4, 0.4)        # cloth-like stretch along axis, relative to centre (m)
    profile: dict = field(default_factory=dict)      # bin -> nearest range (for flutter)

    def summary(self):
        return {'dist': round(self.dist, 2), 'bearing_deg': round(math.degrees(self.bearing), 1),
                'width': round(self.width, 2), 'wavy': round(self.wavy, 4), 'thick': round(self.thick, 4),
                'pen': round(self.pen, 3), 'flutter': round(self.flutter, 4), 'tall': round(self.tall, 2),
                'score': round(self.score, 3)}


def _robust_std(v):
    v = np.asarray(v, dtype=float)
    if v.size < 3:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _norm(v, lo, hi):
    return float(np.clip((v - lo) / max(1e-9, hi - lo), 0.0, 1.0))


class CurtainDetector:

    def __init__(self, cfg):
        self.c = cfg
        self.prev = None                     # previous scan's profile {bin: r}
        self.flutters = collections.deque(maxlen=4)      # per-bin flutter residuals of the last scans
        self.hist = collections.deque(maxlen=int(cfg['cur_persist_n']))
        self.last = None

    def update(self, p_level, floor_z, t, gyro_z=0.0):
        """One scan (levelled body frame). Returns the Surface ahead (or None)."""
        c = self.c
        p = np.asarray(p_level, dtype=float).reshape(-1, 3)
        s = self._surface(p, floor_z, t, gyro_z)
        self.last = s
        ok = s is not None and s.score >= c['cur_suspect_thr'] and s.dist <= c['cur_detect_m']
        self.hist.append((s.centre, s.bearing, s.score) if ok else None)
        return s

    def suspect(self):
        """Like confident() at the lower cur_suspect_thr: worth a camera question, no blind push."""
        return self.confident(self.c['cur_suspect_thr'])

    def confident(self, thr=None):
        """cur_persist_k of the last cur_persist_n scans scored as cloth ON THE SAME PATCH: window
        centres within cur_persist_m (the dog moves < 0.1 m in 0.5 s) and the SAME PLANE, normals
        within cur_persist_deg - a hanging sheet fits the same line every scan, clutter does not
        (the toy market stall, run 113959 t=64 s: 0, 12, 25, 11 deg in four scans)."""
        thr = self.c['cur_score_thr'] if thr is None else thr
        good = [g for g in self.hist if g is not None and g[2] >= thr]
        if len(good) < self.c['cur_persist_k']:
            return False
        tol = math.radians(self.c['cur_persist_deg'])
        for (rc_, rb, _) in good:
            same = [1 for (cc, bb, _) in good if math.hypot(cc[0] - rc_[0], cc[1] - rc_[1]) <= self.c['cur_persist_m']
                    and abs(G.wrap(bb - rb)) <= tol]
            if len(same) >= self.c['cur_persist_k']:
                return True
        return False

    # ------------------------------------------------------------------ geometry
    def _split_straight(self, sg, rn, min_bins=5):
        """Split a run of bins at its largest deviation from the end-to-end chord while that
        deviation exceeds cur_corner_m (cloth folds of a few cm stay below it)."""
        if len(sg) < 2 * min_bins:
            return [sg]
        ang = np.radians((np.array(sg) + 0.5) * BIN_DEG - HALF_FOV_DEG)
        P = np.stack([rn[sg] * np.cos(ang), rn[sg] * np.sin(ang)], axis=1)
        AB = P[-1] - P[0]
        L = float(np.linalg.norm(AB))
        if L < 1e-6:
            return [sg]
        dev = np.abs((P - P[0]) @ np.array([-AB[1], AB[0]]) / L)
        i = int(np.argmax(dev))
        if dev[i] > self.c['cur_corner_m'] and min_bins <= i <= len(sg) - min_bins:
            return self._split_straight(sg[:i + 1], rn, min_bins) + self._split_straight(sg[i:], rn, min_bins)
        return [sg]

    def _surface(self, p, floor_z, t, gyro_z):
        c = self.c
        if p.shape[0] == 0:
            self.prev = None
            return None
        h = p[:, 2] - floor_z
        r = np.hypot(p[:, 0], p[:, 1])
        a = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        inview = (np.abs(a) < HALF_FOV_DEG) & (r > 0.35) & (r < c['cur_max_r'])
        band = inview & (h > c['cur_h_lo']) & (h < c['cur_h_hi'])
        if band.sum() < 30:
            self.prev = None
            return None
        b = ((a + HALF_FOV_DEG) / BIN_DEG).astype(int).clip(0, NB - 1)
        rn = np.full(NB, np.inf)
        cnt = np.zeros(NB, int)
        for k in np.unique(b[band]):
            rk = r[band & (b == k)]
            cnt[k] = rk.size
            if rk.size >= 3:
                rn[k] = float(np.percentile(rk, 20))
        # continuous segments (one empty bin allowed)
        segs, cur, last = [], [], None
        for k in range(NB):
            if not np.isfinite(rn[k]):
                continue
            if cur and (k - last > 2 or abs(rn[k] - rn[last]) > c['cur_seg_jump_m']):
                segs.append(cur)
                cur = []
            cur.append(k)
            last = k
        if cur:
            segs.append(cur)
        # corners: two walls meeting would look "wavy" as one line - split where the profile bends
        segs = [piece for sg in segs for piece in self._split_straight(sg, rn)]
        # flutter: each bin's offset from ITS scan's fitted line, compared with the previous scan's.
        # The dog's own motion moves the line, not the offsets from it: a rigid wall stays at the
        # noise level even walked past obliquely; moving / folded cloth does not (2026-11-23: the
        # first version subtracted a common shift and saw 'flutter' on oblique flat walls)
        offs = {}

        # a curtain is a PATCH of a surface (a doorway in a panel wall, flush with it): every segment
        # is scored in sliding windows of ~a doorway's width; the most cloth-like window ahead wins,
        # or (nothing cloth-like) the window nearest straight ahead (for the 'stuck' camera check)
        slices = np.arange(c['cur_h_lo'], c['cur_tall_top_m'], 0.1)
        best, nearest = None, None
        for sg in segs:
            if len(sg) < 5:
                continue
            ang = np.radians((np.array(sg) + 0.5) * BIN_DEG - HALF_FOV_DEG)
            xy = np.stack([rn[sg] * np.cos(ang), rn[sg] * np.sin(ang)], axis=1)
            if float(np.linalg.norm(xy[-1] - xy[0])) < c['cur_min_width_m']:
                continue
            cen_s = xy.mean(axis=0)
            _, _, vt = np.linalg.svd(xy - cen_s, full_matrices=False)
            d = vt[0]
            if float((xy[-1] - xy[0]) @ d) < 0.0:
                d = -d                                  # 'along' grows with the bin index (windows)
            n = np.array([-d[1], d[0]])
            if float(n @ cen_s) < 0.0:
                n = -n                                  # n points from the dog into the surface
            for k_, o_ in zip(sg, (xy - cen_s) @ n):
                offs[int(k_)] = float(o_)
            if abs(math.degrees(math.atan2(n[1], n[0]))) > c['cur_normal_deg']:
                continue                                # seen too obliquely to square up to it
            # the points of this segment's bearings, once
            segmask = inview & np.isin(b, sg)
            ps, hs_all, bs = p[segmask], h[segmask], b[segmask]
            sd_all = (ps[:, :2] - cen_s) @ n            # + = behind the surface line
            along = xy @ d
            seg_wins = []                               # (along_lo, along_hi, Surface) of this segment
            for i in range(len(sg)):
                j = i
                while j + 1 < len(sg) and along[j + 1] - along[i] <= c['cur_win_m'] + 1e-9:
                    j += 1
                if abs(along[j] - along[i]) < c['cur_min_width_m']:
                    continue
                win = sg[i:j + 1]
                inner = win[1:-1] if len(win) > 6 else win          # window ends: posts / frame edges
                wxy = xy[i:j + 1]
                cen = wxy.mean(axis=0)
                cang = math.degrees(math.atan2(cen[1], cen[0]))
                if abs(cang) > c['cur_ahead_deg']:
                    continue
                sw = np.isin(bs, inner)
                inner_xy = xy[[sg.index(k) for k in inner]]
                wavy = _robust_std((inner_xy - cen_s) @ n)
                band_w = sw & (hs_all > c['cur_h_lo']) & (hs_all < c['cur_h_hi'])
                near = band_w & (np.abs(sd_all) < 0.12)
                thick = _robust_std(sd_all[near]) if near.sum() >= 10 else 0.0
                pw = sw & (hs_all > 0.15) & (hs_all < c['cur_h_hi'])
                behind = int((pw & (sd_all > 0.30) & (sd_all < 3.0)).sum())
                nearp = int((pw & (np.abs(sd_all) < 0.12)).sum())
                pen = behind / max(1, behind + nearp)
                hs = hs_all[sw & (np.abs(sd_all) < 0.10)]
                tall = float(np.mean([((hs >= z) & (hs < z + 0.1)).sum() >= 3 for z in slices])) if slices.size else 0.0
                fr = [r_[k] for r_ in self.flutters for k in inner if k in r_]
                flutter = _robust_std(fr) if len(fr) >= 8 else 0.0      # bins seen in both scans
                # a hanging sheet's folds are bounded; shelves / stalls / trolleys are rougher (the toy
                # market stall + wire trolley, run 113959 t=64 s: wavy 0.055-0.068)
                too_rough = wavy > c['cur_wavy_max']
                score = 0.0 if (tall < c['cur_tall_min'] or too_rough) else (
                    c['cur_w_wavy'] * _norm(wavy, c['cur_wavy_lo'], c['cur_wavy_hi'])
                    + c['cur_w_thick'] * _norm(thick, c['cur_thick_lo'], c['cur_thick_hi'])
                    + c['cur_w_pen'] * _norm(pen, c['cur_pen_lo'], c['cur_pen_hi'])
                    + c['cur_w_flutter'] * _norm(flutter, c['cur_flutter_lo'], c['cur_flutter_hi']))
                sf = Surface(t=t, dist=float(n @ cen), bearing=float(math.atan2(n[1], n[0])),
                             centre=(float(cen[0]), float(cen[1])), width=float(along[j] - along[i]),
                             wavy=wavy, thick=thick, pen=pen, flutter=flutter, tall=tall, score=score,
                             n_bins=len(win), axis=(float(d[0]), float(d[1])))
                seg_wins.append((float(along[i]), float(along[j]), sf))
                if nearest is None or abs(cang) < nearest[0]:
                    nearest = (abs(cang), sf)
            # the cloth-like stretch: windows overlapping the best one of this segment and scoring nearly
            # as much (cur_span_frac; a curtain across a whole corridor is wider than one window, but
            # a window half over the wall beside a curtained doorway must not widen it)
            if seg_wins:
                lo_b, hi_b, sb = max(seg_wins, key=lambda w_: w_[2].score)
                if sb.score > 0.0:
                    lo, hi = lo_b, hi_b
                    changed = True
                    while changed:
                        changed = False
                        for a_, b_, s_ in seg_wins:
                            if s_.score >= c['cur_span_frac'] * sb.score and a_ <= hi + 1e-6 and b_ >= lo - 1e-6 and (a_ < lo or b_ > hi):
                                lo, hi = min(lo, a_), max(hi, b_)
                                changed = True
                    mid = float(np.array(sb.centre) @ d)
                    sb.span = (lo - mid, hi - mid)
                if best is None or sb.score > best.score:
                    best = sb
        if self.prev is not None and abs(gyro_z) < 0.10:
            self.flutters.append({k: o - self.prev[k] for k, o in offs.items() if k in self.prev})
        self.prev = offs
        if best is not None and best.score > 0.0:
            return best
        return nearest[1] if nearest is not None else None


# ====================================================================== pass controller
IDLE, ALIGN, INSPECT, LINEUP, APPROACH, PUSH, RETREAT = (
    'IDLE', 'ALIGN', 'INSPECT', 'LINEUP', 'APPROACH', 'PUSH', 'RETREAT')


class CurtainPass:
    """Driver mode CURTAIN. step() returns the command while active, None when it is over."""

    def __init__(self, cfg, ev=None):
        self.c = cfg
        self.ev = ev or (lambda t, **kw: None)
        self.phase = IDLE
        self.push = False                 # guard: front stop relaxed (PUSH only)
        self.asks = []                    # camera questions to publish on /quad/vlm_ask
        self.answer = None                # True / False / None (unknown)
        self.ask_id = None
        self.n_ask = 0
        self.last_resort = False
        self.attempts = 0
        self.cooldown_until = -1e9
        self.marks = []                   # world (x, y, kind): 'wall' / 'curtain' / 'not_curtain'
        self.surf_w = None                # world (x, y) of the surface being handled
        self.t0 = self.t_phase = 0.0
        self.goal_yaw = 0.0
        self.push_from = None
        self.push_need = 0.0
        self.prog = None                  # (t, x, y) of the last progress
        self.moves = []                   # LINEUP: ('turn', yaw) / ('walk', metres), one at a time
        self.walk_from = None
        self.span, self.span_sign = (-0.4, 0.4), 1.0
        self.cam_down_until = -1e9        # unanswered camera question: skip camera-only starts for a while
        self.result = ''

    @property
    def active(self):
        return self.phase != IDLE

    # ---------------------------------------------------------------- inputs
    def on_answer(self, d):
        """/quad/vlm_answer from perception: {'id', 'answer', 'text', 'failed'?}."""
        if self.ask_id is None or d.get('id') != self.ask_id:
            return
        txt = str(d.get('text', ''))
        if d.get('failed') or txt.startswith('no image') or txt.startswith('query failed'):
            self.answer = 'unknown'
        else:
            self.answer = bool(d.get('answer'))

    # ---------------------------------------------------------------- trigger
    def _marked(self, w_xy):
        # a camera 'no' / a lost patch marks less ground than a touched wall or a passed curtain:
        # it must not hide a curtained doorway right next to a rejected wall patch
        for x, y, kind in self.marks:
            r = self.c['cur_mark_m'] if kind in ('wall', 'curtain') else self.c['cur_mark_soft_m']
            if math.hypot(w_xy[0] - x, w_xy[1] - y) < r:
                return True
        return False

    def wants(self, t, pose, surf, confident, last_resort, suspect=False):
        """confident: lidar says cloth (persistent) -> camera, else lidar + touch test;
        suspect: slightly cloth-like (persistent) and last_resort: stuck in front of a tall
        surface -> the camera alone decides (no push without its 'yes')."""
        c = self.c
        if (self.active or c['curtain'] not in ('pass', 'detect') or surf is None or t < self.cooldown_until
                or self.attempts >= c['cur_max_attempts'] or surf.dist > c['cur_detect_m']
                or abs(math.degrees(math.atan2(surf.centre[1], surf.centre[0]))) > c['cur_ahead_deg']):
            return None
        w_xy = G.body_to_world(np.array([surf.centre]), pose)[0]
        if self._marked(w_xy):
            return None
        if confident:
            return 'lidar'
        if t < self.cam_down_until:
            return None                    # camera not answering: a camera-only start is pointless
        if suspect:
            return 'suspect'
        if last_resort and surf.tall >= c['cur_tall_min']:
            return 'last_resort'
        return None

    def start(self, t, pose, surf, why):
        self.attempts += 1
        self.t0 = t
        self.last_resort = why in ('last_resort', 'suspect')       # camera 'yes' required
        self.surf_w = tuple(G.body_to_world(np.array([surf.centre]), pose)[0])
        # cloth stretch along the surface; its axis in the world vs the 'along' direction used in
        # _lineup ((-sin g, cos g) of the square heading g) - same or opposite sign
        self.span = surf.span
        ax_w = G.body_to_world(np.array([surf.axis]), (0.0, 0.0, pose[2]))[0]
        g0 = G.wrap(pose[2] + surf.bearing)
        self.span_sign = 1.0 if (ax_w[0] * -math.sin(g0) + ax_w[1] * math.cos(g0)) >= 0.0 else -1.0
        self.answer, self.ask_id = None, None
        self.push = False
        self.goal_yaw = G.wrap(pose[2] + surf.bearing)     # square to the surface
        self.ev(t, curtain='start', why=why, attempt=self.attempts, **surf.summary())
        self._phase(ALIGN, t)

    def abort(self, t, why='interrupted'):
        if self.active:
            self.ev(t, curtain=why)
            self.phase, self.push, self.result = IDLE, False, why
            self.cooldown_until = t + 5.0

    # ---------------------------------------------------------------- control
    def _phase(self, ph, t):
        self.phase, self.t_phase = ph, t

    def _finish(self, t, result, mark=None, **info):
        if mark and self.surf_w is not None:
            self.marks.append((float(self.surf_w[0]), float(self.surf_w[1]), mark))
        if result != 'passed':
            self.cooldown_until = t + self.c['cur_cooldown_s']
        self.ev(t, curtain=result, s=round(t - self.t0, 1), **info)
        self.phase, self.push, self.result = IDLE, False, result
        return None

    def _ask(self, t):
        if t < self.cam_down_until:
            self.answer = 'unknown'          # the camera did not answer a moment ago: do not wait again
            self.ev(t, curtain='camera_down_skip')
            return
        self.n_ask += 1
        self.ask_id = 'rc%d' % self.n_ask
        self.answer = None
        self.asks.append({'id': self.ask_id, 'kind': 'curtain'})
        self.ev(t, curtain='ask_camera', id=self.ask_id)

    def take_asks(self):
        a, self.asks = self.asks, []
        return a

    def step(self, t, pose, surf, front_base, obs_body, progress_xy):
        """front_base: free distance straight ahead from base_link (raw scan, body corridor);
        progress_xy: position for the touch test (FAST-LIO when usable, else leg odometry)."""
        c = self.c
        x, y, yaw = pose
        if t - self.t0 > c['cur_max_s']:
            return self._finish(t, 'timeout', mark='wall' if self.phase == PUSH else None)

        if self.phase == ALIGN:
            # turn to the heading measured at the start (surfaces flicker while turning); once
            # square, the surface must be back ahead within 1 s, else it was not a sheet ahead
            err = G.wrap(self.goal_yaw - yaw)
            if abs(err) <= math.radians(c['cur_align_deg']):
                ahead = (surf is not None and surf.dist <= c['cur_detect_m'] + 0.5
                         and abs(math.degrees(math.atan2(surf.centre[1], surf.centre[0]))) <= c['cur_ahead_deg'])
                if ahead:
                    self._phase(INSPECT, t)
                    self._ask(t)
                    return Cmd()
                self.aligned_t = getattr(self, 'aligned_t', None) or t
                if t - self.aligned_t > 1.0:
                    self.aligned_t = None
                    return self._finish(t, 'lost', mark='lost')
                return Cmd()
            self.aligned_t = None
            obs = np.asarray(obs_body, dtype=float).reshape(-1, 2)
            if obs.shape[0] and np.hypot(obs[:, 0], obs[:, 1]).min() < c['cur_turn_clear_m']:
                if abs(err) <= math.radians(25.0):        # close enough to square: go as it is
                    self._phase(INSPECT, t)
                    self._ask(t)
                    return Cmd()
                return self._finish(t, 'no_room_to_turn')
            if t - self.t_phase > 6.0:
                return self._finish(t, 'align_timeout')
            wz = float(np.clip(1.5 * err, -c['turn_wz'], c['turn_wz']))
            if abs(wz) < 0.12:
                wz = math.copysign(0.12, err)
            return Cmd(0.0, wz)

        if self.phase == INSPECT:
            waited = t - self.t_phase
            ans = self.answer
            if ans is True:
                self.ev(t, curtain='camera_yes')
                return self._lineup(t, pose, surf)
            elif ans is False:
                return self._finish(t, 'camera_no', mark='not_curtain')
            elif ans == 'unknown' or waited > c['cur_vlm_wait_s']:
                self.cam_down_until = t + c['cur_cam_retry_s']
                if self.last_resort:
                    # camera-only start and no camera: no blind push, but no mark either - if the
                    # lidar becomes sure of it later (closer, more scans), that path may still act
                    self.attempts -= 1
                    self.cooldown_until = t + 3.0
                    self.ev(t, curtain='no_camera_answer', s=round(t - self.t0, 1))
                    self.phase, self.result = IDLE, 'no_camera_answer'
                    return None
                self.ev(t, curtain='no_camera_answer_touch_test')
                return self._lineup(t, pose, surf)
            return Cmd()

        if self.phase == LINEUP:
            return self._do_moves(t, x, y, yaw, front_base)

        if self.phase == APPROACH:
            if front_base <= c['stop_m'] + c['cur_sensor_x'] + 0.04:
                self.push = True
                self.push_from = (x, y)
                self.prog = (t,) + tuple(progress_xy)
                dist = surf.dist if surf is not None and surf.dist < 1.0 else front_base
                self.push_need = dist + c['cur_clear_m']
                self.ev(t, curtain='push', need=round(self.push_need, 2))
                self._phase(PUSH, t)
                return Cmd(c['cur_push_vx'], 0.0)
            if t - self.t_phase > c['cur_detect_m'] / c['cur_approach_vx'] + 5.0:
                return self._finish(t, 'approach_timeout')
            return Cmd(c['cur_approach_vx'], 0.0)

        if self.phase == PUSH:
            done = math.hypot(x - self.push_from[0], y - self.push_from[1])
            if done >= self.push_need:
                self.ev(t, curtain='through', d=round(done, 2))
                return self._finish(t, 'passed', mark='curtain')
            t_p, px, py = self.prog
            if math.hypot(progress_xy[0] - px, progress_xy[1] - py) > c['cur_stall_m']:
                self.prog = (t, progress_xy[0], progress_xy[1])
            elif t - t_p > c['cur_stall_s'] and t - self.t_phase > c['cur_grace_s']:
                # touch test: pushing but not moving -> rigid (spec: back off, mark as wall). With the
                # nose on it a turn on the spot would sweep the front corners into it: first back off
                # <= cur_retreat_m straight over the ground just walked in on, then hand back
                return self._wall(t, x, y, done)
            if t - self.t_phase > self.push_need / c['cur_push_vx'] + 6.0:
                return self._wall(t, x, y, done, why='push_timeout')
            return Cmd(c['cur_push_vx'], 0.0)

        if self.phase == RETREAT:
            back = math.hypot(x - self.retreat_from[0], y - self.retreat_from[1])
            if back >= self.retreat_need or t - self.t_phase > self.retreat_need / c['cur_retreat_vx'] + 2.0:
                return self._finish(t, self.retreat_why, mark='wall', back=round(back, 2))
            return Cmd(-c['cur_retreat_vx'], 0.0)
        return None

    def _lineup(self, t, pose, surf):
        """Get in front of the cloth patch before pushing: square to the plane is not enough - the
        most cloth-like window can sit beside the dog's line (a curtained doorway in a wall: pushing
        straight on would hit the wall next to it). Standoff point cur_standoff_m out from the patch
        centre along the normal; turn - walk - turn (walk OR turn, team rule)."""
        c = self.c
        x, y, yaw = pose
        if surf is not None:
            w_xy = G.body_to_world(np.array([surf.centre]), pose)[0]
            if math.hypot(w_xy[0] - self.surf_w[0], w_xy[1] - self.surf_w[1]) < 0.5:
                self.surf_w = (float(w_xy[0]), float(w_xy[1]))         # same patch, fresher fix
                self.span = surf.span
        g = self.goal_yaw
        nx, ny = math.cos(g), math.sin(g)                              # into the surface
        tx, ty = -ny, nx                                               # along it
        px, py = self.surf_w
        a = (x - px) * tx + (y - py) * ty          # where the dog's straight line meets the surface
        lo, hi = self.span
        if self.span_sign < 0:
            lo, hi = -hi, -lo
        m = c['cur_body_half_m']
        lo_ok, hi_ok = lo + m, hi - m
        if lo_ok > hi_ok:                          # stretch narrower than the body: its centre
            lo_ok = hi_ok = 0.5 * (lo + hi)
        # strict: the whole body must meet cloth. Outside the safe stretch -> walk to a point
        # cur_lineup_m inside it (a tolerance outside it put the body edge on the door post)
        self.moves = []
        if not (lo_ok - 1e-6 <= a <= hi_ok + 1e-6):
            inset = min(c['cur_lineup_m'], 0.5 * (hi_ok - lo_ok))
            target = min(max(a, lo_ok + inset), hi_ok - inset)
            qx, qy = px + target * tx, py + target * ty
            sx, sy = qx - nx * c['cur_standoff_m'], qy - ny * c['cur_standoff_m']
            lat = a - target
            d = math.hypot(sx - x, sy - y)
            self.moves = [('turn', math.atan2(sy - y, sx - x)), ('walk', d), ('turn', g)]
            self.ev(t, curtain='lineup', lateral=round(lat, 2), walk=round(d, 2))
        self._phase(LINEUP, t)
        return Cmd()

    def _do_moves(self, t, x, y, yaw, front_base):
        c = self.c
        if not self.moves:
            self._phase(APPROACH, t)
            return Cmd()
        kind, val = self.moves[0]
        if kind == 'turn':
            err = G.wrap(val - yaw)
            if abs(err) <= math.radians(3.0):
                self.moves.pop(0)
                self.t_phase = t
                return Cmd()
            if t - self.t_phase > 8.0:
                return self._finish(t, 'lineup_timeout')
            wz = float(np.clip(1.5 * err, -c['turn_wz'], c['turn_wz']))
            return Cmd(0.0, wz if abs(wz) >= 0.12 else math.copysign(0.12, err))
        if self.walk_from is None:
            self.walk_from = (x, y)
        moved = math.hypot(x - self.walk_from[0], y - self.walk_from[1])
        if moved >= val:
            self.moves.pop(0)
            self.walk_from = None
            self.t_phase = t
            return Cmd()
        if front_base <= c['stop_m'] + c['cur_sensor_x'] + 0.05 or t - self.t_phase > val / c['cur_approach_vx'] + 4.0:
            self.walk_from = None
            return self._finish(t, 'lineup_blocked')
        return Cmd(c['cur_approach_vx'], 0.0)

    def _wall(self, t, x, y, done, why='wall'):
        self.push = False
        # never further back than the dog came in straight (push + approach), at most cur_retreat_m
        self.retreat_need = min(self.c['cur_retreat_m'], done + 0.3)
        self.retreat_why = why
        if self.retreat_need <= 0.01:
            return self._finish(t, why, mark='wall', d=round(done, 2))
        self.retreat_from = (x, y)
        self.ev(t, curtain='touch_wall', d=round(done, 2), retreat=round(self.retreat_need, 2))
        self._phase(RETREAT, t)
        return Cmd()
