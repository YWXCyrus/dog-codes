"""Driving state machine (spec sections 6 and 9.3).

Modes: CRUISE, TURN, BLOCKED, HOLD, STAIRS (phases LOOK, APPROACH, ENTER, CLIMB, DESCEND, EXIT),
BAR (SQUARE, CROSS: a low cylinder across the lane, rc/bar.py), CURTAIN (ALIGN, PUSH: rc/curtain.py).
The driver only proposes a command; the guard (rc/guard.py) filters every command afterwards.
Headings in lanes are body-relative (0 = straight ahead, + = left); targets are stored as
odom-frame yaw so they survive the dog's own rotation.
"""
import collections
import math
from dataclasses import dataclass, field

import numpy as np

from rc import gait as Gt
from rc import geometry as G
from rc.bar import BarResult
from rc.config import mode as cfg_mode
from rc.curtain import CurtainResult
from rc.guard import Cmd
from rc.stairs import StairsResult

CRUISE, TURN, BLOCKED, HOLD, STAIRS = 'CRUISE', 'TURN', 'BLOCKED', 'HOLD', 'STAIRS'
BAR, CURTAIN = 'BAR', 'CURTAIN'


@dataclass
class DriverInput:
    t: float
    lanes: list
    obs: np.ndarray                     # persistent obstacles (body frame), for the wall-side check
    mem: np.ndarray                     # FAST-LIO map obstacle / drop cells (body frame); empty without it
    pose: tuple                         # odom frame (x, y, yaw)
    odom_ok: bool
    travelled: float
    stairs: StairsResult = field(default_factory=StairsResult)
    gyro_z: float = 0.0
    tilt_deg: float = 0.0
    pitch_deg: float = 0.0              # + = nose down (ROS)
    last_cmd: Cmd = field(default_factory=Cmd)   # previous GUARDED command (what the dog was sent)
    dist_from_start: float = 0.0
    front_dist: float = math.inf        # free distance straight ahead (raw scan), for the odometry cross-check
    hint_yaw: float = None              # frontier planner: yaw (odom frame) to head for; None = no advice
    bar: BarResult = field(default_factory=BarResult)
    curtain: CurtainResult = field(default_factory=CurtainResult)


@dataclass
class DriverOutput:
    cmd: Cmd
    mode: str
    phase: str = ''
    codes: list = field(default_factory=list)
    events: list = field(default_factory=list)
    stairs_mode: bool = False
    ignore_front_x: float = 0.0         # guard: front stop ignores obstacles nearer than this (bar / curtain)


class Driver:

    def __init__(self, cfg, grid):
        self.c = cfg
        self.grid = grid
        self.mode = CRUISE
        self.phase = ''
        self.hold_reason = ''
        self.side = 1 if cfg['wall_side'] == 'left' else -1
        self.t_start = None
        self.target_yaw = None
        self.target_t = -1e9
        self.turn_dir = 0
        self.turn_t = -1e9
        self.blocked_turned = 0.0
        self.blocked_left_t = -1e9      # when BLOCKED last ended (blocked_memory_s)
        self.blocked_dir = 0            # BLOCKED turn direction, kept across short re-entries
        self.free_streak = 0            # decisions in a row with a free lane in the same direction
        self.streak_yaw = None
        self.last_yaw = None
        self.last_new_cell_t = None
        self.prog = collections.deque()
        self.stairs_done = False
        self.ignore_unsure_until = -1e9
        self.seq = None
        self.seq_kind = ''
        self.stair = {}
        self.events = []
        self.codes = []
        self.last_code_t = -1e9
        self.stairs_mode = False
        self._resume_phase = None
        self.odom_frozen = False        # leg odometry did not move while the lidar saw us move
        self.cmd_dist = 0.0             # distance by the guarded command (fallback when odometry is frozen)
        self._last_step_t = None
        self.front_ignore_x = 0.0       # set while crossing a bar / pushing a curtain, else 0
        self.hint_failed = False        # set when the planner's hint led into BLOCKED (core tells the planner)
        self.bar_s = {}
        self.bar_cool_until = -1e9
        self.bar_seen_t = -1e9
        self.curtain_s = {}
        self.curtain_cool_until = -1e9
        self.curtain_seen_t = -1e9

    # ---------------------------------------------------------------- helpers
    def _dist(self, di):
        return self.cmd_dist if self.odom_frozen else di.travelled
    def _ev(self, t, **kw):
        kw['t'] = round(t, 2)
        self.events.append(kw)

    def _hold(self, t, reason, keep_stairs=False):
        if self.mode == HOLD and self.hold_reason == reason:
            return
        self._resume_phase = self.phase if keep_stairs else None
        self.front_ignore_x = 0.0
        self.mode, self.hold_reason = HOLD, reason
        if not keep_stairs:
            self.phase = ''
        self._ev(t, hold=reason)

    def resume(self, t=0.0):
        if self.mode != HOLD:
            return
        reason = self.hold_reason
        self._ev(t, resume=reason)
        self.hold_reason = ''
        self.prog.clear()
        if self._resume_phase in ('CLIMB', 'DESCEND'):
            self.mode, self.phase = STAIRS, self._resume_phase
            self.stair['t_phase'] = t
            self.stair['prog_t'] = t
            self.stair['prog_d'] = None
        elif reason == 'rug_no_response' or self._resume_phase in ('ENTER', 'ALIGN', 'EXIT', 'ABORT'):
            # the dog may be in AI state: always leave it through the full exit sequence
            self._start_seq(t, Gt.EXIT_AI, 'EXIT')
        else:
            if reason == 'stairs_unsure':
                self.ignore_unsure_until = t + 10.0
            self.mode, self.phase = CRUISE, ''
            self.stairs_mode = False
        self._resume_phase = None

    @staticmethod
    def _lane_near(lanes, h):
        return min(lanes, key=lambda l: abs(G.wrap(l.heading - h)))

    def _wall_on_side(self, di):
        pts = [np.asarray(di.obs).reshape(-1, 2), np.asarray(di.mem).reshape(-1, 2)]
        p = np.vstack(pts)
        if p.shape[0] == 0:
            return False
        s = self.side * p[:, 1]
        return bool(np.any((p[:, 0] > -0.4) & (p[:, 0] < 0.8) & (s > 0.2) & (s < self.c['wall_dist_m'] + 0.6)))

    def _choose(self, di):
        """The lane to take; None if nothing is free.

        nav_mode planner: the free lane closest to the frontier planner's hint; without a usable hint
        straight ahead if free, else the smallest turn (the wall side only breaks ties). The old
        "leftmost free lane" rule turned the dog whenever a wall was near for no reason (2026-11-25).
        nav_mode wall: the old wall following."""
        free = [l for l in di.lanes if l.free]
        if not free:
            return None
        # lanes within +-90 deg first (wall side first); lanes towards the rear only when nothing
        # ahead is free - the rear is blind, a "free" rear lane is unverified (trial 2: low wall)
        ahead = [l for l in free if abs(l.heading) <= math.pi / 2 + 1e-6]
        rear = [l for l in free if abs(l.heading) > math.pi / 2 + 1e-6]
        if cfg_mode(self.c, 'nav_mode') == 'planner':
            if di.hint_yaw is not None:
                h = G.wrap(di.hint_yaw - di.pose[2])
                dev = math.radians(self.c['plan_max_dev_deg'])
                cand = [l for l in free if abs(G.wrap(l.heading - h)) <= dev]
                if cand:
                    return min(cand, key=lambda l: abs(G.wrap(l.heading - h)))
            return (sorted(ahead, key=lambda l: (round(abs(l.heading), 3), -self.side * l.heading))
                    + sorted(rear, key=lambda l: abs(l.heading)))[0]
        order = (sorted(ahead, key=lambda l: -self.side * l.heading)
                 + sorted(rear, key=lambda l: abs(l.heading)))
        if not self._wall_on_side(di):
            straight = self._lane_near(di.lanes, 0.0)
            if straight.free and abs(straight.heading) < 1e-6:
                order = [straight] + [l for l in order if l is not straight]
        cell = self.grid.current
        if cell is not None and self.grid.origin is not None:
            ok = [l for l in order
                  if not self.grid.is_dead_end(cell, self.grid.heading_dir(di.pose[2] + l.heading))]
            order = ok or order
            first = order[0]
            near = [l for l in order if abs(G.wrap(l.heading - first.heading)) <= math.radians(15.0) + 1e-6]
            unvisited = [l for l in near if self.grid.neighbour(
                cell, self.grid.heading_dir(di.pose[2] + l.heading)) not in self.grid.visited]
            if unvisited:
                return unvisited[0]
        return order[0]

    def _progress(self, di):
        """True if the dog was sent forward for progress_s but moved < progress_ratio of it."""
        c = self.c
        t = di.t
        self.prog.append((t, di.last_cmd.vx, di.travelled, di.front_dist))
        while self.prog and t - self.prog[0][0] > c['progress_s']:
            self.prog.popleft()
        if not di.odom_ok or len(self.prog) < 3 or t - self.prog[0][0] < c['progress_s'] * 0.95:
            return False
        if any(p[1] < 0.08 for p in self.prog):
            return False
        sent = sum(p[1] for p in self.prog) * (t - self.prog[0][0]) / max(1, len(self.prog) - 1)
        moved = self.prog[-1][2] - self.prog[0][2]
        if moved >= c['progress_ratio'] * sent:
            return False
        # odometry says "not moving": does the lidar agree? If the wall ahead came closer by at
        # least half the commanded distance, the odometry is frozen, not the dog.
        f0, f1 = self.prog[0][3], self.prog[-1][3]
        if f0 < 2.9 and f1 < 2.9 and (f0 - f1) > 0.5 * sent:
            self.odom_frozen = True
            self.prog.clear()
            self._ev(t, odom_frozen='leg odometry moved %.2f m, lidar %.2f m' % (moved, f0 - f1))
            return False
        return True

    def _start_seq(self, t, steps, kind, turn_sign=1):
        self.mode, self.phase = STAIRS, kind
        self.seq = Gt.GaitSequencer(steps, self.c, turn_sign=turn_sign, last_code_t=self.last_code_t)
        self.seq_kind = kind
        self.stairs_mode = True
        self._ev(t, stairs=kind)

    def _run_seq(self, di):
        st, cmd, code = self.seq.tick(di.t, di.gyro_z)
        if code is not None:
            self.codes.append(code)
            self.last_code_t = di.t
        return st, cmd

    # ---------------------------------------------------------------- main
    def step(self, di):
        c = self.c
        t = di.t
        self.events, self.codes = [], []
        if self.t_start is None:
            self.t_start = t
            self.last_new_cell_t = t
        yaw = di.pose[2]
        dyaw = 0.0 if self.last_yaw is None else G.wrap(yaw - self.last_yaw)
        self.last_yaw = yaw
        if self._last_step_t is not None:
            self.cmd_dist += max(0.0, di.last_cmd.vx) * max(0.0, min(t - self._last_step_t, 0.5))
        self._last_step_t = t

        if self.grid.origin is not None and di.odom_ok:
            new = self.grid.update(di.pose)
            if new is not None:
                self.last_new_cell_t = t
                self._ev(t, cell=list(new))
        if t - self.last_new_cell_t > c['side_switch_s']:
            self.side = -self.side
            self.last_new_cell_t = t
            self._ev(t, wall_side='left' if self.side > 0 else 'right')

        # run timer: never in the middle of the stairs (waits until they are done or abandoned)
        on_stairs = self.mode == STAIRS and self.phase not in ('LOOK', 'APPROACH')
        if c['run_time_s'] > 0 and t - self.t_start > c['run_time_s'] and self.mode != HOLD and not on_stairs:
            self._hold(t, 'time')

        if self.mode not in (BAR, CURTAIN):
            self.front_ignore_x = 0.0

        if self.mode == HOLD:
            return self._out(Cmd())

        if self.mode == STAIRS:
            return self._out(self._stairs(di, dyaw))
        if self.mode == BAR:
            return self._out(self._bar(di))
        if self.mode == CURTAIN:
            return self._out(self._curtain(di))

        # ---- stairs triggers (normal modes) ----
        st = di.stairs
        if c['stairs'] != 'off' and not self.stairs_done:
            if st.kind == 'up' and st.confident and st.edge_x < 2.5:
                if c['stairs'] == 'climb':
                    self._begin_approach(di)
                    return self._out(self._stairs(di, dyaw))
                self._ev(t, stairs_seen='up', heading=round(math.degrees(st.heading), 1),
                         edge=round(st.edge_x, 2))
            elif (st.kind == 'unsure' and st.edge_x < 1.5 and t > self.ignore_unsure_until
                  and c['stairs'] == 'climb'):
                self.mode, self.phase = STAIRS, 'LOOK'
                self.stair = {'t_phase': t, 'last': 'unsure'}
                self._ev(t, stairs='LOOK', edge=round(st.edge_x, 2))
                return self._out(Cmd())
            elif st.kind in ('up', 'unsure') and c['stairs'] == 'detect':
                self._ev(t, stairs_seen=st.kind, edge=round(st.edge_x, 2))

        cmd = self._obstacle_triggers(di)
        if cmd is not None:
            return self._out(cmd)

        if self._progress(di):
            cell = self.grid.current
            if cell is not None and self.grid.origin is not None:
                self.grid.mark_dead_end(cell, self.grid.heading_dir(yaw))
            self._hold(t, 'no_progress')
            return self._out(Cmd())

        return self._out(self._navigate(di, dyaw))

    def _navigate(self, di, dyaw):
        c = self.c
        t = di.t
        yaw = di.pose[2]
        # keep a committed target while it is still free
        lane = None
        if self.target_yaw is not None and t - self.target_t < c['commit_s']:
            l = self._lane_near(di.lanes, G.wrap(self.target_yaw - yaw))
            if l.free:
                lane = l
        if lane is None:
            lane = self._choose(di)
            if lane is not None:
                self.target_yaw = G.wrap(yaw + lane.heading)
                self.target_t = t
        # a lane must be free blocked_confirm_n decisions in a row (same direction) before BLOCKED ends:
        # 061517 - single-scan flickers alternately left and right made the dog swing back and forth
        if lane is None:
            self.free_streak, self.streak_yaw = 0, None
        else:
            h = G.wrap(yaw + lane.heading)
            same = self.streak_yaw is not None and abs(G.wrap(h - self.streak_yaw)) <= math.radians(20.0)
            self.free_streak = self.free_streak + 1 if same else 1
            self.streak_yaw = h
        if lane is None or (self.mode == BLOCKED and self.free_streak < c['blocked_confirm_n']):
            if self.mode != BLOCKED and di.hint_yaw is not None:
                self.hint_failed = True
            return self._blocked(di, dyaw)
        if self.mode == BLOCKED:
            self.blocked_left_t = t
        # turn on the spot OR walk straight, never both (2026-11-23: slips at ledges while doing both):
        # heading error > head_err_turn_deg -> TURN until it is below turn_done_deg, then walk with wz = 0
        err = G.wrap(self.target_yaw - yaw)
        if self.mode in (BLOCKED, TURN):
            if abs(err) > math.radians(c['turn_done_deg']):
                if self.mode == BLOCKED:
                    self.mode = TURN
                    self.turn_t = t
                    self.turn_dir = 1 if err > 0 else -1
                elif abs(err) < math.pi / 2:
                    self.turn_dir = 1 if err > 0 else -1      # overshoot: turn back, not round a circle
                return self._turn(err, c['turn_wz'])
            self.mode = CRUISE
        if abs(err) > math.radians(c['head_err_turn_deg']):
            if self.mode != TURN or t - self.turn_t > c['commit_s']:
                self.turn_dir = 1 if err > 0 else -1
                self.turn_t = t
            self.mode = TURN
            return self._turn(err, c['turn_wz'])
        self.mode = CRUISE
        f0 = self._lane_near(di.lanes, 0.0).free_dist
        k = max(0.0, min(1.0, (f0 - 0.8) / 1.2))
        vx = c['cruise_vx_min'] + (c['cruise_vx'] - c['cruise_vx_min']) * k
        return Cmd(vx, 0.0)

    def _blocked(self, di, dyaw):
        """No (confirmed) free lane: turn on the spot one way. Blocked again within blocked_memory_s
        keeps the direction and the turned angle, so flickering lanes cannot make it swing back and
        forth; a full circle without a lane -> HOLD blocked."""
        c, t = self.c, di.t
        if self.mode != BLOCKED:
            if t - self.blocked_left_t > c['blocked_memory_s'] or self.blocked_dir == 0:
                self.blocked_turned = 0.0
                if self.mode == TURN and self.turn_dir:
                    self.blocked_dir = self.turn_dir          # already turning: keep going that way
                else:
                    best = max(di.lanes, key=lambda l: l.free_dist)
                    self.blocked_dir = 1 if best.heading >= 0 else -1
            self.mode = BLOCKED
            self._ev(t, mode=BLOCKED)
        self.turn_dir = self.blocked_dir
        self.blocked_turned += abs(dyaw)
        if self.blocked_turned > 2 * math.pi:
            self._hold(t, 'blocked')
            return Cmd()
        return Cmd(0.0, self.blocked_dir * c['turn_wz'])

    def _turn(self, err, wz_max, direction=None):
        """On-the-spot turn; slows to turn_wz_min in the last turn_slow_deg (the dog answers ~0.3 s
        late, full speed to the last degree overshoots past turn_done_deg)."""
        c = self.c
        d = self.turn_dir if direction is None else direction
        rate = wz_max * abs(err) / math.radians(c['turn_slow_deg'])
        rate = max(min(c['turn_wz_min'], wz_max), min(wz_max, rate))
        return Cmd(0.0, d * rate)

    # ---------------------------------------------------------------- stairs
    def _begin_approach(self, di):
        st = di.stairs
        x, y, yaw = di.pose
        h = yaw + st.heading
        self.mode, self.phase = STAIRS, 'APPROACH'
        self.stairs_mode = False
        self.stair = {'t_phase': di.t, 'yaw': G.wrap(h), 'yaws': [],
                      'edge': (x + st.edge_x * math.cos(h), y + st.edge_x * math.sin(h)),
                      't0': di.t}
        self._ev(di.t, stairs='APPROACH', edge=round(st.edge_x, 2), heading=round(math.degrees(st.heading), 1))

    @staticmethod
    def _median_angle(angles):
        ref = angles[-1]
        return G.wrap(ref + float(np.median([G.wrap(a - ref) for a in angles])))

    def _along(self, pose, point, yaw):
        """Signed distance from pose to point along direction yaw."""
        return (point[0] - pose[0]) * math.cos(yaw) + (point[1] - pose[1]) * math.sin(yaw)

    def _stairs(self, di, dyaw):
        c = self.c
        t = di.t
        s = self.stair
        st = di.stairs
        x, y, yaw = di.pose

        if self.phase == 'LOOK':
            if st.kind == 'up' and st.confident:
                self._begin_approach(di)
                return Cmd()
            if st.kind in ('up', 'unsure'):
                s['last'] = st.kind
            if t - s['t_phase'] >= c['stair_look_s']:
                if s.get('last') == 'unsure' and st.kind != 'none':
                    self._hold(t, 'stairs_unsure')
                else:
                    self.mode, self.phase = CRUISE, ''
                    self.ignore_unsure_until = t + 10.0
                    self._ev(t, stairs='LOOK_none')
            return Cmd()

        if self.phase == 'APPROACH':
            settling = t - s['t0'] < 1.0       # stand still 1 s: headings from a still dog are far better
            if st.kind == 'up' and st.confident and st.edge_x > (0.7 if settling else 0.9):
                h = yaw + st.heading
                s['yaws'] = (s['yaws'] + [G.wrap(h)])[-9:]
                s['yaw'] = self._median_angle(s['yaws'])      # one noisy scan must not steer
                s['edge'] = (x + st.edge_x * math.cos(h), y + st.edge_x * math.sin(h))
            if settling:
                return Cmd()
            err = G.wrap(s['yaw'] - yaw)
            dist = self._along(di.pose, s['edge'], s['yaw'])
            # turn on the spot OR walk straight: error > 5 deg -> turn until below turn_done_deg
            if abs(err) > math.radians(5.0):
                s['turning'] = True
            if s.get('turning'):
                if abs(err) > math.radians(c['turn_done_deg']):
                    return self._turn(err, 0.25, direction=1 if err > 0 else -1)
                s['turning'] = False
            if dist <= c['stair_standoff_m']:
                self._start_seq(t, Gt.ENTER_AI, 'ENTER')
                return Cmd()
            if t - s['t0'] > 30.0:
                self._hold(t, 'stairs_approach')
                return Cmd()
            return Cmd(0.10, 0.0)

        if self.phase in ('ENTER', 'EXIT', 'ABORT'):
            st_, cmd = self._run_seq(di)
            if st_ == 'running':
                return cmd
            if self.phase == 'ENTER':
                if st_ == 'done':
                    self.phase = 'ALIGN'                  # the turn test rotated the dog: face the stairs again
                    s.update(t_phase=t)
                    self._ev(t, stairs='ALIGN')
                else:
                    self._ev(t, stairs_fail=self.seq.failed_reason)
                    self.stairs_done = True              # the Captain takes the stairs
                    self._start_seq(t, Gt.EXIT_AI, 'ABORT')
                return Cmd()
            if self.phase == 'ABORT':
                self.stairs_mode = False
                self._hold(t, 'ai_no_response' if st_ == 'done' else 'rug_no_response')
                return Cmd()
            # EXIT
            if st_ == 'done':
                self.stairs_done = True
                self.stairs_mode = False
                self.mode, self.phase = CRUISE, ''
                self.target_yaw = None
                self._ev(t, stairs='DONE')
            else:
                self._hold(t, 'rug_no_response')
            return Cmd()

        if self.phase == 'ALIGN':
            err = G.wrap(s['yaw'] - yaw)
            if abs(err) > math.radians(4.0) and t - s['t_phase'] < 10.0:
                return Cmd(0.0, math.copysign(0.2, err))
            self.phase = 'CLIMB'
            s.update(t_phase=t, t_climb=t, prog_t=t, prog_d=None, flat_since=None, flat_d=None)
            self._ev(t, stairs='CLIMB')
            return Cmd()

        if self.phase in ('CLIMB', 'DESCEND'):
            if t - s['t_climb'] > c['stair_max_s']:
                self._hold(t, 'stairs_timeout', keep_stairs=True)
                return Cmd()
            # tilt: the guard stops and resumes by itself (no HOLD)
            # no progress on a step (only while the guard let us move)
            d_now = self._dist(di)
            if di.last_cmd.vx >= 0.05 and not self.odom_frozen:
                if s.get('prog_d') is None:
                    s['prog_d'], s['prog_t'] = d_now, t
                elif d_now - s['prog_d'] > 0.05:
                    s['prog_d'], s['prog_t'] = d_now, t
                elif t - s['prog_t'] > c['stair_no_progress_s']:
                    self._hold(t, 'stairs_no_progress', keep_stairs=True)
                    return Cmd()
            else:
                s['prog_d'], s['prog_t'] = None, t
            # walk straight, never turn on a step; drifted too far -> HOLD in AI, the Captain lines it up
            err = G.wrap(s['yaw'] - yaw)
            if abs(err) > math.radians(c['stair_hold_err_deg']):
                self._hold(t, 'stairs_heading', keep_stairs=True)
                return Cmd()
            wz = 0.0
            if self.phase == 'CLIMB':
                # fallback: the lidar may never see the way down from the platform (the real Mid-360
                # gets only 1-2 rings per tread). The same ~1 s pitch filter as DESCEND: a sustained
                # nose-down pitch means the dog is already going down -> DESCEND, went_down already set.
                pc = s.get('pitch_c', di.pitch_deg)
                pc += 0.1 * (di.pitch_deg - pc)
                s['pitch_c'] = pc
                by = 'lidar' if st.kind == 'down' else ('imu' if pc > 10.0 else None)
                if by is not None:
                    if by == 'lidar':
                        h = yaw + st.heading
                        s['fall'] = (x + st.edge_x * math.cos(h), y + st.edge_x * math.sin(h))
                    self.phase = 'DESCEND'
                    s['t_phase'] = t
                    s['flat_since'], s['flat_d'] = None, None
                    s['pitch_f'] = pc if by == 'imu' else di.pitch_deg
                    if by == 'imu':
                        s['went_down'] = True
                    self._ev(t, stairs='DESCEND', by=by,
                             edge=round(st.edge_x, 2) if by == 'lidar' else None)
                return Cmd(c['stair_vx_up'], wz)
            # DESCEND: exit when the IMU says we went down (nose-down pitch) and are level again,
            # nothing lower is seen ahead, and that for stair_flat_exit_m of travel. (The down-edge
            # distance seen from the platform is unreliable: the lidar sees the lower steps far out.)
            # filtered pitch (~1 s): walking wobble on flat floor reaches 5-12 deg in single samples
            # (trial 2), the stairs give a sustained ~27 deg
            pf = s.get('pitch_f', di.pitch_deg)
            pf += 0.1 * (di.pitch_deg - pf)
            s['pitch_f'] = pf
            if pf > 10.0:
                s['went_down'] = True
            if not s.get('went_down') and d_now - s.get('desc_d0', d_now) > 4.0:
                self._hold(t, 'stairs_lost', keep_stairs=True)
                return Cmd()
            s.setdefault('desc_d0', d_now)
            level = abs(pf) < 6.0 and st.kind != 'down'
            if s.get('went_down') and level:
                if s['flat_d'] is None:
                    s['flat_d'] = d_now
                if d_now - s['flat_d'] >= c['stair_flat_exit_m']:
                    self._start_seq(t, Gt.EXIT_AI, 'EXIT')
                    return Cmd()
            else:
                s['flat_d'] = None
            return Cmd(c['stair_vx_down'], wz)
        return Cmd()

    # ---------------------------------------------------------------- bar / curtain
    def _obstacle_triggers(self, di):
        """Start a bar crossing or a curtain push; returns the command for this tick or None."""
        c, t = self.c, di.t
        bm, cm = cfg_mode(c, 'bar'), cfg_mode(c, 'curtain')
        cu = di.curtain
        if cm != 'off' and cu.found and cu.dist < c['curtain_trigger_m'] and t > self.curtain_cool_until:
            if cm == 'push':
                self.mode, self.phase = CURTAIN, 'ALIGN'
                self.curtain_s = {'t0': t, 'tp': t, 'dist': cu.dist, 'bearing': cu.bearing,
                                  'yaw': G.wrap(di.pose[2] + cu.bearing)}
                self._ev(t, curtain='ALIGN', dist=round(cu.dist, 2), bearing=round(math.degrees(cu.bearing), 1),
                         width=round(cu.width, 2), conf=round(cu.conf, 2))
                return self._curtain(di)
            if t - self.curtain_seen_t > 3.0:
                self.curtain_seen_t = t
                self._ev(t, curtain_seen=round(cu.dist, 2), bearing=round(math.degrees(cu.bearing), 1))
        b = di.bar
        if bm != 'off' and b.confident and b.dist < c['bar_trigger_m'] and t > self.bar_cool_until:
            if bm == 'cross':
                self.mode, self.phase = BAR, 'SQUARE'
                self.bar_s = {'t0': t, 'tp': t, 'x0': b.dist, 'depth': b.depth, 'yaws': [],
                              'yaw': G.wrap(di.pose[2] + b.corr)}
                self._ev(t, bar='SQUARE', dist=round(b.dist, 2), depth=round(b.depth, 2),
                         height=round(b.height, 2), corr=round(math.degrees(b.corr), 1))
                return self._bar(di)
            if t - self.bar_seen_t > 3.0:
                self.bar_seen_t = t
                self._ev(t, bar_seen=round(b.dist, 2), height=round(b.height, 2))
        return None

    def _finish_obstacle(self, t, ev, cool_attr, cool_s):
        self.mode, self.phase = CRUISE, ''
        self.front_ignore_x = 0.0
        self.target_yaw = None
        setattr(self, cool_attr, t + cool_s)
        self.prog.clear()
        self._ev(t, **ev)

    def _bar(self, di):
        """SQUARE: stand 0.5 s, turn to face the bar; CROSS: walk straight bar_vx, front stop ignores the
        bar's own cells; HOLD on heading drift / no progress (no reverse, no turning on the bar)."""
        c, t = self.c, di.t
        s = self.bar_s
        yaw = di.pose[2]
        if t - s['t0'] > 45.0:
            self._hold(t, 'bar_timeout')
            return Cmd()
        if self.phase == 'SQUARE':
            if di.bar.found:
                s['x0'], s['depth'] = di.bar.dist, di.bar.depth
                s['yaws'] = (s['yaws'] + [G.wrap(yaw + di.bar.corr)])[-9:]
                s['yaw'] = self._median_angle(s['yaws'])
            if t - s['tp'] < 0.5:
                return Cmd()
            err = G.wrap(s['yaw'] - yaw)
            if abs(err) > math.radians(5.0):
                s['turning'] = True
            if s.get('turning'):
                if abs(err) > math.radians(c['turn_done_deg']):
                    return self._turn(err, 0.25, direction=1 if err > 0 else -1)
                s['turning'] = False
            self.phase = 'CROSS'
            s.update(d0=self._dist(di), prog_d=None, prog_t=t, tc=t)
            self._ev(t, bar='CROSS', x0=round(s['x0'], 2), depth=round(s['depth'], 2))
            return Cmd()
        # CROSS
        done = self._dist(di) - s['d0']
        need = s['x0'] + s['depth'] + c['bar_extra_m']
        if done >= need:
            self._finish_obstacle(t, {'bar': 'DONE', 'travelled': round(done, 2)}, 'bar_cool_until', 3.0)
            return Cmd()
        err = G.wrap(s['yaw'] - yaw)
        if abs(err) > math.radians(c['bar_hold_err_deg']):
            self._hold(t, 'bar_heading')
            return Cmd()
        if di.last_cmd.vx >= 0.05 and not self.odom_frozen:
            if s['prog_d'] is None or self._dist(di) - s['prog_d'] > 0.05:
                s['prog_d'], s['prog_t'] = self._dist(di), t
            elif t - s['prog_t'] > c['bar_no_progress_s']:
                self._hold(t, 'bar_no_progress')
                return Cmd()
        else:
            s['prog_d'], s['prog_t'] = None, t
        self.front_ignore_x = max(0.0, s['x0'] + s['depth'] + 0.15 - done)
        return Cmd(c['bar_vx'], 0.0)

    def _curtain(self, di):
        """ALIGN: turn to the curtain's bearing; PUSH: walk straight curtain_vx through it. The front stop
        ignores the cloth (curtain_ignore_m behind the face). Touch test: the face does not come closer
        although the dog walks for curtain_stall_s -> it is a wall: stop, ignore the camera for a while."""
        c, t = self.c, di.t
        s = self.curtain_s
        yaw = di.pose[2]
        if self.phase == 'ALIGN' and t - s['t0'] > 15.0 or self.phase == 'PUSH' and t > s['limit']:
            self._hold(t, 'curtain_timeout')
            return Cmd()
        if self.phase == 'ALIGN':
            if di.curtain.found:
                s['dist'] = di.curtain.dist
                s['yaw'] = G.wrap(yaw + di.curtain.bearing)
            err = G.wrap(s['yaw'] - yaw)
            if abs(err) > math.radians(c['turn_done_deg']) and t - s['tp'] < 12.0:
                return self._turn(err, 0.25, direction=1 if err > 0 else -1)
            self.phase = 'PUSH'
            # bounded push: 1.3 x the time the walk needs at curtain_vx + 3 s. A wall taken for a curtain
            # leaves lidar view at ~0.3 m, so the touch test cannot see the contact: this limit is the only stop
            s.update(d0=self._dist(di), tc=t, fd=collections.deque(),
                     limit=t + 1.3 * (s['dist'] + c['curtain_through_m']) / c['curtain_vx'] + 3.0)
            self._ev(t, curtain='PUSH', dist=round(s['dist'], 2))
            return Cmd()
        # PUSH
        done = self._dist(di) - s['d0']
        if done >= s['dist'] + c['curtain_through_m']:
            self._finish_obstacle(t, {'curtain': 'DONE', 'travelled': round(done, 2)}, 'curtain_cool_until', 5.0)
            return Cmd()
        err = G.wrap(s['yaw'] - yaw)
        if abs(err) > math.radians(c['curtain_hold_err_deg']):
            self._hold(t, 'curtain_heading')
            return Cmd()
        # touch test: judged only while the face is still seen. Inside curtain_min range the lidar
        # drops it (zero-range filter, near_drop_m), which looks like "gone", never like a stall.
        if di.front_dist < 2.0 and done < s['dist']:
            s['fd'].append((t, di.front_dist, di.last_cmd.vx))
        else:
            s['fd'].clear()
        while s['fd'] and t - s['fd'][0][0] > c['curtain_stall_s']:
            s['fd'].popleft()
        if (len(s['fd']) >= 5 and t - s['fd'][0][0] > 0.9 * c['curtain_stall_s']
                and all(v >= 0.08 for _, _, v in s['fd'])
                and s['fd'][0][1] - di.front_dist < c['curtain_stall_close_m']):
            self._finish_obstacle(t, {'curtain': 'WALL', 'front_m': round(di.front_dist, 2)},
                                  'curtain_cool_until', c['curtain_cooldown_s'])
            return Cmd()
        self.front_ignore_x = max(0.0, s['dist'] - done) + c['curtain_ignore_m']
        return Cmd(c['curtain_vx'], 0.0)

    def _out(self, cmd):
        return DriverOutput(cmd=cmd, mode=self.mode, phase=self.phase, codes=list(self.codes),
                            events=list(self.events), stairs_mode=self.stairs_mode,
                            ignore_front_x=self.front_ignore_x)

