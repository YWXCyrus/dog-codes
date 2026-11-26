"""Grid navigator (driver mode GRID): cell centre to cell centre over the arena map.

Each decision (10 Hz) it knows the arena pose (rc/localize.py), the map (rc/arena.py) and the plan
(rc/cellplan.py, refreshed every plan_period_s). Driving rules:
  * targets are CELL CENTRES; turns happen only near a centre (centre_tol_m), where the dog's turning
    circle (~0.39 m) fits inside the 1.02 m cell - this is what keeps the corners clear;
  * turn on the spot OR walk straight (walk_wz_max > 0 allows a small steering term while walking);
  * straight runs through already visited cells go on without stopping; the dog stops at every NEW
    cell centre for cell_pause_s (perception gets a still picture: scene query on a new cell);
  * before walking, the lane ahead (lane_w wide) must be free up to the target; a blocked border is
    marked (temporarily impassable + WALL evidence) and the plan is redone - no pushing into walls;
  * no progress while walking -> the border is blocked and the plan redone; three times -> HOLD.
Special borders on the path:
  CURTAIN  centre, square up, (camera question), PUSH at cur_push_vx with the guard's front stop relaxed
           for the cloth band only; touch test: < cur_stall_m in cur_stall_s = rigid -> back off
           cur_retreat_m (the only reverse), border = WALL, replan.
  stairs   centre, square up, then the driver's stairs sequence takes over (want_stairs).
  PROBE    (nothing else to explore, arena cells unvisited behind a wall): centre, face it, camera.
Survivor room with the QR not read: LOOK at its walls (square + skewed to the entrance half).
FINAL: the final cell's centre -> FINISH (HOLD 'finish', the dog stands).
"""
import collections
import math
import uuid
from dataclasses import dataclass, field

import numpy as np

from rc import arena as A
from rc import geometry as G
from rc import lanes as L
from rc.cellplan import CellPlanner, Plan
from rc.guard import Cmd


@dataclass
class NavIn:
    t: float
    apose: tuple                          # arena (x, y, yaw)
    pts: np.ndarray                       # body-frame blocking points (obstacles + drops + map)
    travelled: float = 0.0
    odom_ok: bool = True
    elapsed: float = 0.0                  # s since Enter
    last_vx: float = 0.0                  # last guarded forward command
    mission: dict = None                  # /quad/mission_status
    curtain_det: dict = None              # persistent camera curtain detection {'bearing', 'conf'}
    answers: list = field(default_factory=list)   # /quad/vlm_answer messages since the last step
    stairs_cue: dict = None               # stairs straight ahead: camera model or lidar detector


@dataclass
class NavOut:
    cmd: Cmd
    phase: str
    events: list = field(default_factory=list)
    hold: str = ''
    want_stairs: dict = None              # {'bearing': rad body-relative, 'edge': m to the stairs border}
    ask: dict = None                      # question for perception (/quad/vlm_ask)
    front_ignore: tuple = None            # body-x band the guard's front stop ignores (curtain push)
    reverse_ok: bool = False
    fallback: bool = False                # no plan: let the reactive driver wander for a while


class CellNav:

    def __init__(self, cfg, amap):
        self.c = cfg
        self.m = amap
        self.planner = CellPlanner(cfg, amap)
        self.plan = Plan()
        self.plan_t = -1e9
        self.goal_mode = 'explore'          # explore | final
        self.phase = 'PLAN'
        self.sub = {}                       # per-phase scratch
        self.turning = False
        self.prog = collections.deque()
        self.fails = collections.deque()
        self.paused = set()                 # cells where the arrival pause was done
        self.looked = set()                 # cells whose walls were looked at (QR)
        self.events = []
        self.finished = False
        self.last_cell = None
        self.came_from = {}                 # cell -> direction the dog came from
        self.stairs_crossings = 0
        self.stairs_asked = set()
        self.realigned = {}

    # ------------------------------------------------------------------ helpers
    def _ev(self, t, **kw):
        kw['t'] = round(t, 2)
        self.events.append(kw)

    def _set(self, phase, t, **kw):
        self.phase = phase
        self.sub = dict(kw, t0=t)
        self.turning = False

    def _turn_cmd(self, err):
        c = self.c
        rate = c['turn_wz'] * abs(err) / math.radians(c['turn_slow_deg'])
        rate = max(min(c['turn_wz_min'], c['turn_wz']), min(c['turn_wz'], rate))
        return Cmd(0.0, math.copysign(rate, err))

    def _turn_to(self, yaw_now, yaw_goal, tol_deg=None):
        """On-the-spot turn to an arena yaw. None when within tolerance."""
        tol = math.radians(self.c['turn_done_deg'] if tol_deg is None else tol_deg)
        err = G.wrap(yaw_goal - yaw_now)
        if abs(err) <= tol:
            return None
        return self._turn_cmd(err)

    def _blocked_ahead(self, ni, dist):
        """Free distance straight ahead (lane_w wide) if it is shorter than needed, else None."""
        need = min(dist + 0.10, self.c['lane_d'])
        fd = L.free_distance(ni.pts, 0.0, self.c['lane_w'], dmax=need + 0.5)
        return fd if fd < need else None

    def _drive_to(self, ni, px, py, tol=None, vx=None, strict=False):
        """Turn-or-walk towards a point. Returns (cmd or None when arrived, info). Not strict: a point
        beside or just behind the dog (< 0.30 m) counts as reached instead of turning round for it."""
        c = self.c
        x, y, yaw = ni.apose
        dx, dy = px - x, py - y
        dist = math.hypot(dx, dy)
        tol = c['arrive_tol_m'] if tol is None else tol
        if dist <= tol:
            self.turning = False
            return None, {'dist': dist}
        err = G.wrap(math.atan2(dy, dx) - yaw)
        along = dist * math.cos(err)
        if not strict and dist < 0.30 and along < 0.05:
            self.turning = False                    # beside / past it: do not circle around it
            return None, {'dist': dist}
        on = math.radians(c['head_err_turn_deg'] if dist > 0.40 else 25.0)
        off = math.radians(c['turn_done_deg'] if dist > 0.40 else 8.0)
        if self.turning and abs(err) <= off:
            self.turning = False
        elif not self.turning and abs(err) > on:
            self.turning = True
        if self.turning:
            return self._turn_cmd(err), {'dist': dist, 'err': err}
        v = c['cruise_vx'] if vx is None else vx
        if dist < 0.35:
            v = min(v, c['grid_vx_near'])
        wz = 0.0
        if c['walk_wz_max'] > 0:
            wz = max(-c['walk_wz_max'], min(c['walk_wz_max'], c['walk_wz_k'] * err))
        info = {'dist': dist, 'err': err}
        fd = self._blocked_ahead(ni, along)
        if fd is not None:
            info['blocked'] = fd                  # the caller decides: stop, or creep up to it
        return Cmd(v, wz), info

    def _progress_fail(self, ni):
        """Walking forward for grid_prog_s with < grid_prog_m done."""
        c, t = self.c, ni.t
        self.prog.append((t, ni.last_vx, ni.travelled))
        while self.prog and t - self.prog[0][0] > c['grid_prog_s']:
            self.prog.popleft()
        if not ni.odom_ok or t - self.prog[0][0] < c['grid_prog_s'] * 0.95:
            return False
        if any(p[1] < 0.08 for p in self.prog):
            return False
        return self.prog[-1][2] - self.prog[0][2] < c['grid_prog_m']

    def _fail(self, t, why):
        """Count a failure; True when there were too many recently (-> HOLD)."""
        self.fails.append(t)
        while self.fails and t - self.fails[0] > 40.0:
            self.fails.popleft()
        self._ev(t, grid_fail=why)
        return len(self.fails) >= self.c['grid_max_fails']

    def _replan(self, ni, cur, force=False):
        c, t = self.c, ni.t
        if not force and t - self.plan_t < c['plan_period_s'] and cur in self.plan.path:
            return
        x, y, yaw = ni.apose
        hd = A.yaw_dir(yaw)
        old = self.plan
        # time budget: go to the final cell while there is still time to get there (checked every 3 s and
        # only in the last 3 minutes: the extra search costs time on the dog's CPU)
        if (self.goal_mode == 'explore' and c['explore_budget_s'] > 0
                and ni.elapsed > c['explore_budget_s'] - 180.0 and t - getattr(self, '_budget_t', -1e9) > 3.0):
            self._budget_t = t
            fp = self.planner.plan(cur, hd, t, 'final')
            if fp.goal is not None:
                eta = self.planner.eta_s(fp.path, hd)
                if ni.elapsed + eta + c['budget_margin_s'] >= c['explore_budget_s']:
                    self.goal_mode = 'final'
                    self._ev(t, grid='time budget: going to the final cell (eta %.0f s)' % eta)
        p = self.planner.plan(cur, hd, t, self.goal_mode)
        if self.goal_mode == 'explore' and p.reason == 'none':
            fp = self.planner.plan(cur, hd, t, 'final')
            if fp.goal is not None:
                self.goal_mode = 'final'
                self._ev(t, grid='exploration done (%d cells visited): going to the final cell %s'
                         % (len(self.m.order), fp.goal))
                p = fp
        # commitment: keep the old goal unless the new one is clearly better
        if (old.goal is not None and p.goal is not None and p.goal != old.goal and old.reason == p.reason
                and not self.m.visited(old.goal) and old.reason == 'explore'):
            keep = self.planner.plan_to(cur, hd, t, old.goal)
            if keep is not None and keep.cost <= p.cost + c['plan_commit_margin']:
                keep.targets = p.targets
                p = keep
        if p.goal != old.goal or p.reason != old.reason:
            self._ev(t, plan=p.reason, goal=None if p.goal is None else list(p.goal),
                     path=[list(q) for q in p.path], cost=None if p.cost == math.inf else round(p.cost, 1))
        self.plan = p
        self.plan_t = t

    # ------------------------------------------------------------------ main
    def step(self, ni):
        self.events = []
        out = self._step(ni)
        out.events = self.events + out.events
        out.phase = self.phase_str()
        return out

    def phase_str(self):
        p = self.phase
        if self.plan.goal is not None and p in ('GO', 'PLAN'):
            return '%s %s->%s' % (p, self.goal_mode[:3], tuple(self.plan.goal))
        return p

    def _step(self, ni):
        c, m, t = self.c, self.m, ni.t
        x, y, yaw = ni.apose
        cur = m.cell_of(x, y)
        if cur != self.last_cell:
            if self.last_cell is not None and A.dir_between(cur, self.last_cell):
                self.came_from[cur] = A.dir_between(cur, self.last_cell)
            self.last_cell = cur
        if self.finished:
            return NavOut(Cmd(), 'FINISH')

        if self.phase == 'PAUSE':
            if t - self.sub['t0'] < c['cell_pause_s']:
                return NavOut(Cmd(), '')
            looks = self._qr_looks(ni, cur)
            if looks:
                self._set('LOOK', t, yaws=looks, k=0, since=None)
                self._ev(t, grid='LOOK at the walls of %s for the QR' % (cur,))
            else:
                self._set('PLAN', t)
        if self.phase == 'LOOK':
            return self._look(ni)
        if self.phase == 'BACK':
            return self._back(ni)
        if self.phase == 'STAIRS_ASK':
            return self._stairs_ask(ni)
        if self.phase == 'REALIGN':
            return self._realign_step(ni)
        if self.phase in ('CURTAIN_ALIGN', 'CURTAIN_ASK', 'CURTAIN_PUSH', 'CURTAIN_BACK', 'PROBE_ALIGN',
                          'PROBE_ASK'):
            return self._curtain(ni, cur)
        if self.phase == 'STAIRS_WAIT':
            return self._stairs_wait(ni, cur)

        self._replan(ni, cur, force=self.phase == 'PLAN')
        self.phase = 'GO'
        p = self.plan
        if p.goal is None:
            self.phase = 'PLAN'
            return NavOut(Cmd(), '', fallback=True)
        if cur not in p.path:
            self._replan(ni, cur, force=True)
            p = self.plan
            if cur not in p.path:
                return NavOut(Cmd(), '', fallback=True)
        rest = p.path[p.path.index(cur):]
        cx, cy = m.centre(cur)

        # ---- a cell entered for the first time: its centre, then a still pause (perception)
        if c['cell_pause_s'] > 0 and cur not in self.paused and not m.is_stairs(cur):
            cmd, info = self._drive_to(ni, cx, cy, vx=c['grid_vx_near'])
            if cmd is not None:
                return self._walk_out(ni, cmd, info, cur, None)
            self.paused.add(cur)
            self._set('PAUSE', t)
            return NavOut(Cmd(), '')

        # ---- at the goal cell: centre, then the goal action
        if len(rest) == 1:
            cmd, info = self._drive_to(ni, cx, cy, vx=c['grid_vx_near'])
            if cmd is not None:
                return self._walk_out(ni, cmd, info, cur, None)
            return self._arrived(ni, cur)

        nxt = rest[1]
        d = A.dir_between(cur, nxt)
        if d is None:
            self._replan(ni, cur, force=True)
            return NavOut(Cmd(), '')
        e = m.edges.get(A.edge_key(cur, nxt))
        special = None
        if m.is_stairs(nxt) and not m.is_stairs(cur):
            special = 'stairs'
        elif e is not None and (e.state == A.CURTAIN or e.was_curtain):
            special = 'curtain'                   # every crossing of a curtain (also back through it) is a push
        yaw_d = A.DIR_YAW[d]
        turn_needed = abs(G.wrap(yaw_d - yaw)) > math.radians(30.0)
        off = math.hypot(x - cx, y - cy)
        if special is not None or (turn_needed and off > c['centre_tol_m']):
            # corners / special borders: get to the centre of this cell first, then square up
            if off > c['centre_tol_m']:
                cmd, info = self._drive_to(ni, cx, cy, tol=c['centre_tol_m'] * 0.7, vx=c['grid_vx_near'],
                                           strict=True)
                if cmd is not None:
                    return self._walk_out(ni, cmd, info, cur, None)
            if special is not None:
                cmd = self._turn_to(yaw, yaw_d)
                if cmd is not None:
                    return NavOut(cmd, '')
                if special == 'stairs':
                    self._set('STAIRS_WAIT', t, nxt=nxt, d=d)
                    self._ev(t, grid='at the stairs: %s -> %s' % (cur, nxt))
                    return self._stairs_wait(ni, cur)
                self._set('CURTAIN_ALIGN', t, nxt=nxt, d=d, cur=cur, k=A.edge_key(cur, nxt), probe=False)
                self._ev(t, grid='curtain border %s -> %s' % (cur, nxt))
                return self._curtain(ni, cur)
        # straight run: up to grid_run_cells along d through visited, ordinary cells
        j = 1
        while (j + 1 < len(rest) and j < c['grid_run_cells'] and m.visited(rest[j])
               and A.dir_between(rest[j], rest[j + 1]) == d and not m.is_stairs(rest[j + 1])
               and m.state(rest[j], rest[j + 1]) != A.CURTAIN):
            j += 1
        ex_, ey_ = m.centre(rest[j])
        # aim at a point on the cell centre line grid_lookahead_m ahead (never past the run's end):
        # a sideways offset becomes a heading error and is turned out (keeps the dog in the middle)
        ux, uy = math.cos(yaw_d), math.sin(yaw_d)
        s_dog = (x - cx) * ux + (y - cy) * uy
        s_end = (ex_ - cx) * ux + (ey_ - cy) * uy
        s_aim = min(s_end, s_dog + c['grid_lookahead_m'])
        ax, ay = cx + s_aim * ux, cy + s_aim * uy
        cmd, info = self._drive_to(ni, ax, ay)
        if cmd is None and s_end - s_dog > 0.15:
            cmd, info = self._drive_to(ni, ex_, ey_)
        if cmd is None:
            return NavOut(Cmd(), '')
        return self._walk_out(ni, cmd, info, cur, nxt)

    def _walk_out(self, ni, cmd, info, cur, nxt):
        """Blocked lane / no progress handling for a walking command."""
        c, m, t = self.c, self.m, ni.t
        if 'blocked' in info and nxt is not None and not m.visited(nxt) and self._stairs_check(ni, cur, nxt):
            return NavOut(Cmd(), '', ask=self.sub.get('ask_q'))
        if 'blocked' in info:
            fd = info['blocked']
            x, y, yaw = ni.apose
            dist_b = None
            if nxt is not None:
                bx = (m.centre(cur)[0] + m.centre(nxt)[0]) / 2.0
                by = (m.centre(cur)[1] + m.centre(nxt)[1]) / 2.0
                dist_b = (bx - x) * math.cos(yaw) + (by - y) * math.sin(yaw)
            beyond = dist_b is None or fd > dist_b + 0.15
            if beyond and fd > c['grid_obst_stop_m']:
                # an obstacle INSIDE a cell (a prop, a cabinet), not on the border: creep up to it
                cmd = Cmd(min(cmd.vx, c['grid_vx_near']), 0.0)
            elif not beyond and self._realign(ni, cur, nxt):
                return NavOut(Cmd(), '')          # first: square up exactly and look again
            elif not beyond:
                # the border itself is blocked: impassable for now, and evidence that it is a wall
                e = m.edge_between(cur, nxt)
                if not e.traversed:
                    e.mh += c['grid_block_evidence']
                    m._classify(e)
                self._block(ni, cur, nxt, 'lane blocked %.2f m ahead' % fd)
                return NavOut(Cmd(), '')
            else:
                # stopped in front of an obstacle inside a cell: close enough counts as a visit
                for cl in ([nxt] if nxt is not None else []) + [cur]:
                    d = math.hypot(x - m.centre(cl)[0], y - m.centre(cl)[1])
                    if not m.visited(cl) and d <= c['visit_blocked_r_m']:
                        m.force_visit(cl, t)
                        self.paused.add(cl)
                        self._ev(t, grid='%s visited from %.2f m (obstacle in the cell)' % (cl, d))
                if nxt is not None:
                    m.cell(nxt).obstructed += 1           # twice: not planned through any more
                    self._block(ni, cur, nxt, 'obstacle in %s' % (nxt,))
                else:
                    self.paused.add(cur)
                    self._set('PLAN', t)
                    self.plan_t = -1e9
                self._start_back(ni, cur)
                return NavOut(Cmd(), '')
        if cmd.vx > 0 and self._progress_fail(ni):
            self.prog.clear()
            if nxt is not None:
                self._block(ni, cur, nxt, 'no progress')
            if self._fail(t, 'no progress'):
                return NavOut(Cmd(), '', hold='no_progress')
            return NavOut(Cmd(), '')
        return NavOut(cmd, '')

    def _realign(self, ni, cur, nxt):
        """A border looked blocked while the dog walked a few degrees off the cell axis (its lane then clips
        the next cell's side panel; sim 2026-11-26 after the stairs): square up to the axis exactly and stand
        still grid_realign_s (the snap runs) before believing it. Once per border per 20 s."""
        k = A.edge_key(cur, nxt)
        last = self.realigned.get(k, -1e9)
        if ni.t - last < 20.0:
            return False
        self.realigned[k] = ni.t
        self._set('REALIGN', ni.t, yaw=A.DIR_YAW[A.dir_between(cur, nxt)], still=None)
        self._ev(ni.t, grid='lane %s -> %s blocked: squaring up to look again' % (cur, nxt))
        return True

    def _realign_step(self, ni):
        s, t = self.sub, ni.t
        cmd = self._turn_to(ni.apose[2], s['yaw'], tol_deg=1.5)
        if cmd is not None and t - s['t0'] < 8.0:
            return NavOut(cmd, '')
        if s['still'] is None:
            s['still'] = t
        if t - s['still'] < self.c['grid_realign_s']:
            return NavOut(Cmd(), '')
        self._set('GO', t)
        return NavOut(Cmd(), '')

    def _stairs_check(self, ni, cur, nxt):
        """A lane into an unvisited cell is blocked. Stairs? The camera model / the lidar stairs detector
        looking that way -> mark the cell; else, if the lidar sees through the border above the obstacle
        (something low: a first step), ask perception's camera once. True = handled (stop this tick)."""
        c, m, t = self.c, self.m, ni.t
        if c['stairs'] != 'climb' or m.is_stairs(nxt):
            return False
        if ni.stairs_cue is not None:
            if m.mark_stairs(nxt, ni.stairs_cue['src']):
                self._ev(t, grid='stairs in %s (%s)' % (nxt, ni.stairs_cue['src']), stairs_cell=list(nxt))
                self._set('PLAN', t)
                self.plan_t = -1e9
                return True
            return False
        e = m.edges.get(A.edge_key(cur, nxt))
        k = A.edge_key(cur, nxt)
        if (not c['stairs_ask_camera'] or e is None or k in self.stairs_asked
                or not (e.state == A.OPEN or e.mp > e.mh)):
            return False
        self.stairs_asked.add(k)
        self._set('STAIRS_ASK', t, nxt=nxt, cur=cur)
        self.sub['ask_q'] = self._ask(ni, 'stairs')
        return True

    def _stairs_ask(self, ni):
        c, s, t = self.c, self.sub, ni.t
        if ni.stairs_cue is not None:
            ans = {'answer': True}
        else:
            ans = self._answer(ni)
            if ans is None and t - s.get('ask_t', s['t0']) < c['cur_vlm_wait_s']:
                return NavOut(Cmd(), '')
        yes = bool(ans and ans.get('answer')) and not (ans or {}).get('failed')
        self._ev(t, stairs_answer='yes' if yes else 'no', cell=list(s['nxt']))
        if yes:
            self.m.mark_stairs(s['nxt'], 'camera')
        self._set('PLAN', t)
        self.plan_t = -1e9
        return NavOut(Cmd(), '')

    def _start_back(self, ni, cur):
        """Off a cell centre in front of an obstacle: back straight to the centre of the cell the dog came
        from (over the ground it just walked) - turning there would sweep the body into the obstacle /
        the door posts. Only when that centre is straight behind; grid_back_max_m at most."""
        c, m = self.c, self.m
        if c['grid_back_max_m'] <= 0:
            return
        x, y, yaw = ni.apose
        came = self.came_from.get(cur)
        tgt = A.step(cur, came) if came else cur
        tx, ty = m.centre(tgt)
        if math.hypot(tx - x, ty - y) <= c['centre_tol_m']:
            tgt = cur
            tx, ty = m.centre(cur)
        d = math.hypot(tx - x, ty - y)
        if d <= c['centre_tol_m']:
            return
        back = G.wrap(math.atan2(ty - y, tx - x) - (yaw + math.pi))
        if abs(back) > math.radians(25.0):
            return
        self._set('BACK', ni.t, tx=tx, ty=ty, d0=ni.travelled)
        self._ev(ni.t, grid='backing %.2f m to %s' % (d, tgt))

    def _back(self, ni):
        c, s = self.c, self.sub
        x, y, yaw = ni.apose
        along = (s['tx'] - x) * math.cos(yaw + math.pi) + (s['ty'] - y) * math.sin(yaw + math.pi)
        if along <= 0.05 or ni.travelled - s['d0'] >= c['grid_back_max_m'] or ni.t - s['t0'] > 10.0:
            self._set('PLAN', ni.t)
            self.plan_t = -1e9
            return NavOut(Cmd(), '')
        return NavOut(Cmd(-c['cur_retreat_vx'], 0.0), '', reverse_ok=True)

    def _block(self, ni, a, b, why):
        self.m.block(a, b, ni.t, self.c['grid_block_s'])
        self._ev(ni.t, grid_block='%s -> %s: %s' % (a, b, why))
        self._set('PLAN', ni.t)
        self.plan_t = -1e9

    def _arrived(self, ni, cur):
        c, m, t = self.c, self.m, ni.t
        p = self.plan
        if p.reason == 'final':
            self.finished = True
            self._ev(t, grid='FINISH in the final cell %s' % (cur,), finish=list(cur))
            return NavOut(Cmd(), '', hold='finish')
        if p.reason == 'probe' and p.probe_edge is not None:
            a, b = A.edge_cells(p.probe_edge)
            nxt = b if a == cur else a
            self._set('PROBE_ALIGN', t, nxt=nxt, d=A.dir_between(cur, nxt), cur=cur, k=p.probe_edge, probe=True)
            self._ev(t, grid='probe: is the wall %s -> %s a curtain?' % (cur, nxt))
            return self._curtain(ni, cur)
        self._set('PLAN', t)
        self.plan_t = -1e9
        return NavOut(Cmd(), '')

    # ------------------------------------------------------------------ QR: look at the walls
    def _qr_looks(self, ni, cur):
        c, m = self.c, self.m
        if not c['qr_look'] or cur in self.looked:
            return []
        ms = ni.mission or {}
        qr = ms.get('qr') or {}
        if qr.get('read'):
            return []
        tasks = ms.get('tasks') or {}
        if 'M04' in tasks and tasks['M04'].get('reported'):
            return []
        surv = [tuple(s) for s in ms.get('survivor_cells') or []]
        here = cur in surv or (c['qr_look_on_seen'] and qr.get('seen_not_read'))
        if not here:
            return []
        self.looked.add(cur)
        came = self.came_from.get(cur)
        yaws = []
        for d in ('N', 'W', 'S', 'E'):
            if m.state(cur, A.step(cur, d)) != A.WALL or d == came:
                continue
            base = A.DIR_YAW[d]
            if came is not None and A.DIR_YAW[came] is not None:
                # nearer half = the half towards the entrance
                s = 1.0 if G.wrap(A.DIR_YAW[came] - base) > 0 else -1.0
                yaws += [G.wrap(base + s * math.radians(c['qr_look_skew_deg'])), base]
            else:
                yaws += [base]
        return yaws

    def _look(self, ni):
        c, t = self.c, ni.t
        s = self.sub
        if s['k'] >= len(s['yaws']):
            self._set('PLAN', t)
            return NavOut(Cmd(), '')
        cmd = self._turn_to(ni.apose[2], s['yaws'][s['k']], tol_deg=5.0)
        if cmd is not None:
            s['since'] = None
            return NavOut(cmd, '')
        if s['since'] is None:
            s['since'] = t
        if t - s['since'] >= c['qr_look_s'] or ((ni.mission or {}).get('qr') or {}).get('read'):
            s['k'] = s['k'] + 1 if not ((ni.mission or {}).get('qr') or {}).get('read') else len(s['yaws'])
            s['since'] = None
        return NavOut(Cmd(), '')

    # ------------------------------------------------------------------ stairs
    def _stairs_wait(self, ni, cur):
        """Squared up at the centre before the stairs: ask the driver for the stairs sequence."""
        s = self.sub
        x, y, yaw = ni.apose
        cx, cy = self.m.centre(cur)
        d = s['d']
        bx = cx + 0.5 * self.m.L * math.cos(A.DIR_YAW[d])
        by = cy + 0.5 * self.m.L * math.sin(A.DIR_YAW[d])
        edge = (bx - x) * math.cos(yaw) + (by - y) * math.sin(yaw)
        return NavOut(Cmd(), '', want_stairs={'bearing': G.wrap(A.DIR_YAW[d] - yaw), 'edge': max(0.0, edge),
                                              't0': s['t0'], 'cell': list(s['nxt'])})

    def on_stairs_started(self, t):
        self._ev(t, grid='stairs sequence started')

    def on_stairs_done(self, t, cells):
        """The driver's stairs sequence finished (exit AI done). cells: cells passed while on the stairs."""
        for cl in cells:
            if cl in self.paused and not self.m.is_stairs(cl):
                continue                              # an ordinary cell walked before (pose error at the landing)
            self.m.mark_stairs(cl, 'crossed')
            self.m.force_visit(cl, t)
            self.paused.add(cl)
        self.stairs_crossings += 1
        self._set('PLAN', t)
        self.plan_t = -1e9
        self._ev(t, grid='stairs crossed (%d cells on the stairs)' % len(cells))

    def on_stairs_aborted(self, t, nxt=None):
        if nxt is not None:
            self.m.block(self.last_cell, tuple(nxt), t, 60.0)
        self._set('PLAN', t)
        self.plan_t = -1e9

    # ------------------------------------------------------------------ curtain / probe
    def _ask(self, ni, kind):
        qid = 'nav-%s-%s' % (kind, uuid.uuid4().hex[:6])
        self.sub['ask_id'] = qid
        self.sub['ask_t'] = ni.t
        self._ev(ni.t, ask=kind, id=qid)
        return {'id': qid, 'kind': kind}

    def _answer(self, ni):
        qid = self.sub.get('ask_id')
        for a in ni.answers:
            if qid is not None and a.get('id') == qid:
                return a
        return None

    def _curtain(self, ni, cur):
        c, m, t = self.c, self.m, ni.t
        s = self.sub
        x, y, yaw = ni.apose
        k = s['k']
        e = m.edge(k)
        yaw_d = A.DIR_YAW[s['d']]
        a0 = s['cur']
        bx = (m.centre(a0)[0] + m.centre(s['nxt'])[0]) / 2.0
        by = (m.centre(a0)[1] + m.centre(s['nxt'])[1]) / 2.0
        past = (x - bx) * math.cos(yaw_d) + (y - by) * math.sin(yaw_d)        # + = beyond the border

        if self.phase in ('CURTAIN_ALIGN', 'PROBE_ALIGN'):
            cmd = self._turn_to(yaw, yaw_d)
            if cmd is not None:
                if t - s['t0'] > 20.0:
                    self._block(ni, a0, s['nxt'], 'could not square up to the curtain')
                    return NavOut(Cmd(), '')
                return NavOut(cmd, '')
            cam = ni.curtain_det is not None and abs(ni.curtain_det.get('bearing', 9.0)) < math.radians(20.0)
            if cam:
                e.result = 'camera_yes'
                m._classify(e)
            need_cam = s['probe'] or (c['cur_ask_camera'] and e.result not in ('camera_yes', 'passed')
                                      and not e.traversed and e.cue < c['cur_cue_min'])
            if need_cam:
                self._set('PROBE_ASK' if s['probe'] else 'CURTAIN_ASK', t, **{kk: s[kk] for kk in
                                                                          ('nxt', 'd', 'cur', 'k', 'probe')})
                return NavOut(Cmd(), '', ask=self._ask(ni, 'curtain'))
            return self._start_push(ni)

        if self.phase in ('CURTAIN_ASK', 'PROBE_ASK'):
            if ni.curtain_det is not None and abs(ni.curtain_det.get('bearing', 9.0)) < math.radians(20.0):
                ans = {'answer': True, 'src': 'camera_model'}
            else:
                ans = self._answer(ni)
            waited = t - s.get('ask_t', s['t0'])
            if ans is None and waited < c['cur_vlm_wait_s']:
                return NavOut(Cmd(), '')
            failed = ans is None or ans.get('failed')
            yes = bool(ans and ans.get('answer')) and not failed
            self._ev(t, curtain_answer='yes' if yes else ('none' if failed else 'no'), edge=list(k))
            if yes:
                e.result = 'camera_yes'
                m._classify(e)
                return self._start_push(ni)
            if failed and ((s['probe'] and c['cur_probe_touch']) or (not s['probe'] and e.state == A.CURTAIN)):
                return self._start_push(ni)          # no camera: the touch test decides
            e.result = 'camera_no'
            m._classify(e)
            if e.state == A.CURTAIN:
                e.state = A.WALL
            self._set('PLAN', t)
            self.plan_t = -1e9
            return NavOut(Cmd(), '')

        if self.phase == 'CURTAIN_PUSH':
            # cloth band in the body frame: the border is `-past` ahead of base_link
            ahead = -past
            band = (ahead - 0.25, ahead + c['cur_ignore_m'])
            if past >= c['cur_clear_m']:
                e.result = 'passed'
                e.was_curtain = True
                m._classify(e)
                self._ev(t, curtain='passed', edge=list(k))
                self._set('PLAN', t)
                self.plan_t = -1e9
                return NavOut(Cmd(), '')
            if t - s['t0'] > c['cur_max_s']:
                self._ev(t, curtain='timeout', edge=list(k))
                return NavOut(Cmd(), '', hold='curtain_timeout')
            err = G.wrap(yaw_d - yaw)
            if abs(err) > math.radians(c['curtain_hold_err_deg']):
                self._ev(t, curtain='heading drift %.0f deg' % math.degrees(err), edge=list(k))
                return self._rigid(ni, 'heading drift')
            # touch test once the nose can be on the cloth
            s['hist'].append((t, ni.travelled, ni.last_vx))
            while s['hist'] and t - s['hist'][0][0] > c['cur_stall_s']:
                s['hist'].popleft()
            touching = past > -(c['cur_nose_m'] + 0.10)
            if (touching and t - s['t0'] > c['cur_stall_s'] and len(s['hist']) > 3
                    and all(h[2] > 0.05 for h in s['hist']) and t - s['hist'][0][0] >= c['cur_stall_s'] * 0.95
                    and s['hist'][-1][1] - s['hist'][0][1] < c['cur_stall_m']):
                return self._rigid(ni, 'touch test: rigid')
            return NavOut(Cmd(c['cur_push_vx'], 0.0), '', front_ignore=band)

        if self.phase == 'CURTAIN_BACK':
            if ni.travelled - s['d0'] >= c['cur_retreat_m'] or t - s['t0'] > 6.0:
                self._set('PLAN', t)
                self.plan_t = -1e9
                return NavOut(Cmd(), '')
            return NavOut(Cmd(-c['cur_retreat_vx'], 0.0), '', reverse_ok=True)
        return NavOut(Cmd(), '')

    def _start_push(self, ni):
        s = self.sub
        self._set('CURTAIN_PUSH', ni.t, **{kk: s[kk] for kk in ('nxt', 'd', 'cur', 'k', 'probe')})
        self.sub['hist'] = collections.deque()
        self._ev(ni.t, curtain='push', edge=list(self.sub['k']))
        return NavOut(Cmd(), '')

    def _rigid(self, ni, why):
        c, m = self.c, self.m
        s = self.sub
        e = m.edge(s['k'])
        e.result = 'rigid'
        m._classify(e)
        self._ev(ni.t, curtain='rigid - a wall (%s)' % why, edge=list(s['k']))
        if c['cur_retreat_m'] > 0:
            self._set('CURTAIN_BACK', ni.t, **{kk: s[kk] for kk in ('nxt', 'd', 'cur', 'k', 'probe')})
            self.sub['d0'] = ni.travelled
        else:
            self._set('PLAN', ni.t)
            self.plan_t = -1e9
        return NavOut(Cmd(), '')

    # ------------------------------------------------------------------ viewer
    def snapshot(self):
        p = self.plan
        return {'phase': self.phase_str(), 'mode': self.goal_mode, 'goal': list(p.goal) if p.goal else None,
                'reason': p.reason, 'path': [list(q) for q in p.path], 'targets': p.targets,
                'finished': self.finished, 'probe': list(p.probe_edge) if p.probe_edge else None}
