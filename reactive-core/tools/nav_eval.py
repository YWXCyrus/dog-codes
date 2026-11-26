"""Navigation acceptance run (2026-11-25): course-shaped worlds x noise seeds, with odometry drift,
checked against the agreed criteria. Pure Python (the simulator), runs jobs in parallel.

  python tools/nav_eval.py                     # nav_mode=cells, full matrix
  python tools/nav_eval.py --nav wall          # the old navigation on the same matrix
  python tools/nav_eval.py --only W1,W6:3      # some jobs
  python tools/nav_eval.py --seeds 0 --json out.json

Criteria (per run):
  C1 no collision            C2 never reverses
  C3 no silent stall: motion commanded (before the guard), dog not moving > 3 s, not in HOLD
  C4 no on-the-spot turn while the dog is tilted > 10 deg (true tilt, outside STAIRS)
  C5 no more than 360 deg of turning without 0.3 m of progress (outside HOLD)
  C6 corridors (W1-W5): goal within 60 s of the start; mazes (W6, W7): >= 90% of the cells within
     300 s; W8: the existing arena / stairs test conditions
  C7 = all of the above on noise seeds 0, 1, 2 with odometry drift (scale 1.05, gyro bias 0.001 rad/s)
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DRIFT = (1.05, 0.001)


def world_for(job):
    from sim import course as C
    from sim import world as W
    name = job.split(':')[0]
    seed = int(job.split(':')[1]) if ':' in job else 0
    if name == 'W1':
        return C.l_corner('left')
    if name == 'W2':
        return C.l_corner('right')
    if name == 'W3':
        return C.t_junction()
    if name == 'W4':
        return C.dead_end()
    if name == 'W5':
        return C.start_bay()
    if name == 'W5b':
        return C.slab_wall()
    if name == 'W6':
        return C.maze(seed)
    if name == 'W7':
        return C.messy(seed)
    if name == 'W8a':
        return C.CourseWorld(W.arena_default(), 'W8a_arena', kind='regression', T=150.0)
    if name == 'W8s':
        return C.CourseWorld(W.stairs_world(), 'W8s_stairs', kind='regression', T=100.0)
    raise ValueError(job)


class Probe:
    def __init__(self):
        self.t0 = None
        self.path = []
        self.goal_t = None
        self.stall = []               # (t_from, t_to)
        self.tilt_turn = []           # times
        self.spin_max = 0.0
        self._st = None               # stall anchor (t, x, y, yaw)
        self._sp = None               # spin anchor (x, y)
        self._spin = 0.0
        self._last_yaw = None
        self.goal = None
        import collections
        self._win = collections.deque()
        self._yaw_acc = 0.0
        self._n = 0

    def __call__(self, t, dog, core, cmd):
        if self.t0 is None:
            self.t0 = t
        tr = t - self.t0
        self.path.append((dog.x, dog.y, tr))
        if self.goal is not None and self.goal_t is None and self.goal(dog.x, dog.y):
            self.goal_t = tr
        mode = core.driver.mode
        drv = core.drive_cmd if mode != 'HOLD' else None
        moving_cmd = drv is not None and (drv.vx > 0.05 or abs(drv.wz) > 0.1)
        # C3: motion commanded (before the guard) but the dog neither moved 5 cm nor turned 3 deg
        if not moving_cmd:
            self._st = None
        elif self._st is None or math.hypot(dog.x - self._st[1], dog.y - self._st[2]) > 0.05 \
                or abs(math.remainder(dog.yaw - self._st[3], 2 * math.pi)) > math.radians(3.0):
            self._st = (tr, dog.x, dog.y, dog.yaw)
        elif tr - self._st[0] > 3.0:
            if not self.stall or self.stall[-1][1] < self._st[0]:
                self.stall.append([round(self._st[0], 1), round(tr, 1)])
            else:
                self.stall[-1][1] = round(tr, 1)
        # C3 (livelock, added in iteration 2): 10 s outside HOLD / STAIRS with < 0.3 m of movement
        # and < 30 deg of turning - e.g. a decide / go / stop loop that never commands motion for 3 s
        if self._last_yaw is not None:
            self._yaw_acc += abs(math.remainder(dog.yaw - self._last_yaw, 2 * math.pi))
        self._win.append((tr, dog.x, dog.y, self._yaw_acc, mode not in ('HOLD', 'STAIRS')))
        while self._win and tr - self._win[0][0] > 10.0:
            self._win.popleft()
        if self._n % 10 == 0 and tr - self._win[0][0] >= 9.9 and all(w[4] for w in self._win):
            x0, y0 = self._win[0][1], self._win[0][2]
            if max(math.hypot(w[1] - x0, w[2] - y0) for w in self._win) < 0.3 and \
                    self._yaw_acc - self._win[0][3] < math.radians(30.0):
                if not self.stall or self.stall[-1][1] < self._win[0][0]:
                    self.stall.append([round(self._win[0][0], 1), round(tr, 1), 'livelock'])
                else:
                    self.stall[-1][1] = round(tr, 1)
        self._n += 1
        # C4 (true tilt)
        tilt = max(abs(math.degrees(dog.pitch)), abs(math.degrees(dog.roll)))
        if mode != 'STAIRS' and tilt > 10.0 and abs(dog.wz) > 0.1 and abs(dog.vx) < 0.05:
            self.tilt_turn.append(round(tr, 2))
        # C5
        if self._last_yaw is not None:
            dyaw = abs(math.remainder(dog.yaw - self._last_yaw, 2 * math.pi))
            if self._sp is None or math.hypot(dog.x - self._sp[0], dog.y - self._sp[1]) > 0.3:
                self._sp, self._spin = (dog.x, dog.y), 0.0
            if mode != 'HOLD':
                self._spin += dyaw
                self.spin_max = max(self.spin_max, self._spin)
        self._last_yaw = dog.yaw


def run_job(args):
    job, seed, nav, drift = args
    from rc import config
    from sim import course as C
    from sim.harness import simulate
    cw = world_for(job)
    cfg = config.load(overrides=['nav_mode=%s' % nav])
    pr = Probe()
    pr.goal = cw.goal
    dk = dict(cw.dog_kwargs)
    if drift:
        dk['odom_drift'] = DRIFT
    t_wall = time.time()
    r = simulate(cw.world, cfg, T=cw.T, start=cw.start, seed=seed, dog_kwargs=dk, probe=pr)
    out = {'job': job, 'world': cw.name, 'kind': cw.kind, 'seed': seed, 'nav': nav, 'drift': drift,
           'wall_s': round(time.time() - t_wall, 1), 'startup': r.startup,
           'collisions': r.collisions, 'reversed': r.reversed, 'final': r.final_mode, 'hold': r.hold_reason,
           'stalls': pr.stall, 'tilt_turn': pr.tilt_turn[:5], 'n_tilt_turn': len(pr.tilt_turn),
           'spin_max_deg': round(math.degrees(pr.spin_max), 1), 'end': [round(v, 2) for v in r.end_pose[:2]]}
    crit = {'C1': r.collisions == 0, 'C2': not r.reversed, 'C3': not pr.stall,
            'C4': not pr.tilt_turn, 'C5': pr.spin_max <= 2 * math.pi + 1e-6}
    if cw.kind == 'corridor':
        out['goal_t'] = pr.goal_t
        crit['C6'] = pr.goal_t is not None and pr.goal_t <= 60.0
    elif cw.kind == 'maze':
        pts = [(x, y) for x, y, tr in pr.path if tr <= 300.0]
        got = C.maze_coverage(cw, pts)
        need = math.ceil(0.9 * cw.cells)
        out['cov'] = '%d/%d' % (len(got), cw.cells)
        t90 = None
        if len(got) >= need:
            seen = set()
            for x, y, tr in pr.path:
                seen |= C.maze_coverage(cw, [(x, y)])
                if len(seen) >= need:
                    t90 = round(tr, 1)
                    break
        out['t90'] = t90
        crit['C6'] = len(got) >= need
    elif cw.kind == 'regression':
        if cw.name.startswith('W8a'):
            far = max(math.hypot(x, y) for x, y, _ in pr.path)
            crit['C6'] = r.cells >= 8 and far > 2.0 and r.startup == 'ready'
            out['cells'], out['far'] = r.cells, round(far, 2)
        else:
            seq = [n for _, n in r.codes]
            crit['C6'] = (seq[:3] == ['manual', 'enter_ai', 'ai_basic']
                          and seq[3:8] == ['exit_ai', 'move_mode', 'rug_general', 'rug_general', 'auto']
                          and r.max_support >= 0.44 and r.end_pose[0] > 6.0
                          and any(e.get('stairs') == 'DONE' for e in r.events))
            out['codes'] = seq[:8]
    out['crit'] = crit
    out['pass'] = all(crit.values())
    out['events'] = [e for e in r.events if 'cell' not in e][-12:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nav', default='cells')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--only', default='')
    ap.add_argument('--no-drift', action='store_true')
    ap.add_argument('--procs', type=int, default=18)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    jobs = (['W1', 'W2', 'W3', 'W4', 'W5', 'W5b'] + ['W6:%d' % s for s in range(10)]
            + ['W7:%d' % s for s in range(3)] + ['W8a', 'W8s'])
    if a.only:
        jobs = a.only.split(',')
    seeds = [int(s) for s in a.seeds.split(',')]
    work = [(j, s, a.nav, not a.no_drift) for j in jobs for s in seeds]
    work.sort(key=lambda w: 0 if w[0].startswith(('W6', 'W7')) else 1)      # long ones first
    t0 = time.time()
    with mp.Pool(min(a.procs, len(work))) as pool:
        res = pool.map(run_job, work, chunksize=1)
    res.sort(key=lambda r: (jobs.index(r['job']), r['seed']))
    for r in res:
        extra = ''
        if 'goal_t' in r:
            extra = 'goal %s s' % r['goal_t']
        if 'cov' in r:
            extra = 'cov %s t90 %s' % (r['cov'], r['t90'])
        if 'cells' in r:
            extra = 'cells %s far %s' % (r['cells'], r['far'])
        if 'codes' in r:
            extra = 'codes %s' % r['codes']
        fails = [k for k, v in r['crit'].items() if not v]
        print('%-14s s%d %-4s %s  %-28s final %-8s %-10s spin %5.0f  %s' % (
            r['world'], r['seed'], 'PASS' if r['pass'] else 'FAIL', ''.join(
                ('%s.' % k[1:]) if v else ('%s!' % k[1:]) for k, v in r['crit'].items()),
            extra, r['final'], r['hold'], r['spin_max_deg'], ('fails ' + ','.join(fails)) if fails else ''))
    n = sum(r['pass'] for r in res)
    print('\n%d / %d runs pass  (%.0f s wall)' % (n, len(res), time.time() - t0))
    for cn in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6'):
        bad = [('%s:s%d' % (r['world'], r['seed'])) for r in res if not r['crit'].get(cn, True)]
        print('  %s: %s' % (cn, 'pass (all runs)' if not bad else 'FAIL in ' + ', '.join(bad)))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == '__main__':
    main()
