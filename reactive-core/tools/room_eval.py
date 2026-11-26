"""Room looks + reported cells on the booklet arenas (nav_mode cells, stage 2, 2026-11-26).

A fake perception stands in for frontier-fastlio/dog_codes/perception.py: from /rc/arena_odom (here
core.arena_pose()) it takes the cell the camera looks at the way perception does in pos_mode 'view'
(view_max_m 1.0 ahead, snapped to the 1.02 m cells) and answers /quad/scene_state for it 3.5 s after
the dog came to a stand facing it (the scene server's ~3.5 s). Checked per room look:
  cell   the cell perception would report (from our position) == the true cell the camera looks at
  room   == the room the navigator meant to look at
and per run: coverage, no collisions, stays in the arena.

  python3 tools/room_eval.py [--seeds 0,1,2,3,4,5] [--procs 6]
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

P = 1.02
ANSWER_S = 3.5


def view_cell(x, y, yaw, d=1.0):
    return int(round((x + d * math.cos(yaw)) / P)), int(round((y + d * math.sin(yaw)) / P))


def run(seed):
    from rc import config
    from sim import course as C
    from sim.harness import simulate
    from tools.nav_eval import DRIFT
    cw = C.arena35(seed)
    x0, y0, yaw0 = cw.start
    st = {'look_t0': None, 'looks': [], 'answered': set()}

    def probe(t, dog, core, cmd):
        nav = core.driver.cells
        if nav is None or nav.arena_pose is None:
            return
        if core.driver.phase != 'cells:LOOK':
            st['look_t0'] = None
            return
        x, y, yaw, v = nav.arena_pose
        if st['look_t0'] is None:
            st['look_t0'] = t
            # the truth in the arena frame
            dx, dy = dog.x - x0, dog.y - y0
            c, s = math.cos(-yaw0), math.sin(-yaw0)
            tx, ty, tyaw = c * dx - s * dy, s * dx + c * dy, dog.yaw - yaw0
            st['looks'].append({'t': round(t, 1), 'est': view_cell(x, y, yaw), 'true': view_cell(tx, ty, tyaw),
                                'room': tuple(nav.look['cell']) if nav.look else None,
                                'pos_err': round(math.hypot(x - tx, y - ty), 3)})
        if t - st['look_t0'] >= ANSWER_S:
            core.on_scene_state(view_cell(x, y, yaw), t)

    cfg = config.load(overrides=['nav_mode=cells', 'cells_wall_half_t=0.05'])
    r = simulate(cw.world, cfg, T=cw.T, start=cw.start, seed=0, dog_kwargs=dict(cw.dog_kwargs, odom_drift=DRIFT),
                 probe=probe)
    got = C.maze_coverage(cw, r.path) | C.looked_cells(cw, r.events)      # looked into = covered
    out_n = sum(1 for x, y in r.path if C.outside_arena(cw, x, y))
    # true 3-wall cells (rooms / dead ends) of this arena, in the arena frame, from the world's walls
    looks = st['looks']
    ok = all(lk['est'] == lk['true'] == lk['room'] for lk in looks)
    return {'seed': seed, 'cov': len(got), 'cells': cw.cells, 'collisions': r.collisions, 'outside': out_n,
            'hold': r.hold_reason, 'looks': looks, 'cells_ok': ok,
            'pass': ok and r.collisions == 0 and out_n == 0 and len(got) >= cw.cells - 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2,3,4,5')
    ap.add_argument('--procs', type=int, default=6)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    t0 = time.time()
    with mp.Pool(min(a.procs, len(seeds))) as pool:
        res = pool.map(run, seeds)
    for r in res:
        print('A:%d %s cov %d/%d coll %d out %d hold %-9s looks %d cells_ok %s' % (
            r['seed'], 'PASS' if r['pass'] else 'FAIL', r['cov'], r['cells'], r['collisions'], r['outside'],
            r['hold'], len(r['looks']), r['cells_ok']))
        for lk in r['looks']:
            flag = '' if lk['est'] == lk['true'] == lk['room'] else '   <-- WRONG'
            print('     t %6.1f  reported %-8s true %-8s meant %-8s pos err %.2f%s' % (
                lk['t'], lk['est'], lk['true'], lk['room'], lk['pos_err'], flag))
    print('\n%d / %d pass  (%.0f s)' % (sum(r['pass'] for r in res), len(res), time.time() - t0))


if __name__ == '__main__':
    main()
