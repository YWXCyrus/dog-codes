#!/usr/bin/env python3
"""Run the grid-navigation scenarios of sim/scenarios.py in parallel and print a table.

  python3 tools/sim_batch.py                         all scenarios, ideal odometry
  python3 tools/sim_batch.py --drift                 with leg-odometry / gyro drift
  python3 tools/sim_batch.py maze1 stairs --set cell_pause_s=0 -v
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def run_one(args):
    name, sets, drift, T, verbose = args
    from rc import arena as A
    from rc import config
    from sim import scenarios as SC
    from sim.harness import simulate
    cfg = config.load(overrides=sets)
    w, info = SC.SCENARIOS[name]()
    t0 = time.time()
    res = simulate(w, cfg, T=T, lidar_kwargs=SC.FAST_LIDAR, camera=True,
                   dog_kwargs=SC.DRIFT if drift else None, resume_on_hold=True)
    core = res.core
    cells = set(w.layout['cells'])
    vis = set(core.amap.order)
    fin = info['final']
    ex, ey = res.end_pose[0], res.end_pose[1]
    in_final = abs(ex - fin[0] * 1.02) < 0.45 and abs(ey - fin[1] * 1.02) < 0.45
    ap = core.apose(res.t_end)
    err = math.hypot(ap[0] - ex, ap[1] - ey) if ap else None
    wrong = 0
    bad = []
    for k, e in core.amap.edges.items():
        if e.state == A.UNKNOWN:
            continue
        a, b = A.edge_cells(k)
        allc = cells | {fin}
        if a not in allc or b not in allc:
            continue
        truth = 'open' if k in info['open_edges'] else ('curtain' if k in info.get('curtains', ()) else 'wall')
        st = e.state
        if truth == 'curtain' and st == 'open' and e.traversed:
            st = 'curtain'
        wrong += st != truth
        if st != truth:
            bad.append((k, st, truth, [round(v) for v in (e.mh, e.mp, e.uh, e.up)], round(e.cue, 1), e.result))
    holds = [e['hold'] for e in res.events if 'hold' in e]
    cur = [e['curtain'] for e in res.events if 'curtain' in e]
    out = {'name': name, 'cells': '%d/%d' % (len(vis & cells), len(cells)), 'missing': sorted(cells - vis),
           'finished': res.finished, 'in_final': in_final, 't': round(res.t_end), 'coll': res.collisions,
           'rev': res.reversed, 'err': None if err is None else round(err, 2), 'wrong_edges': wrong,
           'holds': holds, 'curtain': cur, 'bad': bad, 'stairs': [e['stairs'] for e in res.events if 'stairs' in e][-1:],
           'wall_s': round(time.time() - t0)}
    if verbose:
        out['log'] = [e for e in res.events if set(e) & {'plan', 'grid', 'grid_block', 'grid_fail', 'curtain', 'hold',
                                                         'stairs', 'arena', 'stairs_grid', 'stairs_cell', 'curtain_answer'}]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--drift', action='store_true')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('-T', type=float, default=480.0)
    ap.add_argument('-v', action='store_true')
    a = ap.parse_args()
    from sim import scenarios as SC
    names = a.names or list(SC.SCENARIOS)
    jobs = [(n, a.set, a.drift, a.T, a.v) for n in names]
    with mp.Pool(min(len(jobs), max(1, mp.cpu_count() - 2))) as pool:
        rows = pool.map(run_one, jobs)
    bad = 0
    for r in rows:
        ok = r['finished'] and r['in_final'] and r['coll'] == 0 and not r['missing']
        bad += not ok
        print('%-4s %-14s cells %-6s fin %-5s inF %-5s t %4s coll %d err %s wrongE %d holds %s curtain %s stairs %s missing %s (%ss)' % (
            'ok' if ok else 'FAIL', r['name'], r['cells'], r['finished'], r['in_final'], r['t'], r['coll'], r['err'],
            r['wrong_edges'], r['holds'], r['curtain'][-3:], r['stairs'], r['missing'], r['wall_s']))
        if r.get('bad') and a.v:
            print('       WRONG BORDERS', r['bad'])
        for e in r.get('log', []):
            print('      ', str(e)[:190])
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
