"""Run one nav_eval job and draw it: walls, the dog's true path, the decisions (cells events).
  python tools/nav_debug.py W7:1 --seed 0 --out run.png [--T 300] [--nav cells]"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.nav_eval import DRIFT, Probe, world_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('job')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--nav', default='cells')
    ap.add_argument('--T', type=float, default=0.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-drift', action='store_true')
    a = ap.parse_args()
    from rc import config
    from sim import course as C
    from sim.harness import simulate
    cw = world_for(a.job)
    pr = Probe()
    pr.goal = cw.goal
    dk = dict(cw.dog_kwargs)
    if not a.no_drift:
        dk['odom_drift'] = DRIFT
    r = simulate(cw.world, config.load(overrides=['nav_mode=%s' % a.nav]), T=a.T or cw.T, start=cw.start,
                 seed=a.seed, dog_kwargs=dk, probe=pr)
    fig, ax = plt.subplots(figsize=(9, 9))
    for b in cw.world.boxes:
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        xs, ys = [], []
        for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
            xs.append(b.cx + c * u * b.sx / 2 - s * v * b.sy / 2)
            ys.append(b.cy + s * u * b.sx / 2 + c * v * b.sy / 2)
        ax.fill(xs, ys, color='#444' if b.name == 'wall' else '#c90', alpha=0.8)
    px = [p[0] for p in pr.path]
    py = [p[1] for p in pr.path]
    ax.plot(px, py, color='#1f77b4', lw=1)
    ax.plot(px[0], py[0], 'go')
    ax.plot(px[-1], py[-1], 'rs')
    # decision points: the dog's true position at the event time
    times = [p[2] for p in pr.path]
    import bisect
    for e in r.events:
        if 'cells' in e and pr.t0 is not None:
            i = min(len(times) - 1, bisect.bisect_left(times, e['t'] - pr.t0))
            ax.plot(px[i], py[i], '.', color={'new': 'g', 'frontier': 'm', 'explored': 'r', 'dead_end': 'k',
                                              'look': 'c'}.get(e['cells'], 'y'), ms=7)
    cov = ''
    if cw.kind == 'maze':
        got = C.maze_coverage(cw, [(x, y) for x, y, tr in pr.path if tr <= 300])
        cov = ' cov %d/%d' % (len(got), cw.cells)
        cr, sr = math.cos(cw.rot), math.sin(cw.rot)
        for i in range(len(cw.xb) - 1):
            for j in range(len(cw.yb) - 1):
                mx, my = (cw.xb[i] + cw.xb[i + 1]) / 2, (cw.yb[j] + cw.yb[j + 1]) / 2
                ax.text(cr * mx - sr * my, sr * mx + cr * my, 'v' if (i, j) in got else 'X',
                        ha='center', va='center', color='g' if (i, j) in got else 'r', fontsize=9)
    ax.set_aspect('equal')
    ax.set_title('%s seed %d nav %s: final %s %s goal %s%s stalls %s tilt_turn %d' % (
        cw.name, a.seed, a.nav, r.final_mode, r.hold_reason, pr.goal_t, cov, pr.stall, len(pr.tilt_turn)), fontsize=8)
    fig.savefig(a.out, dpi=100)
    ev = [e for e in r.events if 'cells' in e or 'hold' in e or 'cells_stall' in e]
    print('events (%d):' % len(ev))
    for e in ev:
        print('  ', e)


if __name__ == '__main__':
    main()
