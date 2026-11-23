#!/usr/bin/env python3
"""Curtain features of the surface ahead, from a recorded run (or several). Needs ROS sourced.

  python3 tools/curtain_probe.py runs/<date_time>/bag [more bags]          # percentiles per bag
  python3 tools/curtain_probe.py runs/<date_time>/bag --frames             # every scan
  python3 tools/curtain_probe.py runs/<date_time>/bag --from 12 --to 20    # a time window (s)

Practice day: stand / walk the dog towards the real curtain (and towards a plain panel) with the
recorder on, then run this on both recordings. Set cur_*_lo above what the panel reaches and
cur_*_hi near what the curtain reaches (config.yaml), check with tests/test_curtain.py.
The same pipeline as the live code: rc.core.Core levels the scan with the IMU and the leg floor.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rc import config, geometry as G                   # noqa: E402
from rc.core import Core                               # noqa: E402
from rc.curtain import CurtainDetector                 # noqa: E402
from rc.pc2 import cloud_to_xyz                        # noqa: E402

FEATS = ('dist', 'width', 'wavy', 'thick', 'pen', 'flutter', 'score')


def probe(bag, cfg, t_from=0.0, t_to=1e9, frames=False):
    from tools import bagread
    core = Core(cfg)
    det = CurtainDetector(cfg)
    t0 = bagread.start_ns(bag)
    rows = []
    for topic, m, tn in bagread.read(bag, ['/livox/points', '/imu/data', '/joint_states', '/leg_odom']):
        t = (tn - t0) / 1e9
        if topic == '/imu/data':
            a, w = m.linear_acceleration, m.angular_velocity
            core.on_imu((a.x, a.y, a.z), (w.x, w.y, w.z), t)
        elif topic == '/joint_states':
            vel = list(m.velocity) if len(m.velocity) == len(m.name) else [0.0] * len(m.name)
            core.on_joints(list(m.name), list(m.position), vel, t)
        elif topic == '/leg_odom':
            p, q = m.pose.pose.position, m.pose.pose.orientation
            core.on_odom(p.x, p.y, G.yaw_from_quat(q.x, q.y, q.z, q.w), t)
        else:
            core.on_cloud(cloud_to_xyz(m).astype(float), t)
            if not (t_from <= t <= t_to):
                continue
            s = det.update(core.p_level, core.floor_z, t, core.gyro[2])
            if s is None:
                continue
            d = s.summary()
            d['t'] = round(t, 1)
            d['confident'] = det.confident()
            d['suspect'] = det.suspect()
            rows.append(d)
            if frames:
                print('t %6.1f  %s%s' % (t, '  '.join('%s %s' % (k, d[k]) for k in
                                                     ('dist', 'bearing_deg', 'width', 'tall', 'wavy', 'thick', 'pen', 'flutter', 'score')),
                                          '  CANDIDATE' if d['confident'] else ''))
        if t > t_to:
            break
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--frames', action='store_true')
    ap.add_argument('--from', dest='t_from', type=float, default=0.0)
    ap.add_argument('--to', dest='t_to', type=float, default=1e9)
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    cfg = config.load(overrides=a.set)
    allrows = []
    for bag in a.bags:
        rows = probe(bag, cfg, a.t_from, a.t_to, a.frames)
        allrows += rows
        near = [r for r in rows if r['dist'] <= cfg['cur_detect_m']]
        print('%s: %d surfaces ahead, %d within %.1f m, %d candidate scans'
              % (bag, len(rows), len(near), cfg['cur_detect_m'], sum(r['confident'] for r in rows)))
    near = [r for r in allrows if r['dist'] <= cfg['cur_detect_m']]
    if near:
        print('\nwithin %.1f m, all bags (%d scans):   p50     p90     p99     max' % (cfg['cur_detect_m'], len(near)))
        for k in ('tall', 'wavy', 'thick', 'pen', 'flutter', 'score'):
            v = np.array([r[k] for r in near])
            print('  %-8s %s' % (k, '  '.join('%6.4f' % x for x in (np.percentile(v, 50), np.percentile(v, 90),
                                                                  np.percentile(v, 99), v.max()))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
