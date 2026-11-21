#!/usr/bin/env python3
"""Offline measuring on a bag recorded by run.py (read-only). Needs ROS 2 Humble sourced.

  python3 tools/measure.py <run folder>/bag [--from S] [--to S]

Uses the same sensing code as the live run (IMU levelling, obstacle cells, height levels) and prints:
  - free width ahead while walking (narrowest gap passed ~ doorway width -> check lane_w)
  - height levels seen ahead (raised boards, steps: heights above the stance floor -> step_limit)
  - stairs: rise and tread length of each level when the detector sees stairs
  - roll / pitch range while moving and while standing (-> tilt limits)
  - leg odometry vs FAST-LIO: path length and speed (-> is leg odometry usable)
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bagread                                   # noqa: E402
from rc import config, geometry as G            # noqa: E402
from rc import stairs as S                      # noqa: E402
from rc.attitude import Attitude                # noqa: E402
from rc.pc2 import cloud_to_xyz                 # noqa: E402
from rc.sensing import Mount, ObstacleDetector, filter_sensor, level_points   # noqa: E402

TOPICS = ['/livox/points', '/imu/data', '/leg_odom', '/leg_odom2', '/odom', '/tf_static']


def free_width(obs, x0=-0.2, x1=0.6, side=0.2):
    """Gap between the nearest obstacle left and right BESIDE the dog (doorway / corridor width).
    Frames with something straight ahead (|y| < side) are skipped (a wall ahead is not a gap)."""
    if obs.shape[0] == 0:
        return math.inf
    m = (obs[:, 0] > x0) & (obs[:, 0] < x1)
    o = obs[m]
    if np.any(np.abs(o[:, 1]) < side):
        return math.inf
    left = o[o[:, 1] > side, 1]
    right = o[o[:, 1] < -side, 1]
    if left.size == 0 or right.size == 0:
        return math.inf
    return float(left.min() - right.max())


def path_len(poses):
    if len(poses) < 2:
        return 0.0
    p = np.array([q[1:3] for q in poses])
    return float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--from', dest='t_from', type=float, default=0.0)
    ap.add_argument('--to', dest='t_to', type=float, default=1e9)
    a = ap.parse_args()
    cfg = config.load()
    t0 = bagread.start_ns(a.bag)
    att = Attitude()
    mount = Mount(np.array(cfg['mount_xyz']), math.radians(cfg['mount_pitch_deg']))
    det = ObstacleDetector(cfg)
    sdet = S.StairDetector(cfg)
    floor = -cfg['body_h_default']
    leg, lio = [], []
    tilt_move, tilt_still = [], []
    widths, levels_seen, stairs_seen = [], [], []
    last_imu = None
    speed = 0.0
    n_cloud = 0
    for topic, m, tns in bagread.read(a.bag, TOPICS):
        t = (tns - t0) * 1e-9
        if t < a.t_from or t > a.t_to:
            continue
        if topic == '/tf_static':
            for tf in m.transforms:
                if tf.child_frame_id.lstrip('/') == 'livox_frame':
                    tr, q = tf.transform.translation, tf.transform.rotation
                    mount = Mount(np.array([tr.x, tr.y, tr.z]), G.pitch_from_quat(q.x, q.y, q.z, q.w))
        elif topic == '/imu/data':
            dt = 0.005 if last_imu is None else t - last_imu
            last_imu = t
            acc = (m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z)
            gyr = (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)
            att.update(acc, gyr, dt)
            (tilt_move if speed > 0.05 else tilt_still).append((math.degrees(att.roll), math.degrees(att.pitch)))
        elif topic in ('/leg_odom', '/leg_odom2'):
            p, q = m.pose.pose.position, m.pose.pose.orientation
            if topic == '/leg_odom' or not leg:
                leg.append((t, p.x, p.y, G.yaw_from_quat(q.x, q.y, q.z, q.w)))
                if len(leg) > 10 and leg[-1][0] - leg[-10][0] > 0:
                    speed = math.hypot(leg[-1][1] - leg[-10][1], leg[-1][2] - leg[-10][2]) / (leg[-1][0] - leg[-10][0])
        elif topic == '/odom':
            p = m.pose.pose.position
            lio.append((t, p.x, p.y))
        elif topic == '/livox/points':
            n_cloud += 1
            pl = level_points(filter_sensor(cloud_to_xyz(m), cfg['near_drop_m'], cfg['max_range_m']),
                              mount, att.roll, att.pitch)
            s = det.update(pl, floor, t)
            if speed > 0.05:
                widths.append((t, free_width(s.obs)))
            u, _, z = S.lane_points(pl, 0.0, cfg['lane_w'], floor)
            lv = S.levels(u, z)
            levels_seen.append((t, [(round(l[0], 3), round(l[2], 2), round(l[4], 2)) for l in lv]))
            r = sdet.update(pl, floor, t, 10.0)
            if r.kind != 'none':
                stairs_seen.append((t, r.kind, r.edge_x, math.degrees(r.heading), lv))

    print('bag: %s   clouds %d   leg odom %d   fast-lio %d' % (a.bag, n_cloud, len(leg), len(lio)))
    print('\n== free width ahead while walking (0.3-1.0 m ahead; lane_w now %.2f m)' % cfg['lane_w'])
    fin = [(t, w) for t, w in widths if w != math.inf]
    if fin:
        ws = np.array([w for _, w in fin])
        t_min, w_min = min(fin, key=lambda tw: tw[1])
        print('  narrowest %.2f m at t=%.1f s;  10th percentile %.2f m;  %d frames with walls both sides'
              % (w_min, t_min, float(np.percentile(ws, 10)), len(fin)))
    else:
        print('  never had obstacles on both sides')

    print('\n== height levels ahead (above the stance floor; strips/steps -> step_limit %.2f m)' % cfg['step_limit'])
    allz = [l[0] for _, lv in levels_seen for l in lv if abs(l[0]) > 0.04]
    if allz:
        h, e = np.histogram(allz, np.arange(-0.6, 0.8, 0.05))
        for k, n in enumerate(h):
            if n:
                print('  %+.2f..%+.2f m: seen in %d frames' % (e[k], e[k + 1], n))
    else:
        print('  only the stance floor level')

    print('\n== stairs detector')
    if not stairs_seen:
        print('  nothing')
    for kind in ('up', 'unsure', 'down'):
        rows = [x for x in stairs_seen if x[1] == kind]
        if not rows:
            continue
        print('  %-6s %d frames, first t=%.1f s edge %.2f m heading %+.0f deg' % (kind, len(rows), rows[0][0], rows[0][2], rows[0][3]))
        if kind == 'up':
            rises, treads = [], []
            for row in rows:
                ups = [l for l in row[4] if l[0] > 0.04]
                prev = 0.0
                for l in ups:
                    rises.append(l[0] - prev)
                    treads.append(l[4] - l[2])
                    prev = l[0]
            if rises:
                print('     rise median %.3f m (10-90%%: %.3f-%.3f), tread median %.2f m over %d levels'
                      % (float(np.median(rises)), float(np.percentile(rises, 10)), float(np.percentile(rises, 90)),
                         float(np.median(treads)), len(rises)))

    print('\n== tilt (deg)')
    for name, arr in (('moving', tilt_move), ('standing', tilt_still)):
        if arr:
            r = np.array(arr)
            print('  %-8s roll %+.1f..%+.1f  pitch %+.1f..%+.1f  (95%% |max| %.1f)' % (
                name, r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max(),
                float(np.percentile(np.abs(r).max(axis=1), 95))))

    print('\n== odometry')
    print('  leg odometry path %.2f m;  FAST-LIO path %.2f m' % (path_len(leg), path_len([(q[0], q[1], q[2]) for q in lio])))
    if len(leg) > 2:
        d = np.array(leg)
        sp = np.hypot(np.diff(d[:, 1]), np.diff(d[:, 2])) / np.maximum(np.diff(d[:, 0]), 1e-3)
        print('  leg odometry max speed %.2f m/s, max jump %.3f m' % (float(sp.max()), float(np.max(np.hypot(np.diff(d[:, 1]), np.diff(d[:, 2]))))))


if __name__ == '__main__':
    main()
