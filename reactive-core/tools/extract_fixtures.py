#!/usr/bin/env python3
"""Trial bags -> tests/data/*.npz (read-only on the bags). Run in WSL with ROS sourced.

The trial bags have no raw /livox/points, only the old stack's /quad/cloud (odom frame, z =
height above the feet, built with FAST-LIO). Each chosen frame is moved into the LEVELLED body
frame with the /odom pose at that time: xy rotated by -yaw about the robot, z_body = z - BODY_H.

  python3 tools/extract_fixtures.py --light <bag> --heavy <bagheavy> --out tests/data/x.npz \
      --times 0,1,2,3,4,39,40,40.5,41,41.5,42
Times are seconds after the start of the LIGHT bag.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bagread                          # noqa: E402
from rc import geometry as G            # noqa: E402
from rc.pc2 import cloud_to_xyz         # noqa: E402

BODY_H = 0.32
MAX_PTS = 6000
SENSOR = np.array([0.18678, 0.0, 0.12498])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--light', required=True)
    ap.add_argument('--heavy', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--times', required=True)
    a = ap.parse_args()
    times = [float(x) for x in a.times.split(',')]
    t0 = bagread.start_ns(a.light)

    odom, imu, cmd = [], [], []
    for topic, m, t in bagread.read(a.light, ['/odom', '/imu/data', '/cmd_vel']):
        ts = (t - t0) * 1e-9
        if topic == '/odom':
            p, q = m.pose.pose.position, m.pose.pose.orientation
            odom.append((ts, p.x, p.y, p.z, G.yaw_from_quat(q.x, q.y, q.z, q.w)))
        elif topic == '/imu/data':
            imu.append((ts, m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
        else:
            cmd.append((ts, m.linear.x, m.angular.z))
    odom = np.array(odom)
    cmd = np.array(cmd) if cmd else np.zeros((1, 3))
    print('odom msgs %d, imu %d, cmd %d' % (len(odom), len(imu), len(cmd)))

    clouds = []
    for _, m, t in bagread.read(a.heavy, ['/quad/cloud']):
        clouds.append(((t - t0) * 1e-9, m))
    ct = np.array([c[0] for c in clouds])
    print('clouds %d (%.1f .. %.1f s)' % (len(clouds), ct.min(), ct.max()))

    rng = np.random.default_rng(1)
    out = {'times': np.array(times), 'yaw': [], 'cmd_vx': [], 'cloud_t': []}
    for i, tq in enumerate(times):
        j = int(np.argmin(np.abs(ct - tq)))
        tc, msg = clouds[j]
        k = int(np.argmin(np.abs(odom[:, 0] - tc)))
        _, x, y, _, yaw = odom[k]
        w = cloud_to_xyz(msg).astype(float)
        xy = G.world_to_body(w[:, :2], (x, y, yaw))
        p = np.column_stack([xy, w[:, 2] - BODY_H])
        # the live code drops the zero-range blob in the sensor frame (rc.sensing.filter_sensor)
        p = p[np.linalg.norm(p - SENSOR, axis=1) >= 0.30]
        if p.shape[0] > MAX_PTS:
            p = p[rng.choice(p.shape[0], MAX_PTS, replace=False)]
        kc = int(np.argmin(np.abs(cmd[:, 0] - tc)))
        out['pts_%d' % i] = p.astype(np.float32)
        out['yaw'].append(yaw)
        out['cmd_vx'].append(cmd[kc, 1])
        out['cloud_t'].append(tc)
        front = p[(p[:, 0] > 0.19) & (np.abs(p[:, 1]) < 0.3) & (p[:, 2] > -0.32 + 0.25) & (p[:, 2] < 0.9)]
        near = (front[:, 0].min() - 0.18678) if front.shape[0] else float('nan')
        print('t=%5.1f cloud %.2f s pts %5d  pose (%.2f, %.2f, %4.0f deg)  cmd vx %.2f  nearest above 0.25 m: %.2f m'
              % (tq, tc, p.shape[0], x, y, math.degrees(yaw), cmd[kc, 1], near))
    for k in ('yaw', 'cmd_vx', 'cloud_t'):
        out[k] = np.array(out[k])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print('saved %s (%.0f kB)' % (a.out, os.path.getsize(a.out) / 1024))


if __name__ == '__main__':
    main()
