"""Ask Nav2's planner server for paths on the sim arena map and judge them (viability experiment).

Needs the stack from run_probe.sh. For each start/goal pair it measures latency, path length and checks
that every path point is free in the SOURCE map (not the inflated costmap), then turns the path into the
same 'heading hint' the driver uses (rc.planner.lookahead_yaw, 1 m ahead).

    python3 tools/nav2/probe.py [map_dir]
"""
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rc.planner import lookahead_yaw  # noqa: E402
from sim import world as W  # noqa: E402

PAIRS = [  # name, start (x, y, yaw), goal (x, y)
    ('start pit to the room door', (0.0, 0.0, 0.0), (5.25, 2.2)),
    ('into the room', (5.25, 2.0, math.pi / 2), (6.2, 4.2)),
    ('across the arena, behind the block', (0.5, -0.5, 0.0), (6.3, 0.6)),
    ('goal inside a wall (must fail or relocate)', (0.0, 0.0, 0.0), (3.5, 3.5)),
    ('goal behind the dog', (3.0, 0.0, 0.0), (0.5, 0.0)),
]


def pose(x, y, yaw, stamp):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = stamp
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


def main():
    world = W.arena_default()
    rclpy.init()
    node = rclpy.create_node('nav2_probe')
    ac = ActionClient(node, ComputePathToPose, 'compute_path_to_pose')
    if not ac.wait_for_server(timeout_sec=20.0):
        print('FAIL: no compute_path_to_pose action server (is run_probe.sh running?)')
        return 2
    ok = True
    for name, (sx, sy, syaw), (gx, gy) in PAIRS:
        goal = ComputePathToPose.Goal()
        now = node.get_clock().now().to_msg()
        goal.goal = pose(gx, gy, 0.0, now)
        goal.start = pose(sx, sy, syaw, now)
        goal.use_start = True
        t0 = time.monotonic()
        fut = ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
        h = fut.result()
        if h is None or not h.accepted:
            print('%-45s REJECTED' % name)
            ok = False
            continue
        rf = h.get_result_async()
        rclpy.spin_until_future_complete(node, rf, timeout_sec=10.0)
        res = rf.result()
        dt = (time.monotonic() - t0) * 1000.0
        if res is None or res.status != 4 or len(res.result.path.poses) == 0:
            print('%-45s no path (status %s) in %.0f ms' % (name, None if res is None else res.status, dt))
            continue
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in res.result.path.poses])
        length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
        inside = [b.name for b in world.boxes if b.name not in W.PASS and b.z1 >= 0.25
                  and np.any(b.contains_xy(pts[:, 0], pts[:, 1]))]
        yaw = lookahead_yaw(pts.tolist(), (sx, sy, syaw), 1.0)
        straight = math.hypot(gx - sx, gy - sy)
        print('%-45s path %2d pts  %.2f m (straight %.2f)  %4.0f ms  hint yaw %5.1f deg  %s' % (
            name, len(pts), length, straight, dt, math.degrees(yaw),
            'CROSSES ' + ','.join(sorted(set(inside))) if inside else 'clear of walls'))
        ok = ok and not inside
    node.destroy_node()
    rclpy.shutdown()
    print('RESULT', 'paths avoid every wall' if ok else 'PROBLEM: see above')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
