#!/usr/bin/env python3
"""End-to-end check of the navigation cues on ROS_DOMAIN_ID=42 (used by test_cues_ros.sh).

Publishes fake camera-model detections (/rc/detections: stairs ahead, slightly left), asks
/quad/vlm_ask {'kind': 'stairs'} after a few seconds, and records /quad/event, /quad/vlm_answer and
/quad/mission_status. Prints PASS / FAIL.
"""
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '42':
        print('refusing: run with ROS_DOMAIN_ID=42')
        return 2
    run_s = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    rclpy.init()
    n = Node('check_cues')
    det_pub = n.create_publisher(String, '/rc/detections', 10)
    ask_pub = n.create_publisher(String, '/quad/vlm_ask', 10)
    got = {'events': [], 'answers': [], 'status': None}
    n.create_subscription(String, '/quad/event', lambda m: got['events'].append(json.loads(m.data)), 50)
    n.create_subscription(String, '/quad/vlm_answer', lambda m: got['answers'].append(json.loads(m.data)), 10)
    n.create_subscription(String, '/quad/mission_status', lambda m: got.__setitem__('status', json.loads(m.data)), 10)

    def pub_det():
        s = String()
        s.data = json.dumps({'t': time.time(), 'dets': [{'cls': 'stairs', 'conf': 0.83, 'bearing': 0.12, 'w': 0.45}]})
        det_pub.publish(s)

    n.create_timer(0.2, pub_det)
    t0 = time.monotonic()
    asked = False
    while time.monotonic() - t0 < run_s:
        rclpy.spin_once(n, timeout_sec=0.1)
        if not asked and time.monotonic() - t0 > run_s / 2:
            s = String()
            s.data = json.dumps({'id': 'nav7', 'kind': 'stairs'})
            ask_pub.publish(s)
            asked = True
    n.destroy_node()
    rclpy.try_shutdown()

    stairs_ev = [e for e in got['events'] if e.get('type') == 'stairs_seen']
    ans = [a for a in got['answers'] if a.get('id') == 'nav7']
    st = got['status'] or {}
    checks = [
        ('stairs_seen event from the camera model', bool(stairs_ev) and stairs_ev[0].get('src') == 'yolo'
         and abs(stairs_ev[0].get('bearing', 0) - 0.12) < 1e-6),
        ('stairs question answered yes at once by the camera model', bool(ans) and ans[0].get('answer') is True
         and ans[0].get('src') == 'yolo'),
        ('mission_status published', bool(st) and 'open' in st and 'suggest' in st),
        ('M04 reported in mission_status after the QR was read', st.get('tasks', {}).get('M04', {}).get('reported') is True),
        ('stairs cell recorded', bool(st.get('stairs_cells'))),
    ]
    for name, ok in checks:
        print(('PASS ' if ok else 'FAIL ') + name)
    print('events:', [e.get('type') for e in got['events']][:12])
    print('status open:', st.get('open'), 'suggest:', st.get('suggest'))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
