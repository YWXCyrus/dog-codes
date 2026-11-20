#!/usr/bin/env python3
"""reporter: submit task results to the competition scoring server (HTTP POST, JSON).

Input : /quad/report (JSON from perception: task_id, task_type, scene_type, label, pos, message)
Output: POST http://192.168.10.141:8001/task_report  (booklet v5.1 section F)

* one HTTP request per task result; only the LATEST valid submission per task_id counts, so a
  newer/better result simply replaces the old one;
* fixed vocabularies are validated locally (a rejected submission wastes the whole task);
* network failures are retried with back-off while the result is still the latest;
* every submission and server answer is appended to run_logs/reports_<time>.jsonl.
"""
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))


def validate(rep, group_id):
    """Return (payload, error). Applies the booklet's 'unused field' conventions."""
    tt = rep.get('task_type')
    if tt not in C.TASK_TYPES:
        return None, 'bad task_type %r' % tt
    scene = rep.get('scene_type') or ''
    if scene and scene not in C.SCENE_TYPES:
        return None, 'bad scene_type %r' % scene
    label = rep.get('label') or ''
    pos = rep.get('pos')
    msg = rep.get('message') or ''
    if tt == 'object':
        if label not in C.OBJECT_LABELS:
            return None, 'bad object label %r' % label
    elif tt == 'hazard':
        if label not in C.HAZARD_LABELS:
            return None, 'bad hazard label %r' % label
    elif tt == 'survivor':
        label, msg = '', ''
    elif tt == 'qr':
        label, pos = '', None
        if not msg:
            return None, 'qr without message'
    if tt != 'qr':
        if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
            return None, 'bad pos %r' % (pos,)
        pos = [float(pos[0]), float(pos[1])]
        if not scene:
            return None, 'missing scene_type'
    return {'group_id': group_id, 'task_id': rep['task_id'], 'task_type': tt, 'scene_type': scene,
            'label': label, 'pos': pos, 'message': msg}, None


class Reporter(Node):

    def __init__(self):
        super().__init__('reporter')
        p = C.declare(self, {
            'url': 'http://192.168.10.141:8001/task_report',
            'group_id': 'group02_XXXX',
            'dry_run': False, 'timeout_s': 6.0, 'max_backoff_s': 10.0,
            'log_dir': os.path.join(HERE, 'run_logs'),
        })
        self.p = p
        os.makedirs(p['log_dir'], exist_ok=True)
        self.log_path = os.path.join(p['log_dir'], time.strftime('reports_%Y%m%d_%H%M%S.jsonl'))
        self.latest = {}          # task_id -> payload (latest wanted)
        self.sent = {}            # task_id -> payload last accepted
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.status = {'accepted': 0, 'rejected': 0, 'failed': 0, 'last': ''}
        self.create_subscription(String, C.T_REPORT, self.on_report, 10)
        self.status_pub = self.create_publisher(String, '/quad/reporter_status', 10)
        self.create_timer(1.0, lambda: self.status_pub.publish(C.jmsg(self.status)))
        threading.Thread(target=self.loop, daemon=True).start()
        if 'XXXX' in p['group_id']:
            self.get_logger().error('group_id %s is a PLACEHOLDER - set the registered id in '
                                    'config/params.yaml before the run' % p['group_id'])
        self.get_logger().info('reporter up -> %s (dry_run=%s)' % (p['url'], p['dry_run']))

    def log(self, obj):
        try:
            with open(self.log_path, 'a', encoding='utf8') as f:
                f.write(json.dumps({'t': time.time(), **obj}) + '\n')
        except Exception:
            pass

    def on_report(self, m):
        rep = C.jload(m.data, None)
        if not rep:
            return
        payload, err = validate(rep, self.p['group_id'])
        if err:
            self.get_logger().warn('report %s dropped: %s' % (rep.get('task_id'), err))
            self.log({'dropped': rep, 'why': err})
            return
        with self.lock:
            if self.sent.get(payload['task_id']) == payload:
                return
            self.latest[payload['task_id']] = payload
        self.wake.set()

    def post(self, payload):
        data = json.dumps(payload).encode('utf8')
        req = urllib.request.Request(self.p['url'], data=data, method='POST',
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.p['timeout_s']) as r:
                return r.status, json.loads(r.read().decode('utf8') or '{}')
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode('utf8') or '{}')
            except Exception:
                return e.code, {}
        except Exception as e:
            return 0, {'error': str(e)}

    def loop(self):
        backoff = {}
        while True:
            self.wake.wait(timeout=1.0)
            self.wake.clear()
            with self.lock:
                todo = [(k, v) for k, v in self.latest.items() if self.sent.get(k) != v]
            for tid, payload in todo:
                if time.time() < backoff.get(tid, (0, 0))[0]:
                    continue
                if self.p['dry_run']:
                    self.log({'dry_run': payload})
                    with self.lock:
                        self.sent[tid] = payload
                    continue
                code, resp = self.post(payload)
                st = resp.get('status', '')
                self.log({'post': payload, 'http': code, 'resp': resp})
                if code == 200 and st == 'accepted':
                    self.status['accepted'] += 1
                    self.status['last'] = '%s accepted' % tid
                    with self.lock:
                        self.sent[tid] = payload
                    backoff.pop(tid, None)
                elif st == 'rejected':
                    self.status['rejected'] += 1
                    self.status['last'] = '%s rejected: %s' % (tid, resp.get('reason'))
                    self.get_logger().error('REJECTED %s: %s  payload=%s' % (tid, resp.get('reason'),
                                                                            payload))
                    with self.lock:
                        self.sent[tid] = payload   # do not hammer; a new result replaces it
                else:
                    self.status['failed'] += 1
                    n = backoff.get(tid, (0, 0))[1] + 1
                    wait = min(self.p['max_backoff_s'], 1.0 * 2 ** min(n, 4))
                    backoff[tid] = (time.time() + wait, n)
                    self.status['last'] = '%s failed http=%s' % (tid, code)
                    self.get_logger().warn('post %s failed (http %s %s), retry in %.0fs'
                                           % (tid, code, resp, wait), throttle_duration_sec=3.0)


def main():
    rclpy.init()
    n = Reporter()
    try:
        C.safe_spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
