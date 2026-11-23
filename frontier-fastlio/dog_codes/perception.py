#!/usr/bin/env python3
"""perception: camera -> mission results.

* QR (M04): OpenCV decoder at ~6 Hz; message parsed for a scene hint (-> M05 target room).
  If a QR is located but not decodable (too small) -> event qr_seen; in mode QR_APPROACH
  this node creeps towards it until decoded.
* Hazard signs (M02): local yellow-triangle detector + template classifier (vision.py),
  position projected onto the arena grid using the lidar range along the sign's bearing.
* Scene understanding (M01/M03/M05, room type): trigger-based queries to the organisers'
  server (one in flight, worker thread; failures are tolerated), parsed into the fixed
  vocabulary. Survivors/objects are assigned to task ids from mission.yaml.
* Answers camera questions from the curtain node (/quad/vlm_ask -> /quad/vlm_answer).
Results go to /quad/report (JSON) -> reporter node -> scoring server.
"""
import collections
import math
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

import common as C
import scene_client as SC
import vision as V


class Perception(Node):

    def __init__(self):
        super().__init__('perception')
        p = C.declare(self, {
            'image_topic': '/camera/color/image_raw',
            'fx': 462.6, 'cx': 320.1,
            'mission_file': os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'config', 'mission.yaml'),
            'group_id': 'group02_XXXX',
            'scene_server': 'http://192.168.10.141:8000',
            'scene_path': '/analyze', 'scene_image_field': 'image',
            'scene_prompt_field': 'prompt', 'scene_group_field': 'group_id',
            'scene_group_short': True,      # send "group02" (as the stress test did) not full id
            'scene_enabled': True, 'scene_min_gap_s': 2.0, 'scene_stationary_gap_s': 9.0,
            'scene_max_speed': 0.4, 'blur_min': 25.0, 'jpeg_quality': 85,
            'scene_conf_min': 0.35, 'survivor_conf_min': 0.5,
            'qr_hz': 6.0, 'hazard_hz': 4.0, 'hz_min_score': 0.55, 'hz_min_margin': 0.04,
            'hz_votes': 3, 'pos_mode': 'view', 'view_max_m': 1.0, 'hazard_pos_inset_m': 0.25,
            'qr_assist_max_s': 15.0, 'qr_assist_w_px': 140.0, 'qr_assist_speed': 0.07,
            'qr_persist_n': 3, 'qr_persist_s': 1.0, 'qr_persist_px': 50.0, 'qr_w_min': 20.0, 'qr_w_max': 700.0,
            # standalone use without the frontier supervisor (e.g. next to reactive-core): no /quad/mode
            # arrives, so start in this mode ('EXPLORE' enables the scene queries); /quad/mode still wins
            'start_mode': 'IDLE', 'odom_topic': '/odom',
        })
        self.p = p
        self.qr_filter = C.QrPersist(p['qr_persist_n'], p['qr_persist_s'], p['qr_persist_px'],
                                     p['qr_w_min'], p['qr_w_max'], 2.0 * p['cx'])
        self.mission = self.load_mission(p['mission_file'])
        self.px = self.py = self.yaw = 0.0
        self.speed = 0.0
        self.wz = 0.0
        self.have_odom = False
        self.mode = p['start_mode']
        self.img = None
        self.img_t = 0.0
        self.img_pose = (0, 0, 0)
        self.last_qr_proc = 0.0
        self.last_hz_proc = 0.0
        self.cloud_xy = None       # (N,3) rel x,y,h of last cloud (odom frame)
        self.qr_seen = None
        self.qr_text = None
        self.qr_assist_t0 = 0.0
        self.hz_votes = collections.deque(maxlen=6)
        self.scene_by_cell = {}    # cell -> {scene: weight}
        self.scene_hist = collections.deque(maxlen=12)   # (t, scene, conf)
        self.best = {}             # task_id -> report dict
        self.survivors = {}        # cell -> {'scene','conf','first_t','n'}
        self.m05_scene_hint = ''
        self.qr_cell = None
        self.obj_votes = {}
        self.haz_votes_w = {}
        self.last_cell = None
        self.last_query_t = 0.0
        self.last_query_cell = None
        self.jobs = queue.Queue(maxsize=3)
        self.results = queue.Queue()
        self.in_flight = False
        self.vlm_pending = None
        self.fail_n = 0
        self.last_report_t = {}

        grp = p['group_id'][:7] if p['scene_group_short'] else p['group_id']
        self.client = SC.SceneClient(p['scene_server'], grp, p['scene_path'], p['scene_image_field'],
                                     p['scene_prompt_field'], p['scene_group_field'],
                                     logger=lambda m: self.get_logger().info(m))
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.report_pub = self.create_publisher(String, C.T_REPORT, 10)
        self.evt_pub = self.create_publisher(String, C.T_EVENT, 10)
        self.cmd_pub = self.create_publisher(Twist, C.T_CMD, 10)
        self.ans_pub = self.create_publisher(String, '/quad/vlm_answer', 10)
        self.dbg_pub = self.create_publisher(String, '/quad/scene_state', 10)
        self.create_subscription(Image, p['image_topic'], self.on_image, C.SENSOR_QOS)
        self.create_subscription(Odometry, p['odom_topic'], self.on_odom, C.SENSOR_QOS)
        self.create_subscription(PointCloud2, C.T_CLOUD, self.on_cloud, C.SENSOR_QOS)
        self.create_subscription(String, C.T_MODE, lambda m: setattr(self, 'mode', m.data), 10)
        self.create_subscription(String, '/quad/vlm_ask', self.on_ask, 10)
        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.drain_results)
        self.create_timer(1.0, self.scheduler)
        self.get_logger().info('perception up: group=%s tasks=%s mode=%s odom=%s'
                               % (grp, list(self.mission), self.mode, p['odom_topic']))

    # ------------------------------------------------------------ config
    def load_mission(self, path):
        d = {'M01': {'task_type': 'object', 'label': ''}, 'M02': {'task_type': 'hazard', 'label': ''},
             'M03': {'task_type': 'survivor', 'scene_type': ''}, 'M04': {'task_type': 'qr'},
             'M05': {'task_type': 'survivor', 'scene_type': ''}}
        try:
            with open(path, 'r', encoding='utf8') as f:
                y = yaml.safe_load(f) or {}
            for k, v in (y.get('tasks') or {}).items():
                d.setdefault(k, {}).update(v or {})
            self.mission_text = str(y.get('mission_text', ''))
        except Exception as e:
            self.get_logger().warn('mission file %s not loaded (%s): defaults' % (path, e))
            self.mission_text = ''
        return d

    # ------------------------------------------------------------ inputs
    def on_odom(self, m):
        self.px, self.py = m.pose.pose.position.x, m.pose.pose.position.y
        q = m.pose.pose.orientation
        self.yaw = C.quat_to_rpy(q.x, q.y, q.z, q.w)[2]
        v = m.twist.twist.linear
        self.speed = math.hypot(v.x, v.y)
        self.wz = m.twist.twist.angular.z
        self.have_odom = True
        cell = C.cell_of(self.px, self.py)
        if cell != self.last_cell:
            self.last_cell = cell

    def on_cloud(self, msg):
        pts = C.cloud_to_xyz(msg)
        if pts.shape[0]:
            self.cloud_xy = np.stack([pts[:, 0] - self.px, pts[:, 1] - self.py, pts[:, 2]], 1)

    def on_image(self, msg):
        try:
            self.img = C.image_to_bgr(msg)
        except Exception as e:
            self.get_logger().warn('image decode: %s' % e, throttle_duration_sec=10.0)
            return
        self.img_t = time.monotonic()
        self.img_pose = (self.px, self.py, self.yaw)

    # ------------------------------------------------------------ helpers
    def range_along(self, world_angle, cone=0.09, lo=0.15, hi=1.6):
        """Distance to the nearest obstacle point near `world_angle` (rad), or None."""
        c = self.cloud_xy
        if c is None:
            return None
        r = np.hypot(c[:, 0], c[:, 1])
        a = np.arctan2(c[:, 1], c[:, 0])
        d = np.abs((a - world_angle + math.pi) % (2 * math.pi) - math.pi)
        sel = (d < cone) & (c[:, 2] > lo) & (c[:, 2] < hi) & (r > 0.3)
        if sel.sum() < 3:
            return None
        return float(np.percentile(r[sel], 15))

    def view_point(self):
        """World point the camera is looking at (robot + min(obstacle range - 0.25, view_max) ahead)."""
        if self.p['pos_mode'] == 'robot':
            return self.px, self.py
        rg = self.range_along(self.yaw, cone=0.3, lo=0.2)
        d = self.p['view_max_m'] if rg is None else C.clamp(rg - 0.25, 0.0, self.p['view_max_m'])
        return self.px + d * math.cos(self.yaw), self.py + d * math.sin(self.yaw)

    def pos_here(self, pose=None, ahead=None):
        x, y = self.view_point()
        return C.snap_to_cell_centre(x, y)

    def scene_now(self, cell=None):
        """Best-guess room type for reporting (never empty)."""
        if cell is not None and cell in self.scene_by_cell and self.scene_by_cell[cell]:
            d = self.scene_by_cell[cell]
            return max(d, key=d.get)
        now = time.monotonic()
        recent = [(c, s) for t, s, c in self.scene_hist if now - t < 25 and s]
        if recent:
            return max(recent)[1]
        tot = collections.Counter()
        for d in self.scene_by_cell.values():
            for k, v in d.items():
                tot[k] += v
        if tot:
            return tot.most_common(1)[0][0]
        return 'workshop'

    def emit(self, task_id, task_type, scene='', label='', pos=None, message='', score=0.5,
             src='', cell=None, force=False):
        cfg = self.mission.get(task_id, {})
        if cfg.get('scene_type'):
            scene = cfg['scene_type']
        rep = {'task_id': task_id, 'task_type': task_type, 'scene_type': scene, 'label': label,
               'pos': pos, 'message': message, 'score': round(float(score), 3), 'src': src,
               'cell': list(cell) if cell else None}
        prev = self.best.get(task_id)
        if prev and not force and rep['score'] < prev['score'] - 1e-6 and task_id != 'M04':
            return   # keep the more confident observation
        t = time.monotonic()
        key = (task_id, scene, label, str(pos), message)
        if self.last_report_t.get(key, 0) + 8.0 > t:
            return
        self.last_report_t[key] = t
        self.best[task_id] = rep
        try:     # snapshot of what the camera saw when this result was produced (for later review)
            if self.img is not None:
                d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_logs', 'snaps')
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, '%s_%s_%s_%s.jpg' % (time.strftime('%H%M%S'), task_id,
                            label or scene or 'x', src)), self.img)
        except Exception:
            pass
        self.report_pub.publish(C.jmsg(rep))
        self.get_logger().info('REPORT %s' % rep)

    # ------------------------------------------------------------ QR + hazards
    def tick(self):
        if self.img is None or time.monotonic() - self.img_t > 1.5 or not self.have_odom:
            return
        t = time.monotonic()
        img = self.img
        pose = self.img_pose
        if t - self.last_qr_proc > 1.0 / self.p['qr_hz']:
            self.last_qr_proc = t
            self.do_qr(img, pose)
        if t - self.last_hz_proc > 1.0 / self.p['hazard_hz']:
            self.last_hz_proc = t
            self.do_hazard(img, pose)
        self.qr_assist()

    def do_qr(self, img, pose):
        try:
            r = V.decode_qr(img)
        except Exception as e:
            self.get_logger().warn('qr: %s' % e, throttle_duration_sec=10)
            return
        if r is None:
            return
        text, w, c = r
        if text:
            if text != self.qr_text:
                self.get_logger().warn('QR decoded: %r' % text)
            self.qr_text = text
            self.qr_seen = None
            self.qr_cell = C.cell_of(pose[0], pose[1])
            low = text.lower().replace('_', ' ')
            for s in C.SCENE_TYPES:
                if s.replace('_', ' ') in low:
                    self.m05_scene_hint = s
                    break
            self.emit('M04', 'qr', scene=self.scene_now(C.cell_of(pose[0], pose[1])), message=text,
                      score=1.0, src='qr')
            self.evt_pub.publish(C.jmsg({'type': 'qr_decoded', 'text': text,
                                         'hint': self.m05_scene_hint,
                                         'w': round(w, 1), 'cx': round(c[0], 1), 'cy': round(c[1], 1)}))
            self.assign_survivors()
        else:
            if not self.qr_filter.update(time.monotonic(), w, c[0]):
                return                      # implausible or a one-frame flicker: do not chase it
            self.qr_seen = {'w': w, 'cx': c[0], 't': time.monotonic()}
            # cy: reactive-core's QR approach stops getting closer when the code nears the picture top
            self.evt_pub.publish(C.jmsg({'type': 'qr_seen', 'w': round(w, 1), 'cx': round(c[0], 1),
                                         'cy': round(c[1], 1)}))

    def qr_assist(self):
        if self.mode != 'QR_APPROACH':
            self.qr_assist_t0 = 0.0
            return
        t = time.monotonic()
        if self.qr_assist_t0 == 0.0:
            self.qr_assist_t0 = t
        tw = Twist()
        done = None
        s = self.qr_seen
        if self.qr_text and t - self.qr_assist_t0 > 0.5:
            done = 'decoded'
        elif t - self.qr_assist_t0 > self.p['qr_assist_max_s']:
            done = 'timeout'
        elif s is None or t - s['t'] > 1.5:
            if t - self.qr_assist_t0 > 3.0:
                done = 'lost'
        else:
            if s['w'] >= self.p['qr_assist_w_px']:
                done = 'close_enough'
            else:
                err = (s['cx'] - self.p['cx']) / 320.0
                tw.angular.z = C.clamp(-1.2 * err, -0.4, 0.4)
                ahead = self.range_along(self.yaw, cone=0.35, lo=0.2)
                if abs(err) < 0.25 and (ahead is None or ahead > 0.40):
                    tw.linear.x = self.p['qr_assist_speed']
        if done:
            self.cmd_pub.publish(Twist())
            self.evt_pub.publish(C.jmsg({'type': 'qr_assist_done', 'why': done}))
            self.qr_assist_t0 = 0.0
            self.qr_seen = None
        else:
            self.cmd_pub.publish(tw)

    def do_hazard(self, img, pose):
        try:
            hz = V.detect_hazards(img)
        except Exception as e:
            self.get_logger().warn('hazard: %s' % e, throttle_duration_sec=10)
            return
        p = self.p
        votes = [(h['label'], h) for h in hz if h['label'] and h['score'] >= p['hz_min_score']
                 and h['margin'] >= p['hz_min_margin']]
        if abs(self.wz) > 0.12:       # image/cloud/pose are not time-synced: skip while turning
            return
        self.hz_votes.append(votes)
        counts = collections.Counter(l for vs in self.hz_votes for l, _ in vs)
        if not counts:
            return
        label, n = counts.most_common(1)[0]
        if n < p['hz_votes']:
            return
        cfg = self.mission.get('M02', {})
        if cfg.get('label') and cfg['label'] != label:
            return
        hh = [h for vs in self.hz_votes for l, h in vs if l == label]
        h = max(hh, key=lambda d: d['score'])
        bearing = -math.atan((h['cx'] - p['cx']) / p['fx'])
        wa = pose[2] + bearing
        rng = self.range_along(wa)
        if rng is None:
            rng = 1.0
        rng = max(rng - p['hazard_pos_inset_m'], 0.0)
        x = pose[0] + rng * math.cos(wa)
        y = pose[1] + rng * math.sin(wa)
        pos = C.snap_to_cell_centre(x, y)
        cell = C.cell_of(x, y)
        off = abs(h['cx'] - p['cx']) / 320.0
        quality = 0.5 * min(1.0, h['score']) + 0.3 * (1 - C.clamp(rng / 2.5, 0, 1)) + 0.2 * (1 - off)
        key = (label, tuple(cell))
        self.haz_votes_w[key] = self.haz_votes_w.get(key, 0.0) + quality
        (bl, bcell), w = max(self.haz_votes_w.items(), key=lambda kv: kv[1])
        self.emit('M02', 'hazard', scene=self.scene_now(bcell), label=bl,
                  pos=C.snap_to_cell_centre(*C.cell_centre(bcell)), score=min(0.99, 0.4 + 0.05 * w),
                  src='hazard', cell=bcell, force=True)

    # ------------------------------------------------------------ VLM scheduling
    def on_ask(self, m):
        d = C.jload(m.data, {})
        if d.get('kind') == 'curtain' and self.img is not None:
            self.vlm_pending = d.get('id')
            self.push_job({'kind': 'curtain', 'id': d.get('id'), 'img': self.img.copy(),
                           'prompt': SC.CURTAIN_PROMPT}, front=True)
        else:
            self.ans_pub.publish(C.jmsg({'id': d.get('id'), 'answer': False, 'text': 'no image'}))

    def push_job(self, job, front=False):
        try:
            if front:
                while not self.jobs.empty():
                    try:
                        self.jobs.get_nowait()
                    except queue.Empty:
                        break
            self.jobs.put_nowait(job)
        except queue.Full:
            pass

    def scheduler(self):
        p = self.p
        if not p['scene_enabled'] or self.img is None or not self.have_odom:
            return
        if self.mode in ('STAIRS', 'CURTAIN', 'DONE', 'IDLE', 'FINAL'):
            return
        t = time.monotonic()
        if self.in_flight or not self.jobs.empty() or t - self.img_t > 1.0:
            return
        if self.speed > p['scene_max_speed']:
            return
        vx_, vy_ = self.view_point()
        cell = C.cell_of(vx_, vy_)
        gap = t - self.last_query_t
        new_cell = cell != self.last_query_cell
        if not ((new_cell and gap > p['scene_min_gap_s']) or gap > p['scene_stationary_gap_s']):
            return
        img = self.img
        if cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < p['blur_min']:
            return
        self.last_query_t = t
        self.last_query_cell = cell
        self.push_job({'kind': 'scene', 'img': img.copy(), 'prompt': SC.build_prompt(self.mission_text),
                       'pose': self.img_pose, 'cell': cell, 'view': (vx_, vy_)})

    def worker_loop(self):
        discovered = False
        while True:
            job = self.jobs.get()
            self.in_flight = True
            try:
                if not discovered:
                    self.client.discover()
                    discovered = True
                ok, buf = cv2.imencode('.jpg', job['img'], [cv2.IMWRITE_JPEG_QUALITY,
                                                           self.p['jpeg_quality']])
                if not ok:
                    continue
                res, info = self.client.analyse(buf.tobytes(), job['prompt'])
                self.results.put((job, res, info))
            except Exception as e:  # keep the worker alive no matter what
                self.results.put((job, None, 'exception %s' % e))
            finally:
                self.in_flight = False

    def drain_results(self):
        while True:
            try:
                job, res, info = self.results.get_nowait()
            except queue.Empty:
                break
            if res is None:
                self.fail_n += 1
                self.get_logger().warn('scene query failed (%s): %s' % (job['kind'], info),
                                       throttle_duration_sec=5.0)
                if job['kind'] == 'curtain':
                    self.ans_pub.publish(C.jmsg({'id': job['id'], 'answer': False,
                                                 'text': 'query failed: ' + info, 'failed': True}))
                continue
            self.fail_n = 0
            if job['kind'] == 'curtain':
                txt = (res['desc'] + ' ' + ' '.join(map(str, res['evidence']))).lower()
                yes = res['relevant'] or any(w in txt for w in ('curtain', 'cloth', 'fabric', 'drape'))
                self.ans_pub.publish(C.jmsg({'id': job['id'], 'answer': bool(yes), 'text': txt[:120]}))
                continue
            self.handle_scene(job, res)

    # ------------------------------------------------------------ scene results
    def handle_scene(self, job, r):
        p = self.p
        cell = job['cell']
        pose = job['pose']
        self.dbg_pub.publish(C.jmsg({'cell': list(cell), 'scene': r['scene'], 'conf': r['conf'],
                                     'objects': r['objects'], 'hazards': r['hazards'],
                                     'survivor': r['survivor'], 'desc': r['desc']}))
        if r['scene'] and r['conf'] >= p['scene_conf_min']:
            d = self.scene_by_cell.setdefault(cell, {})
            d[r['scene']] = d.get(r['scene'], 0.0) + r['conf']
            self.scene_hist.append((time.monotonic(), r['scene'], r['conf']))
            self.refresh_scene_for_cell(cell)
        scene = self.scene_now(cell)
        pos = C.snap_to_cell_centre(*job['view'])
        # --- objects (M01)
        cfg = self.mission.get('M01', {})
        want = cfg.get('label', '')
        for o in r['objects']:
            if want and o != want:
                continue
            key = (o, tuple(C.cell_of(*job['view'])))
            self.obj_votes[key] = self.obj_votes.get(key, 0.0) + r['conf']
            (bo, bcell), w = max(self.obj_votes.items(), key=lambda kv: kv[1])
            self.emit('M01', 'object', scene=self.scene_now(bcell), label=bo,
                      pos=C.snap_to_cell_centre(*C.cell_centre(bcell)), score=min(0.99, 0.3 + 0.1 * w),
                      src='vlm', cell=bcell, force=True)
            break
        # --- hazards from the VLM (lower priority than the local detector)
        cfg2 = self.mission.get('M02', {})
        for hzl in r['hazards']:
            if cfg2.get('label') and cfg2['label'] != hzl:
                continue
            prev = self.best.get('M02')
            if prev and prev['src'] == 'hazard':
                break
            self.emit('M02', 'hazard', scene=scene, label=hzl, pos=pos, score=0.4 * r['conf'],
                      src='vlm', cell=cell)
            break
        # --- survivors
        if r['survivor'] and r['conf'] >= p['survivor_conf_min']:
            s = self.survivors.setdefault(cell, {'first_t': time.monotonic(), 'n': 0, 'scene': '',
                                                 'conf': 0.0, 'pos': pos})
            s['n'] += 1
            if r['conf'] >= s['conf']:
                s['conf'] = r['conf']
                s['pos'] = pos
            if s['n'] == 1:
                self.evt_pub.publish(C.jmsg({'type': 'survivor_seen', 'cell': list(cell)}))
            self.assign_survivors()

    def refresh_scene_for_cell(self, cell):
        """A better scene guess for `cell` exists: re-emit reports tied to that cell."""
        for tid, rep in list(self.best.items()):
            if rep.get('cell') and tuple(rep['cell']) == tuple(cell) and tid != 'M04':
                new = self.scene_now(cell)
                if new != rep['scene_type'] and not self.mission.get(tid, {}).get('scene_type'):
                    self.emit(tid, rep['task_type'], scene=new, label=rep['label'], pos=rep['pos'],
                              message=rep['message'], score=rep['score'], src=rep['src'],
                              cell=cell, force=True)
        for c, s in self.survivors.items():
            if tuple(c) == tuple(cell):
                s['scene'] = ''     # recomputed in assign_survivors
        self.assign_survivors()

    def assign_survivors(self):
        if not self.survivors:
            return
        cands = []
        for cell, s in self.survivors.items():
            sc = self.scene_now(cell)
            cands.append({'cell': cell, 'scene': sc, 'conf': s['conf'], 't': s['first_t'],
                          'pos': s['pos']})
        cands.sort(key=lambda d: d['t'])
        hint = self.mission.get('M05', {}).get('scene_type') or self.m05_scene_hint
        m3cfg = self.mission.get('M03', {}).get('scene_type')
        m3 = m5 = None
        if self.qr_cell is not None:
            # the QR is on the wall of the room with the FIRST survivor (the one giving the clue)
            near = sorted(cands, key=lambda c: math.hypot(c['cell'][0] - self.qr_cell[0],
                                                          c['cell'][1] - self.qr_cell[1]))
            if near and math.hypot(near[0]['cell'][0] - self.qr_cell[0],
                                   near[0]['cell'][1] - self.qr_cell[1]) <= 1.5:
                m3 = near[0]
                rest = [c for c in cands if c is not m3]
                if hint:
                    hs = [c for c in rest if c['scene'] == hint]
                    m5 = max(hs, key=lambda d: d['conf']) if hs else (rest[0] if rest else None)
                else:
                    m5 = rest[0] if rest else None
                for tid, c in (('M03', m3), ('M05', m5)):
                    if c:
                        self.emit(tid, 'survivor', scene=c['scene'], label='', pos=c['pos'],
                                  score=c['conf'], src='vlm', cell=c['cell'], force=True)
                return
        if hint:
            m5s = [c for c in cands if c['scene'] == hint]
            if m5s:
                m5 = max(m5s, key=lambda d: d['conf'])
            rest = [c for c in cands if c is not m5]
            if rest:
                m3 = rest[0]
        elif m3cfg:
            m3s = [c for c in cands if c['scene'] == m3cfg]
            m3 = max(m3s, key=lambda d: d['conf']) if m3s else None
            rest = [c for c in cands if c is not m3]
            m5 = rest[0] if rest else None
        else:
            m3 = cands[0]
            m5 = cands[1] if len(cands) > 1 else None
        for tid, c in (('M03', m3), ('M05', m5)):
            if c:
                self.emit(tid, 'survivor', scene=c['scene'], label='', pos=c['pos'], score=c['conf'],
                          src='vlm', cell=c['cell'])


def main():
    rclpy.init()
    n = Perception()
    try:
        C.safe_spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
