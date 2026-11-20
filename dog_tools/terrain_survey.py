#!/usr/bin/env python3
"""terrain_survey: READ-ONLY survey while you drive the dog by hand.

Subscribes to /odom, /imu/data, /livox/points (+TF) and builds
  * a top-down terrain map (floor height + roughness, walls) in the odom frame,
  * a 5 Hz sample log (pose, tilt, wobble, what the dog's rough-terrain detector would say, gait label),
  * a periodic report (summary table + map PNGs) in the run folder.
The gait label comes from gait_label.py (a second terminal), through the file labels.csv.

THIS NODE CREATES NO PUBLISHER, NO SERVICE-CLIENT, NOTHING THAT CAN COMMAND THE DOG
(rosout and parameter events are switched off; it logs "publishers=0" at start-up).
"""
import csv
import json
import math
import os
import signal
import sys
import time
import traceback
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, PointCloud2
from tf2_ros import Buffer, TransformListener

import core
import report

BEST = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=3)
BEST_IMU = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)


class Survey(Node):

    def __init__(self):
        # no /rosout publisher, no /parameter_events publisher: the node publishes nothing at all
        super().__init__('terrain_survey', enable_rosout=False, start_parameter_services=False)
        defaults = {
            'run_root': '~/dog_tools_runs', 'cloud_topic': '/livox/points', 'odom_topic': '/odom',
            'imu_topic': '/imu/data', 'odom_frame': 'odom', 'sensor_frame': 'livox_frame',
            'max_range': 8.0, 'max_points': 12000, 'body_h': 0.0, 'body_h_default': 0.30,
            'sample_hz': 5.0, 'report_every_s': 20.0,
            'map_x0': -6.0, 'map_x1': 14.0, 'map_y0': -8.0, 'map_y1': 8.0, 'map_res': 0.1,
            'wall_h': 0.60,
            'rough_std': 0.050, 'rough_frac': 0.35, 'rough_on_n': 4, 'rough_off_n': 15,
        }
        self.p = {}
        for k, v in defaults.items():
            self.declare_parameter(k, v)
            self.p[k] = self.get_parameter(k).value
        p = self.p
        # rclpy always creates a /parameter_events publisher; parameters are read now, so drop it:
        # the node then has no publisher whatsoever
        try:
            self.destroy_publisher(self._parameter_event_publisher)
            self._parameter_event_publisher = None
        except Exception as e:
            self.get_logger().warn('could not remove the /parameter_events publisher (it only carries '
                                   'parameter-change notices, harmless): %s' % e)

        # ---- run folder
        root = os.path.expanduser(p['run_root'])
        self.run_dir = os.path.join(root, 'survey_' + time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(root, 'LATEST'), 'w') as f:
            f.write(self.run_dir + '\n')
        self.labels_path = os.path.join(self.run_dir, 'labels.csv')
        with open(self.labels_path, 'w', newline='') as f:
            csv.writer(f).writerow(['t_wall', 'kind', 'text'])
        with open(os.path.join(self.run_dir, 'params.json'), 'w') as f:
            json.dump(p, f, indent=1)
        self.samples_f = open(os.path.join(self.run_dir, 'samples.csv'), 'w', buffering=1, newline='')
        self.samples_w = csv.writer(self.samples_f)
        self.samples_w.writerow(core.SAMPLE_COLS)

        # ---- state
        self.t0 = time.time()
        self.grid = core.GridMap(p['map_x0'], p['map_x1'], p['map_y0'], p['map_y1'], p['map_res'],
                                 wall_h=p['wall_h'])
        self.calib = core.BodyHeightCalib(default=p['body_h_default'], fixed=p['body_h'])
        self.floor0 = None
        self.det = core.RoughDetector(p['rough_std'], p['rough_frac'], p['rough_on_n'], p['rough_off_n'])
        self.have_odom = False
        self.px = self.py = self.pz = self.yaw = 0.0
        self.roll_o = self.pitch_o = float('nan')
        self.odom_hist = deque()
        self.odom_t = 0.0
        self.imu_t = 0.0
        self.cloud_t = 0.0
        self.cloud_n = 0
        self.cb_ms = 0.0
        self.cloud_times = deque(maxlen=30)
        self.tf_ok = False
        self.imu_sum = np.zeros(3)
        self.imu_n = 0
        self.gyro_sq = 0.0
        self.last_tilt_imu = (float('nan'), float('nan'))
        self.feat = None
        self.feat_t = 0.0
        self.hit = False
        self.last_det = 0.0
        self.label = core.UNLABELLED
        self.lab_off = 0
        self.n_marks = 0
        self.n_rows = 0
        self.last_warn = {}
        self.last_map_save = time.time()
        self.finalised = False

        # ---- inputs (subscriptions only)
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)
        self.create_subscription(Odometry, p['odom_topic'], self.on_odom, BEST)
        self.create_subscription(Imu, p['imu_topic'], self.on_imu, BEST_IMU)
        self.create_subscription(PointCloud2, p['cloud_topic'], self.on_cloud, BEST)
        self.create_timer(1.0 / p['sample_hz'], self.on_sample)
        self.create_timer(0.5, self.poll_labels)
        self.create_timer(5.0, self.heartbeat)
        self.create_timer(p['report_every_s'], self.periodic_report)
        self.get_logger().info('terrain_survey up. run folder: %s' % self.run_dir)
        self.get_logger().info('publishers=%d (read-only: this node cannot command the dog); '
                               'domain=%s' % (len(list(self.publishers)), os.environ.get('ROS_DOMAIN_ID', '(unset)')))

    # ------------------------------------------------------------------ helpers
    def warn_throttled(self, key, msg, every=5.0):
        t = time.monotonic()
        if t - self.last_warn.get(key, -1e9) >= every:
            self.last_warn[key] = t
            self.get_logger().warn(msg)

    # ------------------------------------------------------------------ inputs
    def on_odom(self, m):
        pos = m.pose.pose.position
        q = m.pose.pose.orientation
        r, pch, yaw = core.quat_to_rpy(q.x, q.y, q.z, q.w)
        self.px, self.py, self.pz, self.yaw = pos.x, pos.y, pos.z, yaw
        self.roll_o, self.pitch_o = r, pch
        self.have_odom = True
        t = time.time()
        self.odom_t = t
        self.odom_hist.append((t, pos.x, pos.y))
        while self.odom_hist and t - self.odom_hist[0][0] > 1.2:
            self.odom_hist.popleft()

    def on_imu(self, m):
        a = m.linear_acceleration
        g = m.angular_velocity
        self.imu_sum += (a.x, a.y, a.z)
        self.imu_n += 1
        self.gyro_sq += g.x * g.x + g.y * g.y
        self.imu_t = time.time()

    def on_cloud(self, msg):
        t0 = time.perf_counter()
        try:
            self._on_cloud(msg)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.cb_ms = ms if self.cb_ms == 0.0 else 0.9 * self.cb_ms + 0.1 * ms

    def _on_cloud(self, msg):
        if not self.have_odom:
            return
        t = time.time()
        self.cloud_t = t
        self.cloud_n += 1
        self.cloud_times.append(t)
        try:
            try:
                tf = self.tfbuf.lookup_transform(self.p['odom_frame'], self.p['sensor_frame'],
                                                 rclpy.time.Time.from_msg(msg.header.stamp))
            except Exception:
                tf = self.tfbuf.lookup_transform(self.p['odom_frame'], self.p['sensor_frame'],
                                                 rclpy.time.Time())
            self.tf_ok = True
        except Exception as e:
            self.tf_ok = False
            self.warn_throttled('tf', 'no TF %s -> %s yet: %s' % (self.p['odom_frame'],
                                                                  self.p['sensor_frame'], e))
            return
        xyz = core.cloud_to_xyz(msg)
        if xyz.shape[0] == 0:
            return
        q, tr = tf.transform.rotation, tf.transform.translation
        R = core.quat_to_rot(q.x, q.y, q.z, q.w).astype(np.float32)
        w = xyz @ R.T + np.array([tr.x, tr.y, tr.z], np.float32)
        w = w[np.hypot(w[:, 0] - self.px, w[:, 1] - self.py) < self.p['max_range']]
        if w.shape[0] == 0:
            return
        if not self.calib.calibrated:
            self.calib.update(w, self.px, self.py, self.pz)
            if not self.calib.calibrated:
                self.warn_throttled('calib', 'waiting for body-height calibration: stand on flat floor '
                                    '(or restart with -p body_h:=0.30)')
                return
            self.get_logger().info('body height calibrated: %.3f m  -> mapping starts' % self.calib.value)
        if self.floor0 is None:                 # world z of the floor where we calibrated (fixed reference)
            self.floor0 = self.pz - self.calib.value
        # detector: height above the dog's CURRENT foot plane (as dog_codes does); map: above the fixed
        # start floor level (so a stair tread keeps the same height whether seen from below or above)
        h_loc = w[:, 2] - (self.pz - self.calib.value)
        h_map = w[:, 2] - self.floor0
        xyh = np.column_stack([w[:, 0], w[:, 1], h_loc]).astype(np.float32)
        xyh_map = np.column_stack([w[:, 0], w[:, 1], h_map]).astype(np.float32)
        if xyh.shape[0] > self.p['max_points']:
            idx = np.random.default_rng(self.cloud_n).choice(xyh.shape[0], self.p['max_points'], replace=False)
            xyh = xyh[idx]
            xyh_map = xyh_map[idx]
        self.grid.add(xyh_map)
        feat = core.rough_features(xyh, self.px, self.py, self.yaw)
        self.feat, self.feat_t = feat, t
        if t - self.last_det >= 0.2:                      # the dog's detector runs at 5 Hz
            self.last_det = t
            self.hit, on, flip = self.det.update(feat)
            if flip:
                self.get_logger().info('detector rough flag -> %s at (%.2f, %.2f) std=%s' % (
                    flip.upper(), self.px, self.py, ('%.3f' % feat[0]) if feat else '-'))

    # ------------------------------------------------------------------ labels
    def poll_labels(self):
        try:
            rows, self.lab_off = core.read_new_rows(self.labels_path, self.lab_off)
        except OSError:
            return
        for row in rows:
            kind, text = row[1], row[2]
            if kind == 'gait':
                self.label = text or core.UNLABELLED
                self.get_logger().info('GAIT LABEL -> %s' % self.label)
            elif kind == 'mark':
                self.n_marks += 1
                self.get_logger().info('MARK #%d at (%.2f, %.2f) [%s]' % (self.n_marks, self.px, self.py, self.label))
            elif kind == 'note':
                self.get_logger().info('NOTE at (%.2f, %.2f): %s' % (self.px, self.py, text))

    # ------------------------------------------------------------------ samples
    def on_sample(self):
        if not self.have_odom:
            return
        t = time.time()
        speed = 0.0
        if len(self.odom_hist) >= 2:
            a, b = self.odom_hist[0], self.odom_hist[-1]
            if b[0] - a[0] >= 0.4:
                speed = math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0])
        if self.imu_n > 0:
            m = self.imu_sum / self.imu_n
            tl = core.tilt_from_accel(m[0], m[1], m[2])
            roll_i, pitch_i = (math.degrees(tl[0]), math.degrees(tl[1])) if tl else (float('nan'),) * 2
            gyro = math.degrees(math.sqrt(self.gyro_sq / self.imu_n))
        else:
            roll_i = pitch_i = gyro = float('nan')
        self.imu_sum[:] = 0
        self.imu_n = 0
        self.gyro_sq = 0.0
        fresh = self.feat is not None and (t - self.feat_t) < 1.0
        f = self.feat if fresh else None
        row = [
            '%.3f' % t, '%.2f' % (t - self.t0), '%.3f' % self.px, '%.3f' % self.py,
            '%.1f' % math.degrees(self.yaw), '%.3f' % speed,
            '%.2f' % math.degrees(self.roll_o) if self.roll_o == self.roll_o else 'nan',
            '%.2f' % math.degrees(self.pitch_o) if self.pitch_o == self.pitch_o else 'nan',
            '%.2f' % roll_i, '%.2f' % pitch_i, '%.2f' % gyro,
            '%.4f' % f[0] if f else 'nan', '%.3f' % f[1] if f else 'nan', str(f[2]) if f else '0',
            '1' if (self.hit and fresh) else '0', '1' if self.det.on else '0', self.label]
        self.samples_w.writerow(row)
        self.n_rows += 1

    # ------------------------------------------------------------------ status / report
    def heartbeat(self):
        t = time.time()
        hz = 0.0
        if len(self.cloud_times) >= 2:
            span = self.cloud_times[-1] - self.cloud_times[0]
            hz = (len(self.cloud_times) - 1) / span if span > 0 else 0.0

        def age(x):
            return 'none' if x == 0.0 else '%.1fs ago' % (t - x)
        f = self.feat if (self.feat and t - self.feat_t < 1.0) else None
        self.get_logger().info(
            't=%3.0fs cloud=%.1fHz(%s,%.0fms) odom=%s imu=%s tf=%s calib=%s label=%s pos=(%.2f,%.2f) rough=%s '
            'on=%s map=%d cells rows=%d' % (
                t - self.t0, hz, age(self.cloud_t), self.cb_ms, age(self.odom_t), age(self.imu_t),
                'ok' if self.tf_ok else 'NO', ('%.3f' % self.calib.value) if self.calib.calibrated else 'WAIT',
                self.label, self.px, self.py, ('%.3f' % f[0]) if f else '-', int(self.det.on),
                int(self.grid.observed().sum()), self.n_rows))
        if t - self.t0 > 6.0:
            if self.odom_t == 0.0:
                self.warn_throttled('no_odom', 'NO %s received. Is terminal 1 (FAST-LIO bring-up) running and '
                                    'ROS_DOMAIN_ID right?' % self.p['odom_topic'], 15)
            if self.cloud_t == 0.0:
                self.warn_throttled('no_cloud', 'NO %s received.' % self.p['cloud_topic'], 15)
            if self.imu_t == 0.0:
                self.warn_throttled('no_imu', 'NO %s received (tilt/wobble columns will be empty).' %
                                    self.p['imu_topic'], 15)

    def save_map(self):
        arrays = self.grid.to_arrays()
        arrays['meta_json'] = np.array(json.dumps({'body_h': self.calib.value, 'run': self.run_dir}))
        path = os.path.join(self.run_dir, 'map.npz')
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)

    def periodic_report(self):
        try:
            self.save_map()
            report.build_report(self.run_dir)
        except Exception:
            self.get_logger().error('periodic report failed (continuing):\n' + traceback.format_exc())

    def finalise(self):
        if self.finalised:
            return
        self.finalised = True
        try:
            self.poll_labels()
            self.samples_f.flush()
            self.samples_f.close()
            self.save_map()
            r = report.build_report(self.run_dir)
            print('\n' + r['summary_text'])
            print('REPORT FOLDER: %s' % os.path.join(self.run_dir, 'report'))
            print('files: %s' % ', '.join(r['files']))
        except Exception:
            print('final report failed:\n' + traceback.format_exc())
            print('data is safe in %s; run: python3 report.py "%s"' % (self.run_dir, self.run_dir))


def main():
    rclpy.init()
    node = Survey()
    signal.signal(signal.SIGTERM, lambda *a: rclpy.try_shutdown())
    try:
        while rclpy.ok():
            try:
                rclpy.spin(node)
                break
            except (KeyboardInterrupt, ExternalShutdownException):     # Ctrl+C / SIGTERM: normal exit
                break
            except Exception:
                node.get_logger().error('callback exception (continuing):\n' + traceback.format_exc())
    finally:
        node.finalise()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
