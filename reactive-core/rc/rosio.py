"""ROS 2 I/O for reactive-core (the only ROS code besides run.py).

Subscribes the dog's topics, feeds Core with ARRIVAL times (time.monotonic()), publishes the
guarded command to /nav_cmd_vel (TA twist_mux, priority 10 below the controller) and gait codes
to /simple_cmd. With publish=False (dry run) no publisher is even created.
"""
import json
import math
import threading
import time

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState, PointCloud2
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from rc import geometry as G
from rc.gait import CODES, AiAxis
from rc.pc2 import cloud_to_xyz

try:
    from transfer_interfaces.msg import MotionSimpleCMD
except Exception:  # pragma: no cover
    MotionSimpleCMD = None

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
TF_STATIC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)


class RosIO:

    def __init__(self, node, core, cfg, publish):
        self.node = node
        self.core = core
        self.c = cfg
        self.publish = publish
        self.lock = threading.Lock()
        self.fastlio = None
        self.mount_from_tf = False
        self.last_code_t = -1e9
        self.code_queue = []
        self.cloud_ms = 0.0
        n = node
        n.create_subscription(PointCloud2, '/livox/points', self.on_cloud, SENSOR_QOS)
        n.create_subscription(Imu, '/imu/data', self.on_imu, SENSOR_QOS)
        n.create_subscription(JointState, '/joint_states', self.on_joints, SENSOR_QOS)
        if cfg['odom_type'] == 'odometry':
            n.create_subscription(Odometry, cfg['odom_topic'], self.on_odom_odometry, SENSOR_QOS)
        else:
            n.create_subscription(PoseWithCovarianceStamped, cfg['odom_topic'], self.on_odom_pose, SENSOR_QOS)
        n.create_subscription(Odometry, '/odom', self.on_fastlio, SENSOR_QOS)
        n.create_subscription(Odometry, cfg['lio_topic'], self.on_lio, SENSOR_QOS)
        n.create_subscription(TFMessage, '/tf_static', self.on_tf_static, TF_STATIC_QOS)
        self.status_pub = n.create_publisher(String, '/rc/status', 10)
        # perception (perception/README.md): task status, events, camera answers, camera-model detections
        n.create_subscription(String, '/quad/mission_status', self.on_mission, 10)
        n.create_subscription(String, '/quad/event', self.on_qevent, 50)
        n.create_subscription(String, '/quad/vlm_answer', self.on_vlm_answer, 10)
        n.create_subscription(String, cfg.get('detections_topic', '/rc/detections'), self.on_detections, 10)
        self.ask_pub = n.create_publisher(String, '/quad/vlm_ask', 10)      # a question, never a motion
        # scene look from the doorway (rc/cellnav.py SCENE): perception's answer per cell in, the cell looked at out
        n.create_subscription(String, '/quad/scene_state', self.on_scene_state, 10)
        self.look_pub = n.create_publisher(String, cfg.get('look_cell_topic', '/rc/look_cell'), 10)
        # the snapped arena pose for perception (launch it with odom_topic:=/rc/odom): positions to report
        self.aodom_pub = n.create_publisher(Odometry, '/rc/odom', 10) if cfg.get('publish_arena_odom') else None
        self._aprev = None
        self.cmd_pub = None
        self.simple_pub = None
        if publish:
            self.cmd_pub = n.create_publisher(Twist, '/nav_cmd_vel', 10)
            if MotionSimpleCMD is None:
                raise RuntimeError('transfer_interfaces not found: source the workspace that has it')
            # the bridge subscribes TRANSIENT_LOCAL: a VOLATILE publisher is incompatible (codes dropped).
            # depth 10: in AI the axis commands (40 msgs/s) must not overwrite a gait code before delivery
            dur = (DurabilityPolicy.TRANSIENT_LOCAL if cfg['simple_cmd_qos'] == 'transient_local'
                   else DurabilityPolicy.VOLATILE)
            self.simple_pub = n.create_publisher(
                MotionSimpleCMD, '/simple_cmd',
                QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                           depth=10, durability=dur))
        self.ai = AiAxis(cfg)               # in AI the commands go out as axis commands on /simple_cmd

    # ---- inputs (executor thread) ----
    def on_cloud(self, m):
        t = time.monotonic()
        xyz = cloud_to_xyz(m)
        with self.lock:
            t0 = time.monotonic()
            self.core.on_cloud(xyz, t)
            self.cloud_ms = (time.monotonic() - t0) * 1000.0

    def on_imu(self, m):
        a, g = m.linear_acceleration, m.angular_velocity
        with self.lock:
            self.core.on_imu((a.x, a.y, a.z), (g.x, g.y, g.z), time.monotonic())

    def on_joints(self, m):
        vel = list(m.velocity) if len(m.velocity) == len(m.name) else [0.0] * len(m.name)
        with self.lock:
            self.core.on_joints(list(m.name), list(m.position), vel, time.monotonic())

    def _odom(self, p, q):
        with self.lock:
            self.core.on_odom(p.x, p.y, G.yaw_from_quat(q.x, q.y, q.z, q.w), time.monotonic())

    def on_odom_pose(self, m):
        self._odom(m.pose.pose.position, m.pose.pose.orientation)

    def on_odom_odometry(self, m):
        self._odom(m.pose.pose.position, m.pose.pose.orientation)

    def on_fastlio(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.fastlio = (p.x, p.y, G.yaw_from_quat(q.x, q.y, q.z, q.w))

    def on_lio(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        with self.lock:
            self.core.on_lio((p.x, p.y, p.z), (q.x, q.y, q.z, q.w), time.monotonic())

    def on_tf_static(self, m):
        for tf in m.transforms:
            if tf.header.frame_id.lstrip('/') == 'base_link' and tf.child_frame_id.lstrip('/') == 'livox_frame':
                tr, q = tf.transform.translation, tf.transform.rotation
                with self.lock:
                    self.core.set_mount((tr.x, tr.y, tr.z), G.pitch_from_quat(q.x, q.y, q.z, q.w))
                if not self.mount_from_tf:
                    self.node.get_logger().info('mount from /tf_static: xyz (%.3f, %.3f, %.3f) pitch %.1f deg'
                                                % (tr.x, tr.y, tr.z, math.degrees(G.pitch_from_quat(q.x, q.y, q.z, q.w))))
                self.mount_from_tf = True

    @staticmethod
    def _json(m):
        try:
            v = json.loads(m.data)
            return v if isinstance(v, dict) else None
        except (ValueError, TypeError):
            return None

    def on_mission(self, m):
        v = self._json(m)
        if v is not None:
            with self.lock:
                self.core.on_mission_status(v, time.monotonic())

    def on_qevent(self, m):
        v = self._json(m)
        if v is not None:
            with self.lock:
                self.core.on_quad_event(v, time.monotonic())

    def on_vlm_answer(self, m):
        v = self._json(m)
        if v is not None:
            with self.lock:
                self.core.on_vlm_answer(v, time.monotonic())

    def on_detections(self, m):
        v = self._json(m)
        if v is not None:
            with self.lock:
                self.core.on_detections(v.get('dets') or [], time.monotonic())

    def on_scene_state(self, m):
        v = self._json(m)
        if v is not None:
            with self.lock:
                self.core.on_scene_state(v, time.monotonic())

    def publish_asks(self, asks):
        for q in asks:
            s = String()
            s.data = json.dumps(q, separators=(',', ':'))
            self.ask_pub.publish(s)

    def publish_look(self, look):
        """/rc/look_cell (JSON {cell, pos, look, kind}): the arena cell the dog is looking at from its doorway.
        Perception takes the scene / QR / survivor / hazard cell from it while it is fresh. Never a motion."""
        if look is None:
            return
        s = String()
        s.data = json.dumps(look, separators=(',', ':'))
        self.look_pub.publish(s)

    def publish_arena_odom(self, apose):
        """nav_msgs/Odometry in frame 'arena' (booklet frame), with the speed perception's scene trigger uses."""
        if self.aodom_pub is None or apose is None:
            return
        now = time.monotonic()
        o = Odometry()
        o.header.stamp = self.node.get_clock().now().to_msg()
        o.header.frame_id = 'arena'
        o.child_frame_id = 'base_link'
        o.pose.pose.position.x, o.pose.pose.position.y = float(apose[0]), float(apose[1])
        o.pose.pose.orientation.z = math.sin(apose[2] / 2.0)
        o.pose.pose.orientation.w = math.cos(apose[2] / 2.0)
        if self._aprev is not None and now - self._aprev[0] > 1e-3:
            dt = now - self._aprev[0]
            vx = (apose[0] - self._aprev[1][0]) / dt
            vy = (apose[1] - self._aprev[1][1]) / dt
            c, s = math.cos(apose[2]), math.sin(apose[2])
            o.twist.twist.linear.x = c * vx + s * vy
            o.twist.twist.linear.y = -s * vx + c * vy
            o.twist.twist.angular.z = G.wrap(apose[2] - self._aprev[1][2]) / dt
        self._aprev = (now, apose)
        self.aodom_pub.publish(o)

    # ---- outputs (main thread) ----
    def _simple(self, code, value=0):
        if self.simple_pub is not None:
            msg = MotionSimpleCMD()
            msg.cmd_code = int(code)
            msg.size = int(value)
            msg.type = 0
            self.simple_pub.publish(msg)

    def send(self, cmd, codes=()):
        """Publish the guarded command and queue gait codes (min gap simple_cmd_gap_s). In AI the
        command goes out as axis commands on /simple_cmd (/nav_cmd_vel gets zero)."""
        self.code_queue.extend(codes)
        now = time.monotonic()
        if self.code_queue and now - self.last_code_t >= self.c['simple_cmd_gap_s']:
            code = self.code_queue.pop(0)
            self.last_code_t = now
            self._simple(code)
            self.ai.on_code(code)
            self.node.get_logger().info('%s simple_cmd 0x%08X%s' % (
                '' if self.publish else '[dry-run]', code, '  (AI: axis commands on)' if self.ai.active
                and code == CODES['enter_ai'] else ''))
        axis = self.ai.messages(cmd)
        for code, value in axis:
            self._simple(code, value)
        if self.cmd_pub is not None:
            tw = Twist()
            if not axis:
                tw.linear.x = float(max(0.0, cmd.vx))   # never reverse (also enforced by the guard)
                tw.angular.z = float(cmd.wz)
            self.cmd_pub.publish(tw)

    def zero_for(self, seconds, rate=20.0):
        from rc.guard import Cmd
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.send(Cmd())
            time.sleep(1.0 / rate)

    def publish_status(self, status):
        m = String()
        m.data = json.dumps(status, separators=(',', ':'))
        self.status_pub.publish(m)
