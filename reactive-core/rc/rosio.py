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
        n.create_subscription(String, cfg['detections_topic'], self.on_detections, 10)   # rc/vision_node.py
        self.status_pub = n.create_publisher(String, '/rc/status', 10)
        # nav_mode cells: position in the arena frame (booklet E) for perception (odom_topic:=/rc/arena_odom)
        # and perception's scene results (a room look ends when its cell has been analysed)
        self.arena_pub = n.create_publisher(Odometry, '/rc/arena_odom', 10)
        n.create_subscription(String, '/quad/scene_state', self.on_scene_state, 10)
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

    def on_detections(self, m):
        try:
            dets = json.loads(m.data).get('dets', [])
        except Exception:
            return
        with self.lock:
            self.core.on_vision(dets, time.monotonic())

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

    def on_scene_state(self, m):
        try:
            cell = json.loads(m.data).get('cell')
        except Exception:
            return
        if isinstance(cell, (list, tuple)) and len(cell) == 2:
            with self.lock:
                self.core.on_scene_state(cell, time.monotonic())

    def publish_arena(self, pose):
        """pose = (x, y, yaw, speed) arena frame (core.arena_pose()); information only, also in dry runs."""
        if pose is None:
            return
        x, y, yaw, v = pose
        m = Odometry()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = 'arena'
        m.child_frame_id = 'base_link'
        m.pose.pose.position.x = float(x)
        m.pose.pose.position.y = float(y)
        m.pose.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.pose.orientation.w = math.cos(yaw / 2.0)
        m.twist.twist.linear.x = float(v)
        self.arena_pub.publish(m)

    def publish_status(self, status):
        m = String()
        m.data = json.dumps(status, separators=(',', ':'))
        self.status_pub.publish(m)
