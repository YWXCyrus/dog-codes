"""Shared helpers for the dog_codes nodes (pure python + numpy, ROS 2 Humble, py3.10).

Conventions
-----------
* Arena frame == odom frame. Origin = centre of the start cell, +X = initial forward,
  +Y = left. Cells are CELL=1.02 m apart.
* Internal events/status use std_msgs/String carrying JSON (no custom msg build needed).
* All subscribers to dog sensor topics use BEST_EFFORT QoS (can connect to either kind).
"""
import collections
import json
import math
import time

import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

CELL = 1.02

# ---- fixed vocabularies from the booklet (v6.3) ----------------------------------------
# booklet v6.3 section E (fixed vocabulary; a value outside these lists makes the submission invalid)
SCENE_TYPES = ['clothes_shop', 'fitness_centre', 'kitchen', 'market', 'workshop', 'clinic', 'cleaning_room',
               'furniture_shop']
OBJECT_LABELS = ['medical_equipment', 'hand_tools', 'food', 'water', 'utensils']
HAZARD_LABELS = ['high_voltage', 'radioactive', 'chemical', 'biological', 'flammable']
TASK_TYPES = ['object', 'hazard', 'survivor', 'qr']

# ---- topics ---------------------------------------------------------------------------
T_CMD = '/quad/cmd_vel'          # behaviours -> safety gate (geometry_msgs/Twist)
T_GAIT_REQ = '/quad/gait_req'    # String: gait name or "code:0x...."
T_STATUS = '/quad/status'        # gate -> all (JSON)
T_MODE = '/quad/mode'            # supervisor -> all (String)
T_LIMITS = '/quad/limits'        # supervisor -> gate (JSON)
T_EVENT = '/quad/event'          # anyone -> supervisor (JSON)
T_ARM = '/quad/arm'              # std_msgs/Bool
T_REPORT = '/quad/report'        # perception -> reporter (JSON)
T_CLOUD = '/quad/cloud'          # cloud_proc -> consumers (PointCloud2, odom frame, z=height)
T_CALIB = '/quad/calib'          # cloud_proc -> consumers (JSON)
T_EXPLORE_CTRL = '/quad/explore_ctrl'   # supervisor -> explorer (JSON)
T_EXPLORE_STATE = '/quad/explore_state'  # explorer -> all (JSON)
T_BEHAV = '/quad/behaviour'      # supervisor -> terrain/curtain (JSON commands)

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=2)
RELIABLE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, depth=10)
LATCHED_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)

# ---- math ------------------------------------------------------------------------------


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def quat_to_rpy(x, y, z, w):
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(clamp(sinp, -1.0, 1.0))
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cell_of(x, y):
    """Lattice cell index containing point (x, y)."""
    return int(round(x / CELL)), int(round(y / CELL))


def cell_centre(ij):
    return ij[0] * CELL, ij[1] * CELL


def snap_to_cell_centre(x, y):
    c = cell_centre(cell_of(x, y))
    return [round(c[0], 2), round(c[1], 2)]


# ---- JSON helpers ----------------------------------------------------------------------


def jmsg(obj):
    m = String()
    m.data = json.dumps(obj, separators=(',', ':'))
    return m


def jload(s, default=None):
    try:
        return json.loads(s)
    except Exception:
        return default


def declare(node, defaults):
    """declare_parameter for each key and return plain dict of values."""
    out = {}
    for k, v in defaults.items():
        node.declare_parameter(k, v)
        out[k] = node.get_parameter(k).value
    return out


class QrPersist:
    """Accept a 'QR located but not decodable' detection only if it is plausible (width and position possible for
    the image) AND persists: at least n detections within window_s whose centres lie within px of each other.
    A one-frame flicker on a textured surface is then ignored instead of sending the dog towards it."""

    def __init__(self, n=3, window_s=1.0, px=50.0, w_min=20.0, w_max=700.0, img_w=640.0):
        self.n, self.window_s, self.px = n, window_s, px
        self.w_min, self.w_max, self.img_w = w_min, w_max, img_w
        self.hist = collections.deque(maxlen=16)

    def plausible(self, w, cx):
        return self.w_min <= w <= self.w_max and 0.0 <= cx <= self.img_w

    def update(self, now, w, cx):
        if not self.plausible(w, cx):
            return False
        self.hist.append((now, cx))
        k = sum(1 for (t, x) in self.hist if now - t <= self.window_s and abs(x - cx) <= self.px)
        return k >= self.n


class BumpDetector:
    """'Told to walk forward but not getting anywhere' = pushing into something the lidar cannot see (the lidar is
    pitched down, so behind the dog it only looks UP: a low wall or board there is invisible). Feed it every control
    tick with the pose and the forward speed that was commanded. It fires when, over the last window_s seconds, at
    least frac_cmd of the ticks commanded vx >= min_cmd_vx, the commanded distance was at least min_cmd_dist, and the
    dog moved less than `ratio` of that. After a hit the history is cleared (so it needs a fresh window to fire again)."""

    def __init__(self, window_s=3.5, min_cmd_vx=0.10, frac_cmd=0.8, ratio=0.2, min_cmd_dist=0.35):
        self.window_s, self.min_cmd_vx, self.frac_cmd = window_s, min_cmd_vx, frac_cmd
        self.ratio, self.min_cmd_dist = ratio, min_cmd_dist
        self.hist = collections.deque()          # (t, x, y, vx_cmd)
        self.last = None                         # (cmd_dist, moved) of the last evaluation, for logging

    def reset(self):
        self.hist.clear()

    def update(self, t, x, y, vx_cmd):
        self.hist.append((t, x, y, vx_cmd))
        while self.hist and t - self.hist[0][0] > self.window_s + 0.5:
            self.hist.popleft()
        sel = [h for h in self.hist if t - h[0] <= self.window_s]
        if len(sel) < 4 or t - sel[0][0] < 0.9 * self.window_s:
            return False
        fwd = sum(1 for h in sel if h[3] >= self.min_cmd_vx)
        if fwd < self.frac_cmd * len(sel):
            return False
        cmd_dist = 0.0
        for a, b in zip(sel[:-1], sel[1:]):
            cmd_dist += max(a[3], 0.0) * (b[0] - a[0])
        moved = math.hypot(sel[-1][1] - sel[0][1], sel[-1][2] - sel[0][2])
        self.last = (cmd_dist, moved)
        if cmd_dist >= self.min_cmd_dist and moved < self.ratio * cmd_dist:
            self.hist.clear()
            return True
        return False


class OdomWatchdog:
    """Flags an odometry that claims an impossible speed (FAST-LIO divergence: real run 20 Nov 10:07, the pose went to
    284 m in 25 s while the IMU said the dog was standing still). Speed = net displacement over the last window_s
    seconds. Fires when it stays above max_speed for hold_s. The dog's real speed is <= ~0.3 m/s in explore mode."""

    def __init__(self, max_speed=0.6, hold_s=0.5, window_s=1.0, min_span_s=0.4):
        self.max_speed, self.hold_s, self.window_s, self.min_span_s = max_speed, hold_s, window_s, min_span_s
        self.hist = collections.deque()          # (t, x, y)
        self.since = None
        self.speed = 0.0

    def reset(self):
        self.hist.clear()
        self.since = None
        self.speed = 0.0

    def update(self, t, x, y):
        """Returns True while the fault condition has held for hold_s."""
        self.hist.append((t, x, y))
        while self.hist and t - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        span = t - self.hist[0][0]
        self.speed = 0.0 if span < self.min_span_s else \
            math.hypot(x - self.hist[0][1], y - self.hist[0][2]) / span
        if self.speed > self.max_speed:
            if self.since is None:
                self.since = t
            return t - self.since >= self.hold_s
        self.since = None
        return False


def rear_fence_mask(X, Y, start_xy, start_yaw, radius, margin=0.15):
    """Boolean mask (same shape as X, Y = world coordinates of grid cells): cells within `radius` of the start pose that lie
    BEHIND the start line (more than `margin` behind it, measured along the start heading)."""
    dx = X - start_xy[0]
    dy = Y - start_xy[1]
    along = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
    return (np.hypot(dx, dy) < radius) & (along < -margin)


class Rate:
    """Tiny helper to run something at most every `period` seconds."""

    def __init__(self, period):
        self.period = period
        self.t = 0.0

    def ready(self, now=None):
        now = time.monotonic() if now is None else now
        if now - self.t >= self.period:
            self.t = now
            return True
        return False


# ---- PointCloud2 helpers (numpy only, no sensor_msgs_py dependency) ---------------------

_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32,
       7: np.float32, 8: np.float64}


def cloud_to_xyz(msg):
    """PointCloud2 -> (N,3) float32 array of x,y,z (finite points only)."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32)
    off = {f.name: f for f in msg.fields}
    if not all(k in off for k in 'xyz'):
        return np.zeros((0, 3), np.float32)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = msg.point_step
    buf = buf[:n * step].reshape(n, step)
    cols = []
    for k in 'xyz':
        f = off[k]
        dt = np.dtype(_DT[f.datatype]).newbyteorder('>' if msg.is_bigendian else '<')
        sz = dt.itemsize
        col = np.ascontiguousarray(buf[:, f.offset:f.offset + sz]).view(dt).reshape(n)
        cols.append(col.astype(np.float32))
    xyz = np.stack(cols, axis=1)
    ok = np.isfinite(xyz).all(axis=1)
    return xyz[ok]


def xyz_to_cloud(xyz, frame_id, stamp):
    from sensor_msgs.msg import PointCloud2, PointField
    m = PointCloud2()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    m.height = 1
    m.width = int(xyz.shape[0])
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate('xyz')]
    m.is_bigendian = False
    m.point_step = 12
    m.row_step = 12 * m.width
    m.is_dense = True
    m.data = xyz.tobytes()
    return m


def image_to_bgr(msg):
    """sensor_msgs/Image -> BGR uint8 ndarray (rgb8/bgr8/mono8/rgba8/bgra8)."""
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ('rgb8', 'bgr8'):
        a = buf[:h * msg.step].reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        return a[:, :, ::-1].copy() if enc == 'rgb8' else a.copy()
    if enc in ('rgba8', 'bgra8'):
        a = buf[:h * msg.step].reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)[:, :, :3]
        return a[:, :, ::-1].copy() if enc == 'rgba8' else a.copy()
    if enc in ('mono8', '8uc1'):
        a = buf[:h * msg.step].reshape(h, msg.step)[:, :w]
        return np.repeat(a[:, :, None], 3, axis=2)
    raise ValueError('unsupported image encoding ' + msg.encoding)


def stamp_s(node):
    return node.get_clock().now().nanoseconds * 1e-9


def filter_obstacle_points(pts, res=0.1, lo=0.30, hi=1.6, extent=0.18, tall=0.70, rmin=0.0, origin=(0.0, 0.0)):
    """Keep only points that belong to OBSTACLE cells: a res-sized column of points that reaches above
    `lo` and has a vertical extent >= `extent` (or reaches `tall`). Raised flat floor (stair treads,
    platforms) and low risers are not obstacles. pts: (N,3) x,y,height."""
    if pts.shape[0] == 0:
        return pts
    h = pts[:, 2]
    r = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
    m = (h >= -0.15) & (h <= hi) & (r > rmin)
    p = pts[m]
    if p.shape[0] == 0:
        return p
    ci = np.floor(p[:, 0] / res).astype(np.int64)
    cj = np.floor(p[:, 1] / res).astype(np.int64)
    key = (cj + 100000) * 200000 + (ci + 100000)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    hs = p[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    hmax = np.maximum.reduceat(hs, starts)
    hmin = np.minimum.reduceat(hs, starts)
    cnt = np.diff(np.r_[starts, ks.size])
    isobs = (hmax >= lo) & (((hmax - hmin) >= extent) | (hmax >= tall)) & (cnt >= 2)
    keep_cell = np.repeat(isobs, cnt)
    return p[order][keep_cell]


def safe_spin(node, executor=None):
    """Spin forever; an exception inside a callback is logged and the node keeps running."""
    import traceback
    import rclpy
    while rclpy.ok():
        try:
            if executor is not None:
                executor.spin()
            else:
                rclpy.spin(node)
            break
        except KeyboardInterrupt:
            break
        except Exception:
            try:
                node.get_logger().error('callback exception (continuing):\n' + traceback.format_exc())
            except Exception:
                pass
