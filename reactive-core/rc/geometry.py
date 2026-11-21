"""Small geometry helpers. Frames follow ROS: x forward, y left, z up; positive pitch = nose down."""
import math

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    """Angle to (-pi, pi]."""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a == -math.pi else a


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pitch_from_quat(x, y, z, w):
    s = 2.0 * (w * y - z * x)
    return math.asin(max(-1.0, min(1.0, s)))


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def footprint_corners(length, width, offset_x=0.0):
    """Rectangle corners (4,2) in the body frame, centred at (offset_x, 0)."""
    hl, hw = length / 2.0, width / 2.0
    return np.array([[offset_x + hl, hw], [offset_x + hl, -hw],
                     [offset_x - hl, -hw], [offset_x - hl, hw]])


def points_in_rect(pts_xy, cx, cy, yaw, length, width):
    """Mask of points inside a rectangle centred at (cx, cy) rotated by yaw."""
    pts_xy = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
    if pts_xy.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    c, s = math.cos(yaw), math.sin(yaw)
    dx = pts_xy[:, 0] - cx
    dy = pts_xy[:, 1] - cy
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return (np.abs(u) <= length / 2.0) & (np.abs(v) <= width / 2.0)


def world_to_body(pts_xy, pose):
    """2-D points from the odom frame into the body frame of pose (x, y, yaw)."""
    pts_xy = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx = pts_xy[:, 0] - x
    dy = pts_xy[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def body_to_world(pts_xy, pose):
    pts_xy = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * pts_xy[:, 0] - s * pts_xy[:, 1],
                     y + s * pts_xy[:, 0] + c * pts_xy[:, 1]], axis=1)
