import math

import numpy as np
import pytest

from rc import config
from rc import geometry as G
from rc.sensing import Mount, ObstacleDetector, calibrate_floor, filter_sensor, level_points
from tests import helpers as H

CFG = config.load()
MOUNT = Mount(np.array([0.18678, 0.0, 0.12498]), math.radians(15))


def to_sensor(p_level, roll, pitch, mount=MOUNT):
    """Inverse of level_points: levelled body frame -> sensor frame."""
    R = G.rot_y(pitch) @ G.rot_x(roll)
    p_body = p_level @ R            # R^T applied to row vectors
    return (p_body - mount.xyz) @ G.rot_y(mount.pitch)


def det(stairs_mode=False, frames=5, pts=None):
    d = ObstacleDetector(CFG)
    s = None
    for i in range(frames):
        s = d.update(pts, H.FLOOR, float(i) * 0.1, stairs_mode=stairs_mode)
    return s


def ahead(cells, x0, x1, y=0.35):
    if cells.shape[0] == 0:
        return cells
    m = (cells[:, 0] > x0) & (cells[:, 0] < x1) & (np.abs(cells[:, 1]) < y)
    return cells[m]


def test_level_points_recovers_floor_when_pitched():
    pts = H.floor()
    roll, pitch = math.radians(3), math.radians(10)
    back = level_points(to_sensor(pts, roll, pitch), MOUNT, roll, pitch)
    assert np.allclose(back, pts, atol=1e-6)


def test_mount_pitch_points_beams_down():
    # a return 2 m straight along the sensor's x axis is below the sensor (mount pitched down)
    p = level_points(np.array([[2.0, 0.0, 0.0]]), MOUNT, 0.0, 0.0)
    assert p[0, 2] < MOUNT.xyz[2] - 0.4


def test_filter_sensor_removes_zero_range_blob_and_far():
    blob = np.zeros((2000, 3))
    far = np.array([[7.0, 0.0, 0.0]])
    ok = np.array([[2.0, 0.5, -0.3]])
    out = filter_sensor(np.vstack([blob, far, ok, [[np.nan, 0, 0]]]), 0.30, 6.0)
    assert out.shape == (1, 3)


def test_close_wall_without_floor_is_obstacle():
    # floor only visible from 1.4 m, and a wall at 0.55 m is only seen from ~0.31 m above the floor
    # (blind zone). Measured against its own lowest points the wall would be only ~0.19 m "high".
    pts = np.vstack([H.floor(), H.wall_x(0.55, -0.6, 0.6, H.FLOOR + 0.31, H.FLOOR + 0.50)])
    s = det(pts=pts)
    assert ahead(s.obs, 0.45, 0.65).shape[0] > 0
    assert ahead(s.obs_now, 0.45, 0.65).shape[0] > 0


def test_low_board_is_ground():
    pts = np.vstack([H.floor(), H.box_top(1.5, 1.8, -0.5, 0.5, 0.15)])
    s = det(pts=pts)
    assert ahead(s.obs, 1.3, 2.0).shape[0] == 0


def test_box_above_step_limit_is_obstacle():
    pts = np.vstack([H.floor(), H.box_top(2.0, 2.3, -0.3, 0.3, 0.30)])
    s = det(pts=pts)
    assert ahead(s.obs, 1.9, 2.4).shape[0] > 0


def test_drop_detected_only_in_normal_mode():
    near = H.floor(1.4, 2.0)
    low = H.floor(2.05, 3.0, z=H.FLOOR - 0.30)
    pts = np.vstack([near, low])
    s = det(pts=pts)
    assert s.drops.shape[0] > 0
    assert s.drops[:, 0].min() > 1.9
    s2 = det(pts=pts, stairs_mode=True)
    assert s2.drops.shape[0] == 0


def test_stairs_obstacle_in_normal_not_in_stairs_mode():
    st = H.stairs(1.6, 2, 0.15, 0.30)
    back_wall = H.wall_x(2.6, -0.6, 0.6, H.FLOOR + 0.30, H.FLOOR + 1.1)
    pts = np.vstack([H.floor(1.4, 1.59), st, back_wall])
    s = det(pts=pts)
    assert ahead(s.obs, 1.85, 2.25).shape[0] > 0          # second tread is 0.30 above the stance floor
    s2 = det(pts=pts, stairs_mode=True)
    assert ahead(s2.obs, 1.55, 2.25).shape[0] == 0         # treads are ground in stairs mode
    assert ahead(s2.obs, 2.5, 2.7).shape[0] > 0            # the wall is still a wall


def test_ceiling_ignored():
    pts = np.vstack([H.floor(), H.grid_pts(0.5, 3.0, -1.0, 1.0, 2.0)])
    s = det(pts=pts)
    assert s.obs.shape[0] == 0


def test_persistence():
    wall = np.vstack([H.floor(), H.wall_x(1.0, -0.4, 0.4)])
    empty = H.floor()
    c35 = dict(CFG, persist_k=3, persist_n=5)
    d = ObstacleDetector(c35)
    seq = [wall, empty, wall, empty, empty]
    for i, p in enumerate(seq):
        s = d.update(p, H.FLOOR, i * 0.1)
    assert ahead(s.obs, 0.9, 1.1).shape[0] == 0            # 2 of 5
    d = ObstacleDetector(c35)
    for i, p in enumerate([wall, empty, wall, empty, wall]):
        s = d.update(p, H.FLOOR, i * 0.1)
    assert ahead(s.obs, 0.9, 1.1).shape[0] > 0             # 3 of 5


def test_default_persistence_off():
    # default 1 of 1 (2026-11-23): this scan only
    wall = np.vstack([H.floor(), H.wall_x(1.0, -0.4, 0.4)])
    d = ObstacleDetector(CFG)
    assert ahead(d.update(wall, H.FLOOR, 0.0).obs, 0.9, 1.1).shape[0] > 0
    assert ahead(d.update(H.floor(), H.FLOOR, 0.1).obs, 0.9, 1.1).shape[0] == 0


def test_calibrate_floor():
    assert calibrate_floor(H.floor()) == pytest.approx(H.FLOOR, abs=0.01)
    assert calibrate_floor(H.floor(z=-0.05)) is None
    assert calibrate_floor(np.zeros((0, 3))) is None


def test_empty_cloud():
    s = det(pts=np.zeros((0, 3)))
    assert s.obs.shape == (0, 2) and s.drops.shape == (0, 2)


def test_scan_reports_ground_and_high_points_of_this_scan():
    fl, box = H.floor(), H.box_top(2.0, 2.3, -0.3, 0.3, 0.30)
    s = det(pts=np.vstack([fl, box]), frames=1)
    high = box[box[:, 2] - H.FLOOR > CFG['step_limit']]               # box sides start at the floor
    assert s.high_now.shape[0] == high.shape[0]
    assert s.ground_now.shape[0] == fl.shape[0] + box.shape[0] - high.shape[0]
    assert np.all(s.high_now[:, 0] > 1.9)
