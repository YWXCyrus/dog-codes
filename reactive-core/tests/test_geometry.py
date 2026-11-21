import math

import numpy as np
import pytest

from rc import geometry as G


def test_rot_y_pitch_down_convention():
    # ROS: positive rotation about y turns +x towards -z (nose down)
    v = G.rot_y(math.pi / 2) @ np.array([1.0, 0, 0])
    assert np.allclose(v, [0, 0, -1], atol=1e-9)


def test_rot_z():
    v = G.rot_z(math.pi / 2) @ np.array([1.0, 0, 0])
    assert np.allclose(v, [0, 1, 0], atol=1e-9)


def test_yaw_from_quat():
    a = math.pi / 2
    assert G.yaw_from_quat(0, 0, math.sin(a / 2), math.cos(a / 2)) == pytest.approx(a)


def test_pitch_from_quat():
    a = math.radians(15)
    assert G.pitch_from_quat(0, math.sin(a / 2), 0, math.cos(a / 2)) == pytest.approx(a)


def test_wrap():
    assert G.wrap(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert G.wrap(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    assert G.wrap(0.3) == pytest.approx(0.3)


def test_points_in_rect():
    pts = np.array([[0.30, 0.0], [0.34, 0.0], [0.0, 0.20], [0.0, 0.22]])
    m = G.points_in_rect(pts, 0.0, 0.0, 0.0, 0.65, 0.42)
    assert list(m) == [True, False, True, False]
    m90 = G.points_in_rect(np.array([[0.0, 0.30], [0.30, 0.0]]), 0.0, 0.0, math.pi / 2, 0.65, 0.42)
    assert list(m90) == [True, False]


def test_points_in_rect_empty():
    assert G.points_in_rect(np.zeros((0, 2)), 0, 0, 0, 1, 1).shape == (0,)


def test_footprint_corners():
    c = G.footprint_corners(0.6, 0.4, 0.1)
    assert c.shape == (4, 2)
    assert c[:, 0].max() == pytest.approx(0.4)
    assert c[:, 0].min() == pytest.approx(-0.2)
    assert c[:, 1].max() == pytest.approx(0.2)


def test_transform_points_2d():
    p = G.world_to_body(np.array([[1.0, 0.0]]), (1.0, 0.0, math.pi / 2))
    assert np.allclose(p, [[0.0, 0.0]], atol=1e-9)
    p = G.world_to_body(np.array([[1.0, 1.0]]), (1.0, 0.0, math.pi / 2))
    assert np.allclose(p, [[1.0, 0.0]], atol=1e-9)
    w = G.body_to_world(np.array([[1.0, 0.0]]), (1.0, 0.0, math.pi / 2))
    assert np.allclose(w, [[1.0, 1.0]], atol=1e-9)
