import math

import pytest

from rc.attitude import Attitude

G = 9.81


def run(att, acc, n=300, dt=0.005, gyro=(0.0, 0.0, 0.0)):
    for _ in range(n):
        att.update(acc, gyro, dt)
    return att


def test_level():
    a = run(Attitude(), (0.0, 0.0, G))
    assert abs(a.roll) < 1e-6 and abs(a.pitch) < 1e-6
    assert a.tilt_deg() < 1e-4


def test_pitch_nose_down_positive():
    th = math.radians(10)
    # ROS convention: nose down = positive pitch; gravity reaction seen as ax = -g sin(pitch)
    a = run(Attitude(), (-G * math.sin(th), 0.0, G * math.cos(th)))
    assert math.degrees(a.pitch) == pytest.approx(10.0, abs=0.1)


def test_roll():
    ph = math.radians(-8)
    a = run(Attitude(), (0.0, G * math.sin(ph), G * math.cos(ph)))
    assert math.degrees(a.roll) == pytest.approx(-8.0, abs=0.1)
    assert a.tilt_deg() == pytest.approx(8.0, abs=0.1)


def test_accel_in_g():
    a = run(Attitude(), (0.0, 0.0, 1.0))
    assert abs(a.pitch) < 1e-6 and a.ok


def test_gyro_short_term():
    # a short fast rotation is followed by the gyro before the accelerometer catches up
    a = Attitude()
    for _ in range(20):
        a.update((0.0, 0.0, G), (0.0, 1.0, 0.0), 0.005)   # 0.1 rad in 0.1 s, accel still says level
    assert 0.05 < a.pitch < 0.1


def test_zero_accel_ignored():
    a = Attitude()
    a.update((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.005)
    assert not a.ok
