"""Roll / pitch from the dog's IMU (/imu/data): complementary filter of accelerometer and gyro.

ROS convention: positive roll = right side down, positive pitch = nose down. At rest the
accelerometer reads the reaction to gravity: (-g sin(pitch), g sin(roll) cos(pitch), g cos...).
The dog IMU reports m/s^2; the Livox IMU reports g - both are accepted (norm < 3 => g).
ASSUMPTION: the dog IMU axes are aligned with base_link (the old gate used the same formula).
"""
import math

G = 9.81


class Attitude:

    def __init__(self, alpha=0.02):
        self.alpha = alpha          # weight of the accelerometer per update (200 Hz -> ~0.25 s)
        self.roll = 0.0
        self.pitch = 0.0
        self.ok = False             # True after the first valid accelerometer sample

    def update(self, acc, gyro, dt):
        ax, ay, az = acc
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-3:
            return self.roll, self.pitch
        k = G if n < 3.0 else 1.0
        ax, ay, az = ax * k, ay * k, az * k
        r_acc = math.atan2(ay, az)
        p_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        if not self.ok:
            self.roll, self.pitch, self.ok = r_acc, p_acc, True
            return self.roll, self.pitch
        dt = max(0.0, min(dt, 0.1))
        r_gyr = self.roll + gyro[0] * dt
        p_gyr = self.pitch + gyro[1] * dt
        a = self.alpha
        self.roll = (1.0 - a) * r_gyr + a * r_acc
        self.pitch = (1.0 - a) * p_gyr + a * p_acc
        return self.roll, self.pitch

    def tilt_deg(self):
        return max(abs(math.degrees(self.roll)), abs(math.degrees(self.pitch)))
