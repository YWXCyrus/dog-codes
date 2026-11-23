"""Sitting / standing from the knee joints (no ROS topic carries the robot state; spec section 4),
and the body height above the feet from the leg joints."""
import math

LEGS = ('LF', 'RF', 'LB', 'RB')
LEG_H_MIN, LEG_H_MAX = 0.20, 0.42


def leg_height(names, positions, cfg):
    """Height of the hips above the stance feet (m), or None if the joints are missing / implausible.

    Joints per leg: <leg>_Joint = hip X (abduction), <leg>_Joint_1 = hip Y, <leg>_Joint_2 = knee.
    Foot drop below the hip = cos(hipX) * (thigh cos(q1) + shank cos(q1 + q2)). The two largest drops
    are the stance legs while trotting (swing legs are lifted). Run 113959: 0.327 m at the default
    height, 0.272 m with the body lowered by the controller (the lidar agreed within ~2-3 cm)."""
    q = dict(zip(names, positions))
    drops = []
    for leg in LEGS:
        try:
            hx, q1, q2 = q[leg + '_Joint'], q[leg + '_Joint_1'], q[leg + '_Joint_2']
        except KeyError:
            return None
        drops.append(math.cos(hx) * (cfg['thigh_m'] * math.cos(q1) + cfg['shank_m'] * math.cos(q1 + q2)))
    drops.sort()
    h = (drops[-1] + drops[-2]) / 2.0 + cfg['leg_h_add']
    return h if LEG_H_MIN <= h <= LEG_H_MAX else None


def classify(names, positions, cfg):
    knees = set(cfg['knee_names'])
    vals = [p for n, p in zip(names, positions) if n in knees]
    if len(vals) != len(knees):
        return 'unclear'
    m = sum(vals) / len(vals)
    if m < cfg['sit_knee']:
        return 'sitting'
    if m > cfg['stand_knee']:
        return 'standing'
    return 'unclear'


class Settle:
    """True once every joint speed stayed below settle_vel for settle_s."""

    def __init__(self, cfg):
        self.c = cfg
        self.since = None

    def update(self, velocities, t):
        if velocities and max(abs(v) for v in velocities) < self.c['settle_vel']:
            if self.since is None:
                self.since = t
            return t - self.since >= self.c['settle_s']
        self.since = None
        return False
