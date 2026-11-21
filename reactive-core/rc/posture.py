"""Sitting / standing from the knee joints (no ROS topic carries the robot state; spec section 4)."""


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
