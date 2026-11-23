"""Core: everything between the sensors and the guarded command (no ROS).

Feed it with on_cloud / on_imu / on_joints / on_odom / on_lio (arrival time t from time.monotonic()
or the simulator clock), call start(t) when the operator presses Enter, then tick(t) at
cmd_rate_hz. tick() returns the GUARDED command plus any /simple_cmd codes to send.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from rc import geometry as G
from rc import lanes as L
from rc import posture as P
from rc.attitude import Attitude
from rc.driver import Driver, DriverInput, HOLD
from rc.grid import Grid
from rc.guard import Cmd, Guard, GuardInput
from rc.curtain import CurtainDetector
from rc.liomap import LioMap
from rc.odom import LegOdom
from rc.sensing import Mount, ObstacleDetector, Scan, filter_sensor, level_points
from rc.stairs import StairDetector, StairsResult

FEEDS = ('livox', 'imu', 'joints', 'odom')


@dataclass
class CoreOut:
    cmd: Cmd
    codes: list = field(default_factory=list)
    events: list = field(default_factory=list)
    status: dict = field(default_factory=dict)
    guard: list = field(default_factory=list)
    asks: list = field(default_factory=list)       # camera questions for /quad/vlm_ask (curtain)


class Core:

    def __init__(self, cfg):
        self.c = cfg
        self.mount = Mount(np.array(cfg['mount_xyz'], dtype=float), math.radians(cfg['mount_pitch_deg']))
        self.att = Attitude()
        self.det = ObstacleDetector(cfg)
        self.stairs_det = StairDetector(cfg)
        self.odom = LegOdom(cfg)
        self.lio = LioMap(cfg)                     # FAST-LIO-placed local map (replaces the old memory)
        self.curtain_det = CurtainDetector(cfg)    # cloth-like surface ahead (rc/curtain.py)
        self.curtain_surf = None
        self.det_map = ObstacleDetector(cfg)       # obstacle rule on live scan + map points
        self.map_extra = np.zeros((0, 2))          # map obstacle + drop cells (body frame) for the lanes
        self.grid = Grid(cfg['cell_m'])
        self.guard = Guard(cfg, sensor_x=float(self.mount.xyz[0]))
        self.driver = Driver(cfg, self.grid)
        self.floor_z = -cfg['body_h_default']      # until the legs give a height (floor_src: legs)
        self.floor_src = 'default'
        self.joints_t = None
        self.yaw_g = None
        self.arrival = {k: None for k in FEEDS}
        self.scan = Scan(floor_z=self.floor_z)
        self.stairs = StairsResult()
        self.gyro = (0.0, 0.0, 0.0)
        self.imu_t = None
        self.joints = ([], [], [])
        self.posture = 'unclear'
        self.settle = P.Settle(cfg)
        self.settled = False
        self.started = False
        self.start_pose = None
        self.last_cmd = Cmd()
        self.last_drive_t = -1e9
        self.drive_cmd = Cmd()
        self.drive_out = None
        self.odom_was_ok = True
        self.pending_codes = []
        self.pending_events = []
        self.path = []
        self.last_guard = []

    # ---------------------------------------------------------------- inputs
    def set_mount(self, xyz, pitch):
        self.mount = Mount(np.array(xyz, dtype=float), float(pitch))
        self.guard.sensor_x = float(self.mount.xyz[0])

    def on_imu(self, acc, gyro, t):
        dt = 0.005 if self.imu_t is None else t - self.imu_t
        self.imu_t = t
        self.att.update(acc, gyro, dt)
        self.gyro = tuple(gyro)
        if self.yaw_g is not None:
            # heading from the gyro (yaw rate around gravity): does not depend on /leg_odom's yaw
            # being valid while walking. Real gyro noise 0.0016 rad/s, bias ~0.0003 rad/s (session 1 bag).
            self.yaw_g = G.wrap(self.yaw_g + gyro[2] * max(0.0, min(dt, 0.1)))
        self.arrival['imu'] = t

    def on_joints(self, names, pos, vel, t):
        self.joints = (list(names), list(pos), list(vel))
        self.posture = P.classify(names, pos, self.c)
        self.settled = self.settle.update(list(vel), t)
        if self.c['floor_src'] == 'legs':
            h = P.leg_height(names, pos, self.c)
            if h is not None:                      # lying / sitting / missing joints: keep the last value
                if self.floor_src != 'legs':
                    self.floor_z, self.floor_src = -h, 'legs'
                else:                              # low-pass ~0.5 s: trotting swing legs, body wobble
                    dt = 0.0 if self.joints_t is None else max(0.0, min(t - self.joints_t, 0.5))
                    self.floor_z += (1.0 - math.exp(-dt / 0.5)) * (-h - self.floor_z)
        self.joints_t = t
        self.arrival['joints'] = t

    def on_odom(self, x, y, yaw, t):
        if self.c['yaw_source'] == 'gyro':
            if self.yaw_g is None:
                self.yaw_g = yaw                   # start aligned with the odom frame
            yaw = self.yaw_g
        self.odom.update(x, y, yaw, t, abs(self.last_cmd.vx))
        self.arrival['odom'] = t

    def on_vlm_answer(self, d, t):
        """/quad/vlm_answer from perception (camera check of a curtain)."""
        self.driver.curtain.on_answer(d)

    def on_qr_event(self, d, t):
        """A perception event from /quad/event (qr_seen / qr_decoded) for the QR approach."""
        text = self.driver.qr.on_event(d, t)
        if text is not None:
            self.pending_events.append({'t': round(t, 2), 'qr_read': text})

    def on_lio(self, pos, quat, t):
        bad = self.lio.on_pose(pos, quat, t)
        if bad is not None:
            self.map_extra = np.zeros((0, 2))
            self.pending_events.append({'t': round(t, 2), 'lio_bad': bad + ' - map off, live scan only'})

    @property
    def odom_ok(self):
        return self.odom.ok and not self.driver.odom_frozen

    def on_cloud(self, xyz_sensor, t):
        self.arrival['livox'] = t
        xs = filter_sensor(xyz_sensor, self.c['near_drop_m'], self.c['max_range_m'])
        p = level_points(xs, self.mount, self.att.roll, self.att.pitch)
        self.p_level = p
        stairs_mode = self.driver.stairs_mode
        self.scan = self.det.update(p, self.floor_z, t, stairs_mode=stairs_mode)
        # FAST-LIO map: earlier scans in the current sensor frame, levelled like this one; the same
        # obstacle rule on live + map points gives the cells for the lanes (front stop: live only)
        m = self.lio.points_sensor(t)
        if m.shape[0]:
            pm = level_points(m, self.mount, self.att.roll, self.att.pitch)
            sm = self.det_map.update(np.vstack([p, pm]), self.floor_z, t, stairs_mode=stairs_mode)
            self.map_extra = np.vstack([sm.obs, sm.drops])
        else:
            self.map_extra = np.zeros((0, 2))
        self.lio.add(xs, t)
        self.stairs = self.stairs_det.update(p, self.floor_z, t, self.dist_from_start())
        if self.c['curtain'] != 'off':
            self.curtain_surf = self.curtain_det.update(p, self.floor_z, t, self.gyro[2])

    # ---------------------------------------------------------------- helpers
    def ages(self, t):
        return {k: (math.inf if v is None else t - v) for k, v in self.arrival.items()}

    def feeds_ok(self, t, max_age=0.5):
        return {k: a <= max_age for k, a in self.ages(t).items()}

    def dist_from_start(self):
        if self.start_pose is None or self.odom.pose is None:
            return 0.0
        return math.hypot(self.odom.pose[0] - self.start_pose[0], self.odom.pose[1] - self.start_pose[1])

    def _mem_body(self, t):
        """Map obstacle / drop cells (body frame) for the lanes and the wall check; empty without FAST-LIO."""
        return self.map_extra if self.lio.usable(t) else np.zeros((0, 2))

    def guard_input(self, t):
        return GuardInput(obs_now=self.scan.obs_now, ages=self.ages(t), tilt_deg=self.att.tilt_deg(), t=t,
                          push=self.driver.curtain.push and self.driver.mode == 'CURTAIN',
                          retreat=self.driver.curtain.phase == 'RETREAT' and self.driver.mode == 'CURTAIN')

    def filter(self, cmd, t):
        """Guard a command (used for start-up and gait tests too)."""
        out, reasons = self.guard.filter(cmd, self.guard_input(t))
        self.last_cmd = out
        self.last_guard = reasons
        return out, reasons

    # ---------------------------------------------------------------- run
    def start(self, t):
        pose = self.odom.pose or (0.0, 0.0, 0.0)
        self.start_pose = pose
        self.grid.set_origin(pose)
        self.grid.update(pose)
        self.started = True
        self.pending_events.append({'t': round(t, 2), 'start': [round(v, 3) for v in pose],
                                    'floor_z': round(self.floor_z, 3)})

    def resume(self, t):
        self.driver.resume(t)

    def tick(self, t):
        if not self.started:
            return CoreOut(cmd=Cmd(), status=self.status(t))
        events = self.pending_events
        self.pending_events = []
        codes = []
        ok = self.odom_ok
        if ok != self.odom_was_ok:
            if not ok:
                self.grid.clear()
                events.append({'t': round(t, 2), 'odom_inconsistent':
                               'frozen (lidar saw motion)' if self.driver.odom_frozen else self.odom.reason})
            else:
                events.append({'t': round(t, 2), 'odom_ok': True})
            self.odom_was_ok = ok
        if t - self.last_drive_t >= 1.0 / self.c['rate_hz'] - 1e-6:
            self.last_drive_t = t
            mem = self._mem_body(t)
            lanes = L.evaluate(self.scan.obs, self.scan.drops, self.c, extra=mem)
            if not any(l.free for l in lanes) and self.c['lane_w_min'] < self.c['lane_w']:
                # tight spot (1 m start bay with the dog off-centre, narrow door): try the minimum
                # lane width before turning on the spot (no turn guard: turning there hits the walls)
                lanes = L.evaluate(self.scan.obs, self.scan.drops, self.c, width=self.c['lane_w_min'],
                                   extra=mem)
            di = DriverInput(t=t, lanes=lanes, obs=self.scan.obs, mem=mem,
                             pose=self.odom.pose or (0.0, 0.0, 0.0), odom_ok=self.odom_ok,
                             travelled=self.odom.travelled, stairs=self.stairs, gyro_z=self.gyro[2],
                             tilt_deg=self.att.tilt_deg(), pitch_deg=math.degrees(self.att.pitch),
                             last_cmd=self.last_cmd, dist_from_start=self.dist_from_start(),
                             front_dist=L.free_distance(self.scan.obs_now, 0.0, self.c['footprint_w']),
                             curtain=self.curtain_surf, curtain_confident=self.curtain_det.confident(),
                             curtain_suspect=self.curtain_det.suspect(),
                             lio_xy=(tuple(self.lio.p[:2]) if self.lio.usable(t) else None))
            out = self.driver.step(di)
            self.drive_out = out
            self.drive_cmd = out.cmd
            codes.extend(out.codes)
            events.extend(out.events)
            if self.odom.pose is not None and (not self.path or math.hypot(
                    self.path[-1][0] - self.odom.pose[0], self.path[-1][1] - self.odom.pose[1]) > 0.05):
                self.path.append(self.odom.pose[:2])
        cmd = self.drive_cmd if self.driver.mode != HOLD else Cmd()
        gcmd, reasons = self.filter(cmd, t)
        if self.guard.latched and self.driver.mode != HOLD:
            self.driver._hold(t, 'tilt_latch', keep_stairs=self.driver.mode == 'STAIRS')
        return CoreOut(cmd=gcmd, codes=codes, events=events, status=self.status(t), guard=reasons,
                       asks=self.driver.curtain.take_asks())

    # ---------------------------------------------------------------- status / viewer
    def status(self, t):
        d = self.driver
        cell = self.grid.current
        head = None
        if self.grid.origin is not None and self.odom.pose is not None:
            head = self.grid.heading_dir(self.odom.pose[2])
        front = L.free_distance(self.scan.obs_now, 0.0, self.c['footprint_w'])
        return {
            'mode': d.mode, 'phase': d.phase, 'hold': d.hold_reason,
            'cell': list(cell) if cell else None, 'dir': head,
            'front_m': round(front, 2), 'tilt': round(self.att.tilt_deg(), 1),
            'floor_m': round(-self.floor_z, 3),
            'odom_ok': self.odom_ok, 'posture': self.posture, 'lio': self.lio.status(t),
            'ages': {k: (None if math.isinf(v) else round(v, 2)) for k, v in self.ages(t).items()},
            'stairs': self.stairs.kind, 'wall_side': 'left' if d.side > 0 else 'right',
            'guard': list(self.last_guard),
        }

    def snapshot(self, t=None):
        """JSON-ready state for the viewer and snapshots.jsonl (grid metres: +x right, +y forward)."""
        import time as _time
        t = _time.monotonic() if t is None else t
        path = []
        if self.grid.origin is not None:
            path = [[round(v, 2) for v in self.grid.to_grid((p[0], p[1], 0.0))] for p in self.path[-600:]]
        dog = self.dog_pose_grid()
        return {'status': self.status(t), 'grid': self.grid.snapshot(),
                'dog': None if dog is None else [round(dog[0], 2), round(dog[1], 2), round(dog[2], 3)],
                'path': path, 'obstacles': self.obstacles_grid(),
                'stairs': {'kind': self.stairs.kind, 'edge_x': None if self.stairs.edge_x == math.inf
                           else round(self.stairs.edge_x, 2)}}

    def dog_pose_grid(self):
        if self.grid.origin is None or self.odom.pose is None:
            return None
        gx, gy = self.grid.to_grid(self.odom.pose)
        return gx, gy, self.grid.grid_yaw(self.odom.pose[2])

    def obstacles_grid(self):
        """Current persistent obstacles in grid metres (for the viewer)."""
        if self.grid.origin is None or self.odom.pose is None or self.scan.obs.shape[0] == 0:
            return []
        w = G.body_to_world(self.scan.obs, self.odom.pose)
        return [list(map(lambda v: round(v, 2), self.grid.to_grid((p[0], p[1], 0.0)))) for p in w[:400]]
