"""Core: everything between the sensors and the guarded command (no ROS).

Feed it with on_cloud / on_imu / on_joints / on_odom / on_lio (arrival time t from time.monotonic()
or the simulator clock), call start(t) when the operator presses Enter, then tick(t) at
cmd_rate_hz. tick() returns the GUARDED command plus any /simple_cmd codes to send.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from rc import geometry as G
from rc import lanes as L
from rc import posture as P
from rc.arena import ArenaMap
from rc.attitude import Attitude
from rc.camcues import DetectionPersist
from rc.cellnav import CellNav
from rc.driver import Driver, DriverInput, HOLD, STAIRS
from rc.localize import GridLocalizer
from rc.grid import Grid
from rc.guard import Cmd, Guard, GuardInput
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


class Core:

    def __init__(self, cfg):
        self.c = cfg
        self.mount = Mount(np.array(cfg['mount_xyz'], dtype=float), math.radians(cfg['mount_pitch_deg']))
        self.att = Attitude()
        self.det = ObstacleDetector(cfg)
        self.stairs_det = StairDetector(cfg)
        self.odom = LegOdom(cfg)
        self.lio = LioMap(cfg)                     # FAST-LIO-placed local map (replaces the old memory)
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
        # ---- grid navigation (arena map, grid snap, cell planner / navigator)
        self.amap = ArenaMap(cfg)
        self.loc = GridLocalizer(cfg)
        if cfg['nav_mode'] == 'grid':
            self.driver.nav = CellNav(cfg, self.amap)
        self.src_name = None
        self.odom_raw = None
        self.obs_t = -1e9
        self.apath = []                            # arena path (viewer)
        self.t_go = None
        self.mission = None                        # /quad/mission_status (perception)
        self.mission_t = None
        self.answers = []                          # /quad/vlm_answer since the last decision
        self.pending_asks = []                     # -> /quad/vlm_ask
        self.pending_look = None                   # -> /rc/look_cell (scene look from the doorway)
        self.scene_seen = {}                       # cell -> time of the last /quad/scene_state for it
        self.qevents = []                          # recent /quad/event (viewer)
        self.cam = {k: DetectionPersist(k, cfg['cam_det_conf'], cfg['cam_det_n'], cfg['cam_det_window_s'],
                                        cfg['cam_det_cone_deg']) for k in ('curtain', 'stairs', 'human')}
        self.cue_t = -1e9
        self.stairs_hints = {}
        self.last_snap = None
        self.cpu = {}                              # processing times (ms): cloud callback, decision tick

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
        self.odom_raw = (x, y, yaw)
        if self.c['yaw_source'] == 'gyro':
            if self.yaw_g is None:
                self.yaw_g = yaw                   # start aligned with the odom frame
            yaw = self.yaw_g
        self.odom.update(x, y, yaw, t, abs(self.last_cmd.vx))
        self.arrival['odom'] = t
        if self.started and self.src_name == 'leg':
            self.loc.feed('leg', self.odom_raw, self.heading(), self.last_cmd.vx)
            if self.loc.events:
                self.pending_events.extend({'t': round(t, 2), 'loc': e} for e in self.loc.events)
                self.loc.events = []

    def on_lio(self, pos, quat, t):
        bad = self.lio.on_pose(pos, quat, t)
        if bad is None and self.started and self.src_name == 'lio':
            self.loc.feed('lio', self.lio_base_pose(), self.heading(), self.last_cmd.vx)
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
        t_cpu = time.perf_counter()
        use_map = self.c['nav_mode'] != 'grid' or self.c['grid_use_lio_map']
        m = self.lio.points_sensor(t) if use_map else np.zeros((0, 3))
        if m.shape[0]:
            pm = level_points(m, self.mount, self.att.roll, self.att.pitch)
            sm = self.det_map.update(np.vstack([p, pm]), self.floor_z, t, stairs_mode=stairs_mode)
            self.map_extra = np.vstack([sm.obs, sm.drops])
        else:
            self.map_extra = np.zeros((0, 2))
        if use_map:
            self.lio.add(xs, t)
        self.stairs = self.stairs_det.update(p, self.floor_z, t, self.dist_from_start())
        self._grid_cloud(p, t)
        dt_ms = (time.perf_counter() - t_cpu) * 1000.0
        self.cpu['cloud'] = 0.8 * self.cpu.get('cloud', dt_ms) + 0.2 * dt_ms
        self.cpu['cloud_max'] = max(self.cpu.get('cloud_max', 0.0) * 0.98, dt_ms)

    # ---------------------------------------------------------------- grid navigation
    def lio_base_pose(self):
        """base_link pose (x, y, yaw) from FAST-LIO, in the base frame at FAST-LIO's start
        (lidar_init is the tilted sensor frame: T = T_base_lidar . T_init_lidar(t) . T_lidar_base)."""
        if not self.lio.ok or self.lio.pose_t is None:
            return None
        Rm = G.rot_y(self.mount.pitch)
        tm = np.asarray(self.mount.xyz, dtype=float)
        R = Rm @ self.lio.R @ Rm.T
        pt = Rm @ (self.lio.R @ (-Rm.T @ tm) + self.lio.p) + tm
        return float(pt[0]), float(pt[1]), math.atan2(R[1, 0], R[0, 0])

    def heading(self):
        """The dog's heading (gyro-integrated yaw with yaw_source gyro)."""
        return None if self.odom.pose is None else self.odom.pose[2]

    def _pick_src(self, t):
        want = 'leg'
        if self.c['pose_src'] == 'lio' and self.lio.usable(t) and self.lio_base_pose() is not None:
            want = 'lio'
        if want != self.src_name:
            self.pending_events.append({'t': round(t, 2), 'pose_src': want})
            self.src_name = want

    def apose(self, t=None):
        """Arena pose (x, y, yaw): booklet frame, origin start cell centre, +X start heading, +Y left."""
        if not self.started:
            return None
        return self.loc.pose(self.heading())

    def _grid_cloud(self, p, t):
        if not self.started:
            return
        c = self.c
        hd = self.heading()
        if hd is None:
            return
        on_stairs = self.driver.mode == STAIRS or self.driver.stairs_mode
        pushing = self.driver.front_ignore is not None
        level = self.att.tilt_deg() < c['snap_max_tilt_deg']
        self.last_snap = self.loc.update(p, self.floor_z, hd, t, gyro_z=self.gyro[2],
                                         tilt_deg=self.att.tilt_deg(), allowed=not on_stairs and not pushing) or self.last_snap
        if on_stairs or not level or t - self.obs_t < 1.0 / c['observe_hz'] - 1e-6:
            return
        self.obs_t = t
        ap = self.loc.pose(hd)
        if ap is None or p.shape[0] == 0:
            return
        cs, sn = math.cos(ap[2]), math.sin(ap[2])
        P_ = np.empty((p.shape[0], 3))
        P_[:, 0] = ap[0] + cs * p[:, 0] - sn * p[:, 1]
        P_[:, 1] = ap[1] + sn * p[:, 0] + cs * p[:, 1]
        P_[:, 2] = p[:, 2] - self.floor_z
        sx, sy, sz = level_points(np.zeros((1, 3)), self.mount, self.att.roll, self.att.pitch)[0]
        S = (ap[0] + cs * sx - sn * sy, ap[1] + sn * sx + cs * sy, sz - self.floor_z)
        self.amap.observe(P_, S, t)

    # ---- perception inputs (std_msgs/String JSON, see perception/README.md)
    def on_detections(self, dets, t):
        for d in self.cam.values():
            d.update(dets, t)

    def on_mission_status(self, st, t):
        self.mission, self.mission_t = st, t
        m = self.amap
        try:
            for key, scene in (st.get('scenes') or {}).items():
                i, j = (int(v) for v in str(key).split(','))
                m.set_label((i, j), 'scene', scene)
            for cl in st.get('survivor_cells') or []:
                m.set_label(tuple(cl), 'survivor', True)
            for tid, tk in (st.get('tasks') or {}).items():
                if tk.get('reported') and tk.get('pos'):
                    cl = m.cell_of(float(tk['pos'][0]), float(tk['pos'][1]))
                    lab = tk.get('label') or tk.get('type') or ''
                    m.set_label(cl, tid, lab)
            if (st.get('qr') or {}).get('read'):
                for cl in st.get('survivor_cells') or []:
                    m.set_label(tuple(cl), 'qr', (st['qr'].get('text') or '')[:40])
        except (TypeError, ValueError, AttributeError):
            pass

    def on_quad_event(self, ev, t):
        self.qevents = (self.qevents + [dict(ev, rt=round(t, 2))])[-12:]

    def on_vlm_answer(self, ans, t):
        self.answers.append(ans)
        self.pending_events.append({'t': round(t, 2), 'vlm_answer': ans})

    def on_scene_state(self, st, t):
        """/quad/scene_state (perception: one scene-server answer, with the arena cell it was for): ends the
        front look at a scene room (rc/cellnav.py SCENE)."""
        try:
            cell = tuple(int(v) for v in st.get('cell'))
        except (TypeError, ValueError, AttributeError):
            return
        if len(cell) == 2:
            self.scene_seen[cell] = t
            self.pending_events.append({'t': round(t, 2), 'scene_state': list(cell), 'scene': st.get('scene'),
                                        'survivor': st.get('survivor')})

    def _off_body(self, pts):
        """Drop points inside the dog's own outline: standing still, the lidar sees its own front legs
        (live 2026-11-26: lane 'blocked' 0.05 m ahead of base_link, the guard's front said 3 m free)."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if pts.shape[0] == 0:
            return pts
        inside = (np.abs(pts[:, 0]) <= self.c['body_x_m']) & (np.abs(pts[:, 1]) <= self.c['body_y_m'])
        return pts[~inside]

    def _nav_pts(self, t):
        mem = self._mem_body(t) if self.c['grid_use_lio_map'] else np.zeros((0, 2))
        return self._off_body(np.vstack([self.scan.obs, self.scan.drops, np.asarray(mem).reshape(-1, 2)]))

    def _stairs_cue(self, t):
        """Stairs straight ahead (body bearing, rad) from the camera model or the lidar stairs detector, or None."""
        sd = self.cam['stairs'].seen(t)
        if sd is not None and abs(sd['bearing']) < math.radians(25.0):
            return {'bearing': sd['bearing'], 'src': 'camera'}
        st = self.stairs
        if st.kind in ('up', 'unsure') and st.edge_x < 1.6 and abs(st.heading) < math.radians(25.0):
            return {'bearing': st.heading, 'src': 'lidar_' + st.kind}
        return None

    def _grid_cues(self, t, ap):
        """Camera / lidar cues into the map (2 Hz)."""
        if ap is None or t - self.cue_t < 0.5:
            return
        self.cue_t = t
        m = self.amap
        x, y, yaw = ap
        here = m.cell_of(x, y)
        cd = self.cam['curtain'].seen(t)
        if cd is not None:
            k = m.add_curtain_cue(x, y, yaw + cd['bearing'], t)
            if k is not None and m.edge(k).cue == self.c['cur_cue_min']:
                self.pending_events.append({'t': round(t, 2), 'curtain_cue': list(k)})
        st = self.stairs
        if st.kind == 'up' and st.confident and st.edge_x < 2.5:
            h = yaw + st.heading
            d = st.edge_x + 0.25
            cl = m.cell_of(x + d * math.cos(h), y + d * math.sin(h))
            if cl != here and not m.visited(cl) and m.mark_stairs(cl, 'lidar'):
                self.pending_events.append({'t': round(t, 2), 'stairs_cell': list(cl), 'src': 'lidar'})
        sd = self.cam['stairs'].seen(t)
        if sd is not None:                     # camera-model stairs: a hint for the viewer; the navigator uses
            h = yaw + sd['bearing']            # it (NavIn.stairs_cue) when a lane into a cell is blocked
            cl = m.cell_of(x + 1.2 * math.cos(h), y + 1.2 * math.sin(h))
            self.stairs_hints[cl] = self.stairs_hints.get(cl, 0) + 1

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
                          front_ignore=self.driver.front_ignore, reverse_ok=self.driver.reverse_ok)

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
        self.t_go = t
        self.loc.start(self.heading() if self.heading() is not None else 0.0)
        self.src_name = None
        self._pick_src(t)
        self.loc.feed(self.src_name, self.lio_base_pose() if self.src_name == 'lio' else self.odom_raw,
                      self.heading())
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
        ap = None
        if t - self.last_drive_t >= 1.0 / self.c['rate_hz'] - 1e-6:
            self._pick_src(t)
            ap = self.apose(t)
            if ap is not None:
                if self.driver.mode != STAIRS or self.driver.phase in ('LOOK', 'APPROACH'):
                    new = self.amap.on_pose(ap[0], ap[1], t)
                    if new is not None:
                        events.append({'t': round(t, 2), 'arena_cell': list(new),
                                       'pos': [round(new[0] * self.amap.L, 2), round(new[1] * self.amap.L, 2)]})
                elif self.driver.phase in ('CLIMB', 'DESCEND'):
                    cl = self.amap.cell_of(ap[0], ap[1])
                    if not self.driver.stair_cells or self.driver.stair_cells[-1] != cl:
                        self.driver.stair_cells.append(cl)
                events.extend(self.amap.events)
                self.amap.events = []
                self._grid_cues(t, ap)
                if not self.apath or math.hypot(self.apath[-1][0] - ap[0], self.apath[-1][1] - ap[1]) > 0.05:
                    self.apath.append((round(ap[0], 3), round(ap[1], 3)))
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
                             apose=ap, nav_pts=self._nav_pts(t),
                             elapsed=t - (self.t_go if self.t_go is not None else t), mission=self.mission,
                             curtain_det=self.cam['curtain'].seen(t), answers=self.answers,
                             stairs_cue=self._stairs_cue(t),
                             perception=self.mission_t is not None and t - self.mission_t < 5.0,
                             scene_seen=self.scene_seen)
            self.answers = []
            t_cpu = time.perf_counter()
            out = self.driver.step(di)
            dt_ms = (time.perf_counter() - t_cpu) * 1000.0
            self.cpu['tick_max'] = max(self.cpu.get('tick_max', 0.0) * 0.98, dt_ms)
            if out.ask is not None:
                self.pending_asks.append(out.ask)
            self.pending_look = out.look
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
        return CoreOut(cmd=gcmd, codes=codes, events=events, status=self.status(t), guard=reasons)

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
            'nav': self.c['nav_mode'], 'acell': self._acell(t), 'apos': self._apos(t),
            'src': self.src_name, 'elapsed': None if self.t_go is None else round(t - self.t_go, 1),
            'visited': len(self.amap.order), 'bounds': self.amap.bounds_known(),
            'cpu': {k: round(v) for k, v in self.cpu.items()},
            # free distance straight ahead (lane_w wide) per source: live obstacles / drops / FAST-LIO map
            'ahead': [round(L.free_distance(self._off_body(self.scan.obs), 0.0, self.c['lane_w']), 2),
                      round(L.free_distance(self._off_body(self.scan.drops), 0.0, self.c['lane_w']), 2),
                      round(L.free_distance(self._off_body(self._mem_body(t)), 0.0, self.c['lane_w']), 2)],
        }

    def _acell(self, t):
        ap = self.apose(t) if self.started else None
        return None if ap is None else list(self.amap.cell_of(ap[0], ap[1]))

    def _apos(self, t):
        ap = self.apose(t) if self.started else None
        return None if ap is None else [round(ap[0], 2), round(ap[1], 2), round(math.degrees(ap[2]), 1)]

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
