"""Closed loop: DogModel + Lidar -> Core (+ Startup) -> guarded command -> DogModel.
Simulated clock, 20 Hz ticks, lidar at 10 Hz. Returns a SimResult for the tests."""
import math
from dataclasses import dataclass, field

from rc.core import Core
from rc.startup import Startup
from sim.dogmodel import DogModel
from sim.lidar import Lidar


@dataclass
class SimResult:
    startup: str = ''
    startup_msg: str = ''
    collisions: int = 0
    reversed: bool = False
    path: list = field(default_factory=list)
    modes: list = field(default_factory=list)
    events: list = field(default_factory=list)
    codes: list = field(default_factory=list)
    final_mode: str = ''
    hold_reason: str = ''
    cells: int = 0
    max_support: float = 0.0
    end_pose: tuple = (0.0, 0.0, 0.0)
    guard_counts: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)      # every 0.5 s: t, x, y, yaw, vx, wz, support, pitch, mode, phase, stairs
    finished: bool = False
    t_end: float = 0.0


def _seg_hit(p0, p1, q0, q1):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q0, q1, p0), orient(q0, q1, p1)
    d3, d4 = orient(p0, p1, q0), orient(p0, p1, q1)
    return d1 * d2 < 0 and d3 * d4 < 0


def _visible(world, x, y, tx, ty):
    for b in world.boxes:
        if b.name != 'wall':
            continue
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        h = b.sx / 2.0
        if _seg_hit((x, y), (tx, ty), (b.cx - c * h, b.cy - s * h), (b.cx + c * h, b.cy + s * h)):
            return False
    return True


def fake_camera(world, dog, hfov_deg=70.0, rng_m=2.6):
    """Camera-model detections like rc/vision_node.py: curtains and stairs in view."""
    dets = []
    for b in world.boxes:
        cls = 'curtain' if b.name == 'curtain' else ('stairs' if b.name == 'step' else None)
        if cls is None:
            continue
        dx, dy = b.cx - dog.x, b.cy - dog.y
        d = math.hypot(dx, dy)
        br = math.atan2(dy, dx) - dog.yaw
        br = math.atan2(math.sin(br), math.cos(br))
        if d > rng_m or abs(br) > math.radians(hfov_deg / 2.0) or not _visible(world, dog.x, dog.y, b.cx, b.cy):
            continue
        dets.append({'cls': cls, 'conf': 0.8, 'bearing': round(br, 4), 'w': 0.3})
    return dets


def vlm_answer(world, dog, q, mode):
    """Perception's /quad/vlm_answer to a /quad/vlm_ask (mode truth | no | down)."""
    if mode == 'down':
        return {'id': q['id'], 'kind': q['kind'], 'answer': False, 'failed': True}
    if mode == 'no':
        return {'id': q['id'], 'kind': q['kind'], 'answer': False}
    name = 'curtain' if q['kind'] == 'curtain' else 'step'
    yes = False
    for b in world.boxes:
        if b.name != name:
            continue
        dx, dy = b.cx - dog.x, b.cy - dog.y
        br = math.atan2(dy, dx) - dog.yaw
        br = math.atan2(math.sin(br), math.cos(br))
        if math.hypot(dx, dy) < 1.4 and abs(br) < math.radians(30.0):
            yes = True
    return {'id': q['id'], 'kind': q['kind'], 'answer': yes, 'src': 'sim'}


def simulate(world, cfg, T=120.0, start=(0.0, 0.0, 0.0), dog_kwargs=None, seed=0, dt=0.05,
             resume_on_hold=False, verbose=False, camera=False, vlm='truth', lidar_kwargs=None,
             lidar_every=2, stop_on_finish=True, mission=None):
    dog = DogModel(world, *start, seed=seed, **(dog_kwargs or {}))
    lidar = Lidar(mount_xyz=cfg['mount_xyz'], mount_pitch_deg=cfg['mount_pitch_deg'], seed=seed,
                  **(lidar_kwargs or {}))
    pending = []
    core = Core(cfg)
    su = Startup(cfg)
    res = SimResult()
    t, k, cmd, started = 0.0, 0, None, False
    from rc.guard import Cmd
    cmd = Cmd()
    last_mode = None
    while t < T:
        dog.step(cmd.vx, cmd.wz, dt)
        acc, gyro = dog.imu(dt)
        core.on_imu(acc, gyro, t)
        n, p, v = dog.joints()
        core.on_joints(n, p, v, t)
        core.on_odom(*dog.leg_odom(), t)
        if k % lidar_every == 0:
            pts = lidar.scan(world, dog.x, dog.y, dog.base_z(), dog.yaw, dog.roll, dog.pitch)
            core.on_cloud(pts, t)
        if camera and k % 4 == 0:
            core.on_detections(fake_camera(world, dog), t)
        if mission is not None and k % 20 == 0:
            core.on_mission_status(mission(core, dog, t) if callable(mission) else mission, t)
        for q in core.pending_asks:
            pending.append((t + 3.0, q))
        core.pending_asks = []
        for item in [p for p in pending if p[0] <= t]:
            pending.remove(item)
            core.on_vlm_answer(vlm_answer(world, dog, item[1], vlm), t)
        if not started:
            so = su.tick(t, core.feeds_ok(t))
            cmd, _ = core.filter(so.cmd, t)
            for c in so.codes:
                dog.on_code(c, t)
            if so.status == 'failed':
                res.startup, res.startup_msg = 'failed', so.msg
                break
            if so.status == 'ready':
                res.startup = 'ready'
                core.start(t)
                started = True
        else:
            out = core.tick(t)
            cmd = out.cmd
            for c in out.codes:
                dog.on_code(c, t)
            res.events.extend(out.events)
            for r in out.guard:
                res.guard_counts[r] = res.guard_counts.get(r, 0) + 1
            m = (out.status['mode'], out.status['phase'])
            if m != last_mode:
                res.modes.append((round(t, 2),) + m)
                if verbose:
                    print('%6.2f %s %s %s' % (t, m[0], m[1], out.status['hold']))
                last_mode = m
            if resume_on_hold and core.driver.mode == 'HOLD' and core.driver.hold_reason != 'finish':
                core.resume(t)
            if stop_on_finish and core.driver.hold_reason == 'finish':
                res.finished = True
                break
        if k % 10 == 0:
            res.path.append((round(dog.x, 3), round(dog.y, 3)))
            res.trace.append((round(t, 2), round(dog.x, 2), round(dog.y, 2), round(math.degrees(dog.yaw), 1),
                              round(cmd.vx, 2), round(cmd.wz, 2), round(dog.support, 2),
                              round(math.degrees(dog.pitch), 1), core.driver.mode, core.driver.phase,
                              core.stairs.kind))
        t += dt
        k += 1
    res.t_end = t
    res.collisions = dog.collisions
    res.reversed = dog.reversed
    res.codes = dog.code_log
    res.final_mode = core.driver.mode
    res.hold_reason = core.driver.hold_reason
    res.cells = len(core.grid.visited)
    res.max_support = dog.max_support
    res.end_pose = (dog.x, dog.y, dog.yaw)
    res.core = core
    res.dog = dog
    return res
