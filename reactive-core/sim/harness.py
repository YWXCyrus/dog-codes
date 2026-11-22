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


def simulate(world, cfg, T=120.0, start=(0.0, 0.0, 0.0), dog_kwargs=None, seed=0, dt=0.05,
             resume_on_hold=False, verbose=False):
    dog = DogModel(world, *start, seed=seed, **(dog_kwargs or {}))
    lidar = Lidar(mount_xyz=cfg['mount_xyz'], mount_pitch_deg=cfg['mount_pitch_deg'], seed=seed)
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
        if k % 2 == 0:
            pts = lidar.scan(world, dog.x, dog.y, dog.base_z(), dog.yaw, dog.roll, dog.pitch)
            core.on_cloud(pts, t)
        if not started:
            gi = core.guard_input(t)
            sp = core.guard._sweep_pts(gi)
            lb = core.guard._sweep_hit(sp, 0.0, cfg['test_wz'])
            rb = core.guard._sweep_hit(sp, 0.0, -cfg['test_wz'])
            so = su.tick(t, core.feeds_ok(t), core.posture, core.settled, core.gyro[2],
                         False, turn_sign=-1 if lb else 1, turn_blocked=lb and rb)
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
            if resume_on_hold and core.driver.mode == 'HOLD':
                core.resume(t)
        if k % 10 == 0:
            res.path.append((round(dog.x, 3), round(dog.y, 3)))
            res.trace.append((round(t, 2), round(dog.x, 2), round(dog.y, 2), round(math.degrees(dog.yaw), 1),
                              round(cmd.vx, 2), round(cmd.wz, 2), round(dog.support, 2),
                              round(math.degrees(dog.pitch), 1), core.driver.mode, core.driver.phase,
                              core.stairs.kind))
        t += dt
        k += 1
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
