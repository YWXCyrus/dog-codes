#!/usr/bin/env python3
"""reactive-core: one script for the Lite3 (spec docs/specs/2026-11-20-reactive-nav-design.md).

  python3 run.py                 DRY RUN (default): everything runs, nothing is sent to the dog
  python3 run.py --live          the real dog (needs ROS_DOMAIN_ID=1 and the TA bring-up running)
  python3 run.py --sim           publishes, but only on a ROS_DOMAIN_ID other than 1 (fake dog tests)

  --set key=value                override a parameter from config.yaml (repeatable)
  --gait-test ai_basic           only: start-up, Enter AI -> AI Basic, 1 s turn, 1 s forward, Exit AI
  --no-record / --record-camera  run recording (bag) off / with the camera
  --no-viewer                    no web viewer

Start-up: feeds (all four topics arriving) -> "Press Enter to go". Nothing is sent during
start-up: stand the dog up and choose the gait with the controller first. Enter during a HOLD resumes. Ctrl+C: zero speed for 1 s, exit, the dog
stays standing. An e-stop is never recovered: if the dog's feed is lost the script stops and exits.
"""
import argparse
import datetime
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rc import config  # noqa: E402

RECORD_TOPICS = ['/livox/points', '/leg_odom', '/leg_odom2', '/odom', '/imu/data', '/joint_states',
                 '/nav_cmd_vel', '/cmd_vel', '/simple_cmd', '/tf_static', '/rc/status']
CAMERA_TOPIC = '/camera/color/image_raw'
FEED_LOST_EXIT_S = 5.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='reactive-core for the Lite3 (default: dry run)')
    ap.add_argument('--live', action='store_true', help='command the real dog (ROS_DOMAIN_ID=1)')
    ap.add_argument('--sim', action='store_true', help='publish on a non-1 domain (fake dog)')
    ap.add_argument('--config', default=None)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--gait-test', choices=['ai_basic'], default=None)
    ap.add_argument('--no-record', action='store_true')
    ap.add_argument('--record-camera', action='store_true')
    ap.add_argument('--no-viewer', action='store_true')
    ap.add_argument('--auto-go', action='store_true', help='(only with --sim) do not wait for Enter')
    ap.add_argument('--runs-dir', default=os.path.join(HERE, 'runs'))
    return ap.parse_args(argv)


def safety_check(args, env):
    """-> (ok, exit_code, message_or_mode). Pure function (tested)."""
    dom = str(env.get('ROS_DOMAIN_ID', '0')).strip() or '0'
    if args.live and args.sim:
        return False, 2, 'choose either --live or --sim'
    if args.live and dom != '1':
        return False, 2, '--live needs ROS_DOMAIN_ID=1 (the dog); it is %s' % dom
    if args.sim and dom == '1':
        return False, 2, '--sim must not run on ROS_DOMAIN_ID=1 (that is the dog)'
    if args.auto_go and not args.sim:
        return False, 2, '--auto-go is only allowed with --sim'
    if args.gait_test and not (args.live or args.sim):
        return False, 2, '--gait-test needs --live (or --sim)'
    return True, 0, 'live' if args.live else 'sim' if args.sim else 'dry-run'


class Console:
    """Reads lines from stdin in a thread (Enter = go / resume)."""

    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for line in sys.stdin:
            self.q.put(line.strip())

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out


def start_recorder(run_dir, args):
    topics = RECORD_TOPICS + ([CAMERA_TOPIC] if args.record_camera else [])
    log = open(os.path.join(run_dir, 'recorder.log'), 'w')
    try:
        p = subprocess.Popen(['ros2', 'bag', 'record', '-o', os.path.join(run_dir, 'bag')] + topics,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except FileNotFoundError:
        print('WARNING: ros2 not found - no recording')
        return None
    return p


def stop_recorder(p):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGINT)
        p.wait(timeout=8)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def fmt_status(t0, st):
    cell = 'cell (%d,%d) %s' % (st['cell'][0], st['cell'][1], st['dir'] or '-') if st['cell'] else 'cell -'
    mode = st['mode'] + (' ' + st['phase'] if st['phase'] else '')
    return '[%6.1fs] %-16s %-14s front %4.2f m  tilt %4.1f  odom %s  stairs %-6s  wall %s%s' % (
        time.monotonic() - t0, mode, cell, st['front_m'], st['tilt'], 'ok' if st['odom_ok'] else 'BAD',
        st['stairs'], st['wall_side'], ('  guard ' + ','.join(st['guard'])) if st['guard'] else '')


def main(argv=None):
    args = parse_args(argv)
    ok, code, mode = safety_check(args, os.environ)
    if not ok:
        print('REFUSED: ' + mode)
        return code
    cfg = config.load(args.config, args.set)
    publish = args.live or args.sim
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.runs_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config_used.json'), 'w') as f:
        json.dump({'mode': mode, 'changed': config.diff(cfg), 'all': cfg}, f, indent=1)

    print('=' * 78)
    print('reactive-core  mode: %s   ROS_DOMAIN_ID=%s   run folder: %s'
          % (mode.upper(), os.environ.get('ROS_DOMAIN_ID', '0'), run_dir))
    print('changed parameters: %s' % (config.diff(cfg) or 'none'))
    if not publish:
        print('DRY RUN: nothing is published to the dog.')
    print('=' * 78)

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rc.core import Core
    from rc.gait import ENTER_AI, EXIT_AI, GaitSequencer
    from rc.guard import Cmd
    from rc.rosio import RosIO
    from rc.startup import Startup

    rclpy.init()
    node = rclpy.create_node('reactive_core')
    core = Core(cfg)
    io = RosIO(node, core, cfg, publish)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    console = Console()
    rec = None
    viewer = None
    dec = open(os.path.join(run_dir, 'decisions.jsonl'), 'a')
    snaps = open(os.path.join(run_dir, 'snapshots.jsonl'), 'a')
    t0 = time.monotonic()
    dt = 1.0 / cfg['cmd_rate_hz']
    rc_exit = 0

    def log(obj):
        obj = dict(obj)
        obj.setdefault('t', round(time.monotonic() - t0, 2))
        dec.write(json.dumps(obj, separators=(',', ':')) + '\n')
        dec.flush()

    try:
        if args.live:
            end = time.monotonic() + 3.0
            while node.count_subscribers('/nav_cmd_vel') == 0 and time.monotonic() < end:
                time.sleep(0.1)
            if node.count_subscribers('/nav_cmd_vel') == 0:
                print('REFUSED: nobody subscribes /nav_cmd_vel - start the TA bring-up first:\n'
                      '  ros2 launch lite3_launcher lite3.launch.py navigation:=False')
                return 3
        if cfg['record'] and not args.no_record:
            rec = start_recorder(run_dir, args)
            print('recording to %s/bag' % run_dir)
        from viewer.layers import build_registry
        registry = build_registry(core)
        if cfg['viewer'] and not args.no_viewer:
            try:
                from viewer.server import start_viewer
                viewer = start_viewer(registry, io.lock, cfg['viewer_port'])
                print('viewer: http://<container-ip>:%d  (e.g. http://192.168.10.72:%d)'
                      % (cfg['viewer_port'], cfg['viewer_port']))
            except Exception as e:
                print('viewer not started: %s' % e)

        # ---------------- start-up ----------------
        su = Startup(cfg, dry_run=not publish)
        last_msg = None
        while True:
            t = time.monotonic()
            with io.lock:
                so = su.tick(t, core.feeds_ok(t))
                cmd, _ = core.filter(so.cmd, t)
            io.send(cmd, so.codes)
            if so.msg != last_msg:
                print('start-up: ' + so.msg)
                log({'startup': so.msg})
                last_msg = so.msg
            if so.status == 'failed':
                print('START-UP FAILED: %s' % so.msg)
                return 4
            if so.status == 'ready':
                break
            time.sleep(dt)
        print('floor %.3f m below base_link (fixed, body_h_default)' % -core.floor_z)

        # ---------------- gait test only ----------------
        if args.gait_test:
            return gait_test(core, io, cfg, GaitSequencer, ENTER_AI, EXIT_AI, Cmd, log)

        # ---------------- wait for Enter ----------------
        if not args.auto_go:
            console.poll()                        # drop any Enter typed during start-up
            print('\nReady. Press Enter to go (Ctrl+C aborts).')
            while not console.poll():
                time.sleep(0.05)
        t_go = time.monotonic()
        with io.lock:
            core.start(t_go)
        print('GO')
        log({'go': True})

        # ---------------- run ----------------
        last_line, last_status_t, last_snap_t, lost_since, last_mode = '', 0.0, 0.0, None, None
        while True:
            t = time.monotonic()
            lines = console.poll()
            with io.lock:
                if lines and core.driver.mode == 'HOLD':
                    core.resume(t)
                    print('resumed by the operator')
                out = core.tick(t)
                st = out.status
                ages = core.ages(t)
                posture = core.posture
            io.send(out.cmd, out.codes)
            for e in out.events:
                e = dict(e, t=round(e.get('t', t) - t0, 2))      # core times are monotonic: make them run time
                log(e)
                if 'hold' in e:
                    print('\n*** HOLD: %s - the Captain may drive by controller; press Enter to resume ***\n' % e['hold'])
                elif 'cell' not in e:
                    print('event: %s' % json.dumps(e))
            if out.guard:
                log({'guard': out.guard, 'cmd': [round(out.cmd.vx, 3), round(out.cmd.wz, 3)]})
            mode_now = (st['mode'], st['phase'], tuple(st['cell'] or ()))
            every = 10.0 if st['mode'] == 'HOLD' else 2.0
            if mode_now != last_mode or t - last_status_t > every:
                line = fmt_status(t0, st)
                if line != last_line or t - last_status_t > every:
                    print(line)
                    last_line, last_status_t = line, t
                last_mode = mode_now
            if t - last_snap_t >= 1.0:
                last_snap_t = t
                with io.lock:
                    snap = registry.snapshot()        # same format as the live viewer: replayable
                snaps.write(json.dumps(snap, separators=(',', ':')) + '\n')
                snaps.flush()
                io.publish_status(st)
            stale = [k for k, a in ages.items() if a > 2.0]
            # after a STOP the dog collapses (knees like sitting) or its feed dies. 'unclear' is NOT
            # used here: on the stairs the knee angles leave the standing band legitimately.
            if stale or posture == 'sitting':
                lost_since = lost_since or t
                if t - lost_since > FEED_LOST_EXIT_S:
                    print('\nFEED LOST / DOG DOWN for %.0f s (%s) - STOP pressed? The code never '
                          'recovers after an e-stop. Exiting.' % (FEED_LOST_EXIT_S, ', '.join(stale) or 'posture sitting'))
                    log({'exit': 'feed_lost', 'stale': stale, 'posture': posture})
                    rc_exit = 5
                    break
            else:
                lost_since = None
            time.sleep(max(0.0, dt - (time.monotonic() - t)))
    except KeyboardInterrupt:
        print('\nCtrl+C: zero speed, stopping (the dog stays standing)')
    finally:
        if publish:
            try:
                io.zero_for(1.0, cfg['cmd_rate_hz'])
            except Exception:
                pass
        stop_recorder(rec)
        if viewer is not None:
            try:
                viewer.shutdown()
            except Exception:
                pass
        dec.close()
        snaps.close()
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except Exception:
            pass
        print('run saved in %s' % run_dir)
    return rc_exit


def gait_test(core, io, cfg, GaitSequencer, ENTER_AI, EXIT_AI, Cmd, log):
    """Enter AI -> AI Basic -> turn test -> 1 s forward 0.1 m/s -> Exit AI -> RUG general -> turn test."""
    dt = 1.0 / cfg['cmd_rate_hz']

    def run_seq(name, steps):
        seq = GaitSequencer(steps, cfg)
        gyro = []
        while True:
            t = time.monotonic()
            with io.lock:
                st, cmd, code = seq.tick(t, core.gyro[2])
                gcmd, _ = core.filter(cmd, t)
                gyro.append(core.gyro[2])
            io.send(gcmd, [code] if code is not None else [])
            if st != 'running':
                print('%s: %s %s (max |gyro z| %.2f rad/s)' % (name, st.upper(), seq.failed_reason,
                                                               max(abs(g) for g in gyro)))
                log({'gait_test': name, 'status': st, 'reason': seq.failed_reason})
                return st == 'done'
            time.sleep(dt)

    if not run_seq('enter AI Basic', ENTER_AI):
        run_seq('exit AI', EXIT_AI)
        return 6
    with io.lock:
        d0 = core.odom.travelled
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        t = time.monotonic()
        with io.lock:
            gcmd, reasons = core.filter(Cmd(0.10, 0.0), t)
        io.send(gcmd)
        time.sleep(dt)
    io.zero_for(0.5)
    with io.lock:
        moved = core.odom.travelled - d0
    print('forward 1 s at 0.10 m/s in AI Basic: leg odometry moved %.3f m' % moved)
    log({'gait_test': 'forward', 'moved_m': round(moved, 3)})
    ok = run_seq('exit AI -> RUG general', EXIT_AI)
    return 0 if ok else 7


if __name__ == '__main__':
    sys.exit(main())
