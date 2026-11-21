# reactive-core — reactive lidar navigation for the Lite3

One script, `run.py`. Reactive wall following in RUG general on the raw lidar (no map, no FAST-LIO
for driving), an independent guard, a 1 m grid memory, stairs (up - platform - down) in AI Basic,
a read-only live map viewer, and recording of every run. Design: `docs/specs/2026-11-20-reactive-nav-design.md`.
Plain Python + numpy; nothing to build.

**Safety.** Someone holds the controller e-stop for every run on the dog. The controller always
wins over this code (twist_mux priority 100 vs 10). The code never reverses. An e-stop is never
recovered by the code: after STOP the script stops commanding and exits.

## 1. Run it on the dog

```bash
# every terminal on the container:
source /opt/ros/humble/setup.bash
source ~/lite3_ws/install/setup.bash        # transfer_interfaces (MotionSimpleCMD)
source ~/ros2_ws/install/setup.bash         # lite3_launcher (TA bring-up)
export ROS_DOMAIN_ID=1                       # the dog

# terminal 1 - TA bring-up without their Nav2 (twist_mux /nav_cmd_vel -> /cmd_vel, joystick, FAST-LIO):
ros2 launch lite3_launcher lite3.launch.py navigation:=False

# terminal 2:
cd ~/dog-codes/reactive-core
python3 run.py                      # 1) DRY RUN first: everything runs, NOTHING is sent to the dog
python3 run.py --live --set cruise_vx=0.12 --set run_time_s=120     # 2) first real run: slower, 2 min
python3 run.py --live               # normal run
```

`--live` refuses to start unless `ROS_DOMAIN_ID=1` and someone subscribes `/nav_cmd_vel` (the TA
bring-up). Auto mode must be on in the app (otherwise the start-up turn test fails and says so).

### What happens
1. Start-up (printed line by line): feeds alive (`/livox/points`, `/imu/data`, `/joint_states`,
   `/leg_odom`) -> posture from the knees: sitting -> ONE Stand/Sit toggle; unclear (after a STOP) ->
   exit -> RUG general (sent twice) -> turn test (1 s at 0.3 rad/s, the gyro must see it) -> floor
   height check -> **"Ready. Press Enter to go"**.
2. Enter: autonomous. Status line every ~2 s:
   `[  42.0s] CRUISE  cell (1,3) N  front 1.20 m  tilt 4.1  odom ok  stairs none  wall left`
   Grid: start (0,0), +y = the start heading ("N"), +x = right, 1 m cells.
3. Stairs: confident stairs ahead -> APPROACH (square to the steps, 0.6 m short) -> ENTER
   (Enter AI -> AI Basic -> turn test) -> ALIGN -> CLIMB -> DESCEND -> EXIT (Exit AI -> Move Mode ->
   RUG general -> turn test) -> wall following continues. Unsure stairs: stop 2 s, look again.
4. **HOLD** (`*** HOLD: <reason> ... ***`): the dog stands still, the Captain may drive by
   controller, **Enter resumes** autonomous mode. Reasons:

   | reason | meaning |
   |---|---|
   | `blocked` | no free lane all round / both turn directions blocked |
   | `no_progress` | sent forward 3 s but did not move (marked as dead end) |
   | `time` | `run_time_s` reached |
   | `stairs_unsure` | something stair-like, not sure -> Captain decides |
   | `ai_no_response` | AI Basic did not obey `/cmd_vel` (turn test); AI already exited -> Captain takes the stairs |
   | `rug_no_response` | after Exit AI the dog does not respond; Enter retries the exit sequence |
   | `stairs_no_progress`, `stairs_tilt`, `stairs_timeout`, `stairs_lost` | trouble on the stairs: the dog stays in AI Basic (never exits AI on a step) |
   | `tilt_latch` | tilt above 30 deg (40 deg pitch on stairs): latched until restart |

5. Ctrl+C: zero speed for 1 s, recorder closed, exit. The dog stays standing.
6. Feed lost / posture unclear for 5 s (STOP pressed): the script exits.

### Viewer (read-only)
`http://192.168.10.72:8081` from the PC while `run.py` runs (`--no-viewer` turns it off). Grid,
path, dog; tick `obstacles`, `stairs`, `status` for more. Replay: open the page (or the file
`viewer/index.html` directly) and pick a run's `snapshots.jsonl`. Not tested on the lab network.

### Parameters
All in `config.yaml`; override with `--set key=value` (repeatable), printed at start and saved in
the run folder. The ones to tune on the course:

| key | start | what |
|---|---|---|
| `step_limit` | 0.25 | higher than this above the local ground = obstacle (strips below it are walked over) |
| `lane_w` | 0.70 | lane width; must fit the doorways (min 0.50) |
| `cruise_vx` / `turn_wz` | 0.18 / 0.40 | speeds |
| `wall_side` | left | wall-following side (arenas are mirrored) |
| `stop_m` / `slow_m` | 0.35 / 0.80 | front guard (from the sensor) |
| `footprint_l` / `footprint_w` / `base_offset_x` | 0.65 / 0.42 / 0.0 | turn-sweep footprint (measure the dog) |
| `stairs` | climb | `off` / `detect` (log only) / `climb` |
| `stair_rise_min` / `stair_rise_max` / `stair_tread_min` | 0.08 / 0.20 / 0.15 | stair geometry limits |
| `stair_vx_up` / `stair_vx_down` | 0.15 / 0.10 | stair speeds |
| `tilt_stop_deg` / `tilt_stop_stairs_deg` | 20 / 35 | tilt stop (stairs: pitch; roll keeps 20) |
| `run_time_s` | 0 (off) | HOLD after this many seconds |
| `odom_topic` / `odom_type` | /leg_odom / pose | position source (`/leg_odom2` + `odometry` as alternative) |
| `simple_cmd_qos` | volatile | `transient_local` if gait codes do not arrive |

### Optional checks
- `python3 run.py --live --gait-test ai_basic` — dog standing in open floor, e-stop held: start-up,
  Enter AI -> AI Basic -> turn test -> 1 s forward at 0.1 m/s -> Exit AI -> RUG general -> turn test,
  printing what the gyro and leg odometry saw. Answers "does `/cmd_vel` drive AI Basic?" before a run.

## 2. What a run leaves behind
`runs/<date_time>/`: `bag/` (raw `/livox/points`, `/leg_odom(2)`, `/odom`, IMU, joints, every
command, `/simple_cmd`; camera with `--record-camera`), `decisions.jsonl` (every event, HOLD,
guard intervention), `snapshots.jsonl` (viewer replay, 1 Hz), `config_used.json`, `recorder.log`.
Bring the folder back to the PC after each run.

Measure from a recording (doorway widths, step heights, stair rise/tread, wobble, leg odometry vs
FAST-LIO): `python3 tools/measure.py runs/<date_time>/bag`. Practice-day procedure: start
`python3 run.py` (dry run, records) and drive the dog by controller over the strips, rubble,
through doorways and up to / over the stairs; then run `measure.py` and set `step_limit`,
`lane_w`, the stair limits.

## 3. Tests (PC, WSL Ubuntu-22.04, never on domain 1)
```bash
cd /mnt/c/Users/Admin/Desktop/dog-codes/reactive-core
bash tests/run_tests.sh                               # unit + real trial-3 frames + simulator (~2 min)
ROS_DOMAIN_ID=42 bash sim/run_ros_test.sh default 60  # ROS wiring: fake dog + run.py --sim
ROS_DOMAIN_ID=42 bash sim/run_ros_test.sh stairs 110  # same, with the stairs world
```

## 4. Files
| file | job |
|---|---|
| `run.py` | CLI, safety checks, start-up, Enter / Ctrl+C, recorder, logs, viewer |
| `rc/rosio.py` | the only ROS code besides run.py: topics, QoS, publishers |
| `rc/core.py` | wires everything: sensors in, guarded command + gait codes out |
| `rc/sensing.py`, `rc/attitude.py` | IMU levelling, local ground (stance-floor fallback), obstacle / drop cells |
| `rc/lanes.py`, `rc/driver.py` | lane rule, wall following, HOLD, stairs sequence |
| `rc/guard.py` | front stop/slow, no reverse, turn sweep, stale feeds, tilt |
| `rc/stairs.py` | stairs detection (height levels) |
| `rc/gait.py`, `rc/startup.py`, `rc/posture.py` | gait codes and sequences, start-up, sitting/standing |
| `rc/grid.py`, `rc/memory.py`, `rc/odom.py` | 1 m grid, short-term obstacle memory, leg odometry check |
| `viewer/` | read-only web viewer (layers, server, page) |
| `sim/` | pure-Python simulator, fake dog ROS node, wiring test |
| `tools/` | `measure.py`, `extract_fixtures.py` (trial bags -> test frames), `bagread.py` |

## 5. Not verified yet (first seen on the course)
`/cmd_vel` in AI Basic (doc 03 hints it may not work; the code then HOLDs at the stairs);
leg odometry quality; the dog stopping within 0.5 s when the script dies; braking distance;
real stair / doorway dimensions; footprint and base_link position; lowest lidar beam angle.
