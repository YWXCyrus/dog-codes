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
python3 run.py --live --set run_time_s=60      # 2) first real run: 1 min, e-stop held
python3 run.py --live               # normal run
```

`--live` refuses to start unless `ROS_DOMAIN_ID=1` and someone subscribes `/nav_cmd_vel` (the TA
bring-up). Before every run: auto mode on (`lite3_auto`), stand the dog up and choose RUG general
with the controller - start-up sends nothing to the dog.

### What happens
1. Start-up: `run.py` stops any running FAST-LIO (`pkill -f fastlio_mapping`) and starts a fresh one
   (`lio_cmd`, output in `lio.log`) - **keep the dog still until "Ready"**. Then the four feeds
   (`/livox/points`, `/imu/data`, `/joint_states`, `/leg_odom`) arrive for 1 s ->
   `floor 0.327 m below base_link (from the leg joints)` -> `FAST-LIO: pose arriving - map on` ->
   **"Ready. Press Enter to go"**. Nothing is sent to the dog during start-up.
2. Enter: autonomous. Status line every ~2 s:
   `[  42.0s] CRUISE  cell (1,3) N  front 1.20 m  tilt 4.1  floor 0.33  odom ok  stairs none  wall left`
   Driving: **turn on the spot OR walk straight, never both** (the dog slipped at ledges doing both).
   Walking: constant `cruise_vx` (0.20 m/s), no steering. Heading error above `head_err_turn_deg`
   (10 deg) -> stop, turn on the spot until below `turn_done_deg` (4 deg; slowing to 0.2 rad/s in the
   last 20 deg), then walk again. The guard drops the turn of any command that still does both
   (`walk_no_turn` in the logs = a driver path was missed).
   Grid = the arena frame of booklet v6.3 E: start cell (0,0), +X = the start heading ("N"), +Y = its
   left, cell centres 1.02 m apart; cell (i,j) is at (1.02 i, 1.02 j) (status `pos`, the value to report).
   BLOCKED (no free lane): turn on the spot ONE way until a lane is free `blocked_confirm_n` (3)
   decisions in a row; blocked again within `blocked_memory_s` (5 s) keeps the direction and the
   turned angle (no left-right-left dithering); a full circle without a lane -> HOLD `blocked`.
3. Stairs: confident stairs ahead -> APPROACH (square to the steps, 0.6 m short) -> ENTER
   (manual mode -> Enter AI -> AI Basic -> turn test) -> ALIGN -> CLIMB -> DESCEND -> EXIT (Exit AI ->
   Move Mode -> RUG general -> auto mode -> turn test) -> wall following continues. Unsure stairs:
   stop 2 s, look again.
   **In AI the dog ignores `/cmd_vel`**: the code drives it with axis commands on `/simple_cmd`
   (forward `0x21010130` + turn `0x21010135` every time, value in `size`, 20 Hz; axis value = dead
   zone + gain x speed, `ai_*` keys), and forward only works in **manual** control mode - so the
   controller has the speed while on the stairs: hands off the sticks. Ctrl+C / a crash in AI leaves
   the dog standing in AI + manual: finish with `ai_exit` by hand.
   Approach: turn on the spot when the stairs are > 5 deg off, walk straight otherwise. Climb and
   descent: straight, no turning on a step; drifted more than `stair_hold_err_deg` (15) -> HOLD.
4. **HOLD** (`*** HOLD: <reason> ... ***`): the dog stands still, the Captain may drive by
   controller, **Enter resumes** autonomous mode. Reasons:

   | reason | meaning |
   |---|---|
   | `blocked` | no free lane after a full turn on the spot |
   | `no_progress` | sent forward 3 s but did not move (marked as dead end) |
   | `time` | `run_time_s` reached |
   | `stairs_unsure` | something stair-like, not sure -> Captain decides |
   | `ai_no_response` | AI Basic did not obey the turn-axis command (turn test); AI already exited -> Captain takes the stairs |
   | `rug_no_response` | after Exit AI the dog does not respond; Enter retries the exit sequence |
   | `stairs_no_progress`, `stairs_timeout`, `stairs_lost` | trouble on the stairs: the dog stays in AI Basic (never exits AI on a step) |
   | `stairs_heading` | heading drifted > 15 deg on the stairs (no turning on a step): stays in AI Basic; line it up by controller, Enter resumes |
   | `tilt_latch` | tilt above 60 deg (the dog going over): latched until restart |

   Not a HOLD: tilt above 35 deg zeroes the command until the tilt is back below 30 deg, then the dog
   drives on by itself (`tilt_stop` in the logs).

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
| `lane_w` | 0.60 | lane width; must fit the doorways and the ~0.9 m start bay (min 0.50) |
| `lio_restart` / `lio_cmd` | true / `ros2 launch lite3_launcher fast_lio.launch.py` | fresh FAST-LIO every run (`false`: leave the running one alone) |
| `lio_radius_m` / `lio_time_s` / `lio_blind_m` | 3.0 / 30 / 0.50 | FAST-LIO map kept around the dog; points nearer than blind (own legs) never stored |
| `lio_max_speed` / `lio_jump_m` | 1.0 / 0.50 | FAST-LIO pose faster / jumpier than this = diverged -> map off for the rest of the run |
| `cruise_vx` / `cruise_vx_min` / `turn_wz` | 0.20 / 0.20 / 0.40 | speeds (min = max: constant forward speed) |
| `head_err_turn_deg` / `turn_done_deg` | 10 / 4 | walking straight: stop and turn on the spot above this heading error, until below the second |
| `turn_wz_min` / `turn_slow_deg` | 0.20 / 20 | on-the-spot turns slow to this rate in the last degrees (the dog answers ~0.3 s late) |
| `stair_hold_err_deg` | 15 | climb / descent walk straight; heading error above this -> HOLD `stairs_heading` |
| `wall_side` | left | wall-following side (arenas are mirrored) |
| `stop_m` | 0.24 | front stop, from the sensor (nose ~0.14 m ahead of it -> ~0.10 m gap); no slow zone. Walls at lidar height vanish inside `near_drop_m`, so in practice it stops at ~0.30 m |
| `near_drop_m` | 0.30 | lidar returns closer than this to the sensor are dropped (zero-range blob; at 0.15 the front legs caused false front stops) |
| `footprint_w` | 0.42 | front-stop corridor width |
| `persist_k` / `persist_n` | 1 / 1 | obstacle cell seen in k of the last n scans (1/1 = this scan only) |
| `floor_src` / `leg_h_add` | legs / 0.0 | floor = body height from the leg joints (`fixed` = `body_h_default`) |
| `stairs` | climb | `off` / `detect` (log only) / `climb` |
| `stair_rise_min` / `stair_rise_max` / `stair_tread_min` | 0.08 / 0.20 / 0.15 | stair geometry limits |
| `stair_vx_up` / `stair_vx_down` | 0.15 / 0.10 | stair speeds |
| `tilt_stop_deg` / `tilt_resume_deg` / `tilt_latch_deg` | 35 / 30 / 60 | stop above, drive on by itself below resume (must stay above the ~27 deg stair slope), latch above |
| `run_time_s` | 0 (off) | HOLD after this many seconds |
| `odom_topic` / `odom_type` | /leg_odom / pose | position source (`/leg_odom2` + `odometry` as alternative) |
| `simple_cmd_qos` | transient_local | must match the `/simple_cmd` bridge (`lite3_domain_bridge_1`, TRANSIENT_LOCAL); with `volatile` every gait code was silently dropped (warning "incompatible QoS ... DURABILITY") |
| `ai_fwd_dz` / `ai_fwd_gain` | 6553 / 56000 | AI forward axis = dead zone + gain x m/s (25000 walked ~0.33 m/s, rough) |
| `ai_turn_dz` / `ai_turn_gain` | 9553 / 12000 | AI turn axis = dead zone + gain x rad/s (15000 turned ~0.45 rad/s, rough) |
| `blocked_confirm_n` / `blocked_memory_s` | 3 / 5.0 | BLOCKED ends on a lane free this many decisions in a row; re-entry within this keeps direction + turned angle |

### QR approach (task M04, `rc/qr.py`)
Needs perception running next to `run.py` (`perception/perception.launch.py`, start it after "Ready").
Perception reads QR codes by itself; when it only *locates* one it cannot read (too far / off to the
side), it says so on `/quad/event`, and the driver switches to mode **QR**:

`LOOK` (stand still 0.4 s, then measure: sharp pictures, no latency) -> `TURN` (face it, on the gyro)
-> `STEP` (walk part of the way, on the leg odometry) -> `LOOK` ... -> `READ` (stand in front of it
up to 3 s) -> back to wall following. Status line: `QR LOOK`, `QR STEP` ...
Events in `decisions.jsonl`: `qr: start / read / decoded / unreadable / lost / timeout /
no_room_to_turn` and `qr_read: <text>`.

- Located shapes wider than `qr_seek_w_max` (190 px) are false locations on panels and frames (a real
  QR that big is read) and are never chased; 2 stable sightings within 1 s start an approach.
- At most `qr_max_attempts` (3) per run, `qr_max_s` (30 s) each; a failed spot is not tried again
  (`qr_blacklist_m`), 15 s cool-down. Off for the rest of the run once any QR was read.
- Never closer than `qr_front_min_m` (0.55 m free ahead of base_link), no turn on the spot with an
  obstacle within `qr_turn_clear_m` (0.42 m); the guard still filters every command.
- **Mounting height limit (camera tilted 20 deg down, picture top = 7.3 deg above horizontal):** a
  15 cm QR can only be read from close up if its top edge is below ~0.45 m. Higher codes rise out of
  the picture first; the dog then stops there (`why: top_of_picture`) and gives up after 3 s. Above
  ~0.85 m a code is not even in the picture from 3 m away. **Check the QR height on the course.**
- `--set qr_approach=false` turns it off. Simulator: `tests/test_qr.py` (simulated camera +
  perception, `sim/qrcam.py`): a 0.30 m-high QR 0.5 m off-axis 2.7 m away is read in ~20 s.

### Curtain (`rc/curtain.py`, spec 9.4: lidar candidate AND camera AND touch test)
Mode **CURTAIN**: `ALIGN` (square to the surface) -> `INSPECT` (camera question via perception
`/quad/vlm_ask`, answer `/quad/vlm_answer`; perception must run) -> `LINEUP` (only if the dog's line
would put its body on the door post: turn - walk to a point 0.9 m in front of the cloth - turn) ->
`APPROACH` -> `PUSH` (guard front stop relaxed, 0.12 m/s straight) -> through, or `RETREAT` -> wall.
Events in `decisions.jsonl`: `curtain: start (why lidar|suspect|last_resort, features) / ask_camera /
camera_yes / camera_no / lineup / push / through / passed / touch_wall / wall / ...`.

**Detection** (every scan): the surface ahead is scored in 0.8 m windows (a curtained doorway is a
patch of a panel wall), only if it is a **hanging sheet** (returns on the surface line in 0.55-0.95 m:
booklet v6.3 D.4, cloth hung from a bar at ~1 m) seen within 50 deg of square. The strongest cue is
**open below** (`open_below`): no return on the surface line in 0.10-0.40 m - the booklet curtain's
hem is ~0.45 m, a wall reaches the floor (measured from ~0.75 m away, where the lowest beam reaches).
Texture adds a little: folds (`wavy`), thickness, see-through (`pen`), flutter (offsets from each
scan's own fitted line, scan to scan - the dog's motion cancels). Clutter is rejected: rougher than a
sheet (`cur_wavy_max`) or a plane that changes from scan to scan (the toy market stall + trolley).
Starts:
- `lidar`: cloth-like on the same plane in 3 of 5 scans AND open below (>= 0.6 in 2 of them):
  camera; no answer -> touch test anyway. The only start that may push without the camera.
- `suspect` (texture only, or score >= 0.05) and `last_resort` (BLOCKED / no new cell for 40 s, tall
  surface ahead): **camera only** - no push without its "yes" (a floor-length curtain lands here).
Thresholds from 9 real runs of 2026-11-22/23 (9.3 min of driving): texture `*_lo` at ~p95-p98 of
real tall surfaces, `open_below` <= 0.5 on every real surface. Result: 0 `lidar` starts, 3 camera
questions (~4 s each). **No real curtain has been recorded yet** - check it on the practice arena.

**Touch test** (the final word): pushing at 0.12 m/s, progress measured by FAST-LIO when it is usable
(feet slip against a wall, leg odometry would still count) else leg odometry; < 0.04 m in 2 s = rigid
-> back off 0.15 m straight over the ground it just walked in on (**the only reverse in the code**,
`cur_retreat_m=0` turns it off; turning with the nose on a wall would sweep the front corners into it)
-> marked as a wall, wall following continues. A passed curtain / touched wall is not tried again
within 0.8 m, a camera "no" within 0.5 m.

**Practice day:** record the dog standing ~1 m in front of the real curtain and of a plain panel,
then `python3 tools/curtain_probe.py runs/<curtain>/bag runs/<panel>/bag --frames` and set
`cur_*_lo/hi` from what they show (`tests/test_curtain.py` keeps the real-wall regression).
`--set curtain=detect` logs candidates without moving for them, `curtain=off` disables it.

Simulator (`sim/lidar.py` cloth: folds, flutter, see-through; `sim/vlmsim.py`: camera answers;
`tests/test_curtain.py`): corridor and curtained doorway head-on, -25 and +20 deg (line-up), camera
ok / down / "no", a rigid look-alike (camera fooled: the touch test catches it, contact only while
pushing), 0 collisions. Not passed (the dog does not push): a doorway it never faces within ~30 deg,
and a lightly folded curtain with the camera down (looks like a wall to any lidar).

### Optional checks
- `python3 run.py --live --gait-test ai_basic` — dog standing in open floor, e-stop held: start-up,
  manual -> Enter AI -> AI Basic -> turn test (turn axis) -> 1 s forward at 0.1 m/s (forward axis) ->
  Exit AI -> RUG general -> auto -> turn test (`/cmd_vel`), printing what the gyro and leg odometry
  saw. Checks the whole AI path before a run; the leg-odometry distance calibrates `ai_fwd_gain`.
- Manual AI (aliases in `~/.bashrc` on the dog): `lite3_manual; ai_mode` ... `ai_exit` (Exit AI ->
  move mode -> RUG general x2 -> auto). Axis by hand: `ros2 topic pub -r 20 -t 60 /simple_cmd
  transfer_interfaces/msg/MotionSimpleCMD "{cmd_code: 0x21010130, size: 25000, type: 0}"` (send the
  turn axis `0x21010135` with `size: 0` alongside; never Ctrl+C a backgrounded `&` publisher - it
  keeps sending: check `ros2 topic info /simple_cmd -v`).

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
| `rc/guard.py` | stale feeds, tilt (stop / resume / latch), no reverse, front stop - nothing else |
| `rc/stairs.py` | stairs detection (height levels) |
| `rc/gait.py`, `rc/startup.py`, `rc/posture.py` | gait codes and sequences, start-up, sitting/standing, body height from the legs |
| `rc/grid.py`, `rc/liomap.py`, `rc/odom.py` | 1 m grid, local 3-D map placed with the FAST-LIO pose, leg odometry check |
| `viewer/` | read-only web viewer (layers, server, page) |
| `sim/` | pure-Python simulator, fake dog ROS node, wiring test |
| `tools/` | `measure.py`, `extract_fixtures.py` (trial bags -> test frames), `bagread.py` |

## 5. Not verified yet (first seen on the course)
`/cmd_vel` in AI Basic (doc 03 hints it may not work; the code then HOLDs at the stairs);
leg odometry quality; the dog stopping within 0.5 s when the script dies; braking distance;
real stair / doorway dimensions; footprint and base_link position; lowest lidar beam angle.

## 6. Course findings 2026-11-22 (runs `runs/20261122_*`, bags analysed offline)
**Start bay is ~1 m wide.** The two "walls" at |y| = 0.45-0.55 m seen in every run are the
bay's side panels (real). In a 1 m bay the turn sweep (footprint corner radius 0.39 m +
`sweep_margin_m`) blocks turning on the spot, so BLOCKED / `HOLD: blocked` right after Enter
is the guard working as designed; the dog can still walk straight out (run 110743 did).
Do not "fix" this by filtering those points - they are walls.

**Getting stuck on the wooden grid frame in front of the bay (RUG general).** Run 110743:
- approach at 0.11 m/s: with anything closer than 0.8 m ahead the driver uses `cruise_vx_min`
  (0.10), whatever `cruise_vx` is;
- the front feet reached the frame at x ~0.45 m, pitch 12 -> 22 deg in 0.5 s;
- `tilt_slow` (15) halved, `tilt_stop` (20) zeroed and `tilt_latch` (30) locked the command
  mid-step; with no command and the legs in the frame the dog sagged to 46 deg pitch.

Hand-driven crossings of the same obstacles (runs 112732, 113959): approach 0.30-0.40 m/s
(leg odometry), body tilt peaks 15-28 deg, the dog walks through.

**Fixed in the guard (tilt push-through, flat ground only; stairs unchanged):** if the dog was
walking forward when the tilt rose above `tilt_slow_deg`, the guard keeps the forward command at
full value and only zeroes turning (`tilt_push` in the logs) for up to `tilt_push_s` (1.0 s);
a tilt that lasts longer slows / stops as before. `tilt_latch` now needs `tilt_latch_s` (0.5 s)
above `tilt_latch_deg` (a step passes, a fall stays); above `tilt_latch_hard_deg` (45) it latches
at once. Front stop, stale feeds and no-reverse still apply during the push. Replaying run
110743's IMU: the old guard slowed at 18.31 s and latched at 18.60 s; the new one keeps walking
through the whole 16-33 deg spike. Approach speed is still `cruise_vx_min` near obstacles, so
the next test (e-stop held) also raises it:

```bash
python3 run.py --live --set cruise_vx=0.30 --set cruise_vx_min=0.25 --set run_time_s=60
```
The front guard still slows from `slow_m` and stops at `stop_m` for walls. If this works, make
the speeds the default. Not verified on the dog yet.

**Typos in `--set`.** Several runs had `exit` typed onto the command line
(`sweep_margin_m=0.05exit`, `mem_time_s=1eexit`) and ran with a text value. `run.py` now
refuses non-numbers for number parameters (`REFUSED: bad parameter ...`).

**Frozen leg odometry in open space** is not recognised (it needs a wall within 2.9 m ahead to
compare against); the dog then stops with `no_progress` HOLDs. Seen only in the simulator.

## 7. 2026-11-23: FAST-LIO map instead of the short-term memory
Live runs `20261123_033725/033854/034135` stalled in BLOCKED / `HOLD: blocked` with a clear path:
the old memory kept ghost cells 0.2-0.5 m ahead of base_link (own legs, noise) where the lidar never
sees floor. The FAST-LIO left running across those runs had diverged by km; a freshly started one
tracked a 1.5 m walk forward and back to ~0.13 m. Now:
- `run.py` restarts FAST-LIO per run; the map is our `/livox/points` placed with `/fastlio/odometry`
  (FAST-LIO's own clouds are 0.5 m voxels), kept 3 m / 30 s, nothing nearer than 0.5 m to the sensor.
- Lanes and the wall check use live scan + map; the **front stop uses the live scan only**.
- Divergence check: the map switches off for the rest of the run (`lio BAD`, event `lio_bad`); the dog
  then drives on the live scan. Replaying `034135` it fired at 12.8 s, before the km jump at 13.2 s.
- The recorder also saves `/fastlio/odometry`, `/livox/lidar`, `/livox/imu` (FAST-LIO inputs).
- Status line: `lio ok` / `lio -` (no fresh pose) / `lio BAD`.
- Feed check before a run: `ros2 topic hz /imu/data` (~200) and `/livox/points` (~10); today they fell
  to 24 -> 7.5 -> 3.9 Hz over three runs.
