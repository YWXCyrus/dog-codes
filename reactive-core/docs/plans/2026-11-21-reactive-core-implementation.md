# reactive-core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single `run.py` on the dog container that starts the Lite3 safely, explores autonomously by reactive lidar wall following in RUG general, climbs the up-platform-down stairs in AI Basic and returns to RUG general, with an independent guard, a 1 m grid memory, a read-only live viewer, run recording and an offline measuring tool.

**Architecture:** All decision logic is pure Python + numpy (no ROS imports) in small modules, wrapped by one `Core` object that takes sensor data and returns a guarded command plus gait codes. `run.py` + `rosio.py` are the only ROS code: subscriptions, publishers, start-up, Enter / Ctrl+C, recording. The same `Core` runs inside a new pure-Python simulator for closed-loop tests and inside a fake-dog ROS node for the WSL wiring test.

**Tech Stack:** Python 3.10, numpy 1.21 (no newer API), ROS 2 Humble (rclpy, rosbag2_py, `transfer_interfaces/MotionSimpleCMD`), pytest 6.2 (WSL only), stdlib `http.server` for the viewer, plain HTML + canvas JS (no external libraries).

**Spec:** `reactive-core/docs/specs/2026-11-20-reactive-nav-design.md` (read it first; this plan argues from it).

**Deviation from the writing-plans template (agreed for speed, user values efficiency):** steps give exact interfaces, algorithms, parameter values and the test cases with their assertions, not the full implementation text. Code is written once, during execution.

## Global Constraints
- All code under `C:\Users\Admin\Desktop\dog-codes\reactive-core\`. Nothing imported from `frontier-fastlio/` (copy helpers instead).
- NO git commits during the build; one commit of everything at the end, only when the user says so. Never push without asking.
- Tests and simulation in WSL `Ubuntu-22.04` (user `dev`) only, ROS tests on `ROS_DOMAIN_ID=42`. Never anything on domain 1 from Claude.
- Frames: body frame = base_link, x forward, y left, z up (ROS). Grid (console / viewer): start cell (0,0), +y = start heading, +x = right, 1.0 m cells, fixed to the start pose.
- Freshness by ARRIVAL time (`time.monotonic()`), never by message stamps.
- No reversing ever: guard clamps vx < 0 to 0.
- AI mode ONLY for the stairs; RUG general (0x21010401) everywhere else. AI Basic only (no fallback gait).
- Codes: Stand/Sit 0x21010202, RUG general 0x21010401, Move Mode 0x21010D06, Enter AI 0x21010528, Exit AI 0x2101052B, AI Basic 0x2101052A.
- `run.py` default = dry run. `--live` requires `ROS_DOMAIN_ID=1`; `--sim` (publishes) requires `ROS_DOMAIN_ID != 1`.
- Line endings LF (repo `.gitattributes` already enforces for py/sh/yaml/md).

## Impact, risk, reversibility, assumptions (user's planning rule)
- **Impact:** new files only, all inside `reactive-core/` (+ `reactive-core/.gitignore`). `frontier-fastlio/`, the TA workspaces, the SDK repos and the dog are untouched. Nothing is sent to the robot during the build.
- **Risk (what could break):**
  1. Lidar geometry assumptions (lowest beam -7 deg, mount pitch sign, IMU pitch sign) wrong -> levelling or blind-zone handling wrong. Mitigation: Task 2 tests sign conventions on synthetic data AND on real trial frames; `run.py` start-up calibrates the floor and refuses to go if the floor height is implausible.
  2. Stair thresholds never seen real stairs -> missed stairs (Captain) or false stairs in the strip pit (dog enters AI in the pit). Mitigation: start-exclusion radius, confident/unsure split, real-frame "must not trigger" tests, every candidate logged.
  3. `/cmd_vel` ignored in AI Basic (doc 03 hints) -> in-run turn test catches it, Exit AI, HOLD.
  4. First on-dog run is the first real test (no hoist) -> dry-run default, lower speed caps on the first run, e-stop held, Ctrl+C zero-speed.
  5. Time: 16 tasks for one day. Order puts the safety-critical path first (sensing -> guard -> driver -> run.py); viewer and measure.py last and are cut first.
- **Reversibility:** fully reversible (new uncommitted files; delete `reactive-core/` code files to undo). The only hard-to-undo action is the final commit/push, which needs the user's go-ahead.
- **Key assumptions:** `/livox/points` is PointCloud2 in `livox_frame` at 10 Hz; `/tf_static` base_link->livox_frame = (0.18678, 0, 0.12498), ~15 deg pitch; `/leg_odom` = PoseWithCovarianceStamped (doc 02 5.3.2); knee joints `LF/RF/LB/RB_Joint_2`; MotionSimpleCMD fields `cmd_code`, `size`, `type` (the old gate sets `size = 0`, `type = 0`); twist_mux subscribes `/nav_cmd_vel`; container has numpy but maybe no pytest (tests run in WSL).

## File structure (all under `reactive-core/`)
| File | Responsibility |
|---|---|
| `config.yaml`, `rc/config.py` | all parameters (flat keys), load + `--set key=value` overrides |
| `rc/geometry.py` | rotations, quaternion->yaw, footprint rectangle, point-in-rotated-rectangle |
| `rc/attitude.py` | roll/pitch complementary filter (accel + gyro) |
| `rc/sensing.py` | levelling, near/range filter, floor calibration, local ground + fallback, OBSTACLE / DROP cells with persistence, stairs-mode trust band |
| `rc/memory.py` | short-term obstacle memory moved with odometry |
| `rc/odom.py` | leg-odometry pose, travelled distance, consistency check |
| `rc/lanes.py` | lane test, free distance per heading |
| `rc/guard.py` | front stop/slow, no reverse, turn sweep, stale feeds, tilt |
| `rc/grid.py` | 1 m grid in the start frame, visited, dead-end marks |
| `rc/stairs.py` | stair profile, confident/unsure UP, DOWN edge, square heading |
| `rc/gait.py` | gait table, timed gait sequences incl. turn test |
| `rc/posture.py` | sitting / standing / unclear, settled |
| `rc/driver.py` | state machine CRUISE / TURN / BLOCKED / HOLD / STAIRS, wall following, progress check, run timer |
| `rc/startup.py` | start-up sequence (pure logic, ticked) |
| `rc/core.py` | wires everything: `Core.on_*()` inputs, `Core.tick()` output |
| `rc/rosio.py` | ROS node: subscriptions, QoS, publishers, PointCloud2 helpers |
| `run.py` | CLI, safety checks, start-up, Enter / Ctrl+C, recorder, decision log, viewer start |
| `viewer/layers.py`, `viewer/server.py`, `viewer/index.html` | read-only modular viewer + replay |
| `sim/world.py`, `sim/lidar.py`, `sim/dogmodel.py`, `sim/harness.py` | pure-Python simulator |
| `sim/fake_dog_node.py`, `sim/run_ros_test.sh` | ROS wiring test (domain 42) |
| `tools/extract_fixtures.py` | trial bags -> `tests/data/*.npz` |
| `tools/measure.py` | offline measuring on a recorded bag |
| `tests/test_*.py`, `tests/run_tests.sh` | tests |
| `README.md` | how to run on the dog |

Test command (used in every task), from Windows via WSL:
`wsl -d Ubuntu-22.04 -u dev -- bash -lc "cd /mnt/c/Users/Admin/Desktop/dog-codes/reactive-core && python3 -m pytest -q tests"`
(`tests/run_tests.sh` wraps this; a single file: append `tests/test_x.py`.)

---

### Task 1: Skeleton, config, geometry

**Files:** Create `rc/__init__.py`, `rc/config.py`, `config.yaml`, `rc/geometry.py`, `.gitignore` (`runs/`, `__pycache__/`, `*.pyc`), `tests/conftest.py` (adds `reactive-core/` to `sys.path`), `tests/run_tests.sh`, `tests/test_config.py`, `tests/test_geometry.py`.

**Interfaces (produces):**
- `config.load(path: str | None = None, overrides: list[str] = ()) -> dict` (flat dict; overrides `"key=value"` parsed with `yaml.safe_load` for the value; unknown key -> `KeyError`).
- `geometry.rot_x(a)`, `rot_y(a)`, `rot_z(a) -> (3,3)`; `yaw_from_quat(x,y,z,w) -> float`; `wrap(a) -> float` in (-pi, pi]; `footprint_corners(length, width, offset_x) -> (4,2)`; `points_in_rect(pts_xy, cx, cy, yaw, length, width) -> bool mask`.

`config.yaml` (flat keys, values from the spec):
```yaml
# sensing
mount_xyz: [0.18678, 0.0, 0.12498]
mount_pitch_deg: 15.0          # overwritten by /tf_static when available
near_drop_m: 0.30
max_range_m: 6.0
patch_m: 0.30
ground_pct: 10
step_limit: 0.25
obs_h_max: 1.20               # above local ground; ignore ceiling returns
n_min: 5
obs_cell_m: 0.10
persist_k: 3
persist_n: 5
body_h_default: 0.32
stairs_trust_m: 0.80          # STAIRS mode: patch ground trusted within this of the stance floor
# memory / odometry
mem_time_s: 10.0
mem_dist_m: 3.0
odom_topic: /leg_odom
odom_type: pose               # pose (PoseWithCovarianceStamped) | odometry (nav_msgs/Odometry)
odom_jump_m: 0.30
odom_speed_factor: 1.5
odom_speed_add: 0.30
# lanes / driving
lane_w: 0.70
lane_w_min: 0.50
lane_d: 1.0
lane_fov_deg: 135
lane_step_deg: 15
rate_hz: 10.0
cmd_rate_hz: 20.0
cruise_vx: 0.18
cruise_vx_min: 0.10
turn_wz: 0.40
walk_wz_max: 0.20
head_err_turn_deg: 30
commit_s: 2.0
wall_side: left
wall_dist_m: 0.60
side_switch_s: 60.0
progress_s: 3.0
progress_ratio: 0.2
run_time_s: 0.0               # 0 = off
# guard
stop_m: 0.35
slow_m: 0.80
guard_corridor_margin_m: 0.05
footprint_l: 0.65
footprint_w: 0.42
base_offset_x: 0.0            # base_link position along the footprint (measure)
sweep_margin_m: 0.10
sweep_dt_s: 0.5
stale_s: 0.30
tilt_slow_deg: 15
tilt_stop_deg: 20
tilt_latch_deg: 30
tilt_stop_stairs_deg: 25
wz_no_memory: 0.20
# grid
cell_m: 1.0
# stairs
stairs: climb                 # off | detect | climb
stair_rise_min: 0.08
stair_rise_max: 0.20
stair_min_rises: 2
stair_bin_m: 0.15
stair_persist: 5
stair_start_excl_m: 1.5
stair_standoff_m: 0.60
stair_look_s: 2.0
stair_vx_up: 0.15
stair_vx_down: 0.10
stair_flat_exit_m: 1.0
stair_max_s: 90.0
stair_no_progress_s: 3.0
# start-up / posture
knee_names: [LF_Joint_2, RF_Joint_2, LB_Joint_2, RB_Joint_2]
sit_knee: -2.4
stand_knee: -1.6
settle_s: 1.0
settle_vel: 0.05
stand_timeout_s: 15.0
test_wz: 0.30
test_s: 1.0
test_ratio: 0.5
simple_cmd_gap_s: 0.4
# viewer / recording
viewer: true
viewer_port: 8081
snapshot_hz: 2.0
record: true
record_camera: false
```

- [ ] Step 1: write `tests/test_config.py`: defaults load (`cfg['step_limit'] == 0.25`); override `step_limit=0.2` -> 0.2 (float); override `wall_side=right` -> `'right'`; unknown key -> `KeyError`.
- [ ] Step 2: write `tests/test_geometry.py`: `rot_y(pi/2) @ [1,0,0] ~ [0,0,-1]`; `yaw_from_quat` of a 90 deg z rotation = pi/2; `wrap(3pi/2) = -pi/2`; `points_in_rect` with a 0.65x0.42 rect at origin: (0.30,0) in, (0.34,0) out, rotated 90 deg: (0,0.30) in.
- [ ] Step 3: run tests -> fail (modules missing).
- [ ] Step 4: implement `config.py`, `config.yaml`, `geometry.py`, skeleton files.
- [ ] Step 5: run tests -> pass.

### Task 2: Attitude + sensing (levelling, ground with fallback, obstacles, drops, persistence)

**Files:** Create `rc/attitude.py`, `rc/sensing.py`, `tests/test_attitude.py`, `tests/test_sensing.py`.

**Interfaces:**
- `Attitude(alpha=0.02)`: `.update(acc_xyz, gyro_xyz, dt) -> (roll, pitch)`; accel auto-scaled (norm < 3 => in g, x 9.81); roll = atan2(ay, az), pitch = atan2(-ax, sqrt(ay^2+az^2)) blended with gyro integration (complementary filter). `.tilt_deg() -> float`. `.roll`, `.pitch`.
- `Mount(xyz, pitch_rad)` dataclass; `level_points(xyz_sensor, mount, roll, pitch) -> (N,3)`: `p_body = rot_y(mount.pitch) @ p + xyz`, then `p_level = rot_y(pitch) @ rot_x(roll) @ p_body` (levelled, yaw-free body frame).
- `filter_points(p_level, mount, near_m, max_range_m) -> (N,3)`: drop within `near_m` of the sensor position, beyond `max_range_m` horizontally, non-finite.
- `calibrate_floor(p_level) -> float | None`: points with 1.4 <= x <= 3.0, |y| <= 1.0; returns the 10th-percentile z if >= 150 points and -0.50 < z < -0.15, else None. (floor_z = -body height.)
- `@dataclass Scan`: `pts` (N,3) levelled, `obs_now` (M,2) obstacle cell centres this scan, `obs` (M,2) persistent obstacle cells, `drops` (K,2) persistent drop cells, `floor_z` float, `t` float.
- `ObstacleDetector(cfg)`: `.update(p_level, floor_z, t, stairs_mode=False) -> Scan`; `.reset()`.
  Algorithm: patch index = floor(xy / patch_m); per patch `g = percentile(z, ground_pct)`; if patch has < 3 points or `|g - floor_z| > step_limit` (normal) / `> stairs_trust_m` (stairs mode) -> `g = floor_z` (FALLBACK); `h = z - g`; obstacle points = `step_limit < h < obs_h_max`; cells (obs_cell_m) with >= n_min such points = `obs_now`; DROP cell = patch whose trusted percentile `g < floor_z - step_limit` (only patches with >= 10 points, only in normal mode); persistence = cell present in >= persist_k of the last persist_n scans.

- [ ] Step 1: `tests/test_attitude.py`: static accel (0,0,9.81) -> roll = pitch = 0; accel for 10 deg nose-down pitch (ax = +9.81 sin 10deg in ROS convention: positive pitch = nose down) converges to +10 deg within 300 updates at dt 0.005; accel in g (0,0,1.0) treated as 9.81.
- [ ] Step 2: `tests/test_sensing.py` (synthetic, built by a helper `make_floor(...)`, `make_wall(x, h)`, `make_board(x0, x1, h)`):
  - `level_points`: a flat floor 0.32 m below base_link, generated in the sensor frame with the dog pitched 10 deg nose-down, levels back to z = -0.32 +- 0.01.
  - Wall at x = 0.55 m (0.6 m tall) with NO floor points within 1.4 m: wall cells appear in `obs_now` (the ground-fallback bug from the spec is covered).
  - Board 0.15 m high from x = 1.5-1.8 m: not an obstacle (below step_limit).
  - Box 0.30 m high at x = 2.0 m: obstacle.
  - Floor drop of 0.30 m beyond x = 2.0 m: DROP cells there in normal mode; none with `stairs_mode=True`.
  - Two stair treads +0.15 / +0.30 m at x = 1.6 / 1.9 m: obstacles in normal mode (second tread > step_limit), none in stairs mode; a 0.8 m wall behind them is an obstacle in both.
  - Ceiling points at z = +2.0: ignored.
  - Persistence: an obstacle present in only 2 of 5 scans is not in `obs`; in 3 of 5 it is.
  - Zero-range blob (2,000 points at the sensor position) is removed by `filter_points`.
- [ ] Step 3: run -> fail. Step 4: implement. Step 5: run -> pass.

### Task 3: Real-data fixtures and replay tests

**Files:** Create `tools/extract_fixtures.py`, `tests/data/trial3_run1.npz`, `tests/test_realdata.py`.

**Interfaces:** `extract_fixtures.py --light <bag dir> --heavy <bag dir> --out <npz> --times 0,1,2,3,4,39,40,40.5,41,41.5,42` (bag-relative seconds). For each time: nearest `/quad/cloud` frame (odom frame, z = height above the feet) and the `/odom` pose at that stamp; convert to the levelled body frame: `p = rot_z(-yaw) @ (p_world_xy - t_xy)`, `z_body = z - BODY_H` with `BODY_H = 0.32` (the old `cloud_proc` default; the fixture also stores the IMU roll/pitch and `/cmd_vel` at that time). Saves arrays `t`, `pts_<i>`, `yaw`, `cmd_vx`. Runs in WSL with ROS sourced (`rosbag2_py`), source bags `C:\Users\Admin\Desktop\run_logs_test3\bag_20261120_111136` + `bagheavy_20261120_111136`. Downsample each frame to <= 6,000 points (fixed seed) to keep the file < 2 MB.

- [ ] Step 1: write `extract_fixtures.py`, run it in WSL, print per-frame point counts and the nearest-obstacle distance ahead (sanity: at 40.5 / 41.5 s about 0.36 / 0.26 m from the sensor, as found in session 5).
- [ ] Step 2: `tests/test_realdata.py` (skips if the npz is missing):
  - frames 40.5 and 41.5 s: `ObstacleDetector` (fed the same frame 5 times for persistence) has obstacle cells within 0.5 m ahead of the sensor (|y| < 0.3).
  - start frames (0-4 s, strip pit): the 0-degree lane of width 0.70 m is reported by `lanes.free_distance` (Task 5) - skip this assertion until Task 5 exists; in this task assert only that no obstacle cell lies within the board strip region the session-5 analysis described (boards below 0.25 m) - concretely: fewer than 3 obstacle cells with 0.5 < x < 1.4 m and |y| < 0.35 m.
- [ ] Step 3: run -> pass (adjust only parameters with a written reason, never the test, if real data disagrees; report it).

### Task 4: Odometry + short-term memory

**Files:** Create `rc/odom.py`, `rc/memory.py`, `tests/test_odom_memory.py`.

**Interfaces:**
- `LegOdom(cfg)`: `.update(x, y, yaw, t, cmd_speed) -> None`; properties `.pose -> (x, y, yaw)` (odom frame), `.ok -> bool`, `.travelled -> float`, `.speed -> float` (from successive poses over >= 0.2 s). Inconsistent if a step > `odom_jump_m` or speed > `odom_speed_factor * cmd_speed + odom_speed_add` for > 0.5 s; once inconsistent, `.ok` stays False for 5 s after the last bad sample. `.reset_reason -> str`.
- `ObstacleMemory(cfg)`: `.add(obs_body_xy, pose, t, travelled)`; `.query(pose, t, travelled) -> (N,2)` body-frame points (odom-frame storage, transformed back); entries expire after `mem_time_s` OR `mem_dist_m` of travel; `.clear()`; dedup on a 0.1 m grid.

- [ ] Step 1 tests: memory point added at body (1.0, 0) with pose (0,0,0); after the dog moves to (1.0, 0, +90 deg) the query returns ~(0, 0) ... (exact: body coordinates of world (1,0) seen from (1,0,90deg) = (0,0)); expiry by time (11 s) and by travel (3.1 m); odom jump 0.5 m -> `ok` False, recovers after 5 s of good data; speed 1.0 m/s while commanded 0.15 -> `ok` False.
- [ ] Step 2 run -> fail; Step 3 implement; Step 4 run -> pass.

### Task 5: Lanes

**Files:** Create `rc/lanes.py`, `tests/test_lanes.py`; enable the lane assertion in `tests/test_realdata.py`.

**Interfaces:**
- `lane_free(obs_xy, heading, width, depth, x0=0.0) -> bool`: rotate points by -heading; any point with `x0 < x < depth` and `|y| < width/2` -> False. `x0` = 0 (lane starts at base_link).
- `free_distance(obs_xy, heading, width, dmax=3.0) -> float`: distance to the first point in the lane strip, `dmax` if none.
- `candidate_headings(cfg) -> np.ndarray` (radians, every `lane_step_deg` within +-`lane_fov_deg`).
- `evaluate(obs_xy, drops_xy, cfg, width=None) -> list[Lane]`, `Lane(heading, free: bool, free_dist: float)`; drops veto like obstacles.

- [ ] Tests: empty -> all free; a wall across at x = 0.8 -> heading 0 blocked, +-90 deg free; a 0.80 m wide doorway in a wall at x = 0.6: heading 0 free with width 0.70, blocked with 0.90; one drop cell at (0.5, 0) vetoes heading 0; real-data start frames: heading 0 free (strip boards do not veto). Run -> fail -> implement -> pass.

### Task 6: Guard

**Files:** Create `rc/guard.py`, `tests/test_guard.py`.

**Interfaces:**
- `@dataclass Cmd: vx: float = 0.0; wz: float = 0.0`.
- `@dataclass GuardInput: obs_now (M,2), mem (N,2), mem_ok: bool, ages: dict[str, float], tilt_deg: float, stairs: bool = False`.
- `Guard(cfg, sensor_x=0.18678)`: `.filter(cmd, gi) -> tuple[Cmd, list[str]]` (reasons, empty when untouched); `.latched -> bool` (tilt above latch; stays latched).
  Rules, in this order:
  1. any age in `ages` > `stale_s` -> Cmd(0,0), reason `stale:<name>`.
  2. latched or tilt > latch -> zero, `tilt_latch`. tilt > stop (stairs: `tilt_stop_stairs_deg`) -> zero, `tilt_stop`. tilt > slow -> vx * 0.5, `tilt_slow`.
  3. vx < 0 -> 0, `no_reverse`.
  4. FRONT on `obs_now` only: corridor |y| < footprint_w/2 + guard_corridor_margin_m, x > sensor_x; d = min(x) - sensor_x; d < stop_m -> vx = 0 (`front_stop`); d < slow_m -> vx *= (d - stop_m) / (slow_m - stop_m) (`front_slow`).
  5. TURN SWEEP on `obs_now` + (`mem` if `mem_ok`): rectangle (footprint_l + 2*sweep_margin, footprint_w + 2*sweep_margin) centred at (base_offset_x, 0); for yaw steps of 5 deg from 0 to `wz * sweep_dt_s` (plus the forward shift `vx * sweep_dt_s` when walking), any point inside the rotated rectangle that is NOT inside the un-rotated footprint at yaw 0 (self / already-touching points) -> wz = 0 (`sweep_block_left` / `_right`). If not `mem_ok`: |wz| clamped to `wz_no_memory` (`no_memory`).
  `.blocked_both(gi) -> bool` = both turn directions blocked (used by the driver for HOLD).

- [ ] Tests: stale imu (0.4 s) -> zero; tilt 22 -> zero; tilt 32 -> latched forever; vx -0.1 -> 0; real-data frame 41.5 s -> vx 0 (`front_stop`); obstacle at 0.6 m ahead -> vx scaled between 0 and cmd; box 0.25 m left of the rear corner (from `mem`) blocks a left turn (wz > 0) but not a right turn; same box with `mem_ok=False` -> not seen, and |wz| <= 0.2; point inside the footprint itself ignored. Run -> fail -> implement -> pass.

### Task 7: Grid

**Files:** Create `rc/grid.py`, `tests/test_grid.py`.

**Interfaces:**
- `Grid(cell_m=1.0)`: `.set_origin(pose)` (start pose in the odom frame); `.to_grid(pose) -> (gx, gy)` metres in the grid frame (gx = right, gy = forward); `.cell(pose) -> (cx, cy)` ints (round); `.update(pose) -> (cx, cy) | None` (new cell entered); `.visited: set`; `.heading_dir(yaw) -> 'N'|'E'|'S'|'W'` (N = start heading); `.mark_dead_end(cell, dir)`, `.is_dead_end(cell, dir) -> bool`; `.all_marked(cell, dirs) -> bool`; `.clear()`; `.snapshot() -> dict` (lists for JSON).

- [ ] Tests: origin at (2, 3, yaw 90 deg) odom; dog 1 m along the start heading -> cell (0, 1); 1 m to the start's right -> (1, 0); `update` returns the cell only once; dead-end marks per (cell, dir); snapshot JSON-serialisable. Run -> fail -> implement -> pass.

### Task 8: Stairs detector

**Files:** Create `rc/stairs.py`, `tests/test_stairs.py`.

**Interfaces:**
- `profile(p_level, heading, width, bin_m, x_max=3.0) -> (centres, heights)`: points rotated by -heading, |y| < width/2, z < floor + 0.6 above the stance floor and z > floor - 0.6; per bin the median z (NaN if < 5 points).
- `analyse(centres, heights, cfg) -> dict(kind, rises, edge_x)`: walk the bins, ignoring NaN; a RISE = jump in [rise_min, rise_max] between consecutive valid bins (gap <= 2 bins), a FALL = drop in that range; alternating rise/fall within 4 bins = strips -> kind 'none'; `>= stair_min_rises` monotonic rises with flat treads (|diff| < 0.03) between -> 'up'; exactly 1 rise in range followed by valid higher bins that do not form a second rise yet -> 'unsure'; a FALL >= rise_min after >= 0.3 m flat, or valid bins ending at an edge with the next >= 3 bins missing -> 'down'. `edge_x` = position of the first rise / fall.
- `StairDetector(cfg)`: `.update(p_level, floor_z, t, dist_from_start) -> StairsResult`; evaluates headings -20..+20 deg in 5 deg steps; `StairsResult(kind: 'none'|'unsure'|'up'|'down', heading, edge_x, confident: bool)`; 'up' is `confident` only when seen in >= `stair_persist` of the last `stair_persist + 2` scans; `heading` = the heading with the smallest `edge_x` among 'up' results (square to the stairs); suppressed ('none') while `dist_from_start < stair_start_excl_m`.

- [ ] Tests (synthetic profiles and point sets): 3 steps of 0.15 rise / 0.30 tread starting 1.6 m ahead -> 'up', confident after 5 scans, edge_x ~1.6; same stairs rotated 15 deg -> chosen heading within 5 deg of 15; strip boards (up 0.10, down 0.10, up 0.10) -> 'none'; a single 0.15 board -> 'unsure' or 'none' (never 'up'); platform edge with a 0.15 fall at 1.0 m -> 'down'; any 'up' within 1.5 m of the start -> 'none'; real-data start frames -> never 'up'. Run -> fail -> implement -> pass.

### Task 9: Gait sequences + posture

**Files:** Create `rc/gait.py`, `rc/posture.py`, `tests/test_gait_posture.py`.

**Interfaces:**
- `CODES = {'stand_sit': 0x21010202, 'rug_general': 0x21010401, 'move_mode': 0x21010D06, 'enter_ai': 0x21010528, 'exit_ai': 0x2101052B, 'ai_basic': 0x2101052A}`.
- Sequence steps: `('zero', s)`, `('code', name)`, `('wait', s)`, `('turn_test',)`.
- `ENTER_AI = [('zero',1.0),('code','enter_ai'),('wait',2.0),('code','ai_basic'),('wait',1.0),('turn_test',)]`; `EXIT_AI = [('zero',1.0),('code','exit_ai'),('wait',2.0),('code','move_mode'),('wait',0.5),('code','rug_general'),('wait',0.5),('code','rug_general'),('wait',1.0),('turn_test',)]`; `SET_RUG = [('code','rug_general'),('wait',0.5),('code','rug_general'),('wait',1.0)]`.
- `GaitSequencer(steps, cfg)`: `.tick(now, gyro_z) -> tuple[str, Cmd, int | None]` = (status 'running'|'done'|'failed', command to send, code to send now or None). `turn_test`: command wz = test_wz for test_s, pass if mean |gyro_z| over the last half >= test_ratio * test_wz, then 0.5 s zero. `.failed_reason`.
- `posture.classify(names, positions, cfg) -> 'sitting'|'standing'|'unclear'` (mean of `knee_names`; < sit_knee sitting, > stand_knee standing). `posture.Settle(cfg)`: `.update(velocities, t) -> bool settled` (all |v| < settle_vel for settle_s).

- [ ] Tests: ENTER_AI emits enter_ai then ai_basic with the waits (fake clock), turn test passes with gyro 0.3, fails with gyro 0.0 (status 'failed'); EXIT_AI emits exit_ai, move_mode, rug_general x2; codes never emitted closer than `simple_cmd_gap_s`; classify: knees -2.75 sitting, -1.29 standing, -2.0 unclear, missing names unclear. Run -> fail -> implement -> pass.

### Task 10: Driver (state machine incl. stairs) + Core

**Files:** Create `rc/driver.py`, `rc/core.py`, `tests/test_driver.py`.

**Interfaces:**
- `@dataclass DriverInput: t, scan: Scan, lanes: list[Lane], mem: (N,2), pose, odom_ok, travelled, stairs: StairsResult, gyro_z, tilt_deg, guard_reasons: list[str], dist_from_start`.
- `Driver(cfg, grid)`: `.step(di) -> DriverOutput(cmd: Cmd, mode: str, codes: list[int], events: list[dict], stairs_mode: bool)`; `.resume()` (Enter on the console while in HOLD); `.mode`.
  - CRUISE: pick the target heading (below); |error| > head_err_turn_deg -> TURN; else vx = cruise_vx scaled by free distance (cruise_vx_min at 0.8 m), wz = clamp(k * error, +-walk_wz_max).
  - Target heading (wall following, side = `wall_side`): for left side scan the free lanes from the most-left heading (+lane_fov) to the most-right; the first free lane whose free_dist >= lane_d is the target, EXCEPT when no wall is within wall_dist_m + 0.6 m on that side (open space) and heading 0 is free -> keep straight (seek a wall). Ties within 15 deg: prefer lanes leading into an unvisited, non-dead-end grid cell. Side switches after `side_switch_s` without a new cell.
  - TURN: in place, wz = +-turn_wz toward the target, direction committed for commit_s; done when |error| < 10 deg.
  - BLOCKED: no free lane -> turn toward the lane with the largest free_dist; full 360 deg turned with nothing free, or the guard reports both sweep directions blocked -> HOLD('blocked').
  - Progress check: vx >= 0.08 commanded for progress_s and odometry travelled < progress_ratio * commanded distance -> mark (cell, dir) dead end, HOLD('no_progress'). Skipped if `odom_ok` is False.
  - HOLD(reason): cmd zero, event `{'hold': reason}`, stays until `.resume()`; resume -> CRUISE (or back into the stairs phase if on a step: mode STAIRS continues with its current phase).
  - Run timer: `run_time_s > 0` and elapsed > it -> HOLD('time').
  - STAIRS (only if cfg stairs == 'climb'; in 'detect' only events are logged): phases
    `LOOK` (unsure within 1.5 m: stop stair_look_s; becomes confident -> APPROACH; still unsure -> HOLD('stairs_unsure'); gone -> CRUISE),
    `APPROACH` (turn to the stairs heading, walk until edge_x <= stair_standoff_m + sensor offset, vx 0.10),
    `ENTER` (GaitSequencer(ENTER_AI); failed -> run EXIT_AI then HOLD('ai_no_response')),
    `CLIMB` (vx stair_vx_up, wz = small correction to keep heading; detector 'down' seen -> DESCEND),
    `DESCEND` (vx stair_vx_down; after the fall edge is passed and the profile ahead shows no rise/fall for stair_flat_exit_m travelled and tilt < 5 deg for 1 s -> EXIT),
    `EXIT` (GaitSequencer(EXIT_AI); failed -> HOLD('rug_no_response'); done -> mark stairs done, CRUISE).
    Failures in CLIMB / DESCEND (no progress stair_no_progress_s, tilt_stop, stair_max_s exceeded) -> HOLD in AI Basic (never Exit AI on a step). `stairs_mode` output = True from ENTER to EXIT (switches the detector trust band and guard tilt limits).
- `Core(cfg)`: inputs `.on_cloud(xyz_sensor, t)`, `.on_imu(acc, gyro, t)`, `.on_joints(names, pos, vel, t)`, `.on_odom(x, y, yaw, t)`, `.set_mount(xyz, pitch)`; `.start(t)` (after Enter: sets grid origin, start pose); `.tick(t) -> CoreOut(cmd: Cmd, codes: list[int], status: dict, events: list[dict])` at cmd_rate_hz; driver step at rate_hz; guard applied to every command (also during gait sequences: the turn test passes through the guard); when `LegOdom.ok` turns False: `ObstacleMemory.clear()` + `Grid.clear()` + event `odom_inconsistent` (plain wall following continues, spec section 8); camera is not a start-up feed in this version (camera reading deferred, only recorded with `--record-camera`); `.resume()`; `.snapshot() -> dict` (for viewer / logs).

- [ ] Tests (pure, scripted inputs): open floor -> CRUISE forward; wall ahead at 0.6 m -> TURN, never vx < 0; corridor with a left wall -> keeps the wall on the left; all lanes blocked + both sweeps blocked -> HOLD('blocked'), cmd zero, `resume()` -> leaves HOLD; no progress for 3 s -> HOLD('no_progress'); stairs confident ahead -> APPROACH -> ENTER emits enter_ai/ai_basic codes; turn test failing (gyro 0) -> exit codes then HOLD('ai_no_response'); unsure for 2 s -> HOLD('stairs_unsure'); `stairs='detect'` -> no codes, only events; run timer -> HOLD('time'). Run -> fail -> implement -> pass.

### Task 11: Simulator + closed-loop tests

**Files:** Create `sim/__init__.py`, `sim/world.py`, `sim/lidar.py`, `sim/dogmodel.py`, `sim/harness.py`, `tests/test_sim.py`.

**Interfaces:**
- `World`: list of boxes `Box(cx, cy, sx, sy, z0, z1, yaw)`; helpers `World.room_with_door(...)`, `World.strip_pit(...)`, `World.stairs(x0, y0, heading, n_up, rise, tread, platform, n_down, width)`, `World.arena_default()` (start pit with strips, corridor, a room with a 0.9 m door, stairs up-platform-down). `World.floor_height(x, y) -> float` (top of the highest box under the point that is <= 0.25 m above the current foot height, else 0); `World.collides(footprint_poly) -> bool` (any box with z1 > foot height + step_limit intersecting).
- `Lidar(mount, elev_deg=np.arange(-7, 53, 2), az_step_deg=1.0, max_range=10)`: `.scan(world, pose3d) -> (N,3)` points in the SENSOR frame (vectorised ray vs box slab test + floor plane), noise 0.01 m; 2,000 zero-range points added at the origin (like the real sensor).
- `DogModel(world, start, sitting=False, ai_ignores_cmd=False)`: `.step(cmd, dt)` (unicycle in the gait's speed limits, first-order lag 0.2 s; z follows `floor_height`; pitch from the terrain under front/rear feet + wobble +-5 deg), `.on_code(code)` (Stand/Sit toggles posture after 3 s; Enter AI / Exit AI / AI Basic / RUG general / Move Mode change `state`; in AI with `ai_ignores_cmd` commands are ignored), `.imu() -> (acc, gyro)`, `.joints() -> (names, pos, vel)`, `.leg_odom() -> (x, y, yaw)` (+ small drift), `.collided -> bool`.
- `harness.simulate(world, cfg, T, seed=0, dog_kwargs={}) -> SimResult(collisions: int, path: list, modes: list[str], events: list, codes: list[int], final_state: str, reversed: bool)`: runs `Startup` (Task 12; until then a stub that marks ready immediately) then `Core` at 20 Hz with lidar at 10 Hz, simulated clock.

- [ ] Tests: lidar on an empty floor: lowest floor return ~1.3-1.5 m ahead (matches the real 1.4 m); `arena_default` 240 s: 0 collisions, never reversed, leaves the start pit (|travel| > 2 m), enters >= 6 grid cells; door 0.9 m: the dog passes; stairs: reaches the platform and the lower floor, codes contain enter_ai ... exit_ai in that order, ends in CRUISE in RUG general; `ai_ignores_cmd=True`: exit codes sent, final HOLD('ai_no_response'), 0 collisions; strip pit start: no enter_ai code in the first 30 s. Run -> fail -> implement -> pass (tune parameters, record any change in `config.yaml` comments).

### Task 12: Start-up logic

**Files:** Create `rc/startup.py`, `tests/test_startup.py`.

**Interfaces:** `Startup(cfg)`: `.tick(t, feeds_ok: dict[str,bool], posture: str, settled: bool, gyro_z: float, floor_z: float | None) -> StartOut(status: 'running'|'ready'|'failed', cmd: Cmd, codes: list[int], msg: str)`. Steps: FEEDS (all of livox, imu, joints, odom alive for 1 s; fail after 5 s with the missing names) -> POSTURE (sitting: send stand_sit ONCE, wait for standing + settled up to stand_timeout_s; unclear: fail 'posture unclear - stand the dog by hand / check STOP'; never a second toggle) -> GAIT (SET_RUG) -> TURN TEST (`GaitSequencer([('turn_test',)])`; fail = 'dog does not respond: auto mode in the app? feed after STOP?') -> FLOOR (calibrate_floor median of 8 frames; implausible -> fail) -> ready.

- [ ] Tests: missing odom -> failed with 'odom' in msg, no codes ever; sitting -> exactly one stand_sit code, then ready after standing+settled; unclear -> failed, no codes; turn test fails -> failed; standing from the start -> no stand_sit code. Replace the harness stub with `Startup`. Run -> fail -> implement -> pass.

### Task 13: ROS shell (`rosio.py`, `run.py`), recording, decision log

**Files:** Create `rc/rosio.py`, `run.py`, `tests/test_run_args.py`.

**Interfaces:**
- `rosio.RosIO(node, core, cfg, publish: bool)`: subscriptions `/livox/points` (BEST_EFFORT, PointCloud2 -> `cloud_to_xyz` copied from `frontier-fastlio/dog_codes/common.py`), `/imu/data`, `/joint_states`, `odom_topic` (pose or odometry type per `odom_type`), `/odom` (FAST-LIO, logged only), `/tf_static` (TRANSIENT_LOCAL; base_link->livox_frame -> `core.set_mount`); arrival time = `time.monotonic()`; publishers only if `publish`: `/nav_cmd_vel` (Twist), `/simple_cmd` (MotionSimpleCMD, `size=0`, `type=0`, min gap simple_cmd_gap_s), `/rc/status` (String JSON). `.send(cmd, codes)`; `.zero_for(seconds)`.
- `run.py` CLI: `--live`, `--sim`, (default dry run), `--config PATH`, `--set KEY=VALUE` (repeatable), `--gait-test ai_basic`, `--no-record`, `--record-camera`, `--no-viewer`, `--auto-go` (only with `--sim`: skip the Enter prompt, for the wiring test). Checks before anything: `--live` and `ROS_DOMAIN_ID != '1'` -> exit 2; `--sim` and domain == '1' -> exit 2; `--live`: `/nav_cmd_vel` must have >= 1 subscriber within 3 s (twist_mux) else exit 3. Flow: print config diffs vs defaults -> start recorder (`ros2 bag record -o runs/<stamp>/bag <topics>` subprocess; camera only with `--record-camera`) -> spin node in a thread -> `Startup` until ready/failed -> "Ready. Press Enter to go (Ctrl+C aborts)" -> `core.start()` -> 20 Hz loop: `core.tick()`, `rosio.send()`, status line on mode/cell change, HOLD message "HOLD: <reason> - Captain may drive; press Enter to resume"; Enter during HOLD -> `core.resume()`. Decision log `runs/<stamp>/decisions.jsonl` (every driver step with a change + every guard intervention + every event). Ctrl+C anywhere -> zero speed 20 Hz for 1 s, stop recorder (SIGINT, wait 5 s), exit; never sends Stand/Sit on exit. `--gait-test ai_basic`: Startup, then ENTER_AI sequence, 1 s forward at 0.1, EXIT_AI, printing gyro and odom response; requires `--live` or `--sim`.
- Topics recorded: `/livox/points /leg_odom /leg_odom2 /odom /imu/data /joint_states /nav_cmd_vel /cmd_vel /simple_cmd /tf_static /rc/status` (+ `/camera/color/image_raw`).

- [ ] Tests (`test_run_args.py`, no ROS needed: the checks live in a pure function `run.safety_check(args, env) -> (ok, code, msg)`): `--live` on domain 42 -> refused; `--sim` on domain 1 -> refused; default -> dry run (publish False); `--auto-go` without `--sim` -> refused; `--set` parsed. Run -> fail -> implement -> pass. `python3 -m py_compile run.py rc/rosio.py` in WSL.

### Task 14: Fake dog + ROS wiring test (WSL, domain 42)

**Files:** Create `sim/fake_dog_node.py`, `sim/run_ros_test.sh`.

**Interfaces:** `fake_dog_node.py [--world default|stairs] [--sitting] [--ai-ignores-cmd] [--duration S]`: wraps `World` + `Lidar` + `DogModel`; publishes `/livox/points` (PointCloud2, frame `livox_frame`, 10 Hz, BEST_EFFORT), `/imu/data` 200 Hz, `/joint_states` 50 Hz, `/leg_odom` (PoseWithCovarianceStamped) 50 Hz, `/tf_static` (base_link->livox_frame, TRANSIENT_LOCAL); subscribes `/nav_cmd_vel` (stands in for twist_mux + the dog) and `/simple_cmd` (MotionSimpleCMD from `~/ws/install/transfer_interfaces`); prints collisions and the code log on exit. `run_ros_test.sh`: refuses unless `ROS_DOMAIN_ID=42`; sources `/opt/ros/humble/setup.bash` + `~/ws/install/setup.bash`; starts the fake dog (sitting), runs `run.py --sim --auto-go --set run_time_s=90 --no-viewer` with a 120 s timeout, then sends SIGINT; checks: start-up stood the dog up with ONE stand_sit code, decisions.jsonl exists, recorder bag exists, fake dog reports 0 collisions, after SIGINT the last `/nav_cmd_vel` messages are zero.

- [ ] Step 1: implement; Step 2: run in WSL; Step 3: fix until all checks pass; also run `--world stairs` once and check enter_ai/exit_ai in the code log.

### Task 15: Viewer

**Files:** Create `viewer/__init__.py`, `viewer/layers.py`, `viewer/server.py`, `viewer/index.html`, `tests/test_viewer.py`.

**Interfaces:**
- `layers.Registry`: `.layer(name, type, fn, style=None, default_on=True)` (types: cells, points, path, markers, text); `.snapshot() -> dict` `{'t':..., 'layers': [{'name','type','style','default_on','data'}]}`; a provider exception -> that layer's `data` = None + `error` string (never breaks the snapshot).
- `server.start(registry, port) -> ThreadingHTTPServer` (daemon thread; GET `/` -> index.html, GET `/state.json` -> snapshot; any other method -> 405; binds 0.0.0.0).
- `index.html`: canvas grid view (1 m cells, +y up, +x right, start cell highlighted), per-layer toggles, polls `/state.json` at 2 Hz, "Open log" file picker replays a `snapshots.jsonl` with a time slider. Renderers for the five types. No external scripts.
- `Core` registers `grid` (cells: visited / dead_end / start) and `dog` (markers + path); prepared but `default_on=False`: `obstacles` (points), `guard` (text), `stairs` (markers). `run.py` writes `runs/<stamp>/snapshots.jsonl` at 1 Hz.

- [ ] Tests: dummy layer appears in the snapshot JSON; a raising provider gives `error` and the rest still serialises; server on a free port answers `/state.json` with valid JSON and `POST` with 405. Manual: open `http://localhost:<port>` from Windows while the WSL wiring test runs; replay a `snapshots.jsonl`.

### Task 16: measure.py, README, full verification

**Files:** Create `tools/measure.py`, `README.md`; modify root `dog-codes/README.md` (reactive-core row: "code + tests", how to run).

**Interfaces:** `measure.py <bag dir> [--from S --to S]` (ROS sourced, read-only): runs Attitude + level + ObstacleDetector + StairDetector over `/livox/points`; prints: free lane width ahead per second while moving (min = narrowest doorway passed), largest height step crossed below the dog's path, stair rise / tread / width estimates when a stair profile is seen, roll/pitch range while moving vs standing, leg-odom vs FAST-LIO path difference. Test on a bag recorded by the Task 14 wiring test (known world: door 0.9 m, rise 0.15 m) - values within 0.05 m.

`README.md`: two terminals on the container (TA bring-up `ros2 launch lite3_launcher lite3.launch.py navigation:=False`, then `python3 run.py --live` from `~/dog-codes/reactive-core`), first-run settings (`--set cruise_vx=0.12 --set run_time_s=120`), what each console message means (HOLD reasons), Ctrl+C behaviour, e-stop note (never auto-recovered), viewer URL, where runs are saved, practice-day measuring procedure (`--dry-run` recording while driving by controller, then `tools/measure.py`), the parameters to tune (step_limit, lane_w, footprint, stair_*).

- [ ] Step 1: implement measure.py + README.
- [ ] Step 2: full test suite in WSL -> all pass; WSL wiring test -> pass; report exact counts.
- [ ] Step 3: spec coverage check against the spec sections 3-11 (list anything not done).
- [ ] Step 4: STOP. Ask the user before the single commit (`git add reactive-core README.md` + one commit with the attribution line). No push.

---

## Order and cut line
Tasks 1-13 are the minimum for a first autonomous trial. 14 (wiring test) is strongly recommended before the dog. 15 (viewer) and 16 (measure.py) are cut first if time runs out; README is never cut.
