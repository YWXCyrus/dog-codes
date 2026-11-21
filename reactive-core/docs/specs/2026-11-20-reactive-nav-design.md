# Reactive navigation for the Lite3 — design spec (draft for review)

Date: 2026-11-20, revised 2026-11-21. Status: agreed with the user 2026-11-21 (implementation plan pending), no code written. Competition 27 Nov 2026, practice 23-26 Nov.

## 1. Why a fresh start (evidence from trials 1-3)
- Trial 3 run 1 (`bag_20261120_111136`): the lidar saw the wall at 0.36 m then 0.26 m ahead (64-69 points within 0.5 m at wall height) while the dog was still commanded forward at 0.22 / 0.15 m/s. The old code had no guard on the raw cloud (explorer `min_r` 0.35 m, obstacle band from 0.30 m up).
- In the same run the explorer picked 4 goals in 7 s in opposite directions; `/odom` showed 0.79 m/s while the commands were <= 0.23 m/s.
- At 43.4-44.0 s ALL dog-side topics stopped together (`/odom`, `/tf`, `/quad/cloud`, `/imu/data`, `/joint_states`); the camera and our nodes kept running and `/nav_cmd_vel` kept going at 20 Hz for 33 s. The user believes this followed a STOP. The gate had no stale-sensor check.
- Runs 2-4 of trial 3 had 0 `/odom` messages and the explorer pose stayed at (0,0): no motion command was ever sent. So "the dog ignored the code" was most likely "the sensor feed never came back" (inference).
- About 20 % of every `/quad/cloud` frame (~2,200 points) is zero-range returns at the sensor position (0.17 m ahead of base_link, 0.45 m high). Floor returns start about 1.4 m ahead of a standing dog.
- All nine heavy-bag frames show the same start: a pit of raised boards (strips) with troughs, enclosed by plywood walls, a small stair-like structure at the left edge. To a lidar the boards look like low walls with gaps.
- Not tested (data missing): leg-odometry drift (no `/leg_odom` and no raw `/livox/points` in the bags).

## 2. Goal and scope
Reliable subset first: leave the start, explore rooms by wall following, never hit walls, climb stairs when found (section 9), stop when told. Captain teleop is the fallback for anything the dog cannot do (booklet rule 3: no points for that obstacle).

First version (built 2026-11-21): core (sections 3-8b) + stairs (section 9). Deferred: curtain, camera reading / reporting, final-area recognition (waiting for organiser confirmation).

Out of scope: global map, Nav2, SLAM, pre-mapping, any rough-terrain mode (uneven ground is handled by the same lidar logic + RUG general gait + the step limit).

## 3. Process model
- Two terminals on the container: (1) the TA bring-up `ros2 launch lite3_launcher lite3.launch.py navigation:=False` (needed for twist_mux `/nav_cmd_vel` -> `/cmd_vel`; it also starts FAST-LIO), (2) `reactive-core/run.py`. `run.py` refuses to start if `/cmd_vel` has no subscriber or twist_mux is not running.
- `run.py` runs all logic in one process. Ctrl+C: send zero speed at 20 Hz for ~1 s, shut down, leave the dog STANDING (no Stand/Sit on exit). No `arm.sh` / `disarm.sh` / `stop.sh`.
- SAFE BY DEFAULT: without `--live`, `run.py` is a dry run (everything runs, nothing is published to the dog). `--live` additionally requires `ROS_DOMAIN_ID=1`; otherwise it exits.
- Parameters in `config.yaml`, any value overridable on the command line (`--set step_limit=0.20`), so practice-day tuning needs no code edits. The values used are printed at start and saved with the run.
- Expected back-stop: the dog drops axis commands older than 250 ms and the TA twist_mux times out at 0.5 s, so a killed script should stop the dog within 0.5 s. NOT VERIFIED (there is no hoist; first seen on the course with the e-stop held).
- Only the guard publishes motion (`/nav_cmd_vel` -> TA twist_mux -> `/cmd_vel`); the controller (priority 100) always wins. Gait / mode codes go via `/simple_cmd`.
- The physical STOP stays the primary safety device; a person stands on it for every on-dog test.

## 4. Start-up sequence (inside run.py)
1. Check feeds alive at their normal rates: `/livox/points`, `/imu/data`, `/joint_states`, `/leg_odom`, camera. Fail -> print which, exit without commanding anything.
2. Posture from the mean knee angle of `/joint_states` (sitting ~ -2.75 rad, standing ~ -1.29 rad): below -2.4 = sitting, above -1.6 = standing, between = UNCLEAR (e.g. after STOP / Lose Control Protection) -> print and exit. "Settled" = all joint speeds near zero for 1 s. If sitting: send Stand/Sit (0x21010202) ONCE, wait until standing and settled (timeout -> exit). Never toggle twice without re-checking. There is no ROS topic for the robot state or gait (UDP only), so posture is inferred.
3. Set the gait RUG general (0x21010401), sent twice 0.5 s apart. The gait cannot be read back; step 4 is the proof that commands are obeyed.
4. Test command: turn in place at 0.3 rad/s for 1 s; the IMU gyro must show the turn (> 50 % of the commanded rate). A turn instead of a forward step: it is clearly visible on the gyro and moves the dog toward nothing. This also catches "auto mode off in the app" and "feed dead after STOP".
5. Check the stance floor height while standing: the default body height 0.32 m (measured on the dog in trial 2) is kept unless the floor seen 1.4-3.0 m ahead agrees within 0.05 m (raised strip boards around the start would otherwise shift every height; seen in the simulator).
6. Print "Ready. Press Enter to go" and WAIT. Enter starts walking. Ctrl+C aborts at any point.

## 5. Sensing (raw lidar only, body frame, no map, no FAST-LIO)
- Input `/livox/points` (PointCloud2, livox_frame, BEST_EFFORT, 10 Hz) with the fixed mount transform from `/tf_static` (0.187 m forward, 0.125 m up from base_link, ~15 deg pitch).
- LEVELLING: rotate the cloud by the current roll/pitch from the IMU (accelerometer low-pass + gyro, as in the old gate's tilt estimate), so heights are measured against gravity. Without it, the 5-12 deg walking wobble shifts heights at 1 m by up to ~0.2 m, close to STEP_LIMIT.
- Drop points within 0.30 m of the sensor (zero-range blob). Keep returns up to ~6 m.
- Ground: per 0.3 m patch, local ground height = a low percentile of the heights in that patch. Obstacle height = point height above the local ground of its patch.
- Ground fallback: a patch with no floor returns (everything nearer than ~1.4 m, and any patch that holds only wall points) uses the known stance floor height (floor = -body height below base_link, calibrated while standing) instead of its own percentile. Without this, a wall close to the dog becomes its own "ground" and is never an obstacle (the trial-3 wall at 0.26 m would be missed). A patch percentile is trusted only if it lies within STEP_LIMIT of the stance floor.
- DROP (negative obstacle): a patch whose ground lies more than STEP_LIMIT BELOW the stance floor (a platform edge, a step down, a deep trough) vetoes a lane like an obstacle, except inside STAIRS descent. Without this the dog would walk off the stair platform or a step down outside stair mode.
- OBSTACLE (precise): at least N_MIN points (start 5) in a cell whose height above local ground exceeds STEP_LIMIT (start 0.25 m), present in at least 3 of the last 5 scans (~0.5 s). Features below STEP_LIMIT are ground, so raised boards / rubble do not stop the dog; walls do. STEP_LIMIT is tuned on practice day 1 (largest step RUG general crosses safely).
- BLIND ZONES (geometry: sensor ~0.45 m above the floor, 15 deg down at the front, lowest beam ~ -7 deg relative to the sensor (Livox Mid-360 data sheet, from memory - verify), consistent with the measured 1.4 m floor distance):
  - ahead: an obstacle of height h is seen only beyond about (0.45 - h) / 0.40 m; a 0.25 m box disappears inside ~0.5 m;
  - sides: the mount pitch does not help, a 0.25 m box beside the dog is seen only beyond ~1.6 m;
  - rear: the beams point upward, low obstacles behind are never seen.
  Tall walls are seen at all distances.
  Real trial-3 frames show board tops (+0.10 m) already 0.72 m ahead, i.e. the real lidar sees lower/closer than this estimate (the "1.4 m floor" was the floor hidden by boards). The estimate is kept as the conservative case (the simulator uses it).
- SHORT-TERM OBSTACLE MEMORY: obstacle cells (0.1 m) seen in the last ~10 s or ~3 m of travel are kept and moved with the leg odometry, so things that were seen ahead and are now beside / behind the dog (in a blind zone) still count. Used by the lanes and the turn check (section 7). If odometry is flagged inconsistent (section 8) the memory is cleared and turns are slowed (0.2 rad/s).
- Timing: freshness is measured by ARRIVAL time on the container (monotonic clock), never by the message stamps - the dog's IMU / joint / odom stamps count from robot power-on, not wall clock.

## 6. Lane rule and driving
- Lane = a rectangle in the body frame, width LANE_W (parameter, start 0.70 m = 0.37 m body + ~0.15 m margin each side; minimum allowed 0.50 m), depth 1.0 m. A heading is WALKABLE if the lane contains no OBSTACLE (section 5, including the short-term memory). One obstacle cell vetoes it. LANE_W is checked against the doorway widths measured on practice day (section 11).
- Driving state machine (10 Hz): CRUISE (walk 0.15-0.20 m/s toward the chosen lane; if the heading error exceeds ~30 deg stop and turn first, never large turns while walking, |wz| <= 0.2 rad/s while vx > 0), TURN (in place, up to 0.4 rad/s, direction committed for ~2 s), BLOCKED (no walkable lane: stop, turn in place toward the widest free angle; NEVER reverse - the lidar looks upward behind the dog and cannot see a low obstacle there (trial 2); after a full circle with nothing, HOLD), HOLD (section 9.3: stand, print why, Enter resumes), STAIRS (section 9).
- Preference: keep a wall on a fixed side at ~0.6 m (left-hand wall following; side is a parameter, arenas are mirrored). A timer switches side if no new cell is visited for a set time (free-standing walls).
- Lanes within +-90 deg are preferred (wall side first); lanes towards the rear only when nothing ahead is free: the rear is blind, so a "free" rear lane is unverified (the simulator reproduced the trial-2 failure: turning back into the invisible low wall). If the guard blocks the committed turn direction, the driver turns the other way.
- A progress check: commanded forward for ~3 s but the free distance ahead did not shrink as expected -> stop and report (blocked, or the dog ignores commands).
- Run timer (parameter, off by default): after the set time, stop and stay standing. Final-area recognition deferred.

## 7. Guard (the last layer, between every command and the dog)
- FRONT (raw cloud only, independent of odometry and memory): zones from the front edge of the sensor; stop below 0.35 m, linear slow-down from 0.80 m.
- NO REVERSE: negative vx is always clamped to 0 (rear is blind).
- TURN SWEEP: the dog turning in place sweeps a circle around base_link. Footprint = body 0.548 m x 0.370 m (doc 01 1.2.1) + feet / head allowance -> start 0.65 m x 0.42 m (MEASURE on practice day: base_link position along the body, nose and tail overhang), plus a 0.10 m margin. Its half-diagonal + margin gives the sweep radius (start ~0.49 m).
  - Before any turn, the guard checks the motion of the next 0.5 s (yaw step = wz x 0.5 s): a turn direction is blocked (wz set to 0 for that sign) if it brings the footprint CLOSER than the margin to any OBSTACLE point (raw cloud OR short-term memory). Points the turn moves away from never block, so the dog can always turn away from a wall it is close to.
  - If both directions are blocked: no turning, BLOCKED -> HOLD (no reversing).
  - Walking with a small wz: the same footprint is swept along the short arc of the next 0.5 s.
  - The memory is required here because the sides and rear are blind to low obstacles; if the memory is cleared (bad odometry), the turn rate is limited to 0.2 rad/s and only the raw cloud is used (a known gap: a low obstacle beside / behind the dog may then be missed).
- Stops on: stale `/livox/points` / `/imu/data` / `/joint_states` / `/leg_odom` (older than 0.3 s by arrival time), tilt above the soft limit (start 15 deg slow, 20 deg stop; hard latch 30 deg; flat-floor wobble was 5-12 deg while walking; stair limits in section 9), a failed progress check.
- On a STOP / lost feed: stop, print what was lost, do not command. Restoring the feed is a manual step (section 13, question 1).

## 8. Memory and coordinates (grid)
- Grid fixed to the starting pose: start cell (0,0), +y = start heading ("forward"), +x = right, 1.0 m cells. One cell forward = (0,1). The grid does not rotate with the dog.
- Printed to the console on every cell change and as a status line (current cell, heading, mode, nearest obstacle), also logged to a file and published as a topic. Internal position stays in metres (reports may need metres).
- Stored per cell: visited, and dead-end marks attached to (cell, heading). At a junction prefer unvisited, unmarked lanes; if all are marked, HOLD. The memory only breaks ties between lanes that are already safe; it never overrides the guard.
- Position source: leg odometry (`/leg_odom`, geometry_msgs/PoseWithCovarianceStamped per doc 02 5.3.2; `/leg_odom2` nav_msgs/Odometry as an alternative, parameter), speed computed from successive poses. It cannot jump; it drifts slowly, 1 m cells absorb it. NOT TESTED (never recorded; `/leg_odom2` has empty frame ids and zero covariances).
- Session-1 bag check (dog sitting still): `/leg_odom` exists, 198 Hz, PoseWithCovarianceStamped, valid quaternion, no drift at rest. Behaviour while walking still unknown.
- HEADING comes from the integrated gyro (`/imu/data` z, started from the first odom yaw; real gyro noise 0.0016 rad/s, bias ~0.0003 rad/s), not from `/leg_odom`'s yaw (`yaw_source` parameter). Position (x, y) from `/leg_odom`.
- FROZEN odometry check: if the odometry shows no motion while the guarded command was forward for 3 s AND the lidar saw the wall ahead come closer by at least half the commanded distance, the odometry is declared frozen for the run: memory and grid off, progress check off, stair distances from the commanded speed.
- FAST-LIO `/odom` (started by the TA bring-up anyway) is only LOGGED next to the leg odometry for comparison, never used to drive (it diverged to 67 m/s in trial 2).
- Consistency check on the leg odometry: speed above 1.5 x commanded + 0.3 m/s, or a jump > 0.3 m between messages -> flag inconsistent: clear the grid memory and the short-term obstacle memory, keep plain wall following, print it.

## 8b. Viewer (view-only, modular)
- Goal: see the grid memory map in a PC browser during a run, and replay a saved run afterwards.
- `run.py` starts a small HTTP server (pure Python, no extra packages) on the container, default port 8081 (parameter). Open `http://192.168.10.72:8081`. It is READ-ONLY: it serves a static page and a JSON snapshot; it accepts no commands. Whether the lab network reaches that port is untested.
- Snapshot at about 2 Hz at `/state.json`, plus a log file of snapshots (JSON lines, ~1 Hz or on change) in the run folder. The same page opens a saved log file for replay (file picker, no server needed), including a log built from past bag data.
- MODULAR by layers. A layer is a name, a type and a provider function that returns the current data; modules register layers without touching the page:
  `viewer.layer(name, type, fn, style={...}, default_on=True)`.
  Generic layer types the page already knows: `cells` (list of (cx, cy, class)), `points` (x, y in metres), `path` (polyline), `markers` (pose + heading + label), `text` (key/value status). The page draws any registered layer from its type and style and gives each one an on/off toggle. A new future feature = one new `viewer.layer(...)` call in Python; a genuinely new drawing type = one small renderer added to the page.
- Layers in the first version (minimal): `grid` (cells: visited, dead-end marks at (cell, heading), start cell (0,0) highlighted, 1 m cells, +y up, +x right, same convention as the console) and `dog` (marker with heading, plus the path trail).
- Prepared but off until asked for: lidar obstacle points with the guard's slow/stop zones and the turn-sweep circle, status strip (mode, feed rates, tilt), stairs candidates, camera frame.
- The page shows what the dog's code believes; if the position estimate drifts, the map drifts too.

## 9. Stairs (first version) and later add-ons

### 9.1 What the SDK says about AI mode (doc 03, 1.2.2 / 1.2.3.2 / 1.2.5 / 1.2.7 / 1.2.13)
- "AI" is a BASIC STATE (16), entered from standing (Torque-control, 6) with Enter AI 0x21010528 and left with Exit AI 0x2101052B (back to Torque-control).
- Inside AI state there are four AI gaits: Basic 0x2101052A, Fast 0x2101052E, Jump 0x21010529, Stand 0x2101052C. Jump and Stand are for posture adjustment only - never walk in them. Which AI gait is active right after Enter AI is not documented -> always send AI Basic explicitly.
- Normal gaits (RUG general etc.) work only outside AI state and in move mode; AI gaits only inside AI state.
- Axis commands (joystick style) work in AI state. BUT speed commands (0x0140 / 0x0141 / 0x0145) need navigation mode, and "switching to navigation mode is not supported in AI state". How message_transformer turns `/cmd_vel` into UDP (speed commands or axis commands) is not in our copies of the source. So `/cmd_vel` in AI state is the BIGGEST UNKNOWN for stairs: it may be ignored.
- After Exit AI the dog is in Torque-control; whether move mode (0x21010D06) and the app's auto mode survive is unknown -> the exit sequence re-sends Move Mode + RUG general and then repeats the turn test of section 4 step 4.

### 9.2 Gait manager
One entry per gait: code, state required (normal / AI), enter steps, exit steps, speed limit, tilt limits. Two gaits only - NO fallback stair gait (user: only the AI gait was found to work on the stairs; AI Basic or no stairs).
- `rug_general`: normal state, move mode, 0x21010401; cruise speeds; tilt 15/20 deg.
- `ai_basic`: enter = zero speed 1 s, Enter AI, wait 2 s, AI Basic, wait 1 s, small turn test (gyro must respond, else exit AI, back to RUG general, HOLD); exit = zero speed 1 s, Exit AI, wait 2 s, Move Mode, RUG general, turn test.
- AI mode is ONLY for the stairs. Every other terrain (strips, rubble, flat) uses RUG general.
- If `/cmd_vel` does not drive the dog in AI Basic (the in-run turn test after Enter AI, on flat ground before the first step), the code exits AI, returns to RUG general and goes to HOLD; the Captain takes the stairs (teleop, 0 points for them).

### 9.3 Stairs behaviour (layout: 2-3 steps up, a platform, 2-3 steps down)
- Detection (lidar, levelled cloud): a height profile along each lane in 0.15 m bins; UP candidate = at least 2 consecutive monotonic rises of 0.08-0.20 m each (parameters; real riser height measured on practice day), step edges roughly parallel. Alternating up/down = strip boards, NOT stairs. Must persist over >= 5 scans. Not triggered inside a start-exclusion radius (parameter, start 1.5 m; trial-1 false stairs at the start). Camera confirmation deferred (camera reading not in the first version).
- METHOD (as built): "height levels". Per candidate heading the lane points are histogrammed by height; horizontal surfaces (floor, treads, platform) give peaks = levels, each with a distance range. UP = >= 2 levels each one riser higher AND clearly further away, ranges not interleaved, rises similar (<= 0.05 m); a rise from the dog's own floor counts only if that floor is visible; a level reached by a rise must be >= stair_tread_min (0.15 m) long. Strip boards interleave with the floor -> not stairs (regression test on the real trial-3 strip pit, which a first per-bin profile method mis-read as stairs). Heading: all headings that see the stairs vote, then an edge-line fit over 7 lateral slices; the driver uses the median of recent detections.
- DOWN candidate (from the platform): a level one riser or more BELOW the stance floor, further away than the visible stance. The edge distance seen from the platform is unreliable (the lower steps are seen far out), so it is not used to time the exit.
- Approach: first stand still 1 s and take the median heading of the confident detections (headings seen while walking were off by up to 11 deg in the simulator; from a still dog within 0.5 deg), keep refining while the edge is > 0.9 m away, turn square to it (in place, turn sweep applies), stop ~0.6 m short (parameter) using odometry to the remembered edge (the first riser is inside the blind zone at the stand-off).
- All stair tilt decisions use a ~1 s filtered pitch or a sustained limit (> 1 s), never single samples: flat-floor walking wobble reaches 5-12 deg (trial 2).
- One AI session for the whole obstacle: enter AI Basic before the first step (then ALIGN: the turn test rotated the dog, turn back to the stairs heading), climb, cross the platform, descend, and exit AI when the IMU showed the descent (nose-down pitch >= 8 deg) and the dog has then been level (< 5 deg) with nothing lower ahead for stair_flat_exit_m (1 m) of travel. No nose-down pitch within 4 m after 'down' was seen -> HOLD('stairs_lost'). No gait change on the platform (fewer transitions; the platform length is unknown).
- Climb / descend: walk straight at stair speed (parameter, start 0.15 m/s; slower down, start 0.10 m/s), wz only for small heading corrections. Guard during STAIRS: the front stop uses height above the NEXT step instead of the stance floor (otherwise the steps themselves are walls); drop vetoes (section 5) are off inside STAIRS. Tilt on stairs: PITCH is expected (0.15/0.30 m stairs = 27 deg slope): pitch stop 35 deg, latch 40 deg; ROLL keeps the normal limits (stop 20, latch 30). (A first value of 25 deg would have stopped every climb.)
- After the exit sequence: continue wall following in RUG general; the stairs (cell, heading) are marked done so they are not climbed twice.
- Any failure (gait turn test fails, no progress for 3 s, tilt stop) -> zero speed, stay in place in AI Basic (do not exit AI on a step), HOLD (teleop fallback). Exit AI only on flat ground.
- MODE SWITCH `stairs: off | detect | climb` (parameter). First trial: `climb` (fully autonomous). A CONFIDENT candidate -> AI Basic and climb; an UNSURE candidate (some rises but not all criteria met) -> HOLD at the stairs. Every candidate is logged with its profile and shown on the viewer, so the first trial also confirms the detection.
- After the descent the code exits AI by itself and continues autonomous wall following in RUG general.
- HOLD (replaces self-recovery in the first version; self-recovery is planned for later): zero speed, the dog stays standing (in AI Basic if on a step - never Exit AI on a step), the console prints why, the Captain may drive by controller (priority 100), Enter on the console resumes autonomous mode. Used for: unsure stairs, failed turn test after Enter / Exit AI, no progress on a step, repeated tilt stops, blocked all round, all lanes dead-end-marked.
- An e-stop is never recovered by the code: after STOP the dog is in Lose Control Protection; the code sees the lost feed / unclear posture, stops commanding and exits.
- Optional flat-floor check: `run.py --live --gait-test ai_basic` runs ONLY the enter sequence, a 1 s turn and a 1 s forward at 0.1 m/s, and the exit sequence, printing gyro and leg-odom responses. Dog standing in open floor, e-stop held. Purpose: learn before a run whether `/cmd_vel` drives AI Basic and whether RUG general responds after Exit AI.

### 9.4 Deferred add-ons
- Camera reading: QR, hazard signs, survivors, room type and reporting stay separate from driving; they may only ask for a ~1 s pause. Copy `perception.py`, `reporter.py`, `scene_client.py`, `qr_decode.py`, `vision.py` from `frontier-fastlio/dog_codes/` into `reactive-core/` (copied, not imported, so each folder stands alone) and verify them there. QR was never decoded in any trial (w 170-500 px) - inspect real frames first. Scene server: one request at a time, ~3.5 s, group id still 422 `invalid_group`. Camera confirmation of stairs plugs in here.
- Curtain (cloth hung from a bar): lidar candidate (thin surface, flutter, returns behind it) AND camera confirmation (scene server) AND a slow touch test (creep <= 0.15 m/s, <= ~1 m; if progress stops it was a wall: back off by turning, mark as wall). The guard front zone is relaxed only inside this mode. Fallback if the scene server is unusable: lidar + touch test.
- Final-area recognition: waiting for organiser confirmation (section 13, question 4).

## 10. Files (folder `reactive-core/`; the old code lives separately in `frontier-fastlio/` as frozen reference, and the two folders are independent)
- Pure-Python logic (no ROS imports, testable and simulated directly): `sensing.py` (levelling, filter, ground, obstacles, short-term memory), `lanes.py`, `driver.py`, `guard.py` (front, turn sweep, stale, tilt), `grid.py`, `stairs.py`, `gait.py` (gait table and sequences), `posture.py`, `config.py`.
- ROS shell: `run.py` (lifecycle, start-up, Ctrl+C, `--live`, `--dry-run`, `--gait-test`, recorder), `rosio.py` (subscriptions, QoS, publishers, arrival-time freshness).
- `viewer/` (`server.py`, `layers.py`, `index.html`), `config.yaml`, `tools/measure.py` (section 11), `tests/`, `sim/`.

## 11. Testing and recording
- Test data note: the trial bags contain no raw `/livox/points`, only the old stack's `/quad/cloud` (odom frame, z = height above the feet, built with FAST-LIO). Test frames are rebuilt in the body frame from `/quad/cloud` + the bag's `/odom` pose; the live code reads `/livox/points`. Replace them with raw recordings from practice day 1.
- Unit tests on REAL data: replay cloud frames from the bags (trial 3 run 1 at 40-42 s: the guard must slow/stop; start frames of the strip pit: boards below STEP_LIMIT must not veto a lane; the stair-like structure at the pit edge must not trigger stairs inside the start-exclusion radius). Synthetic tests for levelling, the ground fallback, the lane rule, the turn sweep, the short-term memory, the stair profile (monotonic vs alternating), posture thresholds, arrival-time freshness, the grid and dead-end logic. Viewer: a unit test that registers a dummy layer and checks the snapshot JSON; check the page in a browser with a replayed log in WSL.
- NEW simulator `sim/` (the old `sim_world.py` is NOT reused): pure Python, no ROS. 2D world of wall segments, raised boards and a staircase; a fake lidar with the real mount geometry and beam pattern (levelled body-frame points); a simple dog model (integrates the guarded command, optional lag and wobble). Runs the real driver + guard + stairs logic in a closed loop, faster than real time, as part of the tests (checks: no wall contact, no reversing, leaves the start pit, follows a wall through a door).
- ROS wiring test in WSL on `ROS_DOMAIN_ID=42` only: a tiny fake-dog node (publishes `/livox/points`, `/imu/data`, `/joint_states`, `/leg_odom`, `/tf_static`; subscribes `/cmd_vel`, `/simple_cmd`) to test `run.py` start-up, QoS, Ctrl+C and `--dry-run`. On the container only a non-1 domain (CLAUDE.md rule).
- RECORDING: every `run.py` run (also dry runs) records its own bag into the run folder: `/livox/points`, `/leg_odom`, `/leg_odom2`, `/odom`, `/imu/data`, `/joint_states`, `/nav_cmd_vel`, `/cmd_vel`, `/simple_cmd`, `/tf_static`, camera optional (`--record-camera`), plus a decision log (JSON lines: mode, chosen lane, guard action and reason, stairs candidates).
- `tools/measure.py` (offline, read-only, runs on a recorded bag with the same sensing code): prints wall gaps along the walk (doorway widths, to check LANE_W), the largest height steps crossed (to set STEP_LIMIT), stair rise / run / width when stairs are in view, wobble range, and the leg-odom vs FAST-LIO difference. Practice day 1: drive by controller over the strips, rubble, through doorways and up to the stairs while `run.py` records in dry-run mode.
- First on-dog tests: directly on the course (no hoist), a person holding the controller e-stop, dry run first, lower speed caps for the first run (`--set cruise_vx=0.12`). Watch for: the dog stops within 0.5 s when the script is killed, the start-up sequence from sitting, the AI Basic enter / exit at the stairs (optionally `--gait-test ai_basic` on open floor first).

## 12. Assumptions not yet verified
- `/cmd_vel` works in AI state (doc 03 suggests it may NOT: navigation mode unsupported there); the dog stops by itself if commands stop; the lidar resolves the board edges and stair risers; leg odometry is usable for 1 m cells and the short-term memory; `/leg_odom` message type as in doc 02; the stop distance 0.35 m is enough at 0.2 m/s (braking distance unknown); base_link position along the body and the true footprint; lowest Livox beam angle; AI Basic climbs AND descends these stairs (user: AI gait worked on stairs before); whether move mode / auto mode survive Exit AI; the Captain's button / auto-trigger rules for obstacles (ask the marshal).

## 13. Open questions
1. What restores the dog's sensor feed after a STOP (restart a program, power cycle, TA)? Draft a TA question with the timestamps of the feed loss.
2. Real group id for the reporter (422 `invalid_group`).
3. Strip, rubble and stair dimensions on the practice course (stair layout known: 2-3 steps up, platform, 2-3 steps down; riser height, tread depth, width and platform length unknown); whether the start area is the same on competition day (the layout changes that morning).
4. Where the final / Drone Launch Area is and how the dog should recognise it without a map (user waiting for organiser confirmation).
5. Which arena (mirrored) the group runs, to set the wall-following side.
6. How message_transformer maps `/cmd_vel` to UDP (speed commands in navigation mode, or axis commands)? Answers whether `/cmd_vel` can drive AI Basic (no fallback gait: if not, stairs = Captain). Ask the TA, or test with `--gait-test ai_basic` on open floor.
