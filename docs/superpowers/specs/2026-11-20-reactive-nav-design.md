# Reactive navigation for the Lite3 — design spec (draft for review)

Date: 2026-11-20. Status: DRAFT, not approved, no code written. Competition 27 Nov 2026, practice 23-26 Nov.

## 1. Why a fresh start (evidence from trials 1-3)
- Trial 3 run 1 (`bag_20261120_111136`): the lidar saw the wall at 0.36 m then 0.26 m ahead (64-69 points within 0.5 m at wall height) while the dog was still commanded forward at 0.22 / 0.15 m/s. The old code had no guard on the raw cloud (explorer `min_r` 0.35 m, obstacle band from 0.30 m up).
- In the same run the explorer picked 4 goals in 7 s in opposite directions; `/odom` showed 0.79 m/s while the commands were <= 0.23 m/s.
- At 43.4-44.0 s ALL dog-side topics stopped together (`/odom`, `/tf`, `/quad/cloud`, `/imu/data`, `/joint_states`); the camera and our nodes kept running and `/nav_cmd_vel` kept going at 20 Hz for 33 s. The user believes this followed a STOP. The gate had no stale-sensor check.
- Runs 2-4 of trial 3 had 0 `/odom` messages and the explorer pose stayed at (0,0): no motion command was ever sent. So "the dog ignored the code" was most likely "the sensor feed never came back" (inference).
- About 20 % of every `/quad/cloud` frame (~2,200 points) is zero-range returns at the sensor position (0.17 m ahead of base_link, 0.45 m high). Floor returns start about 1.4 m ahead of a standing dog.
- All nine heavy-bag frames show the same start: a pit of raised boards (strips) with troughs, enclosed by plywood walls, a small stair-like structure at the left edge. To a lidar the boards look like low walls with gaps.
- Not tested (data missing): leg-odometry drift (no `/leg_odom` and no raw `/livox/points` in the bags).

## 2. Goal and scope
Reliable subset first: leave the start, explore rooms by wall following, never hit walls, read QR / hazard / survivor / room cues, report, stop in the final area. Curtain and stairs are add-ons built after the core works (section 9). Captain teleop is the fallback for anything the dog cannot do (booklet rule 3: no points for that obstacle).

Out of scope: global map, Nav2, SLAM, pre-mapping, any rough-terrain mode (uneven ground is handled by the same lidar logic + RUG general gait + the step limit).

## 3. Process model
- ONE script, `reactive/run.py`, started by hand. It runs every node in one process. Ctrl+C: send zero speed at 20 Hz for ~1 s, shut down, leave the dog STANDING (no Stand/Sit on exit). No `arm.sh` / `disarm.sh` / `stop.sh`.
- `--dry-run`: everything runs, nothing is published to the dog (used for container and WSL tests).
- Expected back-stop: the dog drops axis commands older than 250 ms and the TA twist_mux times out at 0.5 s, so a killed script should stop the dog within 0.5 s. VERIFY on the hoist.
- Only the guard publishes motion (`/nav_cmd_vel` -> TA twist_mux -> `/cmd_vel`); the controller (priority 100) always wins. Gait / mode codes go via `/simple_cmd`.
- The physical STOP stays the primary safety device; a person stands on it for every on-dog test.

## 4. Start-up sequence (inside run.py)
1. Check feeds alive at their normal rates: lidar, `/imu/data`, `/joint_states`, camera. Fail -> print which, exit without commanding anything.
2. Read posture from the knee joints (sitting ~ -2.75, standing ~ -1.29). If sitting: send Stand/Sit (0x21010202) ONCE, wait until standing and settled. Never toggle twice without re-checking; if the posture is unclear, stop and ask.
3. Set the gait RUG general (0x21010401); confirm (`/quad/status`-style readback or a responding test).
4. Small test command (about 0.1 m/s for 1 s): lidar/IMU must show real movement. This also catches "auto mode off in the app" and "feed dead after STOP".
5. Print "Ready. Press Enter to go" and WAIT. Enter starts walking. Ctrl+C aborts at any point.

## 5. Sensing (raw lidar only, sensor/body frame, no map, no FAST-LIO)
- Input `/livox/points` with the fixed mount transform from `/tf_static` (0.187 m forward, 0.125 m up from base_link, ~15 deg pitch).
- Drop points within 0.30 m of the sensor (zero-range blob). Keep returns up to ~6 m.
- Ground: per 0.3 m patch, local ground height = a low percentile of the heights in that patch. Obstacle height = point height above the local ground of its patch.
- OBSTACLE (precise): at least N_MIN points (start 5) in a cell whose height above local ground exceeds STEP_LIMIT (start 0.25 m), present in at least 3 of the last 5 scans (~0.5 s). Features below STEP_LIMIT are ground, so raised boards / rubble do not stop the dog; walls do. STEP_LIMIT is tuned on practice day 1 (largest step RUG general crosses safely).
- Floor is visible from ~1.4 m ahead only; nothing under or just ahead of the body is seen.

## 6. Lane rule and driving
- Lane = a rectangle in the body frame, width 0.70 m (0.37 m body + ~0.15 m margin each side), depth 1.0 m. A heading is WALKABLE if the lane contains no OBSTACLE (section 5). One obstacle cell vetoes it.
- Driving state machine (10 Hz): CRUISE (walk 0.15-0.20 m/s toward the chosen lane; if the heading error exceeds ~30 deg stop and turn first, never large turns while walking), TURN (in place, up to 0.4 rad/s, direction committed for ~2 s), BLOCKED (no walkable lane: stop, turn toward the widest free angle; reverse only if the rear is clear; after a full circle with nothing, STOPPED and tell the Captain), STOPPED (stays stopped; needs a restart of the script).
- Preference: keep a wall on a fixed side at ~0.6 m (left-hand wall following; side is a parameter, arenas are mirrored). A timer switches side if no new cell is visited for a set time (free-standing walls).
- A progress check: commanded forward for ~3 s but the free distance ahead did not shrink as expected -> stop and report (blocked, or the dog ignores commands).

## 7. Guard (the last layer, independent of driving, map and odometry)
- Between every command and the dog. Zones from the front edge of the sensor: stop below 0.35 m, linear slow-down from 0.80 m; side zones block turns toward something within ~0.30 m; rear zone required to reverse.
- Stops on: stale lidar / IMU / joint data (older than 0.3 s), tilt above the soft limit (start 15 deg slow, 20 deg stop; hard latch 30 deg; flat-floor wobble was 5-12 deg while walking), a failed progress check.
- On a STOP / lost feed: stop, print what was lost, do not command. Restoring the feed is a manual step (section 13, question 1).

## 8. Memory and coordinates (grid)
- Grid fixed to the starting pose: start cell (0,0), +y = start heading ("forward"), +x = right, 1.0 m cells. One cell forward = (0,1). The grid does not rotate with the dog.
- Printed to the console on every cell change and as a status line (current cell, heading, mode, nearest obstacle), also logged to a file and published as a topic. Internal position stays in metres (reports may need metres).
- Stored per cell: visited, and dead-end marks attached to (cell, heading). At a junction prefer unvisited, unmarked lanes; if all are marked, STOPPED + tell the Captain. The memory only breaks ties between lanes that are already safe; it never overrides the guard.
- Position source: leg odometry + IMU (cannot jump; drifts slowly, 1 m cells absorb it). NOT TESTED (no `/leg_odom` in the bags). Fallback: FAST-LIO with a speed-consistency check (it diverged to 67 m/s in trial 2); if inconsistent, drop the memory and fall back to plain wall following.

## 9. Add-ons (after the core works)
- Camera reading: QR, hazard signs, survivors, room type and reporting stay separate from driving; they may only ask for a ~1 s pause. Reuse `perception.py`, `reporter.py`, `scene_client.py`, `qr_decode.py`, `vision.py` after verification. QR was never decoded in any trial (w 170-500 px) - inspect real frames first. Scene server: one request at a time, ~3.5 s, group id still 422 `invalid_group`.
- Curtain (cloth hung from a bar): lidar candidate (thin surface, flutter, returns behind it) AND camera confirmation (scene server) AND a slow touch test (creep <= 0.15 m/s, <= ~1 m; if progress stops it was a wall: back off, mark as wall). The guard front zone is relaxed only inside this mode. Fallback if the scene server is unusable: lidar + touch test.
- Stairs: lidar sees a MONOTONIC rise over several steps (alternating up/down = strip boards, not stairs) AND camera confirms; stop ~0.6 m short and square to the stairs; Enter AI (0x21010528) -> AI Basic (0x2101052A); walk straight at stair speed with a stricter tilt limit; at the top Exit AI (0x2101052B) -> RUG general (0x21010401). Not triggered inside a start-exclusion zone (stair false alarm in trial 1).
- Gait manager: one entry per gait (code, required mode normal/AI, enter/exit steps, speed + tilt limits). Normal gait codes work only outside AI state in move mode; AI gaits only inside AI state (doc 03, 1.2.5 / 1.2.13).

## 10. Files (new folder `reactive/`, old code stays as reference)
`run.py` (lifecycle, start-up, Ctrl+C), `sensing.py` (filter, ground, obstacles), `lanes.py`, `driver.py`, `guard.py`, `health.py` (feed rates), `grid.py`, `gait.py`, later `curtain.py`, `stairs.py`; unit tests under `tests/`, simulator under `sim/`.

## 11. Testing
- Unit tests on REAL data: replay cloud frames from the bags (trial 3 run 1 at 40-42 s: the guard must slow/stop; start frames of the strip pit: boards below STEP_LIMIT must not veto a lane). Synthetic tests for the lane rule, grid and dead-end logic.
- WSL simulation on `ROS_DOMAIN_ID=42` only; on the container only a non-1 domain (CLAUDE.md rule).
- First on-dog tests: hoist, a person on STOP, `--dry-run` first. Verify: the dog stops within 0.5 s when the script is killed, `/cmd_vel` works in AI state, the start-up sequence from sitting.
- Practice-day data (record raw `/livox/points`, `/leg_odom`, IMU, joints, camera): drive across the strips and rubble by controller in RUG general to find STEP_LIMIT, the wobble range and the leg-odometry drift.

## 12. Assumptions not yet verified
- `/cmd_vel` is bridged and works in AI state; the dog stops by itself if commands stop; the lidar resolves the board edges; leg odometry is usable for 1 m cells; the stop distance 0.35 m is enough at 0.2 m/s (braking distance unknown); the Captain's button / auto-trigger rules for obstacles (ask the marshal).

## 13. Open questions
1. What restores the dog's sensor feed after a STOP (restart a program, power cycle, TA)? Draft a TA question with the timestamps of the feed loss.
2. Real group id for the reporter (422 `invalid_group`).
3. Strip, rubble and stair dimensions on the practice course; whether the start area is the same on competition day (the layout changes that morning).
4. Where the final / Drone Launch Area is and how the dog should recognise it without a map.
5. Which arena (mirrored) the group runs, to set the wall-following side.
