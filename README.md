# dog_codes — autonomous run for the Legged Robot Challenge (Jueying Lite3)

One launch file starts every node. Copy this folder to the dog container (e.g. `~/dog_codes`), start the TAs'
bring-up, run `ros2 launch`, arm. Nothing is built (plain python, no colcon, no pip).

## 1. Run it

```bash
# every terminal on the container:
source /opt/ros/humble/setup.bash
source ~/lite3_ws/install/setup.bash        # transfer_interfaces (MotionSimpleCMD)
source ~/ros2_ws/install/setup.bash         # lite3_launcher, FAST-LIO
export ROS_DOMAIN_ID=<the domain the dog topics are on>     # `ros2 topic list` must show /odom after bring-up

# terminal 1 - TA bring-up WITHOUT their Nav2 (FAST-LIO + /odom + TF + twist_mux + joystick):
ros2 launch lite3_launcher lite3.launch.py navigation:=False

# terminal 2 - this package (group id = the one registered with the organisers):
ros2 launch ~/dog_codes/mission.launch.py group_id:=group02_XXXX
#   add  dry_run:=true   to run everything but NEVER publish anything towards the dog (first check)

# terminal 3 - only with the dog on the hoist / free area, someone holding STOP, joystick in hand:
bash ~/dog_codes/scripts/arm.sh             # dog stands up if sitting, calibrates, starts exploring
bash ~/dog_codes/scripts/status.sh          # gate + supervisor state
bash ~/dog_codes/scripts/stop.sh            # software stop (latches) ; disarm.sh clears it
bash ~/dog_codes/scripts/final.sh           # Captain/Marshal: "robot is in the final cell" (manual trigger)
```

Fill `config/mission.yaml` with the morning's task text (labels / rooms) and `group_id` before the run.
## Data it leaves behind (bring `run_logs/` and `~/.ros/log/` back after each test)

- `run_logs/bag_<time>/` - rosbag (auto-started by the launch file): robot state, every command sent, every supervisor
  mode/event, map exploration state, curtain scores, scene answers, odometry/IMU/joints/TF. Replay or `ros2 bag info`.
  `record:=false` turns it off; `record_heavy:=true` also records `/quad/cloud` + camera images (large; use for short
  stair / curtain tests so I can retune from the real lidar data).
- `run_logs/reports_*.jsonl` - every result submitted to the scoring server and its answer.
- `run_logs/snaps/` - camera snapshot taken each time a result was produced (to check what it actually saw).
- `~/.ros/log/<latest>/launch.log` - all node console output (stairs approach numbers, gait changes, stuck events).
- Optional map image: `DOG_DEBUG_PNG=/tmp/grid.png` before launch; `DOG_DEBUG=1` adds terrain profile prints.

## 2. What each node does

| file | role |
|---|---|
| `safety_gate.py` | the ONLY node that talks to the dog: arm/disarm, speed caps, ramp, 0.35 s watchdog, tilt latch, joystick yield, gait/stand commands (`/simple_cmd`), `dry_run` |
| `cloud_proc.py` | `/livox/points` -> `/quad/cloud` (odom frame, z = height above the feet), floor calibration |
| `explorer.py` | own 2D map, frontier exploration, then visits every arena cell centre, A* + pure-pursuit, keep-out zones, GOTO |
| `terrain.py` | stairs detection (monotonic floor rise/drop ahead), stair-climb executor with gait fallback list, rubble detection |
| `curtain.py` | scores wall-like patches for "cloth-ness", pushes through with a touch test (stalls => it was a wall, back off) |
| `perception.py` | QR (own pure-python decoder: OpenCV's QUIRC is not linked on Ubuntu), hazard signs (yellow triangle + template match), scene-server queries, mission task assignment |
| `reporter.py` | HTTP POST to the scoring server, vocabulary validation, retry, latest-wins |
| `supervisor.py` | state machine; mode owner; final-cell stop + face the opening + `/quad/ready_to_launch` |
| `vision.py`, `qr_decode.py`, `scene_client.py`, `common.py` | helpers |

Mission flow: STANDUP -> EXPLORE (rooms, survivors, signs, QR; stairs/curtains are remembered and deferred) ->
STAIRS / CURTAIN when the open area is exhausted or after `defer_until_s` -> on the organiser's final-cell message:
FINAL (stop, turn to the open wall) -> DONE (`/quad/ready_to_launch` published; launch the drone manually).
Joystick use at any time => PAUSED (counts as a manual recovery), resumes 3 s after release.

## 3. Stage ladder for the first runs (do not skip)

1. `dry_run:=true`, dog anywhere: all nodes up? `ros2 topic echo /quad/status`, `/quad/explore_state`, `/quad/supervisor`.
   Check the map: `DOG_DEBUG_PNG=/tmp/grid.png` env var before launch writes the explorer's map as an image.
2. Dog on the hoist, someone on STOP, `arm.sh`, feet in the air: does `/nav_cmd_vel` -> `/cmd_vel` reach the dog and does
   it react? (`status.sh`: `responding` flag). Confirm the stand toggle / start gait codes do what the doc says.
3. Floor, open area, small speeds (`lim_explore_vx: 0.15`): exploration in the practice arena.
4. Stairs alone: put the dog 2 m from the stairs; check `stairs_ahead` event distance vs reality; tune `blind_bias_m`,
   `approach_dist`, `stair_gaits` order, `min_rise`.
5. Curtain, rubble, QR/hazard/scene tasks one by one (`defer_until_s: 0` makes obstacles come first), then a full run.

## 4. Assumptions that are NOT verified on the real robot (check first)

- `/nav_cmd_vel` -> twist_mux -> `/cmd_vel` and `/simple_cmd` reach the motion host; dog is in auto mode (app).
- **Single gait: RUG general for the whole run** (start, flat, rough, stairs; `rough_enabled: false`, `stair_gaits: ['general']`).
  Select RUG general in the app before arming; the code also sends it at arm. Change it back in `params.yaml` if needed.
- Gait/stand codes from doc 03 (`flat_slow 0x21010300`, `general 0x21010401`, `hstep 0x21010407`, `grip 0x21010402`,
  `stand_toggle 0x21010202`) behave as documented; which gait climbs the stairs is unknown (`stair_gaits` list).
- Camera topic `/camera/color/image_raw`; lidar `/livox/points` with TF `odom -> livox_frame`; knee joints `*_Joint_2`.
- Scene server: route/field names are discovered from `/openapi.json`; if that fails the defaults in `perception.py` are guesses.
  `ros2 param set /perception scene_path ...` / restart with other values. Print the spec: `curl http://192.168.10.141:8000/openapi.json`.
- The organiser's "robot reached the final cell" message: topic name/type unknown (`supervisor.final_topic`). Without it use `final.sh`.
- Arena is a 1.02 m grid aligned with the start pose (+X forward at start); stairs/curtains are square to it.
- Curtain cloth-ness thresholds (`curtain.py`) are guesses; the touch test (stall => wall) is the real decision.
- Rubble/stairs detection thresholds were tuned on a simulation, not on the real lidar.
- Perception reports the cell the camera looks at (~1 m ahead); the 0.5 m localisation criterion needs the exact cell.

## 4b. Trial-2 findings and fixes (20 Nov 2026)

What the bags of the two live runs showed (`sim_tests/real_data/` holds the replayed series):
- 10:03: odometry fine, but the dog turned round and pushed against something BEHIND its start pose for ~35 s (commanded
  0.22 m/s, measured 0.04 m/s). The cloud shows nothing there: the lidar (0.425 m up, pitched down 15 degrees) looks UP behind the
  dog, so a low wall or board is invisible. The explorer's goals were all behind the start.
- 10:07: FAST-LIO odometry diverged while the dog turned in place near the walls (pose reached 284 m while the IMU said it stood
  still). The explorer chased the ghost pose.

Fixes (all in `explorer.py` / `safety_gate.py`, parameters in the file headers, switch off by setting the value to 0 / false):
| what | parameter(s) | effect |
|---|---|---|
| bump detection | `explorer.bump_*` | commanded forward >= 0.10 m/s for 3.5 s but moved < 20 % of it => obstacle marked 0.35-0.50 m ahead (valid 60 s), goal blacklisted, back off 1.5 s. Log line `BUMP #n`, event `bump`, `bumps` in `/quad/explore_state` |
| rear fence | `explorer.rear_fence_m` (1.5), `rear_fence_timeout_s` (8) | until the dog has travelled 1.5 m from its start, cells within 1.5 m BEHIND the start line are not traversable. No goal for 8 s => start guards lifted (log `guards lifted`) |
| goal commitment | `explorer.goal_commit_s` (3) + turn hysteresis | no flipping between left / right / behind |
| odometry watchdog | `safety_gate.odom_fault_*` | net odom speed above max(0.6, 1.5 x speed cap) m/s for 0.5 s while ARMED => disarm + latch `odometry diverged`. Recover: `bash scripts/disarm.sh`, restart the bring-up (terminal 1) with the dog at its start pose, `bash scripts/arm.sh` |

Not fixed: the cause of the FAST-LIO divergence itself (unproven; turning in a ~1.4 m gap is the suspect, try
`ros2 param set /supervisor lim_explore_wz 0.3`).

## 5. Tests that were run (WSL, simulated dog; `codes/sim_tests/`)

`run_gate_test.sh` (19 safety checks, incl. the odometry watchdog), `run_dry_test.sh`, `run_manual_test.sh` (joystick takeover), `run_sim_test.sh`
(full mission on a simulated arena: stand-up, rooms, QR, hazard, survivors, curtain, stairs with gait fallback, final cell).
These prove the logic and the plumbing, not the physics.
