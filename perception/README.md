# perception — run the challenge perception with either navigation core

The perception code lives in `frontier-fastlio/dog_codes` (`perception.py`: QR M04, hazard signs M02,
scene server for objects / survivors / room types M01 M03 M05; `reporter.py`: JSON to the scoring
server). This folder only adds a launch file so it can run **without** the frontier stack.

| core | how perception runs |
|---|---|
| frontier-fastlio | `mission.launch.py` already starts perception + reporter. Do not use this folder. |
| reactive-core | start `perception.launch.py` in another terminal (below). |

## Group id (with its PIN) - kept out of the repo
The repo is public on GitHub, so the registered id is NOT in the code. Put it once in a file on the dog
(replace with your registered id):
```bash
echo group02_XXXX > ~/.dog_group_id && chmod 600 ~/.dog_group_id
```
`perception.launch.py`, `mission.launch.py` and `qr-test` read it by default (the same full id goes to
the scene server and the scoring server, booklet v6.3 E); `group_id:=...` still overrides it.

## With reactive-core
```bash
# terminal 1: TA bring-up (camera, odom_relay /fastlio/odometry -> /odom, twist_mux)
ros2 launch lite3_launcher lite3.launch.py navigation:=False
# terminal 2: reactive-core
cd ~/dog-codes/reactive-core && python3 run.py --live
# terminal 3: AFTER run.py printed "Ready" (it restarts FAST-LIO; /odom then starts at the dog's start)
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=61
ros2 launch ~/dog-codes/perception/perception.launch.py
```
Arguments: `dry_run:=true` (reporter logs the JSON, sends nothing), `scene_enabled:=false` (no
scene-server queries), `report_url:=`, `scene_server:=`, `start_mode:=` (default `EXPLORE`),
`odom_topic:=`, `image_topic:=`.

Results: `frontier-fastlio/dog_codes/run_logs/reports_<time>.jsonl` (every JSON and the server's
answer) and `run_logs/snaps/` (the camera picture behind each result).

## What is different from the frontier stack
- **Never moves the dog.** Perception only drives (towards an unreadable QR) in mode `QR_APPROACH`,
  which only the frontier supervisor sets, and it publishes on `/quad/cmd_vel`, which reactive-core
  does not use.
- **No QR approach on main yet.** Perception publishes `qr_seen` (located but not read: `w`, `cx`,
  `cy`) and `qr_decoded` on `/quad/event`, but reactive-core on main does not act on them: a code is
  read only if the driving brings the camera close enough (12 cm code: within ~0.35 m at 640x480,
  `qr-test/README.md`). A QR approach mode (turn / step / stand still in front of the code) exists on
  branch `perception-standalone` (`reactive-core/rc/qr.py`); it touches the driver and is to be merged
  together with the `testing` branch's driver changes.
- **Curtain camera check:** perception answers `/quad/vlm_ask` {'kind': 'curtain'} via the scene
  server on `/quad/vlm_answer` (used by the frontier stack's curtain node and by the curtain mode on
  branch `perception-standalone`). The `testing` branch uses its own camera model (best.pt) instead.
- **Hazard positions are coarser:** `/quad/cloud` (frontier `cloud_proc`) is missing, so the
  distance to a sign falls back to ~0.75 m ahead of the dog.
- **Room type (`scene_type`)** comes from the scene server; without an answer perception reports
  its fallback `workshop`.
- **Start it after run.py's "Ready".** reactive-core restarts FAST-LIO at start-up: an older
  `/odom` frame would put earlier results in the wrong cells.

## For the navigation: stairs and task status (2026-11-25)
Perception only reports; the driving decides. Everything is std_msgs/String JSON.

**Stairs.** The lidar alone may see the stairs as a wall (the first riser is below `step_limit`, the
higher steps + handrails form one face). Perception gives the camera's opinion:
- `/quad/event` `{"type": "stairs_seen", "src": "yolo"|"vlm", "conf", "yaw", "cell": [i, j],
  "bearing", "w", "n"}` - `yolo`: the camera model (`/rc/detections` from `reactive-core/rc/vision_node.py`,
  best.pt) saw `stairs` in >= 3 detections within 1.5 s in one direction (`bearing` rad, + = left,
  max 1 event/s); `vlm`: a scene-server answer named stairs (no bearing).
- Ask on `/quad/vlm_ask` `{"id": "...", "kind": "stairs"}` (e.g. when the dog is blocked by a wide face
  ahead) -> `/quad/vlm_answer` `{"id", "kind": "stairs", "answer": true|false, "src", "text"}`. If the
  camera model saw stairs in the last 1.5 s the answer is immediate (with `bearing`); else the scene
  server is asked (~3.5 s). `"failed": true` = could not check (no image / server error) - not a "no".
- Suggested use (driver): blocked by a wide face + camera says stairs -> treat the face as the first
  riser: square up, stop ~0.6 m short (still in the cell before the stairs), Enter AI, climb, descend,
  walk ~0.5 m clear (inside the cell after), Exit AI - booklet v6.3 F.6: AI only from one cell before
  to one cell after the stairs.

**Task status for backtracking** - `/quad/mission_status`, 1 Hz:
```json
{"t": 123.4, "open": ["M01", "M05"], "all_done": false,
 "tasks": {"M04": {"type": "qr", "reported": true, "message": "My friend is in the market.", ...}, ...},
 "qr": {"read": true, "text": "...", "seen_not_read": false}, "m05_hint": "market",
 "survivor_cells": [[2, 1]], "scenes": {"2,1": "kitchen"}, "stairs_cells": [[3, 1]],
 "suggest": ["find_scene:market", "explore", "go_stairs_when_out_of_time"]}
```
`suggest` is ordered (first = most useful): `read_qr` (a survivor room is known but its QR is not read
- look at its side walls from the doorway), `find_scene:<type>` (an open task names a room type, e.g.
the QR's room for M05), `explore` (tasks open: keep visiting unseen cells, backtrack if needed),
`go_stairs` (all done) / `go_stairs_when_out_of_time`. Cells are the arena cells (1.02 m, +X start
heading, +Y left, booklet E). "reported" = sent to the reporter (latest valid submission counts).

**Survivors:** a persistent `human` detection from the camera model makes perception ask the scene
server about that view at once (instead of waiting for the next new cell).

Tests: `python3 -m pytest -q perception/test_cues.py` (no ROS), `ROS_DOMAIN_ID=42 bash
perception/test_cues_ros.sh` (fake camera + fake camera-model detections -> perception -> PASS x5).

## Test without the dog
```bash
ROS_DOMAIN_ID=42 bash perception/test_standalone.sh
```
Fake camera with a QR code + fake `/odom` -> perception -> reporter (dry run). Prints the JSON and
PASS when it matches `{"group_id", "task_id": "M04", "task_type": "qr", ..., "message": <text>}`.
QR reading limits and a live camera check without ROS nodes: `qr-test/`.
