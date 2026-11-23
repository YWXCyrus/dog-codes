# perception — run the challenge perception with either navigation core

The perception code lives in `frontier-fastlio/dog_codes` (`perception.py`: QR M04, hazard signs M02,
scene server for objects / survivors / room types M01 M03 M05; `reporter.py`: JSON to the scoring
server). This folder only adds a launch file so it can run **without** the frontier stack.

| core | how perception runs |
|---|---|
| frontier-fastlio | `mission.launch.py` already starts perception + reporter. Do not use this folder. |
| reactive-core | start `perception.launch.py` in another terminal (below). |

## With reactive-core
```bash
# terminal 1: TA bring-up (camera, odom_relay /fastlio/odometry -> /odom, twist_mux)
ros2 launch lite3_launcher lite3.launch.py navigation:=False
# terminal 2: reactive-core
cd ~/dog-codes/reactive-core && python3 run.py --live
# terminal 3: AFTER run.py printed "Ready" (it restarts FAST-LIO; /odom then starts at the dog's start)
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=1
ros2 launch ~/dog-codes/perception/perception.launch.py group_id:=<registered id>
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
- **QR approach is done by reactive-core** (mode QR, `reactive-core/README.md`): perception publishes
  `qr_seen` (located, not read: `w`, `cx`, `cy`) and `qr_decoded` on `/quad/event`; reactive-core
  turns and steps towards the code and stands still in front of it until it is read. Codes mounted
  higher than ~0.45 m cannot be read from close up with the 20 deg down-tilted camera.
- **Hazard positions are coarser:** `/quad/cloud` (frontier `cloud_proc`) is missing, so the
  distance to a sign falls back to ~0.75 m ahead of the dog.
- **Room type (`scene_type`)** comes from the scene server; without an answer perception reports
  its fallback `workshop`.
- **Start it after run.py's "Ready".** reactive-core restarts FAST-LIO at start-up: an older
  `/odom` frame would put earlier results in the wrong cells.

## Test without the dog
```bash
ROS_DOMAIN_ID=42 bash perception/test_standalone.sh
```
Fake camera with a QR code + fake `/odom` -> perception -> reporter (dry run). Prints the JSON and
PASS when it matches `{"group_id", "task_id": "M04", "task_type": "qr", ..., "message": <text>}`.
QR reading limits and a live camera check without ROS nodes: `qr-test/`.
