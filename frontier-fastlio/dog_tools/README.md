# dog_tools - terrain survey while you drive the dog by hand

Purpose: find out **which gait crosses which terrain** and **what the dog's rough-terrain detector actually sees**,
so `rough_gait` and the detector thresholds in `dog_codes` can be set from data instead of guesses.

You drive the dog with the controller and pick the gait yourself (as always). This folder only **watches**:

- `terrain_survey.py` - subscribes to `/odom`, `/imu/data`, `/livox/points` (+TF). **It has no publisher at all**
  (it logs `publishers=0` at start) - it cannot command the dog. It builds a terrain map, logs 5 Hz samples
  (pose, tilt, wobble, the detector's verdict, current gait label) and writes a report every 20 s and on exit.
- `compare.py` - compares several sessions (one gait / one driving style each) terrain by terrain (no ROS needed).
- `gait_label.py` - OPTIONAL: type gait labels / stumble marks during a run. Not needed when each session is one
  gait that you note down yourself.
- `report.py` - rebuilds the report from a run folder (no ROS needed).
- `scripts/` - `run_survey.sh` (starts the survey), `record_bag.sh` (optional light rosbag), `make_report.sh`,
  `env.sh` (optional: sources ROS + the TA workspaces).

Do **not** run `dog_codes` (`mission.launch.py`, `arm.sh`) at the same time.

## 0. On the PC (once)

```
# on the dog: git clone https://github.com/YWXCyrus/dog-codes.git ~/dog-codes   (dog_tools is in ~/dog-codes/frontier-fastlio/dog_tools)
```

## 1. On the dog container - 2 SSH terminals (3 with the optional bag)

Environment: if your `~/.bashrc` already sources ROS and the TA workspaces, new terminals are ready
(`ros2 pkg list | grep lite3_launcher` prints `lite3_launcher`). If not, in every terminal:
`source ~/dog-codes/frontier-fastlio/dog_tools/scripts/env.sh` (must be **sourced**; it sources the three setup files and prints the domain).
`echo $ROS_DOMAIN_ID` must be the dog's domain (the one `ros2 topic list` shows `/odom` on), the same in ALL terminals.

Check nothing from the autonomy code is still running (this must print nothing):

```bash
ros2 node list | grep -E "safety_gate|supervisor|explorer"
```

If it prints something, stop that launch (Ctrl+C in its terminal, or `pkill -f dog_codes`) before going on.

### Terminal 1 - TA bring-up (FAST-LIO, /odom, TF), same as trial 1

Put the dog **standing on flat floor at the start pose, facing forward** (this sets the map origin), then:

```bash
ros2 launch lite3_launcher lite3.launch.py navigation:=False
```

Wait ~20 s, then in any terminal: `ros2 topic hz /odom` must show a steady rate, not "no new messages" (Ctrl+C to stop it).

### Terminal 2 - the survey (keep the dog standing still for the first ~5 s)

```bash
bash ~/dog-codes/frontier-fastlio/dog_tools/scripts/run_survey.sh
```

Healthy output (a status line every 5 s):

```
publishers=0 (read-only: this node cannot command the dog); domain=<yours>
body height calibrated: 0.2xx m  -> mapping starts
t= 25s cloud=9.8Hz(0.1s ago,15ms) odom=0.0s ago imu=0.0s ago tf=ok calib=0.2xx label=unlabelled pos=(0.00,0.00) ...
```

What to check: `cloud` about 10 Hz, `odom`/`imu` "ago" under 1 s, `tf=ok`, `calib` a number (not WAIT), and the
processing time in `ms` well below 100. If something is off see section 5.

### Optional - gait labels / stumble marks (not needed for the one-gait-per-session plan)

```bash
python3 ~/dog-codes/frontier-fastlio/dog_tools/gait_label.py
```

Menu: `1 flat_slow  2 flat_medium  3 flat_fast  4 crawl  5 general  6 grip  7 hstep`.
`+name` = your own label (use the controller/app's wording, e.g. `+rugged_high_step`), `-` = none/stopped,
`m` = **MARK** (stumble, slip, you had to catch the dog), `n text` = note at the current spot
(e.g. `n uneven cell starts`), `q` = quit. Terminal 2 echoes every label, so you can see it arrived.

### Optional terminal 4 - light rosbag (lets me tune the detector offline)

```bash
bash ~/dog-codes/frontier-fastlio/dog_tools/scripts/record_bag.sh 240      # max seconds, stops by itself; ~1 GB per 4 min; refuses if disk is short
```

## 2. The runs: 6 sessions = 3 rug gaits x (slow feel-out, then full run at max efficiency)

Someone on STOP, as for any hand-driven test. Terminal 1 (bring-up) stays running for ALL sessions, so the odom
origin stays the same; between sessions put the dog back at the **same start pose and heading**.

For each session:

1. Dog at the start, standing, terminal 1 running. Select the gait on the controller/app (that is the only time the
   gait matters to the tool - **you** note it).
2. Start the survey (terminal 2) and wait for `body height calibrated ... mapping starts` (dog still).
3. Drive the course. Slow session: feel out each terrain slowly. Fast session: the full run at max efficiency.
4. Ctrl+C in terminal 2. It prints the session folder, e.g. `.../dog_tools_runs/survey_20261120_141502/report`.
   **Write down: folder name = gait + slow/fast** (e.g. `survey_20261120_141502 = general_slow`).
5. If the dog falls, or you restart terminal 1, tell me: that session is not comparable.

Suggested order: `general` slow, `general` fast, then `grip`, then `hstep` (any order is fine).

## 2b. Comparing the sessions (PC or dog, no ROS)

After copying `dog_tools_runs` back (section 3), from the `dog_tools` folder:

```
python3 compare.py general_slow=survey_A general_fast=survey_B grip_slow=survey_C grip_fast=survey_D hstep_slow=survey_E hstep_fast=survey_F
```

(use the full folder paths; `NAME=` is your own tag). It sums the maps, classes every cell the dog visited
(flat / rough / raised / obstacle / unmapped), and prints a table per terrain: time, speed, wobble, max tilt, gyro per
session, plus a text grid of the classes and a map-consistency warning if the sessions' start poses differed.
Options: `--cells 1:2,0:0` (only that lattice box), `--override "1,0=rough 2,0=flat"` (force a cell's class),
`--flat-rough`, `--raised-h`, `--wall-pct` (class thresholds are first guesses: read the printed cell table and
tune), `--tilt-warn`. Output goes to a `compare_<time>` folder next to the sessions (`compare.txt`, `compare.csv`).
You can also just send me the folders and I run it.

## 3. Bring it back to the PC

```
scp -r user@192.168.10.72:~/dog_tools_runs "C:\Users\Admin\Desktop\dog_tools_runs"
```

Then tell me the path. I read `report/summary.txt`, `report/cells.csv` and the two PNGs.
To rebuild a report on the dog: `bash ~/dog-codes/frontier-fastlio/dog_tools/scripts/make_report.sh`.

## 4. How to read the output (`<run>/report/`)

- `map_rough.png` - floor-height scatter (std in 0.5 m blocks). Blue = flat, red = rough; the white tick on the colour
  bar is the dog's "rough" threshold (0.05). **Cell labels `i,j` sit in the top-left corner of each 1.02 m cell**;
  (0,0) is the start cell, +X right, +Y up. The path is coloured by gait label; `Mn` = your marks, `Nn` = your notes
  (listed on the right).
- `map_height.png` - mean floor height above the start floor (stairs/ramps/raised platforms show as a gradient).
- `summary.txt/.csv` - one row per stretch under one gait (`segment`) and per gait overall (`total`):
  distance, speed, max tilt, **wobble** (lower = steadier), detector verdicts (`hit%`, `on%`, `flips` = how often the
  flag flapped), your marks. This is the gait-vs-terrain comparison.
- `cells.csv` - per lattice cell: coverage, mean/max height, roughness, and which gaits were used there.
- `../samples.csv`, `../labels.csv`, `../map.npz` - raw data.

Then tell me, per cell or region: which terrain it is, which gait worked, which didn't.

## 5. Limits and troubleshooting

- **The map is blind where the lidar can't see the floor**: within ~1 m of the dog and behind it (the lidar is pitched
  down at the front). The start cell and anything you only walked away from stay grey. Cells ahead of the dog fill in.
- **Curtains/hanging objects show up as a rough stripe**; **stairs look rough** on the roughness map (look at the height
  map to tell stairs from rubble). White = more than 0.6 m tall (wall/object); a very tall step would also turn white.
- Heights are relative to the floor where the dog calibrated. Restarting terminal 1 moves the origin: start a new survey.
- Map accuracy ~0.1-0.3 m (FAST-LIO drift): good for regions, not for centimetres.
- The detector columns copy the rule in `dog_codes/terrain.py` (std > 0.050 or share above 4 cm > 0.35, ON after 4
  hits, OFF after 15 misses) but not its stairs suppression.

| symptom | cause / fix |
|---|---|
| `NO /odom received` | terminal 1 not running or wrong `ROS_DOMAIN_ID` (same in every terminal) |
| `no TF odom -> livox_frame` | terminal 1 bring-up not fully up yet; wait / restart it |
| `calib=WAIT` for > 20 s | stand on flat floor facing forward; or `bash scripts/run_survey.sh -p body_h:=0.30` |
| `ms` above ~80 | container busy; `bash scripts/run_survey.sh -p max_points:=6000` |
| labels not registering | T3 must print `recorded gait: ...` and T2 `GAIT LABEL -> ...`; run both on the container |
| report PNGs missing | OpenCV missing - run `report.py` elsewhere with the run folder copied |
