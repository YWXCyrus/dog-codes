# Competition morning (27 Nov) — branch `grid-nav`

## 0. Branch `integrated` (grid-nav + scene looks / reversing, README 0b)
Use `git checkout integrated` instead of `grid-nav` in A1. grid-nav itself ran live on 26 Nov (runs 20261126_065301 ... 071529).
- [ ] Reversing is OFF by default (runs 071529 / 093607: the dog did not move at -0.12 m/s). `--set grid_reverse=true`
      tries it again; the code notices after 2.5 s without motion and turns round instead.
- [ ] **Feed stalls**: runs 26 Nov had up to ~1 s old feeds (all four at once) while reactive-core used <= 33 ms
      per tick. Dry run with `top` open; then try `--set lio_restart=false` with no FAST-LIO running (grid mode
      does not use it). Compare the `stale:` guard lines in `decisions.jsonl`.
- [ ] **Scene look** (dry run, dog walked by hand to the cell in front of a scene room): terminal 2 prints
      `>> scene detected (i,j): looking from the doorway` (or `dead end`), then `>> <scene label>, (i,j)`;
      `ros2 topic echo /rc/look_cell` shows the room's cell. Live: the dog stops at the cell in front, looks ahead,
      45 deg left, 45 deg right, and never walks in.
- [ ] **Messages**: every `>> ...` line in terminal 2 is one thing the dog did (cells only, forward = (1,0), right = (1,-1)).
      `--set say=false` goes back to the raw `event: {...}` lines.
- [ ] **Wooden boards**: the dog drives over them; no `>> stairs confirm` for them. If the camera is asked, expect
      `>> not stairs (i,j) (camera)`.
- [ ] Perception: plain `ros2 launch ~/dog-codes/perception/perception.launch.py arena:=A` (odom now defaults to /rc/odom).
- [ ] **Camera model (`best.pt`, `rc.vision_node`): NOT started** (2026-11-26: CPU). Nothing else changes: stairs come
      from the lidar (when sure) or the VLM ("stairs?", answer waited 12 s, asked again up to 3x when it times out),
      curtains from the lidar + the VLM + the touch test. Sim without it (stairs, curtain_leaf, curtain_alt; VLM 3 s
      and 9 s): all pass. Expect `>> asking the camera: stairs?` then `>> stairs confirm (i,j) (camera)` at the stairs.

Everything below was built and tested overnight **in the simulator only** (no ROS on the build PC, no dog).
Nothing in it has run on the real dog yet. Prep time 09:00-11:45, prelims 12:15, software freeze before.
The old behaviour is one switch away: `--set nav_mode=wall`.

## A. Before the first run (5 min, at the laptop / container)
1. Get the branch on the dog: `cd ~/dog-codes && git fetch && git checkout grid-nav && git pull`.
2. Group id file: `cat ~/.dog_group_id` must show the registered `group##_XXXX`.
3. Which arena (left = A, right = B)? It goes into the perception launch (`arena:=A|B`, report URL `/task_report_A|B`).
4. `frontier-fastlio/dog_codes/config/mission.yaml`: fill in M02's hazard label if the task text names it
   (M01 is always `hand_tools`). Leave blanks if unknown.
5. (branch `integrated`: skip - the camera model is not started, see section 0.) Camera model: where is `best.pt`
   on the dog, and does `python3 -m rc.vision_node --model <path>` start (needs ultralytics)? If not, the
   curtain/stairs camera cues come only from the scene server (slower, ~4 s).

## B. Dry run with the dog walked by hand (15 min; the most important check)
Terminal 1 bring-up as usual. Terminal 2: `python3 run.py` (dry run, sends nothing). After "Ready", press
Enter, then **drive the dog by controller** through 4-6 cells of the practice arena while watching
`http://<dog>:8081` on the laptop:
- [ ] the dog's arrow moves with the dog and stays in the right cell; the status line `pos (x, y)` matches
      the cell (cell centres are multiples of 1.02 m). If the pose drifts or jumps by a cell: grid snap is
      fooled -> try `--set snap=false` and compare; tell me what the panels look like (height, gaps).
- [ ] borders turn **solid** for panels and disappear for openings. Measured: panels 0.43 m, curtain
      0.15-0.93 m (bands set for that: panel band 0.20-0.38, above-panel band 0.55-0.85, snap 0.18-0.40).
- [ ] the **curtain** border shows **dashed**: the lidar sees it as TALLER than a panel. If it stays solid,
      `--set edge_tall_ratio=0.15`; if real panels turn dashed, `--set edge_tall_ratio=0.5` (or `edge_tall_curtain=false`
      and rely on the camera).
- [ ] the viewer page itself renders (it was written blind: only checked by script, never in a browser).
- [ ] `ros2 topic echo /rc/odom --once` gives the arena pose; no Python tracebacks in terminal 2.

## C. First live runs (e-stop held)
1. `python3 run.py --live --set run_time_s=90` — watch: centring in each cell, turning only at centres,
   stopping 1 s at each new cell, no wall contact.
2. Speed: hand driving crossed the boards at 0.3-0.4 m/s; the code drives 0.20. If C1 looks calm, try
   `--set cruise_vx=0.25` (the whole 15-cell tour took ~200-290 s in the simulator at 0.20).
3. Optional smoother driving: `--set walk_wz_max=0.15` (steer while walking; the team found slips at ledges
   when walking + turning, so only if it behaves on the boards).
4. Stairs: the planner leaves them for last. At the cell before the stairs it squares up and starts the
   existing AI sequence (Enter AI there, exit after ~1 m flat = the cell after: booklet F.6). If the lidar
   detector is not sure after 4 s it approaches on the grid (`stairs_grid_fallback`). Check once.
5. Final cell: the run ends with `*** FINISHED ... ***` and the dog standing at the centre of the +1 cell.

## D. Perception next to it (terminal 3, after "Ready")
`ros2 launch ~/dog-codes/perception/perception.launch.py odom_topic:=/rc/odom arena:=A` (or B).
`odom_topic:=/rc/odom` matters: that is the snapped arena pose, so reported `pos` lands in the right cell.

## E. Decisions only you can make
- [ ] Deploy `grid-nav` or stay on `main` / `testing` / `perception-standalone`? (`grid-nav` is main +
      grid navigation; the curtain/QR modules of `perception-standalone` and the planner/bar code of `testing`
      are NOT merged — only `testing`'s `rc/vision_node.py` was copied.)
- [ ] `explore_budget_s` (450 s): when to give up exploring and head for the final cell (12 min session
      includes up to 3 drone launches).
- [ ] `cur_probe_touch` (false): with exploration done and arena cells still unreached behind a "wall", and
      the camera giving no answer, should the dog push that wall at 0.12 m/s (touch test, may count as a
      collision)? Default: no.
- [ ] Start pose: the dog must start at the start-cell centre facing the designated forward direction
      (the arena frame is defined by it).

## F. If something goes wrong on the course
| symptom | quick fix |
|---|---|
| map/pose nonsense, dog lost | `--set nav_mode=wall` (old behaviour) |
| walls drawn where there are openings | `--set edge_wall_ratio=0.75` |
| stops before doorways (`grid_block ... lane blocked`) | `--set lane_w=0.55` |
| never goes to the final cell | `--set explore_budget_s=300` |
| curtain never tried | check it is dashed in the viewer; `--set cur_ask_camera=false` (push on lidar alone) |
| stairs HOLD `stairs_unsure` | Captain takes them (0 points for the stairs), Enter resumes |
| (integrated) walks into a scene room | its far wall was not seen as WALL yet: nothing to set; `scene_prop_min` only decides scene vs dead end |
| (integrated) treats a corridor end as a scene and skips it | `--set scene_look=false` (drives in as grid-nav did) |
| (integrated) want it to reverse instead of turning round | `--set grid_reverse=true` (off by default: the dog did not move backwards) |
