# sim_tests — offline checks for dog_codes (WSL2 Ubuntu-22.04, ROS 2 Humble, domain 42). NOT for the dog.

Needs: `~/ws` (transfer_interfaces), `python3-qrcode` (apt) for the simulated QR code.
Run from WSL:  `bash run_sim_test.sh <seconds> [--start-sitting] [--stair-ok general,grip] [--start x,y,yaw_deg]`

| script | what |
|---|---|
| `sim_world.py` | simulated dog + arena (walls, rubble, curtain, stairs, hazard sign, QR, survivors), lidar/camera/IMU/joints, mock scene server :8000 and scoring server :8001 (scores M01-M05 against ground truth) |
| `run_sim_test.sh` | full mission launch against the sim, prints supervisor mode every 10 s and the scoreboard |
| `run_gate_test.sh` / `test_gate.py` | 14 behavioural checks of the safety gate |
| `run_dry_test.sh` | dry_run really publishes nothing towards the dog |
| `run_manual_test.sh` | joystick takeover -> PAUSED -> resume |
| `dbg.sh`, `dbg_node.sh`, `summ.sh`, `env_check.sh` | debugging helpers (explorer map PNG, single node + topic echo, log digest) |


## Trial-2 fixes (20 Nov 2026)
- `test_dog_codes_trial2.py` — unit tests of `BumpDetector`, `OdomWatchdog`, `rear_fence_mask` and replays of the real recorded runs
  in `real_data/` (`trial2_run1003.csv`, `trial2_run1007.csv`, `trial1_walk0726.csv`, `survey1_hand_driven_samples.csv`).
  `python3 -B -m unittest -v test_dog_codes_trial2`
- `sim_world.py --low-back` : the wall behind the start is only 0.30 m high, solid but invisible to the (pitched) lidar.
  `sim_world.py --odom-fault-at 20` : 20 s after the first commanded motion the odometry runs away at 1.5 m/s.
- `DOG_CODES=/path/to/other/dog_codes bash run_sim_test.sh 100 --low-back` runs another copy of dog_codes (A/B tests).
