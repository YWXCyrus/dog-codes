# dog-codes — Jueying Lite3 autonomy (Legged Robot Challenge, SUTD 2026)

Two independent code bases. Each folder is self-contained: run it from inside its own folder, nothing in one needs the other.

| Folder | What it is | Status |
|---|---|---|
| `frontier-fastlio/` | First approach: frontier exploration on a map built from FAST-LIO odometry (what was tested as "dog_codes3"). | Frozen reference. Runs as before. |
| `reactive-core/` | New approach: reactive lidar logic (lane rule, wall following, independent guard, 1 m grid memory, stairs in AI Basic, live map viewer). | Code + tests (simulator only); not yet run on the dog. |

On the dog container:

```bash
git clone https://github.com/YWXCyrus/dog-codes.git ~/dog-codes
# old code:   see ~/dog-codes/frontier-fastlio/dog_codes/README.md
# new code:   see ~/dog-codes/reactive-core/README.md   (python3 run.py; dry run by default)
```

## frontier-fastlio/
- `dog_codes/` — the mission nodes (`mission.launch.py`, `scripts/arm.sh` ...). Start with its README.
- `dog_tools/` — hand-drive terrain survey tools.
- `sim_tests/` — WSL2 simulator and unit tests (`ROS_DOMAIN_ID=42` only, never the dog).

## reactive-core/
- `README.md` — how to run it on the dog (`python3 run.py`, dry run by default; `--live` for the dog).
- `docs/specs/2026-11-20-reactive-nav-design.md` — the design; `docs/plans/` — the implementation plan.

## Safety (applies to both)
Commands that move the real robot only with someone holding STOP / the controller e-stop (there is no hoist; tests run on the course). Anything run on the container for testing uses a `ROS_DOMAIN_ID` other than 1.
