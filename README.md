# dog-codes — Jueying Lite3 autonomy (Legged Robot Challenge, SUTD 2026)

Two independent code bases. Each folder is self-contained: run it from inside its own folder, nothing in one needs the other.

| Folder | What it is | Status |
|---|---|---|
| `frontier-fastlio/` | First approach: frontier exploration on a map built from FAST-LIO odometry (what was tested as "dog_codes3"). | Frozen reference. Runs as before. |
| `reactive-core/` | New approach: reactive lidar logic (lane rule, wall following, independent guard, 1 m grid memory, live map viewer). | Design spec only, no code yet. |

On the dog container:

```bash
git clone https://github.com/YWXCyrus/dog-codes.git ~/dog-codes
# old code:   see ~/dog-codes/frontier-fastlio/dog_codes/README.md
# new code:   see ~/dog-codes/reactive-core/   (spec: reactive-core/docs/specs/)
```

## frontier-fastlio/
- `dog_codes/` — the mission nodes (`mission.launch.py`, `scripts/arm.sh` ...). Start with its README.
- `dog_tools/` — hand-drive terrain survey tools.
- `sim_tests/` — WSL2 simulator and unit tests (`ROS_DOMAIN_ID=42` only, never the dog).

## reactive-core/
- `docs/specs/2026-11-20-reactive-nav-design.md` — the design (draft, under review).
- Code will be added here after the spec and the implementation plan are approved.

## Safety (applies to both)
Commands that move the real robot only with the dog on the hoist and someone holding STOP for first tests. Anything run on the container for testing uses a `ROS_DOMAIN_ID` other than 1.
