# Working on rammp

Robot-assisted feeding on a Kinova Gen3 (README.md: how the stack works; docs/notes/: the design history --
what was tried, measured and decided, with dates).

## The arm and the people near it

- **Ask before anything that moves the arm** (a test, a run, a recovery move the user did not ask for): propose it
  in one line and wait for a go. The user attends every motion and is often recording. Keep diagnoses short; the
  quickest safe option first.
- **Never command the gripper.** It holds the utensil; nothing in the tree opens or closes it.
- **Unknown pose** (after a power-cycle, an e-stop, a stop mid-move): `make state` first. `make start` plans a
  collision-free path from anywhere (the planner must be up: `make planner`). Without the planner,
  `make start-direct SPEED=10` only for a workspace the user says is clear: feeding_start parks below the table's
  top, beside it, and a straight joint line can pass close to the table.
- **After a power-cycle**: the utensil may be re-seated (re-measure tip_px, confirm at a 10 mm hover), restart
  `make depth` / `make models`, check `make limits` and `make tool` before feeding.
- The e-stop does not hold the arm still (it has moved 100-160 mm after one).

## Machines

- The Mac is the source of truth: edit here, `tests/offline.sh`, then `./sync.sh <host>` (`--delete`; never
  `config/local.yaml`). Never edit on a replica.
- **The site Jetson `abra@192.168.1.11` is shared.** Other people's `sheppy-*` containers, ports (foxglove :8765)
  and cameras' drivers: never stop, reuse or modify them. Everything of ours lives under `~/cornell/` (names
  `cornell_*` / `cornell/*`, our ROS 2 on domain 42); ask before installing anything there. Do not store its
  password anywhere. The arm there is `192.168.1.10`.
- **GPU work off the Jetson: `debajyoti@cascadilla`** (RTX 3070): `~/cornell_curobo/rammp` (a synced tree; rsync it,
  excluding `.git .venv out __pycache__ config/local.yaml sim/assets/smplx`) and the image
  `cornell/curobo-eval:0.7.8` (`docker/Dockerfile.planner_x86`):
  `docker run --rm --gpus all -e PYTHONUTF8=1 -v $PWD/rammp:/work -w /work cornell/curobo-eval:0.7.8 python3 tests/site_check.py --planner curobo`.
  Blender (OptiX) renders in `~/cornell_render`. `buttermilk` corrupts big docker pulls; the old `jetson` is CUDA 13
  (no cuRobo).
- Installs belong in the repo's setup scripts (`setup_jetson.sh`), never one-off edits inside a venv (e.g.
  `decord2` for LocateAnything on aarch64, not a stub).

## The code

- **Every number that is tuned or changes with the deployment is in `config/`**, units in the key names; a machine's
  own facts in `config/machines/<name>.yaml` -> `config/local.yaml`. Code reads `cfg`, never a literal.
- **Comments say what the code does now** (and a one-line why where it is not obvious, safety reasons above all).
  History -- old values, dated decisions, incidents, disproved theories -- goes to `docs/notes/<area>.md`, never into
  the code or the config.
- **Free-space moves only through `kinova/curobo_client.travel`** (cuRobo plans; our tip floor checks; `run_path`
  runs it under the arm's limits and the torque watch). Contact and straight-line moves (OVER, plunge, scoop,
  approach, back-off) stay scripted through `kinova/motion.py` / `safe_move.py`, with their guards.
- Shared pieces, imported rather than re-defined: `kinova/frames.py` (angles, Kortex poses; no Kortex import),
  `kinova/motion.py` (IK, paths, trajectories, the tip model), `kinova/workspace.py` / `kinova/scene.py` (the table
  and bowl, food_detect_start, the tip floor), `feeding/camera.py` (intrinsics, pixel mapping, the camera mount),
  `runlog.py` (logging, the run record).
- The scene camera is coarse (`scene_camera.extrinsic.error_m`, 5 cm): it only places the overlook, and its error
  is clearance in the planner's world and our floor. The wrist camera aims. Re-run `tests/scene_error_study.py` on a
  CUDA box after changing the overlook height, the margins or the world.
- Everything generated goes under `out/` (git-ignored). The run record lives outside the tree (`runs_dir`).

## Before a sync / a site day

```
tests/offline.sh                                   # the Mac: unit tests + the bite for every camera combination
tests/site_check.py --planner curobo [--camera ..]  # cascadilla: the same bite with the real planner
```

The user's standing choices: a spoon is on the arm but runs with the fork profile unchanged (2026-09-30); on demo
day the food is laid out spaced (piles defeat the wrist camera's detection: no pile tuning).
