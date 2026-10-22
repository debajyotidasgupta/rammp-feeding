# Working on rammp_feeding

Robot-assisted feeding on the RAMMP Kinova Gen3, as a RAMMP module (README.md: the layout; docs/: how it runs;
docs/notes/: the design history -- what was tried, measured and decided, with dates).

## The arm and the people near it

- **Ask before anything that moves the arm** (a test, a run, a recovery move the user did not ask for): propose it
  in one line and wait for a go. The user attends every motion and is often recording. Keep diagnoses short; the
  quickest safe option first.
- **Never command the gripper.** It holds the utensil; nothing in the module opens or closes it.
- **Unknown pose** (after a power-cycle, an e-stop, a stop mid-move): `rammp state` first, then `rammp home` (a planned
  path from anywhere; the planner must be up).
- **After a power-cycle**: the utensil may be re-seated -- re-measure `tip_px` at a 10 mm hover before feeding.
- The e-stop does not hold the arm still (it has moved 100-160 mm after one).

## The platform

- **Reuse what the stack serves; never re-derive it.** The arm's model is `/robot_description`; camera intrinsics are
  `camera_info`; transforms between a camera's frames are its driver's `/tf_static`; detections are SAM3's
  (`/perception/sam3/detect`, boxes and masks) and the scene segmenter's (`/perception/object_detections`); every move
  goes through the arm node's `/execute_joint_trajectory`; free-space paths are the planner's. Ours: the utensil's tip
  model, the wrist camera's mount, the scene camera's pose (we measure it and publish it on `/tf_static`), and the moves
  the planner cannot express (straight lines, the scoop) computed on the arm's own model.
- It deploys as a node in rammp-deployments' `december_2026/sheppy-manifest.yaml`, transcribed from this repo's
  `rammp-alternative.yaml` (keep the two the same). Images are pinned to release tags (or say why not).
- Docs: https://github.com/rammp-org/rammp-org.github.io (the site), and each module's own `docs/`.

## Machines

- The Mac is the source of truth: edit and commit here, never on the robot. **No tests or renders on the Mac**: run
  `tests/offline.sh` and `tests/test_ros.sh` (arm64 Docker) on `debajyoti@jetson`, cuRobo on `debajyoti@cascadilla` or
  the Unicorn cluster's GPU nodes.
- **The site Jetson `abra@192.168.1.11` is shared** (the arm there is `192.168.1.10`). The `sheppy-*` containers are the
  deployment's nodes, run by sheppy: change what runs through the manifest and sheppy, with the other team's go. Our
  state is `/home/abra/.config/rammp_feeding` (mounted at `/config`), our record `/home/abra/rammp_runs`. Do not store
  its password anywhere.
- `debajyoti@jetson` (CUDA 13, an AGX Orin with a 512 GB NVMe at `/mnt/ssd`): evaluations; `/mnt/ssd/sam3eval` holds the
  SAM3 / LocateAnything comparison.

## The code

- **Every number that is tuned or changes with the deployment is in `config/`**, units in the key names; a site's own
  facts in its `/config/local.yaml` (`config/machines/<name>.yaml` is the versioned copy). Code reads `cfg`.
- **Comments say what the code does now** (and a one-line why where it is not obvious, safety reasons above all).
  History -- old values, dated decisions, incidents -- goes to `docs/notes/<area>.md`, never into the code or config.
- `ros.py` is the only module that imports ROS; `tests/fake_robot.py` implements the same interface.
- Free-space moves only through `Arm.travel` (the shared planner's path, checked against our tip floor, reach and the
  utensil's clearance; our world pushed first and the planner's own put back when the command ends). Contact and
  straight-line moves (OVER, plunge, scoop, approach, back-off) are our IK paths through `Arm.run`, with their guards.
- Everything generated goes under `out/` (git-ignored); the run record lives outside the tree (`runs_dir`).

The user's standing choices: delivery is open loop (the mouth measured once, as it opens); on demo day the food is
laid out spaced (no pile tuning); SAM3 is the detector.
