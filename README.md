# rammp — robot-assisted feeding, Kinova Gen3

A **Kinova Gen3 7-DoF** on a wheelchair mount at the user's right holds a **fixed utensil** (a fork or a spoon) in
a Robotiq 2F-85; the gripper is never commanded. There is no wrist force/torque sensor: all force sensing is joint
torque. A wrist RGB-D camera aims every pick; an optional fixed scene camera gives the coarse bowl and table. The
stack runs on a **Jetson AGX Orin** (the arm driver in a container, the models resident on its GPU), the free-space
paths come from **cuRobo** in its own GPU container.

The design history -- what was tried, measured and decided, with dates -- is in `docs/notes/`; this file says how
the stack works now. `CLAUDE.md` holds the working rules.

## Commands

```
make status        what is running, is the arm reachable
make state         where is the arm right now
make preflight     before any motion: tool, other clients, faults, kinematics vs the capture, limits, servers
                   (FIX=1: set our tool + limits). make feed runs it first; nothing moves
make site-ready    ready to feed, nothing moves: models + cameras, preflight, the bowl (scene camera), the detect plan

make planner       the cuRobo planner server (every free-space move needs it)
make start         to feeding_start: a cuRobo path from anywhere
make detect        to food_detect_start, over the bowl (PLAN=1: plan and check only)
make start-direct  to feeding_start by one joint move, no planner (SPEED=<deg/s>; a clear workspace only)
make perceive      at food_detect_start: capture, detect, depth, target
make acquire       over -> plunge -> scoop  (STOP_AFTER=over|plunge; aim checks: STOP_AFTER=hover30|hover10,
                   then START_AT=punch)
make deliver       swing -> present -> bite -> retract, from the end of the scoop
                   (STOP_AFTER=swing|present|bite, START_AT=present|bite|retract, VALIDATE=1)
make retract       back from the user to food_detect_start
make feed          bite after bite until the bowl is empty or the user declines, then feeding_start (BITES=1)

make review        the latest session: timeline, numbers, contact sheet per bite (RUN=<session|bite>)
make locate-scene  the coarse bowl + table from the scene camera -> runs_dir/scene_state.json
make locate-bowl   the bowl + table from the latest perceive (no scene camera: to set `workspace:`)
make capture       re-measure feeding_start from the parked arm
make models        every model loaded and warm, both camera servers up (make feed runs it)
make camera        the selected camera behind the camera server;  make depth: the Kinova vision nodes
make limits        the arm's speed limits (APPLY=1: from deployment.arm.soft_limits)
make tool          the arm's tool frame + mass vs ours (APPLY=1 sets ours, saving theirs; RESTORE=<file>)
make local         a machine's settings: config/machines/<MACHINE>.yaml -> config/local.yaml
make twin          the digital twin: replay a recorded bite in MuJoCo -> out/sim/*.mp4 (the Mac)
```

## A bite

```
feeding_start --cuRobo--> food_detect_start --perceive--> OVER, plunge, scoop --cuRobo, carrying--> at the user
   ^                       (computed over the bowl)       (scripted, guarded)                    look, approach,
   |                                                                                              bite (force)
   +------cuRobo------ food_detect_start <--cuRobo-- back off from the mouth (straight, scripted) <-----+
```

- **food_detect_start** is computed over the bowl the workspace reports (`kinova/scene.py`): the wrist camera
  `workspace.overlook_camera_height_m` (0.45) above the table, its lens axis through the bowl, in the captured
  overlook's orientation, joints by continuous IK out of the captured pose (the arm keeps its shape).
- **Perceive** (`feeding/stage1.sh`): a colour frame and a depth median from the camera server, LocateAnything-3B
  for the bowl and the food (one query per category; the food profile's prompt), depth at each detection, the
  target: the uppermost piece's top. Exit 3 = the bowl is empty (confirmed by depth), 5 = food left, none pickable.
- **Acquire** (`kinova/scripts/acquire.py`, one arm program): OVER translates at the overlook height until the
  utensil tip's vertical line meets the target (at most 10 cm); a fresh depth reading; one smooth plunge to the
  measured bowl floor (or 25 mm below a packed piece's top, never within 10 mm of the table), slowing through the
  food; the scoop turns the utensil horizontal as the food rises (a joint-space path pivoting on the tip), backing
  off toward the scoop end.
- **Deliver** (`kinova/scripts/deliver.py`, `feeding/deliver.sh`): the swing to the user is one cuRobo path with
  the piece on the utensil, to a goal 175 mm above the scoop end, turned 90 deg to the user, the utensil level and
  retracted along itself, 50 mm to the user's right (the food out of the camera's line to the mouth). Then a look
  (MediaPipe face landmarks; LocateAnything as the fallback), the open-mouth wait, a straight approach to 80 mm
  short of the mouth from a detection made at that pose (under 2 min old), the bite (tool force > 1 N for 0.1 s),
  a straight back-off (never stopped on torque: away from the face is the safe direction), and one cuRobo path
  back to food_detect_start.

## Free-space motion: cuRobo (`kinova/curobo_client.py`, `kinova/planner_server.py`)

`make planner` keeps cuRobo warm in its own GPU container (our tree mounted; on the site Jetson the other team's
`rammp-curobo` image, on an x86 box `docker/Dockerfile.planner_x86`). `curobo_client.travel(conn, goal, label)`
asks it for a collision-free joint path from where the arm is, checks every waypoint's utensil tip against our own
floor (`scene.tip_floor`: above the table over its footprint, above the bowl's rim over the bowl), runs it through
`kinova.motion.run_path` (the arm's soft limits, the controller's validation, the torque watch), and settles onto
the goal exactly. Its world (`kinova/curobo_world.py`):

| obstacle | from |
|---|---|
| table | the workspace's table height over `workspace.table_extent_m` |
| bowl | hollow: a floor and 12 wall boxes round the rim (the utensil works over its opening) |
| the user | head and torso boxes round the last measured mouth (`runs_dir/mouth_state.json`, else `motion.free_space.user_mouth_m`) |
| wheelchair | `sim/assets/wheelchair/wheelchair_boxes.yaml` (ramms-sim's Unreal model; rough height) |
| pedestal | the arm's mount |
| utensil, food | spheres on the wrist, from the tip model (`kinova/motion.tip_in_tool`); a piece in the "carrying" planner |

**The scene camera's error is clearance.** Its reading is trusted to `scene_camera.extrinsic.error_m` (5 cm): the
table and rim are that much higher in the planner's world and in our floor, the walls that much thicker. Planned
on what a mis-calibrated scene camera reports and judged on the truth (`tests/scene_error_study.py`, four camera
placements, 25 sampled errors each): at a 0.45 m overlook with the margin every plan succeeded, none touched the
true table or bowl (the tip >= 66 mm clear), and the true bowl was always in the wrist camera's view and within
OVER's reach; the tip peaks 22-25 cm above the table. At 0.40 m the margin leaves no room, and trusting the
reading touched the truth in 1-3 of 25 plans per camera.

Scripted, not planned: the moves that touch or must be straight (OVER, plunge, scoop, approach, back-off,
sidestep) and `make start-direct`.

## Every move goes through `kinova/scripts/safe_move.py` / `kinova/motion.py`

| guard | why |
|---|---|
| the utensil stays down on joint moves | tool pitch is J2+J4+J6; only complete 7-joint targets, time-synchronised, keep the sum interpolating linearly |
| a joint-torque step stops the move | the only contact detector (no wrist F/T, no self-collision model of the utensil) |
| joint limits pre-checked | Kortex silently ignores a target past a limit |
| fail fast | nothing moved within 3 s: stop and report the controller's own error report (read it first; the arm state misleads) |
| the tool frame matches ours | another user's tool frame shifts every cartesian target (`make tool`) |
| trajectories under the arm's limits | speed and acceleration planned under the soft limits and the controller's own acceleration measure, validated before they run |

The e-stop does not hold the arm still: it has moved 100-160 mm after one.

## Configuration (`config/`, read by `settings.py`)

```
config/rammp.yaml             deployment (arm, containers, ports, models), features, motion, guards, acquisition,
                              perception, delivery, the workspace
config/camera/<name>.yaml     intrinsics, depth registration and correction, the mount on the wrist, streams; a
                              scene camera's extrinsic and error
config/utensil/<name>.yaml    camera-to-tip, tip pixel, image region, tine offset (by_camera.<camera>: measured
                              against another camera)
config/food/<name>.yaml       the detector prompt, the aim offset
config/bowl/<name>.yaml       the captured overlook (orientation, arm shape), the scoop end, where they were captured
config/machines/<name>.yaml   a machine's own facts (make local copies it to config/local.yaml, never synced)
```

Later wins: `rammp.yaml` < the selected profiles < `config/local.yaml` < `RAMMP_SET="a.b=1,c=[1, 2]"`. Profiles:
`profiles:` or per run (`RAMMP_CAMERA=realsense_d405`, `RAMMP_UTENSIL`, `RAMMP_FOOD`, `RAMMP_BOWL`,
`RAMMP_SCENE_CAMERA`). Python: `from settings import cfg` (`cfg.delivery.bite_n`, `cfg.get("a.b")`); shell:
`python3 -m settings get <key>`, `python3 -m settings dump`. A machine's file holds only `deployment.root`
(`{root}`: venvs, models, caches, runs), `deployment.name` (`{name}`: containers and images), the arm's IP, its
cameras and a port only if one clashes.

**The workspace** (where the table and bowl are): the scene camera's latest reading when one is configured (`make
locate-scene`, valid `workspace.scene_max_age_s`), else `workspace:` in the config (`make locate-bowl` measures it
from the wrist camera). The scoop end moves with the bowl (`scoop_end_follows: bowl`) or only with the table.

## Cameras

Nothing reads a camera directly: frames and depth medians come from `feeding/camera_server.py` (`/frame`, `/mjpeg`,
`/median`); the profile's `streams.kind` says where it gets them -- `kinova_vision` (the arm's own module: depth
from our ROS 2 nodes, `make depth`; colour over RTSP) or `ros2` (any RGB-D camera publishing topics: the site D405,
the Orbbec scene camera). A new wrist camera: a profile (topics, intrinsics, `mount`: where it sits on the wrist,
or the file of another stack's measurement) and the utensil measured against it under `by_camera.<camera>`. The
scene camera's extrinsic is read in place from its calibration file; it is coarse (5 cm) and only ever places the
overlook -- the wrist camera aims.

## Calibration

| what | where | how |
|---|---|---|
| feeding_start | `calibration/home_pose.json` | park the arm, `make capture` (J1, J3 stored as the exact intended -90 / 0) |
| the captured overlook | `calibration/food_detect_pose.json` (named by the bowl profile) | its orientation and arm shape seed the computed one |
| camera-to-tip, tip pixel, image region | the utensil profile | at food_detect_start: a ruler from the tip to the bare table + the sensor's depth there; the tip where it meets the food (a 10 mm hover settles it) |
| tine offset, aim offset | utensil / food profile | at a 10 mm hover (`make acquire STOP_AFTER=hover10`) |
| depth correction | camera profile `depth.scale` / `offset_m` | a ruler check |
| the arm's tool, limits | `deployment.arm` | `make tool APPLY=1`, `make limits APPLY=1` |

Re-measure the utensil whenever it is changed **or re-seated**: a power-cycle re-initialises the gripper and can
pivot it (check at a 10 mm hover after one).

## The run record (`feeding/lib.sh`, `runlog.py`, `make review`)

```
<runs_dir>/<date>_<what>/          a session: session.json (host, commit, profiles, workspace), config.yaml (merged),
                                   console.log (timestamped), events.jsonl, moves/arm/ (detect / home telemetry)
  bite_NN/perceive/                colour, depth, the DA3 fill, detection, target
  bite_NN/acquire/                 where the floor was read, scoop_log.json, arm/: plan_*.json (every validated
                                   trajectory), *_arm_*.npz (telemetry: joints, velocities, torques, tool pose, force)
  bite_NN/deliver/look_N/          each look at the face: colour, depth, face.json, the open-mouth wait
  bite_NN/summary.json, contact_sheet.jpg
<runs_dir>/latest -> the current bite;  scene_state.json, mouth_state.json: what the planner's world reads
```

Recording never raises and never slows a move; without `RAMMP_RECORD` everything logs to the console only.

## Tests (no arm needed)

```
tests/offline.sh                    (the Mac, ~1 min) the unit tests + the whole bite for every camera combination
.venv/bin/python tests/site_check.py [tests/sites/<site>.yaml] [--planner stub|curobo] [--camera <p>]
                                    [--scene-camera <p>|none] [--dump motions.json] [--render]
python3 tests/scene_error_study.py  (a CUDA box) plans on a mis-calibrated scene camera's report, judged on the truth
bash tests/regress.sh check         (the Jetson) perception / measurement / face outputs on saved runs
.venv/bin/python tests/twin_check.py   the twin's arm vs the controller's tool pose on a recorded session
```

`site_check` runs the arm stages exactly as on the robot (their own checks and refusals) against the Gen3's real
kinematics (`tests/mujoco_kortex.py`: MuJoCo FK / IK on a simulated clock) and a site file: the cameras, the
workspace, the user's mouth; a synthetic depth frame stands in for the camera. `--planner stub` (default) answers
plans without a GPU (`tests/stub_planner.py`: up, across, down; no obstacles -- the plumbing, not collision
avoidance); `--planner curobo` starts the real planner server (a CUDA box: `docker run --gpus all -v $PWD:/work -w
/work cornell/curobo-eval:0.7.8 python3 tests/site_check.py --planner curobo`). Run `tests/offline.sh` before
every sync.

## Simulation (`sim/`)

- **The twin** (`make twin`): a recorded bite replayed in MuJoCo -- the arm from each move's telemetry, the person
  (SMPL-X, `sim/pose_human.py` once, licensed model in `sim/assets/smplx/`) placed on the measured mouth, the food
  where the detection put it, the real and twin wrist-camera frames side by side.
- **A site's bite** (`sim/compare_planners.py`): two `site_check --dump` recordings side by side on the site's
  geometry, in MuJoCo's renderer -- the quick look for experiments.
- **The final film**: `--export` the scene and poses, render on a GPU box with Blender's Cycles
  (`blender -b -P sim/blender_render.py -- out/sim/cmp/left`), `--compose` the frames with the HUD; ~2 s a frame on
  an RTX 3070. MuJoCo poses every frame; Blender only draws.

Everything generated goes under `out/` (git-ignored): `out/sim/` (videos, renders, the procedural textures in
`out/sim/cache/`), `out/captures/` (debug images). The presentation figures keep their own `presentation/out/`.

## Machines

| host | role |
|---|---|
| abra@192.168.1.11 | **the site Jetson** (AGX Orin, JetPack 6.2, Python 3.10), wired to the arm at 192.168.1.10. SHARED: ours is all under `~/cornell/` |
| debajyoti@cascadilla | x86, RTX 3070: the planner tests and the Blender renders (`~/cornell_curobo`, `~/cornell_render`) |
| debajyoti@jetson | the lab Jetson until 2026-09-25 (CUDA 13: not a cuRobo box) |
| the Mac | the source of truth: edit here, `./sync.sh [host]` pushes with `--delete` (never `config/local.yaml`) |

**The shared site Jetson.** Other people's `sheppy-*` containers run on it (their ROS 2 on domain 0, foxglove on
:8765, the cameras' drivers): leave them alone. Ours: `~/cornell/{rammp, venvs/la, venvs/mp, da3src, models,
.cache/huggingface, rammp_runs}`, images `cornell/*`, containers `cornell_*`, our ROS 2 on domain 42. Set up:
`./setup_jetson.sh`, `make local MACHINE=abra_jetson`, `make images driver planner models`. Remove:

```
docker rm -f $(docker ps -aq --filter name=cornell_); docker rmi $(docker images -q 'cornell/*')
pkill -f "cornell/venvs"; rm -rf ~/cornell
```

## Deployment day: a new site, in order

Nothing moves until step 5; every motion is attended.

1. **The machine**: `./setup_jetson.sh`, `make local MACHINE=<name>` (root, name, the arm's IP, the cameras),
   `make images driver planner`.
2. **The arm's settings** (another user may have changed them): `make tool` / `make limits`, `APPLY=1` sets ours
   (the replaced values are saved; `make tool RESTORE=<file>` puts them back).
3. **The cameras**: the scene camera's extrinsic is read from its calibration file; `make models` brings up the
   models and both camera servers.
4. **`make site-ready`**: the preflight, the bowl and table from the scene camera, the plan to food_detect_start.
   A kinematics FAIL: drive the arm to feeding_start and `make capture` (pull calibration/ back to the Mac before
   the next sync). Off site, check the geometry on real kinematics: `tests/site_check.py tests/sites/<site>.yaml`.
5. **Attended, slow**: `make start`, `make detect`, `make perceive`, the 10 mm hover check
   (`make acquire STOP_AFTER=hover10`).
6. `make feed BITES=1`, then `make feed`.

## Layout

```
settings.py, runlog.py       the config loader; logging and the run record
config/                      every tunable, switch and deployment value
calibration/                 home_pose.json, food_detect_pose.json, the Kinova depth sensor's intrinsics
kinova/
  connection.py              session, routers, teardown
  frames.py                  angles and Kortex's pose convention (no Kortex: shared with the detector and the planner)
  motion.py                  IK, continuous IK paths, time scaling, validated and watched trajectories, the tip model
  workspace.py, scene.py     where the table and bowl are; food_detect_start, the tip floor, the scoop end
  curobo_client.py           free-space moves: plan, check, run, settle
  curobo_world.py, planner_server.py   cuRobo's world and the planning service
  scripts/                   sequence (detect / home / direct), acquire, deliver, safe_move, preflight, arm_state,
                             arm_tool, arm_limits, capture_home
feeding/                     the stage scripts (stage1 / acquire / deliver / feed .sh, lib.sh), the camera server,
                             perception (perceive, locate_anything, locate_target, locate_scene, locate_bowl,
                             depth_fill, detect_mouth_mp, detect_mouth), review
tests/                       site_check, stub_planner, mujoco_kortex + fake_kortex, scene_error_study, unit_test,
                             regress.sh, twin_check, sites/
sim/                         the twin and the film: scene, record, twin, compare_planners, blender_render, assets/
docker/                      the driver, vision and x86 planner images
docs/notes/                  the design history
presentation/                the 2026-09 figures (their own README)
third_party/                 kinova-gen3-ros2 (submodule; the ROS 2 driver, not used by the stack)
```

## Known open

- **cuRobo has not driven the real arm yet**: the first `make detect` with it is attended and slow.
- **Mixed plates**: a pile of small pieces often detects as one box on the wrist camera ("food left, none
  pickable"); a sharper image fixes it (even light, a higher-resolution mode, or splitting a box by depth).
- **A new food**: test its prompt on a saved frame first; only fruit cubes and strawberries have been fed.
- **The wheelchair's height** in the planner's world is rough (ramms-sim's model); x / y likely right.
- **The two cameras disagreed by 8 cm** on the bowl at the site (2026-09-30): a tape measurement settles which.
- The bite often reads 0.0-0.3 s (the force is already over 1 N when the wait starts): harmless, not a bite time.
