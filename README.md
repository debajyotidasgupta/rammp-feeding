# rammp_feeding

Robot-assisted feeding on the Kinova Gen3, as a RAMMP module. The module finds the bowl with a fixed scene camera,
aims a utensil held in the gripper at a piece of food with the wrist camera, scoops it, presents it at the user's
mouth once it opens, waits for the bite and returns. It repeats this until the bowl is empty or the user declines.

The module is a ROS 2 package (`rammp_feeding`) with an image (`Dockerfile`, `FROM rammp-base`) and a node for the
deployment manifest (`rammp-alternative.yaml`). It moves the arm only through the arm node. Free-space paths come from
the shared RAMMP-CuRobo planner; straight-line and contact moves are computed on the arm's own model.

Current release: 0.2.3, image `debajyotidasgupta/rammp-feeding:0.2.3` on Docker Hub (arm64, Jetson).

**Deploying on a new Jetson:** follow [docs/porting.md](docs/porting.md), the deployment checklist, from start to end.

## Before starting sheppy: what to fill in

The module's node is the `feeding` node of rammp-deployments' `december_2026/sheppy-manifest.yaml`. Until
[rammp-deployments#139](https://github.com/rammp-org/rammp-deployments/pull/139) is merged, it is on that repo's
`feature/feeding-node` branch. All positions below are in the arm's base frame (x forward, y left, z up), in metres.

**Before `sheppy up`:**

1. **`/srv/rammp_feeding/config/local.yaml`**: the site settings. Copy
   [`config/machines/template.yaml`](config/machines/template.yaml) and fill in everything marked REQUIRED:
   - `profiles.camera`, `profiles.scene_camera`: your wrist and scene camera models (e.g. `realsense_d405`,
     `orbbec_gemini336l`)
   - `utensil.glass_to_tip_m`: ruler, the wrist camera's front glass straight out to the fork tips
   - `bowl.floor_above_table_m`: ruler, the bowl's inner floor above the table
   - `workspace.table_extent_m`: the table's edges from the arm base, `{x: [near, far], y: [right, left]}`
   - `workspace.bowl_centre_m`: where the bowl will be placed, `[x, y]`
   - `motion.free_space.user_mouth_m`: the seated user's mouth, `[x, y, z]`
   - `perception.wrist_calibration.square_m`: the printed ChArUco square's size (only when calibrating with the
     board)
2. **`december_2026/config/robots/<your_robot>/mount.yaml`** (rammp-deployments): `robot: <your_robot>` and
   `rpy_deg: [0.0, 0.0, 0.0]` (the arm must be level). Then
   `uv run scripts/set_robot.py december_2026/config/robots/<your_robot>`.
3. **`december_2026/profiles/<your_robot>.yaml`** (rammp-deployments): what sheppy starts. Use the template in the
   checklist, section 4. Fill in your wrist camera's serial (`serial_no`) and scene camera's serial (`serial_number`).
   Leave `feeding` out for now.
4. **`december_2026/sheppy-manifest.yaml`**: point the `sam3` node's weights mount at your downloaded
   `facebook/sam3` snapshot.

**After `sheppy up` (without `feeding`):**

5. **Arm poses**, not typed by hand: pose the arm and run `rammp capture home`, `overlook`, `scoop_end`, `deliver`.
6. **Calibration**, either:
   - **run it:** `rammp calibrate-wrist`, then `rammp calibrate-scene`, with the printed ChArUco board; or
   - **enter values you already have:**
     - `/srv/rammp_feeding/config/wrist_mount.json`: `camera`, `utensil`, `camera_in_tool_m`, `rotation_quat_xyzw`,
       `tip_px`, `image_region_px`
     - `/srv/rammp_feeding/config/scene_extrinsic.json`: `xyz`, `quat_xyzw`

     The formats and frames are in [the checklist, 8b](docs/porting.md#8b-entering-an-existing-calibration-no-board).
7. **`rammp check`**: every row must read OK. Whatever is missing, it names.

Only then add `feeding` to your profile and start it, with someone at the e-stop.

## Documentation

| Page | Contents |
| --- | --- |
| [docs/index.mdx](docs/index.mdx) | Overview, commands, a feeding session, exit codes |
| [docs/porting.md](docs/porting.md) | The deployment checklist for a new Jetson: hardware, host and network, software, the deployment manifest, the site file and hand measurements, captures, calibrations (with the board, or an existing calibration's values), verification, feeding |
| [docs/interface.md](docs/interface.md) | ROS interfaces used and offered |

## Layout

```
rammp_feeding/   cli.py             the rammp command
                 ros.py             all ROS I/O: the arm, the planner, the cameras, SAM3, tf
                 arm.py             guarded moves (contact, tilt, path checks); cuRobo paths through Arm.travel
                 kinematics.py      FK, Jacobian and IK on the arm's URDF
                 scene.py           the workspace, the tip model, poses, the planner world
                 perception.py      the target piece; what is under the tip
                 acquire.py         over, plunge, scoop; the hover route
                 deliver.py         swing, present, bite, retract
                 face.py            MediaPipe face landmarks, the mouth
                 scene_camera.py    the coarse bowl and table; the scene camera's calibration
                 wrist_calibration.py   the wrist camera's mount and the utensil's pixels, from a ChArUco board
                 camera.py  geometry.py  config.py  record.py
config/          rammp.yaml (every tuned value) and profiles: camera/, utensil/, bowl/, food/;
                 machines/template.yaml (the site file every installation fills in: its local.yaml)
urdf/            the arm driver's model, used when /robot_description is not served
launch/          feed.launch.py (rammp feed, as the deployment runs it)
tests/           offline.sh, test_ros.sh and their fixtures
```

A site's own state (`local.yaml`, the captured poses, the calibrations, the scene reading) lives outside the tree, in
the state directory mounted at `/config`. The run record goes to the directory mounted at `/runs`.

## Build and test

Tests and renders run on an arm64 Docker host (a Jetson), not on a development laptop.

```bash
docker build -t rammp_feeding:dev .
docker run --rm -v "$PWD:/work" -w /work rammp_feeding:dev bash tests/offline.sh   # unit tests and the whole bite
                                                                                   # against tests/fake_robot.py
tests/test_ros.sh                                  # the ROS wiring against kinova-gen3-ros2 --sim and stand-ins
```

`tests/offline.sh` needs no ROS: numpy, scipy, Pillow, PyYAML and OpenCV. `tests/test_ros.sh` runs the real arm driver
in simulation, stand-ins for the planner, SAM3 and the cameras, and `rammp check` against them.

Release: `docker build -t debajyotidasgupta/rammp-feeding:<version> .` on a Jetson from the tagged commit, then
`docker push`; the manifest pins that tag.
