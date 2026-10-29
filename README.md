# rammp_feeding

Robot-assisted feeding on the Kinova Gen3, as a RAMMP module. The module finds the bowl with a fixed scene camera,
aims a utensil held in the gripper at a piece of food with the wrist camera, scoops it, presents it at the user's
mouth once it opens, waits for the bite and returns. It repeats this until the bowl is empty or the user declines.

The module is a ROS 2 package (`rammp_feeding`) with an image (`Dockerfile`, `FROM rammp-base`) and a node for the
deployment manifest (`rammp-alternative.yaml`). It moves the arm only through the arm node. Free-space paths come from
the shared RAMMP-CuRobo planner; straight-line and contact moves are computed on the arm's own model.

Current release: 0.2.0, image `ghcr.io/rammp-org/rammp-feeding:0.2.0`.

## Documentation

| Page | Contents |
| --- | --- |
| [docs/index.mdx](docs/index.mdx) | Overview, commands, a feeding session, exit codes |
| [docs/porting.md](docs/porting.md) | Bringing the module up on a new robot or site: network, deployment, hand measurements, captures, calibrations, verification |
| [docs/interface.md](docs/interface.md) | ROS interfaces used and offered |
| [docs/notes/](docs/notes/README.md) | Engineering notes: measurements, decisions and their rationale |

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
                 machines/<site>.yaml (a site's local.yaml, versioned)
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

Release: `docker build -t ghcr.io/rammp-org/rammp-feeding:<version> .` from the tagged commit; the manifest pins that
tag.
