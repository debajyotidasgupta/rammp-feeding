# rammp_feeding

Robot-assisted feeding on the RAMMP Kinova Gen3, as a RAMMP module: a ROS 2 package (`rammp_feeding`), its image
(`Dockerfile`, `FROM rammp-base`), and its node for the deployment manifest (`rammp-alternative.yaml`, transcribed into
rammp-deployments' `december_2026/sheppy-manifest.yaml`). User docs: [`docs/`](docs/index.mdx); the design history:
[`docs/notes/`](docs/notes/README.md).

```
rammp_feeding/   cli.py (the rammp command)  ros.py (everything ROS: the arm, the planner, the cameras, SAM3, tf)
                 arm.py (guarded moves)  kinematics.py (the arm's model: FK, IK)  scene.py (the workspace, poses, the
                 planner world)  perception.py (the target, under the tip)  face.py (MediaPipe)  acquire.py  deliver.py
                 scene_camera.py (the coarse bowl, the camera's calibration)  camera.py  geometry.py  config.py  record.py
config/          rammp.yaml + profiles (camera, utensil, bowl, food); machines/ (a site's local.yaml)
calibration/     the default captured poses (a site's own: /config)
urdf/            the arm driver's model (the fallback when /robot_description is not published)
tests/           offline.sh (no ROS: unit tests + the whole bite against tests/fake_robot.py)
                 test_ros.sh (the ROS wiring against kinova-gen3-ros2 --sim and stand-ins, in Docker)
```

```bash
PY=.venv/bin/python tests/offline.sh               # any machine: numpy, scipy, Pillow, PyYAML
docker build -t rammp_feeding:dev . && tests/test_ros.sh   # an arm64 host with Docker (a Jetson, an Apple-silicon Mac)
```
