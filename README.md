# rammp_feeding

Robot-assisted feeding on the Kinova Gen3, as a RAMMP module: it finds the bowl, picks a piece with a fork, waits for
the user's open mouth, feeds them and returns, one bite each time the user asks.

It runs as a capability engine (rammp-kitchen-capability-engines' `CapabilityEngine`, engine id `feeding`): the
orchestrator starts its actions, `start_meal`, `feed` (one bite) and `finish_meal`, and nothing moves otherwise
([interface](docs/interface.md)). The `rammp` command runs the same steps by hand, for setting up a site.

**Release 0.4.0:** `debajyotidasgupta/rammp-feeding:0.4.0` (Docker Hub, arm64), for the arm stack's 2.0 generation
(kinova-gen3-ros2 2.0.0, rammp-curobo 2.0.1); 0.2.5 is for the 1.1.0 stack. It is the `feeding` node of
rammp-deployments' `december_2026/sheppy-manifest.yaml`, copied from `rammp-alternative.yaml`.

## Getting started
On a new Jetson, follow the [deployment checklist](docs/porting.md). In short:
1. **Host:** network and software (checklist sections 1–4).
2. **Site file:** `/srv/rammp_feeding/config/local.yaml`, from [`config/machines/template.yaml`](config/machines/template.yaml):
   - the two camera profiles;
   - `utensil.glass_to_tip_m`: ruler, along the camera's axis;
   - `workspace.table_extent_m` and `workspace.bowl_centre_m`: tape, from the arm's base;
   - the printed board's `square_m`.
3. **Calibrate:** `sheppy up` without `feeding`, then `rammp capture home`, `rammp calibrate-wrist`, `rammp setup`
   and `rammp check`.
4. **Feed:** `rammp feed --bites 1` by hand; under the orchestrator, the `feeding` node's actions.

When the arm, the bowl or the user moves, run `rammp setup` (or one of its steps).

## Docs

| Page | Contents |
| --- | --- |
| [docs/index.mdx](docs/index.mdx) | What it does, the commands, the exit codes |
| [docs/porting.md](docs/porting.md) | The deployment checklist, and what to redo when the setup changes |
| [docs/interface.md](docs/interface.md) | The ROS interfaces |

## Layout
```
rammp_feeding/  engine.py (the node)  cli.py (the rammp command)  setup_steps.py (rammp setup)  ros.py (all ROS I/O)
                arm.py (guarded moves)
                kinematics.py  scene.py (workspace, tip model, planner world)  perception.py  acquire.py  deliver.py
                face.py  scene_camera.py  wrist_calibration.py  camera.py  geometry.py  config.py  record.py
config/         rammp.yaml (every tuned value), profiles (camera/, utensil/, bowl/, food/), machines/template.yaml
launch/  tests/
```

## Build and test
Tests run on an arm64 Docker host (a Jetson), not on a laptop:
```bash
docker build -t rammp_feeding:dev .
docker run --rm -v "$PWD:/work" -w /work rammp_feeding:dev bash tests/offline.sh   # unit tests + the whole bite (fake robot)
tests/test_ros.sh                                                                  # the ROS wiring against the simulated driver
ROBOT=<rammp-deployments config/robots/<robot>> tests/test_e2e.sh                  # a bite, and the node's actions (cancels too),
                                                                                   # through the real 2.0 driver, description and
                                                                                   # planner nodes (a CUDA host); then the node as
                                                                                   # sheppy starts it, stopping within 5 s
```
To release: build `debajyotidasgupta/rammp-feeding:<version>` from the tagged commit, push it, and pin that tag in the
manifest.
