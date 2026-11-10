#!/usr/bin/env bash
# One whole bite end to end on the real arm stack, in a ROS domain of its own (any other stack on the host is left
# alone): kinova-gen3-ros2 2.0.0 --sim, its description node rendering a robot's mount, RAMMP-CuRobo 2.0.1's node
# with that robot's planner config, and tests/e2e_bite.py in the module's image (the rendered scene: tests/fake_robot.py).
#   ROBOT=<rammp-deployments config/robots/<robot> folder> tests/test_e2e.sh      (after: docker build -t rammp_feeding:dev .)
# A CUDA host (the planner); the first run compiles the planner's kernels (a few minutes, kept in curobo_kernel_cache).
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=${IMG:-rammp_feeding:dev}
ARM=ghcr.io/rammp-org/kinova-gen3-ros2:2.0.0
PLANNER=ghcr.io/rammp-org/rammp-curobo:2.0.1
ROBOT=${ROBOT:?a robot config folder: mount.yaml and curobo/}
D="-e ROS_DOMAIN_ID=${E2E_DOMAIN:-77}"
names="rammp-e2e-arm rammp-e2e-model rammp-e2e-planner"
docker rm -f $names >/dev/null 2>&1 || true
docker volume rm rammp-e2e-share >/dev/null 2>&1 || true
trap 'docker rm -f $names >/dev/null 2>&1 || true; docker volume rm rammp-e2e-share >/dev/null 2>&1 || true' EXIT
docker run -d --name rammp-e2e-arm $D $ARM /module_ws/install/kinova_gen3_ros2/lib/kinova_gen3_ros2/kinova_gen3_node --sim >/dev/null
docker run -d --name rammp-e2e-model --network container:rammp-e2e-arm $D -v "$ROBOT:/config/robot:ro" $ARM \
  ros2 launch kinova_gen3_description description.launch.py mount_file:=/config/robot/mount.yaml >/dev/null
PARAMS=$(mktemp -d)                              # (as the deployment passes them: a params file)
printf 'rammp_curobo:\n  ros__parameters:\n    config: /config/robot/curobo/planner.yaml\n    world: ""\n' > "$PARAMS/planner.yaml"
docker run -d --name rammp-e2e-planner --network container:rammp-e2e-arm $D --runtime nvidia --gpus all \
  -v "$ROBOT:/config/robot:ro" -v rammp-e2e-share:/feeding:ro -v curobo_kernel_cache:/root/.cache \
  -v "$PARAMS/planner.yaml:/params.yaml:ro" $PLANNER \
  /opt/rammp_curobo/rammp_curobo_ros/lib/rammp_curobo_ros/planner_node --ros-args --params-file /params.yaml >/dev/null
docker run --rm --network container:rammp-e2e-arm $D -v "$PWD/tests:/tests:ro" -v rammp-e2e-share:/share $IMG \
  python3 /tests/e2e_bite.py || { echo "--- the planner's log"; docker logs --tail 30 rammp-e2e-planner; echo "--- the arm's log";
                                   docker logs --tail 20 rammp-e2e-arm; exit 1; }
echo "E2E OK"
