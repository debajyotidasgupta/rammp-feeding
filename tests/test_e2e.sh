#!/usr/bin/env bash
# End to end on the real arm stack, in a ROS domain of its own (any other stack on the host is left alone):
# kinova-gen3-ros2 2.0.0 --sim, its description node rendering a robot's mount, RAMMP-CuRobo 2.0.1's node with that
# robot's planner config; in the module's image, tests/e2e_bite.py (one bite through the pipeline) and
# tests/e2e_engine.py (the feeding engine driven through its actions), the rendered scene tests/fake_robot.py's; then
# the node as sheppy starts it, mock and real: /feeding/status up, and docker stop ends it within 5 s.
#   ROBOT=<rammp-deployments config/robots/<robot> folder> tests/test_e2e.sh      (after: docker build -t rammp_feeding:dev .)
# A CUDA host (the planner); the first run compiles the planner's kernels (a few minutes, kept in curobo_kernel_cache).
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=${IMG:-rammp_feeding:dev}
ARM=ghcr.io/rammp-org/kinova-gen3-ros2:2.0.0
PLANNER=ghcr.io/rammp-org/rammp-curobo:2.0.1
ROBOT=${ROBOT:?a robot config folder: mount.yaml and curobo/}
D="-e ROS_DOMAIN_ID=${E2E_DOMAIN:-77}"
names="rammp-e2e-arm rammp-e2e-model rammp-e2e-planner rammp-e2e-node"
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
logs() { echo "--- the planner's log"; docker logs --tail 30 rammp-e2e-planner; echo "--- the arm's log";
         docker logs --tail 20 rammp-e2e-arm; exit 1; }
for t in e2e_bite e2e_engine; do
  docker run --rm --network container:rammp-e2e-arm $D -v "$PWD/tests:/tests:ro" -v rammp-e2e-share:/share $IMG \
    python3 /tests/$t.py || logs
done
SRC=". /opt/ros/humble/setup.sh && . /ros2_ws/install/setup.sh && . /module_ws/install/setup.sh"
for mode in mock real; do                        # (the image's STOPSIGNAL, SIGINT, as docker stop sends it)
  if [ $mode = mock ]; then cmd="ros2 launch rammp_feeding feeding.launch.py mock:=true"
  else cmd="/module_ws/install/rammp_feeding/lib/rammp_feeding/feeding_node --ros-args -p state_dir:=/tmp/state"; fi
  docker run -d --name rammp-e2e-node --network container:rammp-e2e-arm $D $IMG $cmd >/dev/null
  docker exec rammp-e2e-node bash -c "$SRC && timeout 60 ros2 topic echo --once /feeding/status std_msgs/msg/String" | grep -q '"state": "idle"' \
    || { echo "the $mode node published no /feeding/status"; docker logs --tail 30 rammp-e2e-node; exit 1; }
  t0=$(date +%s%N); docker stop -t 10 rammp-e2e-node >/dev/null; ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  echo "  the $mode node: /feeding/status up; docker stop took ${ms} ms"
  [ "$ms" -lt 5000 ] || { echo "the $mode node took ${ms} ms to stop"; docker logs --tail 20 rammp-e2e-node; exit 1; }
  docker rm -f rammp-e2e-node >/dev/null
done
echo "E2E OK"
