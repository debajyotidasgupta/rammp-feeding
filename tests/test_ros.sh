#!/usr/bin/env bash
# The module's ROS wiring against the real arm driver: kinova-gen3-ros2 1.1.0 in --sim (as rammp-deployments pins it),
# the stand-ins (tests/ros_stubs.py), and tests/ros_wiring.py, all in one network namespace; then `rammp check` there,
# with the test site's measurements and poses. Any arm64 host with Docker.
#   tests/test_ros.sh            (after: docker build -t rammp_feeding:dev .)
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=${IMG:-rammp_feeding:dev}
ARM=ghcr.io/rammp-org/kinova-gen3-ros2:1.1.0
docker rm -f rammp-arm-sim rammp-stubs >/dev/null 2>&1 || true
trap 'docker rm -f rammp-arm-sim rammp-stubs >/dev/null 2>&1 || true' EXIT
docker run -d --name rammp-arm-sim $ARM /module_ws/install/kinova_gen3_ros2/lib/kinova_gen3_ros2/kinova_gen3_node \
  --sim --urdf /module_ws/src/kinova-gen3-driver/models/gen3_7dof_2f85.urdf >/dev/null
docker run -d --name rammp-stubs --network container:rammp-arm-sim -v "$PWD/tests:/tests:ro" -v rammp-share:/share $IMG python3 /tests/ros_stubs.py >/dev/null
sleep 4
docker run --rm --network container:rammp-arm-sim -v "$PWD/tests:/tests:ro" -v rammp-share:/share $IMG python3 /tests/ros_wiring.py
out=$(docker run --rm --network container:rammp-arm-sim -v "$PWD/tests/poses:/tmp/s:ro" -e RAMMP_NO_LOCAL=1 -e RAMMP_STATE_DIR=/tmp/s \
  -e RAMMP_SET="deployment.runs_dir=/tmp/r,utensil.glass_to_tip_m=0.233,workspace.table_extent_m.x=[0.08, 0.76],workspace.table_extent_m.y=[-0.30, 0.80],workspace.bowl_centre_m=[0.290, 0.158],motion.free_space.user_mouth_m=[-0.277, 0.306, 0.428],utensil.tip_px=[456, 374],utensil.image_region_px=[280, 365, 660, 480],camera.mount.camera_in_tool_m=[0.0, 0.075, -0.125],camera.mount.rotation_quat_xyzw=[0.0, 0.0, 1.0, 0.0]" \
  $IMG ros2 run rammp_feeding rammp check || true)
echo "$out"
for name in arm trajectories planner "planner world" sam3 measured poses camera scene_camera; do
  echo "$out" | grep -qE "^  OK +$name( |$)" || { echo "CHECK FAILED: $name"; exit 1; }
done
echo "$out" | grep -qE "^  FAIL +scene pose +not measured" || { echo "CHECK FAILED: the scene pose row"; exit 1; }
echo "ROS OK"
