#!/usr/bin/env bash
# The module's ROS wiring against the real arm driver: kinova-gen3-ros2 1.1.0 in --sim (as rammp-deployments pins it),
# the stand-ins (tests/ros_stubs.py), and tests/ros_wiring.py, all in one network namespace. Any arm64 host with Docker.
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
docker run --rm --network container:rammp-arm-sim -e RAMMP_NO_LOCAL=1 -e RAMMP_STATE_DIR=/tmp/s \
  -e RAMMP_SET=deployment.runs_dir=/tmp/r $IMG ros2 run rammp_feeding rammp check
