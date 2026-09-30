#!/usr/bin/env bash
# Bring up the camera server for one camera role, whatever the camera (make camera / make models):
#
#   feeding/camera_up.sh camera          # the wrist camera (profiles.camera)
#   feeding/camera_up.sh scene_camera    # the fixed scene camera (profiles.scene_camera; nothing if none)
#
# <role>.streams.kind in the selected profile says how:
#   kinova_vision  the arm's vision module: the server runs inside our vision container (make depth starts its
#                  depth nodes), on its depth topics; colour is RTSP from the arm, not served here
#   ros2           any ROS 2 camera's colour + depth topics, published by whatever drives the camera: the server
#                  runs in its own container on that camera's ROS domain / DDS config (<role>.streams.ros)
set -euo pipefail
cd "$(dirname "$0")/.."
ROLE="${1:?camera or scene_camera}"
S() { python3 -m settings get "$1"; }
KIND="$(S "$ROLE.streams.kind" 2>/dev/null || true)"
[ -n "$KIND" ] || { echo "no $ROLE configured (profiles.$ROLE)"; exit 0; }
PORT="$(S deployment.servers.${ROLE}_port)"
if curl -s -m 2 "http://127.0.0.1:$PORT/ping" | grep -q ok; then echo "$ROLE server already up (:$PORT)"; exit 0; fi
DEPTH="$(S "$ROLE.streams.depth_topics")"
SRV=feeding/camera_server.py
case "$KIND" in
  ros2)
    NAME="$(S "deployment.containers.$ROLE")"; IMG="$(S deployment.images.vision)"
    COLOUR="$(S "$ROLE.streams.colour_topic")"
    DOMAIN="$(S "$ROLE.streams.ros.domain_id")"; DDS="$(S "$ROLE.streams.ros.cyclonedds_uri")"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --network host --ipc host --restart unless-stopped \
      -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e ROS_DOMAIN_ID="$DOMAIN" \
      ${DDS:+-e CYCLONEDDS_URI=file:///dds.xml -v "$DDS":/dds.xml:ro} \
      -v "$PWD/$SRV":/camera_server.py:ro --entrypoint bash "$IMG" -lc \
      "source /opt/ros/humble/setup.bash; exec python3 /camera_server.py --port $PORT --depth $DEPTH --colour $COLOUR" >/dev/null
    echo "starting the $ROLE server ($NAME, :$PORT): $COLOUR + $DEPTH, ROS domain $DOMAIN" ;;
  kinova_vision)
    VISION="$(S deployment.containers.vision)"
    docker exec "$VISION" pkill -9 -f "[c]amera_server.py" >/dev/null 2>&1 || true
    docker cp "$SRV" "$VISION":/tmp/camera_server.py >/dev/null
    docker exec -d "$VISION" bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
      cd /tmp && python3 camera_server.py --port $PORT --depth $DEPTH > /tmp/camera_server.log 2>&1"
    echo "starting the $ROLE server (in $VISION, :$PORT): $DEPTH" ;;
  *) echo "$ROLE: unknown streams.kind '$KIND'"; exit 1 ;;
esac
