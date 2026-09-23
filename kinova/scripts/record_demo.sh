#!/usr/bin/env bash
# Record a teleop demo: arm state + wrist colour + wrist depth + the static D435i.
#
# Wall-clock stamped and started as close together as possible; alignment is
# done afterwards on those timestamps, not by assuming lockstep.
#
# The wrist RTSP feed DIES mid-recording (37 s of a 180 s run, previously
# observed; the session drop is a known defect in this camera path). So it is
# wrapped in a restart loop writing numbered segments -- losing three seconds at
# a seam is recoverable, losing the second half of a demonstration is not.
#
#   ./record_demo.sh /tmp/demo2 300
set -uo pipefail
OUT="${1:-/tmp/demo2}"
SECS="${2:-300}"
ARM_IP="${ARM_IP:-192.168.2.67}"
mkdir -p "$OUT"
AVAIL=$(df --output=avail -BM / | tail -1 | tr -d "M ")
echo "recording to $OUT for ${SECS}s   (disk free: ${AVAIL} MB)"
[ "$AVAIL" -lt 2000 ] && echo "  WARNING: under 2 GB free"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$OUT/start_utc.txt"

# 1. arm state, 40 Hz, high-level API (coexists with the gamepad)
docker exec tcp_rec python3 -u -m kinova.scripts.record_demo \
    --out /work_out --hz 40 --seconds "$SECS" > "$OUT/state.log" 2>&1 &

# 2. wrist colour, segmented with restart-on-death
(
  END=$(( $(date +%s) + SECS )); i=0
  while [ "$(date +%s)" -lt "$END" ]; do
    LEFT=$(( END - $(date +%s) )); [ "$LEFT" -le 2 ] && break
    i=$((i+1))
    timeout -s INT "$LEFT" gst-launch-1.0 -q -e \
      rtspsrc location=rtsp://$ARM_IP/color protocols=tcp latency=30 \
      ! rtph264depay ! h264parse config-interval=-1 ! queue \
      ! matroskamux ! filesink location="$OUT/wrist_$(printf %03d $i).mkv" \
      >> "$OUT/wrist.log" 2>&1
    echo "[seg $i ended at $(date -u +%H:%M:%S)]" >> "$OUT/wrist.log"
    sleep 1
  done
) &

# 3. static D435i colour (USB 2.0 -> 640x480@30)
timeout -s INT "$SECS" gst-launch-1.0 -q -e \
  v4l2src device=/dev/video4 ! "video/x-raw,width=640,height=480,framerate=30/1" \
  ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 \
  ! h264parse ! matroskamux ! filesink location="$OUT/face_cam.mkv" \
  > "$OUT/face_cam.log" 2>&1 &

# 4. wrist DEPTH via the already-running vision node
docker exec kinova_vision bash -lc \
  "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
   cd /tmp && rm -rf /tmp/depth_bag && timeout -s INT $SECS ros2 bag record \
   --compression-mode file --compression-format zstd -o /tmp/depth_bag \
   /camera/depth/image_raw /camera/depth/camera_info" > "$OUT/depth.log" 2>&1 &

echo "  RECORDING -- drive the full motion now. stops automatically after ${SECS}s"
wait
echo "done:"; ls -la "$OUT" | tail -8
