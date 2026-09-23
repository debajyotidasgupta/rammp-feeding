#!/usr/bin/env bash
# Stage 1 perception, from food_detect_start: capture -> detect -> locate.
#
#   feeding/stage1.sh                    # results in ~/rammp_runs/<timestamp>/
#   STAGE1_DA3=1 feeding/stage1.sh       # also run the optional DA3 cross-check
#
# Runs ON THE JETSON. Writes OUTSIDE ~/rammp: that tree is a replica that
# sync.sh overwrites with --delete, and results written into it are lost.
#
# Refuses unless the arm is at food_detect_start. locate_target's image->motion
# mapping (+X_base moves the scene down, +Y_base moves it right, at f/Z) was
# measured in exactly that orientation and is wrong in any other.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/rammp_runs/$(date +%Y%m%d_%H%M%S)}"
ARM_IP="${ARM_IP:-192.168.2.67}"
PY="$HOME/venvs/la/bin/python"
mkdir -p "$OUT"

echo "[1/4] arm must be at food_detect_start"
TOOL=$(docker exec -e PYTHONDONTWRITEBYTECODE=1 tcp_rec python3 -c "
import json, sys
from kinova.connection import kinova_connection
from kinova.scripts.safe_move import joints, tool_pose, fold
want = json.load(open('/work/calibration/food_detect_pose.json'))['joints_deg']
with kinova_connection() as c:
    q = joints(c); p = tool_pose(c)
off = max(abs(fold(a - b)) for a, b in zip(q, want))
if off > 1.0:
    sys.exit('not at food_detect_start (%.1f deg away) -- run: make detect' % off)
print('%.4f %.4f %.4f' % tuple(p[:3]))
" 2>/dev/null)
echo "      tool at $TOOL"

echo "[2/4] capture colour + depth"
rm -f /tmp/s1_*.jpg
timeout 22 gst-launch-1.0 rtspsrc location=rtsp://$ARM_IP/color protocols=tcp latency=200 \
  ! rtph264depay ! h264parse ! openh264dec ! videoconvert ! jpegenc \
  ! multifilesink location=/tmp/s1_%03d.jpg >/dev/null 2>&1 || true
cp "$(ls /tmp/s1_*.jpg | tail -1)" "$OUT/colour.jpg"; rm -f /tmp/s1_*.jpg
docker cp feeding/depth_probe.py kinova_vision:/tmp/depth_probe.py >/dev/null
docker exec kinova_vision bash -lc "source /opt/ros/humble/setup.bash; \
  source /ros2_ws/install/setup.bash; cd /tmp && python3 depth_probe.py --out /tmp/s1_depth --frames 20" \
  | grep -E "valid|p50" | sed 's/^/      /'
docker cp kinova_vision:/tmp/s1_depth.npy "$OUT/depth.npy" >/dev/null

echo "[3/4] detect"
PYTHONPATH=. "$PY" -m feeding.perceive --image "$OUT/colour.jpg" --out "$OUT/detection" 2>/dev/null \
  | grep -E "^  [a-z]|^     [0-9]"

DA3_ARG=()
if [ "${STAGE1_DA3:-0}" = 1 ]; then
  PYTHONPATH=.:$HOME/da3src/src "$PY" -m feeding.run_depth_anything3 \
    --image "$OUT/colour.jpg" --out "$OUT/da3" 2>/dev/null | grep -E "time|metric" | sed 's/^/      /'
  DA3_ARG=(--da3 "$OUT/da3.npy")
fi

echo "[4/4] locate"
PYTHONPATH=. "$PY" -m feeding.locate_target --det "$OUT/detection.json" --depth "$OUT/depth.npy" \
  "${DA3_ARG[@]}" --image "$OUT/colour.jpg" --tool $TOOL --out "$OUT/target" 2>/dev/null
echo
echo "results: $OUT"
