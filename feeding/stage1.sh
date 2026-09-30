#!/usr/bin/env bash
# Stage 1 perception, from food_detect_start: capture -> detect -> locate.
#
#   feeding/stage1.sh                    # results in the bite's perceive/ (see feeding/lib.sh)
#   STAGE1_DA3=1 feeding/stage1.sh       # also run the optional DA3 cross-check
#
# Run on its own (make perceive) it starts a session and a bite of its own; under make feed, the bite
# feed.sh started. Exit 3 = no food in the bowl.
#
# Runs ON THE JETSON. Writes OUTSIDE ~/rammp: that tree is a replica that
# sync.sh overwrites with --delete, and results written into it are lost.
#
# Refuses unless the arm is at food_detect_start. locate_target's image->motion
# mapping (+X_base moves the scene down, +Y_base moves it right, at f/Z) was
# measured in exactly that orientation and is wrong in any other.
set -euo pipefail
cd "$(dirname "$0")/.."
source feeding/lib.sh
session_begin perceive
[ -n "${RAMMP_BITE:-}" ] || bite_begin       # make feed starts each bite; on its own, perceive starts one
stage_begin perceive
STILL_SINCE=$(date +%s.%N)                   # the arm is at rest at food_detect_start from here: depth after this counts
OUT="${1:-$STAGE_DIR}"
ARM_IP="${ARM_IP:-$(S deployment.arm.ip)}"
mkdir -p "$OUT"

echo "$(date +%T) [1/4] arm must be at food_detect_start"
TOOL=$(docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work -w /work "$DRIVER" python3 -c "
import json, sys
from settings import cfg
from kinova.connection import kinova_connection
from kinova.scripts.safe_move import joints, tool_pose, fold
from kinova.scene import overlook_joints
with kinova_connection() as c:
    want = overlook_joints(c); q = joints(c); p = tool_pose(c)
off = max(abs(fold(a - b)) for a, b in zip(q, want))
if off > cfg.motion.near_pose_deg:
    sys.exit('not at food_detect_start (%.1f deg away) -- run: make detect' % off)
print('%.4f %.4f %.4f' % tuple(p[:3]))
" 2>/dev/null)
echo "      tool at $TOOL"

echo "$(date +%T) [2/4] capture colour + depth"
# one frame in ~1 s (was: record the stream until a 22 s timeout, keep the last frame)
PYTHONPATH=. "$MPY" -m feeding.grab_frame "$OUT/colour.jpg" || echo "      (no colour frame -- make camera / make models)"
# the resident depth server (make models): fresh frames in ~0.5 s; else start a probe (~5 s)
if ! PYTHONPATH=. "$PY" -m feeding.depth_grab "$OUT/depth.npy" --frames "$FRAMES" --since "$STILL_SINCE" | sed -u 's/^/      /' && [ "$CAM_KIND" = kinova_vision ]; then
  docker cp feeding/depth_probe.py "$VISION":/tmp/depth_probe.py >/dev/null
  docker exec "$VISION" bash -lc "source /opt/ros/humble/setup.bash; \
    source /ros2_ws/install/setup.bash; cd /tmp && python3 depth_probe.py --out /tmp/s1_depth --frames $FRAMES" \
    | grep --line-buffered -E "valid|p50" | sed -u 's/^/      /'
  docker cp "$VISION":/tmp/s1_depth.npy "$OUT/depth.npy" >/dev/null 2>&1 || echo "      (no depth frame from the sensor)"
fi
# fallback: fill sensor holes (or a missing depth frame) from the resident Depth Anything 3
PYTHONPATH=.:$DA3SRC/src "$PY" -m feeding.depth_fill "$OUT" 2>/dev/null | sed -u 's/^/      /' || true

echo "$(date +%T) [3/4] detect"
BOWL_CACHE=(); [ -n "$(S features.bowl_once_per_session)" ] && BOWL_CACHE=(--bowl-cache "$RAMMP_SESSION/bowl.json")
PYTHONPATH=. "$PY" -m feeding.perceive --image "$OUT/colour.jpg" --out "$OUT/detection" "${BOWL_CACHE[@]}" 2>/dev/null \
  | grep --line-buffered -E "^  [a-z]|^     [0-9]"

DA3_ARG=()
if [ "${STAGE1_DA3:-0}" = 1 ]; then
  PYTHONPATH=.:$DA3SRC/src "$PY" -m feeding.run_depth_anything3 \
    --image "$OUT/colour.jpg" --out "$OUT/da3" 2>/dev/null | grep -E "time|metric" | sed 's/^/      /'
  DA3_ARG=(--da3 "$OUT/da3.npy")
fi

echo "$(date +%T) [4/4] locate"
set +e
PYTHONPATH=. "$PY" -m feeding.locate_target --det "$OUT/detection.json" --depth "$OUT/depth.npy" \
  "${DA3_ARG[@]}" --image "$OUT/colour.jpg" --tool $TOOL --out "$OUT/target" 2>/dev/null
rc=$?; set -e
ev perceived ", \"rc\": $rc, \"target\": $([ -f "$OUT/target.json" ] && echo true || echo false)"   # 3 = no food
echo
echo "results: $OUT"
exit $rc                         # 3 = no food in the bowl, 5 = food left but no pickable piece detected
