#!/usr/bin/env bash
# Food delivery, from the end of the scoop. Runs ON THE JETSON. Built 2026-09-24.
#
#   feeding/deliver.sh                         # swing -> present -> bite -> retract (to food_detect_start)
#   STOP_AFTER=present feeding/deliver.sh      # stop at a stage: swing | present | bite
#   START_AT=retract feeding/deliver.sh        # resume: present | bite | retract
#   VALIDATE=1 feeding/deliver.sh              # plan and validate the swing / approach; nothing moves
#   WAIT_OPEN= feeding/deliver.sh              # present: by default it approaches only once the user opens their mouth (60 s); empty = do not wait
#
# present = capture -> detect nose + mouth -> approach to 80 mm short of the mouth. If the food hides the
# mouth: sidestep 50 mm away from the user (-Y) and look again, twice at most; then the nose estimate.
# Every approach uses only the detection made at that pose, just before it.
# Exit 7 = the user did not open their mouth within 60 s: the arm retracted with the food still on it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/venvs/la/bin/python"          # LocateAnything (fallback)
MPY="$HOME/venvs/mp/bin/python"         # MediaPipe face landmarks (first choice, ~25 ms)
STOP_AFTER="${STOP_AFTER:-}"; START_AT="${START_AT:-swing}"; VALIDATE="${VALIDATE:-}"; WAIT_OPEN="${WAIT_OPEN-1}"   # on by default (user, 2026-09-24); WAIT_OPEN= to skip
ARM="docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work -w /work tcp_rec python3 -u -m kinova.scripts.deliver"
VO=""; [ -n "$VALIDATE" ] && VO="--validate-only"
STAGES="swing present bite retract"
case " $STAGES " in *" $START_AT "*) ;; *) echo "START_AT must be one of: $STAGES"; exit 2 ;; esac
started=0
run() { [ "$1" = "$START_AT" ] && started=1; [ "$started" = 1 ]; }
stop_here() { [ "$STOP_AFTER" = "$1" ] && { echo; echo "stopped after $1 (STOP_AFTER)"; exit 0; } || true; }
arm() { $ARM "$@" 2>&1 | { grep -v Deprecation || true; }; return "${PIPESTATUS[0]}"; }

capture_colour() {   # $1 = run dir: one colour frame, ~1 s
  PYTHONPATH=. "$MPY" -m feeding.grab_frame "$1/colour.jpg" >/dev/null 2>&1 && return 0
  rm -f /tmp/dl_*.jpg     # fallback: the old way (records until the 22 s timeout)
  timeout 22 gst-launch-1.0 rtspsrc location=rtsp://192.168.2.67/color protocols=tcp latency=200 \
    ! rtph264depay ! h264parse ! openh264dec ! videoconvert ! jpegenc \
    ! multifilesink location=/tmp/dl_%03d.jpg >/dev/null 2>&1 || true
  cp "$(ls /tmp/dl_*.jpg | tail -1)" "$1/colour.jpg"; rm -f /tmp/dl_*.jpg
}
capture_depth() {    # $1 = run dir: 20-frame median depth
  if ! PYTHONPATH=. "$PY" -m feeding.depth_grab "$1/depth.npy" >/dev/null 2>&1; then     # resident: ~0.5 s
    docker cp feeding/depth_probe.py kinova_vision:/tmp/depth_probe.py >/dev/null
    docker exec kinova_vision bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
      cd /tmp && python3 depth_probe.py --out /tmp/dl_depth --frames 15" >/dev/null 2>&1
    docker cp kinova_vision:/tmp/dl_depth.npy "$1/depth.npy" >/dev/null 2>&1 || true
  fi
  # fallback: fill sensor holes (the food hiding the face from the depth camera) from Depth Anything 3
  PYTHONPATH=.:$HOME/da3src/src "$PY" -m feeding.depth_fill "$1" 2>/dev/null | sed 's/^/    /' || true
}
detect_face() {      # $1 = run dir: MediaPipe, else LocateAnything
  set +e; PYTHONPATH=. "$MPY" -m feeding.detect_mouth_mp "$1" 2>/dev/null | grep -E "^(nose|mouth|mediapipe)" | sed 's/^/    /'
  rc=${PIPESTATUS[0]}; set -e
  if [ "$rc" != 0 ]; then
    echo "    MediaPipe found no face -- LocateAnything"
    PYTHONPATH=. "$PY" -m feeding.detect_mouth "$1" 2>/dev/null | grep -E "^(nose|mouth)" | sed 's/^/    /'
  fi
}

if run swing; then
  echo "$(date +%T) [1/4] SWING to the user"; arm swing $VO
  [ -n "$VALIDATE" ] && { echo; echo "validate-only: the swing was checked; present needs the arm there"; exit 0; }
  stop_here swing
fi
if run present; then
  echo "$(date +%T) [2/4] PRESENT"
  for SHIFT in 0 50 100; do
    D="$HOME/rammp_runs/deliver_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$D"
    if [ -n "$WAIT_OPEN" ] && [ "$SHIFT" = 0 ]; then
      set +e; PYTHONPATH=. "$MPY" -m feeding.detect_mouth_mp "$D" --wait-open 60 2>/dev/null | grep -v -E "^(W0000|I0000|INFO)"; rc=${PIPESTATUS[0]}; set -e
      if [ "$rc" != 0 ]; then       # no open mouth = the user does not want this bite: take it back
        echo "    no open mouth -- not approaching; retracting"
        arm retract; exit 7
      fi
    else
      capture_colour "$D"
    fi
    capture_depth "$D"
    detect_face "$D"
    C="/tmp/dl_$(date +%s)"; docker exec tcp_rec mkdir -p "$C"
    docker cp "$D/face.json" tcp_rec:"$C/face.json" >/dev/null; docker cp "$D/colour.jpg" tcp_rec:"$C/colour.jpg" >/dev/null
    NOSE=""; [ "$SHIFT" = 100 ] && NOSE="--allow-nose"
    set +e; arm approach --face "$C/face.json" --colour "$C/colour.jpg" $NOSE $VO; rc=$?; set -e
    [ $rc = 0 ] && break
    [ $rc != 3 ] && { echo "present refused -- nothing further moved ($D)"; exit 1; }
    [ -n "$VALIDATE" ] && { echo "    validate-only: would sidestep -Y 50 mm and look again"; exit 0; }
    [ "$SHIFT" = 100 ] && { echo "mouth not found after sidestepping 100 mm"; exit 1; }
    arm sidestep --step 0.05
  done
  echo "    detection: $D"
  [ -n "$VALIDATE" ] && exit 0
  stop_here present
fi
if run bite; then
  echo "$(date +%T) [3/4] BITE -- take it when ready"
  set +e; arm bite --timeout 60; rc=$?; set -e
  [ $rc = 4 ] && echo "    no bite: retracting anyway"
  [ $rc != 0 ] && [ $rc != 4 ] && exit 1
  stop_here bite
fi
if run retract; then
  echo "$(date +%T) [4/4] RETRACT to food_detect_start"; arm retract $VO
fi
