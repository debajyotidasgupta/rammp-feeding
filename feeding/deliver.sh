#!/usr/bin/env bash
# Food delivery, from the end of the scoop. Runs ON THE JETSON.
#
#   feeding/deliver.sh                         # swing -> present -> bite -> retract (to food_detect_start)
#   STOP_AFTER=present feeding/deliver.sh      # stop at a stage: swing | present | bite
#   START_AT=retract feeding/deliver.sh        # resume: present | bite | retract
#   VALIDATE=1 feeding/deliver.sh              # plan and validate the swing / approach; nothing moves
#   WAIT_OPEN= feeding/deliver.sh              # present: by default it approaches only once the user opens their mouth (60 s); empty = do not wait
#
# present, features.track_approach: once the mouth opens, the tracked approach (deliver.py track): one continuous
# move steered by the wrist camera to delivery.stop_m short of the mouth, wherever the user is in view and reach;
# stopped for any reason, the arm retracts. Without it:
# present = capture -> detect nose + mouth -> approach to delivery.stop_m (80 mm) short of the mouth. If the food
# hides the mouth: sidestep delivery.sidestep_m (50 mm, -Y) and look again, twice at most; then the nose estimate.
# Every approach uses only the detection made at that pose, just before it.
# Exit 7 = the user did not open their mouth within delivery.open_mouth_timeout_s: the arm retracted with the food.
# Exit 8 = the presentation stopped or was refused (out of reach, no usable mouth, a guard): the arm retracted.
#
# Records into the bite's deliver/: one look_<n>/ per look at the face, arm/ for the arm (feeding/lib.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source feeding/lib.sh                            # PY: LocateAnything (fallback); MPY: MediaPipe (first choice, ~25 ms)
bite_attach; session_begin deliver; stage_begin deliver
OPEN_WAIT_S="$(S delivery.open_mouth_timeout_s)"; FACE_DETECTOR="$(S features.face_detector)"; TRACK="$(S features.track_approach)"
STOP_AFTER="${STOP_AFTER:-}"; START_AT="${START_AT:-swing}"; VALIDATE="${VALIDATE:-}"; WAIT_OPEN="${WAIT_OPEN-$(S features.wait_for_open_mouth)}"   # default from config/; WAIT_OPEN= to skip
ARM="$(arm_cmd deliver)"
VO=""; [ -n "$VALIDATE" ] && VO="--validate-only"
STAGES="swing present bite retract"
case " $STAGES " in *" $START_AT "*) ;; *) echo "START_AT must be one of: $STAGES"; exit 2 ;; esac
started=0
run() { [ "$1" = "$START_AT" ] && started=1; [ "$started" = 1 ]; }
stop_here() { [ "$STOP_AFTER" = "$1" ] && { echo; echo "stopped after $1 (STOP_AFTER)"; exit 0; } || true; }
arm() { $ARM "$@" 2>&1 | { grep --line-buffered -v Deprecation || true; }; return "${PIPESTATUS[0]}"; }

capture_colour() {   # $1 = run dir: one colour frame, ~1 s
  PYTHONPATH=. "$MPY" -m feeding.grab_frame "$1/colour.jpg" >/dev/null 2>&1 && return 0
}
capture_depth() {    # $1 = run dir: median depth (perception.depth_frames), from frames since the arm stopped ($STILL_SINCE)
  if ! PYTHONPATH=. "$PY" -m feeding.depth_grab "$1/depth.npy" --since "${STILL_SINCE:-$(date +%s.%N)}" >/dev/null 2>&1 && [ "$CAM_KIND" = kinova_vision ]; then     # resident: ~0.5 s
    docker cp feeding/depth_probe.py "$VISION":/tmp/depth_probe.py >/dev/null
    docker exec "$VISION" bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
      cd /tmp && python3 depth_probe.py --out /tmp/dl_depth --frames $FRAMES" >/dev/null 2>&1
    docker cp "$VISION":/tmp/dl_depth.npy "$1/depth.npy" >/dev/null 2>&1 || true
  fi
  # fallback: fill sensor holes (the food hiding the face from the depth camera) from Depth Anything 3
  PYTHONPATH=.:$DA3SRC/src "$PY" -m feeding.depth_fill "$1" 2>/dev/null | sed -u 's/^/    /' || true
}
detect_face() {      # $1 = run dir: MediaPipe, else LocateAnything
  rc=5
  if [ "$FACE_DETECTOR" = mediapipe ]; then
    set +e; PYTHONPATH=. "$MPY" -m feeding.detect_mouth_mp "$1" 2>/dev/null | grep --line-buffered -E "^(nose|mouth|mediapipe)" | sed -u 's/^/    /'
    rc=${PIPESTATUS[0]}; set -e
  fi
  if [ "$rc" != 0 ]; then
    [ "$FACE_DETECTOR" = mediapipe ] && echo "    MediaPipe found no face -- LocateAnything"
    PYTHONPATH=. "$PY" -m feeding.detect_mouth "$1" 2>/dev/null | grep --line-buffered -E "^(nose|mouth)" | sed -u 's/^/    /'
  fi
}

if run swing; then
  echo "$(date +%T) [1/4] SWING to the user"; arm swing $VO; STILL_SINCE=$(date +%s.%N)
  [ -n "$VALIDATE" ] && { echo; echo "validate-only: the swing was checked; present needs the arm there"; exit 0; }
  stop_here swing
fi
if run present && [ -n "$TRACK" ]; then
  echo "$(date +%T) [2/4] PRESENT (tracked)"
  D="$STAGE_DIR/look_$(( $(find "$STAGE_DIR" -maxdepth 1 -name 'look_*' | wc -l) + 1 ))"; mkdir -p "$D"; export RAMMP_RECORD="$D"
  if [ -n "$WAIT_OPEN" ] && [ -z "$VALIDATE" ]; then
    set +e; PYTHONPATH=. "$MPY" -m feeding.detect_mouth_mp "$D" --wait-open "$OPEN_WAIT_S" 2>/dev/null | grep --line-buffered -v -E "^(W0000|I0000|INFO)"; rc=${PIPESTATUS[0]}; set -e
    [ "$rc" != 0 ] && { echo "    no open mouth -- not approaching; retracting"; ev declined; arm retract; exit 7; }
  fi
  set +e; arm track $VO; rc=$?; set -e
  if [ $rc != 0 ]; then
    echo "    the tracked approach stopped (rc $rc) -- retracting"
    [ -n "$VALIDATE" ] && exit 1
    arm retract && exit 8 || exit 1                 # 8: stopped, but retracted to food_detect_start
  fi
  [ -n "$VALIDATE" ] && exit 0
  stop_here present
elif run present; then
  echo "$(date +%T) [2/4] PRESENT"
  for SHIFT in 0 50 100; do
    D="$STAGE_DIR/look_$(( $(find "$STAGE_DIR" -maxdepth 1 -name 'look_*' | wc -l) + 1 ))"; mkdir -p "$D"
    export RAMMP_RECORD="$D"
    if [ -n "$WAIT_OPEN" ] && [ "$SHIFT" = 0 ]; then
      set +e; PYTHONPATH=. "$MPY" -m feeding.detect_mouth_mp "$D" --wait-open "$OPEN_WAIT_S" 2>/dev/null | grep --line-buffered -v -E "^(W0000|I0000|INFO)"; rc=${PIPESTATUS[0]}; set -e
      if [ "$rc" != 0 ]; then       # no open mouth = the user does not want this bite: take it back
        echo "    no open mouth -- not approaching; retracting"; ev declined
        arm retract; exit 7
      fi
    else
      capture_colour "$D"
    fi
    capture_depth "$D"
    detect_face "$D"
    C="/tmp/dl_$(date +%s)"; docker exec "$DRIVER" mkdir -p "$C"
    docker cp "$D/face.json" "$DRIVER":"$C/face.json" >/dev/null; docker cp "$D/colour.jpg" "$DRIVER":"$C/colour.jpg" >/dev/null
    NOSE=""; [ "$SHIFT" = 100 ] && NOSE="--allow-nose"
    set +e; arm approach --face "$C/face.json" --colour "$C/colour.jpg" $NOSE $VO; rc=$?; set -e
    if [ $rc = 5 ]; then           # the depth was not a face at the fork (a stale frame): one fresh look from here
      echo "    looking again (fresh colour + depth)"; ev depth_implausible
      D2="${D}_again"; mkdir -p "$D2"; capture_colour "$D2"; capture_depth "$D2"; detect_face "$D2"; D="$D2"
      docker cp "$D/face.json" "$DRIVER":"$C/face.json" >/dev/null; docker cp "$D/colour.jpg" "$DRIVER":"$C/colour.jpg" >/dev/null
      set +e; arm approach --face "$C/face.json" --colour "$C/colour.jpg" $NOSE $VO; rc=$?; set -e
    fi
    [ $rc = 0 ] && break
    [ $rc != 3 ] && { echo "present refused ($D) -- retracting"; [ -n "$VALIDATE" ] && exit 1; arm retract && exit 8 || exit 1; }
    [ -n "$VALIDATE" ] && { echo "    validate-only: would sidestep -Y 50 mm and look again"; exit 0; }
    [ "$SHIFT" = 100 ] && { echo "mouth not found after sidestepping 100 mm -- retracting"; arm retract && exit 8 || exit 1; }
    arm sidestep                    # delivery.sidestep_m
    STILL_SINCE=$(date +%s.%N)
  done
  echo "    detection: $D"
  [ -n "$VALIDATE" ] && exit 0
  stop_here present
fi
if run bite; then
  echo "$(date +%T) [3/4] BITE -- take it when ready"
  set +e; arm bite; rc=$?; set -e
  [ $rc = 4 ] && echo "    no bite: retracting anyway"
  [ $rc != 0 ] && [ $rc != 4 ] && exit 1
  stop_here bite
fi
if run retract; then
  echo "$(date +%T) [4/4] RETRACT to food_detect_start"; arm retract $VO
fi
